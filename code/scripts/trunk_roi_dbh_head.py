"""
TrunkROIDBHHead — DBH prediction dari backbone feature strip di h=1.3m.

Motivasi: DBH head sebelumnya hanya pakai query embedding (abstrak).
Head ini langsung melihat strip feature visual dari feature map res2 (stride=4)
di posisi geometris 1.3m dari tanah → lebih informatif tentang lebar trunk.

Pipeline per instance:
  1. Dari mask + depth: cari row gambar paling dekat ke h=1.3m
  2. Dari res2 (C=192, stride=4): ambil strip ±1 feature row di row tersebut
  3. Hanya ambil kolom feature yang mask-nya aktif + depth proximity filter
  4. Average pool strip → (C,) vector
  5. MLP → log1p(DBH_mm) → eksponensial balik → DBH mm
"""

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Konstanta kamera ORIGINAL (480×270). Di-scale sesuai image size aktual.
_ORIG_H    = 270.0
_ORIG_W    = 480.0
_ORIG_CY   = 135.0
_ORIG_FY   = 240.0
_H_CAM     = 2.0          # tinggi kamera dari tanah (m), SPREAD default
_LOG_DMAX  = math.log1p(237.6)   # normalisasi di mapper


def _denorm_depth(d_norm: torch.Tensor) -> torch.Tensor:
    """Kembalikan depth dari [0,1] log-normalized ke meter."""
    return torch.expm1(d_norm * _LOG_DMAX)


def _find_row_1p3m(mask: torch.Tensor, depth_raw: torch.Tensor,
                   cy: float, fy: float,
                   h_cam: float = _H_CAM,
                   tol_wh: float = 0.25,
                   depth_tol: float = 0.3) -> tuple:
    """
    Cari row di gambar yang paling dekat ke world height = 1.3m.

    Returns:
        (best_row, trunk_cols, d_trunk)  atau  (None, None, None) jika gagal
    """
    rows_with_mask = mask.sum(dim=1).nonzero(as_tuple=False).squeeze(1)
    if len(rows_with_mask) < 3:
        return None, None, None

    best_r, best_wh = None, None
    for r in rows_with_mask.tolist():
        cols = mask[r].nonzero(as_tuple=False).squeeze(1)
        if len(cols) < 2:
            continue
        d_vals = depth_raw[r][cols]
        valid  = (d_vals > 0.1) & (d_vals < 200.0)
        if valid.sum() < 2:
            continue
        d   = d_vals[valid].median().item()
        wh  = h_cam - (r - cy) * d / fy
        if best_r is None or abs(wh - 1.3) < abs(best_wh - 1.3):
            best_r, best_wh = r, wh

    if best_r is None or abs(best_wh - 1.3) > tol_wh:
        return None, None, None

    # Depth proximity filter: hanya kolom dalam TOL dari permukaan depan
    row_cols = mask[best_r].nonzero(as_tuple=False).squeeze(1)
    row_deps = depth_raw[best_r][row_cols]
    valid    = (row_deps > 0.1) & (row_deps < 200.0)
    row_cols = row_cols[valid]
    row_deps = row_deps[valid]
    if len(row_cols) < 2:
        return None, None, None

    d_min    = row_deps.min().item()
    trunk_m  = row_deps <= d_min + depth_tol
    trunk_cols = row_cols[trunk_m]
    d_trunk  = row_deps[trunk_m].mean().item()

    if len(trunk_cols) < 2:
        return None, None, None

    return best_r, trunk_cols, d_trunk


class TrunkROIDBHHead(nn.Module):
    """
    MLP kecil yang memprediksi log1p(DBH_mm) dari strip feature backbone res2.

    Args:
        in_channels: channel backbone res2 (FocalNet-L = 192)
        hidden:      lebar layer tersembunyi
        strip_rows:  jumlah feature row di atas/bawah row 1.3m yang disertakan (±strip_rows)
    """

    def __init__(self, in_channels: int = 192, hidden: int = 256, strip_rows: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.strip_rows  = strip_rows

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        # Inisialisasi output layer ke populasi mean log1p(DBH) ~3.5
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 3.5)

    def _extract_strip_feat(
        self,
        feat: torch.Tensor,       # (C, H_f, W_f)
        row_img: int,
        trunk_cols_img: torch.Tensor,  # 1D tensor of image column indices
    ) -> Optional[torch.Tensor]:
        """
        Ambil feature dari strip ±strip_rows di feature map pada row_img.
        Hanya kolom yang berkorespondensi dengan trunk_cols_img.

        Returns: (C,) tensor atau None jika tidak ada kolom valid.
        """
        C, H_f, W_f = feat.shape
        stride = 4

        r_feat = row_img // stride
        r_lo   = max(0, r_feat - self.strip_rows)
        r_hi   = min(H_f - 1, r_feat + self.strip_rows) + 1

        # Konversi kolom gambar ke kolom feature, unik
        col_feats = (trunk_cols_img // stride).unique()
        col_feats = col_feats[col_feats < W_f]
        if len(col_feats) == 0:
            return None

        # feat[:, r_lo:r_hi, col_feats] → (C, n_rows, n_cols) → mean → (C,)
        strip = feat[:, r_lo:r_hi, :][:, :, col_feats]   # (C, n_rows, n_cols)
        return strip.mean(dim=[1, 2])                      # (C,)

    def forward_single(
        self,
        feat: torch.Tensor,        # (C, H_f, W_f)  res2 satu gambar
        depth_norm: torch.Tensor,  # (H_img, W_img)  depth normalized [0,1]
        mask: torch.Tensor,        # (H_img, W_img)  bool
    ) -> Optional[torch.Tensor]:
        """
        Prediksi log1p(DBH_mm) untuk satu instance.
        Returns: scalar tensor atau None jika gagal ekstrak strip.
        """
        H_img, W_img = depth_norm.shape
        scale = H_img / _ORIG_H
        cy    = _ORIG_CY * scale
        fy    = _ORIG_FY * scale

        depth_raw = _denorm_depth(depth_norm)

        # Coba dulu dengan toleransi ketat (h=1.3±0.5m)
        row_img, trunk_cols, _ = _find_row_1p3m(mask, depth_raw, cy, fy, tol_wh=0.5)

        # Fallback: mask tidak cover h=1.3m (mask berhenti di h=1.5-1.8m)
        # Gunakan row terbaik yang tersedia dari mask, tanpa cutoff toleransi
        if row_img is None:
            row_img, trunk_cols, _ = _find_row_1p3m(
                mask, depth_raw, cy, fy, tol_wh=float("inf"))

        if row_img is None:
            return None

        feat_vec = self._extract_strip_feat(feat, row_img, trunk_cols)
        if feat_vec is None:
            return None

        return self.mlp(feat_vec).squeeze(-1)   # scalar

    def compute_loss(
        self,
        res2: torch.Tensor,          # (B, C, H_f, W_f)
        depth_norms: list,           # list[Tensor (H,W)], len=B
        instances: list,             # list[Instances], len=B
        cat_log_mean: dict,
        cat_log_std:  dict,
    ) -> torch.Tensor:
        """
        Hitung smooth-L1 loss DBH atas semua GT instance dalam batch.
        Normalisasi per-spesies dalam log-space (sama seperti DBHNormCriterion).

        Returns: loss scalar (0 jika tidak ada instance valid)
        """
        device    = res2.device
        preds, gts, cats = [], [], []
        n_total = 0

        for b in range(len(instances)):
            inst  = instances[b]
            if not inst.has("gt_masks") or not inst.has("gt_dbh"):
                continue
            masks_b  = inst.gt_masks          # (N, H, W) bool tensor
            dbhs_b   = inst.gt_dbh            # (N,) float, cm
            labels_b = inst.gt_classes        # (N,) int
            depth_b  = depth_norms[b].to(device)
            feat_b   = res2[b]                # (C, H_f, W_f)

            for i in range(len(inst)):
                dbh_cm = dbhs_b[i].item()
                if dbh_cm <= 0:
                    continue
                n_total += 1
                mask_i = masks_b[i].bool()
                pred   = self.forward_single(feat_b, depth_b, mask_i)
                if pred is None:
                    continue
                preds.append(pred)
                gts.append(math.log1p(dbh_cm * 10.0))   # cm → mm → log1p
                cats.append(labels_b[i].item())

        if not preds:
            return res2.sum() * 0.0, 0, n_total   # keep graph

        pred_t = torch.stack(preds)                          # (N,)
        gt_t   = torch.tensor(gts, device=device)           # (N,)

        # Per-species z-score normalization
        pred_norm = torch.zeros_like(pred_t)
        gt_norm   = torch.zeros_like(gt_t)
        for cid in set(cats):
            idx   = [k for k, c in enumerate(cats) if c == cid]
            idx_t = torch.tensor(idx, device=device)
            mu    = cat_log_mean.get(cid, 3.5)
            sigma = cat_log_std.get(cid, 0.5)
            pred_norm[idx_t] = (pred_t[idx_t] - mu) / sigma
            gt_norm[idx_t]   = (gt_t[idx_t]   - mu) / sigma

        return F.smooth_l1_loss(pred_norm, gt_norm, beta=1.0), len(preds), n_total

    @torch.no_grad()
    def predict_batch(
        self,
        res2: torch.Tensor,
        depth_norms: list,
        pred_masks_list: list,   # list[Tensor (N, H, W) bool]
    ) -> list:
        """
        Inference: prediksi DBH mm per instance, per gambar.
        Returns: list[Tensor (N,) float32]  (DBH mm, 0 jika gagal)
        """
        results = []
        for b in range(len(pred_masks_list)):
            feat_b   = res2[b]
            depth_b  = depth_norms[b].to(res2.device)
            masks_b  = pred_masks_list[b]
            preds_b  = []
            for i in range(len(masks_b)):
                pred = self.forward_single(feat_b, depth_b, masks_b[i].bool())
                if pred is None:
                    preds_b.append(torch.tensor(0.0, device=res2.device))
                else:
                    preds_b.append(torch.expm1(pred))   # log1p → mm
            results.append(torch.stack(preds_b) if preds_b
                           else torch.zeros(0, device=res2.device))
        return results
