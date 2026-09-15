"""
HybridTrunkROIDBHHead — DBH prediction dengan 3 sinyal:
  1. Backbone res2 strip features di h=1.3m  (192 dim)
  2. Geometric prior: dbh_geom + depth_trunk  (2 dim)
  3. Species one-hot                          (13 dim)

Dibanding TrunkROIDBHHead original:
  - Spesies didapat dari gt_classes (training) / pred_classes (inference)
  - Geometric estimate dipakai sebagai explicit input feature (bukan hanya difilter)
  - Training hanya pada clean instances: ratio[0.5,2.0] + wh±0.25m + d=[1,10m) + trunk_px≥5
  - NO fallback tol_wh=inf di training (strict = bersih); fallback tetap ada di inference
"""

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ORIG_H   = 270.0
_ORIG_W   = 480.0
_ORIG_CY  = 135.0
_ORIG_FY  = 240.0
_H_CAM    = 2.0
_LOG_DMAX = math.log1p(237.6)

NUM_SPECIES = 13


def _denorm_depth(d_norm: torch.Tensor) -> torch.Tensor:
    return torch.expm1(d_norm * _LOG_DMAX)


def _find_row_1p3m(mask: torch.Tensor, depth_raw: torch.Tensor,
                   cy: float, fy: float,
                   h_cam: float = _H_CAM,
                   tol_wh: float = 0.5,
                   depth_tol: float = 0.3):
    """
    Returns (best_row, trunk_cols, d_trunk, world_height) atau (None,None,None,None).
    """
    rows_with_mask = mask.sum(dim=1).nonzero(as_tuple=False).squeeze(1)
    if len(rows_with_mask) < 3:
        return None, None, None, None

    best_r, best_wh = None, None
    for r in rows_with_mask.tolist():
        cols = mask[r].nonzero(as_tuple=False).squeeze(1)
        if len(cols) < 2:
            continue
        d_vals = depth_raw[r][cols]
        valid  = (d_vals > 0.1) & (d_vals < 200.0)
        if valid.sum() < 2:
            continue
        d  = d_vals[valid].median().item()
        wh = h_cam - (r - cy) * d / fy
        if best_r is None or abs(wh - 1.3) < abs(best_wh - 1.3):
            best_r, best_wh = r, wh

    if best_r is None or abs(best_wh - 1.3) > tol_wh:
        return None, None, None, None

    row_cols = mask[best_r].nonzero(as_tuple=False).squeeze(1)
    row_deps = depth_raw[best_r][row_cols]
    valid    = (row_deps > 0.1) & (row_deps < 200.0)
    row_cols = row_cols[valid]
    row_deps = row_deps[valid]
    if len(row_cols) < 2:
        return None, None, None, None

    d_min      = row_deps.min().item()
    trunk_m    = row_deps <= d_min + depth_tol
    trunk_cols = row_cols[trunk_m]
    d_trunk    = row_deps[trunk_m].mean().item()
    if len(trunk_cols) < 2:
        return None, None, None, None

    return best_r, trunk_cols, d_trunk, best_wh


class HybridTrunkROIDBHHead(nn.Module):
    """
    Args:
        in_channels:  channel backbone res2 (FocalNet-L = 192)
        hidden:       lebar hidden layer
        strip_rows:   ± feature rows disertakan di sekitar row 1.3m
        num_species:  jumlah spesies (default 13)
    """

    def __init__(self, in_channels: int = 192, hidden: int = 256,
                 strip_rows: int = 1, num_species: int = NUM_SPECIES,
                 dropout: float = 0.1):
        super().__init__()
        self.in_channels  = in_channels
        self.strip_rows   = strip_rows
        self.num_species  = num_species

        # Input: backbone (192) + geometric (2) + species_onehot (13) = 207
        feat_dim = in_channels + 2 + num_species

        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 3.5)

    def _extract_strip_feat(self, feat: torch.Tensor, row_img: int,
                            trunk_cols_img: torch.Tensor) -> Optional[torch.Tensor]:
        C, H_f, W_f = feat.shape
        stride = 4
        r_feat = row_img // stride
        r_lo   = max(0, r_feat - self.strip_rows)
        r_hi   = min(H_f - 1, r_feat + self.strip_rows) + 1
        col_feats = (trunk_cols_img // stride).unique()
        col_feats = col_feats[col_feats < W_f]
        if len(col_feats) == 0:
            return None
        strip = feat[:, r_lo:r_hi, :][:, :, col_feats]
        return strip.mean(dim=[1, 2])   # (C,)

    def forward_single(
        self,
        feat:         torch.Tensor,   # (C, H_f, W_f)
        depth_norm:   torch.Tensor,   # (H_img, W_img) normalized [0,1]
        mask:         torch.Tensor,   # (H_img, W_img) bool
        species_idx:  int,            # 0-based class index
        strict:       bool = True,    # True = no fallback (training); False = fallback (inference)
    ) -> Optional[Tuple]:
        """
        Returns (pred_scalar, dbh_geom_mm, d_trunk, trunk_px, world_height)
        atau None jika gagal.
        """
        H_img, _ = depth_norm.shape
        scale     = H_img / _ORIG_H
        cy        = _ORIG_CY * scale
        fy        = _ORIG_FY * scale
        fx        = _ORIG_FY * scale   # square pixels

        depth_raw = _denorm_depth(depth_norm)

        row_img, trunk_cols, d_trunk, world_h = _find_row_1p3m(
            mask, depth_raw, cy, fy, tol_wh=0.5)

        if row_img is None and not strict:
            row_img, trunk_cols, d_trunk, world_h = _find_row_1p3m(
                mask, depth_raw, cy, fy, tol_wh=float("inf"))

        if row_img is None:
            return None

        trunk_px = len(trunk_cols)

        feat_vec = self._extract_strip_feat(feat, row_img, trunk_cols)
        if feat_vec is None:
            return None

        dbh_geom_mm = trunk_px * d_trunk * 1000.0 / fx

        # Geometric feature: log1p(dbh_geom)/10, depth/10
        geom_feat = torch.tensor(
            [math.log1p(dbh_geom_mm) / 10.0, d_trunk / 10.0],
            device=feat.device, dtype=torch.float32)

        # Species one-hot
        sp_onehot = F.one_hot(
            torch.tensor(species_idx, device=feat.device),
            self.num_species).float()

        x = torch.cat([feat_vec, geom_feat, sp_onehot], dim=0)   # (207,)
        pred = self.mlp(x).squeeze(-1)                             # scalar

        return pred, dbh_geom_mm, d_trunk, trunk_px, world_h

    def compute_loss(
        self,
        res2:          torch.Tensor,   # (B, C, H_f, W_f)
        depth_norms:   list,           # list[Tensor (H,W)]
        instances:     list,           # list[Instances]
        cat_log_mean:  dict,           # {cat_id (1-based): float}
        cat_log_std:   dict,
        # Thresholds untuk clean filter
        min_trunk_px:  int   = 5,
        min_depth:     float = 1.0,
        max_depth:     float = 10.0,
        max_wh_dev:    float = 0.25,   # |world_h - 1.3| ≤ max_wh_dev
        min_ratio:     float = 0.5,
        max_ratio:     float = 2.0,
        species_weights: dict = None,  # {cat_id_1based: float} — None = uniform
    ):
        """
        Loss hanya dihitung pada clean instances (semua filter aktif).
        cat_id = gt_classes + 1  (detectron2 0-based → annotation 1-based)
        """
        device = res2.device
        preds, gts, cat_ids = [], [], []
        n_valid  = 0
        n_total  = 0
        n_skip_filter = 0

        for b in range(len(instances)):
            inst = instances[b]
            if not inst.has("gt_masks") or not inst.has("gt_dbh"):
                continue
            masks_b  = inst.gt_masks
            dbhs_b   = inst.gt_dbh
            classes_b = inst.gt_classes
            depth_b  = depth_norms[b].to(device)
            feat_b   = res2[b]

            for i in range(len(inst)):
                dbh_cm = dbhs_b[i].item()
                if dbh_cm <= 0:
                    continue
                n_total += 1
                gt_mm    = dbh_cm * 10.0
                cat_id   = classes_b[i].item() + 1   # 0-based → 1-based

                result = self.forward_single(
                    feat_b, depth_b, masks_b[i].bool(),
                    species_idx=classes_b[i].item(), strict=True)

                if result is None:
                    n_skip_filter += 1
                    continue

                pred, dbh_geom_mm, d_trunk, trunk_px, world_h = result

                # Clean filters
                if trunk_px < min_trunk_px:
                    n_skip_filter += 1; continue
                if not (min_depth <= d_trunk < max_depth):
                    n_skip_filter += 1; continue
                if abs(world_h - 1.3) > max_wh_dev:
                    n_skip_filter += 1; continue
                if gt_mm > 0:
                    ratio = dbh_geom_mm / gt_mm
                    if not (min_ratio <= ratio <= max_ratio):
                        n_skip_filter += 1; continue

                n_valid += 1
                preds.append(pred)
                gts.append(math.log1p(gt_mm))
                cat_ids.append(cat_id)

        if not preds:
            # Dummy loss melalui parameter head agar backward() tidak crash
            dummy = sum(p.sum() * 0.0 for p in self.mlp.parameters())
            return dummy, 0, n_total

        pred_t = torch.stack(preds)
        gt_t   = torch.tensor(gts, device=device)

        pred_norm = torch.zeros_like(pred_t)
        gt_norm   = torch.zeros_like(gt_t)
        for cid in set(cat_ids):
            idx   = [k for k, c in enumerate(cat_ids) if c == cid]
            idx_t = torch.tensor(idx, device=device)
            mu    = cat_log_mean.get(cid, 3.5)
            sigma = cat_log_std.get(cid, 0.5)
            pred_norm[idx_t] = (pred_t[idx_t] - mu) / sigma
            gt_norm[idx_t]   = (gt_t[idx_t]   - mu) / sigma

        if species_weights is not None:
            w = torch.tensor(
                [species_weights.get(cid, 1.0) for cid in cat_ids],
                device=device, dtype=torch.float32)
            w = w / w.mean()  # normalize: rata-rata weight tetap 1.0
            loss = (F.smooth_l1_loss(pred_norm, gt_norm, beta=1.0, reduction='none') * w).mean()
        else:
            loss = F.smooth_l1_loss(pred_norm, gt_norm, beta=1.0)
        return loss, n_valid, n_total

    @torch.no_grad()
    def predict_batch(self, res2, depth_norms, pred_masks_list, pred_classes_list):
        """
        Inference: pred_classes_list = list[Tensor (N,)] 0-based class indices.
        Returns list[Tensor (N,) float32]  (DBH mm, 0 jika gagal)
        """
        results = []
        for b in range(len(pred_masks_list)):
            feat_b    = res2[b]
            depth_b   = depth_norms[b].to(res2.device)
            masks_b   = pred_masks_list[b]
            classes_b = pred_classes_list[b]
            preds_b   = []
            for i in range(len(masks_b)):
                result = self.forward_single(
                    feat_b, depth_b, masks_b[i].bool(),
                    species_idx=classes_b[i].item(), strict=False)
                if result is None:
                    preds_b.append(torch.tensor(0.0, device=res2.device))
                else:
                    pred = result[0]
                    preds_b.append(torch.expm1(pred))
            results.append(torch.stack(preds_b) if preds_b
                           else torch.zeros(0, device=res2.device))
        return results
