"""
dbh_sandbox_head.py — Tier 1 dari DBH sandbox: trainer cepat berbasis cache.

Baca cache dari precompute_dbh_cache.py (res2 feature map per image + tabel
index per-instance dgn geometri row-1.3m sudah dihitung dgn threshold paling
longgar). SEMUA axis kombinasi (filter data + hyperparameter head) adalah
CLI arg -- generalisasi V1-V6 (finetune_trunk_roi_dbh_hybrid_v6.py dkk) jadi
SATU script, tanpa mengubah file lama itu.

Reuse tanpa modifikasi: HybridTrunkROIDBHHead & _extract_strip_feat
(scripts/trunk_roi_dbh_head_hybrid.py). Filter axis: filters.py.

Karena backbone TIDAK di-forward ulang (feature sudah di-cache), training ini
head-only -> jauh lebih cepat dari finetune_trunk_roi_dbh_hybrid_v6.py yg
~20menit/1000iter. Eval di sini SELALU deterministik (tanpa augmentasi
LSJ apapun -- cache dibangun sekali dgn transform tetap), sehingga menutup
bug is_train=True pada val loader di V6 (lihat plan).

Jalankan (baseline setara V6, tapi universe 11 spesies):
  nohup /scratch2/pr65/anur0018/conda_envs/maskdino/bin/python -u \\
      scripts/dbh_sandbox/dbh_sandbox_head.py \\
      --cache-dir /scratch2/pr65/anur0018/maskdino_output/dbh_sandbox_cache \\
      --output-dir /scratch2/pr65/anur0018/maskdino_output/dbh_sandbox/baseline \\
      --max-iter 20000 --eval-period 500 \\
      > logs/dbh_sandbox_baseline_nohup.log 2>&1 &
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE         = Path(__file__).resolve().parent
SCRIPTS_ROOT = HERE.parent
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(HERE))

from trunk_roi_dbh_head_hybrid import HybridTrunkROIDBHHead
import filters as F_

logger = logging.getLogger("dbh_sandbox_head")

NUM_SPECIES = 13


class FeatureCache:
    """Lazy in-RAM cache utk res2 feature map per image_id (fp16 di disk, fp32 di RAM)."""

    def __init__(self, cache_dir: Path, split: str):
        self.dir = Path(cache_dir) / "features" / split
        self._mem = {}

    def get(self, image_id: int, device) -> torch.Tensor:
        key = int(image_id)
        t = self._mem.get(key)
        if t is None:
            t = torch.load(self.dir / f"{key}.pt", map_location="cpu").float()
            self._mem[key] = t
        return t.to(device)


def load_index(cache_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_pickle(Path(cache_dir) / f"index_{split}.pkl")


def build_feature_vec(head: HybridTrunkROIDBHHead, feat_cache: FeatureCache,
                       row, device, use_geom: bool):
    feat_map = feat_cache.get(row.image_id, device)
    trunk_cols_t = torch.as_tensor(row.trunk_cols, device=device, dtype=torch.long)
    # REUSE _extract_strip_feat apa adanya (tidak reimplement) -- strip_rows
    # sudah ditentukan saat konstruksi `head`.
    feat_vec = head._extract_strip_feat(feat_map, int(row.row_img), trunk_cols_t)
    if feat_vec is None:
        return None

    geom_feat = torch.tensor(
        [math.log1p(max(row.dbh_geom_mm, 0.0)) / 10.0, row.d_trunk / 10.0],
        device=device, dtype=torch.float32)
    if not use_geom:
        geom_feat = geom_feat * 0.0  # ablasi: matikan info geometric, arsitektur tetap sama

    sp_idx = int(row.category_id) - 1  # konvensi proyek 1-based -> 0-based utk one-hot
    sp_onehot = F.one_hot(torch.tensor(sp_idx, device=device), NUM_SPECIES).float()
    return torch.cat([feat_vec, geom_feat, sp_onehot], dim=0)


def compute_metrics(preds_mm: np.ndarray, gts_mm: np.ndarray, cats: np.ndarray,
                     min_species_n: int = 30) -> dict:
    e = np.abs(preds_mm - gts_mm)
    ss_res = np.sum((gts_mm - preds_mm) ** 2)
    ss_tot = np.sum((gts_mm - gts_mm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    per_species = {}
    for cid in sorted(set(cats.tolist())):
        idx = cats == cid
        gi, pi = gts_mm[idx], preds_mm[idx]
        ei = np.abs(gi - pi)
        ss_r = np.sum((gi - pi) ** 2)
        ss_t = np.sum((gi - gi.mean()) ** 2)
        r2_c = 1 - ss_r / ss_t if ss_t > 0 else float("nan")
        per_species[int(cid)] = {"N": int(idx.sum()), "R2": float(r2_c), "MAE": float(ei.mean())}

    # R2 pada N kecil sangat tidak stabil (varians ground-truth di sampel kecil bisa nyaris nol,
    # bikin error absolut wajar jadi R2 ekstrem negatif meski MAE bagus -- lihat kasus Persimmon
    # N=10 R2=-23.9 tapi MAE=36.5mm, salah satu terbaik). mean/worst_species_R2 (metrik ranking
    # utama sweep) HANYA menghitung spesies dgn N>=min_species_n; spesies N-kecil tetap muncul
    # penuh di per_species (N/R2/MAE) sbg info tambahan, tidak dibuang, cuma tidak ikut menurunkan rata2.
    included = {cid: s for cid, s in per_species.items()
                if not math.isnan(s["R2"]) and s["N"] >= min_species_n}
    excluded_small_n = sorted(cid for cid, s in per_species.items() if cid not in included)
    sp_r2_vals = [s["R2"] for s in included.values()]
    mean_per_species_r2 = float(np.mean(sp_r2_vals)) if sp_r2_vals else float("nan")
    worst_species_r2 = float(np.min(sp_r2_vals)) if sp_r2_vals else float("nan")

    return {
        "N": len(preds_mm), "MAE": float(e.mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
        "bias": float((preds_mm - gts_mm).mean()), "R2": float(r2),
        "mean_per_species_R2": mean_per_species_r2, "worst_species_R2": worst_species_r2,
        "min_species_n_threshold": min_species_n,
        "n_species_in_mean": len(sp_r2_vals), "species_excluded_small_n": excluded_small_n,
        "w50mm": float((e < 50).mean() * 100), "w100mm": float((e < 100).mean() * 100),
        "per_species": per_species,
    }


def evaluate(head, feat_cache, df, device, use_geom: bool, min_species_n: int = 30) -> dict:
    head.eval()
    preds, gts, cats = [], [], []
    with torch.no_grad():
        for row in df.itertuples():
            x = build_feature_vec(head, feat_cache, row, device, use_geom)
            if x is None:
                continue
            pred_log = head.mlp(x).squeeze(-1)
            preds.append(torch.expm1(pred_log).item())
            gts.append(row.gt_dbh_mm)
            cats.append(int(row.category_id))
    head.train()
    if not preds:
        return {"N": 0}
    return compute_metrics(np.array(preds), np.array(gts), np.array(cats), min_species_n)


def parse_ratio_bounds(s: str):
    if s is None or s.lower() == "off":
        return None
    lo, hi = s.split(",")
    return (float(lo), float(hi))


def parse_rubberfig_cap(s: str):
    if s is None or s.lower() == "off":
        return None
    return float(s)


def main(args):
    logging.basicConfig(level=logging.INFO,
                         format="[%(asctime)s %(name)s] %(message)s", datefmt="%m/%d %H:%M:%S")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    species_subset = F_.resolve_species_subset(args.species_subset)
    ratio_bounds = parse_ratio_bounds(args.ratio_bounds)
    rubberfig_cap = parse_rubberfig_cap(args.rubberfig_cap_cm)
    use_geom = args.geometric_feature == "on"

    logger.info(f"Combo: {vars(args)}")

    train_df_raw = load_index(cache_dir, "train")
    val_df_raw   = load_index(cache_dir, args.eval_split)

    train_df = F_.apply_filters(train_df_raw, species_subset=species_subset,
                                 min_trunk_px=args.min_trunk_px, min_depth=args.min_depth,
                                 max_depth=args.max_depth, wh_tolerance=args.wh_tolerance,
                                 ratio_bounds=ratio_bounds, rubberfig_cap_cm=rubberfig_cap)
    val_df = F_.apply_filters(val_df_raw, species_subset=species_subset,
                               min_trunk_px=args.min_trunk_px, min_depth=args.min_depth,
                               max_depth=args.max_depth, wh_tolerance=args.wh_tolerance,
                               ratio_bounds=ratio_bounds, rubberfig_cap_cm=rubberfig_cap)

    logger.info(f"Train N={len(train_df)}  {args.eval_split} N={len(val_df)}  "
                f"(species_subset={species_subset})")
    if len(train_df) < 10 or len(val_df) < 5:
        logger.error("Terlalu sedikit instance setelah filter -- kombinasi ini di-skip.")
        with open(out_dir / "metrics.json", "w") as f:
            json.dump([{"iteration": 0, "N": len(train_df), "error": "insufficient_data"}], f)
        return

    for cid, g in train_df.groupby("category_id"):
        logger.info(f"  train {F_.CAT_ID_TO_NAME.get(int(cid), cid):<14} N={len(g)}")

    cat_log_mean, cat_log_std = F_.cat_log_stats(train_df)
    species_weights = F_.species_weight_scheme(train_df, args.species_weight_scheme,
                                                cap=args.species_weight_cap)
    logger.info(f"cat_log_mean={cat_log_mean}")
    logger.info(f"species_weights={species_weights}")

    train_feat_cache = FeatureCache(cache_dir, "train")
    val_feat_cache   = FeatureCache(cache_dir, args.eval_split)

    head = HybridTrunkROIDBHHead(in_channels=192, hidden=args.hidden_width,
                                  strip_rows=args.strip_rows, num_species=NUM_SPECIES,
                                  dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    logger.info(f"HybridTrunkROIDBHHead: {n_params:,} params, strip_rows={args.strip_rows}, "
                f"hidden={args.hidden_width}, dropout={args.dropout}, geom_feature={use_geom}")

    optimizer = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iter, eta_min=args.lr * 0.01)

    loss_fn = F.smooth_l1_loss if args.loss_type == "smooth_l1" else F.mse_loss

    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(train_df))
    cursor = 0
    batch_size = min(args.batch_size, len(train_df))

    metrics_path = out_dir / "metrics.json"
    metrics_log = []
    best_score = -float("inf")
    t0 = time.time()

    head.train()
    for iteration in range(1, args.max_iter + 1):
        if cursor + batch_size > len(order):
            order = rng.permutation(len(train_df))
            cursor = 0
        idx = order[cursor:cursor + batch_size]
        cursor += batch_size
        batch_rows = train_df.iloc[idx]

        xs, gt_logs, cat_ids = [], [], []
        for row in batch_rows.itertuples():
            x = build_feature_vec(head, train_feat_cache, row, device, use_geom)
            if x is None:
                continue
            xs.append(x)
            gt_logs.append(math.log1p(row.gt_dbh_mm))
            cat_ids.append(int(row.category_id))
        if not xs:
            continue

        X = torch.stack(xs)
        preds = head.mlp(X).squeeze(-1)
        gt_t = torch.tensor(gt_logs, device=device, dtype=torch.float32)

        mu    = torch.tensor([cat_log_mean.get(c, 3.5) for c in cat_ids], device=device)
        sigma = torch.tensor([cat_log_std.get(c, 1.0) for c in cat_ids], device=device)
        pred_norm = (preds - mu) / sigma
        gt_norm   = (gt_t - mu) / sigma

        w = torch.tensor([species_weights.get(c, 1.0) for c in cat_ids],
                          device=device, dtype=torch.float32)
        w = w / w.mean()
        loss = (loss_fn(pred_norm, gt_norm, reduction="none") * w).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if iteration % 500 == 0:
            elapsed = time.time() - t0
            logger.info(f"iter {iteration:6d}/{args.max_iter}  loss={loss.item():.4f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed:.0f}s")

        if iteration % args.eval_period == 0 or iteration == args.max_iter:
            metrics = evaluate(head, val_feat_cache, val_df, device, use_geom, args.min_species_n)
            metrics["iteration"] = iteration
            metrics["loss"] = loss.item()
            metrics_log.append(metrics)
            with open(metrics_path, "w") as f:
                json.dump(metrics_log, f, indent=2)

            if "mean_per_species_R2" in metrics:
                score = metrics["mean_per_species_R2"]
                logger.info(f"[EVAL iter {iteration}]  N={metrics['N']}  "
                            f"meanPerSpR2={score:.4f}  aggR2={metrics['R2']:.4f}  "
                            f"MAE={metrics['MAE']:.1f}mm  worstSpR2={metrics['worst_species_R2']:.4f}")
                if score > best_score:
                    best_score = score
                    torch.save(head.state_dict(), out_dir / "head_best.pth")
                    with open(out_dir / "best_metrics.json", "w") as f:
                        json.dump(metrics, f, indent=2)

    torch.save(head.state_dict(), out_dir / "head_final.pth")
    with open(out_dir / "combo.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"DONE. Best mean_per_species_R2={best_score:.4f}  -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--min-species-n", type=int, default=30,
                    help="Spesies dgn N eval < ini dikecualikan dari mean/worst_species_R2 "
                         "(R2 tidak stabil di N kecil) -- tetap dilaporkan penuh di per_species")

    # Data-filter axes
    p.add_argument("--species-subset", default="universe11")
    p.add_argument("--min-trunk-px", type=int, default=5)
    p.add_argument("--min-depth", type=float, default=1.0)
    p.add_argument("--max-depth", type=float, default=20.0)
    p.add_argument("--wh-tolerance", type=float, default=0.5)
    p.add_argument("--ratio-bounds", default="off", help="'lo,hi' atau 'off'")
    p.add_argument("--rubberfig-cap-cm", default="150", help="cm atau 'off'")

    # Model/training axes
    p.add_argument("--strip-rows", type=int, default=5)
    p.add_argument("--species-weight-scheme", default="inverse_freq_capped",
                    choices=["none", "inverse_freq_capped", "sqrt_inverse_freq"])
    p.add_argument("--species-weight-cap", type=float, default=20.0)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden-width", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--loss-type", default="smooth_l1", choices=["smooth_l1", "mse"])
    p.add_argument("--geometric-feature", default="on", choices=["on", "off"])

    # Training loop
    p.add_argument("--max-iter", type=int, default=20000)
    p.add_argument("--eval-period", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    main(args)
