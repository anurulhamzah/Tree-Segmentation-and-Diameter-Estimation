"""
Hybrid Trunk ROI DBH Training — V7 (dataset v3, universe 11 spesies)
==============================================================
Fase D dari DBH Sandbox (scripts/dbh_sandbox/): sanity-check fidelitas kombinasi pemenang
sandbox (baseline: universe11 spesies, semua hyperparameter default -- lihat
maskdino_output/sweep_dbh_hyperparams/phaseC_winner.json) lewat pipeline PRODUKSI ASLI
(LSJ augmentation nyata via COCOInstanceDBHDatasetMapper, backbone di-forward tiap
iterasi -- BUKAN cache seperti scripts/dbh_sandbox/dbh_sandbox_head.py).

Perbedaan dari V6:
  1. Dataset v3 (BARU): universe 11 spesies (exclude HANYA Lemon+LeechVine, blank-DBH >50%),
     bukan 6 spesies v2. Lihat scripts/create_clean_dbh_json_v3_universe11.py.
     N_train=8,437 (vs 6,344 di v2).
  2. CAT_LOG_MEAN/STD/SPECIES_WEIGHT dihitung ulang utk 11 spesies (dari train v3).
  3. BUG DIPERBAIKI (ditemukan di V6 selama sesi DBH Sandbox, 12 Jul 2026): val loader V6
     pakai `COCOInstanceDBHDatasetMapper(cfg, True)` -- is_train=True bahkan utk eval,
     artinya eval JUGA kena random LSJ jitter (scale 0.4-2.0x + crop), bukan deterministic
     resize. Fix: construct mapper dgn is_train=True (supaya annotations tetap diproses,
     tidak di-pop), TAPI override `.tfm_gens` scr manual ke transform DETERMINISTIK
     (build_transform_gen(cfg, is_train=False) -- no flip, no scale jitter, resize tetap).
  4. strip_rows=5, dropout=0.3, hidden=256 -- SAMA PERSIS dgn V6 & baseline sandbox.

Jalankan:
  nohup /scratch2/pr65/anur0018/conda_envs/maskdino/bin/python -u \\
      scripts/finetune_trunk_roi_dbh_hybrid_v7_universe11.py \\
      --config-file configs/maskdino_FocalNet_L_combined_rle_f1000_rgbd_scratch_dbh_json_v3_universe11.yaml \\
      --model-weights /scratch2/pr65/anur0018/maskdino_output/FocalNet_L_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k/model_final.pth \\
      --output-dir /scratch2/pr65/anur0018/maskdino_output/FocalNet_L_trunk_roi_dbh_hybrid_v7_universe11_20k \\
      > logs/trunk_roi_dbh_hybrid_v7_universe11_20k_nohup.log 2>&1 &
"""

import argparse, json, logging, math, os, sys
from pathlib import Path

HERE          = Path(__file__).resolve().parent
PROJECT_ROOT  = HERE.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(HERE))

import torch
import torch.optim as optim
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from maskdino import add_maskdino_config
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import (
    COCOInstanceDBHDatasetMapper)
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
    build_transform_gen)
from register_combined import register_all_combined
from register_rainforests import register_all_rainforests
from register_plantations import register_all_plantations

from trunk_roi_dbh_head_hybrid import HybridTrunkROIDBHHead

logger = logging.getLogger("finetune_hybrid_v7")

# Per-species log1p(DBH_mm) statistics -- dihitung dari dataset v3 train (universe11)
# via: python3 -c "..." atas instances_train.json v3 (lihat sesi 12 Jul 2026)
CAT_LOG_MEAN = {
    1:  5.3710,   # Apple       N=437
    3:  4.6873,   # Loquat      N=892
    4:  5.5160,   # Mango       N=86
    5:  5.2144,   # Orange      N=653
    6:  4.9466,   # Persimmon   N=56
    7:  5.3494,   # Pomegranate N=55
    8:  6.2302,   # AliiFig     N=728
    9:  5.4537,   # BangaloPalm N=3451
    10: 6.1035,   # Fern        N=877
    12: 6.3724,   # RubberFig   N=234
    13: 5.9343,   # Umbrella    N=968
}
CAT_LOG_STD = {
    1:  0.1327,
    3:  0.2960,
    4:  0.2803,
    5:  0.1893,
    6:  0.0843,
    7:  0.2189,
    8:  0.2492,
    9:  0.4202,
    10: 0.2578,
    12: 0.1407,
    13: 0.2813,
}

CAT_NAMES = {
    1: 'Apple', 3: 'Loquat', 4: 'Mango', 5: 'Orange', 6: 'Persimmon', 7: 'Pomegranate',
    8: 'AliiFig', 9: 'BangaloPalm', 10: 'Fern', 12: 'RubberFig', 13: 'Umbrella',
}

# Per-species weight -- inverse frequency, cap 20.0, baseline=BangaloPalm (N=3451)
SPECIES_WEIGHT = {
    1:  7.9,   # Apple:       3451/437 = 7.9
    3:  3.9,   # Loquat:      3451/892 = 3.9
    4:  20.0,  # Mango:       3451/86  = 40.1 → cap 20
    5:  5.3,   # Orange:      3451/653 = 5.3
    6:  20.0,  # Persimmon:   3451/56  = 61.6 → cap 20
    7:  20.0,  # Pomegranate: 3451/55  = 62.7 → cap 20
    8:  4.7,   # AliiFig:     3451/728 = 4.7
    9:  1.0,   # BangaloPalm: baseline
    10: 3.9,   # Fern:        3451/877 = 3.9
    12: 14.7,  # RubberFig:   3451/234 = 14.7
    13: 3.6,   # Umbrella:    3451/968 = 3.6
}


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.model_weights:
        cfg.MODEL.WEIGHTS = args.model_weights
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def build_loaders(cfg):
    mapper_train = COCOInstanceDBHDatasetMapper(cfg, True)
    mapper_val   = COCOInstanceDBHDatasetMapper(cfg, True)
    # FIX bug V6: is_train=True dipakai supaya annotations TETAP diproses (bukan di-pop),
    # tapi transform-nya HARUS deterministik utk eval yg valid -- override manual ke
    # build_transform_gen(cfg, is_train=False) (no flip, no scale jitter, resize tetap).
    mapper_val.tfm_gens = build_transform_gen(cfg, False)
    logger.info(f"[FIX V6 bug] mapper_val.tfm_gens override ke deterministik: {mapper_val.tfm_gens}")

    train_loader = build_detection_train_loader(cfg, mapper=mapper_train)
    val_loader   = build_detection_test_loader(
        cfg, cfg.DATASETS.TEST[0], mapper=mapper_val)
    return train_loader, val_loader


def evaluate(maskdino, trunk_head, val_loader, device):
    """Eval full val set."""
    trunk_head.eval()
    maskdino.eval()

    preds_all, gts_all, cats_all = [], [], []
    n_skip = 0; n_total = 0

    with torch.no_grad():
        for batch in val_loader:
            depths_raw = [x.get("depth") for x in batch]
            if any(d is None for d in depths_raw):
                continue

            from detectron2.structures import ImageList
            depths_norm = [d.to(device) for d in depths_raw]
            depths_4ch  = [d.unsqueeze(0).mul(255.0) for d in depths_norm]
            imgs_4ch    = [torch.cat([x["image"].to(device), d], dim=0)
                           for x, d in zip(batch, depths_4ch)]
            imgs_norm   = [(img - maskdino.pixel_mean) / maskdino.pixel_std
                           for img in imgs_4ch]
            img_list    = ImageList.from_tensors(imgs_norm, maskdino.size_divisibility)
            features    = maskdino.backbone(img_list.tensor)
            res2        = features["res2"]

            instances = [x.get("instances") for x in batch]
            for b, inst in enumerate(instances):
                if inst is None or not inst.has("gt_dbh"):
                    continue
                feat_b    = res2[b]
                depth_b   = depths_norm[b]
                masks_b   = inst.gt_masks.to(device)
                dbhs_b    = inst.gt_dbh.to(device)
                classes_b = inst.gt_classes.to(device)

                for j in range(len(inst)):
                    dbh_cm = dbhs_b[j].item()
                    if dbh_cm <= 0:
                        continue
                    n_total += 1
                    result = trunk_head.forward_single(
                        feat_b, depth_b, masks_b[j].bool(),
                        species_idx=classes_b[j].item(), strict=False)
                    if result is None:
                        n_skip += 1
                        continue
                    pred_mm = torch.expm1(result[0]).item()
                    preds_all.append(pred_mm)
                    gts_all.append(dbh_cm * 10.0)
                    cats_all.append(classes_b[j].item() + 1)

    trunk_head.train()

    if not preds_all:
        return {"n_skip": n_skip, "n_total": n_total}

    p = np.array(preds_all); g = np.array(gts_all); c = np.array(cats_all)
    e = np.abs(p - g)
    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    per_species = {}
    for cid in sorted(set(c.tolist())):
        idx = c == cid
        gi = g[idx]; pi = p[idx]; ei = np.abs(gi - pi)
        ss_r = np.sum((gi - pi)**2); ss_t = np.sum((gi - gi.mean())**2)
        r2_c = 1 - ss_r/ss_t if ss_t > 0 else float("nan")
        per_species[int(cid)] = {"N": int(idx.sum()), "R2": float(r2_c), "MAE": float(ei.mean())}

    # min_species_n=30 -- KONSISTEN dgn fix metrik di dbh_sandbox_head.py (12 Jul 2026):
    # R2 pada N kecil sangat tidak stabil, jangan masukkan ke mean/worst_species_R2.
    MIN_SPECIES_N = 30
    included = {cid: s for cid, s in per_species.items() if s["N"] >= MIN_SPECIES_N}
    sp_r2_vals = [s["R2"] for s in included.values()]
    mean_per_species_r2 = float(np.mean(sp_r2_vals)) if sp_r2_vals else float("nan")
    worst_species_r2 = float(np.min(sp_r2_vals)) if sp_r2_vals else float("nan")

    return {
        "N": len(p), "n_skip": n_skip, "n_total": n_total,
        "MAE": float(e.mean()), "RMSE": float(np.sqrt((e**2).mean())),
        "bias": float((p - g).mean()), "R2": float(r2),
        "mean_per_species_R2": mean_per_species_r2, "worst_species_R2": worst_species_r2,
        "n_species_in_mean": len(sp_r2_vals),
        "w50mm": float((e < 50).mean() * 100),
        "w100mm": float((e < 100).mean() * 100),
        "per_species": per_species,
    }


def main(args):
    register_all_rainforests(os.environ.get("RAINFORESTS_ROOT",
        str(PROJECT_ROOT / "data" / "rainforests")))
    register_all_plantations(os.environ.get("PLANTATIONS_ROOT",
        str(PROJECT_ROOT / "data" / "plantations")))
    register_all_combined(os.environ.get("COMBINED_ROOT",
        str(PROJECT_ROOT / "data" / "combined")))

    cfg    = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── MaskDINO backbone (frozen) ───────────────────────────────────────────
    maskdino = build_model(cfg).to(device)
    DetectionCheckpointer(maskdino).load(cfg.MODEL.WEIGHTS)
    maskdino.eval()
    for p in maskdino.parameters():
        p.requires_grad = False
    logger.info("MaskDINO loaded & frozen.")

    # ── Hybrid head v7: strip_rows=5 (sama dgn V6 & baseline sandbox) ───────
    trunk_head = HybridTrunkROIDBHHead(
        in_channels=192, hidden=256,
        strip_rows=5, num_species=13,
        dropout=0.3,
    ).to(device)
    n_params = sum(p.numel() for p in trunk_head.parameters())
    logger.info(f"HybridTrunkROIDBHHead v7: {n_params:,} params, strip_rows=5, dropout=0.3")

    # ── Dataloader ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(cfg)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(trunk_head.parameters(),
                            lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iter, eta_min=args.lr * 0.01)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics_hybrid_v7.json")
    metrics_log  = []

    trunk_head.train()
    iter_loader = iter(train_loader)
    best_r2  = -float("inf")
    best_mae = float("inf")
    n_clean_acc = 0; n_total_acc = 0

    logger.info(f"V7 training: {args.max_iter} iters, lr={args.lr}, wd=5e-4, dropout=0.3, strip_rows=5")
    logger.info("Dataset v3: universe11 spesies (exclude Lemon+LeechVine, blank-DBH>50%), max_depth=20m")
    logger.info("SPECIES_WEIGHT (inverse freq, cap 20, BangaloPalm=1.0):")
    for cid, w in sorted(SPECIES_WEIGHT.items()):
        logger.info(f"  {CAT_NAMES.get(cid,cid):<14} ×{w:.1f}")

    for iteration in range(1, args.max_iter + 1):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        depths_norm = [x["depth"].to(device) for x in batch
                       if x.get("depth") is not None]
        if len(depths_norm) != len(batch):
            continue

        with torch.no_grad():
            from detectron2.structures import ImageList
            depths_4ch = [d.unsqueeze(0).mul(255.0) for d in depths_norm]
            imgs_4ch   = [torch.cat([x["image"].to(device), d], dim=0)
                          for x, d in zip(batch, depths_4ch)]
            imgs_norm  = [(img - maskdino.pixel_mean) / maskdino.pixel_std
                          for img in imgs_4ch]
            img_list  = ImageList.from_tensors(imgs_norm, maskdino.size_divisibility)
            features  = maskdino.backbone(img_list.tensor)
            res2      = features["res2"]

        instances = [x["instances"].to(device) for x in batch]

        optimizer.zero_grad()
        loss, n_clean, n_total = trunk_head.compute_loss(
            res2, depths_norm, instances, CAT_LOG_MEAN, CAT_LOG_STD,
            min_ratio=0.0, max_ratio=float("inf"),
            species_weights=SPECIES_WEIGHT,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trunk_head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        n_clean_acc += n_clean; n_total_acc += n_total

        if iteration % 100 == 0:
            logger.info(f"iter {iteration:5d}/{args.max_iter}  "
                        f"loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
        if iteration % 500 == 0:
            pct = n_clean_acc / max(n_total_acc, 1) * 100
            logger.info(f"  coverage last 500 iters: "
                        f"{n_clean_acc}/{n_total_acc} ({pct:.1f}%) — wh+depth+px")
            n_clean_acc = n_total_acc = 0

        if iteration % args.eval_period == 0 or iteration == args.max_iter:
            latest_ckpt = os.path.join(args.output_dir, "hybrid_v7_latest.pth")
            torch.save(trunk_head.state_dict(), latest_ckpt)

            try:
                metrics = evaluate(maskdino, trunk_head, val_loader, device)
            except Exception as exc:
                logger.error(f"[EVAL iter {iteration}] EXCEPTION: {exc}")
                metrics = {}

            metrics["iteration"] = iteration
            metrics["loss"]      = loss.item()
            metrics_log.append(metrics)
            with open(metrics_path, "w") as f:
                json.dump(metrics_log, f, indent=2)

            if "N" in metrics:
                r2  = metrics["mean_per_species_R2"]
                mae = metrics["MAE"]
                logger.info(
                    f"[EVAL iter {iteration}]  N={metrics['N']}  "
                    f"meanPerSpR2={r2:.4f}  aggR2={metrics['R2']:.4f}  MAE={mae:.1f}mm  "
                    f"±100mm={metrics['w100mm']:.1f}%  "
                    f"skip={metrics['n_skip']}/{metrics['n_total']}")
                for cid, s in metrics.get("per_species", {}).items():
                    logger.info(f"  {CAT_NAMES.get(cid,cid):<12}  "
                                f"N={s['N']:>4}  R²={s['R2']:>7.4f}  MAE={s['MAE']:>7.1f}mm")
                if not math.isnan(r2) and r2 > best_r2:
                    best_r2 = r2; best_mae = mae
                    ckpt = os.path.join(args.output_dir, "hybrid_v7_best.pth")
                    torch.save(trunk_head.state_dict(), ckpt)
                    logger.info(f"  *** New best meanPerSpR2={best_r2:.4f}  MAE={best_mae:.1f}mm → {ckpt}")
            else:
                logger.warning(f"[EVAL iter {iteration}]  skip={metrics.get('n_skip','?')}/"
                                f"{metrics.get('n_total','?')} — semua instance diskip")

    final_ckpt = os.path.join(args.output_dir, "hybrid_v7_final.pth")
    torch.save(trunk_head.state_dict(), final_ckpt)
    logger.info(f"Done. Final: {final_ckpt}")
    logger.info(f"Best val meanPerSpR2={best_r2:.4f}  MAE={best_mae:.1f}mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",   required=True)
    parser.add_argument("--model-weights", default="",
                        help="Path ke model_final.pth FocalNetL RGBD scratch")
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--num-gpus",      type=int, default=1)
    parser.add_argument("--max-iter",      type=int, default=20000)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--eval-period",   type=int, default=1000)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    setup_logger(name="fvcore")
    logger = setup_logger(name="finetune_hybrid_v7")
    logger.info(f"Args: {args}")
    main(args)
