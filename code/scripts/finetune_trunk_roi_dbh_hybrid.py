"""
Hybrid Trunk ROI DBH Training
==============================
Perbedaan dari finetune_trunk_roi_dbh.py:
  1. Pakai HybridTrunkROIDBHHead: backbone res2 + geometric prior + species one-hot
  2. Clean filter di compute_loss: ratio[0.5,2.0] + wh±0.25m + depth[1,10m) + trunk_px≥5
  3. FIX Bug #6: val mapper pakai is_train=True agar gt_dbh tersedia di eval
  4. Fix cat_id offset: gt_classes (0-based) + 1 = cat_id (handled di head)

Jalankan:
  nohup /scratch2/pr65/anur0018/conda_envs/maskdino/bin/python -u \\
      scripts/finetune_trunk_roi_dbh_hybrid.py \\
      --config-file configs/<cfg>.yaml \\
      --model-weights /path/to/model_final.pth \\
      --output-dir /path/to/output \\
      > logs/hybrid_nohup.log 2>&1 &
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
from register_combined import register_all_combined
from register_rainforests import register_all_rainforests
from register_plantations import register_all_plantations

from trunk_roi_dbh_head_hybrid import HybridTrunkROIDBHHead

logger = logging.getLogger("finetune_hybrid")

# Per-species statistics (log1p(DBH_mm), 1-based cat_id)
CAT_LOG_MEAN = {
    1: 3.0991, 2: 2.8020, 3: 2.3825, 4: 3.2736, 5: 2.9299,
    6: 2.7118, 7: 3.1230, 8: 3.9572, 9: 3.1455, 10: 3.8777,
    11: 3.3954, 12: 5.5931, 13: 3.5910,
}
CAT_LOG_STD = {
    1: 0.1347, 2: 0.0531, 3: 0.3175, 4: 0.2660, 5: 0.1826,
    6: 0.0702, 7: 0.1995, 8: 0.2657, 9: 0.4212, 10: 0.2726,
    11: 0.1704, 12: 0.9506, 13: 0.3434,
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
    # BUG #6 FIX: pakai is_train=True agar annotations (termasuk gt_dbh) tetap ada
    mapper_val   = COCOInstanceDBHDatasetMapper(cfg, True)
    train_loader = build_detection_train_loader(cfg, mapper=mapper_train)
    val_loader   = build_detection_test_loader(
        cfg, cfg.DATASETS.TEST[0], mapper=mapper_val)
    return train_loader, val_loader


def evaluate(maskdino, trunk_head, val_loader, device, max_batches=200):
    """Eval pada val set. Pakai strict=False (dengan fallback) untuk coverage lebih tinggi."""
    trunk_head.eval()
    maskdino.eval()

    preds_all, gts_all = [], []
    n_skip  = 0
    n_total = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
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
                    gt_mm = dbh_cm * 10.0

                    result = trunk_head.forward_single(
                        feat_b, depth_b, masks_b[j].bool(),
                        species_idx=classes_b[j].item(), strict=False)

                    if result is None:
                        n_skip += 1
                        continue

                    pred_mm = torch.expm1(result[0]).item()
                    preds_all.append(pred_mm)
                    gts_all.append(gt_mm)

    trunk_head.train()

    if not preds_all:
        return {"n_skip": n_skip, "n_total": n_total}

    import numpy as np
    p  = np.array(preds_all)
    g  = np.array(gts_all)
    e  = np.abs(p - g)
    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "N":      len(p),
        "n_skip": n_skip,
        "n_total": n_total,
        "MAE":    float(e.mean()),
        "RMSE":   float(np.sqrt((e**2).mean())),
        "bias":   float((p - g).mean()),
        "R2":     float(r2),
        "w50mm":  float((e < 50).mean() * 100),
        "w100mm": float((e < 100).mean() * 100),
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

    # ── MaskDINO (frozen) ───────────────────────────────────────────────────
    maskdino = build_model(cfg).to(device)
    DetectionCheckpointer(maskdino).load(cfg.MODEL.WEIGHTS)
    maskdino.eval()
    for p in maskdino.parameters():
        p.requires_grad = False
    logger.info("MaskDINO loaded & fully frozen.")

    # ── Hybrid head ─────────────────────────────────────────────────────────
    trunk_head = HybridTrunkROIDBHHead(in_channels=192, hidden=256,
                                       strip_rows=1, num_species=13).to(device)
    n_params = sum(p.numel() for p in trunk_head.parameters())
    logger.info(f"HybridTrunkROIDBHHead: {n_params:,} trainable parameters.")

    # ── Dataloader ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(cfg)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(trunk_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iter, eta_min=args.lr * 0.01)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics_hybrid.json")
    metrics_log  = []

    trunk_head.train()
    iter_loader = iter(train_loader)
    best_r2     = -float("inf")
    best_mae    = float("inf")
    # Accumulators untuk log coverage (clean instances yang lolos filter)
    n_clean_acc = 0
    n_total_acc = 0

    logger.info(f"Starting Hybrid training for {args.max_iter} iterations...")
    logger.info("Clean filter: ratio[0.5,2.0] + wh±0.25m + depth[1,10m) + trunk_px≥5")

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
            res2, depths_norm, instances, CAT_LOG_MEAN, CAT_LOG_STD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trunk_head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        n_clean_acc += n_clean
        n_total_acc += n_total

        if iteration % 100 == 0:
            logger.info(f"iter {iteration:5d}/{args.max_iter}  "
                        f"loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
        if iteration % 500 == 0:
            pct = n_clean_acc / max(n_total_acc, 1) * 100
            logger.info(f"  clean filter last 500 iters: "
                        f"{n_clean_acc}/{n_total_acc} ({pct:.1f}%) — ratio+wh+depth+px")
            n_clean_acc = n_total_acc = 0

        if iteration % args.eval_period == 0 or iteration == args.max_iter:
            latest_ckpt = os.path.join(args.output_dir, "hybrid_latest.pth")
            torch.save(trunk_head.state_dict(), latest_ckpt)

            try:
                metrics = evaluate(maskdino, trunk_head, val_loader, device)
            except Exception as exc:
                logger.error(f"[EVAL iter {iteration}] evaluate() EXCEPTION: {exc}")
                metrics = {}

            metrics["iteration"] = iteration
            metrics["loss"]      = loss.item()
            metrics_log.append(metrics)
            with open(metrics_path, "w") as f:
                json.dump(metrics_log, f, indent=2)

            if "N" in metrics:
                r2  = metrics["R2"]
                mae = metrics["MAE"]
                logger.info(
                    f"[EVAL iter {iteration}]  N={metrics['N']}  "
                    f"R²={r2:.3f}  MAE={mae:.1f}mm  "
                    f"±100mm={metrics['w100mm']:.1f}%  skip={metrics['n_skip']}/{metrics['n_total']}"
                )
                if r2 > best_r2:
                    best_r2  = r2
                    best_mae = mae
                    ckpt = os.path.join(args.output_dir, "hybrid_best.pth")
                    torch.save(trunk_head.state_dict(), ckpt)
                    logger.info(f"  New best R²={best_r2:.3f}  MAE={best_mae:.1f}mm → {ckpt}")
            else:
                logger.warning(
                    f"[EVAL iter {iteration}]  n_skip={metrics.get('n_skip','?')}/"
                    f"{metrics.get('n_total','?')}  — semua instance diskip di eval")

    final_ckpt = os.path.join(args.output_dir, "hybrid_final.pth")
    torch.save(trunk_head.state_dict(), final_ckpt)
    logger.info(f"Training done. Final: {final_ckpt}")
    logger.info(f"Best val R²={best_r2:.3f}  MAE={best_mae:.1f}mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",   required=True)
    parser.add_argument("--model-weights", default="",
                        help="Path ke model_final.pth FocalNetL RGBD scratch")
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--num-gpus",      type=int, default=1)
    parser.add_argument("--max-iter",      type=int, default=20000)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--eval-period",   type=int, default=2000)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    setup_logger(name="fvcore")
    logger = setup_logger(name="finetune_hybrid")
    logger.info(f"Args: {args}")

    main(args)
