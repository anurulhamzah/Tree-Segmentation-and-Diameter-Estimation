#!/usr/bin/env python3
"""
Trunk ROI DBH Fine-tune
═══════════════════════
Melatih TrunkROIDBHHead di atas FocalNet-L RGBD Scratch yang dibekukan.

Perbedaan vs finetune_focal_dbh.py:
  - Input ke DBH head: strip feature backbone res2 (C=192) di row h=1.3m
  - Bukan: query embedding + class logits
  - Backbone + seluruh MaskDINO: FROZEN sepenuhnya
  - Hanya TrunkROIDBHHead (MLP kecil) yang dilatih

Usage:
    cd /scratch2/pr65/anur0018/tree_classification
    python scripts/finetune_trunk_roi_dbh.py \
        --config-file configs/maskdino_FocalNet_L_combined_rle_f1000_rgbd_scratch_dbh_finetune_v4_stratified_20k_lr1e3.yaml \
        --model-weights /scratch2/pr65/anur0018/maskdino_output/FocalNet_L_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k/model_final.pth \
        --output-dir /scratch2/pr65/anur0018/maskdino_output/FocalNet_L_trunk_roi_dbh_v4_stratified_20k \
        --num-gpus 1
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(HERE))

from register_combined import register_all_combined
from register_rainforests import register_all_rainforests
from register_plantations import register_all_plantations

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.modeling import build_model

from maskdino import add_maskdino_config
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import COCOInstanceDBHDatasetMapper

from trunk_roi_dbh_head import TrunkROIDBHHead

logger = logging.getLogger("finetune_trunk_roi_dbh")

# Per-species DBH statistics (log-space, dari V4 train set) — sama dengan finetune_focal_dbh.py
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
    mapper_val   = COCOInstanceDBHDatasetMapper(cfg, False)
    train_loader = build_detection_train_loader(cfg, mapper=mapper_train)
    val_loader   = build_detection_test_loader(
        cfg, cfg.DATASETS.TEST[0], mapper=mapper_val)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(maskdino, trunk_head, val_loader, device, max_batches=200):
    """Hitung MAE dan coverage di val set."""
    trunk_head.eval()
    maskdino.eval()

    preds_all, gts_all = [], []
    n_skip = 0
    n_total = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        images = [x["image"].to(device) for x in batch]
        depths_raw = [x.get("depth") for x in batch]
        if any(d is None for d in depths_raw):
            continue

        # Preprocess (same as maskdino.forward but we keep depth separate)
        depths_norm = [d.to(device) for d in depths_raw]
        depths_4ch  = [d.unsqueeze(0).mul(255.0) for d in depths_norm]
        imgs_4ch    = [torch.cat([img.to(device), d], dim=0)
                       for img, d in zip(images, depths_4ch)]
        imgs_norm   = [(x - maskdino.pixel_mean) / maskdino.pixel_std
                       for x in imgs_4ch]
        from detectron2.structures import ImageList
        img_list = ImageList.from_tensors(imgs_norm, maskdino.size_divisibility)
        features = maskdino.backbone(img_list.tensor)
        res2 = features["res2"]   # (B, 192, H_f, W_f)

        instances = [x.get("instances") for x in batch]
        for b, inst in enumerate(instances):
            if inst is None or not inst.has("gt_dbh"):
                continue
            feat_b  = res2[b]
            depth_b = depths_norm[b]
            masks_b = inst.gt_masks.to(device)
            dbhs_b  = inst.gt_dbh.to(device)

            for j in range(len(inst)):
                dbh_cm = dbhs_b[j].item()
                if dbh_cm <= 0:
                    continue
                n_total += 1
                gt_mm = dbh_cm * 10.0
                pred  = trunk_head.forward_single(feat_b, depth_b, masks_b[j].bool())
                if pred is None:
                    n_skip += 1
                    continue
                pred_mm = torch.expm1(pred).item()
                preds_all.append(pred_mm)
                gts_all.append(gt_mm)

    trunk_head.train()
    if not preds_all:
        return {"n_skip": n_skip, "n_total": n_total}   # tanpa 'N' → caller tahu eval kosong

    import numpy as np
    p = np.array(preds_all)
    g = np.array(gts_all)
    e = np.abs(p - g)
    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "N":        len(p),
        "n_skip":   n_skip,
        "MAE":      float(e.mean()),
        "RMSE":     float(np.sqrt((e**2).mean())),
        "bias":     float((p - g).mean()),
        "R2":       float(r2),
        "w50mm":    float((e < 50).mean() * 100),
        "w100mm":   float((e < 100).mean() * 100),
    }


def main(args):
    # Dataset registration
    register_all_rainforests(os.environ.get("RAINFORESTS_ROOT",
        str(PROJECT_ROOT / "data" / "rainforests")))
    register_all_plantations(os.environ.get("PLANTATIONS_ROOT",
        str(PROJECT_ROOT / "data" / "plantations")))
    register_all_combined(os.environ.get("COMBINED_ROOT",
        str(PROJECT_ROOT / "data" / "combined")))

    cfg    = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build & freeze MaskDINO ──────────────────────────────────────────────
    maskdino = build_model(cfg).to(device)
    DetectionCheckpointer(maskdino).load(cfg.MODEL.WEIGHTS)
    maskdino.eval()
    for p in maskdino.parameters():
        p.requires_grad = False
    logger.info("MaskDINO loaded & fully frozen.")

    # ── Build TrunkROIDBHHead ────────────────────────────────────────────────
    trunk_head = TrunkROIDBHHead(in_channels=192, hidden=256, strip_rows=1).to(device)
    n_params = sum(p.numel() for p in trunk_head.parameters())
    logger.info(f"TrunkROIDBHHead: {n_params:,} trainable parameters.")

    # ── Dataloader ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(cfg)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(trunk_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iter, eta_min=args.lr * 0.01)

    # ── Output dir ───────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics_trunk_roi.json")
    metrics_log  = []

    # ── Training loop ────────────────────────────────────────────────────────
    trunk_head.train()
    iter_loader  = iter(train_loader)
    best_mae     = float("inf")
    n_valid_acc  = 0   # instance valid (lolos forward_single) sejak log terakhir
    n_total_acc  = 0   # total instance dengan dbh>0 sejak log terakhir

    logger.info(f"Starting training for {args.max_iter} iterations...")

    for iteration in range(1, args.max_iter + 1):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        # Preprocess (replicate maskdino forward preamble)
        depths_norm = [x["depth"].to(device) for x in batch
                       if x.get("depth") is not None]
        if len(depths_norm) != len(batch):
            continue

        with torch.no_grad():
            depths_4ch = [d.unsqueeze(0).mul(255.0) for d in depths_norm]
            imgs_4ch   = [torch.cat([x["image"].to(device), d], dim=0)
                          for x, d in zip(batch, depths_4ch)]
            imgs_norm  = [(img - maskdino.pixel_mean) / maskdino.pixel_std
                          for img in imgs_4ch]
            from detectron2.structures import ImageList
            img_list  = ImageList.from_tensors(imgs_norm, maskdino.size_divisibility)
            features  = maskdino.backbone(img_list.tensor)
            res2      = features["res2"]   # (B, 192, H_f, W_f)

        instances = [x["instances"].to(device) for x in batch]

        optimizer.zero_grad()
        loss, n_valid, n_total = trunk_head.compute_loss(
            res2, depths_norm, instances, CAT_LOG_MEAN, CAT_LOG_STD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trunk_head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        n_valid_acc += n_valid
        n_total_acc += n_total

        if iteration % 100 == 0:
            logger.info(f"iter {iteration:5d}/{args.max_iter}  "
                        f"loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
        if iteration % 500 == 0:
            cov = n_valid_acc / max(n_total_acc, 1) * 100
            logger.info(f"  coverage last 500 iters: {n_valid_acc}/{n_total_acc} "
                        f"({cov:.1f}%) instances valid")
            n_valid_acc = n_total_acc = 0

        if iteration % args.eval_period == 0 or iteration == args.max_iter:
            # Simpan checkpoint periodic dulu sebelum eval (safety net)
            latest_ckpt = os.path.join(args.output_dir, "trunk_roi_latest.pth")
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
                logger.info(
                    f"[EVAL iter {iteration}]  N={metrics['N']}  "
                    f"MAE={metrics['MAE']:.1f}mm  R²={metrics['R2']:.3f}  "
                    f"±100mm={metrics['w100mm']:.1f}%  skip={metrics['n_skip']}"
                )
                if metrics["MAE"] < best_mae:
                    best_mae = metrics["MAE"]
                    ckpt = os.path.join(args.output_dir, "trunk_roi_best.pth")
                    torch.save(trunk_head.state_dict(), ckpt)
                    logger.info(f"  New best MAE={best_mae:.1f}mm → saved {ckpt}")
            else:
                logger.warning(
                    f"[EVAL iter {iteration}]  n_skip={metrics.get('n_skip', '?')}/"
                    f"{metrics.get('n_total', '?')}  — semua instance diskip"
                )

    # Final save
    final_ckpt = os.path.join(args.output_dir, "trunk_roi_final.pth")
    torch.save(trunk_head.state_dict(), final_ckpt)
    logger.info(f"Training done. Final checkpoint: {final_ckpt}")
    logger.info(f"Best val MAE: {best_mae:.1f}mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",   required=True)
    parser.add_argument("--model-weights", default="",
                        help="Path to FocalNetL RGBD scratch model_final.pth")
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--num-gpus",      type=int, default=1)
    parser.add_argument("--max-iter",      type=int, default=20000)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--eval-period",   type=int, default=2000)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    setup_logger(name="fvcore")
    logger = setup_logger(name="finetune_trunk_roi_dbh")
    logger.info(f"Args: {args}")

    main(args)
