#!/usr/bin/env python3
"""
DBH Evaluation Script — MaskDINO Tree Classification

Evaluates DBH regression quality (MAE, MRE, R²) for E5 (RGB+DBH) and E6v2 (RGBD+DBH).
Re-runs inference and extracts pred_dbh directly from the model decoder output,
since coco_instances_results.json does not save custom regression predictions.

Usage (run from PROJECT_ROOT):
    python scripts/eval_dbh.py --experiment E5
    python scripts/eval_dbh.py --experiment E6v2
    python scripts/eval_dbh.py --experiment E5 --split test
    python scripts/eval_dbh.py --experiment E5 E6v2  (both at once)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
OUTPUT_ROOT = PROJECT_ROOT.parent / "maskdino_output"
REPORT_DIR = PROJECT_ROOT / "reports" / "dbh_eval"

sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import COCOInstanceDBHDatasetMapper
from detectron2.structures import ImageList
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config
from pycocotools import mask as maskUtils

from maskdino import add_maskdino_config
from register_rainforests import register_all_rainforests

# ── Experiment registry ───────────────────────────────────────────────────────

EXPERIMENTS = {
    "E5": {
        "output_dir": "SwinT_rf_rle_f1000_dbh",
        "config": "maskdino_SwinT_rf_rle_f1000_dbh.yaml",
        "checkpoint": "model_0044999.pth",  # best AP50 @44999
        "label": "SwinT RGB+DBH (E5)",
    },
    "E6v2": {
        "output_dir": "SwinT_rf_rle_f1000_rgbd_dbh_75k",
        "config": "maskdino_SwinT_rf_rle_f1000_rgbd_dbh_75k.yaml",
        "checkpoint": "model_0069999.pth",  # best AP50 @69999
        "label": "SwinT RGBD+DBH (E6v2)",
    },
}

SCORE_THRESH = 0.5
IOU_THRESH   = 0.5

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_cfg(exp_key):
    exp = EXPERIMENTS[exp_key]
    out_dir     = OUTPUT_ROOT / exp["output_dir"]
    config_path = PROJECT_ROOT / "configs" / exp["config"]
    ckpt_path   = out_dir / exp["checkpoint"]

    assert config_path.exists(), f"Config not found: {config_path}"
    assert ckpt_path.exists(),   f"Checkpoint not found: {ckpt_path}"

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR = str(out_dir)
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(ckpt_path))
    model.eval()
    print(f"[{exp_key}] Model loaded from {ckpt_path.name} on {cfg.MODEL.DEVICE}")
    return model, cfg

# ── Data loading ──────────────────────────────────────────────────────────────

def build_loader(cfg, split="val"):
    dataset_name = cfg.DATASETS.TEST[0]
    if split == "test":
        dataset_name = dataset_name.replace("_val", "_test")
    print(f"[data] Loading dataset: {dataset_name}")
    # Test mapper — image only, no GT instances (annotations di-drop)
    mapper = COCOInstanceDBHDatasetMapper(cfg, is_train=False)
    return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def build_gt_index(cfg, split="val"):
    """Load GT annotations (class, dbh, mask RLE) per image_id from annotation JSON."""
    import json
    from detectron2.data import MetadataCatalog
    dataset_name = cfg.DATASETS.TEST[0]
    if split == "test":
        dataset_name = dataset_name.replace("_val", "_test")
    json_file = MetadataCatalog.get(dataset_name).json_file
    with open(json_file) as f:
        coco = json.load(f)
    # cat_id → 0-indexed class
    cat_id_to_cls = {c["id"]: i for i, c in enumerate(coco["categories"])}
    gt_index = {}  # image_id → list of {class, dbh, rle}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid not in gt_index:
            gt_index[iid] = []
        seg = ann.get("segmentation")
        if seg is None:
            continue
        # RLE or polygon — store as-is for maskUtils.iou
        if isinstance(seg, dict):
            rle = seg
        else:
            # polygon — encode via maskUtils
            img_info = next(im for im in coco["images"] if im["id"] == iid)
            rle = maskUtils.frPyObjects(seg, img_info["height"], img_info["width"])
            rle = maskUtils.merge(rle)
        gt_index[iid].append({
            "class": cat_id_to_cls.get(ann["category_id"], -1),
            "dbh":   float(ann.get("dbh", 0.0) or 0.0),
            "rle":   rle,
        })
    return gt_index

# ── Inference with DBH extraction ─────────────────────────────────────────────

@torch.no_grad()
def infer_with_dbh(model, data_loader, gt_index):
    """
    Runs inference and returns per-image dicts containing:
        image_id, pred_scores, pred_classes, pred_masks_rle, pred_dbh_mm,
        gt_dbh, gt_classes, gt_masks_rle
    GT is loaded from gt_index (built from annotation JSON) to avoid
    test-mapper dropping annotations.
    """
    device   = next(model.parameters()).device
    n_cls    = model.sem_seg_head.num_classes
    topk     = model.test_topk_per_image
    is_rgbd  = (model.pixel_mean.shape[0] == 4)
    results  = []

    for i, batch in enumerate(data_loader):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1} images...")

        inp = batch[0]

        # ── Forward pass (handles both RGB 3-ch and RGBD 4-ch) ───────────────
        imgs = []
        for x in batch:
            img = x["image"].to(device).float()  # (3, H, W)
            if is_rgbd:
                depth = x.get("depth")
                if depth is not None:
                    depth = depth.to(device).float().unsqueeze(0)  # (1, H, W)
                else:
                    depth = torch.zeros(1, img.shape[1], img.shape[2], device=device)
                img = torch.cat([img, depth], dim=0)  # (4, H, W)
            imgs.append(img)
        imgs = [(x - model.pixel_mean) / model.pixel_std for x in imgs]
        images = ImageList.from_tensors(imgs, model.size_divisibility)
        features = model.backbone(images.tensor)
        outputs, _ = model.sem_seg_head(features)

        # Capture pred_dbh BEFORE outputs is cleared
        pred_dbh_all = outputs.get("pred_dbh", None)  # (1, Q)

        mask_cls  = outputs["pred_logits"][0]   # (Q, C)
        mask_pred = outputs["pred_masks"][0]     # (Q, H', W')
        mask_box  = outputs["pred_boxes"][0]     # (Q, 4)

        img_size  = images.image_sizes[0]
        height    = inp.get("height", img_size[0])
        width     = inp.get("width",  img_size[1])

        # Upsample masks to padded image size
        mask_pred = F.interpolate(
            mask_pred.unsqueeze(0),
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear", align_corners=False,
        )[0]

        # ── Replicate topk selection from instance_inference ─────────────────
        scores = mask_cls.sigmoid()  # (Q, C)
        labels_all = (
            torch.arange(n_cls, device=device)
            .unsqueeze(0).repeat(topk, 1).flatten(0, 1)
        )
        # Need Q*C labels; repeat per query
        labels_flat = (
            torch.arange(n_cls, device=device)
            .unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1)
        )
        scores_flat = scores.flatten()
        scores_topk, topk_idx = scores_flat.topk(topk, sorted=True)
        labels_topk = labels_flat[topk_idx]
        query_idx   = topk_idx // n_cls  # map (Q*C) → Q

        # Filter by score threshold
        keep = scores_topk >= SCORE_THRESH
        scores_topk  = scores_topk[keep].cpu()
        labels_topk  = labels_topk[keep].cpu()
        query_idx_k  = query_idx[keep]

        if len(query_idx_k) == 0:
            results.append({
                "image_id":     inp["image_id"],
                "pred_scores":  np.array([], dtype=np.float32),
                "pred_classes": np.array([], dtype=np.int64),
                "pred_rle":     [],
                "pred_dbh_mm":  np.array([], dtype=np.float32),
                "gt_classes":   np.array([], dtype=np.int64),
                "gt_dbh":       np.array([], dtype=np.float32),
                "gt_rle":       [],
            })
            continue

        # ── Crop masks to original image size ────────────────────────────────
        h_pad, w_pad = mask_pred.shape[-2:]
        scale_h = h_pad / img_size[0]
        scale_w = w_pad / img_size[1]
        # crop back to un-padded size
        mask_pred_crop = mask_pred[query_idx_k, :img_size[0], :img_size[1]]  # (K, H, W)
        # resize to original
        mask_pred_out = F.interpolate(
            mask_pred_crop.unsqueeze(0),
            size=(height, width),
            mode="bilinear", align_corners=False,
        )[0]  # (K, H_orig, W_orig)
        pred_masks_bin = (mask_pred_out > 0).cpu().numpy().astype(np.uint8)

        # ── DBH predictions ──────────────────────────────────────────────────
        if pred_dbh_all is not None:
            pred_dbh_log = pred_dbh_all[0][query_idx_k].cpu()  # (K,)
            # Model predicts log1p(dbh_mm); invert
            pred_dbh_mm = torch.expm1(pred_dbh_log).clamp(min=0).numpy()
        else:
            pred_dbh_mm = np.zeros(len(query_idx_k))

        # ── Encode masks to RLE for IoU computation ──────────────────────────
        pred_rle = [maskUtils.encode(np.asfortranarray(m)) for m in pred_masks_bin]

        # ── GT annotations — dari JSON index, bukan test loader ──────────────
        image_id = inp["image_id"]
        gt_anns  = gt_index.get(image_id, [])
        if gt_anns:
            gt_classes  = np.array([a["class"] for a in gt_anns], dtype=np.int64)
            gt_dbh_vals = np.array([a["dbh"]   for a in gt_anns], dtype=np.float32)
            gt_rle      = [a["rle"] for a in gt_anns]
        else:
            gt_classes  = np.array([], dtype=np.int64)
            gt_dbh_vals = np.array([], dtype=np.float32)
            gt_rle      = []

        results.append({
            "image_id":     inp["image_id"],
            "pred_scores":  scores_topk.numpy(),
            "pred_classes": labels_topk.numpy(),
            "pred_rle":     pred_rle,
            "pred_dbh_mm":  pred_dbh_mm,
            "gt_classes":   gt_classes,
            "gt_dbh":       gt_dbh_vals,
            "gt_rle":       gt_rle,
        })

    return results

# ── Greedy IoU matching ───────────────────────────────────────────────────────

def match_predictions(results, category_names):
    """
    Greedy match: sort preds by score desc, match to best unmatched GT (IoU ≥ threshold).
    Returns list of matched pairs: (pred_dbh_mm, gt_dbh_mm, category_name)
    Only includes pairs where gt_dbh > 0 (DBH annotated).
    """
    matched_pairs = []  # (pred_dbh, gt_dbh, cat_name)

    for r in results:
        n_gt   = len(r["gt_rle"])
        n_pred = len(r["pred_rle"])
        if n_gt == 0 or n_pred == 0:
            continue

        gt_matched = np.zeros(n_gt, dtype=bool)

        # Predictions already sorted by score (topk sorted=True)
        for p_idx in range(n_pred):
            if len(r["gt_rle"]) == 0:
                break
            # Compute IoU vs all unmatched GTs
            ious = maskUtils.iou(
                [r["pred_rle"][p_idx]],
                r["gt_rle"],
                [0] * n_gt,
            )[0]  # (n_gt,)
            ious[gt_matched] = 0.0  # ignore already matched

            best_gt = int(np.argmax(ious))
            if ious[best_gt] >= IOU_THRESH:
                gt_matched[best_gt] = True
                gt_dbh = r["gt_dbh"][best_gt]
                if gt_dbh > 0:  # only instances with annotated DBH
                    cat_idx  = int(r["gt_classes"][best_gt])
                    cat_name = category_names[cat_idx] if cat_idx < len(category_names) else str(cat_idx)
                    matched_pairs.append((
                        float(r["pred_dbh_mm"][p_idx]),
                        float(gt_dbh),
                        cat_name,
                    ))

    return matched_pairs

# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(pairs):
    """Returns dict with MAE, MRE, RMSE, R², N for a list of (pred, gt, cat) pairs."""
    if len(pairs) == 0:
        return {"N": 0, "MAE": None, "MRE": None, "RMSE": None, "R2": None}

    pred = np.array([p[0] for p in pairs])
    gt   = np.array([p[1] for p in pairs])

    errors = pred - gt
    abs_errors = np.abs(errors)
    mae   = float(np.mean(abs_errors))
    mre   = float(np.mean(abs_errors / (gt + 1e-6)))
    rmse  = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
    r2    = 1.0 - ss_res / (ss_tot + 1e-10)

    return {"N": len(pairs), "MAE": mae, "MRE_pct": mre * 100, "RMSE": rmse, "R2": r2}

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_scatter(pairs, exp_label, out_path):
    pred = np.array([p[0] for p in pairs])
    gt   = np.array([p[1] for p in pairs])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"DBH Evaluation — {exp_label}", fontsize=13)

    # Scatter: pred vs GT
    ax = axes[0]
    ax.scatter(gt, pred, alpha=0.3, s=12, color="steelblue")
    lim = max(gt.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="y=x (perfect)")
    ax.set_xlabel("GT DBH (mm)")
    ax.set_ylabel("Pred DBH (mm)")
    ax.set_title("Pred vs GT DBH")
    ax.legend(fontsize=9)

    # Error histogram
    ax = axes[1]
    errors = pred - gt
    ax.hist(errors, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", lw=1.5, linestyle="--")
    ax.set_xlabel("Pred − GT (mm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution  MAE={np.mean(np.abs(errors)):.1f} mm")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter plot: {out_path}")


def plot_per_class(pairs, category_names, exp_label, out_path):
    from collections import defaultdict
    by_class = defaultdict(list)
    for pred, gt, cat in pairs:
        by_class[cat].append((pred, gt))

    cats = sorted(by_class.keys())
    maes = [np.mean(np.abs(np.array([p[0] for p in by_class[c]]) -
                            np.array([p[1] for p in by_class[c]])))
            for c in cats]
    ns   = [len(by_class[c]) for c in cats]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(cats, maes, color="steelblue", edgecolor="white")
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("MAE (mm)")
    ax.set_title(f"Per-Class MAE — {exp_label}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved per-class plot: {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_experiment(exp_key, split="val"):
    print(f"\n{'='*60}")
    print(f"  Evaluating {exp_key}: {EXPERIMENTS[exp_key]['label']}")
    print(f"  Split: {split}")
    print(f"{'='*60}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.environ["RAINFORESTS_ROOT"] = str(
        PROJECT_ROOT / "data" / "rainforests"
    )
    register_all_rainforests(os.environ["RAINFORESTS_ROOT"])

    model, cfg = load_model_and_cfg(exp_key)

    # Category names from dataset
    from detectron2.data import MetadataCatalog
    dataset_name = cfg.DATASETS.TEST[0]
    if split == "test":
        dataset_name = dataset_name.replace("_val", "_test")
    meta = MetadataCatalog.get(dataset_name)
    category_names = meta.thing_classes if hasattr(meta, "thing_classes") else []
    print(f"  Categories ({len(category_names)}): {category_names}")

    loader   = build_loader(cfg, split)
    gt_index = build_gt_index(cfg, split)

    print(f"  Running inference ({len(loader)} images)...")
    results = infer_with_dbh(model, loader, gt_index)
    print(f"  Inference done. Total images: {len(results)}")

    # Match and compute metrics
    pairs = match_predictions(results, category_names)
    print(f"  Matched pairs (with DBH annotation): {len(pairs)}")

    metrics_overall = compute_metrics(pairs)
    print(f"\n  === OVERALL METRICS ({exp_key}) ===")
    print(f"  N matched (with DBH): {metrics_overall['N']}")
    if metrics_overall["N"] > 0:
        print(f"  MAE:  {metrics_overall['MAE']:.2f} mm")
        print(f"  MRE:  {metrics_overall['MRE_pct']:.2f} %")
        print(f"  RMSE: {metrics_overall['RMSE']:.2f} mm")
        print(f"  R²:   {metrics_overall['R2']:.4f}")

    # Per-class metrics
    from collections import defaultdict
    by_class = defaultdict(list)
    for pred, gt, cat in pairs:
        by_class[cat].append((pred, gt))

    per_class = {}
    print(f"\n  === PER-CLASS METRICS ===")
    for cat in sorted(by_class.keys()):
        cat_pairs = [(p, g, cat) for p, g in by_class[cat]]
        m = compute_metrics(cat_pairs)
        per_class[cat] = m
        print(f"  {cat:<20} N={m['N']:4d}  MAE={m['MAE']:.2f}mm  "
              f"MRE={m['MRE_pct']:.1f}%  R²={m['R2']:.3f}")

    # Save JSON report
    report = {
        "experiment": exp_key,
        "label": EXPERIMENTS[exp_key]["label"],
        "checkpoint": EXPERIMENTS[exp_key]["checkpoint"],
        "split": split,
        "score_thresh": SCORE_THRESH,
        "iou_thresh": IOU_THRESH,
        "overall": metrics_overall,
        "per_class": per_class,
    }
    report_path = REPORT_DIR / f"dbh_metrics_{exp_key}_{split}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved metrics: {report_path}")

    # Plots
    if len(pairs) > 0:
        exp_label = EXPERIMENTS[exp_key]["label"]
        plot_scatter(
            pairs, exp_label,
            REPORT_DIR / f"dbh_scatter_{exp_key}_{split}.png"
        )
        plot_per_class(
            pairs, category_names, exp_label,
            REPORT_DIR / f"dbh_per_class_{exp_key}_{split}.png"
        )

    return report


def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Evaluate DBH regression quality")
    parser.add_argument("--experiment", nargs="+", choices=list(EXPERIMENTS.keys()),
                        default=["E5", "E6v2"], help="Which experiments to evaluate")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Dataset split to evaluate on")
    args = parser.parse_args()

    all_reports = {}
    for exp_key in args.experiment:
        report = evaluate_experiment(exp_key, args.split)
        all_reports[exp_key] = report

    # Summary comparison
    if len(all_reports) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*60}")
        header = f"  {'Model':<25} {'N':>6} {'MAE':>8} {'MRE':>8} {'R²':>8}"
        print(header)
        print("  " + "-" * 57)
        for exp_key, rep in all_reports.items():
            m = rep["overall"]
            if m["N"] > 0:
                print(f"  {rep['label']:<25} {m['N']:>6} "
                      f"{m['MAE']:>7.1f}mm {m['MRE_pct']:>7.1f}%  {m['R2']:>7.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
