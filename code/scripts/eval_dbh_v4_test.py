#!/usr/bin/env python3
"""
DBH Evaluation — SwinB Combined RGBD+DBH v4 Stratified, TEST split

Usage:
    cd /scratch2/pr65/anur0018/tree_classification
    nohup python scripts/eval_dbh_v4_test.py > /tmp/eval_dbh_v4_test.log 2>&1 &
"""

import json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
OUTPUT_ROOT   = PROJECT_ROOT.parent / "maskdino_output"
REPORT_DIR    = PROJECT_ROOT / "reports" / "dbh_eval"

sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.structures import ImageList
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config
from pycocotools import mask as maskUtils

from maskdino import add_maskdino_config
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import COCOInstanceDBHDatasetMapper
from register_combined import register_all_combined

EXPERIMENTS = {
    "swinb": {
        "name":         "SwinB_combined_rgbd_dbh_v4_stratified_75k",
        "label":        "Swin-B RGBD Pretrained + Head DBH (v4 stratified, 75k)",
        "output_dir":   "SwinB_combined_rle_f1000_rgbd_dbh_repaired_v4_stratified_75k",
        "checkpoint":   "model_final.pth",
        "dataset_test": "combined_inst_rle_f1000_repaired_v4_stratified_test",
    },
    "focalnetl": {
        "name":         "FocalNetL_combined_rgbd_dbh_finetune_v4_stratified_20k",
        "label":        "FocalNet-L RGBD Scratch + DBH Fine-tune (v4 stratified, 20k)",
        "output_dir":   "FocalNet_L_combined_rle_f1000_rgbd_scratch_dbh_finetune_v4_stratified_20k_lr1e3",
        "checkpoint":   "model_final.pth",
        "dataset_test": "combined_inst_rle_f1000_repaired_v4_stratified_test",
    },
    "joint": {
        "name":         "FocalNetL_dbh_joint_plainext_20k",
        "label":        "FocalNet-L plain_ext20k init + JOINT DBH (backbone unfrozen, 20k)",
        "output_dir":   "FocalNet_L_dbh_joint_plainext_20k",
        "checkpoint":   "model_final.pth",
        "dataset_test": "combined_inst_rle_f1000_repaired_v4_stratified_test",
    },
    "frozen": {
        "name":         "FocalNetL_dbh_frozen_plainext_20k",
        "label":        "FocalNet-L plain_ext20k init + FROZEN DBH (backbone beku, 20k)",
        "output_dir":   "FocalNet_L_dbh_frozen_plainext_20k",
        "checkpoint":   "model_final.pth",
        "dataset_test": "combined_inst_rle_f1000_repaired_v4_stratified_test",
    },
}
# Default; override via --model {swinb,focalnetl}
EXPERIMENT = EXPERIMENTS["swinb"]

SCORE_THRESH = 0.35
IOU_THRESH   = 0.50


def load_model(exp):
    out_dir  = OUTPUT_ROOT / exp["output_dir"]
    cfg_path = out_dir / "config.yaml"
    ckpt     = out_dir / exp["checkpoint"]

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(cfg_path))
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR   = str(out_dir)
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(ckpt))
    model.eval()
    print(f"Loaded {ckpt.name} on {cfg.MODEL.DEVICE}")
    return model, cfg


def build_gt_index(dataset_name):
    from detectron2.data import MetadataCatalog
    json_file = MetadataCatalog.get(dataset_name).json_file
    coco = json.load(open(json_file))
    cat_id_to_cls = {c["id"]: i for i, c in enumerate(coco["categories"])}
    gt_index = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        gt_index.setdefault(iid, [])
        seg = ann.get("segmentation")
        if seg is None:
            continue
        if isinstance(seg, dict):
            rle = seg
        else:
            img_info = next(im for im in coco["images"] if im["id"] == iid)
            rle = maskUtils.frPyObjects(seg, img_info["height"], img_info["width"])
            rle = maskUtils.merge(rle)
        gt_index[iid].append({
            "class": cat_id_to_cls.get(ann["category_id"], -1),
            "dbh":   float(ann.get("dbh") or 0.0),
            "rle":   rle,
        })
    return gt_index


@torch.no_grad()
def infer_with_dbh(model, data_loader, gt_index):
    device = next(model.parameters()).device
    n_cls  = model.sem_seg_head.num_classes
    topk   = model.test_topk_per_image
    results = []

    for i, batch in enumerate(data_loader):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(data_loader)} images...", flush=True)

        inp  = batch[0]
        imgs = []
        for x in batch:
            img = x["image"].to(device).float()
            depth = x.get("depth")
            if depth is not None:
                depth = depth.to(device).float().unsqueeze(0)
            else:
                depth = torch.zeros(1, img.shape[1], img.shape[2], device=device)
            imgs.append(torch.cat([img, depth], dim=0))

        imgs   = [(x - model.pixel_mean) / model.pixel_std for x in imgs]
        images = ImageList.from_tensors(imgs, model.size_divisibility)
        feats  = model.backbone(images.tensor)
        outputs, _ = model.sem_seg_head(feats)

        pred_dbh_all = outputs.get("pred_dbh", None)
        mask_cls  = outputs["pred_logits"][0]
        mask_pred = outputs["pred_masks"][0]

        img_size = images.image_sizes[0]
        height   = inp.get("height", img_size[0])
        width    = inp.get("width",  img_size[1])

        mask_pred = F.interpolate(
            mask_pred.unsqueeze(0),
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear", align_corners=False,
        )[0]

        scores      = mask_cls.sigmoid()
        labels_flat = torch.arange(n_cls, device=device).unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1)
        scores_flat = scores.flatten()
        scores_topk, topk_idx = scores_flat.topk(topk, sorted=True)
        labels_topk = labels_flat[topk_idx]
        query_idx   = topk_idx // n_cls

        keep        = scores_topk >= SCORE_THRESH
        scores_topk = scores_topk[keep].cpu()
        labels_topk = labels_topk[keep].cpu()
        query_idx_k = query_idx[keep]

        if len(query_idx_k) == 0:
            results.append({
                "image_id": inp["image_id"],
                "pred_scores": np.array([], dtype=np.float32),
                "pred_classes": np.array([], dtype=np.int64),
                "pred_rle": [],
                "pred_dbh_mm": np.array([], dtype=np.float32),
                "gt_classes": np.array([], dtype=np.int64),
                "gt_dbh": np.array([], dtype=np.float32),
                "gt_rle": [],
            })
            continue

        mask_pred_crop = mask_pred[query_idx_k, :img_size[0], :img_size[1]]
        mask_pred_out  = F.interpolate(
            mask_pred_crop.unsqueeze(0),
            size=(height, width),
            mode="bilinear", align_corners=False,
        )[0]
        pred_masks_bin = (mask_pred_out > 0).cpu().numpy().astype(np.uint8)

        if pred_dbh_all is not None:
            pred_dbh_log = pred_dbh_all[0][query_idx_k].cpu()
            pred_dbh_mm  = torch.expm1(pred_dbh_log).clamp(min=0).numpy()
        else:
            pred_dbh_mm = np.zeros(len(query_idx_k))

        pred_rle = [maskUtils.encode(np.asfortranarray(m)) for m in pred_masks_bin]

        gt_anns = gt_index.get(inp["image_id"], [])
        results.append({
            "image_id":     inp["image_id"],
            "pred_scores":  scores_topk.numpy(),
            "pred_classes": labels_topk.numpy(),
            "pred_rle":     pred_rle,
            "pred_dbh_mm":  pred_dbh_mm,
            "gt_classes":   np.array([a["class"] for a in gt_anns], dtype=np.int64),
            "gt_dbh":       np.array([a["dbh"] for a in gt_anns], dtype=np.float32),
            "gt_rle":       [a["rle"] for a in gt_anns],
        })

    return results


def match_predictions(results, category_names):
    pairs = []
    for r in results:
        if not r["gt_rle"] or not r["pred_rle"]:
            continue
        gt_matched = np.zeros(len(r["gt_rle"]), dtype=bool)
        for p_idx in range(len(r["pred_rle"])):
            ious = maskUtils.iou([r["pred_rle"][p_idx]], r["gt_rle"], [0] * len(r["gt_rle"]))[0]
            ious[gt_matched] = 0.0
            best_gt = int(np.argmax(ious))
            if ious[best_gt] >= IOU_THRESH:
                gt_matched[best_gt] = True
                gt_dbh = r["gt_dbh"][best_gt]
                if gt_dbh > 0:
                    cat_idx  = int(r["gt_classes"][best_gt])
                    cat_name = category_names[cat_idx] if cat_idx < len(category_names) else str(cat_idx)
                    pairs.append((float(r["pred_dbh_mm"][p_idx]), float(gt_dbh), cat_name))
    return pairs


def compute_metrics(pairs):
    if not pairs:
        return {"N": 0, "MAE": None, "MRE_pct": None, "RMSE": None, "R2": None,
                "bias": None, "w50mm": None, "w100mm": None}
    pred = np.array([p[0] for p in pairs])
    gt   = np.array([p[1] for p in pairs])
    err  = pred - gt
    mae  = float(np.mean(np.abs(err)))
    mre  = float(np.mean(np.abs(err) / (gt + 1e-6))) * 100
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2   = float(1.0 - ss_res / (ss_tot + 1e-10))
    # Metrik standar literatur DBH (bias = mean signed error; within-X-mm = akurasi threshold)
    bias   = float(np.mean(err))
    w50mm  = float(np.mean(np.abs(err) <= 50)) * 100
    w100mm = float(np.mean(np.abs(err) <= 100)) * 100
    return {"N": len(pairs), "MAE": mae, "MRE_pct": mre, "RMSE": rmse, "R2": r2,
            "bias": bias, "w50mm": w50mm, "w100mm": w100mm}


def plot_results(pairs, label, suffix=""):
    pred = np.array([p[0] for p in pairs])
    gt   = np.array([p[1] for p in pairs])
    errors = pred - gt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"DBH Evaluation — {label}", fontsize=12)

    ax = axes[0]
    ax.scatter(gt, pred, alpha=0.3, s=10, color="steelblue")
    lim = max(gt.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="y=x (perfect)")
    ax.set_xlabel("GT DBH (mm)"); ax.set_ylabel("Pred DBH (mm)")
    ax.set_title("Pred vs GT DBH"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(errors, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", lw=1.5, linestyle="--")
    ax.set_xlabel("Pred − GT (mm)"); ax.set_ylabel("Count")
    ax.set_title(f"Error  MAE={np.mean(np.abs(errors)):.1f} mm  bias={np.mean(errors):.1f} mm")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = REPORT_DIR / f"dbh_scatter_v4_test{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Per-class MAE
    from collections import defaultdict
    by_class = defaultdict(list)
    for p, g, c in pairs:
        by_class[c].append((p, g))
    cats = sorted(by_class.keys())
    maes = [np.mean(np.abs(np.array([x[0] for x in by_class[c]]) - np.array([x[1] for x in by_class[c]]))) for c in cats]
    ns   = [len(by_class[c]) for c in cats]

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(cats, maes, color="steelblue", edgecolor="white")
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("MAE (mm)"); ax.set_title(f"Per-Class MAE — {label}")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    out2 = REPORT_DIR / f"dbh_per_class_v4_test{suffix}.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(EXPERIMENTS.keys()), default="swinb")
    args = ap.parse_args()
    exp = EXPERIMENTS[args.model]

    setup_logger()
    os.makedirs(REPORT_DIR, exist_ok=True)

    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_combined(os.environ["COMBINED_ROOT"])

    print(f"\n{exp['label']}")
    print(f"Dataset: {exp['dataset_test']}")
    print(f"Score threshold: {SCORE_THRESH}  |  IoU threshold: {IOU_THRESH}")

    model, cfg = load_model(exp)

    from detectron2.data import MetadataCatalog
    category_names = MetadataCatalog.get(exp["dataset_test"]).thing_classes
    print(f"Classes ({len(category_names)}): {category_names}\n")

    loader   = build_detection_test_loader(
        cfg, exp["dataset_test"],
        mapper=COCOInstanceDBHDatasetMapper(cfg, is_train=False)
    )
    gt_index = build_gt_index(exp["dataset_test"])
    print(f"Running inference on {len(loader)} images...", flush=True)

    results = infer_with_dbh(model, loader, gt_index)
    pairs   = match_predictions(results, category_names)
    print(f"\nMatched TP pairs with DBH annotation: {len(pairs)}")

    overall = compute_metrics(pairs)
    print(f"\nOVERALL:")
    print(f"  N     = {overall['N']}")
    print(f"  MAE   = {overall['MAE']:.2f} mm")
    print(f"  MRE   = {overall['MRE_pct']:.1f} %")
    print(f"  RMSE  = {overall['RMSE']:.2f} mm")
    print(f"  R²    = {overall['R2']:.4f}")

    from collections import defaultdict
    by_class = defaultdict(list)
    for p, g, c in pairs:
        by_class[c].append((p, g, c))

    per_class = {}
    print(f"\nPER-CLASS:")
    for cat in sorted(by_class.keys()):
        m = compute_metrics(by_class[cat])
        per_class[cat] = m
        print(f"  {cat:<20} N={m['N']:4d}  MAE={m['MAE']:.1f} mm  MRE={m['MRE_pct']:.1f}%  R²={m['R2']:.3f}")

    report = {
        "experiment":   exp["name"],
        "label":        exp["label"],
        "dataset":      exp["dataset_test"],
        "score_thresh": SCORE_THRESH,
        "iou_thresh":   IOU_THRESH,
        "overall":      overall,
        "per_class":    per_class,
    }
    out_json = REPORT_DIR / f"dbh_metrics_v4_test_{args.model}.json"
    json.dump(report, open(out_json, "w"), indent=2)
    print(f"\nSaved: {out_json}")

    if pairs:
        plot_results(pairs, exp["label"], suffix=f"_{args.model}")

    print("\nDone.")


if __name__ == "__main__":
    main()
