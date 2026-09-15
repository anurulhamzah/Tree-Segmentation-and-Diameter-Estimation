#!/usr/bin/env python3
"""
Visualisasi perbandingan prediksi semua repaired_v4 model vs GT.
Menghasilkan HTML report dengan colored mask overlays.

Usage:
  python visualize_model_comparison.py --output /path/to/reports/model_comparison.html
"""

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pycocotools import mask as mask_util
from pathlib import Path
import io
import base64
from collections import defaultdict
import argparse
import sys

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_BASE = Path("/scratch2/pr65/anur0018/maskdino_output")
COMBINED_ANN  = Path("/scratch2/pr65/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v4/instances_val.json")
PLANT_ANN     = Path("/scratch2/pr65/anur0018/tree_classification/data/plantations/annotations_inst/filtered_rle_f1000_repaired_v4/instances_val.json")
SCORE_THR = 0.30   # threshold optimal untuk kanopi padat

# Selected images: (image_id, description)
# Plantation images (ada di kedua val set → semua 7 model bisa dibandingkan)
PLANTATION_IDS = [
    (48,  "Plantation — Apple + Loquat (9 trees)"),
    (617, "Plantation — Loquat + Orange (5 trees)"),
    (826, "Plantation — Dense Apple (12 trees)"),
]
# Rainforest images (combined val only → 6 combined models)
RAINFOREST_IDS = [
    (300202, "Rainforest — All 6 categories (24 trees)"),
    (300833, "Rainforest — All 6 categories (20 trees)"),
]

# ── Model definitions ─────────────────────────────────────────────────────────
# Format: (run_dir_name, display_name, dataset, pred_json_path_relative_to_eval_best)
MODELS = [
    {
        "name": "SwinB Plantations RGB",
        "short": "SwinB<br>Plantations",
        "dir":  "SwinB_pl_rle_f1000_repaired_v4_75k",
        "dataset": "plantation",  # only for plantation images
        "ann_file": str(PLANT_ANN),
    },
    {
        "name": "FocalNet-L Combined RGBD",
        "short": "FocalNet-L<br>RGBD",
        "dir":  "FocalNet_L_combined_rle_f1000_rgbd_scratch_repaired_v4_75k",
        "dataset": "combined",
        "ann_file": str(COMBINED_ANN),
    },
    {
        "name": "SwinL Combined RGBD",
        "short": "SwinL<br>RGBD",
        "dir":  "SwinL_combined_rle_f1000_rgbd_scratch_repaired_v4_75k",
        "dataset": "combined",
        "ann_file": str(COMBINED_ANN),
    },
    {
        "name": "SwinB Combined RGB (pretrained)",
        "short": "SwinB<br>RGB",
        "dir":  "SwinB_combined_rle_f1000_repaired_v4_75k",
        "dataset": "combined",
        "ann_file": str(COMBINED_ANN),
    },
    {
        "name": "SwinB Combined RGBD (scratch)",
        "short": "SwinB<br>RGBD scratch",
        "dir":  "SwinB_combined_rle_f1000_rgbd_scratch_repaired_v4_75k",
        "dataset": "combined",
        "ann_file": str(COMBINED_ANN),
    },
    {
        "name": "R50 Combined RGB",
        "short": "R50<br>RGB",
        "dir":  "R50_combined_rle_f1000_repaired_v4_75k",
        "dataset": "combined",
        "ann_file": str(COMBINED_ANN),
    },
    {
        "name": "SwinB Combined RGB (scratch)",
        "short": "SwinB<br>RGB scratch",
        "dir":  "SwinB_combined_rle_f1000_rgb_scratch_repaired_v4_75k",
        "dataset": "combined",
        "ann_file": str(COMBINED_ANN),
    },
]

# ── Colors per category ───────────────────────────────────────────────────────
CATEGORY_COLORS = {
    # Plantation
    "Apple":        (220,  50,  50),   # red
    "Lemon":        (255, 200,   0),   # yellow
    "Loquat":       (200, 100, 200),   # purple
    "Mango":        (255, 140,   0),   # orange
    "Orange":       (255, 165,   0),   # light orange
    "Persimmon":    (180,  80,   0),   # dark orange
    "Pomegranate":  (160,   0,  80),   # dark red
    # Rainforest
    "AliiFig":      ( 50, 180,  50),   # green
    "BangaloPalm":  ( 30, 130, 200),   # blue
    "Fern":         (100, 200, 100),   # light green
    "LeechVine":    (  0, 160, 160),   # teal
    "RubberFig":    ( 80,  80, 200),   # indigo
    "Umbrella":     (180, 120,  50),   # brown
    # fallback
    "__unknown__":  (128, 128, 128),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def decode_rle(seg, h, w):
    """Decode COCO RLE or polygon to binary mask."""
    if isinstance(seg, dict):
        return mask_util.decode(seg).astype(bool)
    else:
        from pycocotools import mask as mask_util2
        rle = mask_util2.frPyObjects(seg, h, w)
        merged = mask_util2.merge(rle)
        return mask_util2.decode(merged).astype(bool)


def overlay_masks(rgb_img, instances, cat_dict, alpha=0.50):
    """
    instances: list of dict with keys: mask (bool HxW), cat_name (str), label (str)
    Returns PIL Image with colored mask overlays.
    """
    img_arr = np.array(rgb_img).copy()
    h, w = img_arr.shape[:2]
    overlay = img_arr.copy().astype(float)

    for inst in instances:
        mask = inst["mask"]
        color = CATEGORY_COLORS.get(inst["cat_name"], CATEGORY_COLORS["__unknown__"])
        for c in range(3):
            overlay[:, :, c][mask] = (
                alpha * color[c] + (1 - alpha) * overlay[:, :, c][mask]
            )

    result = overlay.clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(result)
    draw = ImageDraw.Draw(pil)

    # Draw labels
    for inst in instances:
        mask = inst["mask"]
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        text = inst["label"]
        # Small white background for readability
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 10)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x0, y0 = max(0, cx - tw//2), max(0, cy - th//2)
        draw.rectangle([x0-1, y0-1, x0+tw+1, y0+th+1], fill=(0, 0, 0, 180))
        draw.text((x0, y0), text, fill=(255, 255, 255), font=font)

    return pil


def render_gt(img_path, img_info, anns, cat_dict):
    """Render GT masks on image."""
    rgb = Image.open(img_path).convert("RGB")
    h, w = rgb.size[1], rgb.size[0]
    instances = []
    for ann in anns:
        cat_name = cat_dict.get(ann["category_id"], "?")
        try:
            m = decode_rle(ann["segmentation"], h, w)
        except Exception:
            continue
        instances.append({
            "mask": m,
            "cat_name": cat_name,
            "label": cat_name[:3],
        })
    return overlay_masks(rgb, instances, cat_dict)


def render_pred(img_path, img_id, preds_by_img, cat_dict, score_thr=SCORE_THR):
    """Render model predictions on image."""
    rgb = Image.open(img_path).convert("RGB")
    h, w = rgb.size[1], rgb.size[0]
    preds = preds_by_img.get(img_id, [])
    preds = [p for p in preds if p["score"] >= score_thr]
    # Sort by score descending (draw highest confidence last → on top)
    preds = sorted(preds, key=lambda p: p["score"])
    instances = []
    for p in preds:
        cat_name = cat_dict.get(p["category_id"], "?")
        try:
            m = decode_rle(p["segmentation"], h, w)
        except Exception:
            continue
        instances.append({
            "mask": m,
            "cat_name": cat_name,
            "label": f"{cat_name[:3]} {p['score']:.2f}",
        })
    n_pred = len(instances)
    pil = overlay_masks(rgb, instances, cat_dict)
    return pil, n_pred


def pil_to_b64(img, scale=1.0, quality=85):
    """Convert PIL image to base64 JPEG string (smaller than PNG for photos)."""
    if scale != 1.0:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "jpg", base64.b64encode(buf.getvalue()).decode()


def legend_html(categories):
    """Generate color legend HTML."""
    items = []
    for cat in categories:
        rgb = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["__unknown__"])
        hex_c = "#{:02x}{:02x}{:02x}".format(*rgb)
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin:2px 8px 2px 0;">'
            f'<span style="width:14px;height:14px;background:{hex_c};border-radius:3px;'
            f'display:inline-block;margin-right:4px;border:1px solid #ccc;"></span>'
            f'<span style="font-size:12px;color:#374151;">{cat}</span></span>'
        )
    return "".join(items)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/scratch2/pr65/anur0018/tree_classification/reports/model_comparison.html")
    parser.add_argument("--score-thr", type=float, default=SCORE_THR)
    args = parser.parse_args()

    score_thr = args.score_thr
    print(f"Score threshold: {score_thr}")

    # Load annotations
    print("Loading annotations...")
    with open(COMBINED_ANN) as f:
        ann_cb = json.load(f)
    with open(PLANT_ANN) as f:
        ann_pl = json.load(f)

    cb_img_dict  = {img["id"]: img for img in ann_cb["images"]}
    pl_img_dict  = {img["id"]: img for img in ann_pl["images"]}
    cb_cat_dict  = {c["id"]: c["name"] for c in ann_cb["categories"]}
    pl_cat_dict  = {c["id"]: c["name"] for c in ann_pl["categories"]}

    cb_ann_by_img = defaultdict(list)
    for a in ann_cb["annotations"]:
        cb_ann_by_img[a["image_id"]].append(a)
    pl_ann_by_img = defaultdict(list)
    for a in ann_pl["annotations"]:
        pl_ann_by_img[a["image_id"]].append(a)

    # Load predictions per model
    print("Loading model predictions...")
    for m in MODELS:
        pred_path = OUTPUT_BASE / m["dir"] / "eval_best" / "inference" / "coco_instances_results.json"
        if not pred_path.exists():
            print(f"  WARNING: predictions not found for {m['name']}: {pred_path}")
            m["preds"] = {}
            continue
        with open(pred_path) as f:
            preds = json.load(f)
        preds_by_img = defaultdict(list)
        for p in preds:
            preds_by_img[p["image_id"]].append(p)
        m["preds"] = preds_by_img
        print(f"  Loaded {len(preds)} preds for {m['name']}")

    # All plantation category names
    all_pl_cats = [c["name"] for c in ann_pl["categories"]]
    all_rf_cats = [c["name"] for c in ann_cb["categories"] if c["name"] not in all_pl_cats]

    # ── Generate image panels ────────────────────────────────────────────────
    sections = []

    def process_image_set(img_id_list, image_set_title, use_models, img_dict, ann_by_img, cat_dict, cat_names, gt_label="GT"):
        section_rows = []
        for img_id, desc in img_id_list:
            if img_id not in img_dict:
                print(f"  SKIP: image {img_id} not in {gt_label} annotations")
                continue
            img_info = img_dict[img_id]
            img_path = img_info["file_name"]
            if not Path(img_path).exists():
                print(f"  SKIP: image file not found: {img_path}")
                continue
            print(f"\n  Image {img_id}: {img_path.split('/')[-1]} — {desc}")

            anns = ann_by_img[img_id]
            print(f"    GT: {len(anns)} instances")

            # Render GT
            gt_pil = render_gt(img_path, img_info, anns, cat_dict)
            _, gt_b64 = pil_to_b64(gt_pil, scale=1.0)

            # Render each model
            model_cells = []
            for m in use_models:
                if m["dataset"] == "plantation" and "plantation" not in img_path.lower():
                    # Plantations model on rainforest image → N/A
                    model_cells.append({"b64": None, "n_pred": 0, "na": True})
                    continue
                preds_by_img = m["preds"]
                # Map image: use combined ann cat_dict for combined models
                if m["dataset"] == "plantation":
                    pred_cat_dict = {c["id"]: c["name"] for c in ann_pl["categories"]}
                else:
                    pred_cat_dict = cat_dict
                pred_pil, n_pred = render_pred(img_path, img_id, preds_by_img, pred_cat_dict, score_thr)
                _, pred_b64 = pil_to_b64(pred_pil, scale=1.0)
                model_cells.append({"b64": pred_b64, "n_pred": n_pred, "na": False})
                print(f"    {m['short'].replace('<br>', ' ')}: {n_pred} predictions")

            section_rows.append({
                "img_id": img_id,
                "desc": desc,
                "fname": img_path.split("/")[-1],
                "gt_b64": gt_b64,
                "gt_n": len(anns),
                "model_cells": model_cells,
            })
        return section_rows

    print("\n=== Processing Plantation images ===")
    pl_rows = process_image_set(
        PLANTATION_IDS,
        "Plantation Images — All 7 Models",
        MODELS,
        cb_img_dict, cb_ann_by_img, cb_cat_dict, all_pl_cats,
    )

    print("\n=== Processing Rainforest images ===")
    cb_only_models = [m for m in MODELS if m["dataset"] == "combined"]
    rf_rows = process_image_set(
        RAINFOREST_IDS,
        "Rainforest Images — Combined Models Only",
        cb_only_models,
        cb_img_dict, cb_ann_by_img, cb_cat_dict, all_rf_cats,
    )

    # ── Build HTML ───────────────────────────────────────────────────────────
    print("\nBuilding HTML...")

    def model_header_row(use_models):
        cells = ['<th style="background:#f1f5f9;padding:6px 8px;font-size:11px;font-weight:700;color:#1e293b;border:1px solid #e2e8f0;min-width:200px;">Image</th>',
                 '<th style="background:#dcfce7;padding:6px 8px;font-size:11px;font-weight:700;color:#166534;border:1px solid #e2e8f0;min-width:200px;">Ground Truth</th>']
        for m in use_models:
            cells.append(
                f'<th style="background:#eff6ff;padding:6px 8px;font-size:11px;font-weight:600;color:#1e3a8a;border:1px solid #e2e8f0;min-width:200px;">'
                f'{m["short"].replace(chr(10), " ")}</th>'
            )
        return "<tr>" + "".join(cells) + "</tr>"

    def row_html(row, use_models):
        # Info cell
        info_cell = (
            f'<td style="vertical-align:top;padding:8px;border:1px solid #e2e8f0;background:#fafafa;">'
            f'<div style="font-size:11px;font-weight:600;color:#374151;">{row["fname"]}</div>'
            f'<div style="font-size:10px;color:#6b7280;margin-top:2px;">{row["desc"]}</div>'
            f'<div style="font-size:10px;color:#9ca3af;margin-top:2px;">id={row["img_id"]}</div>'
            f'</td>'
        )
        # GT cell
        gt_cell = (
            f'<td style="vertical-align:top;padding:4px;border:1px solid #e2e8f0;background:#f0fdf4;">'
            f'<img src="data:image/jpeg;base64,{row["gt_b64"]}" '
            f'style="width:100%;display:block;border-radius:4px;">'
            f'<div style="text-align:center;font-size:10px;color:#166534;margin-top:3px;">'
            f'GT: {row["gt_n"]} instances</div>'
            f'</td>'
        )
        # Model cells
        model_tds = []
        for i, cell in enumerate(row["model_cells"]):
            if cell.get("na"):
                model_tds.append(
                    '<td style="vertical-align:middle;text-align:center;padding:4px;border:1px solid #e2e8f0;background:#f9fafb;">'
                    '<span style="color:#9ca3af;font-size:11px;">N/A</span></td>'
                )
            else:
                color = "#1e40af"
                model_tds.append(
                    f'<td style="vertical-align:top;padding:4px;border:1px solid #e2e8f0;background:#f8fafc;">'
                    f'<img src="data:image/jpeg;base64,{cell["b64"]}" '
                    f'style="width:100%;display:block;border-radius:4px;">'
                    f'<div style="text-align:center;font-size:10px;color:{color};margin-top:3px;">'
                    f'{cell["n_pred"]} predictions</div>'
                    f'</td>'
                )
        return "<tr>" + info_cell + gt_cell + "".join(model_tds) + "</tr>"

    def section_html(title, rows, use_models):
        if not rows:
            return f'<p style="color:#9ca3af;">No images in section: {title}</p>'
        header = model_header_row(use_models)
        row_htmls = [row_html(r, use_models) for r in rows]
        table = (
            f'<table style="border-collapse:collapse;width:100%;table-layout:fixed;">'
            + header
            + "".join(row_htmls)
            + "</table>"
        )
        return (
            f'<div style="margin-bottom:40px;">'
            f'<h2 style="font-size:18px;font-weight:700;color:#1e293b;margin:0 0 12px 0;">{title}</h2>'
            + table +
            f'</div>'
        )

    pl_legend = legend_html(all_pl_cats)
    rf_legend = legend_html(all_rf_cats)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Repaired_v4 — Model Prediction Comparison</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 24px; }}
  h1   {{ font-size: 22px; font-weight: 800; color: #0f172a; margin: 0 0 4px 0; }}
  .sub {{ font-size: 13px; color: #64748b; margin-bottom: 24px; }}
  .meta-bar {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:28px; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; }}
  .badge-blue {{ background:#dbeafe; color:#1e40af; }}
  .badge-green {{ background:#dcfce7; color:#166534; }}
  .badge-gray {{ background:#f1f5f9; color:#475569; }}
  .legend {{ background:white; border:1px solid #e2e8f0; border-radius:8px; padding:10px 14px; margin-bottom:20px; }}
  .section {{ margin-bottom:40px; }}
  table img {{ border-radius:4px; }}
  .note {{ background:#fef9c3; border:1px solid #fde68a; border-radius:6px; padding:10px 14px; font-size:12px; color:#92400e; margin-bottom:20px; }}
</style>
</head>
<body>
<h1>Repaired_v4 — Model Prediction Comparison</h1>
<p class="sub">Generated 2026-06-27 &nbsp;|&nbsp; Score threshold: {score_thr:.2f} &nbsp;|&nbsp; Mask: segmentation (COCO RLE)</p>

<div class="meta-bar">
  <span class="badge badge-blue">7 Models</span>
  <span class="badge badge-green">5 Sample Images</span>
  <span class="badge badge-gray">Val Set: repaired_v4</span>
  <span class="badge badge-gray">Threshold: {score_thr:.2f}</span>
</div>

<div class="note">
  <strong>Catatan:</strong> Setiap kolom adalah prediksi model yang berbeda pada gambar yang sama.
  Mask diwarnai per kategori (lihat legend). Label = kategori + confidence score.
  SwinB Plantations tidak dijalankan pada rainforest images (N/A).
</div>

<div class="legend">
  <div style="font-size:11px;font-weight:700;color:#374151;margin-bottom:6px;">Plantation Categories</div>
  {pl_legend}
  <div style="font-size:11px;font-weight:700;color:#374151;margin:8px 0 6px 0;">Rainforest Categories</div>
  {rf_legend}
</div>

{section_html("Plantation Images — Semua 7 Model", pl_rows, MODELS)}
{section_html("Rainforest Images — Combined Models (6)", rf_rows, cb_only_models)}

<p style="font-size:11px;color:#94a3b8;margin-top:32px;">
  Training index: <a href="repaired_v4_training_index.html" style="color:#3b82f6;">repaired_v4_training_index.html</a>
</p>
</body>
</html>
"""

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
