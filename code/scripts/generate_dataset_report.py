#!/usr/bin/env python3
"""
Generate HTML dataset summary report.
Usage: python generate_dataset_report.py --output reports/dataset_summary.html
"""

import json
import argparse
from pathlib import Path
from collections import Counter

DATA_ROOT = Path("/scratch2/pr65/anur0018/tree_classification/data")

DATASETS = {
    "Plantations": {
        "path": DATA_ROOT / "plantations/annotations_inst/filtered_rle_f1000_repaired_v4_stratified",
        "color": "#16a34a",
        "bg": "#f0fdf4",
        "badge_bg": "#dcfce7",
        "badge_txt": "#166534",
        "desc": "7 orchard species, RGB + depth, Unreal Engine synthetic",
        "note": "",
    },
    "Rainforests": {
        "path": DATA_ROOT / "rainforests/annotation_inst/filtered_rle_f1000_repaired_v4_stratified",
        "color": "#0369a1",
        "bg": "#f0f9ff",
        "badge_bg": "#dbeafe",
        "badge_txt": "#1e40af",
        "desc": "6 tropical species, RGB + depth, Unreal Engine synthetic",
        "note": "",
    },
    "Combined": {
        "path": DATA_ROOT / "combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified",
        "color": "#7c3aed",
        "bg": "#faf5ff",
        "badge_bg": "#ede9fe",
        "badge_txt": "#5b21b6",
        "desc": "13 species (Plantations + Rainforests), RGB + depth",
        "note": "",
    },
}

SPECIES_COLORS = {
    "Apple": "#dc2626", "Lemon": "#ca8a04", "Loquat": "#9333ea",
    "Mango": "#ea580c", "Orange": "#f97316", "Persimmon": "#b45309",
    "Pomegranate": "#be185d", "AliiFig": "#16a34a", "BangaloPalm": "#0284c7",
    "Fern": "#15803d", "LeechVine": "#0e7490", "RubberFig": "#4f46e5",
    "Umbrella": "#92400e",
}


def load_split(base: Path, split: str):
    path = base / f"instances_{split}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def bar(val, max_val, color, width=120):
    w = int(val / max_val * width) if max_val else 0
    return (
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<div style="width:{w}px;height:10px;background:{color};border-radius:3px;'
        f'flex-shrink:0;"></div>'
        f'<span style="font-size:11px;color:#374151;">{val:,}</span>'
        f'</div>'
    )


def pct_badge(val, total, warn=False):
    p = val / total * 100 if total else 0
    color = "#b45309" if warn else "#374151"
    bg = "#fef9c3" if warn else "transparent"
    return f'<span style="font-size:10px;color:{color};background:{bg};padding:1px 4px;border-radius:3px;">{p:.1f}%</span>'


def species_section(ds_name, ds_info):
    base = ds_info["path"]
    color = ds_info["color"]

    splits_data = {}
    for split in ("train", "val", "test"):
        d = load_split(base, split)
        if d is None:
            return f'<p style="color:red;">Split files not found at {base}</p>'
        splits_data[split] = d

    cat_dict = {c["id"]: c["name"] for c in splits_data["train"]["categories"]}
    counts = {
        split: Counter(a["category_id"] for a in splits_data[split]["annotations"])
        for split in ("train", "val", "test")
    }
    img_counts = {split: len(splits_data[split]["images"]) for split in ("train", "val", "test")}
    ann_counts = {split: sum(counts[split].values()) for split in ("train", "val", "test")}

    total_imgs = sum(img_counts.values())
    total_anns = sum(ann_counts.values())

    # Max per-category for bar scaling
    max_ann = max(
        counts["train"].get(cid, 0) + counts["val"].get(cid, 0) + counts["test"].get(cid, 0)
        for cid in cat_dict
    )

    # Per-species rows
    rows = ""
    for cid, name in sorted(cat_dict.items()):
        tr = counts["train"].get(cid, 0)
        va = counts["val"].get(cid, 0)
        te = counts["test"].get(cid, 0)
        tot = tr + va + te
        sp_color = SPECIES_COLORS.get(name, "#6b7280")
        warn_va = va < 50
        warn_te = te < 50
        dot = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{sp_color};margin-right:6px;flex-shrink:0;"></span>'
        rows += f"""
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:6px 8px;white-space:nowrap;">{dot}<span style="font-size:12px;font-weight:500;color:#1e293b;">{name}</span></td>
          <td style="padding:6px 8px;">{bar(tr, max_ann, sp_color)}</td>
          <td style="padding:6px 8px;text-align:right;font-size:12px;color:#374151;">{va:,} {pct_badge(va, tot, warn_va)}</td>
          <td style="padding:6px 8px;text-align:right;font-size:12px;color:#374151;">{te:,} {pct_badge(te, tot, warn_te)}</td>
          <td style="padding:6px 8px;text-align:right;font-size:12px;font-weight:600;color:#0f172a;">{tot:,}</td>
        </tr>"""

    # Totals row
    rows += f"""
        <tr style="background:#f8fafc;font-weight:700;border-top:2px solid #e2e8f0;">
          <td style="padding:7px 8px;font-size:12px;color:#1e293b;">TOTAL</td>
          <td style="padding:7px 8px;font-size:12px;color:#374151;">{ann_counts['train']:,} instances</td>
          <td style="padding:7px 8px;text-align:right;font-size:12px;color:#374151;">{ann_counts['val']:,} {pct_badge(ann_counts['val'], total_anns)}</td>
          <td style="padding:7px 8px;text-align:right;font-size:12px;color:#374151;">{ann_counts['test']:,} {pct_badge(ann_counts['test'], total_anns)}</td>
          <td style="padding:7px 8px;text-align:right;font-size:12px;color:#0f172a;">{total_anns:,}</td>
        </tr>
        <tr style="background:#f8fafc;">
          <td style="padding:5px 8px;font-size:11px;color:#6b7280;">Images</td>
          <td style="padding:5px 8px;font-size:11px;color:#6b7280;">{img_counts['train']:,}</td>
          <td style="padding:5px 8px;text-align:right;font-size:11px;color:#6b7280;">{img_counts['val']:,} {pct_badge(img_counts['val'], total_imgs)}</td>
          <td style="padding:5px 8px;text-align:right;font-size:11px;color:#6b7280;">{img_counts['test']:,} {pct_badge(img_counts['test'], total_imgs)}</td>
          <td style="padding:5px 8px;text-align:right;font-size:11px;color:#6b7280;">{total_imgs:,}</td>
        </tr>"""

    badge_bg  = ds_info["badge_bg"]
    badge_txt = ds_info["badge_txt"]

    return f"""
<div style="background:white;border:1px solid #e2e8f0;border-radius:10px;
            overflow:hidden;margin-bottom:28px;box-shadow:0 1px 3px rgba(0,0,0,.06);">
  <div style="background:{ds_info['bg']};border-bottom:2px solid {color};padding:14px 20px;">
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <h2 style="font-size:16px;font-weight:700;color:{color};margin:0;">{ds_name}</h2>
      <span style="background:{badge_bg};color:{badge_txt};padding:2px 8px;border-radius:10px;
                   font-size:11px;font-weight:600;">{len(cat_dict)} species</span>
      <span style="background:{badge_bg};color:{badge_txt};padding:2px 8px;border-radius:10px;
                   font-size:11px;font-weight:600;">{total_imgs:,} images</span>
      <span style="background:{badge_bg};color:{badge_txt};padding:2px 8px;border-radius:10px;
                   font-size:11px;font-weight:600;">{total_anns:,} instances</span>
      <span style="background:#dcfce7;color:#166534;padding:2px 8px;border-radius:10px;
                   font-size:11px;font-weight:600;">✓ scene-stratified</span>
    </div>
    <p style="font-size:12px;color:#64748b;margin:6px 0 0 0;">{ds_info['desc']}</p>
  </div>
  <div style="overflow-x:auto;padding:0 4px 4px 4px;">
    <table style="width:100%;border-collapse:collapse;min-width:500px;">
      <thead>
        <tr style="background:#f8fafc;border-bottom:1px solid #e2e8f0;">
          <th style="padding:7px 8px;text-align:left;font-size:11px;font-weight:700;
                     color:#64748b;white-space:nowrap;">Species</th>
          <th style="padding:7px 8px;text-align:left;font-size:11px;font-weight:700;color:#64748b;">
            Train (instances)</th>
          <th style="padding:7px 8px;text-align:right;font-size:11px;font-weight:700;color:#64748b;">Val</th>
          <th style="padding:7px 8px;text-align:right;font-size:11px;font-weight:700;color:#64748b;">Test</th>
          <th style="padding:7px 8px;text-align:right;font-size:11px;font-weight:700;color:#64748b;">Total</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


def overview_card(ds_name, ds_info):
    base = ds_info["path"]
    total_imgs, total_anns, n_cats = 0, 0, 0
    for split in ("train", "val", "test"):
        d = load_split(base, split)
        if d:
            total_imgs += len(d["images"])
            total_anns += len(d["annotations"])
            n_cats = len(d["categories"])
    color = ds_info["color"]
    return f"""
<div style="flex:1;min-width:200px;background:white;border:1px solid #e2e8f0;
            border-top:4px solid {color};border-radius:8px;padding:16px 20px;
            box-shadow:0 1px 3px rgba(0,0,0,.06);">
  <div style="font-size:14px;font-weight:700;color:{color};margin-bottom:8px;">{ds_name}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div><div style="font-size:11px;color:#94a3b8;">Species</div>
         <div style="font-size:20px;font-weight:700;color:#0f172a;">{n_cats}</div></div>
    <div><div style="font-size:11px;color:#94a3b8;">Images</div>
         <div style="font-size:20px;font-weight:700;color:#0f172a;">{total_imgs:,}</div></div>
    <div><div style="font-size:11px;color:#94a3b8;">Instances</div>
         <div style="font-size:20px;font-weight:700;color:#0f172a;">{total_anns:,}</div></div>
    <div><div style="font-size:11px;color:#94a3b8;">Split</div>
         <div style="font-size:12px;font-weight:600;color:#16a34a;margin-top:4px;">70/15/15</div></div>
  </div>
</div>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/scratch2/pr65/anur0018/tree_classification/reports/dataset_summary.html")
    args = parser.parse_args()

    overview_cards = "".join(overview_card(n, i) for n, i in DATASETS.items())
    species_sections = "".join(species_section(n, i) for n, i in DATASETS.items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Summary — Repaired V4 Stratified</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#f8fafc; color:#1e293b; font-family:'Segoe UI',system-ui,sans-serif;
          font-size:14px; line-height:1.5; padding:28px 32px; }}
  h1   {{ font-size:21px; font-weight:800; color:#0f172a; }}
  .sub {{ font-size:12px; color:#64748b; margin-top:4px; margin-bottom:24px; }}
</style>
</head>
<body>

<h1>Dataset Summary — Repaired V4 Stratified Split</h1>
<p class="sub">
  Annotation: repaired_v4 (mask quality fix + area filter f1000) &nbsp;·&nbsp;
  Split: scene-stratified 70/15/15 &nbsp;·&nbsp;
  0 scene overlap antara train / val / test &nbsp;·&nbsp;
  Generated: 2026-06-27
</p>

<!-- Overview cards -->
<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:28px;">
  {overview_cards}
</div>

<!-- Split strategy note -->
<div style="background:#fefce8;border:1px solid #fde68a;border-radius:8px;
            padding:12px 16px;margin-bottom:28px;font-size:12px;color:#713f12;">
  <strong>Split Strategy:</strong>
  Gambar dikelompokkan per scene (Tree ID).
  Scene-level assignment memastikan tidak ada pohon yang sama muncul di lebih dari satu split.
  Distribusi spesies dipertahankan proporsional antar split (~70% train, ~15% val, ~15% test).
  <br><br>
  <strong>⚠ Catatan Pomegranate:</strong>
  Total hanya 199 instance di seluruh dataset — val=30, test=31.
  AP Pomegranate memiliki variance tinggi dan harus diinterpretasikan dengan hati-hati.
  Ini adalah keterbatasan data, bukan keterbatasan metodologi split.
</div>

<!-- Per-dataset species breakdown -->
<h2 style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:16px;">
  Per-Dataset Species Distribution
</h2>
{species_sections}

<!-- Image size info -->
<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;
            padding:14px 20px;margin-bottom:28px;">
  <h3 style="font-size:13px;font-weight:700;color:#1e293b;margin-bottom:10px;">Image Properties</h3>
  <table style="border-collapse:collapse;font-size:12px;">
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:5px 16px 5px 0;color:#64748b;font-weight:600;">Resolution</td>
      <td style="padding:5px 0;">480 × 270 px (RGB + depth PFM)</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:5px 16px 5px 0;color:#64748b;font-weight:600;">Source</td>
      <td style="padding:5px 0;">Unreal Engine 5 (above-ground synthetic)</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:5px 16px 5px 0;color:#64748b;font-weight:600;">Annotation</td>
      <td style="padding:5px 0;">COCO RLE instance segmentation, area ≥ 1000 px²</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:5px 16px 5px 0;color:#64748b;font-weight:600;">Depth</td>
      <td style="padding:5px 0;">PFM format, jarak kamera ke pohon (bukan elevation)</td>
    </tr>
    <tr>
      <td style="padding:5px 16px 5px 0;color:#64748b;font-weight:600;">Mask repair</td>
      <td style="padding:5px 0;">v4 — 1.3% mask diperbaiki (union COCO + semantic segmentation)</td>
    </tr>
  </table>
</div>

<p style="font-size:11px;color:#94a3b8;">
  Training index: <a href="repaired_v4_training_index.html" style="color:#3b82f6;">repaired_v4_training_index.html</a>
  &nbsp;·&nbsp;
  Model comparison: <a href="model_comparison.html" style="color:#3b82f6;">model_comparison.html</a>
</p>

</body>
</html>"""

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    print(f"Saved: {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
