#!/usr/bin/env python3
"""
Refined geometric DBH eval menggunakan trunk isolation dari _find_row_1p3m().

Perbedaan dari geometric sebelumnya (R²=-5.28):
  LAMA: min lebar mask di window bottom rows (termasuk cabang/crown)
  BARU: cari row di h=1.3m secara geometris, lalu filter kolom dengan depth
        proximity (hanya trunk depan, bukan cabang belakang)
        → DBH_mm = trunk_px * d_trunk * 1000 / f_x

Evaluasi menggunakan GT masks untuk menguji prinsip geometris murni:
  - R² masih negatif → geometry tidak bisa solve ini (perlu neural)
  - R² positif       → masalah ada di mask quality, geometry sendiri valid

Usage:
    cd /scratch2/pr65/anur0018/tree_classification
    nohup /scratch2/pr65/anur0018/conda_envs/maskdino/bin/python \\
        scripts/eval_geometric_refined.py \\
        > logs/eval_geometric_refined.out 2>&1 &
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Untuk decode RLE mask
from pycocotools import mask as coco_mask

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from trunk_roi_dbh_head import _find_row_1p3m, _denorm_depth, _LOG_DMAX

# Semua gambar test set = 480×270
CY = 135.0
FY = 240.0
FX = 240.0   # f_x = f_y (kamera square pixel)

ANN_FILE = (
    HERE.parent
    / "data/combined/annotation_inst"
    / "filtered_rle_f1000_repaired_v4_stratified"
    / "instances_test.json"
)

OUTPUT_FILE = HERE.parent / "reports/dbh_eval/geometric_refined_gt_masks.json"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").strip()
        assert header in ("PF", "Pf"), f"Not a PFM file: {path}"
        w, h = (int(v) for v in f.readline().decode("latin-1").split())
        scale = float(f.readline().decode("latin-1").strip())
        endian = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(f.read(), dtype=endian).reshape(h, w)
    return np.flipud(data).copy()


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth < 65000
    out = np.where(valid, np.log1p(depth), 0.0) / _LOG_DMAX
    return out.astype(np.float32)


def depth_path(img_path: str) -> str:
    return img_path.replace("rgb_resized", "depth_pfm").replace(".png", ".pfm")


# ─── Main eval ────────────────────────────────────────────────────────────────

def main():
    print(f"Loading annotations from {ANN_FILE}")
    ann_data = json.load(open(ANN_FILE))

    imgid_to_info = {img["id"]: img for img in ann_data["images"]}

    # Group annotations by image; skip instances with dbh=0 (vine/no-measurement)
    imgid_to_anns: dict = {}
    for ann in ann_data["annotations"]:
        if ann.get("dbh", 0) <= 0:
            continue
        imgid_to_anns.setdefault(ann["image_id"], []).append(ann)

    print(f"Images with DBH annotations: {len(imgid_to_anns)}")
    total_anns = sum(len(v) for v in imgid_to_anns.values())
    print(f"Total instances (dbh>0): {total_anns}")

    preds_mm, gts_mm, cat_ids = [], [], []
    n_skip_no_row = 0    # _find_row_1p3m returned None
    n_skip_depth  = 0    # depth file missing
    n_processed   = 0

    for img_id, anns in imgid_to_anns.items():
        img_info = imgid_to_info[img_id]
        img_path = img_info["file_name"]
        dep_path = depth_path(img_path)

        try:
            depth_raw = _denorm_depth(
                torch.from_numpy(_normalize_depth(_read_pfm(dep_path)))
            )
        except FileNotFoundError:
            n_skip_depth += len(anns)
            continue

        for ann in anns:
            gt_mm = ann["dbh"] * 10.0   # cm → mm

            # Decode RLE mask → bool tensor (H, W)
            seg = ann["segmentation"]
            if isinstance(seg, dict):               # already RLE
                rle = seg
            else:                                   # polygon list
                rle = coco_mask.frPyObjects(seg, img_info["height"], img_info["width"])
                rle = coco_mask.merge(rle)
            mask = torch.from_numpy(coco_mask.decode(rle).astype(bool))

            row, trunk_cols, d_trunk = _find_row_1p3m(mask, depth_raw, CY, FY)
            if row is None:
                n_skip_no_row += 1
                continue

            trunk_px = len(trunk_cols)
            dbh_mm   = trunk_px * d_trunk * 1000.0 / FX

            preds_mm.append(float(dbh_mm))
            gts_mm.append(float(gt_mm))
            cat_ids.append(ann["category_id"])
            n_processed += 1

        if n_processed % 500 == 0 and n_processed > 0:
            print(f"  processed {n_processed} instances so far...")

    # ─── Metrics ──────────────────────────────────────────────────────────────
    p = np.array(preds_mm)
    g = np.array(gts_mm)
    e = np.abs(p - g)
    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Per-category metrics
    CATNAMES = {
        1: "Apple", 2: "Lemon", 3: "Loquat", 4: "Mango",
        5: "Orange", 6: "Persimmon", 7: "Pomegranate",
        8: "AliiFig", 9: "BangaloPalm", 10: "Fern",
        11: "LeechVine", 12: "RubberFig", 13: "Umbrella",
    }
    per_cat = {}
    for cid, cname in CATNAMES.items():
        idx = [i for i, c in enumerate(cat_ids) if c == cid]
        if not idx:
            continue
        pi, gi = p[idx], g[idx]
        ei = np.abs(pi - gi)
        ss_r = np.sum((gi - pi) ** 2)
        ss_t = np.sum((gi - gi.mean()) ** 2)
        per_cat[cname] = {
            "N":    len(idx),
            "MAE":  float(ei.mean()),
            "RMSE": float(np.sqrt((ei**2).mean())),
            "bias": float((pi - gi).mean()),
            "R2":   float(1 - ss_r / ss_t) if ss_t > 0 else float("nan"),
        }

    results = {
        "method":        "geometric_refined_gt_masks",
        "description":   "_find_row_1p3m + depth_proximity_filter, GT masks, f_x=240",
        "N":             int(len(p)),
        "n_skip_no_row": int(n_skip_no_row),
        "n_skip_depth":  int(n_skip_depth),
        "total_instances": int(total_anns),
        "coverage_pct":  float(len(p) / total_anns * 100),
        "MAE":           float(e.mean()),
        "RMSE":          float(np.sqrt((e**2).mean())),
        "bias":          float((p - g).mean()),
        "R2":            float(r2),
        "within_50mm":   float((e < 50).mean() * 100),
        "within_100mm":  float((e < 100).mean() * 100),
        "median_pred":   float(np.median(p)),
        "median_gt":     float(np.median(g)),
        "per_category":  per_cat,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUTPUT_FILE, "w"), indent=2)

    print("\n" + "=" * 55)
    print(f"  Method  : Refined Geometric (GT masks)")
    print(f"  N valid : {results['N']} / {total_anns} ({results['coverage_pct']:.1f}%)")
    print(f"  Skipped : {n_skip_no_row} (no 1.3m row) + {n_skip_depth} (depth file missing)")
    print(f"  MAE     : {results['MAE']:.1f} mm")
    print(f"  RMSE    : {results['RMSE']:.1f} mm")
    print(f"  Bias    : {results['bias']:+.1f} mm")
    print(f"  R²      : {results['R2']:.4f}")
    print(f"  ±50mm   : {results['within_50mm']:.1f}%")
    print(f"  ±100mm  : {results['within_100mm']:.1f}%")
    print(f"  Median pred/GT: {results['median_pred']:.0f} / {results['median_gt']:.0f} mm")
    print("=" * 55)
    print("\nPer category:")
    for cname, m in per_cat.items():
        print(f"  {cname:12s}  N={m['N']:3d}  MAE={m['MAE']:6.1f}  R²={m['R2']:+.3f}  bias={m['bias']:+.1f}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
