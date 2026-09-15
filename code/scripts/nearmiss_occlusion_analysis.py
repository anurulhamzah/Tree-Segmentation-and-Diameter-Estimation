#!/usr/bin/env python
"""
Analisis near-miss vs true-miss: apakah recall yang hilang disebabkan OKLUSI
(deteksi ada & terlokalisasi tapi IoU 0,3-0,5 -- resolusi TAK menolong) atau
TRUE-MISS (IoU<0,1, benar-benar tak terdeteksi -- resolusi/retrain mungkin
menolong). Pakai mask-IoU (metrik proyek yg benar) dari PREDIKSI TERSIMPAN model
terbaik (FocalNetL-classweight), tanpa GPU.

Kategori per GT (best mask-IoU dgn prediksi se-kategori, skor>=0.35):
  HIT       : IoU>=0.5  (dihitung di recall@0.5)
  NEAR-MISS : 0.3<=IoU<0.5  (terdeteksi+terlokalisasi, gagal threshold -> oklusi/ambiguitas mask)
  WEAK      : 0.1<=IoU<0.3
  TRUE-MISS : IoU<0.1  (praktis tak terdeteksi)
Dipilah per bin-depth dan per ukuran objek COCO (area GT).
"""
import os, sys, json
from collections import defaultdict
import numpy as np
from pycocotools import mask as M

USER_ROOT = "/scratch2/pr65/anur0018"
PROJECT = f"{USER_ROOT}/tree_classification"
GT = f"{PROJECT}/data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json"
PRED = f"{USER_ROOT}/maskdino_output/FocalNet_L_combined_rgbd_scratch_v4_stratified_mosaic_classweight_20k/eval_test_v4/inference/coco_instances_results.json"
OUT = f"{PROJECT}/reports/nearmiss_occlusion_analysis.json"

SCORE = 0.35
DEPTH_BINS = [(0, 8), (8, 12), (12, 18), (18, 30), (30, 1e9)]
DEPTH_LAB = ["<8m", "8-12m", "12-18m", "18-30m", ">30m"]
# COCO area thresholds (px^2 at 480x270): small<32^2, medium 32^2..96^2, large>96^2
SIZE_LAB = ["small(<1024)", "medium(1024-9216)", "large(>9216)"]


def size_bin(area):
    if area < 1024:
        return 0
    if area < 9216:
        return 1
    return 2


def depth_bin(d):
    for i, (lo, hi) in enumerate(DEPTH_BINS):
        if lo <= d < hi:
            return i
    return len(DEPTH_BINS) - 1


def to_rle(seg, h, w):
    if isinstance(seg, list):
        r = M.frPyObjects(seg, h, w)
        return M.merge(r)
    if isinstance(seg["counts"], list):
        return M.frPyObjects(seg, h, w)
    return seg


def main():
    gt = json.load(open(GT))
    imgs = {im["id"]: im for im in gt["images"]}
    catname = {c["id"]: c["name"] for c in gt["categories"]}
    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]].append(a)

    preds = json.load(open(PRED))
    pr_by_img = defaultdict(list)
    for p in preds:
        if p["score"] >= SCORE:
            pr_by_img[p["image_id"]].append(p)

    # category_id space check: preds may be 0-based or 1-based. GT cats are 1..13.
    pcats = set(p["category_id"] for p in preds)
    gcats = set(catname)
    offset = 1 if (min(pcats) == 0 and max(pcats) == max(gcats) - 1) else 0

    CATS = ["HIT", "NEAR", "WEAK", "TRUEMISS"]

    def cat_of(iou):
        if iou >= 0.5: return 0
        if iou >= 0.3: return 1
        if iou >= 0.1: return 2
        return 3

    overall = [0, 0, 0, 0]
    by_depth = {i: [0, 0, 0, 0] for i in range(len(DEPTH_BINS))}
    by_size = {i: [0, 0, 0, 0] for i in range(len(SIZE_LAB))}
    # for size, also record what fraction of TRUE-MISS are small
    truemiss_sizes = [0, 0, 0]
    nearmiss_sizes = [0, 0, 0]
    n_gt = 0

    for iid, gl in gt_by_img.items():
        im = imgs[iid]; h, w = im["height"], im["width"]
        pl = pr_by_img.get(iid, [])
        # build compressed RLE lists
        g_rles = [to_rle(a["segmentation"], h, w) for a in gl]
        g_cats = [a["category_id"] for a in gl]
        p_rles = [to_rle(p["segmentation"], h, w) for p in pl]
        p_cats = [p["category_id"] + offset for p in pl]
        if len(p_rles):
            iou = M.iou(p_rles, g_rles, [0] * len(g_rles))  # [P x G]
            iou = np.asarray(iou).reshape(len(p_rles), len(g_rles))
        else:
            iou = np.zeros((0, len(g_rles)))
        for gi, a in enumerate(gl):
            same = [pi for pi in range(len(p_rles)) if p_cats[pi] == g_cats[gi]]
            best = max((iou[pi, gi] for pi in same), default=0.0)
            c = cat_of(best)
            overall[c] += 1
            d = a.get("depth_mean")
            db = depth_bin(float(d)) if isinstance(d, (int, float)) and d > 0 else None
            if db is not None:
                by_depth[db][c] += 1
            sb = size_bin(a.get("area", 0))
            by_size[sb][c] += 1
            if c == 3:
                truemiss_sizes[sb] += 1
            if c == 1:
                nearmiss_sizes[sb] += 1
            n_gt += 1

    def pct(row):
        t = sum(row)
        return {CATS[i]: f"{100*row[i]/max(t,1):.1f}%" for i in range(4)} | {"N": t}

    res = {
        "model": "FocalNetL-classweight (combined test 843 img)",
        "score_thr": SCORE, "N_gt": n_gt,
        "recall@0.5": round(overall[0] / max(n_gt, 1), 4),
        "recall@0.3 (HIT+NEAR)": round((overall[0] + overall[1]) / max(n_gt, 1), 4),
        "overall_pct": pct(overall),
        "by_depth": {DEPTH_LAB[i]: pct(by_depth[i]) for i in range(len(DEPTH_BINS))},
        "by_size": {SIZE_LAB[i]: pct(by_size[i]) for i in range(len(SIZE_LAB))},
        "truemiss_size_share": {SIZE_LAB[i]: truemiss_sizes[i] for i in range(3)},
        "nearmiss_size_share": {SIZE_LAB[i]: nearmiss_sizes[i] for i in range(3)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    print(f"Model: {res['model']}  N_gt={n_gt}")
    print(f"recall@0.5={res['recall@0.5']:.4f}   recall@0.3={res['recall@0.3 (HIT+NEAR)']:.4f}  "
          f"(sanity: known Recall@0.35 classweight=0.5739)")
    print(f"\nOVERALL: {res['overall_pct']}")
    print("\n=== per DEPTH bin (% of GT in that bin) ===")
    print(f"{'bin':10s} {'N':>6s} {'HIT':>7s} {'NEAR':>7s} {'WEAK':>7s} {'TRUEMISS':>9s}")
    for i in range(len(DEPTH_BINS)):
        r = by_depth[i]; t = sum(r)
        print(f"{DEPTH_LAB[i]:10s} {t:>6d} {100*r[0]/max(t,1):>6.1f}% {100*r[1]/max(t,1):>6.1f}% "
              f"{100*r[2]/max(t,1):>6.1f}% {100*r[3]/max(t,1):>8.1f}%")
    print("\n=== per SIZE (COCO area) ===")
    print(f"{'size':18s} {'N':>6s} {'HIT':>7s} {'NEAR':>7s} {'WEAK':>7s} {'TRUEMISS':>9s}")
    for i in range(len(SIZE_LAB)):
        r = by_size[i]; t = sum(r)
        print(f"{SIZE_LAB[i]:18s} {t:>6d} {100*r[0]/max(t,1):>6.1f}% {100*r[1]/max(t,1):>6.1f}% "
              f"{100*r[2]/max(t,1):>6.1f}% {100*r[3]/max(t,1):>8.1f}%")
    tm = sum(truemiss_sizes)
    print(f"\nTRUE-MISS breakdown by size: " +
          ", ".join(f"{SIZE_LAB[i]}={truemiss_sizes[i]} ({100*truemiss_sizes[i]/max(tm,1):.0f}%)" for i in range(3)))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
