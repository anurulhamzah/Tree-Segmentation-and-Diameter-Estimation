#!/usr/bin/env python
"""
Kurva Precision/Recall/F1 vs score-threshold (0,05-0,50) untuk model TERBAIK
(FocalNetL-classweight), mask-IoU>=0.5, matching greedy per-kategori (konvensi
F1@0.35 proyek). Menghasilkan figur tesis (PNG+PDF) + tabel JSON.

IoU matrix dihitung SEKALI per gambar (semua pred vs GT), lalu tiap threshold
cuma re-filter+greedy-match dari cache -> cepat.
"""
import os, json
from collections import defaultdict
import numpy as np
from pycocotools import mask as M
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER_ROOT = "/scratch2/pr65/anur0018"
PROJECT = f"{USER_ROOT}/tree_classification"
GT = f"{PROJECT}/data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json"
PRED = f"{USER_ROOT}/maskdino_output/FocalNet_L_combined_rgbd_scratch_v4_stratified_mosaic_classweight_20k/eval_test_v4/inference/coco_instances_results.json"
FIGDIR = f"{PROJECT}/reports/figures"
OUT_JSON = f"{PROJECT}/reports/threshold_sweep.json"

IOU_THR = 0.5
THRS = [round(0.05 * k, 2) for k in range(1, 11)]  # 0.05..0.50

# Okabe-Ito colourblind-safe; identity also carried by marker+linestyle (grayscale-safe)
C_REC, C_PREC, C_F1 = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d8d8"


def to_rle(seg, h, w):
    if isinstance(seg, list):
        return M.merge(M.frPyObjects(seg, h, w))
    if isinstance(seg["counts"], list):
        return M.frPyObjects(seg, h, w)
    return seg


def main():
    gt = json.load(open(GT))
    imgs = {im["id"]: im for im in gt["images"]}
    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]].append(a)
    preds = json.load(open(PRED))
    pr_by_img = defaultdict(list)
    for p in preds:
        pr_by_img[p["image_id"]].append(p)
    pcats = set(p["category_id"] for p in preds)
    gcats = set(c["id"] for c in gt["categories"])
    offset = 1 if (min(pcats) == 0 and max(pcats) == max(gcats) - 1) else 0

    # Precompute per-image: IoU matrix [P x G], pred scores, pred cats, gt cats
    cache = []
    for iid, gl in gt_by_img.items():
        im = imgs[iid]; h, w = im["height"], im["width"]
        pl = pr_by_img.get(iid, [])
        g_rles = [to_rle(a["segmentation"], h, w) for a in gl]
        g_cats = np.array([a["category_id"] for a in gl])
        if pl:
            p_rles = [to_rle(p["segmentation"], h, w) for p in pl]
            iou = np.asarray(M.iou(p_rles, g_rles, [0] * len(g_rles))).reshape(len(pl), len(gl))
            p_sc = np.array([p["score"] for p in pl])
            p_ct = np.array([p["category_id"] + offset for p in pl])
        else:
            iou = np.zeros((0, len(gl))); p_sc = np.zeros(0); p_ct = np.zeros(0, int)
        cache.append((iou, p_sc, p_ct, g_cats))
    n_gt = sum(len(c[3]) for c in cache)

    rows = []
    for thr in THRS:
        TP = FP = FN = 0
        for iou, p_sc, p_ct, g_cats in cache:
            keep = np.where(p_sc >= thr)[0]
            order = keep[np.argsort(-p_sc[keep])]  # high score first
            gt_taken = np.zeros(len(g_cats), bool)
            tp = 0
            for pi in order:
                cand = np.where((~gt_taken) & (g_cats == p_ct[pi]) & (iou[pi] >= IOU_THR))[0]
                if len(cand):
                    best = cand[np.argmax(iou[pi, cand])]
                    gt_taken[best] = True
                    tp += 1
            TP += tp
            FP += len(order) - tp
            FN += len(g_cats) - tp
        prec = TP / max(TP + FP, 1)
        rec = TP / max(TP + FN, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({"thr": thr, "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4), "TP": TP, "FP": FP, "FN": FN})

    f1s = [r["f1"] for r in rows]
    best_i = int(np.argmax(f1s))
    json.dump({"iou_thr": IOU_THR, "n_gt": n_gt, "best_thr": rows[best_i]["thr"],
               "rows": rows}, open(OUT_JSON, "w"), indent=2)

    # sanity print
    print(f"N_gt={n_gt}  (sanity @thr0.35 should ~ F1 0.6385 / R 0.5739 / P 0.7195)")
    print(f"{'thr':>5s} {'prec':>7s} {'rec':>7s} {'f1':>7s}")
    for r in rows:
        star = " <- F1 max" if r["thr"] == rows[best_i]["thr"] else ""
        print(f"{r['thr']:>5.2f} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}{star}")

    # ---- figure ----
    x = [r["thr"] for r in rows]
    rec = [r["recall"] for r in rows]; prec = [r["precision"] for r in rows]; f1 = [r["f1"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(x, rec, color=C_REC, lw=2, marker="o", ms=6, ls="-", label="Recall", zorder=3)
    ax.plot(x, prec, color=C_PREC, lw=2, marker="s", ms=5.5, ls="--", label="Precision", zorder=3)
    ax.plot(x, f1, color=C_F1, lw=2.6, marker="^", ms=6.5, ls="-", label="F1", zorder=4)

    # F1 peak marker (= project's operating point 0.35)
    lo = min(min(rec), min(prec), min(f1)); hi = max(max(rec), max(prec), max(f1))
    ax.set_ylim(lo - 0.05, hi + 0.11)

    # recall ceiling (top-left, empty region): recall saturates even at thr 0.05
    rec_ceiling = max(rec)
    ax.axhline(rec_ceiling, color=C_REC, lw=1.0, ls=":", zorder=1)
    ax.text(0.052, rec_ceiling + 0.013,
            f"recall ceiling $\\approx$ {rec_ceiling:.2f}  ($\\approx$23% true-miss, unreachable by threshold)",
            fontsize=7.8, color=C_REC, ha="left", va="bottom")

    # F1 peak marker (= project's operating point 0.35); label in empty top-centre with arrow
    bx, by = x[best_i], f1[best_i]
    ax.scatter([bx], [by], s=120, facecolor="none", edgecolor=C_F1, lw=1.8, zorder=5)
    ax.annotate(f"F1 max = {by:.3f} @ thr {bx:.2f}\n(current operating point)",
                xy=(bx, by), xytext=(0.235, hi + 0.055), fontsize=8.5, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))

    ax.set_xlabel("Score threshold", fontsize=11, color=INK)
    ax.set_ylabel("Metric", fontsize=11, color=INK)
    ax.set_title("Operating-point trade-off — FocalNet-L (mask IoU $\\geq$ 0.5)",
                 fontsize=11, color=INK)
    ax.set_xticks(x)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.legend(frameon=False, fontsize=10, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGDIR}/threshold_sweep_focalnetl.{ext}", bbox_inches="tight")
    print(f"\nSaved figure -> {FIGDIR}/threshold_sweep_focalnetl.[png|pdf]")
    print(f"Saved table  -> {OUT_JSON}")


if __name__ == "__main__":
    main()
