#!/usr/bin/env python
"""Figur pendukung tesis (dataviz style: Okabe-Ito, clean academic, grayscale-safe).
Menghasilkan PDF+PNG di reports/figures/:
  fig_seg_ladder      : AP50 test 9 model (arsitektur -> backbone -> modality -> finetune)
  fig_resolution_test : recall RF vs rute inference-only (semua GAGAL)
  fig_size_occlusion  : komposisi HIT/NEAR/WEAK/TRUE-MISS per ukuran objek
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = "/scratch2/pr65/anur0018/tree_classification"
FIGDIR = f"{PROJECT}/reports/figures"
os.makedirs(FIGDIR, exist_ok=True)

INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#dcdcdc"
BLUE, ORANGE, GREEN, SKY, AMBER, GREY = "#0072B2", "#D55E00", "#009E73", "#56B4E9", "#E69F00", "#999999"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGDIR}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.[pdf|png]")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


# ---------- Figure 1: segmentation ladder ----------
def seg_ladder():
    # (label, AP50, family)  family: 'mrcnn' | 'maskdino' | 'best'
    rows = [
        ("Mask R-CNN R50 (RGBD scratch)", 6.86, "mrcnn"),
        ("Mask R-CNN R50 (RGB pretrained)", 15.89, "mrcnn"),
        ("MaskDINO R50 (RGB scratch)", 32.00, "maskdino"),
        ("MaskDINO Swin-B (RGBD pretrained)", 40.16, "maskdino"),
        ("MaskDINO Swin-B (RGB pretrained)", 44.23, "maskdino"),
        ("MaskDINO Swin-B (RGBD scratch)", 48.63, "maskdino"),
        ("MaskDINO FocalNet-L (RGBD scratch)", 57.18, "maskdino"),
        ("  + mosaic fine-tune", 60.02, "maskdino"),
        ("  + class reweighting (final)", 61.17, "best"),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = {"mrcnn": GREY, "maskdino": BLUE, "best": GREEN}
    colors = [cols[r[2]] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=200)
    ax.barh(y, vals, color=colors, height=0.68, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.7, yi, f"{v:.1f}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlabel("Segmentation AP$_{50}$ on held-out test set (%)", fontsize=10, color=INK)
    ax.set_xlim(0, 68)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    clean(ax)
    # legend by family
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREY, label="Mask R-CNN"),
                       Patch(color=BLUE, label="MaskDINO"),
                       Patch(color=GREEN, label="Best model")],
              frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); save(fig, "fig_seg_ladder")


# ---------- Figure 2: resolution test ----------
def resolution_test():
    j = json.load(open(f"{PROJECT}/reports/res_test_rf_tahap1.json"))
    jt = json.load(open(f"{PROJECT}/reports/res_test_rf_tiling.json"))
    base = j["arm480_p640"]["overall_recall"]
    bars = [
        ("whole-image 480\n(baseline, proc 640)", base, GREEN),
        ("native 960\nproc 640", j["arm960_p640"]["overall_recall"], BLUE),
        ("native 960\nproc 960", j["arm960_p960"]["overall_recall"], GREY),
        ("native 960\nproc 1280", j["arm960_p1280"]["overall_recall"], GREY),
        ("tiling / SAHI\n(480 tiles)", jt["tiling960"]["overall_recall"], ORANGE),
    ]
    x = np.arange(len(bars))
    vals = [b[1] for b in bars]
    fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
    ax.bar(x, vals, color=[b[2] for b in bars], width=0.62, zorder=3)
    ax.axhline(base, color=GREEN, lw=1.0, ls=":", zorder=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("Recall (RF test, IoU $\\geq$ 0.5)", fontsize=10, color=INK)
    ax.set_ylim(0, 0.60)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.text(0.985, base - 0.004, "baseline", color=GREEN, fontsize=8, ha="right", va="top",
            transform=ax.get_yaxis_transform())
    clean(ax)
    fig.tight_layout(); save(fig, "fig_resolution_test")


# ---------- Figure 3: size / occlusion composition ----------
def size_occlusion():
    j = json.load(open(f"{PROJECT}/reports/nearmiss_occlusion_analysis.json"))
    order = ["large(>9216)", "medium(1024-9216)", "small(<1024)"]
    disp = {"large(>9216)": "large\n(n=928)", "medium(1024-9216)": "medium\n(n=6860)",
            "small(<1024)": "small\n(n=137)"}
    cats = ["HIT", "NEAR", "WEAK", "TRUEMISS"]
    catcol = {"HIT": GREEN, "NEAR": SKY, "WEAK": AMBER, "TRUEMISS": GREY}
    catlab = {"HIT": "HIT (IoU$\\geq$0.5)", "NEAR": "near-miss (0.3-0.5)",
              "WEAK": "weak (0.1-0.3)", "TRUEMISS": "true-miss (<0.1)"}
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=200)
    y = np.arange(len(order))
    for i, sz in enumerate(order):
        left = 0.0
        vals = {c: float(j["by_size"][sz][c].rstrip("%")) for c in cats}
        for c in cats:
            w = vals[c]
            ax.barh(y[i], w, left=left, color=catcol[c], height=0.62, zorder=3,
                    edgecolor="white", linewidth=1.4)
            if w >= 6:
                ax.text(left + w / 2, y[i], f"{w:.0f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            left += w
    ax.set_yticks(y); ax.set_yticklabels([disp[s] for s in order], fontsize=9, color=INK)
    ax.set_xlabel("Share of ground-truth instances (%)", fontsize=10, color=INK)
    ax.set_xlim(0, 100)
    clean(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=catcol[c], label=catlab[c]) for c in cats],
              frameon=False, fontsize=8, loc="upper center", ncol=4,
              bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout(); save(fig, "fig_size_occlusion")


if __name__ == "__main__":
    seg_ladder()
    resolution_test()
    size_occlusion()
    print("done")
