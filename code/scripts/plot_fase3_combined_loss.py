#!/usr/bin/env python3
"""Loss curves Fase 3 Combined — tiap chart disimpan sebagai file terpisah (lebih besar)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_ROOT = "/fs04/scratch2/pr65/anur0018/maskdino_output"
OUT_DIR  = "/fs04/scratch2/pr65/anur0018/tree_classification/reports/deep_eval/fase3_combined"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = [
    ("SwinT Combined 75k",          "SwinT_combined_rle_f1000_75k",      "#1976D2"),
    ("SwinB Combined 75k",          "SwinB_combined_rle_f1000_75k",      "#E67E22"),
    ("SwinL Combined 75k",          "SwinL_combined_rle_f1000_75k",      "#00897B"),
    ("FocalNet-L Combined 75k",     "FocalNet_L_combined_rle_f1000_75k", "#C62828"),
    ("R50 Combined 50k (baseline)", "R50_combined_rle_f1000_50k",        "#999999"),
]

def load(dirn):
    rows = [json.loads(l) for l in open(f"{OUT_ROOT}/{dirn}/metrics.json") if l.strip()]
    loss = pd.DataFrame([r for r in rows if "total_loss" in r]).sort_values("iteration")
    ap   = pd.DataFrame([r for r in rows if "segm/AP50" in r]).sort_values("iteration")
    return loss, ap

data = {label: (*load(d), c) for label, d, c in MODELS}
def smooth(s, w=30): return s.rolling(w, min_periods=1, center=True).mean()

panels = [
    ("total_loss", "Total Loss",          "01_total_loss"),
    ("loss_ce",    "Classification (CE)",  "02_loss_ce"),
    ("loss_mask",  "Mask BCE",             "03_loss_mask"),
    ("loss_dice",  "Dice",                 "04_loss_dice"),
    ("loss_giou",  "GIoU",                 "05_loss_giou"),
    ("loss_bbox",  "BBox L1",              "06_loss_bbox"),
]

WARMUP = 3000   # abaikan spike awal saat menentukan rentang sumbu-y

# Jadwal LR decay sesuai field 'lr' di metrics.json tiap model:
#   model 75k (SwinT/B/L, FocalNet-L) drop @60k & 70k
#   R50 baseline 50k drop @40k & 47k
LR_DROPS = [
    (60000, "red",     "LR↓ @60k/70k (model 75k)"),
    (70000, "red",     None),
    (40000, "#777777", "LR↓ @40k/47k (R50 50k)"),
    (47000, "#777777", None),
]

def draw_lr_lines(ax):
    for xv, col, lbl in LR_DROPS:
        ax.axvline(xv, color=col, ls=":", alpha=0.45, lw=1.2)
        if lbl:
            # y dalam koordinat fraksi axes supaya konsisten lintas chart
            ax.text(xv, 0.985, lbl, transform=ax.get_xaxis_transform(),
                    fontsize=8.5, color=col, ha="center", va="top")

saved = []
for col, title, fname in panels:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted = False
    lo, hi = np.inf, -np.inf
    for label, (loss, ap, color) in data.items():
        if col not in loss.columns: continue
        sm = smooth(loss[col])
        ax.plot(loss["iteration"], loss[col], color=color, alpha=0.12, lw=0.8)
        ax.plot(loss["iteration"], sm, color=color, lw=2.4, label=label)
        plotted = True
        # rentang sumbu-y dari data setelah warmup (buang spike iter awal)
        post = sm[loss["iteration"] >= WARMUP]
        if len(post):
            lo = min(lo, float(post.min())); hi = max(hi, float(post.max()))
    if not plotted:
        plt.close(fig); continue
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        pad = (hi - lo) * 0.08
        ax.set_ylim(lo - pad, hi + pad)
    draw_lr_lines(ax)
    ax.set_title(f"Fase 3 Combined — {title}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Iterasi", fontsize=12); ax.set_ylabel("Loss", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=11, loc="upper right")
    p = f"{OUT_DIR}/{fname}.png"
    plt.tight_layout(); plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    saved.append(p)

# chart progres AP50 (terpisah)
fig, ax = plt.subplots(figsize=(11, 6.5))
for label, (loss, ap, color) in data.items():
    if "segm/AP50" in ap.columns and len(ap):
        ax.plot(ap["iteration"], ap["segm/AP50"], "o-", color=color, lw=2.4, ms=7, label=label)
        bi = ap["segm/AP50"].idxmax()
        ax.scatter(ap.loc[bi,"iteration"], ap.loc[bi,"segm/AP50"], s=180, color=color,
                   marker="*", zorder=6, edgecolor="white")
        ax.annotate(f"{ap.loc[bi,'segm/AP50']:.1f}%", (ap.loc[bi,"iteration"], ap.loc[bi,"segm/AP50"]),
                    textcoords="offset points", xytext=(6,6), fontsize=9, color=color, fontweight="bold")
draw_lr_lines(ax)
ax.set_title("Fase 3 Combined — Progres segm AP50 (★ = best checkpoint)", fontsize=15, fontweight="bold")
ax.set_xlabel("Iterasi", fontsize=12); ax.set_ylabel("segm AP50 (%)", fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=11, loc="lower right")
p = f"{OUT_DIR}/07_ap50_progress.png"
plt.tight_layout(); plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
saved.append(p)

print(f"SAVED {len(saved)} chart ke {OUT_DIR}/")
for s in saved: print("  -", os.path.basename(s))
