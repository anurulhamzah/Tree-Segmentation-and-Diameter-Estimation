#!/usr/bin/env python
"""Figure: DBH head progression V1(V2)->V6->Sandbox, mean per-species R2 climbing
from deep negative to positive. Matches style of thesis_figures.py."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = "/scratch2/pr65/anur0018/tree_classification/reports/figures"
os.makedirs(FIGDIR, exist_ok=True)
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#dcdcdc"
BLUE, GREEN, GREY = "#0072B2", "#009E73", "#999999"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


rows = [
    ("V2\n(6 sp)", -0.416, GREY),
    ("V3\n(6 sp)", -0.375, GREY),
    ("V4\n(6 sp)", -0.248, GREY),
    ("V5\n(6 sp)", -0.222, GREY),
    ("V6\n(6 sp)", -0.125, GREY),
    ("Sandbox\n(11 sp, 3-seed)", 0.136, GREEN),
]
x = list(range(len(rows)))
y = [r[1] for r in rows]
colors = [r[2] for r in rows]

fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
ax.axhline(0, color=INK, lw=1.0, zorder=2)
ax.bar(x, y, color=colors, width=0.6, zorder=3)
for xi, v in zip(x, y):
    va = "bottom" if v >= 0 else "top"
    off = 0.012 if v >= 0 else -0.012
    ax.text(xi, v + off, f"{v:+.3f}", ha="center", va=va, fontsize=8.5, color=INK)
ax.plot(x, y, color=BLUE, lw=1.4, ls="--", zorder=2, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_ylabel("Mean per-species $R^2$", fontsize=10, color=INK)
ax.set_title("DBH head progression: mean per-species $R^2$", fontsize=11, color=INK)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.set_ylim(-0.5, 0.25)
clean(ax)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{FIGDIR}/fig_dbh_progression.{ext}", bbox_inches="tight")
print("saved fig_dbh_progression.[png|pdf]")
