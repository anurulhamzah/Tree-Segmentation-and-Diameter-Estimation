"""
Training progress comparison — Combined dataset (fase 2)
Segm AP50 vs iteration up to 50k
"""

import re, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

LOG_DIR = '/scratch2/pr65/anur0018/tree_classification/logs'

def extract_segm_ap50(logname, max_iter=50000):
    logf = os.path.join(LOG_DIR, f'{logname}.out')
    if not os.path.exists(logf):
        return [], []
    data_lines = []
    with open(logf) as f:
        for line in f:
            if 'copypaste:' not in line:
                continue
            if 'Task:' in line or 'AP,AP50' in line:
                continue
            m = re.search(r'copypaste: ([\d.,\-]+)', line)
            if m:
                data_lines.append(m.group(1))
    # segm = odd indices (0-based): bbox@n=even, segm@n=odd
    iters, ap50s = [], []
    for idx in range(1, len(data_lines), 2):
        it = ((idx // 2) + 1) * 5000
        if it > max_iter:
            break
        try:
            ap50 = float(data_lines[idx].split(',')[1])
            iters.append(it)
            ap50s.append(ap50)
        except (IndexError, ValueError):
            pass
    return iters, ap50s


# ── Data ─────────────────────────────────────────────────────────────────────
BACKBONE_MODELS = [
    dict(label='SwinT Combined (C1)',  log='SwinT_combined_rle_f1000_75k_55768234',        color='#1976D2', ls='-',  lw=1.8, marker='o',  ms=5),
    dict(label='SwinL Combined (C2)',  log='SwinL_combined_rle_f1000_75k_55786426',        color='#7B1FA2', ls='-',  lw=1.8, marker='s',  ms=5),
    dict(label='SwinB Combined (C3)',  log='SwinB_combined_rle_f1000_75k_55797820',        color='#2E7D32', ls='-',  lw=2.5, marker='D',  ms=5.5),
    dict(label='FocalNet-L (running)', log='FocalNet_L_combined_rle_f1000_75k_55884266',  color='#C62828', ls='-',  lw=2.5, marker='^',  ms=5.5),
]

ABLATION_MODELS = [
    dict(label='SwinB RGB only (C3 baseline)',  log='SwinB_combined_rle_f1000_75k_55797820',        color='#2E7D32', ls='-',  lw=2.5, marker='D',  ms=5.5),
    dict(label='SwinB RGB + DBH (C4)',          log='SwinB_combined_rle_f1000_75k_dbh_55848391',    color='#E65100', ls='--', lw=2.0, marker='o',  ms=5),
    dict(label='SwinB RGBD (no DBH, running)',  log='SwinB_combined_rle_f1000_rgbd_75k_55884237',  color='#0097A7', ls='--', lw=2.0, marker='s',  ms=5),
    dict(label='SwinB RGBD + DBH (C5)',         log='SwinB_combined_rle_f1000_rgbd_dbh_75k_55848452', color='#9C27B0', ls=':', lw=2.0, marker='^', ms=5),
]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11.5, 'axes.labelsize': 10.5,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
fig.suptitle('Training Progress — Combined Dataset (Fase 2)  ·  Segm AP50 @ Val Set  ·  eval tiap 5 000 iter',
             fontsize=12, fontweight='bold', y=1.01)

XTICKS = list(range(0, 55000, 5000))
XTICK_LABELS = [f'{x//1000}k' for x in XTICKS]

# ── Panel 1: Backbone scaling ─────────────────────────────────────────────────
for m in BACKBONE_MODELS:
    iters, ap50s = extract_segm_ap50(m['log'])
    if not iters:
        continue
    is_running = 'running' in m['label']
    lbl = m['label'] + (f'  [last: {ap50s[-1]:.1f}%@{iters[-1]//1000}k]' if is_running
                        else f'  [final @50k: {ap50s[-1]:.1f}%]')
    ax1.plot(iters, ap50s, color=m['color'], ls=m['ls'], lw=m['lw'],
             marker=m['marker'], ms=m['ms'], markevery=1, label=lbl, zorder=3)
    # Annotate last point
    ax1.annotate(f"{ap50s[-1]:.1f}%",
                 xy=(iters[-1], ap50s[-1]),
                 xytext=(iters[-1] + 800, ap50s[-1] + 0.4),
                 fontsize=8, color=m['color'], fontweight='bold')

ax1.axvline(40000, color='#BDBDBD', lw=1.0, ls=':', zorder=1)
ax1.axvline(47000, color='#BDBDBD', lw=1.0, ls=':', zorder=1)
ax1.text(40300, 12, 'LR×0.1\n@60k', fontsize=7.5, color='#9E9E9E')
ax1.text(47300, 12, 'LR×0.01\n@70k', fontsize=7.5, color='#9E9E9E')

ax1.set_xlabel('Iteration')
ax1.set_ylabel('Segm AP50 (%)')
ax1.set_title('Backbone Scaling — Swin T/B/L vs FocalNet-L')
ax1.set_xlim(0, 51500)
ax1.set_ylim(10, 52)
ax1.set_xticks(XTICKS)
ax1.set_xticklabels(XTICK_LABELS)
ax1.legend(loc='upper left', framealpha=0.88)
ax1.grid(axis='y', alpha=0.25)
ax1.grid(axis='x', alpha=0.12)

# ── Panel 2: Depth/DBH ablation ───────────────────────────────────────────────
for m in ABLATION_MODELS:
    color = m.get('color', '#333')
    iters, ap50s = extract_segm_ap50(m['log'])
    if not iters:
        continue
    is_running = 'running' in m['label']
    lbl = m['label'] + (f'  [last: {ap50s[-1]:.1f}%@{iters[-1]//1000}k]' if is_running
                        else f'  [@50k: {ap50s[-1]:.1f}%]')
    ax2.plot(iters, ap50s, color=color, ls=m['ls'], lw=m['lw'],
             marker=m['marker'], ms=m['ms'], markevery=1, label=lbl, zorder=3)
    ax2.annotate(f"{ap50s[-1]:.1f}%",
                 xy=(iters[-1], ap50s[-1]),
                 xytext=(iters[-1] + 800, ap50s[-1] + 0.4),
                 fontsize=8, color=color, fontweight='bold')

# Cold-start annotation
ax2.annotate('Cold-start\n(4-ch init)',
             xy=(5000, 6.01), xytext=(8000, 6.5),
             fontsize=8, color='#9C27B0',
             arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=1.0))
ax2.annotate('Cold-start\n(4-ch init)',
             xy=(5000, 5.31), xytext=(10000, 3.5),
             fontsize=8, color='#0097A7',
             arrowprops=dict(arrowstyle='->', color='#0097A7', lw=1.0))

ax2.set_xlabel('Iteration')
ax2.set_ylabel('Segm AP50 (%)')
ax2.set_title('Depth & DBH Ablation — SwinB Combined')
ax2.set_xlim(0, 51500)
ax2.set_ylim(0, 52)
ax2.set_xticks(XTICKS)
ax2.set_xticklabels(XTICK_LABELS)
ax2.legend(loc='upper left', framealpha=0.88)
ax2.grid(axis='y', alpha=0.25)
ax2.grid(axis='x', alpha=0.12)

# Catatan running models
fig.text(0.5, -0.03,
         '★ FocalNet-L dan SwinB RGBD masih training — data belum lengkap sampai 50k.',
         ha='center', fontsize=9, color='#616161', style='italic')

plt.tight_layout()
out = '/scratch2/pr65/anur0018/tree_classification/reports/deep_eval/training_progress_50k.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'✓ Saved: {out}')
plt.close()
