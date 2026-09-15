"""
Plot LR schedule comparison — fase 2, 50k iter
Backbone: R50, SwinT, SwinB, SwinL, FocalNet-L
MultiStepLR + linear warmup
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Konfigurasi per model ─────────────────────────────────────────────────────
MODELS = [
    dict(label='ResNet-50',  base_lr=1e-4, warmup=200,  steps=(40000, 47000), color='#78909C', ls='--', lw=1.6),
    dict(label='FocalNet-L', base_lr=1e-4, warmup=1000, steps=(40000, 47000), color='#E65100', ls='-',  lw=2.2),
    dict(label='Swin-T',     base_lr=5e-5, warmup=1000, steps=(40000, 47000), color='#1976D2', ls='-',  lw=2.0),
    dict(label='Swin-B',     base_lr=4e-5, warmup=1000, steps=(40000, 47000), color='#2E7D32', ls='-',  lw=2.0),
    dict(label='Swin-L',     base_lr=3e-5, warmup=1000, steps=(40000, 47000), color='#7B1FA2', ls='-',  lw=2.0),
]

GAMMA = 0.1
STEP1, STEP2, MAX_ITER = 40000, 47000, 50000
ITERS = np.arange(0, MAX_ITER + 1, 50)


def compute_lr(base_lr, warmup, steps, iters):
    lr = np.zeros_like(iters, dtype=float)
    for i, it in enumerate(iters):
        if it < warmup:
            lr[i] = base_lr * max(it / warmup, 1e-6)
        elif it < steps[0]:
            lr[i] = base_lr
        elif it < steps[1]:
            lr[i] = base_lr * GAMMA
        else:
            lr[i] = base_lr * GAMMA * GAMMA
    return lr


# ── Layout ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 9,  'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})

fig = plt.figure(figsize=(16, 9))
fig.suptitle('Learning Rate Schedule Comparison — Fase 2 (50 k Iterations)\n'
             'MultiStepLR  ·  γ = 0.1  ·  step₁ = 40 k  ·  step₂ = 47 k  ·  warmup = linear',
             fontsize=12, fontweight='bold', y=0.98)

# 3-panel layout: log-scale (main), zoom warmup, tabel
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.38],
                      hspace=0.45, wspace=0.30,
                      left=0.07, right=0.97, top=0.88, bottom=0.04)

ax_log   = fig.add_subplot(gs[0, :2])   # main log-scale (2/3 width)
ax_zoom  = fig.add_subplot(gs[0, 2])    # zoom warmup (1/3 width)
ax_table = fig.add_subplot(gs[1, :])    # tabel di bawah
ax_table.axis('off')

# ── Phase shading ─────────────────────────────────────────────────────────────
phase_colors = ['#E3F2FD', '#FFF3E0', '#F3E5F5']
phase_labels = ['Phase 1\n(high LR)', 'Phase 2\n(×0.1)', 'Phase 3\n(×0.01)']
phase_ranges = [(0, STEP1), (STEP1, STEP2), (STEP2, MAX_ITER)]

for ax in [ax_log]:
    for (x0, x1), pc, pl in zip(phase_ranges, phase_colors, phase_labels):
        ax.axvspan(x0/1000, x1/1000, facecolor=pc, alpha=0.55, zorder=0)
        mid = (x0 + x1) / 2 / 1000
        ax.text(mid, 1.6e-4, pl, ha='center', va='bottom', fontsize=8,
                color='#757575', fontweight='bold')

# ── Plot log-scale ────────────────────────────────────────────────────────────
for m in MODELS:
    lr = compute_lr(m['base_lr'], m['warmup'], m['steps'], ITERS)
    ax_log.semilogy(ITERS / 1000, lr, color=m['color'], lw=m['lw'],
                    ls=m['ls'], label=m['label'])

# Vertical decay lines
for xv, lbl in [(STEP1/1000, 'step₁'), (STEP2/1000, 'step₂')]:
    ax_log.axvline(xv, color='#9E9E9E', lw=1.1, ls=':', zorder=1)
    ax_log.text(xv + 0.3, 1.8e-8, lbl, fontsize=8, color='#616161', va='bottom')

ax_log.set_xlabel('Iteration (×1 000)')
ax_log.set_ylabel('Learning Rate (log scale)')
ax_log.set_title('Effective LR — Log Scale')
ax_log.set_xlim(0, 50)
ax_log.set_ylim(5e-9, 3e-4)
ax_log.legend(loc='lower left', framealpha=0.85, ncol=1)
ax_log.grid(axis='y', alpha=0.2, which='both')

# LR ratio annotations (right side)
for m in MODELS:
    final_lr = m['base_lr'] * GAMMA * GAMMA
    ax_log.annotate(f"{m['base_lr']*1e5:.1f}e-5→{final_lr*1e7:.1f}e-7",
                    xy=(50, m['base_lr']), xytext=(50.2, m['base_lr']),
                    fontsize=7, color=m['color'], va='center',
                    annotation_clip=False)

# ── Zoom: warmup phase (0–2k) ─────────────────────────────────────────────────
ax_zoom.axvspan(0, 1.0, facecolor='#E3F2FD', alpha=0.5, zorder=0)

for m in MODELS:
    zoom_iters = np.arange(0, 2001, 10)
    lr = compute_lr(m['base_lr'], m['warmup'], m['steps'], zoom_iters)
    ax_zoom.plot(zoom_iters / 1000, lr * 1e4, color=m['color'], lw=m['lw'],
                 ls=m['ls'], label=m['label'])

# Warmup endpoint annotations
for m in MODELS:
    wu = m['warmup']
    wu_lr = m['base_lr']
    ax_zoom.annotate(f"{wu_lr*1e4:.2f}",
                     xy=(wu/1000, wu_lr*1e4),
                     xytext=(wu/1000 + 0.1, wu_lr*1e4),
                     fontsize=7.5, color=m['color'], va='center')

ax_zoom.axvline(1.0, color='#9E9E9E', lw=1.0, ls=':', zorder=1)
ax_zoom.text(1.05, 0.02, 'warmup\nend\n@1k', fontsize=7.5, color='#616161',
             va='bottom', transform=ax_zoom.get_xaxis_transform())
ax_zoom.set_xlabel('Iteration (×1 000)')
ax_zoom.set_ylabel('Learning Rate (×10⁻⁴)')
ax_zoom.set_title('Warmup Phase Zoom (0–2 k)')
ax_zoom.set_xlim(0, 2)
ax_zoom.grid(axis='y', alpha=0.25)

# ── Tabel LR summary ─────────────────────────────────────────────────────────
col_labels = ['Backbone', 'Base LR', 'Warmup', 'Phase 1\n(0–40k)', 'Phase 2\n(40–47k)', 'Phase 3\n(47–50k)', 'Ratio P1/P3']
col_widths  = [0.13, 0.10, 0.08, 0.14, 0.14, 0.14, 0.12]
x_positions = [0.01]
for w in col_widths[:-1]:
    x_positions.append(x_positions[-1] + w)

y_header = 0.92
ax_table.text(0.5, 1.05, 'LR Schedule Summary', ha='center', va='bottom',
              fontsize=10, fontweight='bold', transform=ax_table.transAxes)

for xi, col in zip(x_positions, col_labels):
    ax_table.text(xi, y_header, col, ha='left', va='top',
                  fontsize=8.5, fontweight='bold', color='#424242',
                  transform=ax_table.transAxes)

ax_table.plot([0, 1], [y_header - 0.08, y_header - 0.08],
              color='#BDBDBD', lw=0.8, transform=ax_table.transAxes)

for row_i, m in enumerate(MODELS):
    y = y_header - 0.18 - row_i * 0.18
    p1 = m['base_lr']
    p2 = p1 * GAMMA
    p3 = p1 * GAMMA * GAMMA
    ratio = p1 / p3

    row_vals = [
        m['label'],
        f"{p1:.1e}",
        f"{m['warmup']:,} iter",
        f"{p1:.2e}",
        f"{p2:.2e}",
        f"{p3:.2e}",
        f"×{ratio:.0f}",
    ]
    for xi, val in zip(x_positions, row_vals):
        ax_table.text(xi, y, val, ha='left', va='top',
                      fontsize=8.5, color=m['color'] if row_vals.index(val) == 0 else '#333333',
                      fontweight='bold' if row_vals.index(val) == 0 else 'normal',
                      transform=ax_table.transAxes)

ax_table.plot([0, 1], [y_header - 0.12 - len(MODELS) * 0.18] * 2,
              color='#BDBDBD', lw=0.5, transform=ax_table.transAxes)

# ── Simpan ────────────────────────────────────────────────────────────────────
out_path = '/scratch2/pr65/anur0018/tree_classification/reports/deep_eval/lr_comparison_50k.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'✓ Saved: {out_path}')
plt.close()
