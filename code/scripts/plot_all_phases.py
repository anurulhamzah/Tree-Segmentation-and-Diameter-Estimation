"""
4 gambar komparasi training per fase sesuai pipeline MaskDINO:
  Fase 1 — Preliminary Sweep (5k): bar chart filter ablation + backbone sweep
  Fase 2 — Training Terpisah (50k): curves RF dan PL
  Fase 3 — Combined PL+RF (75k): backbone scaling curves
  Fase 4 — DBH Regression Extension: curves SwinT RF dan SwinB Combined
"""

import re, os, json
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR   = '/scratch2/pr65/anur0018/tree_classification/logs'
SWEEP_DIR = '/scratch2/pr65/anur0018/maskdino_output/sweep_hyperparams'
OUT_DIR   = '/scratch2/pr65/anur0018/tree_classification/reports/deep_eval'
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11.5, 'axes.labelsize': 10.5,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def sweep_ap50(subdir):
    mf = f'{SWEEP_DIR}/{subdir}/metrics.json'
    if not os.path.exists(mf): return None
    best = None
    with open(mf) as f:
        for line in f:
            try:
                d = json.loads(line)
                if 'segm/AP50' in d: best = d['segm/AP50']
            except: pass
    return best

def extract_curves(logname, max_iter=None):
    logf = f'{LOG_DIR}/{logname}.out'
    if not os.path.exists(logf): return [], []
    data_lines = []
    with open(logf) as f:
        for line in f:
            if 'copypaste:' not in line: continue
            if 'Task:' in line or 'AP,AP50' in line: continue
            m = re.search(r'copypaste: ([\d.,\-]+)', line)
            if m: data_lines.append(m.group(1))
    iters, ap50s = [], []
    for idx in range(1, len(data_lines), 2):
        it = ((idx // 2) + 1) * 5000
        if max_iter and it > max_iter: break
        try:
            ap50 = float(data_lines[idx].split(',')[1])
            iters.append(it); ap50s.append(ap50)
        except: pass
    return iters, ap50s

def extract_pl_multi(logname, n_models=2):
    logf = f'{LOG_DIR}/{logname}.out'
    data_lines = []
    with open(logf) as f:
        for line in f:
            if 'copypaste:' not in line: continue
            if 'Task:' in line or 'AP,AP50' in line: continue
            m = re.search(r'copypaste: ([\d.,\-]+)', line)
            if m: data_lines.append(m.group(1))
    segm_vals = []
    for idx in range(1, len(data_lines), 2):
        try: segm_vals.append(float(data_lines[idx].split(',')[1]))
        except: pass
    result = []
    for mi in range(n_models):
        chunk = segm_vals[mi*10:(mi+1)*10]
        result.append(([((i+1)*5000) for i in range(len(chunk))], chunk))
    return result

def plot_curve(ax, iters, ap50s, label, color, ls='-', lw=2.0, marker='o',
               ms=5, ann_xoff=600, ann_yoff=0.5, running=False):
    if not iters: return
    lbl = label + (f'  [{ap50s[-1]:.1f}%@{iters[-1]//1000}k★]' if running
                   else f'  [{ap50s[-1]:.1f}%@{iters[-1]//1000}k]')
    ax.plot(iters, ap50s, color=color, ls=ls, lw=lw,
            marker=marker, ms=ms, label=lbl, zorder=3)
    ax.annotate(f'{ap50s[-1]:.1f}%',
                xy=(iters[-1], ap50s[-1]),
                xytext=(iters[-1]+ann_xoff, ap50s[-1]+ann_yoff),
                fontsize=7.5, color=color, fontweight='bold', annotation_clip=False)

def add_lr_decay_lines(ax, steps, y_text_pct=0.07):
    ymin, ymax = ax.get_ylim()
    for xv, lbl in steps:
        ax.axvline(xv, color='#BDBDBD', lw=1.0, ls=':', zorder=1)
        ax.text(xv+300, ymin + (ymax-ymin)*y_text_pct, lbl,
                fontsize=7.5, color='#9E9E9E', va='bottom')

XTICK50 = list(range(0, 55000, 5000))
XTICK75 = list(range(0, 80000, 5000))
XLB50 = [f'{x//1000}k' for x in XTICK50]
XLB75 = [f'{x//1000}k' for x in XTICK75]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Preliminary Sweep (5k iter)
# ══════════════════════════════════════════════════════════════════════════════
fig1, (ax_filt, ax_bb) = plt.subplots(1, 2, figsize=(15, 6))
fig1.suptitle('Fase 1 — Preliminary Sweep  (5 000 iterations each)\n'
              'Kiri: Filter Threshold Ablation (R50)  ·  Kanan: Backbone Sweep (f = 1 000)',
              fontsize=12, fontweight='bold', y=1.01)

# Panel kiri: filter ablation
filters   = ['f0','f100','f200','f300','f500','f1000']
xlabels_f = ['f=0\n(no filter)','f=100','f=200','f=300','f=500','f=1000\n(RLE)']
rf_filt   = [sweep_ap50(f'p1_rf_{f}_R50') for f in filters]
pl_filt   = [sweep_ap50(f'p1_pl_{f}_R50') for f in filters]
x = np.arange(len(filters)); w = 0.35

b1 = ax_filt.bar(x - w/2, [v or 0 for v in rf_filt], w, color='#1565C0', alpha=0.85, label='RF (Rainforests)')
b2 = ax_filt.bar(x + w/2, [v or 0 for v in pl_filt], w, color='#E65100', alpha=0.85, label='PL (Plantations)')
for bar, val in list(zip(b1, rf_filt)) + list(zip(b2, pl_filt)):
    if val and val > 0.3:
        ax_filt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax_filt.axvline(4.5, color='#9E9E9E', lw=1.2, ls='--')
ax_filt.text(4.6, 28, '← Pilih\nf=1000', fontsize=8.5, color='#616161')
ax_filt.set_xticks(x); ax_filt.set_xticklabels(xlabels_f)
ax_filt.set_ylabel('Segm AP50 (%) @ 5k iter')
ax_filt.set_title('Filter Threshold Ablation — R50 Baseline')
ax_filt.set_ylim(0, 36); ax_filt.legend(); ax_filt.grid(axis='y', alpha=0.25)

# Panel kanan: backbone sweep
backbones  = ['R50', 'SwinT', 'SwinB', 'SwinL', 'FocalNet-L']
bb_keys    = ['R50', 'SwinT', 'SwinB', 'SwinL', 'FocalNet_L']
bb_colors  = ['#78909C', '#1976D2', '#2E7D32', '#7B1FA2', '#C62828']
rf_bb  = [sweep_ap50(f'p2_rf_f1000_{k}') for k in bb_keys]
pl_bb  = [sweep_ap50(f'p2_pl_f1000_{k}') for k in bb_keys]
x2 = np.arange(len(backbones))

b3 = ax_bb.bar(x2 - w/2, [v or 0 for v in rf_bb], w,
               color=bb_colors, alpha=0.55, label='RF', edgecolor=bb_colors, linewidth=1.2)
b4 = ax_bb.bar(x2 + w/2, [v or 0 for v in pl_bb], w,
               color=bb_colors, alpha=0.92, label='PL')
for bar, val in list(zip(b3, rf_bb)) + list(zip(b4, pl_bb)):
    if val and val > 0.5:
        ax_bb.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax_bb.set_xticks(x2); ax_bb.set_xticklabels(backbones)
ax_bb.set_ylabel('Segm AP50 (%) @ 5k iter')
ax_bb.set_title('Backbone Sweep — f=1000, 5k iter')
ax_bb.set_ylim(0, 46)
ax_bb.legend(handles=[
    plt.Rectangle((0,0),1,1, fc='#444', alpha=0.5, label='RF (light = lower AP)'),
    plt.Rectangle((0,0),1,1, fc='#444', alpha=0.9, label='PL (dark = higher AP)'),
], fontsize=9)
ax_bb.grid(axis='y', alpha=0.25)

fig1.text(0.5, -0.03,
          'Pilihan: f=1000 (RLE) untuk semua experiment selanjutnya.  '
          'Backbone SwinT / SwinB / SwinL dilanjutkan ke Fase 2.',
          ha='center', fontsize=9, style='italic', color='#424242')
plt.tight_layout()
fig1.savefig(f'{OUT_DIR}/fase1_sweep.png', dpi=150, bbox_inches='tight', facecolor='white')
print('✓ Fase 1 saved')
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Training Terpisah (50k iter): RF dan PL
# ══════════════════════════════════════════════════════════════════════════════
fig2, (ax_rf, ax_pl) = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Fase 2 — Training Terpisah  (50 000 iterations)  ·  Segm AP50 @ Val Set  ·  eval tiap 5k',
              fontsize=12, fontweight='bold', y=1.01)

# RF panel
rf_models = [
    ('R50 RF (baseline)', 'r50_rf_rle_f1000_50k_55561077', '#78909C', '--', 1.6, 's'),
    ('SwinT RF (E1)',     'swint_rf_rle_f1000_55561078',   '#1976D2', '-',  2.2, 'o'),
    ('SwinB RF (E2)',     'swinb_rf_f1000_55471809',        '#2E7D32', '-',  2.2, 'D'),
    ('SwinL RF (E3)',     'SwinL_rf_rle_f1000_55534116',    '#7B1FA2', '-',  2.2, '^'),
]
for label, log, color, ls, lw, mk in rf_models:
    iters, ap50s = extract_curves(log, 50000)
    plot_curve(ax_rf, iters, ap50s, label, color, ls, lw, mk)

add_lr_decay_lines(ax_rf, [(40000, 'LR×0.1\n@40k'), (47000, 'LR×0.01\n@47k')])
ax_rf.set_xlabel('Iteration'); ax_rf.set_ylabel('Segm AP50 (%)')
ax_rf.set_title('Rainforests (RF) — 6 kelas, 2 282 gambar')
ax_rf.set_xlim(0, 52000); ax_rf.set_ylim(5, 45)
ax_rf.set_xticks(XTICK50); ax_rf.set_xticklabels(XLB50)
ax_rf.legend(loc='upper left', framealpha=0.88); ax_rf.grid(axis='y', alpha=0.25)
ax_rf.grid(axis='x', alpha=0.12)

# PL panel
pl_curves = extract_pl_multi('pl_rle_f1000_50k_55751990')
pl_models = [
    ('SwinT PL (P1)', '#1976D2', '-',  2.2, 'o'),
    ('SwinB PL (P2)', '#2E7D32', '-',  2.2, 'D'),
]
for (iters, ap50s), (label, color, ls, lw, mk) in zip(pl_curves, pl_models):
    plot_curve(ax_pl, iters, ap50s, label, color, ls, lw, mk)

add_lr_decay_lines(ax_pl, [(40000, 'LR×0.1\n@40k'), (47000, 'LR×0.01\n@47k')])
ax_pl.set_xlabel('Iteration'); ax_pl.set_ylabel('Segm AP50 (%)')
ax_pl.set_title('Plantations (PL) — 7 kelas, 1 683 gambar')
ax_pl.set_xlim(0, 52000); ax_pl.set_ylim(35, 57)
ax_pl.set_xticks(XTICK50); ax_pl.set_xticklabels(XLB50)
ax_pl.legend(loc='upper left', framealpha=0.88); ax_pl.grid(axis='y', alpha=0.25)
ax_pl.grid(axis='x', alpha=0.12)
ax_pl.text(1000, 36.5, 'SwinL PL dilanjutkan\ndi Fase 3 (75k)',
           fontsize=8.5, color='#9E9E9E', style='italic')

fig2.text(0.5, -0.03,
          'PL AP50 lebih tinggi dari RF (kelas buah lebih distinctive). '
          'SwinT dan SwinB dipilih untuk fase selanjutnya.',
          ha='center', fontsize=9, style='italic', color='#424242')
plt.tight_layout()
fig2.savefig(f'{OUT_DIR}/fase2_training_terpisah.png', dpi=150, bbox_inches='tight', facecolor='white')
print('✓ Fase 2 saved')
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — Combined PL+RF (75k iter)
# ══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
fig3.suptitle('Fase 3 — Combined PL+RF  (75 000 iterations)  ·  Segm AP50 @ Val Set  ·  eval tiap 5k',
              fontsize=12, fontweight='bold', y=1.01)

fase3_models = [
    ('SwinT Combined (C1)', 'SwinT_combined_rle_f1000_75k_55768234', '#1976D2', '-',  2.0, 'o',  False),
    ('SwinL Combined (C2)', 'SwinL_combined_rle_f1000_75k_55786426', '#7B1FA2', '-',  2.0, 's',  False),
    ('SwinB Combined (C3)', 'SwinB_combined_rle_f1000_75k_55797820', '#2E7D32', '-',  2.5, 'D',  False),
    ('FocalNet-L Combined', 'FocalNet_L_combined_rle_f1000_75k_55884266', '#C62828', '-', 2.5, '^', True),
]
for label, log, color, ls, lw, mk, running in fase3_models:
    iters, ap50s = extract_curves(log, 75000)
    plot_curve(ax3, iters, ap50s, label, color, ls, lw, mk,
               ann_xoff=600, ann_yoff=0.6, running=running)

add_lr_decay_lines(ax3, [(60000, 'LR×0.1\n@60k'), (70000, 'LR×0.01\n@70k')], y_text_pct=0.05)
ax3.set_xlabel('Iteration'); ax3.set_ylabel('Segm AP50 (%)')
ax3.set_title('Combined Dataset (RF + PL, 13 kelas, 3 964 gambar train) — Backbone Scaling')
ax3.set_xlim(0, 77000); ax3.set_ylim(10, 55)
ax3.set_xticks(XTICK75); ax3.set_xticklabels(XLB75)
ax3.legend(loc='upper left', framealpha=0.88); ax3.grid(axis='y', alpha=0.25)
ax3.grid(axis='x', alpha=0.12)
ax3.annotate('★ masih\nrunning', xy=(30000, 46.85), xytext=(26000, 49),
             fontsize=8, color='#C62828',
             arrowprops=dict(arrowstyle='->', color='#C62828', lw=1))

fig3.text(0.5, -0.03,
          'SwinB Combined (C3) terbaik dari Swin family: 47.06% @75k. '
          'FocalNet-L belum selesai namun sudah melampaui SwinB @30k.',
          ha='center', fontsize=9, style='italic', color='#424242')
plt.tight_layout()
fig3.savefig(f'{OUT_DIR}/fase3_combined.png', dpi=150, bbox_inches='tight', facecolor='white')
print('✓ Fase 3 saved')
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 — DBH Regression Extension  (2×2 ablation: ±Depth × ±DBH)
# ══════════════════════════════════════════════════════════════════════════════
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(15, 6))
fig4.suptitle(
    'Fase 4 — DBH Regression Extension  ·  75k iter  ·  Segm AP50 @ Val Set  ·  eval tiap 5k\n'
    'Ablasi 2×2: RGB / RGBD  ×  tanpa DBH / + DBH head',
    fontsize=12, fontweight='bold', y=1.01)

# Warna konsisten: RGB=biru, RGBD=teal  |  tanpa DBH=solid, +DBH=dashed
COLORS = {
    'rgb':      '#1976D2',  # biru
    'rgb_dbh':  '#E65100',  # oranye
    'rgbd':     '#0097A7',  # teal
    'rgbd_dbh': '#9C27B0',  # ungu
}

# ── SwinT RF — panel kiri ─────────────────────────────────────────────────────
swint_configs = [
    # label                       log                                          color               ls    lw   mk    max_iter running
    ('RGB only (E1, 75k★)',       'SwinT_rf_rle_f1000_100k_55749598',         COLORS['rgb'],      '-',  2.5, 'o',  75000,  False),
    ('RGBD only (E7v2)',          'SwinT_rf_rle_f1000_rgbd_75k_55775967',     COLORS['rgbd'],     '-',  2.0, 's',  75000,  False),
    ('RGB + DBH (E5, 50k)',       'SwinT_rf_rle_f1000_dbh_55764347',          COLORS['rgb_dbh'],  '--', 2.0, 'o',  50000,  False),
    ('RGBD + DBH (E6v2)',         'SwinT_rf_rle_f1000_rgbd_dbh_75k_55775802', COLORS['rgbd_dbh'], '--', 2.0, 's',  75000,  False),
]
for label, log, color, ls, lw, mk, mxi, running in swint_configs:
    iters, ap50s = extract_curves(log, mxi)
    plot_curve(ax4a, iters, ap50s, label, color, ls, lw, mk,
               ann_xoff=400, ann_yoff=0.4, running=running)

# Marker: E5 hanya 50k
ax4a.axvline(50000, color='#BDBDBD', lw=1.0, ls=':', zorder=1)
ax4a.text(50300, 7, 'E5 end\n@50k', fontsize=7.5, color='#9E9E9E')

# LR decay markers
add_lr_decay_lines(ax4a,
    [(40000,'@40k\n×0.1'), (47000,'@47k\n×0.01'),   # 50k schedule
     (60000,'@60k\n×0.1'), (70000,'@70k\n×0.01')],  # 75k schedule
    y_text_pct=0.04)

# Cold-start annotation RGBD
ax4a.annotate('Cold-start\n(4-ch init)',
              xy=(5000, 3.7), xytext=(7500, 5.5),
              fontsize=7.5, color=COLORS['rgbd'],
              arrowprops=dict(arrowstyle='->', color=COLORS['rgbd'], lw=1.0))

ax4a.set_xlabel('Iteration'); ax4a.set_ylabel('Segm AP50 (%)')
ax4a.set_title('SwinT — Rainforests (RF, 6 kelas, 2 282 gambar)')
ax4a.set_xlim(0, 77000); ax4a.set_ylim(0, 42)
ax4a.set_xticks(XTICK75); ax4a.set_xticklabels(XLB75)
ax4a.legend(loc='upper left', framealpha=0.88, fontsize=8.5)
ax4a.grid(axis='y', alpha=0.25); ax4a.grid(axis='x', alpha=0.12)

# ── SwinB Combined — panel kanan ──────────────────────────────────────────────
swinb_configs = [
    ('RGB only (C3)',            'SwinB_combined_rle_f1000_75k_55797820',         COLORS['rgb'],      '-',  2.5, 'o', 75000, False),
    ('RGBD only (running★)',     'SwinB_combined_rle_f1000_rgbd_75k_55884237',    COLORS['rgbd'],     '-',  2.0, 's', 75000, True),
    ('RGB + DBH (C4)',           'SwinB_combined_rle_f1000_75k_dbh_55848391',     COLORS['rgb_dbh'],  '--', 2.0, 'o', 75000, False),
    ('RGBD + DBH (C5)',          'SwinB_combined_rle_f1000_rgbd_dbh_75k_55848452',COLORS['rgbd_dbh'], '--', 2.0, 's', 75000, False),
]
for label, log, color, ls, lw, mk, mxi, running in swinb_configs:
    iters, ap50s = extract_curves(log, mxi)
    plot_curve(ax4b, iters, ap50s, label, color, ls, lw, mk,
               ann_xoff=400, ann_yoff=0.4, running=running)

ax4b.annotate('Cold-start\n(4-ch init)',
              xy=(5000, 5.31), xytext=(7500, 2.5),
              fontsize=7.5, color=COLORS['rgbd'],
              arrowprops=dict(arrowstyle='->', color=COLORS['rgbd'], lw=1.0))
ax4b.annotate('',
              xy=(5000, 6.01), xytext=(7500, 4.2),
              arrowprops=dict(arrowstyle='->', color=COLORS['rgbd_dbh'], lw=1.0))

add_lr_decay_lines(ax4b, [(60000,'@60k\n×0.1'), (70000,'@70k\n×0.01')], y_text_pct=0.04)
ax4b.set_xlabel('Iteration'); ax4b.set_ylabel('Segm AP50 (%)')
ax4b.set_title('SwinB — Combined (RF+PL, 13 kelas, 3 964 gambar)')
ax4b.set_xlim(0, 77000); ax4b.set_ylim(0, 52)
ax4b.set_xticks(XTICK75); ax4b.set_xticklabels(XLB75)
ax4b.legend(loc='upper left', framealpha=0.88, fontsize=8.5)
ax4b.grid(axis='y', alpha=0.25); ax4b.grid(axis='x', alpha=0.12)

# Legend skema warna
from matplotlib.lines import Line2D
legend_scheme = [
    Line2D([0],[0], color=COLORS['rgb'],      lw=2, ls='-',  label='RGB (tanpa depth)'),
    Line2D([0],[0], color=COLORS['rgbd'],     lw=2, ls='-',  label='RGBD (+ depth channel)'),
    Line2D([0],[0], color='#555', lw=2, ls='-',  label='solid = tanpa DBH head'),
    Line2D([0],[0], color='#555', lw=2, ls='--', label='dashed = + DBH head'),
]
fig4.legend(handles=legend_scheme, loc='lower center', ncol=4,
            fontsize=9, framealpha=0.88, bbox_to_anchor=(0.5, -0.08))

fig4.text(0.5, -0.14,
          'SwinT RF: RGBD cold-start berat; RGB+DBH (E5) hanya 50k — RGBD+DBH dan RGBD recover post-decay.\n'
          'SwinB Combined: DBH head -3.4pp vs baseline (gradient competition 13 kelas); RGBD+DBH masih lebih baik dari RGBD+cold-start.',
          ha='center', fontsize=8.5, style='italic', color='#424242')
plt.tight_layout()
fig4.savefig(f'{OUT_DIR}/fase4_dbh_extension.png', dpi=150, bbox_inches='tight', facecolor='white')
print('✓ Fase 4 saved')
plt.close(fig4)

print(f'\nSemua gambar tersimpan di: {OUT_DIR}/')
