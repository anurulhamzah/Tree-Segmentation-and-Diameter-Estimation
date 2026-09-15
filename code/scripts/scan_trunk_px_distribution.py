#!/usr/bin/env python3
"""
Scan semua train annotations dengan dbh > 0.
Untuk setiap instance, jalankan _find_row_1p3m dan catat:
  - trunk_px   : jumlah pixel trunk di row h≈1.3m
  - depth_trunk: depth median pixel trunk (meter)
  - world_height: ketinggian dunia aktual di row yang ditemukan (m)
  - tol_used   : toleransi yang berhasil (0.5 atau inf)
  - dbh_geom_mm: prediksi geometris (trunk_px * d * 1000 / fx)
  - dbh_gt_mm  : GT DBH (cm → mm)

Hasil: CSV + 6-panel visualisasi distribusi.

Usage (nohup di node VS Code):
  nohup /scratch2/pr65/anur0018/conda_envs/maskdino/bin/python \
      scripts/scan_trunk_px_distribution.py \
      > logs/scan_trunk_px.log 2>&1 &
"""
import json, math, sys, time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

BASE   = Path('/scratch2/pr65/anur0018/tree_classification')
OUTDIR = BASE / 'reports/dbh_eval'
OUTDIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE / 'scripts'))
from trunk_roi_dbh_head import _denorm_depth, _find_row_1p3m, _LOG_DMAX

from pycocotools import mask as coco_mask
from PIL import Image

# ── Kamera SPREAD ──────────────────────────────────────────────────────────
_ORIG_H, _ORIG_W = 270.0, 480.0
_ORIG_CY, _ORIG_FY, _ORIG_FX = 135.0, 240.0, 240.0
H_CAM = 2.0

# ── Helpers ────────────────────────────────────────────────────────────────

def read_pfm(path):
    with open(path, 'rb') as f:
        f.readline()
        w, h = (int(v) for v in f.readline().decode('latin-1').split())
        scale = float(f.readline().decode('latin-1').strip())
        endian = '<f4' if scale < 0 else '>f4'
        data = np.frombuffer(f.read(), dtype=endian).reshape(h, w)
    return np.flipud(data).copy()


def load_depth_norm(img_path: Path) -> np.ndarray:
    dep_path = Path(str(img_path)
                    .replace('rgb_resized', 'depth_pfm')
                    .replace('.png', '.pfm'))
    if not dep_path.exists():
        return None
    raw = read_pfm(str(dep_path))
    norm = np.where(raw < 65000, np.log1p(raw), 0.0).astype(np.float32) / _LOG_DMAX
    return norm


def decode_mask(ann, H, W) -> np.ndarray:
    seg = ann['segmentation']
    if isinstance(seg, dict):
        rle = seg
    else:
        rle = coco_mask.merge(coco_mask.frPyObjects(seg, H, W))
    return coco_mask.decode(rle).astype(bool)


# ── Load dataset ───────────────────────────────────────────────────────────
TRAIN_JSON = BASE / 'data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_train.json'
print(f"Loading {TRAIN_JSON} ...")
d = json.load(open(TRAIN_JSON))

cats  = {c['id']: c['name'] for c in d['categories']}
imgs  = {i['id']: i for i in d['images']}

# Group annotations by image (only dbh > 0)
img2anns = defaultdict(list)
for ann in d['annotations']:
    if ann.get('dbh', 0) > 0:
        img2anns[ann['image_id']].append(ann)

n_imgs = len(img2anns)
print(f"Images with ≥1 dbh annotation: {n_imgs}")
print(f"Total dbh annotations: {sum(len(v) for v in img2anns.values())}")

# ── Scan ───────────────────────────────────────────────────────────────────
records = []
t0 = time.time()

for idx, (img_id, anns) in enumerate(img2anns.items()):
    if idx % 200 == 0:
        elapsed = time.time() - t0
        eta = elapsed / max(idx, 1) * (n_imgs - idx)
        print(f"  [{idx}/{n_imgs}] elapsed={elapsed:.0f}s  ETA={eta:.0f}s  records={len(records)}")

    img_info = imgs[img_id]
    img_path = Path(img_info['file_name'])
    H, W     = img_info['height'], img_info['width']

    # Scale kamera ke resolusi aktual gambar
    scale = H / _ORIG_H
    cy    = _ORIG_CY * scale
    fy    = _ORIG_FY * scale
    fx    = _ORIG_FX * scale

    # Load depth
    depth_norm = load_depth_norm(img_path)
    if depth_norm is None:
        for ann in anns:
            records.append({
                'ann_id': ann['id'], 'img_id': img_id,
                'cat_id': ann['category_id'], 'cat': cats[ann['category_id']],
                'dbh_gt_mm': ann['dbh'] * 10.0,
                'valid': False, 'reason': 'no_depth',
                'trunk_px': 0, 'depth_trunk': 0, 'world_height': 0,
                'tol_used': None, 'dbh_geom_mm': 0,
            })
        continue

    depth_t = torch.from_numpy(depth_norm)
    depth_raw = _denorm_depth(depth_t)

    for ann in anns:
        dbh_gt_mm = ann['dbh'] * 10.0
        mask      = decode_mask(ann, H, W)

        # Coba tol 0.5 dulu, fallback ke inf
        row, tcols, d_trunk = _find_row_1p3m(
            torch.from_numpy(mask), depth_raw, cy, fy, tol_wh=0.5)
        tol_used = 0.5

        if row is None:
            row, tcols, d_trunk = _find_row_1p3m(
                torch.from_numpy(mask), depth_raw, cy, fy, tol_wh=float('inf'))
            tol_used = float('inf')

        if row is None:
            records.append({
                'ann_id': ann['id'], 'img_id': img_id,
                'cat_id': ann['category_id'], 'cat': cats[ann['category_id']],
                'dbh_gt_mm': dbh_gt_mm, 'valid': False, 'reason': 'no_row_found',
                'trunk_px': 0, 'depth_trunk': 0, 'world_height': 0,
                'tol_used': None, 'dbh_geom_mm': 0,
            })
            continue

        trunk_px   = len(tcols)
        wh         = H_CAM - (row - cy) * d_trunk / fy
        dbh_geom   = trunk_px * d_trunk * 1000.0 / fx

        records.append({
            'ann_id': ann['id'], 'img_id': img_id,
            'cat_id': ann['category_id'], 'cat': cats[ann['category_id']],
            'dbh_gt_mm': dbh_gt_mm, 'valid': True, 'reason': 'ok',
            'trunk_px': trunk_px, 'depth_trunk': float(d_trunk),
            'world_height': float(wh), 'tol_used': tol_used,
            'dbh_geom_mm': float(dbh_geom),
        })

print(f"\nScan done: {len(records)} records in {time.time()-t0:.0f}s")

# ── Save CSV ────────────────────────────────────────────────────────────────
import csv
CSV_OUT = OUTDIR / 'trunk_px_scan_train.csv'
fieldnames = ['ann_id','img_id','cat_id','cat','dbh_gt_mm','valid','reason',
              'trunk_px','depth_trunk','world_height','tol_used','dbh_geom_mm']
with open(CSV_OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(records)
print(f"CSV saved: {CSV_OUT}")

# ── Analisis & Visualisasi ─────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

valid_recs  = [r for r in records if r['valid']]
all_cats    = sorted(set(r['cat'] for r in records))

# Warna per spesies
CAT_COLORS = {
    'Apple':       '#e74c3c',
    'Orange':      '#e67e22',
    'Lemon':       '#f1c40f',
    'Pomegranate': '#9b59b6',
    'Mango':       '#2ecc71',
    'Loquat':      '#1abc9c',
    'Persimmon':   '#e91e63',
    'BangaloPalm': '#3498db',
    'Fern':        '#27ae60',
    'AliiFig':     '#16a085',
    'RubberFig':   '#8e44ad',
    'Umbrella':    '#2980b9',
    'LeechVine':   '#c0392b',
}

bg   = '#111827'
fg   = '#f9fafb'
grid = '#374151'

fig = plt.figure(figsize=(22, 18), facecolor=bg)
fig.suptitle('Distribusi Pixel Diameter Trunk di h≈1.3m — Train Set (dbh>0)',
             color=fg, fontsize=15, fontweight='bold', y=0.98)

axes = []
for i in range(6):
    ax = fig.add_subplot(3, 2, i+1)
    ax.set_facecolor('#1f2937')
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid)
    axes.append(ax)

# ─ Panel 1: Histogram trunk_px (semua valid, stacked per spesies) ──────────
ax = axes[0]
bins = np.arange(0, 151, 3)
cat_data = {c: [r['trunk_px'] for r in valid_recs if r['cat'] == c] for c in all_cats}
bottoms = np.zeros(len(bins)-1)
for cat in all_cats:
    if not cat_data[cat]: continue
    counts, _ = np.histogram(cat_data[cat], bins=bins)
    ax.bar(bins[:-1], counts, width=3, bottom=bottoms,
           color=CAT_COLORS.get(cat, '#888'), alpha=0.85, label=cat)
    bottoms += counts
ax.set_xlabel('trunk_px (pixel)', color=fg)
ax.set_ylabel('count', color=fg)
ax.set_title(f'Histogram trunk_px — {len(valid_recs)} instance valid', color=fg, fontsize=10)
ax.legend(fontsize=7, facecolor='#1f2937', labelcolor=fg, loc='upper right', ncol=2)
ax.grid(color=grid, alpha=0.4, axis='y')
ax.xaxis.label.set_color(fg); ax.yaxis.label.set_color(fg)

# Tambah garis threshold yang diusulkan
for thr, col, lbl in [(3,'cyan','min=3'), (5,'lime','min=5'), (8,'orange','min=8')]:
    ax.axvline(thr, color=col, lw=1.5, ls='--', alpha=0.8)
    ax.text(thr+0.5, ax.get_ylim()[1]*0.85, lbl, color=col, fontsize=8)

# ─ Panel 2: Box plot trunk_px per spesies ─────────────────────────────────
ax = axes[1]
sorted_cats = sorted(all_cats, key=lambda c: -np.median([r['trunk_px'] for r in valid_recs if r['cat']==c]) if cat_data.get(c) else 0)
box_data    = [[r['trunk_px'] for r in valid_recs if r['cat']==c] for c in sorted_cats]
bp = ax.boxplot(box_data, patch_artist=True, vert=False,
                medianprops=dict(color='white', linewidth=2),
                flierprops=dict(marker='.', markerfacecolor='#6b7280', markersize=3, alpha=0.4),
                whiskerprops=dict(color='#9ca3af'), capprops=dict(color='#9ca3af'),
                boxprops=dict(linewidth=1))
for patch, cat in zip(bp['boxes'], sorted_cats):
    patch.set_facecolor(CAT_COLORS.get(cat, '#888'))
    patch.set_alpha(0.75)
ax.set_yticks(range(1, len(sorted_cats)+1))
ax.set_yticklabels(sorted_cats, color=fg, fontsize=9)
ax.set_xlabel('trunk_px', color=fg)
ax.set_title('Box plot trunk_px per spesies (sorted by median)', color=fg, fontsize=10)
ax.axvline(5, color='lime', lw=1.5, ls='--', alpha=0.7, label='min=5px')
ax.axvline(8, color='orange', lw=1.5, ls='--', alpha=0.7, label='min=8px')
ax.legend(fontsize=8, facecolor='#1f2937', labelcolor=fg)
ax.grid(color=grid, alpha=0.4, axis='x')

# ─ Panel 3: Coverage rate per spesies (berbagai threshold) ─────────────────
ax = axes[2]
thresholds = [1, 3, 5, 8, 10, 15, 20]
total_per_cat = {c: len([r for r in records if r['cat']==c]) for c in all_cats}
x_pos = np.arange(len(all_cats))
bar_w = 0.8 / len(thresholds)
cmap  = plt.cm.plasma(np.linspace(0.15, 0.95, len(thresholds)))

for i, thr in enumerate(thresholds):
    cov = []
    for c in all_cats:
        tot = total_per_cat[c]
        ok  = len([r for r in valid_recs if r['cat']==c and r['trunk_px']>=thr])
        cov.append(ok / tot * 100 if tot > 0 else 0)
    ax.bar(x_pos + i*bar_w - bar_w*len(thresholds)/2,
           cov, bar_w, color=cmap[i], alpha=0.85, label=f'≥{thr}px')

ax.set_xticks(x_pos)
ax.set_xticklabels([c[:8] for c in all_cats], rotation=35, ha='right', color=fg, fontsize=8)
ax.set_ylabel('Coverage (%)', color=fg)
ax.set_title('Coverage rate: % instance dengan trunk_px ≥ threshold', color=fg, fontsize=10)
ax.legend(fontsize=7, facecolor='#1f2937', labelcolor=fg, ncol=4, loc='upper right')
ax.set_ylim(0, 105)
ax.grid(color=grid, alpha=0.4, axis='y')

# ─ Panel 4: Scatter trunk_px vs DBH_gt (mm) — per spesies ─────────────────
ax = axes[3]
for cat in all_cats:
    sub = [r for r in valid_recs if r['cat']==cat and r['trunk_px']>=3]
    if not sub: continue
    xs = [r['trunk_px'] for r in sub]
    ys = [r['dbh_gt_mm'] for r in sub]
    ax.scatter(xs, ys, s=6, alpha=0.25, color=CAT_COLORS.get(cat,'#888'), label=cat)
ax.set_xlabel('trunk_px', color=fg)
ax.set_ylabel('DBH GT (mm)', color=fg)
ax.set_title('trunk_px vs DBH GT (mm) — semua instance trunk_px≥3', color=fg, fontsize=10)
ax.legend(fontsize=7, facecolor='#1f2937', labelcolor=fg, ncol=2, loc='upper left')
ax.grid(color=grid, alpha=0.3)
ax.set_xlim(0, 120); ax.set_ylim(0)

# ─ Panel 5: Geometric pred vs GT (trunk_px ≥ 5) ───────────────────────────
ax = axes[4]
sub5 = [r for r in valid_recs if r['trunk_px'] >= 5]
for cat in all_cats:
    sub = [r for r in sub5 if r['cat']==cat]
    if not sub: continue
    xs = [r['dbh_gt_mm'] for r in sub]
    ys = [r['dbh_geom_mm'] for r in sub]
    ax.scatter(xs, ys, s=6, alpha=0.25, color=CAT_COLORS.get(cat,'#888'), label=cat)

# Perfect prediction line
mx = max((r['dbh_gt_mm'] for r in sub5), default=500)
ax.plot([0, mx], [0, mx], '--', color='white', lw=1.5, alpha=0.6, label='perfect')
ax.set_xlabel('DBH GT (mm)', color=fg)
ax.set_ylabel('DBH Geometric pred (mm)', color=fg)
ax.set_title(f'Geometric pred vs GT — trunk_px≥5 (N={len(sub5)})', color=fg, fontsize=10)
ax.legend(fontsize=7, facecolor='#1f2937', labelcolor=fg, ncol=2)
ax.grid(color=grid, alpha=0.3)

# Hitung R² dan MAE untuk sub5
if sub5:
    g = np.array([r['dbh_gt_mm'] for r in sub5])
    p = np.array([r['dbh_geom_mm'] for r in sub5])
    mae = np.abs(g-p).mean()
    ss_res = np.sum((g-p)**2); ss_tot = np.sum((g-g.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    ax.text(0.05, 0.92, f'N={len(sub5)}  MAE={mae:.0f}mm  R²={r2:.3f}',
            transform=ax.transAxes, color=fg, fontsize=9,
            bbox=dict(fc='#1f2937', ec=grid, pad=4))

# ─ Panel 6: World height distribusi (seberapa dekat ke 1.3m) ───────────────
ax = axes[5]
wh_data = [r['world_height'] for r in valid_recs]
n_exact = sum(1 for wh in wh_data if abs(wh-1.3) <= 0.5)
n_loose = sum(1 for wh in wh_data if abs(wh-1.3) > 0.5)
ax.hist(wh_data, bins=50, color='#3b82f6', alpha=0.7, edgecolor='none')
ax.axvline(1.3, color='cyan', lw=2, ls='--', label='h=1.3m target')
ax.axvspan(0.8, 1.8, color='cyan', alpha=0.06, label='±0.5m band')
ax.set_xlabel('World height di row yang ditemukan (m)', color=fg)
ax.set_ylabel('count', color=fg)
ax.set_title(f'Distribusi world height di row yang dipilih\n'
             f'Dalam ±0.5m: {n_exact} ({n_exact/len(wh_data)*100:.1f}%)  |  '
             f'Fallback: {n_loose} ({n_loose/len(wh_data)*100:.1f}%)',
             color=fg, fontsize=9)
ax.legend(fontsize=8, facecolor='#1f2937', labelcolor=fg)
ax.grid(color=grid, alpha=0.4, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.97])
OUT_PNG = OUTDIR / 'trunk_px_distribution_train.png'
plt.savefig(OUT_PNG, dpi=130, bbox_inches='tight', facecolor=bg)
plt.close()
print(f"Plot saved: {OUT_PNG}")

# ── Summary tabel per spesies ───────────────────────────────────────────────
print("\n=== Summary per spesies (threshold trunk_px ≥ 5) ===")
print(f"{'Spesies':<15} {'N_total':>8} {'N_valid':>8} {'Cov%':>6} {'Cov≥5%':>7} "
      f"{'tpx_med':>7} {'tpx_p10':>7} {'tpx_p90':>7} {'R²_geom':>8}")
for cat in sorted(all_cats):
    total = total_per_cat[cat]
    valid_cat = [r for r in valid_recs if r['cat']==cat]
    v5    = [r for r in valid_cat if r['trunk_px']>=5]
    tpx   = np.array([r['trunk_px'] for r in valid_cat]) if valid_cat else np.array([0])
    cov   = len(valid_cat)/total*100 if total else 0
    cov5  = len(v5)/total*100 if total else 0
    if v5:
        g = np.array([r['dbh_gt_mm'] for r in v5])
        p = np.array([r['dbh_geom_mm'] for r in v5])
        ss_res = np.sum((g-p)**2); ss_tot = np.sum((g-g.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    else:
        r2 = float('nan')
    print(f"{cat:<15} {total:>8} {len(valid_cat):>8} {cov:>6.1f} {cov5:>7.1f} "
          f"{np.median(tpx):>7.1f} {np.percentile(tpx,10):>7.1f} {np.percentile(tpx,90):>7.1f} "
          f"{r2:>8.3f}")

print(f"\nDone. Total time: {time.time()-t0:.0f}s")
print(f"\nKunci threshold untuk training:")
for thr in [3, 5, 8, 10]:
    n_ok = sum(1 for r in valid_recs if r['trunk_px']>=thr)
    pct  = n_ok / len(records) * 100
    print(f"  trunk_px >= {thr:2d}: {n_ok:5d} instances ({pct:.1f}% dari semua dbh>0)")
