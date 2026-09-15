#!/usr/bin/env python3
"""Visualisasi distribusi trunk pixel width per kategori."""

import json, sys, numpy as np, torch
from pycocotools import mask as coco_mask
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from trunk_roi_dbh_head import _denorm_depth, _LOG_DMAX

def _read_pfm(path):
    with open(path, "rb") as f:
        f.readline()
        w, h = (int(v) for v in f.readline().decode("latin-1").split())
        scale = float(f.readline().decode("latin-1").strip())
        endian = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(f.read(), dtype=endian).reshape(h, w)
    return np.flipud(data).copy()

def norm_depth(d):
    valid = d < 65000
    return np.where(valid, np.log1p(d), 0.0).astype(np.float32) / _LOG_DMAX

CY, FY, FX = 135.0, 240.0, 240.0
NAMES = {1:'Apple',2:'Lemon',3:'Loquat',4:'Mango',5:'Orange',6:'Persimmon',7:'Pomegranate',
         8:'AliiFig',9:'BangaloPalm',10:'Fern',11:'LeechVine',12:'RubberFig',13:'Umbrella'}

ANN_FILE = Path(__file__).parent.parent / \
    'data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json'

ann_data = json.load(open(ANN_FILE))
imgid_to_info = {img['id']: img for img in ann_data['images']}
imgid_to_anns = {}
for a in ann_data['annotations']:
    if a.get('dbh', 0) > 0:
        imgid_to_anns.setdefault(a['image_id'], []).append(a)

# Kumpulkan 3 contoh per kategori
bins = {
    '1-2px':  [],
    '3-5px':  [],
    '6-10px': [],
    '>10px':  [],
}
depth_cache = {}
scanned = 0

for img_id, anns in list(imgid_to_anns.items()):
    if all(len(v) >= 3 for v in bins.values()):
        break
    if scanned >= 3000:
        break
    info = imgid_to_info[img_id]
    dep_path = info['file_name'].replace('rgb_resized','depth_pfm').replace('.png','.pfm')
    if dep_path not in depth_cache:
        depth_cache[dep_path] = _denorm_depth(
            torch.from_numpy(norm_depth(_read_pfm(dep_path))))
    depth_raw = depth_cache[dep_path]

    for ann in anns:
        scanned += 1
        gt_mm = ann['dbh'] * 10.0
        seg = ann['segmentation']
        rle = seg if isinstance(seg, dict) else \
              coco_mask.merge(coco_mask.frPyObjects(seg, info['height'], info['width']))
        mask = torch.from_numpy(coco_mask.decode(rle).astype(bool))

        rows_with_mask = mask.sum(dim=1).nonzero(as_tuple=False).squeeze(1)
        if len(rows_with_mask) < 3:
            continue

        best_r, best_wh = None, None
        for r in rows_with_mask.tolist():
            cols = mask[r].nonzero(as_tuple=False).squeeze(1)
            if len(cols) < 2:
                continue
            d_vals = depth_raw[r][cols]
            valid = (d_vals > 0.1) & (d_vals < 200.0)
            if valid.sum() < 2:
                continue
            d = d_vals[valid].median().item()
            wh = 2.0 - (r - CY) * d / FY
            if best_r is None or abs(wh - 1.3) < abs(best_wh - 1.3):
                best_r, best_wh = r, wh

        if best_r is None or abs(best_wh - 1.3) > 0.25:
            continue

        row_cols = mask[best_r].nonzero(as_tuple=False).squeeze(1)
        row_deps = depth_raw[best_r][row_cols]
        valid = (row_deps > 0.1) & (row_deps < 200.0)
        row_cols, row_deps = row_cols[valid], row_deps[valid]
        if len(row_cols) < 2:
            continue

        d_min = row_deps.min().item()
        trunk_m = row_deps <= d_min + 0.3
        trunk_px = int(trunk_m.sum().item())
        d_trunk = row_deps[trunk_m].mean().item()
        trunk_cols = row_cols[trunk_m]
        expected_px = gt_mm * FX / (d_trunk * 1000.0)

        rec = dict(img_id=img_id, ann=ann, mask=mask, best_r=best_r,
                   trunk_cols=trunk_cols, row_cols=row_cols,
                   d_trunk=d_trunk, gt_mm=gt_mm, expected_px=expected_px,
                   trunk_px=trunk_px, best_wh=best_wh, info=info)

        if trunk_px <= 2 and len(bins['1-2px']) < 3:
            bins['1-2px'].append(rec)
        elif 3 <= trunk_px <= 5 and len(bins['3-5px']) < 3:
            bins['3-5px'].append(rec)
        elif 6 <= trunk_px <= 10 and len(bins['6-10px']) < 3:
            bins['6-10px'].append(rec)
        elif trunk_px > 10 and len(bins['>10px']) < 3:
            bins['>10px'].append(rec)

print('Found:', {k: len(v) for k, v in bins.items()})

# ── Visualisasi ─────────────────────────────────────────────────────────────
CAT_COLORS = {
    '1-2px':  '#e74c3c',
    '3-5px':  '#e67e22',
    '6-10px': '#2ecc71',
    '>10px':  '#3498db',
}
CAT_LABELS = {
    '1-2px':  'trunk_px = 1–2   (44.5% dari data)',
    '3-5px':  'trunk_px = 3–5   (30.0% dari data)',
    '6-10px': 'trunk_px = 6–10  (13.9% dari data)',
    '>10px':  'trunk_px > 10    (11.6% dari data)',
}

fig, axes = plt.subplots(4, 3, figsize=(17, 23))
fig.patch.set_facecolor('#111827')

for row_i, (cat, recs) in enumerate(bins.items()):
    for col_i in range(3):
        ax = axes[row_i, col_i]
        ax.set_facecolor('#1f2937')
        if col_i >= len(recs):
            ax.axis('off')
            continue

        rec = recs[col_i]
        info = rec['info']
        rgb = np.array(Image.open(info['file_name']).convert('RGB'))

        # Mask overlay
        mask_np = rec['mask'].numpy()
        overlay = rgb.copy().astype(float)
        overlay[mask_np] = overlay[mask_np] * 0.4 + np.array([100, 180, 255]) * 0.6
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        ax.imshow(overlay)

        # Row 1.3m — garis solid tebal dengan label
        r = rec['best_r']
        H, W = rgb.shape[:2]
        # Shadow hitam dulu supaya kontras di background apapun
        ax.axhline(r, color='black', linewidth=4.0, alpha=0.7)
        ax.axhline(r, color='#facc15', linewidth=2.0, alpha=1.0)
        # Label "h=1.3m" di kiri
        ax.text(4, r - 4, f'h≈{rec["best_wh"]:.2f}m',
                fontsize=7, color='#facc15', fontweight='bold',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.6, ec='none'))

        # Semua mask cols di row itu (putih kecil)
        all_c = rec['row_cols'].numpy()
        ax.scatter(all_c, np.full(len(all_c), r),
                   c='white', s=8, alpha=0.35, marker='s', linewidths=0)

        # Trunk cols (kuning terang, lebih besar)
        tc = rec['trunk_cols'].numpy()
        if len(tc) > 0:
            ax.scatter(tc, np.full(len(tc), r),
                       c='#ff0', s=45, marker='s', linewidths=0, zorder=6)
            # Bracket penanda trunk span
            x0, x1 = int(tc.min()), int(tc.max())
            ax.annotate('', xy=(x1+3, r-4), xytext=(x0-3, r-4),
                        arrowprops=dict(arrowstyle='<->', color='#ff0', lw=1.5))
            ax.text((x0+x1)/2, r-10, f'{len(tc)}px',
                    ha='center', va='bottom', fontsize=7, color='#ff0', fontweight='bold')

        pred_mm = rec['trunk_px'] * rec['d_trunk'] * 1000.0 / FX
        ratio = rec['trunk_px'] / max(rec['expected_px'], 0.1)
        cat_name = NAMES.get(rec['ann']['category_id'], '?')

        title = (f"{cat_name}  |  GT={rec['gt_mm']:.0f}mm   exp={rec['expected_px']:.1f}px\n"
                 f"d={rec['d_trunk']:.2f}m  trunk={rec['trunk_px']}px  "
                 f"pred={pred_mm:.0f}mm  ratio={ratio:.2f}")
        ax.set_title(title, fontsize=8, color='#e2e8f0', pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(CAT_COLORS[cat])
            spine.set_linewidth(2.5)

    # Label baris
    axes[row_i, 0].set_ylabel(CAT_LABELS[cat],
                               color=CAT_COLORS[cat], fontsize=9.5,
                               fontweight='bold', labelpad=10)

legend_txt = ('Biru = mask overlay     '
              'Garis kuning putus = row h≈1.3m     '
              'Titik putih = semua mask cols di row     '
              'Titik kuning = trunk cols (lolos depth filter)')
fig.text(0.5, 0.002, legend_txt, ha='center', fontsize=8, color='#94a3b8')
fig.suptitle('Distribusi Trunk Pixel Width di Row h=1.3m  —  Refined Geometric Method',
             color='white', fontsize=13, fontweight='bold', y=1.002)

plt.tight_layout(pad=0.6, h_pad=1.5, w_pad=0.4)
OUT = Path(__file__).parent.parent / 'reports/dbh_eval/trunk_pixel_distribution.png'
plt.savefig(OUT, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Saved: {OUT}')
