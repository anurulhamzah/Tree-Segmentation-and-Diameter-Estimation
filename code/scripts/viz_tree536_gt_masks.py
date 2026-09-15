#!/usr/bin/env python3
"""
Visualisasi GT mask per-instance untuk Tree536_1721961020.png
+ posisi h=1.3m (DBH measurement point) pada setiap instance.
"""

import json, sys, math
import numpy as np
import torch
from pathlib import Path
from pycocotools import mask as coco_mask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from trunk_roi_dbh_head import _denorm_depth, _LOG_DMAX, _find_row_1p3m

# ── Konstanta kamera ─────────────────────────────────────────────────────────
CY, FY, FX = 135.0, 240.0, 240.0
H_CAM = 2.0

NAMES = {1:'Apple',2:'Lemon',3:'Loquat',4:'Mango',5:'Orange',6:'Persimmon',
         7:'Pomegranate',8:'AliiFig',9:'BangaloPalm',10:'Fern',
         11:'LeechVine',12:'RubberFig',13:'Umbrella'}

# Warna per instance (RGBA, untuk overlay dan anotasi)
INST_COLORS = [
    '#e74c3c',  # merah
    '#2ecc71',  # hijau
    '#3498db',  # biru
    '#f39c12',  # oranye
    '#9b59b6',  # ungu
]

# ── I/O paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
ANN_FILE = BASE / 'data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_train.json'
IMG_NAME = 'Tree536_1721961020'
RGB_PATH = BASE / f'data/plantations/rgb_resized/{IMG_NAME}.png'
DEP_PATH = BASE / f'data/plantations/depth_pfm/{IMG_NAME}.pfm'
OUT_PATH = BASE / f'reports/dbh_eval/{IMG_NAME}_gt_masks_dbh.png'

# ── Load data ────────────────────────────────────────────────────────────────
def read_pfm(path):
    with open(path, 'rb') as f:
        hdr = f.readline().decode('latin-1').strip()
        w, h = (int(v) for v in f.readline().decode('latin-1').split())
        scale = float(f.readline().decode('latin-1').strip())
        endian = '<f4' if scale < 0 else '>f4'
        data = np.frombuffer(f.read(), dtype=endian).reshape(h, w)
    return np.flipud(data).copy()

def norm_depth(d):
    valid = d < 65000
    return np.where(valid, np.log1p(d), 0.0).astype(np.float32) / _LOG_DMAX

rgb = np.array(Image.open(RGB_PATH).convert('RGB'))
depth_raw_np = read_pfm(str(DEP_PATH))
depth_norm = norm_depth(depth_raw_np)
depth_raw = _denorm_depth(torch.from_numpy(depth_norm))  # tensor (H, W), meter

H, W = rgb.shape[:2]

# Load annotations
ann_data = json.load(open(ANN_FILE))
imgid_map = {i['file_name'].split('/')[-1].replace('.png',''): i for i in ann_data['images']}
img_info = imgid_map[IMG_NAME]
img_id = img_info['id']
annotations = [a for a in ann_data['annotations'] if a['image_id'] == img_id]
print(f"Loaded {len(annotations)} annotations for {IMG_NAME}")

# ── Decode setiap mask + hitung row 1.3m ─────────────────────────────────────
instances = []
for ann in annotations:
    seg = ann['segmentation']
    rle = seg if isinstance(seg, dict) else coco_mask.merge(coco_mask.frPyObjects(seg, H, W))
    mask = torch.from_numpy(coco_mask.decode(rle).astype(bool))

    # Hitung row h=1.3m dengan _find_row_1p3m (butuh mask + depth)
    row, trunk_cols, d_trunk = _find_row_1p3m(mask, depth_raw, CY, FY)

    # Fallback: hitung dari depth_mean annotation jika ada
    depth_mean_ann = ann.get('depth_mean', None)

    # Hitung dari bbox center column — fallback depth dari depth map kolom bbox
    x, y, bw, bh = ann['bbox']
    cx_bbox = int(x + bw / 2)
    row_fallback = None
    if row is None and depth_mean_ann is not None:
        d_fb = float(depth_mean_ann)
        row_fallback = int(round(CY + FY * (H_CAM - 1.3) / d_fb)) if d_fb > 0 else None
    elif row is None:
        # Ambil median depth di area bbox kolom tengah
        col_range = slice(max(0, cx_bbox-5), min(W, cx_bbox+5))
        row_range = slice(max(0, int(y)), min(H, int(y+bh)))
        d_bbox = depth_raw[row_range, col_range]
        valid = (d_bbox > 0.1) & (d_bbox < 200)
        if valid.sum() > 0:
            d_fb = float(d_bbox[valid].median())
            row_fallback = int(round(CY + FY * (H_CAM - 1.3) / d_fb))

    # Predicted DBH dari trunk cols jika ada
    if row is not None and trunk_cols is not None and len(trunk_cols) > 0:
        trunk_px = len(trunk_cols)
        dbh_pred_mm = trunk_px * d_trunk * 1000.0 / FX
    else:
        trunk_px = None
        dbh_pred_mm = None

    instances.append({
        'ann': ann,
        'mask': mask.numpy().astype(bool),
        'row': row,                         # dari _find_row_1p3m (bisa None)
        'row_fallback': row_fallback,        # dari depth_mean/bbox
        'trunk_cols': trunk_cols,
        'd_trunk': d_trunk,
        'trunk_px': trunk_px,
        'dbh_pred_mm': dbh_pred_mm,
        'dbh_gt_cm': ann.get('dbh', 0),
        'cat_name': NAMES.get(ann['category_id'], '?'),
        'bbox': ann['bbox'],
    })
    r_used = row if row is not None else row_fallback
    print(f"  ann={ann['id']} {NAMES.get(ann['category_id'],'?')} "
          f"dbh_gt={ann.get('dbh',0):.2f}cm  row_1.3m={r_used}  "
          f"trunk_px={trunk_px}  dbh_pred={dbh_pred_mm:.0f}mm" if dbh_pred_mm else
          f"  ann={ann['id']} {NAMES.get(ann['category_id'],'?')} "
          f"dbh_gt={ann.get('dbh',0):.2f}cm  row_1.3m={r_used}  trunk_px={trunk_px} (no pred)")

# ── Plot ─────────────────────────────────────────────────────────────────────
n_inst = len(instances)
fig = plt.figure(figsize=(16, 10), facecolor='#111827')

# Grid: baris atas = overview (span semua kolom), baris bawah = per-instance
gs = fig.add_gridspec(2, n_inst, height_ratios=[1.6, 1.0], hspace=0.08, wspace=0.04)

# ── Panel atas: overview ──────────────────────────────────────────────────────
ax_main = fig.add_subplot(gs[0, :])
ax_main.set_facecolor('#1f2937')

# Base RGB
composite = rgb.copy().astype(float)

# Overlay semua mask dengan warna berbeda
alpha_mask = 0.45
for i, inst in enumerate(instances):
    color = np.array(to_rgba(INST_COLORS[i % len(INST_COLORS)])[:3]) * 255
    m = inst['mask']
    composite[m] = composite[m] * (1 - alpha_mask) + color * alpha_mask

composite = np.clip(composite, 0, 255).astype(np.uint8)
ax_main.imshow(composite)

# Gambar garis h=1.3m per instance + label
for i, inst in enumerate(instances):
    color = INST_COLORS[i % len(INST_COLORS)]
    r_used = inst['row'] if inst['row'] is not None else inst['row_fallback']
    if r_used is None:
        continue
    x0, y0, bw, bh = inst['bbox']
    xc = x0 + bw / 2

    # Garis tebal di range bbox
    ax_main.plot([x0, x0 + bw], [r_used, r_used],
                 color='black', linewidth=5, alpha=0.6, solid_capstyle='round')
    ax_main.plot([x0, x0 + bw], [r_used, r_used],
                 color=color, linewidth=2.5, alpha=1.0, solid_capstyle='round')

    # Label: nomor instance
    ax_main.text(xc, r_used - 5, f'#{i+1}\nh=1.3m',
                 ha='center', va='bottom', fontsize=7.5, color=color, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.65, ec='none'))

    # Bounding box
    rect = mpatches.Rectangle((x0, y0), bw, bh,
                               linewidth=1.8, edgecolor=color, facecolor='none',
                               linestyle='--', alpha=0.85)
    ax_main.add_patch(rect)

# Legend
legend_handles = [
    mpatches.Patch(color=INST_COLORS[i], label=f'#{i+1}  {inst["cat_name"]}  GT={inst["dbh_gt_cm"]:.2f}cm')
    for i, inst in enumerate(instances)
]
ax_main.legend(handles=legend_handles, loc='upper right', fontsize=8,
               facecolor='#1f2937', edgecolor='#374151', labelcolor='white', framealpha=0.9)
ax_main.set_title(f'{IMG_NAME}  —  GT Masks + Posisi h=1.3m per Instance',
                  color='white', fontsize=11, fontweight='bold', pad=6)
ax_main.axis('off')

# ── Panel bawah: satu per instance ───────────────────────────────────────────
for i, inst in enumerate(instances):
    ax = fig.add_subplot(gs[1, i])
    ax.set_facecolor('#1f2937')
    color = INST_COLORS[i % len(INST_COLORS)]

    # Crop ke bbox (dengan margin)
    x0, y0, bw, bh = inst['bbox']
    pad = 12
    r0 = max(0, int(y0) - pad)
    r1 = min(H, int(y0 + bh) + pad)
    c0 = max(0, int(x0) - pad)
    c1 = min(W, int(x0 + bw) + pad)

    crop_rgb = rgb[r0:r1, c0:c1].copy().astype(float)
    m_crop = inst['mask'][r0:r1, c0:c1]
    color_arr = np.array(to_rgba(color)[:3]) * 255
    crop_rgb[m_crop] = crop_rgb[m_crop] * 0.35 + color_arr * 0.65
    crop_rgb = np.clip(crop_rgb, 0, 255).astype(np.uint8)
    ax.imshow(crop_rgb)

    # Garis h=1.3m dalam crop coordinates
    r_used = inst['row'] if inst['row'] is not None else inst['row_fallback']
    if r_used is not None:
        r_local = r_used - r0
        ax.axhline(r_local, color='black', linewidth=5, alpha=0.6)
        ax.axhline(r_local, color=color, linewidth=2.5, alpha=1.0)

        # Tanda "DBH di sini"
        ax.text((c1 - c0) / 2, r_local - 3, 'DBH',
                ha='center', va='bottom', fontsize=8, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7, ec='none'))

        # Trunk cols jika ada
        if inst['trunk_cols'] is not None and len(inst['trunk_cols']) > 0:
            tc_local = inst['trunk_cols'].numpy() - c0
            valid_tc = (tc_local >= 0) & (tc_local < (c1 - c0))
            if valid_tc.sum() > 0:
                ax.scatter(tc_local[valid_tc], np.full(valid_tc.sum(), r_local),
                           c='yellow', s=25, marker='s', linewidths=0, zorder=8)
                span = tc_local[valid_tc]
                ax.annotate('', xy=(span.max()+2, r_local-4), xytext=(span.min()-2, r_local-4),
                            arrowprops=dict(arrowstyle='<->', color='yellow', lw=1.2))
                ax.text((span.min()+span.max())/2, r_local-10,
                        f'{inst["trunk_px"]}px',
                        ha='center', va='bottom', fontsize=7, color='yellow', fontweight='bold')

    # Frame warna instance
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)
    ax.set_xticks([]); ax.set_yticks([])

    # Title
    r_info = 'mask-based' if inst['row'] is not None else 'depth-fallback'
    dbh_pred_str = f"{inst['dbh_pred_mm']:.0f}mm" if inst['dbh_pred_mm'] else '—'
    title = (f"#{i+1}  {inst['cat_name']}\n"
             f"GT={inst['dbh_gt_cm']:.2f}cm ({inst['dbh_gt_cm']*10:.0f}mm)\n"
             f"pred={dbh_pred_str}  ({r_info})")
    ax.set_title(title, fontsize=7, color='#e2e8f0', pad=3)

# ── Footer ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.002,
         'Garis horizontal = posisi h=1.3m per pohon  |  '
         'Titik kuning = trunk columns (lolos depth filter)  |  '
         'Kotak putus = GT bounding box',
         ha='center', fontsize=8, color='#6b7280')

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'\nSaved: {OUT_PATH}')
