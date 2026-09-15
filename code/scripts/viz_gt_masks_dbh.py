#!/usr/bin/env python3
"""
Visualisasi GT mask per-instance + posisi h=1.3m (DBH point) untuk image manapun.

Usage:
    python viz_gt_masks_dbh.py Tree306_1721040361 Tree318_1721964518
"""

import json, sys, argparse
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

# ── Konstanta ─────────────────────────────────────────────────────────────────
CY, FY, FX, H_CAM = 135.0, 240.0, 240.0, 2.0

NAMES = {1:'Apple',2:'Lemon',3:'Loquat',4:'Mango',5:'Orange',6:'Persimmon',
         7:'Pomegranate',8:'AliiFig',9:'BangaloPalm',10:'Fern',
         11:'LeechVine',12:'RubberFig',13:'Umbrella'}

INST_COLORS = ['#e74c3c','#2ecc71','#3498db','#f39c12','#9b59b6','#1abc9c','#e67e22']

ALL_JSONS = [
    'data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_train.json',
    'data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_val.json',
    'data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json',
    'data/plantations/annotations_inst/instances_train_rle.json',
    'data/plantations/annotations_inst/instances_val_rle.json',
    'data/plantations/annotations_inst/instances_test_rle.json',
]

BASE = Path(__file__).parent.parent


# ── Helpers ───────────────────────────────────────────────────────────────────
def read_pfm(path):
    with open(path, 'rb') as f:
        f.readline()
        w, h = (int(v) for v in f.readline().decode('latin-1').split())
        scale = float(f.readline().decode('latin-1').strip())
        endian = '<f4' if scale < 0 else '>f4'
        data = np.frombuffer(f.read(), dtype=endian).reshape(h, w)
    return np.flipud(data).copy()

def norm_depth(d):
    valid = d < 65000
    return np.where(valid, np.log1p(d), 0.0).astype(np.float32) / _LOG_DMAX


def find_image_annotations(stem):
    """Cari gambar di semua JSONs, return (ann_file, img_info, annotations)."""
    for jpath in ALL_JSONS:
        jf = BASE / jpath
        if not jf.exists():
            continue
        d = json.load(open(jf))
        anns_by_img = {}
        for a in d['annotations']:
            anns_by_img.setdefault(a['image_id'], []).append(a)
        for img in d['images']:
            if img['file_name'].split('/')[-1].replace('.png', '') == stem:
                return jpath, img, anns_by_img.get(img['id'], [])
    return None, None, []


def build_viz(stem, out_dir=None):
    """Buat satu figure untuk gambar dengan nama stem (tanpa .png)."""
    ann_file, img_info, annotations = find_image_annotations(stem)
    if img_info is None:
        print(f"[{stem}] SKIP — tidak ditemukan di semua annotation JSON")
        return

    print(f"\n[{stem}] ditemukan di {ann_file}")
    print(f"  {img_info['width']}x{img_info['height']}  N_ann={len(annotations)}")

    img_path = Path(img_info['file_name'])
    dep_path = Path(str(img_path).replace('rgb_resized', 'depth_pfm').replace('.png', '.pfm'))

    if not img_path.exists():
        print(f"  SKIP — RGB tidak ada: {img_path}")
        return
    if not dep_path.exists():
        print(f"  SKIP — depth tidak ada: {dep_path}")
        return

    rgb = np.array(Image.open(img_path).convert('RGB'))
    H, W = rgb.shape[:2]
    depth_raw = _denorm_depth(torch.from_numpy(norm_depth(read_pfm(str(dep_path)))))

    # ── Decode mask + hitung row h=1.3m ──────────────────────────────────────
    instances = []
    for ann in annotations:
        seg = ann['segmentation']
        rle = seg if isinstance(seg, dict) else coco_mask.merge(
              coco_mask.frPyObjects(seg, H, W))
        mask = torch.from_numpy(coco_mask.decode(rle).astype(bool))

        row, trunk_cols, d_trunk = _find_row_1p3m(mask, depth_raw, CY, FY)

        # Fallback dari depth median di area bbox
        row_fallback = None
        if row is None:
            x, y, bw, bh = ann['bbox']
            cx = int(x + bw / 2)
            c0, c1 = max(0, cx - 8), min(W, cx + 8)
            r0, r1 = max(0, int(y)), min(H, int(y + bh))
            patch = depth_raw[r0:r1, c0:c1]
            valid = (patch > 0.1) & (patch < 200)
            if valid.sum() > 0:
                d_fb = float(patch[valid].median())
                row_fallback = int(round(CY + FY * (H_CAM - 1.3) / d_fb))

        dbh_pred_mm = None
        trunk_px = None
        if row is not None and trunk_cols is not None and len(trunk_cols) > 0:
            trunk_px = len(trunk_cols)
            dbh_pred_mm = trunk_px * d_trunk * 1000.0 / FX

        r_used = row if row is not None else row_fallback
        src = 'mask' if row is not None else 'fallback'
        pred_str = f"{dbh_pred_mm:.0f}mm" if dbh_pred_mm else '—'
        print(f"  ann={ann['id']} {NAMES.get(ann['category_id'],'?')} "
              f"GT={ann.get('dbh',0):.2f}cm  row={r_used}({src})  "
              f"trunk_px={trunk_px}  pred={pred_str}")

        instances.append(dict(
            ann=ann, mask=mask.numpy().astype(bool),
            row=row, row_fallback=row_fallback, row_src=src,
            trunk_cols=trunk_cols, d_trunk=d_trunk,
            trunk_px=trunk_px, dbh_pred_mm=dbh_pred_mm,
            dbh_gt_cm=ann.get('dbh', 0),
            cat_name=NAMES.get(ann['category_id'], '?'),
            bbox=ann['bbox'],
        ))

    n = len(instances)
    if n == 0:
        print(f"  SKIP — tidak ada anotasi valid")
        return

    # ── Layout: atas=overview, bawah=per-instance ────────────────────────────
    ncols = min(n, 6)
    fig = plt.figure(figsize=(max(14, ncols * 3.2), 10), facecolor='#111827')
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.7, 1.0],
                          hspace=0.06, wspace=0.04)

    # Panel overview
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_facecolor('#1f2937')

    composite = rgb.copy().astype(float)
    for i, inst in enumerate(instances):
        c = np.array(to_rgba(INST_COLORS[i % len(INST_COLORS)])[:3]) * 255
        composite[inst['mask']] = composite[inst['mask']] * 0.45 + c * 0.55
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    ax_main.imshow(composite)

    for i, inst in enumerate(instances):
        color = INST_COLORS[i % len(INST_COLORS)]
        r_used = inst['row'] if inst['row'] is not None else inst['row_fallback']
        if r_used is None:
            continue
        x0, y0, bw, bh = inst['bbox']
        ax_main.plot([x0, x0 + bw], [r_used, r_used],
                     color='black', linewidth=5, alpha=0.55, solid_capstyle='round')
        ax_main.plot([x0, x0 + bw], [r_used, r_used],
                     color=color, linewidth=2.5, alpha=1.0, solid_capstyle='round')
        ax_main.text(x0 + bw / 2, r_used - 5, f'#{i+1} h=1.3m',
                     ha='center', va='bottom', fontsize=7.5, color=color, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.65, ec='none'))
        rect = mpatches.Rectangle((x0, y0), bw, bh, linewidth=1.8,
                                   edgecolor=color, facecolor='none',
                                   linestyle='--', alpha=0.8)
        ax_main.add_patch(rect)

    handles = [
        mpatches.Patch(color=INST_COLORS[i % len(INST_COLORS)],
                       label=f'#{i+1} {inst["cat_name"]} GT={inst["dbh_gt_cm"]:.2f}cm')
        for i, inst in enumerate(instances)
    ]
    ax_main.legend(handles=handles, loc='upper right', fontsize=7.5,
                   facecolor='#1f2937', edgecolor='#374151', labelcolor='white', framealpha=0.9)
    ax_main.set_title(f'{stem}  —  GT Masks + Posisi h=1.3m per Instance',
                      color='white', fontsize=11, fontweight='bold', pad=6)
    ax_main.axis('off')

    # Panel per-instance
    for i, inst in enumerate(instances[:ncols]):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor('#1f2937')
        color = INST_COLORS[i % len(INST_COLORS)]

        x0, y0, bw, bh = inst['bbox']
        pad = 14
        r0 = max(0, int(y0) - pad);  r1 = min(H, int(y0 + bh) + pad)
        c0 = max(0, int(x0) - pad);  c1 = min(W, int(x0 + bw) + pad)

        crop = rgb[r0:r1, c0:c1].copy().astype(float)
        m_crop = inst['mask'][r0:r1, c0:c1]
        cv = np.array(to_rgba(color)[:3]) * 255
        crop[m_crop] = crop[m_crop] * 0.35 + cv * 0.65
        ax.imshow(np.clip(crop, 0, 255).astype(np.uint8))

        r_used = inst['row'] if inst['row'] is not None else inst['row_fallback']
        if r_used is not None:
            rl = r_used - r0
            ax.axhline(rl, color='black', linewidth=5, alpha=0.55)
            ax.axhline(rl, color=color, linewidth=2.5)
            ax.text((c1 - c0) / 2, rl - 3, 'DBH',
                    ha='center', va='bottom', fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7, ec='none'))

            if inst['trunk_cols'] is not None and len(inst['trunk_cols']) > 0:
                tc = inst['trunk_cols'].numpy() - c0
                valid = (tc >= 0) & (tc < (c1 - c0))
                if valid.sum() > 0:
                    ax.scatter(tc[valid], np.full(valid.sum(), rl),
                               c='yellow', s=25, marker='s', linewidths=0, zorder=8)
                    span = tc[valid]
                    ax.annotate('', xy=(span.max()+2, rl-4), xytext=(span.min()-2, rl-4),
                                arrowprops=dict(arrowstyle='<->', color='yellow', lw=1.2))
                    ax.text((span.min()+span.max())/2, rl-10,
                            f'{inst["trunk_px"]}px',
                            ha='center', va='bottom', fontsize=7,
                            color='yellow', fontweight='bold')

        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])

        pred_str = f"{inst['dbh_pred_mm']:.0f}mm" if inst['dbh_pred_mm'] else '—'
        gt_mm = inst['dbh_gt_cm'] * 10
        ax.set_title(
            f"#{i+1}  {inst['cat_name']}\n"
            f"GT={inst['dbh_gt_cm']:.2f}cm ({gt_mm:.0f}mm)\n"
            f"pred={pred_str}  ({inst['row_src']})",
            fontsize=7, color='#e2e8f0', pad=3)

    fig.text(0.5, 0.002,
             'Garis = posisi h=1.3m  |  Titik kuning = trunk cols (depth filter)  |  '
             'Kotak putus = GT bbox  |  mask = fallback jika mask tidak cover h=1.3m',
             ha='center', fontsize=8, color='#6b7280')

    if out_dir is None:
        out_dir = BASE / 'reports/dbh_eval'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{stem}_gt_masks_dbh.png'
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stems', nargs='+', help='Image stem(s), e.g. Tree306_1721040361')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    for stem in args.stems:
        build_viz(stem.replace('.png', ''), args.out)
