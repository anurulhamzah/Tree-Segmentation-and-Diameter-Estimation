#!/usr/bin/env python3
"""
Audit GT annotations where bbox and segmentation strongly disagree
(large bbox but tiny mask) -> candidate "main tree not fully annotated".

Uses ONLY existing COCO RLE JSON (no instance-id PNG needed yet).
Outputs:
  - gt_audit_annotations.csv  : one row per flagged annotation (all splits)
  - gt_audit_files_to_fix.csv : one row per image to fix (priority subset)
Run: python3 gt_bbox_seg_audit.py
"""
import json, csv
from pathlib import Path
from collections import defaultdict

DIR = '/home/anur0018/pr65_scratch/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000'
OUT = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/scripts'
SPLITS = ['train', 'val', 'test']

def scene(fn): return 'plantation' if '/plantations/' in fn else ('rainforest' if '/rainforests/' in fn else 'unknown')

# --- flag definitions -------------------------------------------------------
# BROAD  : bbox spans >=40% frame AND mask <=6% frame  (bbox vs seg strongly differ)
# PLANT_ERR (high-confidence GT error): the CLOSEST tree of a plantation image
#            spans >=35% frame, mask <=18% frame, depth_mean <=6 m
def broad(bbox_frac, area_frac):
    return bbox_frac >= 0.40 and area_frac <= 0.06

ann_rows = []          # all flagged annotations
file_rows = []         # priority images to fix
summary = []

for split in SPLITS:
    d = json.load(open(f'{DIR}/instances_{split}.json'))
    catname = {c['id']: c['name'] for c in d['categories']}
    id2img = {im['id']: im for im in d['images']}
    img2ann = defaultdict(list)
    for a in d['annotations']:
        img2ann[a['image_id']].append(a)

    n_broad_ann = 0
    broad_imgs = set()
    sc_broad = defaultdict(int)
    plant_err_imgs = []

    for im in d['images']:
        sc = scene(im['file_name'])
        imgA = im['width'] * im['height']
        fname = Path(im['file_name']).name
        anns = img2ann[im['id']]

        # per-annotation broad flag
        for a in anns:
            bx = a['bbox']; bba = bx[2] * bx[3]
            if bba <= 0:
                continue
            bfrac = bba / imgA
            afrac = a['area'] / imgA
            fill = a['area'] / bba
            if broad(bfrac, afrac):
                n_broad_ann += 1
                broad_imgs.add(im['id'])
                sc_broad[sc] += 1
                ann_rows.append({
                    'split': split, 'scene': sc, 'file_name': fname,
                    'image_id': im['id'], 'ann_id': a['id'],
                    'category': catname.get(a['category_id'], a['category_id']),
                    'bbox': f"{int(bx[0])},{int(bx[1])},{int(bx[2])},{int(bx[3])}",
                    'bbox_pct_frame': round(bfrac * 100, 1),
                    'mask_pct_frame': round(afrac * 100, 1),
                    'fill_ratio': round(fill, 3),
                    'bbox_over_mask': round(1 / fill, 1) if fill > 0 else 999,
                    'depth_mean': a.get('depth_mean'), 'depth_min': a.get('depth_min'),
                    'severity': round(bfrac / max(fill, 1e-3), 1),
                })

        # high-confidence plantation GT error (closest tree under-annotated)
        if sc == 'plantation':
            da = [a for a in anns if a.get('depth_mean') is not None]
            if da:
                near = min(da, key=lambda a: a['depth_mean'])
                bx = near['bbox']; bba = bx[2] * bx[3]
                if bba > 0:
                    bfrac = bba / imgA; afrac = near['area'] / imgA
                    if bfrac >= 0.35 and afrac <= 0.18 and near['depth_mean'] <= 6.0:
                        plant_err_imgs.append(im['id'])
                        file_rows.append({
                            'split': split, 'scene': sc, 'file_name': fname,
                            'image_id': im['id'], 'n_gt': len(anns),
                            'closest_cat': catname.get(near['category_id'], near['category_id']),
                            'closest_bbox_pct': round(bfrac * 100, 1),
                            'closest_mask_pct': round(afrac * 100, 1),
                            'closest_fill': round(near['area'] / bba, 3),
                            'closest_depth_m': round(near['depth_mean'], 2),
                            'priority': 'HIGH (plantation foreground under-annotated)',
                        })

    nimg = len(d['images'])
    summary.append((split, nimg, len(d['annotations']), n_broad_ann, len(broad_imgs),
                    dict(sc_broad), len(plant_err_imgs)))

# sort outputs by severity / priority
ann_rows.sort(key=lambda r: r['severity'], reverse=True)
file_rows.sort(key=lambda r: (r['split'], r['closest_fill']))

# write CSVs
with open(f'{OUT}/gt_audit_annotations.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(ann_rows[0].keys()))
    w.writeheader(); w.writerows(ann_rows)
with open(f'{OUT}/gt_audit_files_to_fix.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(file_rows[0].keys()))
    w.writeheader(); w.writerows(file_rows)

# ---- summary ----
print('=' * 78)
print(f'{"split":<7}{"images":>8}{"anns":>8}{"broadAnn":>10}{"broadImg":>10}{"plant":>7}{"rain":>7}{"PLANT_ERR":>11}')
for s, nimg, nann, ba, bi, scb, pe in summary:
    print(f'{s:<7}{nimg:>8}{nann:>8}{ba:>10}{bi:>10}{scb.get("plantation",0):>7}{scb.get("rainforest",0):>7}{pe:>11}')
print('=' * 78)
print(f'Wrote {len(ann_rows)} flagged annotations -> gt_audit_annotations.csv')
print(f'Wrote {len(file_rows)} HIGH-priority plantation files -> gt_audit_files_to_fix.csv')
print()
print('--- TOP 15 plantation files to fix (lowest fill = worst) ---')
pf = [r for r in file_rows if r['scene'] == 'plantation'][:15]
for r in pf:
    print(f"  [{r['split']:<5}] {r['file_name']:<28} bbox={r['closest_bbox_pct']:>4}%  mask={r['closest_mask_pct']:>4}%  "
          f"fill={r['closest_fill']:.2f}  depth={r['closest_depth_m']}m  n_gt={r['n_gt']}")
