#!/usr/bin/env python3
"""Repair COCO masks menggunakan exact pixel match dari instance PNG baru (flat color).

Instance PNG baru = flat color sempurna (0 blended pixels) → cukup exact match.
Untuk tiap anotasi: sample instance PNG baru di mask lama → temukan dominant
exact palette color → new mask = SEMUA pixel di instance PNG dengan warna itu.

Input:  filtered_rle_f1000_repaired_v2
Output: filtered_rle_f1000_repaired_v4
"""
import numpy as np, json, csv, os, time
from pathlib import Path
from collections import defaultdict
from PIL import Image
from pycocotools import mask as maskUtils

BASE     = '/home/anur0018/pr65_scratch2/anur0018'
SRC      = f'{BASE}/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v2'
DST      = f'{BASE}/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v4'
INST_DIR = f'{BASE}/raw_data/plantations/instance_segmentation'
PAL_CSV  = f'{BASE}/raw_data/plantations/color_palette_Part I.csv'
REPORT   = f'{BASE}/tree_classification/scripts/repair_v4_report.csv'

os.makedirs(DST, exist_ok=True)

PAL    = np.array([(int(r['R']), int(r['G']), int(r['B']))
                   for r in csv.DictReader(open(PAL_CSV))])
PALSET = set(tuple(int(x) for x in c) for c in PAL)

def rle_decode(seg, h, w):
    if isinstance(seg, dict):
        r = seg
    else:
        o = maskUtils.frPyObjects(seg, h, w)
        r = o[0] if len(o) == 1 else maskUtils.merge(o)
    return maskUtils.decode(r).astype(bool)

def rle_encode(m):
    r = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
    r['counts'] = r['counts'].decode('ascii')
    return r

def scene(fn):
    return 'plantation' if '/plantations/' in fn else 'rainforest'

def find_anchor_exact(flat_1d, old_mask_1d):
    """Dominant exact palette color di dalam old mask → anchor."""
    sub = flat_1d[old_mask_1d]
    if len(sub) == 0:
        return None
    uv, uc = np.unique(sub, axis=0, return_counts=True)
    for j in np.argsort(-uc):
        col = tuple(int(x) for x in uv[j])
        if col in PALSET and col != (255, 255, 255):
            return np.array(col, dtype=np.uint8)
    return None

report = open(REPORT, 'w', newline='')
import csv as csvmod
rw = csvmod.writer(report)
rw.writerow(['split', 'stem', 'ann_id', 'scene', 'cat_id',
             'old_area', 'new_area', 'status'])
stats = defaultdict(int)
t0 = time.time()

for split in ['train', 'val', 'test']:
    d = json.load(open(f'{SRC}/instances_{split}.json'))
    img2ann = defaultdict(list)
    for a in d['annotations']:
        img2ann[a['image_id']].append(a)

    out_anns = []
    n_img = len(d['images'])
    done = 0

    for im in d['images']:
        done += 1
        if done % 300 == 0:
            print(f'[{split}] {done}/{n_img} ({time.time()-t0:.0f}s)', flush=True)
            report.flush()

        anns = img2ann[im['id']]
        if not anns:
            continue

        sc = scene(im['file_name'])
        if sc == 'rainforest':
            for a in anns:
                out_anns.append(a)
                stats['rainforest_copy'] += 1
            continue

        H, W = im['height'], im['width']
        nm = Path(im['file_name']).stem
        inst_path = Path(INST_DIR) / f'{nm}.png'

        if not inst_path.exists():
            for a in anns:
                out_anns.append(a)
                stats['FALLBACK_no_png'] += 1
                rw.writerow([split, nm, a['id'], sc, a['category_id'],
                             a['area'], a['area'], 'FALLBACK_no_png'])
            continue

        try:
            inst = np.array(Image.open(inst_path).convert('RGB'))
            if inst.shape[:2] != (H, W):
                inst = np.array(Image.fromarray(inst).resize((W, H), Image.NEAREST))
        except Exception:
            for a in anns:
                out_anns.append(a)
                stats['FALLBACK_read_err'] += 1
            continue

        flat = inst.reshape(-1, 3)

        for a in anns:
            old = rle_decode(a['segmentation'], H, W).reshape(-1)
            oa  = int(old.sum())

            anchor = find_anchor_exact(flat, old)

            if anchor is None:
                out_anns.append(a)
                stats['FALLBACK_no_anchor'] += 1
                rw.writerow([split, nm, a['id'], sc, a['category_id'],
                             oa, oa, 'FALLBACK_no_anchor'])
                continue

            # Exact match: semua pixel yang persis = anchor color
            newm = np.all(flat == anchor, axis=1)
            newm |= old  # pastikan new ⊇ old

            na = int(newm.sum())
            if na < oa:
                newm = old.copy()
                na = oa
                status = 'WARN_shrink'
            else:
                status = 'ok'

            a2 = dict(a)
            a2['segmentation'] = rle_encode(newm.reshape(H, W))
            a2['area'] = na
            a2['bbox'] = [float(x) for x in maskUtils.toBbox(a2['segmentation'])]
            out_anns.append(a2)
            stats[status] += 1
            rw.writerow([split, nm, a['id'], sc, a['category_id'],
                         oa, na, status])

    assert len(out_anns) == len(d['annotations']), \
        f"{split}: {len(out_anns)} != {len(d['annotations'])}"

    d['annotations'] = out_anns
    json.dump(d, open(f'{DST}/instances_{split}.json', 'w'))
    print(f'== {split}: {n_img} img, {len(out_anns)} ann ==', flush=True)

report.close()
print('STATUS:', dict(stats))
print(f'SELESAI → {DST}  ({time.time()-t0:.0f}s)')
