#!/usr/bin/env python3
"""Repair COCO masks menggunakan distance-threshold ke anchor palette color.

Problem dengan NN approach: Unreal render instance PNG dengan lighting →
pixel yang harusnya PAL[i] bervariasi warnanya → NN salah-assign ke PAL lain.

Fix: untuk tiap anotasi, temukan anchor color (palette color di mask lama),
lalu assign SEMUA pixel non-white yang jaraknya < T dari anchor ke anotasi itu.
T = 0.9 * jarak ke palette color terdekat lainnya (adaptive per image).

Input:  filtered_rle_f1000_repaired_v2  (preserves 7 added annotations)
Output: filtered_rle_f1000_repaired_v3
"""
import numpy as np, json, csv, os, time
from pathlib import Path
from collections import defaultdict
from PIL import Image
from pycocotools import mask as maskUtils

BASE   = '/home/anur0018/pr65_scratch2/anur0018'
SRC    = f'{BASE}/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v2'
DST    = f'{BASE}/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v3'
RAW    = f'{BASE}/raw_data'
REPORT = f'{BASE}/tree_classification/scripts/repair_threshold_report.csv'

os.makedirs(DST, exist_ok=True)

PAL_CSV  = f'{RAW}/plantations/color_palette_Part I.csv'
INST_DIR = f'{RAW}/plantations/instance_segmentation'
PAL      = np.array([(int(r['R']), int(r['G']), int(r['B']))
                     for r in csv.DictReader(open(PAL_CSV))])
PALSET   = set(tuple(int(x) for x in c) for c in PAL)

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

def find_anchor_color(flat_1d, old_mask_1d):
    """Cari palette color yang paling banyak exact-match di old mask."""
    sub = flat_1d[old_mask_1d]
    if len(sub) == 0:
        return None
    uv, uc = np.unique(sub, axis=0, return_counts=True)
    for j in np.argsort(-uc):
        col = tuple(int(x) for x in uv[j])
        if col in PALSET and col != (255, 255, 255):
            return np.array(col, dtype=np.int32)
    return None

def expand_threshold(flat, H, W, anchor, present):
    """Expand mask: semua non-white pixel dalam jarak T dari anchor."""
    anchor_i = anchor.astype(np.int32)

    # T adaptive = 0.9 * jarak ke palette color lain yang paling dekat
    anchor_t = tuple(int(x) for x in anchor)
    other = np.array([c for c in present
                      if tuple(int(x) for x in c) != anchor_t], dtype=np.int32)
    if len(other) > 0:
        dists_other = np.sqrt(((other - anchor_i) ** 2).sum(1))
        T = float(dists_other.min()) * 0.90
    else:
        T = 60.0  # fallback jika hanya 1 warna di gambar

    # Pastikan T minimal 30 (tangkap variasi lighting)
    T = max(T, 30.0)

    nonwhite = ~np.all(flat == 255, axis=1)
    idx_nw = np.where(nonwhite)[0]
    if len(idx_nw) == 0:
        return np.zeros(H * W, bool)

    pix = flat[idx_nw].astype(np.int32)
    dist_anchor = np.sqrt(((pix - anchor_i) ** 2).sum(1))
    dist_white  = np.sqrt(((pix - 255) ** 2).sum(1))

    sel = idx_nw[(dist_anchor < T) & (dist_anchor < dist_white)]
    newm = np.zeros(H * W, bool)
    newm[sel] = True
    return newm

report = open(REPORT, 'w', newline='')
import csv as csvmod
rw = csvmod.writer(report)
rw.writerow(['split', 'stem', 'ann_id', 'scene', 'cat_id',
             'old_area', 'new_area', 'T', 'status'])
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
                             a['area'], a['area'], 0, 'FALLBACK_no_png'])
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

        # Palette colors present (exact) in this image
        uimg = np.unique(flat, axis=0)
        present = np.array([c for c in uimg
                            if tuple(int(x) for x in c) in PALSET])

        for a in anns:
            old = rle_decode(a['segmentation'], H, W).reshape(-1)
            oa  = int(old.sum())

            anchor = find_anchor_color(flat, old)
            if anchor is None:
                # Tidak ada exact palette pixel di mask lama → fallback
                out_anns.append(a)
                stats['FALLBACK_no_anchor'] += 1
                rw.writerow([split, nm, a['id'], sc, a['category_id'],
                             oa, oa, 0, 'FALLBACK_no_anchor'])
                continue

            anchor_t = tuple(int(x) for x in anchor)
            other = np.array([c for c in present
                              if tuple(int(x) for x in c) != anchor_t], dtype=np.int32)
            if len(other) > 0:
                dists_other = np.sqrt(((other.astype(np.int32) - anchor) ** 2).sum(1))
                T = float(dists_other.min()) * 0.90
            else:
                T = 60.0
            T = max(T, 30.0)

            newm = expand_threshold(flat, H, W, anchor, present)
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
                         oa, na, round(T, 1), status])

    assert len(out_anns) == len(d['annotations']), \
        f"{split}: {len(out_anns)} != {len(d['annotations'])}"

    d['annotations'] = out_anns
    json.dump(d, open(f'{DST}/instances_{split}.json', 'w'))
    print(f'== {split}: {n_img} img, {len(out_anns)} ann ==', flush=True)

report.close()
print('STATUS:', dict(stats))
print(f'SELESAI → {DST}  ({time.time()-t0:.0f}s)')
