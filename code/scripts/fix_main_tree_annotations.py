#!/usr/bin/env python3
"""
Tambah anotasi pohon utama yang hilang ke filtered_rle_f1000_repaired.

Untuk setiap gambar dalam TARGET_LIST:
- Identifikasi pohon utama (nama sesuai stem file)
- Jika anotasi pohon utama belum tercakup di JSON → generate mask dari raw instance PNG
  menggunakan nearest-neighbor palette expansion (sama dgn repair_coco_tolerant.py)
- Tambahkan annotation baru ke JSON split yg relevan
- Simpan ke OUTPUT_DIR (tidak merusak file asli)
"""
import numpy as np
import json
import csv
import os
import struct
import time
from pathlib import Path
from collections import defaultdict
from PIL import Image
from pycocotools import mask as maskUtils

# ── paths ────────────────────────────────────────────────────────────────────
RAW         = '/home/anur0018/pr65_scratch2/anur0018/raw_data'
INST_PLANT  = f'{RAW}/plantations/instance_segmentation'
DEPTH_DIR   = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/plantations/depth_pfm'
OBJ_CSV     = f'{RAW}/plantations/obj_info_final - plantation.csv'
PAL_CSV     = f'{RAW}/plantations/color_palette_Part I.csv'
SRC         = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired'
OUTPUT_DIR  = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── target images ─────────────────────────────────────────────────────────────
TARGET_LIST = [
    'Tree646_1721022039','Tree570_1721051626','Tree682_1721042820',
    'Tree619_1721028834','Tree611_1721041923','Tree629_1721957387',
    'Tree617_1721040057','Tree654_1721044178','Tree535_1721042579',
    'Tree530_1721964195','Tree589_1721042268','Tree561_1721040669',
    'Tree556_1721023401','Tree655_1721959566','Tree562_1721026682',
    'Tree636_1721962498','Tree526_1721023521','Tree534_1721029764',
    'Tree675_1721963798','Tree685_1721051812','Tree650_1721044966',
    'Tree575_1721039212','Tree558_1721041141','Tree653_1721033352',
    'Tree541_1721026328','Tree664_1721041469','Tree619_1721029284',
    'Tree675_1721959975','Tree587_1721958152','Tree574_1721038275',
    'Tree611_1721031278','Tree686_1721032458','Tree630_1721957942',
    'Tree675_1721955915','Tree649_1721051800','Tree556_1721037451',
    'Tree665_1721030270','Tree627_1721039309','Tree669_1721043650',
    'Tree653_1721047521','Tree632_1721037066','Tree647_1721023489',
    'Tree666_1721036793','Tree650_1721958648','Tree637_1721961716',
    'Tree650_1721023448','Tree683_1721053678','Tree672_1721047447',
    'Tree674_1721028359','Tree675_1721030896','Tree686_1721959228',
    'Tree632_1721038584','Tree562_1721963529','Tree630_1721037428',
    'Tree526_1721028544','Tree606_1721054229','Tree89_1721963027',
    'Tree455_1721028165','Tree457_1721053967','Tree464_1721963455',
    'Tree455_1721022269','Tree473_1721026305','Tree911_1721037417',
    'Tree928_1721040094','Tree914_1721963737','Tree907_1721022865',
    'Tree922_1721965407','Tree918_1721961564','Tree922_1721958044',
    'Tree909_1721053955','Tree911_1721963260','Tree906_1721053213',
    'Tree910_1721039236','Tree910_1721958863','Tree910_1721028532',
    'Tree920_1721962335','Tree909_1721030492','Tree906_1721022829',
]
TARGET_SET = set(TARGET_LIST)

# ── species → category_id ────────────────────────────────────────────────────
SPECIES_TO_CAT = {
    'Apple': 1, 'Lemon': 2, 'Loquat': 3, 'Mango': 4,
    'Orange': 5, 'Persimmon': 6, 'Pomegranate': 7,
    'AliiFig': 8, 'BangaloPalm': 9, 'Fern': 10,
    'LeechVine': 11, 'RubberFig': 12, 'Umbrella': 13,
}

# ── load palette ──────────────────────────────────────────────────────────────
PAL = np.array([(int(r['R']), int(r['G']), int(r['B']))
                for r in csv.DictReader(open(PAL_CSV))])
PALSET = set(tuple(int(x) for x in c) for c in PAL)

# ── load obj_info ─────────────────────────────────────────────────────────────
OBJ_INFO = {}   # tree_name → {DBH, Height, Crown_Diameter, Species}
with open(OBJ_CSV) as f:
    for r in csv.DictReader(f):
        name = r.get('ActorName_Unreal', '').strip()
        if name:
            OBJ_INFO[name] = {
                'dbh':            float(r['DBH']) if r.get('DBH') else 0.0,
                'height_ue':      float(r['Height']) if r.get('Height') else 0.0,
                'crown_diameter': float(r['Crown_Diameter']) if r.get('Crown_Diameter') else 0.0,
                'species':        r.get('Species', '').strip(),
            }

# ── helpers ───────────────────────────────────────────────────────────────────
def rle_decode(seg, h, w):
    if isinstance(seg, dict):
        r = seg
    else:
        o = maskUtils.frPyObjects(seg, h, w)
        r = o[0] if len(o) == 1 else maskUtils.merge(o)
    return maskUtils.decode(r).astype(bool)

def rle_encode(mask_2d):
    r = maskUtils.encode(np.asfortranarray(mask_2d.astype(np.uint8)))
    r['counts'] = r['counts'].decode('ascii')
    return r

def read_pfm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        dims   = f.readline().decode().strip().split()
        W, H   = int(dims[0]), int(dims[1])
        scale  = float(f.readline().decode().strip())
        endian = '<' if scale < 0 else '>'
        data   = np.frombuffer(f.read(), dtype=endian + 'f4').reshape(H, W)
        data   = np.flipud(data)
    return data

def generate_main_tree_mask(inst_png_path, main_color_arr, H, W):
    """
    Hasilkan mask pohon utama dengan nearest-neighbor palette expansion
    (algoritma sama dgn repair_coco_tolerant.py).
    Jamin hasilnya ⊇ exact-color pixels.
    """
    raw = np.array(Image.open(inst_png_path).convert('RGB'))
    if raw.shape[:2] != (H, W):
        raw = np.array(Image.fromarray(raw).resize((W, H), Image.NEAREST))

    flat = raw.reshape(-1, 3)
    uimg = np.unique(flat, axis=0)
    present = np.array([c for c in uimg if tuple(int(x) for x in c) in PALSET])
    present_t = [tuple(int(x) for x in c) for c in present]

    main_color_t = tuple(int(x) for x in main_color_arr)

    # Pastikan warna utama ada di present (mungkin tepat exact)
    if main_color_t not in present_t:
        # Warna hadir tapi tidak sepenuhnya dari PALSET? Kembalikan exact match saja
        exact = np.all(flat.reshape(H, W, 3) == main_color_arr, axis=2)
        return exact

    nonwhite = ~np.all(flat == 255, axis=1)
    idx_nw   = np.where(nonwhite)[0]
    newm     = np.zeros(H * W, bool)

    if len(present) > 0 and len(idx_nw) > 0:
        pix = flat[idx_nw].astype(np.int32)
        # Distance ke setiap palette color yg hadir di gambar
        d2  = ((pix[:, None, :] - present.astype(np.int32)[None, :, :]) ** 2).sum(-1)
        nn  = d2.argmin(1)
        # Valid = lebih dekat ke palette color tsb drpd ke putih
        valid = d2.min(1) < ((pix - 255) ** 2).sum(1)
        bidx  = present_t.index(main_color_t)
        sel   = idx_nw[(nn == bidx) & valid]
        newm[sel] = True

    # Jamin exact-color pixels selalu masuk
    exact_1d = np.all(flat == main_color_arr, axis=1)
    newm |= exact_1d

    return newm.reshape(H, W)

def get_depth_stats(depth_pfm_path, mask_2d):
    """Hitung depth_mean/min/max untuk piksel dalam mask."""
    try:
        depth = read_pfm(depth_pfm_path)
        vals  = depth[mask_2d & (depth < 1e6)]
        if len(vals) == 0:
            return 0.0, 0.0, 0.0
        return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))
    except Exception:
        return 0.0, 0.0, 0.0

# ── load all split JSONs ──────────────────────────────────────────────────────
t0 = time.time()
splits_data = {}
for split in ['train', 'val', 'test']:
    splits_data[split] = json.load(open(f'{SRC}/instances_{split}.json'))

# Build fast lookups: stem → (split, image_entry, [annotations])
stem_to_split = {}
stem_to_im    = {}
stem_to_anns  = {}   # image_id → list[ann]  (shared reference)
stem_to_imgid = {}

for split, d in splits_data.items():
    ai = defaultdict(list)
    for a in d['annotations']:
        ai[a['image_id']].append(a)
    for im in d['images']:
        stem = Path(im['file_name']).stem
        stem_to_split[stem] = split
        stem_to_im[stem]    = im
        stem_to_anns[stem]  = ai[im['id']]   # may be empty list []
        stem_to_imgid[stem] = im['id']

# Max annotation ID globally
max_ann_id = max(
    max((a['id'] for a in d['annotations']), default=0)
    for d in splits_data.values()
)
next_ann_id = max_ann_id + 1

# ── process each target ───────────────────────────────────────────────────────
added = 0
already_ok = 0
skipped = 0
report_rows = [['stem', 'scene', 'main_tree', 'raw_px', 'covered_before', 'action', 'new_ann_id', 'new_ann_area']]

for stem in TARGET_LIST:
    if stem not in stem_to_im:
        print(f"  SKIP {stem}: not in any JSON split")
        skipped += 1
        report_rows.append([stem, '?', '?', 0, 0, 'SKIP_no_json', '', ''])
        continue

    im    = stem_to_im[stem]
    H, W  = im['height'], im['width']
    fn    = im['file_name']
    scene = 'plantation' if '/plantations/' in fn else 'rainforest'
    split = stem_to_split[stem]
    anns  = stem_to_anns[stem]

    inst_png = Path(INST_PLANT) / f'{stem}.png'
    inst_txt = Path(INST_PLANT) / f'{stem}.txt'
    if not inst_png.exists() or not inst_txt.exists():
        print(f"  SKIP {stem}: no raw instance files in plantation dir")
        skipped += 1
        report_rows.append([stem, scene, '?', 0, 0, 'SKIP_no_raw', '', ''])
        continue

    # Identify main tree
    with open(inst_txt) as f:
        entries = {e[0]: int(e[1]) for e in [l.strip().split() for l in f if l.strip()]}
    main_tree = stem.rsplit('_', 1)[0]
    if main_tree not in entries:
        print(f"  SKIP {stem}: {main_tree} not in txt")
        skipped += 1
        report_rows.append([stem, scene, main_tree, 0, 0, 'SKIP_no_entry', '', ''])
        continue

    inst_num   = entries[main_tree]
    main_color = PAL[inst_num]  # numpy array

    # Raw main tree mask (exact color)
    raw_img = np.array(Image.open(inst_png).convert('RGB'))
    if raw_img.shape[:2] != (H, W):
        raw_img = np.array(Image.fromarray(raw_img).resize((W, H), Image.NEAREST))
    raw_main_2d = np.all(raw_img == main_color, axis=2)
    raw_px = int(raw_main_2d.sum())

    if raw_px == 0:
        print(f"  SKIP {stem}: main tree not visible (0 raw pixels)")
        skipped += 1
        report_rows.append([stem, scene, main_tree, 0, 0, 'SKIP_invisible', '', ''])
        continue

    # Check existing coverage
    rep_union = np.zeros((H, W), bool)
    for a in anns:
        rep_union |= rle_decode(a['segmentation'], H, W)
    covered_before = int((raw_main_2d & rep_union).sum())

    if covered_before == raw_px:
        already_ok += 1
        report_rows.append([stem, scene, main_tree, raw_px, covered_before, 'ALREADY_OK', '', ''])
        continue

    # ── Need to add annotation ────────────────────────────────────────────────
    print(f"  FIXING {stem}: {covered_before}/{raw_px}px covered → adding main tree ann", flush=True)

    # Generate full mask via nearest-neighbor expansion
    new_mask = generate_main_tree_mask(inst_png, main_color, H, W)

    # Ensure new mask ⊇ raw exact pixels
    new_mask |= raw_main_2d

    new_area = int(new_mask.sum())

    # Compute bbox
    rle_seg = rle_encode(new_mask)
    bbox    = [float(x) for x in maskUtils.toBbox(rle_seg)]

    # Get metadata from obj_info
    info     = OBJ_INFO.get(main_tree, {})
    cat_name = info.get('species', '')
    cat_id   = SPECIES_TO_CAT.get(cat_name, 5)   # fallback Orange=5

    # Depth stats
    pfm_path = Path(DEPTH_DIR) / f'{stem}.pfm'
    depth_mean, depth_min, depth_max = get_depth_stats(str(pfm_path), new_mask)

    new_ann = {
        'id':             next_ann_id,
        'image_id':       stem_to_imgid[stem],
        'category_id':    cat_id,
        'segmentation':   rle_seg,
        'bbox':           bbox,
        'area':           new_area,
        'iscrowd':        0,
        'height_ue':      round(info.get('height_ue', 0.0), 4),
        'dbh':            round(info.get('dbh', 0.0), 4),
        'crown_diameter': round(info.get('crown_diameter', 0.0), 4),
        'depth_mean':     round(depth_mean, 2),
        'depth_min':      round(depth_min, 2),
        'depth_max':      round(depth_max, 2),
    }

    # Append to the split's annotation list
    splits_data[split]['annotations'].append(new_ann)
    # Also update our local anns reference for consistency
    anns.append(new_ann)

    report_rows.append([stem, scene, main_tree, raw_px, covered_before,
                        'ADDED', next_ann_id, new_area])
    next_ann_id += 1
    added += 1

# ── save updated JSONs ────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_DIR} ...", flush=True)
for split, d in splits_data.items():
    out_path = f'{OUTPUT_DIR}/instances_{split}.json'
    json.dump(d, open(out_path, 'w'))
    n_ann = len(d['annotations'])
    n_img = len(d['images'])
    print(f"  {split}: {n_img} images, {n_ann} annotations → {out_path}")

# ── save report ───────────────────────────────────────────────────────────────
report_path = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/scripts/fix_main_tree_report.csv'
import csv as csv_mod
with open(report_path, 'w', newline='') as f:
    csv_mod.writer(f).writerows(report_rows)

print(f"\n{'='*60}")
print(f"SELESAI ({time.time()-t0:.1f}s)")
print(f"  Already OK  : {already_ok}")
print(f"  Added       : {added}")
print(f"  Skipped     : {skipped}")
print(f"  Report      : {report_path}")
print(f"  Output      : {OUTPUT_DIR}")
