#!/usr/bin/env python3
"""Generate instance_verify_v2.html - Panel 3 shows ALL pixels assigned to main tree
(exact + nearest-neighbor), not just exact palette matches."""

import numpy as np, json, csv, base64, io
from pathlib import Path
from PIL import Image
from pycocotools import mask as maskUtils

PAL_CSV  = '/home/anur0018/pr65_scratch2/anur0018/raw_data/plantations/color_palette_Part I.csv'
INST_DIR = '/home/anur0018/pr65_scratch2/anur0018/raw_data/plantations/instance_segmentation'
RGB_DIR  = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/plantations/rgb_resized'
ANN_V2   = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v2'
OUT_HTML = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/instance_verify_v2.html'

PAL = np.array([(int(r['R']), int(r['G']), int(r['B']))
                for r in csv.DictReader(open(PAL_CSV))])
PALSET = set(tuple(int(x) for x in c) for c in PAL)

STEMS = ['Tree534_1721029764', 'Tree675_1721963798',
         'Tree646_1721022039', 'Tree617_1721040057',
         'Tree556_1721023401', 'Tree611_1721041923']

def img_to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def read_txt_main_color(stem):
    """Tree with index=1 in txt file = pohon utama → PAL[1]."""
    txt = Path(INST_DIR) / f'{stem}.txt'
    if not txt.exists():
        return None
    for line in txt.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and int(parts[1]) == 1:
            return PAL[1]  # index 1 = pohon utama = PAL[1]
    return None

def get_nn_mask(flat, H, W, main_color):
    """Returns (exact_mask, nn_mask, other_mask, bg_mask) all shape (H*W,)."""
    uimg = np.unique(flat, axis=0)
    present = np.array([c for c in uimg if tuple(int(x) for x in c) in PALSET])
    present_t = [tuple(int(x) for x in c) for c in present]
    main_t = tuple(int(x) for x in main_color)

    bg_mask = np.all(flat == 255, axis=1)
    nonwhite = ~bg_mask
    idx_nw = np.where(nonwhite)[0]

    exact_mask = np.all(flat == main_color, axis=1)
    nn_mask = np.zeros(H * W, bool)
    other_mask = np.zeros(H * W, bool)

    if len(present) > 0 and len(idx_nw) > 0 and main_t in present_t:
        pix = flat[idx_nw].astype(np.int32)
        d2 = ((pix[:, None, :] - present.astype(np.int32)[None, :, :]) ** 2).sum(-1)
        nn = d2.argmin(1)
        valid = d2.min(1) < ((pix - 255) ** 2).sum(1)
        bidx = present_t.index(main_t)
        sel = idx_nw[(nn == bidx) & valid & ~exact_mask[idx_nw]]
        nn_mask[sel] = True
        other_sel = idx_nw[valid & (nn != bidx)]
        other_mask[other_sel] = True

    return exact_mask, nn_mask, other_mask, bg_mask

def make_panel3(inst_raw, H, W, main_color):
    """Highlight: bright green=exact, lime=nn-blended, dark=other trees, gray=background."""
    flat = inst_raw.reshape(-1, 3)
    exact_mask, nn_mask, other_mask, bg_mask = get_nn_mask(flat, H, W, main_color)

    vis = np.zeros((H * W, 3), dtype=np.uint8)
    # background → light gray
    vis[bg_mask] = [200, 200, 200]
    # other trees → dim brownish
    vis[other_mask] = (flat[other_mask] * 0.35).astype(np.uint8)
    # nn blended pixels → lime/yellow-green
    vis[nn_mask] = [180, 255, 80]
    # exact palette pixels → bright green
    vis[exact_mask] = [0, 220, 0]

    return vis.reshape(H, W, 3)

def make_panel4(rgb, H, W, stem, ann_data):
    """Annotation overlay: green=main tree ann, orange=others."""
    overlay = rgb.copy().astype(np.float32)
    stem_base = stem.split('_')[0]
    for im_info in ann_data['images']:
        if Path(im_info['file_name']).stem == stem:
            img_id = im_info['id']
            break
    else:
        return rgb

    for a in ann_data['annotations']:
        if a['image_id'] != img_id:
            continue
        m = maskUtils.decode(a['segmentation'] if isinstance(a['segmentation'], dict)
                             else maskUtils.merge(maskUtils.frPyObjects(a['segmentation'], H, W)))
        m = m.astype(bool)
        if a.get('main_tree', False) or \
           (stem_base in str(a.get('extra', {}).get('tree_name', ''))):
            overlay[m] = overlay[m] * 0.4 + np.array([0, 200, 0]) * 0.6
        else:
            overlay[m] = overlay[m] * 0.5 + np.array([255, 140, 0]) * 0.5

    return overlay.clip(0, 255).astype(np.uint8)

# Load annotations
ann_data = {}
for split in ['train', 'val', 'test']:
    d = json.load(open(f'{ANN_V2}/instances_{split}.json'))
    if not ann_data:
        ann_data = d
    else:
        ann_data['images'] += d['images']
        ann_data['annotations'] += d['annotations']

cards_html = []
for stem in STEMS:
    inst_path = Path(INST_DIR) / f'{stem}.png'
    rgb_path  = Path(RGB_DIR) / f'{stem}.png'
    if not inst_path.exists() or not rgb_path.exists():
        print(f'SKIP {stem}: missing files')
        continue

    inst_raw = np.array(Image.open(inst_path).convert('RGB'))
    H, W = inst_raw.shape[:2]
    rgb = np.array(Image.open(rgb_path).convert('RGB'))
    if rgb.shape[:2] != (H, W):
        rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.LANCZOS))

    main_color = read_txt_main_color(stem)
    if main_color is None:
        print(f'SKIP {stem}: pohon utama not found in txt')
        continue

    flat = inst_raw.reshape(-1, 3)
    bg_pct = np.all(flat == 255, axis=1).mean() * 100
    exact_ct = np.all(flat == main_color, axis=1).sum()

    exact_mask, nn_mask, other_mask, bg_mask = get_nn_mask(flat, H, W, main_color)
    nn_total = exact_mask.sum() + nn_mask.sum()

    # Panel 1: RGB
    p1 = img_to_b64(rgb)
    # Panel 2: Raw instance PNG
    p2 = img_to_b64(inst_raw)
    # Panel 3: Highlight (improved)
    p3 = img_to_b64(make_panel3(inst_raw, H, W, main_color))
    # Panel 4: Annotation overlay
    p4 = img_to_b64(make_panel4(rgb, H, W, stem, ann_data))

    stem_base = stem.split('_')[0]
    cards_html.append(f"""
<div class="card">
  <div class="card-title">
    <b>{stem}</b> ({stem_base}) &nbsp;|&nbsp;
    Main color PAL[1]={tuple(int(x) for x in main_color)} &nbsp;|&nbsp;
    Exact px: <b>{exact_ct}</b> &nbsp;|&nbsp;
    NN-expanded px: <b>{nn_total}</b> &nbsp;|&nbsp;
    BG: {bg_pct:.0f}%
  </div>
  <div class="panels">
    <div class="panel">
      <img src="data:image/png;base64,{p1}">
      <div class="label">1. RGB Original</div>
    </div>
    <div class="panel">
      <img src="data:image/png;base64,{p2}">
      <div class="label">2. Instance PNG (raw)</div>
    </div>
    <div class="panel">
      <img src="data:image/png;base64,{p3}">
      <div class="label">3. Highlight: <span style="color:#00dc00">■ Exact PAL[1]</span>
        <span style="color:#b4ff50">■ NN-blended</span>
        <span style="color:#888">■ BG</span>
        <span style="color:#555">■ Other trees</span></div>
    </div>
    <div class="panel">
      <img src="data:image/png;base64,{p4}">
      <div class="label">4. Annotation repaired_v2</div>
    </div>
  </div>
</div>""")
    print(f'{stem}: exact={exact_ct}px, nn_total={nn_total}px, bg={bg_pct:.0f}%')

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Instance Verify v2</title>
<style>
body {{ background:#111; color:#eee; font-family:sans-serif; margin:0; padding:10px; }}
.card {{ background:#1e1e1e; border-radius:8px; margin-bottom:20px; padding:12px; }}
.card-title {{ font-size:13px; margin-bottom:8px; color:#ffd; }}
.panels {{ display:flex; gap:8px; }}
.panel {{ flex:1; }}
.panel img {{ width:100%; border-radius:4px; display:block; }}
.label {{ font-size:11px; color:#aaa; margin-top:4px; text-align:center; }}
</style></head><body>
<h2 style="color:#ffd">Instance Verify v2 — Panel 3: Exact (hijau) + NN-blended (lime) + BG (abu) + Other (gelap)</h2>
{''.join(cards_html)}
</body></html>"""

open(OUT_HTML, 'w').write(html)
print(f'\nSaved: {OUT_HTML} ({Path(OUT_HTML).stat().st_size/1e6:.1f} MB)')
