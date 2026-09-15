#!/usr/bin/env python3
"""Compare v2 vs v3 annotation for sample images."""
import numpy as np, json, csv, base64, io
from pathlib import Path
from PIL import Image
from pycocotools import mask as maskUtils

PAL_CSV  = '/home/anur0018/pr65_scratch2/anur0018/raw_data/plantations/color_palette_Part I.csv'
INST_DIR = '/home/anur0018/pr65_scratch2/anur0018/raw_data/plantations/instance_segmentation'
RGB_DIR  = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/plantations/rgb_resized'
ANN_V2   = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v2'
ANN_V3   = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v3'
OUT_HTML = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/verify_v2_vs_v3.html'

PAL    = np.array([(int(r['R']), int(r['G']), int(r['B']))
                   for r in csv.DictReader(open(PAL_CSV))])
PALSET = set(tuple(int(x) for x in c) for c in PAL)

STEMS = ['Tree534_1721029764', 'Tree675_1721963798',
         'Tree646_1721022039', 'Tree617_1721040057',
         'Tree556_1721023401', 'Tree611_1721041923']

def img_to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def load_anns(ann_dir):
    data = {'images': [], 'annotations': []}
    for split in ['train', 'val', 'test']:
        d = json.load(open(f'{ann_dir}/instances_{split}.json'))
        data['images'] += d['images']
        data['annotations'] += d['annotations']
    # index by stem
    stem2id = {Path(im['file_name']).stem: im['id'] for im in data['images']}
    id2anns = {}
    for a in data['annotations']:
        id2anns.setdefault(a['image_id'], []).append(a)
    return stem2id, id2anns

def make_overlay(rgb, H, W, stem, stem2id, id2anns, main_color):
    if stem not in stem2id:
        return rgb
    img_id = stem2id[stem]
    anns = id2anns.get(img_id, [])
    overlay = rgb.copy().astype(np.float32)
    main_t = tuple(int(x) for x in main_color)

    for a in anns:
        seg = a['segmentation']
        m = maskUtils.decode(seg if isinstance(seg, dict)
                             else maskUtils.merge(maskUtils.frPyObjects(seg, H, W))).astype(bool)
        # Determine if this is the main tree annotation
        # Sample instance PNG at mask pixels to find its anchor color
        is_main = a.get('is_main_tree', False)
        # Fallback: check if annotation area is largest (main tree)
        if is_main:
            overlay[m] = overlay[m] * 0.3 + np.array([0, 230, 0]) * 0.7
        else:
            overlay[m] = overlay[m] * 0.5 + np.array([255, 140, 0]) * 0.5

    return overlay.clip(0, 255).astype(np.uint8)

def make_overlay2(rgb, H, W, stem, stem2id, id2anns, inst_flat, main_color):
    """Overlay coloring main tree green based on anchor color match."""
    if stem not in stem2id:
        return rgb
    img_id = stem2id[stem]
    anns = id2anns.get(img_id, [])
    overlay = rgb.copy().astype(np.float32)
    main_t = tuple(int(x) for x in main_color)

    for a in anns:
        seg = a['segmentation']
        m = maskUtils.decode(seg if isinstance(seg, dict)
                             else maskUtils.merge(maskUtils.frPyObjects(seg, H, W))).astype(bool)
        flat_sub = inst_flat[m.reshape(-1)]
        # Find anchor: dominant exact palette color in mask
        uv, uc = np.unique(flat_sub, axis=0, return_counts=True) if len(flat_sub) > 0 else (np.array([]), np.array([]))
        anchor_t = None
        for j in (np.argsort(-uc) if len(uc) > 0 else []):
            c = tuple(int(x) for x in uv[j])
            if c in PALSET and c != (255, 255, 255):
                anchor_t = c
                break

        if anchor_t == main_t:
            overlay[m] = overlay[m] * 0.3 + np.array([0, 230, 0]) * 0.7
        else:
            overlay[m] = overlay[m] * 0.5 + np.array([255, 140, 0]) * 0.5

    return overlay.clip(0, 255).astype(np.uint8)

v2_stem2id, v2_id2anns = load_anns(ANN_V2)
v3_stem2id, v3_id2anns = load_anns(ANN_V3)

cards = []
for stem in STEMS:
    inst_path = Path(INST_DIR) / f'{stem}.png'
    rgb_path  = Path(RGB_DIR)  / f'{stem}.png'
    if not inst_path.exists() or not rgb_path.exists():
        continue

    inst = np.array(Image.open(inst_path).convert('RGB'))
    H, W = inst.shape[:2]
    rgb  = np.array(Image.open(rgb_path).convert('RGB'))
    if rgb.shape[:2] != (H, W):
        rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.LANCZOS))

    inst_flat = inst.reshape(-1, 3)
    main_color = PAL[1]  # pohon utama = PAL[1]

    # Stats
    exact_ct = np.all(inst_flat == main_color, axis=1).sum()
    v2_anns = v2_id2anns.get(v2_stem2id.get(stem), [])
    v3_anns = v3_id2anns.get(v3_stem2id.get(stem), [])

    # Find main tree annotation area in v2 and v3
    def main_ann_area(anns):
        best_area, best_m = 0, None
        for a in anns:
            seg = a['segmentation']
            m = maskUtils.decode(seg if isinstance(seg, dict)
                                 else maskUtils.merge(maskUtils.frPyObjects(seg, H, W))).astype(bool)
            flat_sub = inst_flat[m.reshape(-1)]
            uv, uc = np.unique(flat_sub, axis=0, return_counts=True) if len(flat_sub) > 0 else ([], [])
            for j in (np.argsort(-uc) if len(uc) > 0 else []):
                c = tuple(int(x) for x in uv[j])
                if c in PALSET and c != (255, 255, 255):
                    if c == tuple(int(x) for x in main_color):
                        if a['area'] > best_area:
                            best_area = a['area']
                            best_m = m
                    break
        return best_area, best_m

    v2_area, v2_main_m = main_ann_area(v2_anns)
    v3_area, v3_main_m = main_ann_area(v3_anns)

    # Make overlays for main tree only
    def make_main_overlay(rgb, main_m):
        ov = rgb.copy().astype(np.float32)
        if main_m is not None:
            ov[main_m] = ov[main_m] * 0.3 + np.array([0, 230, 0]) * 0.7
        return ov.clip(0, 255).astype(np.uint8)

    p1 = img_to_b64(rgb)
    p2 = img_to_b64(inst)
    p3 = img_to_b64(make_main_overlay(rgb, v2_main_m))
    p4 = img_to_b64(make_main_overlay(rgb, v3_main_m))

    stem_base = stem.split('_')[0]
    gain = v3_area - v2_area
    cards.append(f"""
<div class="card">
  <div class="title"><b>{stem}</b> ({stem_base}) &nbsp;|&nbsp;
    Exact PAL[1]: {exact_ct}px &nbsp;|&nbsp;
    V2 main tree: <b>{v2_area:,}px</b> &nbsp;→&nbsp;
    V3 main tree: <b>{v3_area:,}px</b>
    <span style="color:{'#6f6' if gain>0 else '#f66'}"> ({'+' if gain>=0 else ''}{gain:,}px)</span>
  </div>
  <div class="panels">
    <div class="panel"><img src="data:image/png;base64,{p1}"><div class="lbl">1. RGB</div></div>
    <div class="panel"><img src="data:image/png;base64,{p2}"><div class="lbl">2. Instance PNG</div></div>
    <div class="panel"><img src="data:image/png;base64,{p3}"><div class="lbl">3. Annotation V2 (NN)</div></div>
    <div class="panel"><img src="data:image/png;base64,{p4}"><div class="lbl">4. Annotation V3 (threshold)</div></div>
  </div>
</div>""")
    print(f'{stem}: exact={exact_ct}, v2={v2_area:,}, v3={v3_area:,} (Δ{gain:+,})')

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>V2 vs V3 Annotation</title>
<style>
body{{background:#111;color:#eee;font-family:sans-serif;padding:10px}}
.card{{background:#1e1e1e;border-radius:8px;margin-bottom:20px;padding:12px}}
.title{{font-size:13px;margin-bottom:8px;color:#ffd}}
.panels{{display:flex;gap:8px}}
.panel{{flex:1}}.panel img{{width:100%;border-radius:4px}}
.lbl{{font-size:11px;color:#aaa;text-align:center;margin-top:4px}}
</style></head><body>
<h2 style="color:#ffd">V2 (NN) vs V3 (Distance Threshold) — Main Tree Annotation</h2>
{''.join(cards)}
</body></html>"""

open(OUT_HTML, 'w').write(html)
print(f'\nSaved: {OUT_HTML} ({Path(OUT_HTML).stat().st_size/1e6:.1f} MB)')
