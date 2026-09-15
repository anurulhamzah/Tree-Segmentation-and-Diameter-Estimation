#!/usr/bin/env python3
"""Visualisasi 13 gambar unchanged: instance PNG + annotation v2 vs v4."""
import numpy as np, json, csv, base64, io
from pathlib import Path
from PIL import Image
from pycocotools import mask as maskUtils
from collections import defaultdict

PAL_CSV  = '/home/anur0018/pr65_scratch2/anur0018/raw_data/plantations/color_palette_Part I.csv'
INST_DIR = '/home/anur0018/pr65_scratch2/anur0018/raw_data/plantations/instance_segmentation'
RGB_DIR  = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/plantations/rgb_resized'
ANN_V2   = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v2'
ANN_V4   = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v4'
OUT_HTML = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/unchanged_13.html'

STEMS = [
    'Tree655_1721959566','Tree685_1721051812','Tree649_1721051800',
    'Tree665_1721030270','Tree650_1721023448','Tree683_1721053678',
    'Tree672_1721047447','Tree674_1721028359','Tree686_1721959228',
    'Tree562_1721963529','Tree630_1721037428','Tree89_1721963027',
    'Tree906_1721022829',
]

PAL    = np.array([(int(r['R']),int(r['G']),int(r['B']))
                   for r in csv.DictReader(open(PAL_CSV))])
PALSET = set(tuple(int(x) for x in c) for c in PAL)
MAIN   = PAL[1]

def img_to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def load_all_anns(ann_dir):
    stem2id, id2anns = {}, defaultdict(list)
    for split in ['train','val','test']:
        d = json.load(open(f'{ann_dir}/instances_{split}.json'))
        for im in d['images']:
            stem2id[Path(im['file_name']).stem] = im['id']
        for a in d['annotations']:
            id2anns[a['image_id']].append(a)
    return stem2id, id2anns

def find_main_ann(anns, inst_flat, H, W):
    main_t = tuple(int(x) for x in MAIN)
    best, best_area = None, -1
    for a in anns:
        seg = a['segmentation']
        m = maskUtils.decode(seg if isinstance(seg, dict)
                             else maskUtils.merge(maskUtils.frPyObjects(seg, H, W))).astype(bool)
        sub = inst_flat[m.reshape(-1)]
        if not len(sub): continue
        uv, uc = np.unique(sub, axis=0, return_counts=True)
        for j in np.argsort(-uc):
            c = tuple(int(x) for x in uv[j])
            if c in PALSET and c != (255,255,255):
                if c == main_t and a['area'] > best_area:
                    best, best_area = (a, m), a['area']
                break
    return best

def make_overlay(rgb, main_m, other_ms):
    ov = rgb.copy().astype(np.float32)
    for m in other_ms:
        ov[m] = ov[m]*0.55 + np.array([255,140,0])*0.45
    if main_m is not None:
        ov[main_m] = ov[main_m]*0.25 + np.array([0,230,0])*0.75
    return ov.clip(0,255).astype(np.uint8)

def check_blended(flat):
    nonwhite = flat[~np.all(flat==255,axis=1)]
    non_pal  = sum(1 for c in np.unique(nonwhite,axis=0)
                   if tuple(int(x) for x in c) not in PALSET)
    return non_pal

v2_stem2id, v2_id2anns = load_all_anns(ANN_V2)
v4_stem2id, v4_id2anns = load_all_anns(ANN_V4)

cards = []
for stem in STEMS:
    inst_path = Path(INST_DIR)/f'{stem}.png'
    rgb_path  = Path(RGB_DIR)/f'{stem}.png'
    if not inst_path.exists() or not rgb_path.exists(): continue
    if stem not in v2_stem2id: continue

    inst = np.array(Image.open(inst_path).convert('RGB'))
    H, W = inst.shape[:2]
    rgb  = np.array(Image.open(rgb_path).convert('RGB'))
    if rgb.shape[:2] != (H,W):
        rgb = np.array(Image.fromarray(rgb).resize((W,H), Image.LANCZOS))
    flat = inst.reshape(-1,3)

    pal1_exact = np.all(flat==MAIN, axis=1).sum()
    white_pct  = np.all(flat==255, axis=1).mean()*100
    blended    = check_blended(flat)

    # PNG lama atau baru?
    is_new_png = blended == 0
    png_status = 'FLAT COLOR (baru)' if is_new_png else f'BLENDED ({blended} non-pal colors, lama)'

    v2_anns = v2_id2anns[v2_stem2id[stem]]
    v4_anns = v4_id2anns[v4_stem2id[stem]]
    v2_main = find_main_ann(v2_anns, flat, H, W)
    v4_main = find_main_ann(v4_anns, flat, H, W)

    v2_m    = v2_main[1] if v2_main else None
    v4_m    = v4_main[1] if v4_main else None
    v2_area = v2_main[0]['area'] if v2_main else 0
    v4_area = v4_main[0]['area'] if v4_main else 0

    def omasks(anns, main):
        return [maskUtils.decode(a['segmentation'] if isinstance(a['segmentation'],dict)
                else maskUtils.merge(maskUtils.frPyObjects(a['segmentation'],H,W))).astype(bool)
                for a in anns if not (main and a['id']==main[0]['id'])]

    p_inst = img_to_b64(inst)
    p_rgb  = img_to_b64(rgb)
    p_v2   = img_to_b64(make_overlay(rgb, v2_m, omasks(v2_anns, v2_main)))
    p_v4   = img_to_b64(make_overlay(rgb, v4_m, omasks(v4_anns, v4_main)))

    status_col = '#6f6' if is_new_png else '#f88'
    cards.append(f"""
<div class="card">
  <div class="title">
    <b>{stem}</b> &nbsp;|&nbsp;
    PAL[1] exact: <b>{pal1_exact:,}px</b> &nbsp;|&nbsp;
    BG: {white_pct:.0f}% &nbsp;|&nbsp;
    PNG: <span style="color:{status_col}">{png_status}</span>
    &nbsp;|&nbsp; V2={v2_area:,}px → V4={v4_area:,}px
  </div>
  <div class="panels">
    <div class="panel"><img src="data:image/png;base64,{p_inst}">
      <div class="lbl">Instance PNG</div></div>
    <div class="panel"><img src="data:image/png;base64,{p_rgb}">
      <div class="lbl">RGB</div></div>
    <div class="panel"><img src="data:image/png;base64,{p_v2}">
      <div class="lbl">V2 ({v2_area:,}px)</div></div>
    <div class="panel"><img src="data:image/png;base64,{p_v4}">
      <div class="lbl">V4 ({v4_area:,}px)</div></div>
  </div>
</div>""")
    print(f'{stem}: PAL[1]={pal1_exact}px, blended={blended}, v2={v2_area}→v4={v4_area} | {png_status}')

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>13 Unchanged Images</title>
<style>
body{{background:#111;color:#eee;font-family:sans-serif;padding:10px}}
.card{{background:#1e1e1e;border-radius:8px;margin-bottom:16px;padding:10px}}
.title{{font-size:12px;margin-bottom:7px;color:#ffd}}
.panels{{display:flex;gap:6px}}
.panel{{flex:1}}.panel img{{width:100%;border-radius:4px;display:block}}
.lbl{{font-size:10px;color:#aaa;text-align:center;margin-top:3px}}
</style></head><body>
<h2 style="color:#ffd">13 Gambar Unchanged — Diagnosa PNG Lama vs Baru</h2>
{''.join(cards)}
</body></html>"""

open(OUT_HTML,'w').write(html)
print(f'\nSaved: {OUT_HTML} ({Path(OUT_HTML).stat().st_size/1e6:.1f} MB)')
