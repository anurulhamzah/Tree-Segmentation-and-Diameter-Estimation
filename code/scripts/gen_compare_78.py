#!/usr/bin/env python3
"""HTML comparison V2 vs V3 for 81 target images (main tree annotation)."""
import numpy as np, json, csv, base64, io
from pathlib import Path
from PIL import Image
from pycocotools import mask as maskUtils
from collections import defaultdict

PAL_CSV  = '/home/anur0018/pr65_scratch2/anur0018/raw_data/plantations/color_palette_Part I.csv'
INST_DIR = '/home/anur0018/pr65_scratch2/anur0018/raw_data/plantations/instance_segmentation'
RGB_DIR  = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/plantations/rgb_resized'
ANN_V2   = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v2'
ANN_V3   = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v3'
OUT_HTML = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/compare_78_v2_vs_v3.html'

STEMS = list(dict.fromkeys([
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
]))

PAL    = np.array([(int(r['R']), int(r['G']), int(r['B']))
                   for r in csv.DictReader(open(PAL_CSV))])
PALSET = set(tuple(int(x) for x in c) for c in PAL)
MAIN_COLOR = PAL[1]  # pohon utama selalu PAL[1]

def img_to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def load_all_anns(ann_dir):
    stem2id, id2anns = {}, defaultdict(list)
    for split in ['train', 'val', 'test']:
        d = json.load(open(f'{ann_dir}/instances_{split}.json'))
        for im in d['images']:
            stem2id[Path(im['file_name']).stem] = im['id']
        for a in d['annotations']:
            id2anns[a['image_id']].append(a)
    return stem2id, id2anns

def find_main_ann(anns, inst_flat, H, W):
    """Cari anotasi pohon utama: anchor color = PAL[1]."""
    best, best_area = None, -1
    main_t = tuple(int(x) for x in MAIN_COLOR)
    for a in anns:
        seg = a['segmentation']
        m = maskUtils.decode(seg if isinstance(seg, dict)
                             else maskUtils.merge(maskUtils.frPyObjects(seg, H, W))).astype(bool)
        sub = inst_flat[m.reshape(-1)]
        if len(sub) == 0:
            continue
        uv, uc = np.unique(sub, axis=0, return_counts=True)
        for j in np.argsort(-uc):
            c = tuple(int(x) for x in uv[j])
            if c in PALSET and c != (255, 255, 255):
                if c == main_t and a['area'] > best_area:
                    best = (a, m)
                    best_area = a['area']
                break
    return best

def make_overlay(rgb, main_m, other_ms):
    ov = rgb.copy().astype(np.float32)
    for m in other_ms:
        ov[m] = ov[m] * 0.55 + np.array([255, 140, 0]) * 0.45
    if main_m is not None:
        ov[main_m] = ov[main_m] * 0.25 + np.array([0, 230, 0]) * 0.75
    return ov.clip(0, 255).astype(np.uint8)

print('Loading annotations...')
v2_stem2id, v2_id2anns = load_all_anns(ANN_V2)
v3_stem2id, v3_id2anns = load_all_anns(ANN_V3)
print(f'V2: {sum(len(v) for v in v2_id2anns.values())} anns | '
      f'V3: {sum(len(v) for v in v3_id2anns.values())} anns')

cards = []
missing = []
for stem in STEMS:
    inst_path = Path(INST_DIR) / f'{stem}.png'
    rgb_path  = Path(RGB_DIR)  / f'{stem}.png'

    if not inst_path.exists() or not rgb_path.exists():
        missing.append(stem)
        continue
    if stem not in v2_stem2id:
        missing.append(f'{stem} (not in COCO)')
        continue

    inst = np.array(Image.open(inst_path).convert('RGB'))
    H, W = inst.shape[:2]
    rgb  = np.array(Image.open(rgb_path).convert('RGB'))
    if rgb.shape[:2] != (H, W):
        rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.LANCZOS))
    inst_flat = inst.reshape(-1, 3)

    v2_anns = v2_id2anns[v2_stem2id[stem]]
    v3_anns = v3_id2anns[v3_stem2id[stem]]

    v2_main = find_main_ann(v2_anns, inst_flat, H, W)
    v3_main = find_main_ann(v3_anns, inst_flat, H, W)

    v2_main_m = v2_main[1] if v2_main else None
    v3_main_m = v3_main[1] if v3_main else None
    v2_area   = v2_main[0]['area'] if v2_main else 0
    v3_area   = v3_main[0]['area'] if v3_main else 0
    gain      = v3_area - v2_area
    ratio     = v3_area / max(v2_area, 1)

    # Other masks for context
    def other_masks(anns, main_a):
        ms = []
        for a in anns:
            if main_a and a['id'] == main_a[0]['id']:
                continue
            seg = a['segmentation']
            m = maskUtils.decode(seg if isinstance(seg, dict)
                                 else maskUtils.merge(maskUtils.frPyObjects(seg, H, W))).astype(bool)
            ms.append(m)
        return ms

    p_inst = img_to_b64(inst)
    p_v2   = img_to_b64(make_overlay(rgb, v2_main_m, other_masks(v2_anns, v2_main)))
    p_v3   = img_to_b64(make_overlay(rgb, v3_main_m, other_masks(v3_anns, v3_main)))

    gain_col = '#6f6' if gain > 500 else ('#ff6' if gain > 0 else '#f66')
    stem_base = stem.split('_')[0]
    cards.append(f"""
<div class="card">
  <div class="title">
    <b>{stem}</b> &nbsp;|&nbsp;
    V2: {v2_area:,}px &nbsp;→&nbsp; V3: {v3_area:,}px
    <span style="color:{gain_col}"> ({'+' if gain>=0 else ''}{gain:,}px &nbsp; {ratio:.2f}x)</span>
  </div>
  <div class="panels">
    <div class="panel"><img src="data:image/png;base64,{p_inst}">
      <div class="lbl">Instance Segmentation PNG</div></div>
    <div class="panel"><img src="data:image/png;base64,{p_v2}">
      <div class="lbl">V2 — NN ({v2_area:,}px)</div></div>
    <div class="panel"><img src="data:image/png;base64,{p_v3}">
      <div class="lbl">V3 — Threshold ({v3_area:,}px)</div></div>
  </div>
</div>""")
    print(f'{stem}: {v2_area:,} → {v3_area:,} ({gain:+,}px, {ratio:.2f}x)')

if missing:
    print(f'\nMissing/skip: {missing}')

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>V2 vs V3 — 81 Target Images</title>
<style>
body{{background:#111;color:#eee;font-family:sans-serif;padding:10px;margin:0}}
h2{{color:#ffd;margin:10px 0 16px}}
.card{{background:#1e1e1e;border-radius:8px;margin-bottom:16px;padding:10px}}
.title{{font-size:12px;margin-bottom:7px;color:#ffd}}
.panels{{display:flex;gap:6px}}
.panel{{flex:1}}.panel img{{width:100%;border-radius:4px;display:block}}
.lbl{{font-size:10px;color:#aaa;text-align:center;margin-top:3px}}
</style></head><body>
<h2>Perbandingan V2 (NN) vs V3 (Distance Threshold) — {len(cards)} gambar
<br><small style="font-size:12px;color:#aaa">Hijau = pohon utama | Orange = pohon lain</small></h2>
{''.join(cards)}
</body></html>"""

open(OUT_HTML, 'w').write(html)
sz = Path(OUT_HTML).stat().st_size / 1e6
print(f'\nSaved: {OUT_HTML} ({sz:.1f} MB) | {len(cards)} cards | {len(missing)} skipped')
