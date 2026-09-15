#!/usr/bin/env python3
"""
Update comparison_swinb_orig_vs_v4.html: re-render prediction panels
dengan warna TP/FP yang jelas.

Panel per gambar setelah update:
  ① RGB asli (unchanged)
  ② GT repaired-v4 — outline putih/cyan (unchanged semantics)
  ③ SwinB-orig predictions — HIJAU=TP, ORANYE=FP
  ④ SwinB-V4 predictions  — HIJAU=TP, ORANYE=FP

Matching: greedy IoU >= 0.5, score >= 0.35.
"""

import base64, io, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
from scipy.ndimage import binary_erosion

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = '/scratch2/pr65/anur0018'

GT = {
    'orig_val':  f'{BASE}/tree_classification/data/combined/annotation_inst/filtered_rle_f1000/instances_val.json',
    'orig_test': f'{BASE}/tree_classification/data/combined/annotation_inst/filtered_rle_f1000/instances_test.json',
    'v4_val':    f'{BASE}/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_val.json',
    'v4_test':   f'{BASE}/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json',
}

PRED = {
    'orig_val':  f'{BASE}/maskdino_output/SwinB_combined_rle_f1000_75k/eval_val_orig/inference/coco_instances_results.json',
    'orig_test': f'{BASE}/maskdino_output/SwinB_combined_rle_f1000_75k/eval_test_orig/inference/coco_instances_results.json',
    'v4_val':    f'{BASE}/maskdino_output/SwinB_combined_rle_f1000_repaired_v4_stratified_75k/eval_val_v4/inference/coco_instances_results.json',
    'v4_test':   f'{BASE}/maskdino_output/SwinB_combined_rle_f1000_repaired_v4_stratified_75k/eval_test_v4/inference/coco_instances_results.json',
}

_REPORT_DIR = (f'{BASE.replace("/scratch2/pr65", "/home/anur0018/pr65_scratch2")}'
               '/tree_classification/reports/stratified_v4')
HTML_IN  = f'{_REPORT_DIR}/comparison_swinb_orig_vs_v4_backup.html'  # baca dari backup
HTML_OUT = f'{_REPORT_DIR}/Report_Juli_2026.html'                      # tulis ke file utama

SCORE_THR = 0.35
IOU_THR   = 0.50

# Warna overlay (R, G, B float32)
COLOR_TP  = np.array([50,  220,  80], dtype=np.float32)  # hijau cerah
COLOR_FP  = np.array([255, 140,   0], dtype=np.float32)  # oranye
COLOR_FN  = np.array([220,  30,  30], dtype=np.float32)  # merah — GT yang terlewat
COLOR_GT  = np.array([80,  200, 255], dtype=np.float32)  # cyan muda untuk GT

ALPHA_FILL   = 0.40
ALPHA_BORDER = 0.90
BORDER_PX    = 2

# ─── Load semua data ke memori ────────────────────────────────────────────────

print('Loading GT dan predictions...', flush=True)

# stem → {img_info, anns}
gt_index = {}   # key → {stem: {...}}
for key, path in GT.items():
    d = json.load(open(path))
    idx = {}
    ann_by_id = defaultdict(list)
    for ann in d['annotations']:
        ann_by_id[ann['image_id']].append(ann)
    for im in d['images']:
        stem = Path(im['file_name']).stem
        idx[stem] = dict(img_info=im, anns=ann_by_id[im['id']])
    gt_index[key] = idx
    print(f'  GT [{key}]: {len(idx)} images', flush=True)

# stem → [preds filtered by score]
pred_index = {}
for key, path in PRED.items():
    # Need image_id → stem mapping from corresponding GT
    gt_key = key  # same key
    stem2id = {stem: v['img_info']['id'] for stem, v in gt_index[gt_key].items()}
    id2stem = {v: k for k, v in stem2id.items()}

    preds = json.load(open(path))
    by_stem = defaultdict(list)
    for p in preds:
        if p['score'] < SCORE_THR:
            continue
        stem = id2stem.get(p['image_id'])
        if stem:
            by_stem[stem].append(p)
    pred_index[key] = dict(by_stem)
    n_preds = sum(len(v) for v in by_stem.values())
    print(f'  Pred [{key}]: {n_preds} predictions (score>={SCORE_THR})', flush=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def decode_mask(seg, h, w):
    if isinstance(seg, dict):
        return maskUtils.decode(seg).astype(bool)
    rles = maskUtils.frPyObjects(seg, h, w)
    return maskUtils.decode(maskUtils.merge(rles)).astype(bool)


def draw_mask(canvas, mask, color, alpha_fill=ALPHA_FILL, alpha_border=ALPHA_BORDER):
    if not mask.any():
        return
    canvas[mask] = canvas[mask] * (1 - alpha_fill) + color * alpha_fill
    inner  = binary_erosion(mask, iterations=BORDER_PX)
    border = mask & ~inner
    if border.any():
        canvas[border] = canvas[border] * (1 - alpha_border) + color * alpha_border


def match_tp_fp(gt_anns, preds, h, w):
    """Greedy IoU match. Returns pr_is_tp[i], gt_caught[i]."""
    n_gt = len(gt_anns)
    n_pr = len(preds)
    if n_gt == 0 or n_pr == 0:
        return [False] * n_pr, [False] * n_gt

    gt_masks = [decode_mask(a['segmentation'], h, w) for a in gt_anns]
    pr_masks = [decode_mask(p['segmentation'], h, w) for p in preds]

    iou = np.zeros((n_pr, n_gt), dtype=np.float32)
    for pi, pm in enumerate(pr_masks):
        pm_sum = pm.sum()
        if pm_sum == 0:
            continue
        for gi, gm in enumerate(gt_masks):
            inter = (pm & gm).sum()
            if inter == 0:
                continue
            iou[pi, gi] = inter / (pm_sum + gm.sum() - inter)

    order    = np.argsort([p['score'] for p in preds])[::-1]
    gt_taken = [False] * n_gt
    pr_is_tp = [False] * n_pr

    for pi in order:
        best_gi, best_v = -1, IOU_THR
        for gi in range(n_gt):
            if not gt_taken[gi] and iou[pi, gi] >= best_v:
                best_v, best_gi = iou[pi, gi], gi
        if best_gi >= 0:
            pr_is_tp[pi] = True
            gt_taken[best_gi] = True

    return pr_is_tp, gt_taken


def render_four_panels(stem, orig_gt_key, v4_gt_key):
    """
    Render 4 panel: RGB | GT-v4 | orig-pred(TP/FP/FN) | v4-pred(TP/FP/FN)
    Kedua panel prediksi dievaluasi vs GT V4 yang SAMA (apple-to-apple).
    Returns numpy H × (4W + 3*SEP) × 3, atau None jika data tidak ada.
    """
    # Lookup v4 GT — common reference untuk kedua model
    if stem not in gt_index[v4_gt_key]:
        alt = 'v4_test' if 'val' in v4_gt_key else 'v4_val'
        if stem in gt_index.get(alt, {}):
            v4_gt_key = alt
        else:
            print(f'    [WARN] {stem}: V4 GT tidak ditemukan di {v4_gt_key} atau {alt}')
            return None

    v4_data = gt_index[v4_gt_key][stem]
    info    = v4_data['img_info']
    h, w    = info['height'], info['width']
    rgb     = np.array(Image.open(info['file_name']).convert('RGB'))
    gt_anns = v4_data['anns']  # GT V4 — dipakai untuk kedua model

    # Panel ① - original
    p1 = rgb.copy().astype(np.float32)

    # Panel ② - GT repaired v4 (cyan outline)
    p2 = rgb.copy().astype(np.float32)
    for ann in gt_anns:
        mask = decode_mask(ann['segmentation'], h, w)
        draw_mask(p2, mask, COLOR_GT)

    # Cari orig preds (coba orig_gt_key dulu, lalu fallback)
    orig_preds = []
    for k in (orig_gt_key, 'orig_test', 'orig_val'):
        orig_preds = pred_index.get(k, {}).get(stem, [])
        if orig_preds:
            break
    if not orig_preds:
        print(f'    [WARN] {stem}: tidak ada orig preds, panel ③ = foto saja')

    # Cari v4 preds
    v4_preds = []
    for k in (v4_gt_key, 'v4_test', 'v4_val'):
        v4_preds = pred_index.get(k, {}).get(stem, [])
        if v4_preds:
            break

    # Panel ③ - SwinB-orig predictions vs V4 GT (TP/FP/FN)
    p3 = rgb.copy().astype(np.float32)
    pr_is_tp3, gt_caught3 = match_tp_fp(gt_anns, orig_preds, h, w)
    for i, pred in enumerate(orig_preds):
        mask = decode_mask(pred['segmentation'], h, w)
        draw_mask(p3, mask, COLOR_TP if pr_is_tp3[i] else COLOR_FP)
    for i, ann in enumerate(gt_anns):
        if not gt_caught3[i]:
            mask = decode_mask(ann['segmentation'], h, w)
            draw_mask(p3, mask, COLOR_FN)

    # Panel ④ - SwinB-V4 predictions vs V4 GT (TP/FP/FN)
    p4 = rgb.copy().astype(np.float32)
    pr_is_tp4, gt_caught4 = match_tp_fp(gt_anns, v4_preds, h, w)
    for i, pred in enumerate(v4_preds):
        mask = decode_mask(pred['segmentation'], h, w)
        draw_mask(p4, mask, COLOR_TP if pr_is_tp4[i] else COLOR_FP)
    for i, ann in enumerate(gt_anns):
        if not gt_caught4[i]:
            mask = decode_mask(ann['segmentation'], h, w)
            draw_mask(p4, mask, COLOR_FN)

    # Hitung stats untuk label
    n_gt = len(gt_anns)
    tp3  = sum(pr_is_tp3);  fp3 = sum(1 for v in pr_is_tp3 if not v);  fn3 = sum(1 for v in gt_caught3 if not v)
    tp4  = sum(pr_is_tp4);  fp4 = sum(1 for v in pr_is_tp4 if not v);  fn4 = sum(1 for v in gt_caught4 if not v)

    # Gabung 4 panel dengan separator 8px abu-abu
    sep = np.full((h, 8, 3), 210, dtype=np.float32)
    row = np.concatenate([p1, sep, p2, sep, p3, sep, p4], axis=1)
    row = row.clip(0, 255).astype(np.uint8)

    # Tambah header strip dengan label teks di atas setiap panel
    from PIL import ImageDraw, ImageFont
    total_w = row.shape[1]
    hdr_h   = 36
    hdr     = Image.new('RGB', (total_w, hdr_h), (240, 242, 246))
    draw    = ImageDraw.Draw(hdr)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
        font_bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
    except Exception:
        font = font_bold = ImageFont.load_default()

    panel_w = w
    sep_w   = 8
    centers = [
        panel_w // 2,
        panel_w + sep_w + panel_w // 2,
        panel_w * 2 + sep_w * 2 + panel_w // 2,
        panel_w * 3 + sep_w * 3 + panel_w // 2,
    ]
    labels = [
        ('Original', (80, 80, 80), font),
        (f'GT V4  ({n_gt} pohon)', (20, 140, 180), font_bold),
        (f'SwinB-orig   TP={tp3} FP={fp3} FN={fn3}', (30, 120, 30), font_bold),
        (f'SwinB-V4     TP={tp4} FP={fp4} FN={fn4}', (30, 120, 30), font_bold),
    ]
    for cx, (txt, color, fnt) in zip(centers, labels):
        bbox = draw.textbbox((0, 0), txt, font=fnt)
        tw   = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, (hdr_h - (bbox[3] - bbox[1])) // 2), txt, fill=color, font=fnt)

    hdr_arr = np.array(hdr)
    return np.concatenate([hdr_arr, row], axis=0)


def arr_to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG', optimize=False)
    return base64.b64encode(buf.getvalue()).decode()


# ─── Tentukan orig GT key per stem ───────────────────────────────────────────

def find_orig_key(stem):
    """Return 'orig_test' atau 'orig_val' sesuai stem."""
    for k in ('orig_test', 'orig_val'):
        if stem in gt_index.get(k, {}):
            return k
    return 'orig_test'  # fallback


def find_v4_key(stem):
    """Return 'v4_test' atau 'v4_val' sesuai stem."""
    for k in ('v4_test', 'v4_val'):
        if stem in gt_index.get(k, {}):
            return k
    return 'v4_test'  # fallback


# ─── Parse HTML & replace base64 ─────────────────────────────────────────────

print('\nReading HTML dari backup...', flush=True)
html = Path(HTML_IN).read_text(encoding='utf-8')
print(f'  Source : {HTML_IN}', flush=True)
print(f'  Output : {HTML_OUT}', flush=True)

# Pattern: cari pasangan viz-label + img berikutnya
# viz-label bisa multi-word HTML, img langsung setelah baris viz-item/img
# Kita gunakan regex yang robust:
# 1) cari semua <img src="data:image/png;base64,XXX"> — ada 23
# 2) cari viz-label sebelum setiap img

# Ekstrak semua (position, stem) dari viz-label
stem_pattern = re.compile(r'Tree\d+_\d+')

# Cari semua img base64 dengan posisinya
img_pattern = re.compile(r'<img\s[^>]*src="data:image/png;base64,([A-Za-z0-9+/=\n]+)"')

# Cari semua viz-label dengan posisinya
label_pattern = re.compile(r'class="viz-label">([^<]+(?:<[^>]+>[^<]*</[^>]+>[^<]*)*)</div>')

# Collect all img matches
img_matches = list(img_pattern.finditer(html))
label_matches = list(label_pattern.finditer(html))

print(f'Found {len(img_matches)} images, {len(label_matches)} labels', flush=True)

# Untuk setiap img, cari label terdekat sebelumnya
def find_stem_before(pos, html_text):
    """Cari stem Tree... yang muncul sebelum posisi pos dalam 2000 char terakhir."""
    snippet = html_text[max(0, pos - 2000):pos]
    stems   = stem_pattern.findall(snippet)
    return stems[-1] if stems else None


# Buat HTML baru dengan replacement bertahap
print('\nRe-rendering images...', flush=True)
new_html = html

# Process dari belakang supaya posisi tidak bergeser
replacements = []
for match in img_matches:
    stem = find_stem_before(match.start(), html)
    if not stem:
        print(f'  [SKIP] Gambar di pos {match.start()} — stem tidak ditemukan')
        continue
    orig_key = find_orig_key(stem)
    v4_key   = find_v4_key(stem)
    replacements.append((match.start(), match.end(), match.group(1), stem, orig_key, v4_key))

print(f'Processing {len(replacements)} images...', flush=True)

# Process + replace dari belakang
offset = 0
for i, (start, end, old_b64, stem, orig_key, v4_key) in enumerate(replacements):
    print(f'  [{i+1:2d}/{len(replacements)}] {stem} (orig={orig_key}, v4={v4_key})', flush=True)

    arr = render_four_panels(stem, orig_key, v4_key)
    if arr is None:
        print(f'    [SKIP] render gagal')
        continue

    new_b64 = arr_to_b64(arr)
    old_img_tag = f'<img src="data:image/png;base64,{old_b64}"'
    new_img_tag = f'<img src="data:image/png;base64,{new_b64}"'

    if old_img_tag in new_html:
        new_html = new_html.replace(old_img_tag, new_img_tag, 1)
        print(f'    ✓ replaced ({len(new_b64)//1024} KB)', flush=True)
    else:
        # Fallback: replace by position
        actual_start = new_html.find(f'data:image/png;base64,{old_b64[:100]}')
        if actual_start >= 0:
            end_marker = new_html.find('"', actual_start + len('data:image/png;base64,'))
            new_html = (new_html[:actual_start] +
                        f'data:image/png;base64,{new_b64}' +
                        new_html[end_marker:])
            print(f'    ✓ replaced via position ({len(new_b64)//1024} KB)', flush=True)
        else:
            print(f'    [WARN] tidak bisa replace — skip', flush=True)

# ─── Update legend ────────────────────────────────────────────────────────────
# Ganti panel legend lama dengan versi baru yang mencantumkan TP/FP/FN
old_legend = (
    '<div class="panel-legend">\n'
    '  <span>Panel kiri ke kanan:</span>\n'
    '  <strong>① Original</strong>\n'
    '  <strong>② GT (repaired v4)</strong>\n'
    '  <strong>③ SwinB-orig prediction (score ≥ 0.35)</strong>\n'
    '  <strong>④ SwinB-V4 prediction (score ≥ 0.35)</strong>'
)
new_legend = (
    '<div class="panel-legend">\n'
    '  <span>Panel kiri ke kanan:</span>\n'
    '  <strong>① Original</strong>\n'
    '  <strong>② GT V4 (cyan outline)</strong>\n'
    '  <strong>③ SwinB-orig &nbsp;'
    '<span style="color:#1a8a3a">●TP</span> '
    '<span style="color:#c77a00">●FP</span> '
    '<span style="color:#c0392b">●FN</span></strong>\n'
    '  <strong>④ SwinB-V4 &nbsp;'
    '<span style="color:#1a8a3a">●TP</span> '
    '<span style="color:#c77a00">●FP</span> '
    '<span style="color:#c0392b">●FN</span></strong>'
)
if old_legend in new_html:
    new_html = new_html.replace(old_legend, new_legend)
    print('\nLegend updated ✓')
else:
    # Coba replace semua panel-legend yang ada referensi "SwinB-orig prediction"
    new_html = re.sub(
        r'<strong>③ SwinB-orig prediction \(score ≥ 0\.35\)</strong>',
        '<strong>③ SwinB-orig &nbsp;<span style="color:#1a8a3a">●TP</span> <span style="color:#c77a00">●FP</span></strong>',
        new_html
    )
    new_html = re.sub(
        r'<strong>④ SwinB-V4 prediction \(score ≥ 0\.35\)</strong>',
        '<strong>④ SwinB-V4 &nbsp;<span style="color:#1a8a3a">●TP</span> <span style="color:#c77a00">●FP</span></strong>',
        new_html
    )
    print('\nLegend updated via regex ✓')

# ─── Tambah color key setelah judul ──────────────────────────────────────────
color_key_html = (
    '\n<div style="display:flex;gap:22px;align-items:center;padding:7px 14px;'
    'background:#e8f4e8;border-radius:5px;font-size:.85em;font-weight:600;margin-bottom:6px">'
    '<span>Color key panel prediksi (③ &amp; ④ — vs GT V4 yang sama):</span>'
    '<span><span style="display:inline-block;width:14px;height:14px;background:rgb(50,220,80);'
    'border-radius:2px;vertical-align:middle;margin-right:4px"></span>TP (terdeteksi benar)</span>'
    '<span><span style="display:inline-block;width:14px;height:14px;background:rgb(255,140,0);'
    'border-radius:2px;vertical-align:middle;margin-right:4px"></span>FP (false alarm)</span>'
    '<span><span style="display:inline-block;width:14px;height:14px;background:rgb(220,30,30);'
    'border-radius:2px;vertical-align:middle;margin-right:4px"></span>FN (pohon terlewat)</span>'
    '<span><span style="display:inline-block;width:14px;height:14px;background:rgb(80,200,255);'
    'border-radius:2px;vertical-align:middle;margin-right:4px"></span>Panel ② = GT V4</span>'
    '</div>\n'
)

# Sisipkan setelah <div class="section"> pertama yang mengandung visualisasi
viz_section_marker = '<h2>Visualisasi Prediksi'
if viz_section_marker in new_html:
    insert_pos = new_html.find(viz_section_marker)
    new_html = new_html[:insert_pos] + color_key_html + new_html[insert_pos:]
    print('Color key inserted ✓')

# ─── Write output ─────────────────────────────────────────────────────────────
out_path = Path(HTML_OUT)
out_path.write_text(new_html, encoding='utf-8')
size_mb = out_path.stat().st_size / 1e6
print(f'\n✓ Saved → {out_path}  ({size_mb:.1f} MB)')
