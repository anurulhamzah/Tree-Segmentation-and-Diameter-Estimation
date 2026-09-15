#!/usr/bin/env python3
"""
Generate HTML report dengan visualisasi TP/FN per-gambar yang jelas.

Warna overlay:
  Hijau  = TP  — prediksi match GT (IoU >= iou_thr) / GT yang terdeteksi
  Oranye = FP  — prediksi yang tidak match GT
  Merah  = FN  — GT yang tidak terdeteksi

Layout per gambar (3 panel berdampingan):
  ① Foto asli RGB
  ② GT colored   : hijau = terdeteksi (TP-gt), merah = missed (FN)
  ③ Pred colored : hijau = TP, oranye = FP

Jalankan:
  python scripts/gen_tp_fn_viz_html.py                  # default 30 gambar worst F1
  python scripts/gen_tp_fn_viz_html.py --max-imgs 50 --score-thr 0.30
  python scripts/gen_tp_fn_viz_html.py --models SwinB-V4
"""

import argparse
import base64
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from scipy.ndimage import binary_erosion

# ─── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    'SwinB-orig': dict(
        label='SwinB Original (non-repaired)',
        gt_json='/scratch2/pr65/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000/instances_test.json',
        pred_json='/scratch2/pr65/anur0018/maskdino_output/SwinB_combined_rle_f1000_75k/eval_test_orig/inference/coco_instances_results.json',
        color_header='#e67e22',
    ),
    'SwinB-V4': dict(
        label='SwinB Repaired V4 Stratified',
        gt_json='/scratch2/pr65/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json',
        pred_json='/scratch2/pr65/anur0018/maskdino_output/SwinB_combined_rle_f1000_repaired_v4_stratified_75k/eval_test_v4/inference/coco_instances_results.json',
        color_header='#27ae60',
    ),
}

# ─── Warna (R, G, B) ─────────────────────────────────────────────────────────
COLOR_TP  = np.array([50,  220,  80], dtype=np.float32)   # hijau cerah
COLOR_FP  = np.array([255, 140,   0], dtype=np.float32)   # oranye
COLOR_FN  = np.array([220,  30,  30], dtype=np.float32)   # merah solid

ALPHA_FILL   = 0.42
ALPHA_BORDER = 0.90
BORDER_PX    = 2

# ─── Utilities ───────────────────────────────────────────────────────────────

def decode_mask(seg, h, w):
    if isinstance(seg, dict):
        return maskUtils.decode(seg).astype(bool)
    rles = maskUtils.frPyObjects(seg, h, w)
    return maskUtils.decode(maskUtils.merge(rles)).astype(bool)


def img_to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def draw_mask(canvas, mask, color):
    if not mask.any():
        return
    canvas[mask] = canvas[mask] * (1 - ALPHA_FILL) + color * ALPHA_FILL
    inner  = binary_erosion(mask, iterations=BORDER_PX)
    border = mask & ~inner
    if border.any():
        canvas[border] = canvas[border] * (1 - ALPHA_BORDER) + color * ALPHA_BORDER


# ─── Per-image greedy IoU matching ───────────────────────────────────────────

def match_per_image(gt_anns, preds, h, w, iou_thr):
    """
    Greedy match prediksi → GT (descending score).
    Returns:
      pr_is_tp : list[bool] len=len(preds), True jika TP
      gt_caught : list[bool] len=len(gt_anns), True jika terdeteksi
    """
    n_gt = len(gt_anns)
    n_pr = len(preds)

    if n_gt == 0 or n_pr == 0:
        return [False] * n_pr, [False] * n_gt

    gt_masks = [decode_mask(a['segmentation'], h, w) for a in gt_anns]
    pr_masks = [decode_mask(p['segmentation'], h, w) for p in preds]

    # IoU matrix n_pr × n_gt
    iou = np.zeros((n_pr, n_gt), dtype=np.float32)
    for pi, pm in enumerate(pr_masks):
        pm_flat = pm.ravel()
        pm_sum  = pm_flat.sum()
        if pm_sum == 0:
            continue
        for gi, gm in enumerate(gt_masks):
            inter = (pm_flat & gm.ravel()).sum()
            if inter == 0:
                continue
            iou[pi, gi] = inter / (pm_sum + gm.sum() - inter)

    order     = np.argsort([p['score'] for p in preds])[::-1]
    gt_taken  = [False] * n_gt
    pr_is_tp  = [False] * n_pr

    for pi in order:
        row = iou[pi]
        best_gi = -1
        best_v  = iou_thr
        for gi in range(n_gt):
            if gt_taken[gi]:
                continue
            if row[gi] >= best_v:
                best_v  = row[gi]
                best_gi = gi
        if best_gi >= 0:
            pr_is_tp[pi]      = True
            gt_taken[best_gi] = True

    return pr_is_tp, gt_taken


# ─── Render single image ─────────────────────────────────────────────────────

def render_triplet(rgb, h, w, gt_anns, preds, iou_thr):
    pr_is_tp, gt_caught = match_per_image(gt_anns, preds, h, w, iou_thr)

    orig  = rgb.copy().astype(np.float32)
    gt_cv = rgb.copy().astype(np.float32)
    pr_cv = rgb.copy().astype(np.float32)

    for i, ann in enumerate(gt_anns):
        mask = decode_mask(ann['segmentation'], h, w)
        draw_mask(gt_cv, mask, COLOR_TP if gt_caught[i] else COLOR_FN)

    for i, pred in enumerate(preds):
        mask = decode_mask(pred['segmentation'], h, w)
        draw_mask(pr_cv, mask, COLOR_TP if pr_is_tp[i] else COLOR_FP)

    # FN: GT pohon yang tidak terdeteksi — tampilkan di panel prediksi (merah)
    for i, ann in enumerate(gt_anns):
        if not gt_caught[i]:
            mask = decode_mask(ann['segmentation'], h, w)
            draw_mask(pr_cv, mask, COLOR_FN)

    sep = np.full((h, 10, 3), 230, dtype=np.float32)
    row = np.concatenate([orig, sep, gt_cv, sep, pr_cv], axis=1)
    return row.clip(0, 255).astype(np.uint8)


# ─── Process one model ───────────────────────────────────────────────────────

def process_model(cfg, score_thr, iou_thr, max_imgs):
    print('  Loading GT...', flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(cfg['gt_json'])

    print('  Loading predictions...', flush=True)
    preds_all  = json.load(open(cfg['pred_json']))
    preds_filt = [p for p in preds_all if p['score'] >= score_thr]

    pred_by_img = defaultdict(list)
    for p in preds_filt:
        pred_by_img[p['image_id']].append(p)

    gt_by_img = defaultdict(list)
    for ann in coco_gt.dataset['annotations']:
        gt_by_img[ann['image_id']].append(ann)

    # Hitung per-image stats untuk sorting
    print('  Computing per-image stats...', flush=True)
    img_records = []
    for img_info in coco_gt.dataset['images']:
        iid   = img_info['id']
        anns  = gt_by_img[iid]
        preds = pred_by_img[iid]
        n_gt  = len(anns)
        if n_gt == 0:
            continue

        pr_is_tp, gt_caught = match_per_image(
            anns, preds, img_info['height'], img_info['width'], iou_thr)

        tp = sum(pr_is_tp)
        fp = sum(1 for v in pr_is_tp if not v)
        fn = sum(1 for v in gt_caught if not v)
        pr_val = tp / (tp + fp) if (tp + fp) else 0.0
        rc_val = tp / (tp + fn) if (tp + fn) else 0.0
        f1     = 2 * pr_val * rc_val / (pr_val + rc_val) if (pr_val + rc_val) else 0.0

        img_records.append(dict(
            iid=iid, img_info=img_info,
            n_gt=n_gt, tp=tp, fp=fp, fn=fn, f1=f1,
        ))

    # Sort worst F1 first
    img_records.sort(key=lambda x: x['f1'])

    print(f'  Rendering {min(max_imgs, len(img_records))} images...', flush=True)
    cards = []
    for rec in img_records[:max_imgs]:
        iid      = rec['iid']
        info     = rec['img_info']
        h, w     = info['height'], info['width']
        fn_path  = info['file_name']
        anns     = gt_by_img[iid]
        preds    = pred_by_img[iid]

        rgb    = np.array(Image.open(fn_path).convert('RGB'))
        panel  = render_triplet(rgb, h, w, anns, preds, iou_thr)
        b64    = img_to_b64(panel)

        cards.append(dict(
            stem=Path(fn_path).stem,
            b64=b64,
            n_gt=rec['n_gt'], tp=rec['tp'], fp=rec['fp'],
            fn=rec['fn'], f1=rec['f1'],
        ))
        if len(cards) % 5 == 0:
            print(f'    ... {len(cards)} done', flush=True)

    return cards


# ─── HTML generation ─────────────────────────────────────────────────────────

CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;margin:36px;background:#f5f6fa;color:#222}
h1{font-size:1.5em;color:#1a1a2e;border-bottom:3px solid #4a90d9;padding-bottom:8px}
h2{font-size:1.05em;color:#2c3e50;margin-top:26px}
.legend{display:flex;gap:22px;align-items:center;padding:8px 14px;
  background:#e0e3ea;border-radius:6px;margin-bottom:10px;font-size:.86em;font-weight:600}
.swatch{display:inline-block;width:16px;height:16px;border-radius:3px;
  margin-right:5px;vertical-align:middle;border:1px solid #999}
.sub-note{font-size:.82em;color:#666;margin-bottom:16px}
.model-block{background:white;padding:18px;border-radius:8px;margin-bottom:28px;
  box-shadow:0 1px 5px rgba(0,0,0,.1)}
.model-badge{display:inline-block;color:white;padding:4px 14px;border-radius:5px;
  font-weight:bold;font-size:1em;margin-bottom:12px}
.col-labels{display:flex;font-size:.79em;color:#555;font-weight:600;margin-bottom:5px}
.col-label{flex:1;text-align:center;padding:2px 0}
.card{margin-bottom:18px;border-left:4px solid #ddd;padding-left:10px}
.card-meta{font-size:.8em;font-family:monospace;color:#333;margin-bottom:4px}
.card img{max-width:100%;border-radius:4px;display:block}
.tp{color:#1a8a3a;font-weight:bold}
.fp{color:#b36200;font-weight:bold}
.fn{color:#c0392b;font-weight:bold}
.f1g{color:#27ae60;font-weight:bold}
.f1y{color:#c77a00;font-weight:bold}
.f1r{color:#c0392b;font-weight:bold}
"""


def f1_class(f1):
    if f1 >= 0.5:  return 'f1g'
    if f1 >= 0.3:  return 'f1y'
    return 'f1r'


def build_html(all_cards, score_thr, iou_thr, out_path):
    parts = [
        '<!DOCTYPE html><html lang="id"><head><meta charset="UTF-8">',
        '<title>TP/FN Visualization</title>',
        f'<style>{CSS}</style></head><body>',
        '<h1>Visualisasi TP / FP / FN — SwinB Original vs V4 Stratified</h1>',
        f'<p class="sub-note">Score threshold: <b>{score_thr}</b> &nbsp;|&nbsp; '
        f'IoU threshold: <b>{iou_thr}</b> &nbsp;|&nbsp; '
        'Sorting: worst F1 first (hanya gambar dengan GT &gt; 0)</p>',
        # Legend
        '<div class="legend">',
        '<span><span class="swatch" style="background:rgb(50,220,80)"></span>'
        'TP — Prediksi match GT</span>',
        '<span><span class="swatch" style="background:rgb(255,140,0)"></span>'
        'FP — Prediksi tidak match GT</span>',
        '<span><span class="swatch" style="background:rgb(220,30,30)"></span>'
        'FN — GT pohon yang terlewat</span>',
        '</div>',
        '<p class="sub-note">Panel kiri → kanan: '
        '<b>① Foto asli</b> &nbsp;|&nbsp; '
        '<b>② GT</b> (hijau = terdeteksi, merah = FN/missed) &nbsp;|&nbsp; '
        '<b>③ Prediksi</b> (hijau = TP, oranye = FP, merah = FN/missed)</p>',
    ]

    for model_key, cards in all_cards.items():
        cfg = MODELS[model_key]
        parts += [
            '<div class="model-block">',
            f'<div class="model-badge" style="background:{cfg["color_header"]}">'
            f'{cfg["label"]}</div>',
            '<div class="col-labels">',
            '<div class="col-label">① Original</div>',
            '<div class="col-label">② GT &nbsp;(hijau=TP / merah=FN)</div>',
            '<div class="col-label">③ Prediksi &nbsp;(hijau=TP / oranye=FP / merah=FN)</div>',
            '</div>',
        ]
        for c in cards:
            fc = f1_class(c['f1'])
            parts += [
                '<div class="card">',
                f'<div class="card-meta">'
                f'{c["stem"]} &nbsp;|&nbsp; GT={c["n_gt"]} &nbsp;|&nbsp; '
                f'<span class="tp">TP={c["tp"]}</span> '
                f'<span class="fp">FP={c["fp"]}</span> '
                f'<span class="fn">FN={c["fn"]}</span> &nbsp;|&nbsp; '
                f'<span class="{fc}">F1={c["f1"]*100:.1f}%</span>'
                f'</div>',
                f'<img src="data:image/png;base64,{c["b64"]}" loading="lazy">',
                '</div>',
            ]
        parts.append('</div>')

    parts += [
        '<p style="font-size:.8em;color:#aaa;text-align:center;margin-top:20px">'
        'gen_tp_fn_viz_html.py</p>',
        '</body></html>',
    ]

    out_path = Path(out_path)
    out_path.write_text('\n'.join(parts), encoding='utf-8')
    print(f'\n✓ Saved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models',    nargs='+', default=list(MODELS.keys()))
    parser.add_argument('--max-imgs',  type=int,   default=30)
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--iou-thr',   type=float, default=0.50)
    parser.add_argument('--out', default=(
        '/scratch2/pr65/anur0018/tree_classification/'
        'reports/stratified_v4/tp_fn_viz_swinb.html'))
    args = parser.parse_args()

    all_cards = {}
    for key in args.models:
        if key not in MODELS:
            print(f'[WARN] Unknown model: {key}'); continue
        print(f'\n[{key}] {MODELS[key]["label"]}')
        cards = process_model(MODELS[key], args.score_thr, args.iou_thr, args.max_imgs)
        all_cards[key] = cards
        print(f'  → {len(cards)} gambar dirender | '
              f'avg F1={np.mean([c["f1"] for c in cards])*100:.1f}%')

    build_html(all_cards, args.score_thr, args.iou_thr, args.out)


if __name__ == '__main__':
    main()
