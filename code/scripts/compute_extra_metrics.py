"""
Hitung metric tambahan untuk semua model v4 stratified:
  - AR100     : COCO Average Recall @ maxDets=100, IoU=0.50:0.95
  - macro_F1  : macro-averaged F1 per kelas @ score_thr, IoU=0.50
  - macro_Rec : macro-averaged Recall per kelas
  - macro_Prec: macro-averaged Precision per kelas
  - Accuracy  : macro per-class accuracy (= macro Recall, definisi COCO-detection)
  - FPS       : dari log inference (s/iter → FPS)

Update eval_test_stratified_v4.json dengan field baru.
"""
import json, contextlib, io, re, sys
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

OUTPUT_ROOT  = '/scratch2/pr65/anur0018/maskdino_output'
REPORTS_DIR  = '/scratch2/pr65/anur0018/tree_classification/reports'
JSON_PATH    = f'{REPORTS_DIR}/eval_test_stratified_v4.json'

GT_FILES = {
    'combined':    '/scratch2/pr65/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json',
    'plantations': '/scratch2/pr65/anur0018/tree_classification/data/plantations/annotations_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json',
}

MODEL_DIRS = {
    'R50-pt':       'R50_combined_rle_f1000_repaired_v4_stratified_75k',
    'R50-sc':       'R50_combined_rle_f1000_scratch_repaired_v4_stratified_75k',
    'SwinT-pt':     'SwinT_combined_rle_f1000_repaired_v4_stratified_75k',
    'SwinB-pt':     'SwinB_combined_rle_f1000_repaired_v4_stratified_75k',
    'SwinB-sc':     'SwinB_combined_rle_f1000_rgb_scratch_repaired_v4_stratified_75k',
    'SwinB-RGBDp':  'SwinB_combined_rle_f1000_rgbd_repaired_v4_stratified_75k',
    'SwinB-RGBDs':  'SwinB_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k',
    'SwinB-DBH':    'SwinB_combined_rle_f1000_rgbd_dbh_repaired_v4_stratified_75k',
    'SwinB-PL':     'SwinB_pl_rle_f1000_repaired_v4_stratified_75k',
    'SwinL-pt':     'SwinL_combined_rle_f1000_repaired_v4_stratified_75k',
    'FocalNetL-sc': 'FocalNet_L_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k',
}

TEST_DS = {s: 'combined' for s in MODEL_DIRS}
TEST_DS['SwinB-PL'] = 'plantations'

SCORE_THR = 0.35
IOU_THR   = 0.50


def get_fps_from_log(model_dir):
    log = Path(f'{OUTPUT_ROOT}/{model_dir}/eval_test_v4/log.txt')
    if not log.exists():
        return None
    text = log.read_text()
    # Cari "Total inference time: H:MM:SS.ffffff (X s / iter ...)"
    m = re.search(r'Total inference pure compute time:.*?\(([0-9.]+) s / iter', text)
    if m:
        return round(1.0 / float(m.group(1)), 2)
    return None


def compute_ar100(coco_gt, coco_dt):
    ev = COCOeval(coco_gt, coco_dt, 'segm')
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return round(float(ev.stats[8]) * 100, 2)   # AR @ maxDets=100


def compute_macro_metrics(coco_gt, coco_dt_filt, iou_thr=IOU_THR):
    cat_ids = sorted(coco_gt.getCatIds())

    ev = COCOeval(coco_gt, coco_dt_filt, 'segm')
    ev.params.iouThrs    = np.array([iou_thr])
    ev.params.areaRng    = [[0, 1e10]]
    ev.params.areaRngLbl = ['all']
    ev.params.maxDets    = [10000]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()

    per_class = {}
    for cat_id in cat_ids:
        tp = fp = fn = 0
        for ei in ev.evalImgs:
            if ei is None or ei['category_id'] != cat_id:
                continue
            for v, ig in zip(ei['dtMatches'][0], ei['dtIgnore'][0]):
                if ig: continue
                tp += int(v > 0); fp += int(v == 0)
            for v, ig in zip(ei['gtMatches'][0], ei['gtIgnore']):
                if ig: continue
                fn += int(v == 0)
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        per_class[cat_id] = {'tp': tp, 'fp': fp, 'fn': fn,
                              'precision': pr, 'recall': rc, 'f1': f1}

    vals = list(per_class.values())
    cat_names = {c['id']: c['name'] for c in coco_gt.loadCats(cat_ids)}

    return {
        'macro_F1':       round(float(np.mean([v['f1']        for v in vals])) * 100, 2),
        'macro_Recall':   round(float(np.mean([v['recall']    for v in vals])) * 100, 2),
        'macro_Precision':round(float(np.mean([v['precision'] for v in vals])) * 100, 2),
        # Accuracy = macro recall: fraction GT per kelas yang terdeteksi dengan label benar
        'Accuracy':       round(float(np.mean([v['recall']    for v in vals])) * 100, 2),
        'per_class_f1':   {cat_names[c]: round(per_class[c]['f1']*100, 2) for c in cat_ids},
    }


def process_model(short, model_dir, test_ds):
    pred_file = Path(f'{OUTPUT_ROOT}/{model_dir}/eval_test_v4/inference/coco_instances_results.json')
    if not pred_file.exists():
        print(f'  SKIP {short} — no pred file')
        return None

    print(f'  {short:14s} ...', end='', flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt   = COCO(GT_FILES[test_ds])
        preds_raw = json.load(open(pred_file))
        if not preds_raw:
            print(' empty pred')
            return None
        coco_dt      = coco_gt.loadRes(preds_raw)
        preds_filt   = [p for p in preds_raw if p['score'] >= SCORE_THR]
        coco_dt_filt = coco_gt.loadRes(preds_filt) if preds_filt else None

    ar100   = compute_ar100(coco_gt, coco_dt)
    macro   = compute_macro_metrics(coco_gt, coco_dt_filt) if coco_dt_filt else {}
    fps     = get_fps_from_log(model_dir)

    print(f' AR100={ar100:.1f}% macro_F1={macro.get("macro_F1","—")}% FPS={fps}')
    return {'AR100': ar100, 'FPS': fps, **macro}


# ── Main ──────────────────────────────────────────────────────────────────────
data = json.load(open(JSON_PATH))

for m in data['models']:
    short = m['short']
    if short not in MODEL_DIRS:
        print(f'  SKIP {short} — no dir mapping')
        continue
    result = process_model(short, MODEL_DIRS[short], TEST_DS[short])
    if result is None:
        continue
    m['metrics_test'].update({
        'AR100':          result.get('AR100'),
        'macro_F1':       result.get('macro_F1'),
        'macro_Recall':   result.get('macro_Recall'),
        'macro_Precision':result.get('macro_Precision'),
        'Accuracy':       result.get('Accuracy'),
        'per_class_f1':   result.get('per_class_f1'),
        'FPS':            result.get('FPS'),
    })

import datetime
data['created'] = datetime.datetime.now().isoformat()
data['score_thr_macro'] = SCORE_THR

with open(JSON_PATH, 'w') as f:
    json.dump(data, f, indent=2)

print(f'\nSaved → {JSON_PATH}')
