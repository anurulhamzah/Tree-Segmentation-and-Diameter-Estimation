"""
Compute F1@0.35 and Recall@0.35 for Mask R-CNN models from saved predictions.
No re-inference needed — reads coco_instances_results.json directly.
"""
import json
import io
import contextlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_FILE = '/scratch2/pr65/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json'
SCORE_THR = 0.35
IOU_THR = 0.50

MODELS = {
    'Mask R-CNN R50 RGB-pretrained': '/scratch2/pr65/anur0018/maskdino_output/maskrcnn_R50_combined_rgb_pretrained_v4_stratified_75k/eval_test_v4/inference/coco_instances_results.json',
    'Mask R-CNN R50 RGBD-scratch':   '/scratch2/pr65/anur0018/maskdino_output/maskrcnn_R50_combined_rgbd_scratch_v4_stratified_75k/eval_test_v4/inference/coco_instances_results.json',
}

coco_gt = COCO(GT_FILE)
n_gt = len(coco_gt.getAnnIds())
print(f"GT annotations: {n_gt}")
print()

results = {}

for name, pred_path in MODELS.items():
    print(f"=== {name} ===")
    with open(pred_path) as f:
        preds = json.load(f)

    # Filter by score threshold
    preds_filtered = [p for p in preds if p['score'] >= SCORE_THR]
    print(f"  Total preds: {len(preds)}, after score>={SCORE_THR}: {len(preds_filtered)}")

    if len(preds_filtered) == 0:
        print("  WARNING: no predictions after threshold — F1=0, Recall=0")
        results[name] = {'F1': 0.0, 'Recall': 0.0, 'Precision': 0.0}
        continue

    coco_dt = coco_gt.loadRes(preds_filtered)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.params.iouThrs = np.array([IOU_THR])
    coco_eval.params.maxDets = [1, 10, 100]

    # Suppress stdout from evaluate/accumulate
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()

    # precision[T, R, K, A, M]: T=iou_thr, R=recall_thr, K=cat, A=area, M=maxDets
    # We want all categories, all areas, maxDets=100 (index 2)
    prec = coco_eval.eval['precision']  # shape: (T, R, K, A, M)
    rec_thr = coco_eval.params.recThrs  # 101 points from 0 to 1

    # T=0 (only IoU=0.5), A=0 (all areas), M=2 (maxDets=100)
    prec_vals = prec[0, :, :, 0, 2]  # (101 recall_thrs, K cats)
    # Mean over categories (ignore -1 entries which mean no GT)
    prec_mean = np.mean(prec_vals[prec_vals >= 0]) if np.any(prec_vals >= 0) else 0.0

    # AP50 (standard) — mean over recall thresholds
    ap50 = float(np.mean(prec_vals[prec_vals >= 0])) * 100 if np.any(prec_vals >= 0) else 0.0

    # For F1@0.35: we need TP, FP, FN counts directly
    # Use coco_eval.evalImgs to count TP/FP/FN
    TP = FP = FN = 0
    for evalImg in coco_eval.evalImgs:
        if evalImg is None:
            continue
        dt_ids = evalImg['dtIds']
        dt_matches = evalImg['dtMatches'][0]  # T=0, IoU=0.5
        dt_ignore = evalImg['dtIgnore'][0]
        gt_ids = evalImg['gtIds']
        gt_ignore = evalImg['gtIgnore']

        # TP: dt with a match and not ignored
        for i, (match, ign) in enumerate(zip(dt_matches, dt_ignore)):
            if ign:
                continue
            if match > 0:
                TP += 1
            else:
                FP += 1

        # FN: GT not matched and not ignored
        gt_match = evalImg['gtMatches'][0]  # (n_gt,)
        for i, (match, ign) in enumerate(zip(gt_match, gt_ignore)):
            if ign:
                continue
            if match == 0:
                FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"  TP={TP}, FP={FP}, FN={FN}")
    print(f"  Precision@{SCORE_THR}: {precision*100:.2f}%")
    print(f"  Recall@{SCORE_THR}:    {recall*100:.2f}%")
    print(f"  F1@{SCORE_THR}:        {f1*100:.2f}%")
    print(f"  AP50 (from COCOeval):  {ap50:.2f}%")
    print()

    results[name] = {
        'TP': TP, 'FP': FP, 'FN': FN,
        'Precision': round(precision*100, 2),
        'Recall': round(recall*100, 2),
        'F1': round(f1*100, 2),
        'AP50_coco': round(ap50, 2),
    }

print("=== SUMMARY (for Table I) ===")
for name, m in results.items():
    print(f"{name}: AP50=??% (from eval log), F1@0.35={m['F1']}%, Recall@0.35={m['Recall']}%")
