"""
Hitung metrik tambahan (dari riset literatur) untuk semua model V4 stratified,
TANPA perlu retrain — semua dihitung dari coco_instances_results.json yang sudah ada.

Metrik baru yang ditambahkan (vs eval_test_stratified_v4.json yang sudah ada):
  - AR100          : COCO Average Recall @ maxDets=100, IoU=0.50:0.95 (sempat dihitung tapi dibuang)
  - APs/APm/APl    : AP untuk objek kecil/sedang/besar (COCO standard, sempat dihitung tapi dibuang)
  - per_class_AP50 : AP50 per kelas (bukan cuma F1 macro)
  - Boundary IoU/AP: Cheng et al. CVPR 2021 — AP dihitung ulang pakai boundary-only mask
                     (radius dilasi 2% diagonal image), untuk diagnosis kualitas tepi mask
                     yang tidak sensitif di mask-IoU biasa (relevan utk kanopi jagged kita)
  - confusion_matrix: prediksi vs GT per kelas (13x13 utk combined), untuk lihat spesies
                     mana yang saling tertukar

Usage:
    python compute_literature_metrics_v4.py
"""
import json
import contextlib
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

OUTPUT_ROOT = Path("/scratch2/pr65/anur0018/maskdino_output")
DATA_ROOT   = Path("/scratch2/pr65/anur0018/tree_classification/data")
REPORTS_DIR = Path("/scratch2/pr65/anur0018/tree_classification/reports")

GT_FILES = {
    "combined":    DATA_ROOT / "combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json",
    "plantations": DATA_ROOT / "plantations/annotations_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json",
    "rainforests": DATA_ROOT / "rainforests/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json",
    "rainforests_960": DATA_ROOT / "rainforests/annotation_inst/filtered_rle_f1000_960/instances_test.json",
}

SCORE_THR = 0.35
IOU_THR   = 0.50
DILATION_RATIO = 0.02  # standar Boundary IoU (Cheng et al. 2021)


# ─── Boundary IoU (Cheng et al., CVPR 2021) ────────────────────────────────────
def mask_to_boundary(mask_arr, dilation_ratio=DILATION_RATIO):
    h, w = mask_arr.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    new_mask = cv2.copyMakeBorder(mask_arr, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1:h + 1, 1:w + 1]
    return mask_arr - mask_erode


def rle_to_boundary_rle(rle, h, w):
    m = maskUtils.decode(rle).astype(np.uint8)
    b = mask_to_boundary(m)
    if b.sum() == 0:  # instance terlalu kecil, boundary abis dimakan erosion
        b = m  # fallback: pakai mask penuh
    rle_out = maskUtils.encode(np.asfortranarray(b))
    rle_out["counts"] = rle_out["counts"].decode("ascii")
    return rle_out


def compute_boundary_ap(gt_path, dt_path):
    """Hitung Boundary AP: sama seperti Mask AP tapi GT & DT diganti versi boundary-only."""
    coco_gt = COCO(str(gt_path))
    imgs = {im["id"]: (im["height"], im["width"]) for im in coco_gt.dataset["images"]}

    gt_b = json.loads(json.dumps(coco_gt.dataset))  # deep copy murah via json
    for ann in gt_b["annotations"]:
        h, w = imgs[ann["image_id"]]
        seg = ann["segmentation"]
        rle = seg if isinstance(seg, dict) else maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
        ann["segmentation"] = rle_to_boundary_rle(rle, h, w)

    dt_raw = json.load(open(dt_path))
    dt_b = []
    for d in dt_raw:
        h, w = imgs[d["image_id"]]
        seg = d["segmentation"]
        rle = seg if isinstance(seg, dict) else maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
        d2 = dict(d)
        d2["segmentation"] = rle_to_boundary_rle(rle, h, w)
        dt_b.append(d2)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt_b = COCO()
        coco_gt_b.dataset = gt_b
        coco_gt_b.createIndex()
        coco_dt_b = coco_gt_b.loadRes(dt_b)
        E = COCOeval(coco_gt_b, coco_dt_b, "segm")
        E.evaluate(); E.accumulate(); E.summarize()

    return {
        "boundary_AP":   round(float(E.stats[0]) * 100, 2),
        "boundary_AP50": round(float(E.stats[1]) * 100, 2),
        "boundary_AP75": round(float(E.stats[2]) * 100, 2),
    }


# ─── AR100, APs/APm/APl (COCO standard, sempat dihitung tapi dibuang) ─────────
def compute_size_stratified_ap(gt_path, dt_path):
    coco_gt = COCO(str(gt_path))
    dt = json.load(open(dt_path))
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dt)
        E = COCOeval(coco_gt, coco_dt, "segm")
        E.evaluate(); E.accumulate(); E.summarize()
    return {
        "AP":     round(float(E.stats[0]) * 100, 2),
        "AP50":   round(float(E.stats[1]) * 100, 2),
        "AP75":   round(float(E.stats[2]) * 100, 2),
        "APs":    round(float(E.stats[3]) * 100, 2),
        "APm":    round(float(E.stats[4]) * 100, 2),
        "APl":    round(float(E.stats[5]) * 100, 2),
        "AR1":    round(float(E.stats[6]) * 100, 2),
        "AR10":   round(float(E.stats[7]) * 100, 2),
        "AR100":  round(float(E.stats[8]) * 100, 2),
        "ARs":    round(float(E.stats[9]) * 100, 2),
        "ARm":    round(float(E.stats[10]) * 100, 2),
        "ARl":    round(float(E.stats[11]) * 100, 2),
    }


# ─── Per-class AP50 + Confusion Matrix ─────────────────────────────────────────
def compute_per_class_ap_and_confusion(gt_path, dt_path, score_thr=SCORE_THR, iou_thr=IOU_THR):
    coco_gt = COCO(str(gt_path))
    cat_ids = sorted(coco_gt.getCatIds())
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    dt_raw = json.load(open(dt_path))
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt_full = coco_gt.loadRes(dt_raw)
        E = COCOeval(coco_gt, coco_dt_full, "segm")
        E.evaluate(); E.accumulate()

    # Per-class AP50 (IoU idx=0 karena default COCOeval iouThrs[0]=0.5)
    per_class_ap50 = {}
    for k_idx, cat_id in enumerate(cat_ids):
        prec = E.eval["precision"][0, :, k_idx, 0, 2]  # iou=0.5, area=all, maxDets=100
        prec = prec[prec > -1]
        per_class_ap50[cat_names[cat_id]] = round(float(np.mean(prec)) * 100, 2) if len(prec) else 0.0

    # Confusion matrix: untuk tiap GT matched (IoU>=thr, score>=thr), kelas prediksi vs kelas GT
    dt_filt = [d for d in dt_raw if d["score"] >= score_thr]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dt_filt) if dt_filt else None

    confusion = defaultdict(lambda: defaultdict(int))  # confusion[gt_name][pred_name] = count
    unmatched_gt = defaultdict(int)  # miss total per gt class (tidak match apapun)

    imgs = coco_gt.getImgIds()
    for img_id in imgs:
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dt_anns = [d for d in dt_filt if d["image_id"] == img_id]
        if not gt_anns:
            continue
        img_info = coco_gt.loadImgs(img_id)[0]
        h, w = img_info["height"], img_info["width"]

        gt_rles = [ann["segmentation"] if isinstance(ann["segmentation"], dict)
                   else maskUtils.merge(maskUtils.frPyObjects(ann["segmentation"], h, w))
                   for ann in gt_anns]
        if dt_anns:
            dt_rles = [d["segmentation"] if isinstance(d["segmentation"], dict)
                       else maskUtils.merge(maskUtils.frPyObjects(d["segmentation"], h, w))
                       for d in dt_anns]
            ious = maskUtils.iou(dt_rles, gt_rles, [0] * len(gt_rles))
        else:
            ious = np.zeros((0, len(gt_rles)))

        matched_gt = set()
        # greedy match by descending score
        order = sorted(range(len(dt_anns)), key=lambda i: -dt_anns[i]["score"])
        for di in order:
            best_iou, best_gi = 0.0, -1
            for gi in range(len(gt_anns)):
                if gi in matched_gt:
                    continue
                iou = ious[di, gi] if len(gt_anns) else 0.0
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_thr and best_gi >= 0:
                matched_gt.add(best_gi)
                gt_name = cat_names[gt_anns[best_gi]["category_id"]]
                pred_name = cat_names[dt_anns[di]["category_id"]]
                confusion[gt_name][pred_name] += 1

        for gi, ann in enumerate(gt_anns):
            if gi not in matched_gt:
                unmatched_gt[cat_names[ann["category_id"]]] += 1

    # Convert to dense dict (termasuk "MISSED" utk GT yang tidak terdeteksi sama sekali)
    all_names = sorted(cat_names.values())
    confusion_dense = {}
    for gt_name in all_names:
        row = {pred_name: confusion[gt_name].get(pred_name, 0) for pred_name in all_names}
        row["MISSED (FN)"] = unmatched_gt.get(gt_name, 0)
        confusion_dense[gt_name] = row

    return per_class_ap50, confusion_dense


# ─── Model list (19 model V4) ───────────────────────────────────────────────────
MODELS = [
    dict(short="R50-pt",      dir="R50_combined_rle_f1000_repaired_v4_stratified_75k", ds="combined"),
    dict(short="R50-sc",      dir="R50_combined_rle_f1000_scratch_repaired_v4_stratified_75k", ds="combined"),
    dict(short="SwinT-pt",    dir="SwinT_combined_rle_f1000_repaired_v4_stratified_75k", ds="combined"),
    dict(short="SwinB-pt",    dir="SwinB_combined_rle_f1000_repaired_v4_stratified_75k", ds="combined"),
    dict(short="SwinB-sc",    dir="SwinB_combined_rle_f1000_rgb_scratch_repaired_v4_stratified_75k", ds="combined"),
    dict(short="SwinB-RGBDp", dir="SwinB_combined_rle_f1000_rgbd_repaired_v4_stratified_75k", ds="combined"),
    dict(short="SwinB-RGBDs", dir="SwinB_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", ds="combined"),
    dict(short="SwinB-DBH",   dir="SwinB_combined_rle_f1000_rgbd_dbh_repaired_v4_stratified_75k", ds="combined"),
    dict(short="SwinB-PL",    dir="SwinB_pl_rle_f1000_repaired_v4_stratified_75k", ds="plantations"),
    dict(short="SwinL-pt",    dir="SwinL_combined_rle_f1000_repaired_v4_stratified_75k", ds="combined"),
    dict(short="FocalNetL-sc",       dir="FocalNet_L_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", ds="combined"),
    dict(short="FocalNetL-RGBDp",    dir="FocalNet_L_combined_rle_f1000_rgbd_repaired_v4_stratified_75k", ds="combined"),
    dict(short="FocalNetL-PL-RGBDs", dir="FocalNet_L_pl_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", ds="plantations"),
    dict(short="FocalNetL-RF-RGBDs", dir="FocalNet_L_rf_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", ds="rainforests"),
    dict(short="FocalNetL-pt",     dir="FocalNet_L_combined_rle_f1000_repaired_v4_stratified_75k", ds="combined"),
    dict(short="FocalNetL-sc-rgb", dir="FocalNet_L_combined_rle_f1000_rgb_scratch_repaired_v4_stratified_75k", ds="combined"),
    dict(short="FocalNetL-PL-pt",  dir="FocalNet_L_pl_rle_f1000_repaired_v4_stratified_75k", ds="plantations"),
    dict(short="FocalNetB-pt",     dir="FocalNet_B_combined_rle_f1000_repaired_v4_stratified_75k_lr1e4", ds="combined"),
    dict(short="FocalNetT-pt",     dir="FocalNet_T_combined_rle_f1000_repaired_v4_stratified_75k_lr1e4", ds="combined"),
    dict(short="FocalNetL-mosaic",     dir="FocalNet_L_combined_rgbd_scratch_v4_stratified_mosaic_noobj05_20k", ds="combined"),
    dict(short="FocalNetL-classweight", dir="FocalNet_L_combined_rgbd_scratch_v4_stratified_mosaic_classweight_20k", ds="combined"),
    dict(short="FocalNetL-control",     dir="FocalNet_L_combined_rgbd_scratch_v4_stratified_mosaic_control20k", ds="combined"),
    dict(short="FocalNetL-ss-noobj",       dir="FocalNet_L_combined_rgbd_scratch_v4_stratified_singleshot75k_mosaic_noobj", ds="combined"),
    dict(short="FocalNetL-ss-classweight", dir="FocalNet_L_combined_rgbd_scratch_v4_stratified_singleshot75k_mosaic_noobj_classweight", ds="combined"),
]


def main():
    results = {}
    for i, m in enumerate(MODELS):
        pred_file = OUTPUT_ROOT / m["dir"] / "eval_test_v4" / "inference" / "coco_instances_results.json"
        if not pred_file.exists():
            print(f"[{i+1}/{len(MODELS)}] SKIP {m['short']} — prediksi tidak ada")
            continue
        gt_file = GT_FILES[m["ds"]]
        print(f"[{i+1}/{len(MODELS)}] {m['short']} ({m['ds']}) ...", flush=True)

        size_ap = compute_size_stratified_ap(gt_file, pred_file)
        per_class_ap50, confusion = compute_per_class_ap_and_confusion(gt_file, pred_file)
        boundary = compute_boundary_ap(gt_file, pred_file)

        results[m["short"]] = {
            "dir": m["dir"], "test_ds": m["ds"],
            "size_stratified": size_ap,
            "per_class_AP50": per_class_ap50,
            "confusion_matrix": confusion,
            "boundary": boundary,
        }
        print(f"    AR100={size_ap['AR100']:.2f}  APs/m/l={size_ap['APs']:.1f}/{size_ap['APm']:.1f}/{size_ap['APl']:.1f}  "
              f"BoundaryAP50={boundary['boundary_AP50']:.2f} (vs mask AP50={size_ap['AP50']:.2f})", flush=True)

    out_path = REPORTS_DIR / "eval_literature_metrics_v4.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
