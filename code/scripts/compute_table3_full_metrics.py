#!/usr/bin/env python3
"""
compute_table3_full_metrics.py — Metrik segmentasi LENGKAP (val split) untuk 6 backbone
MaskDINO + referensi Mask R-CNN, RGB-D from-scratch 75k, dipakai di tab:backbone tesis.

Table 3 semula cuma AP50. Proposal riset (research_proposal.html Bagian 8) sudah eksplisit
menetapkan hierarki: AP@50:95 dan AP@50 = primary, AP@75 dan AR@100 = secondary. Skrip ini
menghitung keluarga AP/AR itu murah dari prediksi tersimpan (coco_instances_results.json,
VAL split, hasil eval periodik saat training) -- TIDAK ADA forward pass GPU baru.

Pola & fungsi diadaptasi dari scripts/compute_seg_full_metrics_8configs.py (preseden proyek
ini utk metrik AR-family, val split, sudah diverifikasi overlap image_id-nya benar).

Jalankan:
    python scripts/compute_table3_full_metrics.py
"""
import contextlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

GT_FILE = ("/scratch2/pr65/anur0018/tree_classification/data/combined/"
           "annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_val.json")
OUTPUT_ROOT = Path("/scratch2/pr65/anur0018/maskdino_output")
REPORT_DIR = Path("/scratch2/pr65/anur0018/tree_classification/reports")

# 0.30, bukan 0.35 yg dipakai compute_seg_full_metrics_8configs.py -- disamakan dgn threshold
# skor yang SUDAH established di naskah tesis ini sendiri (tab:distance/sec:res-distance,
# kedua gambar kualitatif §3.3), bukan preseden skrip lain yg konteksnya beda.
SCORE_THR = 0.30
IOU_THR = 0.50
# Boundary IoU (Cheng et al., CVPR 2021) -- dilasi 2% diagonal image, preseden proyek ini
# di scripts/compute_literature_metrics_v4.py (dipakai utk laporan lain, TEST split; di sini
# fungsinya sama tapi dijalankan pada prediksi VAL yang sama dgn kolom lain di tabel ini).
DILATION_RATIO = 0.02

# Backbone : (output dir, subpath ke inference/) -- RGB-D from-scratch 75k, sama persis dengan
# yang dipakai tab:depth/tab:backbone. Subpath default "inference" = checkpoint final; Swin-T dan
# Swin-L punya "best" AP50 di iterasi 69999 (bukan final 75000), jadi dipakai prediksi dari
# eval_best69999_val/inference/ (dieval ulang khusus utk konsistensi dgn AP50 yg sudah dipublikasikan
# di tab:depth/tab:backbone -- lihat compute_table3_full_metrics.py commit note).
CONFIGS = [
    ("FocalNet-L", "FocalNet_L_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "inference"),
    ("FocalNet-B", "FocalNet_B_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "inference"),
    ("Swin-T", "SwinT_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "eval_best69999_val/inference"),
    ("Swin-L", "SwinL_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "eval_best69999_val/inference"),
    ("Swin-B", "SwinB_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "inference"),
    ("ResNet-50 (MaskDINO)", "R50_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "inference"),
    ("Mask R-CNN, ResNet-50", "maskrcnn_R50_combined_rgbd_scratch_v4_stratified_75k", "inference"),
]


def ap_ar_family(coco_gt, dt_path, iou_type="segm"):
    """AP, AP50, AP75, APs/m/l, AR1/10/100, ARs/m/l -- threshold-agnostic, standar COCO.
    iou_type="bbox" utk box AP (predictions.json sudah punya field bbox utk semua deteksi)."""
    dt = json.load(open(dt_path))
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dt)
        E = COCOeval(coco_gt, coco_dt, iou_type)
        E.evaluate(); E.accumulate(); E.summarize()
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl",
            "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    return {k: round(float(v) * 100, 2) for k, v in zip(keys, E.stats)}


def per_class_ap50(coco_gt, dt_path):
    """AP50 per spesies -- rumus baku COCOeval (preseden compute_literature_metrics_v4.py)."""
    cat_ids = sorted(coco_gt.getCatIds())
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    dt = json.load(open(dt_path))
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dt)
        E = COCOeval(coco_gt, coco_dt, "segm")
        E.evaluate(); E.accumulate()
    out = {}
    for k_idx, cat_id in enumerate(cat_ids):
        prec = E.eval["precision"][0, :, k_idx, 0, 2]  # iou=0.5, area=all, maxDets=100
        prec = prec[prec > -1]
        out[cat_names[cat_id]] = round(float(np.mean(prec)) * 100, 2) if len(prec) else 0.0
    return out


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
        b = m
    rle_out = maskUtils.encode(np.asfortranarray(b))
    rle_out["counts"] = rle_out["counts"].decode("ascii")
    return rle_out


def boundary_ap(coco_gt, gt_dataset, dt_path):
    """Boundary AP (Cheng et al. CVPR 2021): AP dihitung ulang dgn GT & DT diganti versi
    boundary-only (mask dikurangi hasil erosi 2% diagonal image)."""
    imgs = {im["id"]: (im["height"], im["width"]) for im in gt_dataset["images"]}

    gt_b = json.loads(json.dumps(gt_dataset))  # deep copy murah via json
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

    return {"BoundaryAP": round(float(E.stats[0]) * 100, 2),
            "BoundaryAP50": round(float(E.stats[1]) * 100, 2),
            "BoundaryAP75": round(float(E.stats[2]) * 100, 2)}


def precision_recall_f1(coco_gt, dt_path, score_thr=SCORE_THR, iou_thr=IOU_THR):
    """P/R/F1 pada satu threshold skor tetap -- metrik operasional, bukan threshold-agnostic.
    Metodologi identik compute_seg_full_metrics_8configs.py, threshold disamakan ke 0.30
    (konvensi tesis ini, bukan 0.35 preseden skrip itu)."""
    preds = json.load(open(dt_path))
    preds_f = [p for p in preds if p["score"] >= score_thr]
    if not preds_f:
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0, "TP": 0, "FP": 0, "FN": 0}

    coco_dt = coco_gt.loadRes(preds_f)
    E = COCOeval(coco_gt, coco_dt, iouType="segm")
    E.params.iouThrs = np.array([iou_thr])
    E.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate(); E.accumulate()

    TP = FP = FN = 0
    for ev in E.evalImgs:
        if ev is None:
            continue
        for match, ign in zip(ev["dtMatches"][0], ev["dtIgnore"][0]):
            if ign:
                continue
            TP += 1 if match > 0 else 0
            FP += 0 if match > 0 else 1
        for match, ign in zip(ev["gtMatches"][0], ev["gtIgnore"]):
            if ign:
                continue
            FN += 1 if match == 0 else 0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"Precision": round(precision * 100, 2), "Recall": round(recall * 100, 2),
            "F1": round(f1 * 100, 2), "TP": TP, "FP": FP, "FN": FN}


def main():
    coco_gt = COCO(GT_FILE)
    print(f"GT: {len(coco_gt.getAnnIds())} anotasi, {len(coco_gt.getImgIds())} gambar (val split)\n")

    results = {}
    for label, out_dir, subpath in CONFIGS:
        dt_path = OUTPUT_ROOT / out_dir / subpath / "coco_instances_results.json"
        if not dt_path.exists():
            print(f"  {label:<24} TIDAK ADA prediksi tersimpan ({dt_path}), dilewati")
            continue
        # verifikasi overlap image_id dgn GT val (bukan asumsi dari nama folder)
        dt_raw = json.load(open(dt_path))
        dt_img_ids = {p["image_id"] for p in dt_raw}
        gt_img_ids = set(coco_gt.getImgIds())
        overlap = len(dt_img_ids & gt_img_ids)
        if overlap < 0.9 * len(gt_img_ids):
            print(f"  {label:<24} PERINGATAN overlap image_id cuma {overlap}/{len(gt_img_ids)}, dilewati")
            continue
        fam = ap_ar_family(coco_gt, dt_path, "segm")
        box = ap_ar_family(coco_gt, dt_path, "bbox")
        prf = precision_recall_f1(coco_gt, dt_path)
        pc50 = per_class_ap50(coco_gt, dt_path)
        bap = boundary_ap(coco_gt, coco_gt.dataset, dt_path)
        results[label] = {**fam, "box_AP": box["AP"], "box_AP50": box["AP50"], "box_AP75": box["AP75"],
                           **prf, **bap, "per_class_AP50": pc50,
                           "output_dir": out_dir, "overlap_img_ids": overlap}
        print(f"  {label:<24} AP={fam['AP']:5.2f}  AP50={fam['AP50']:5.2f}  AP75={fam['AP75']:5.2f}  "
              f"AR100={fam['AR100']:5.2f}  boxAP50={box['AP50']:5.2f}  BoundaryAP50={bap['BoundaryAP50']:5.2f}  "
              f"P@.30={prf['Precision']:5.2f}  R@.30={prf['Recall']:5.2f}  F1@.30={prf['F1']:5.2f}")

    out_path = REPORT_DIR / "table3_full_metrics_val.json"
    with open(out_path, "w") as f:
        json.dump({"split": "val", "gt_file": GT_FILE, "results": results}, f, indent=2)
    print(f"\nTersimpan: {out_path}")


if __name__ == "__main__":
    main()
