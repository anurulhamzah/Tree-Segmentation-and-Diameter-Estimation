#!/usr/bin/env python3
"""
compute_seg_full_metrics_8configs.py — Metrik segmentasi LENGKAP (val split) untuk
kedelapan konfigurasi yang dibandingkan di Bagian 1 laporan (dengan/tanpa loss DBH).

Kenapa dibuat. Tabel "Metrik segmentasi lengkap" semula hanya memuat AP/AP50/AP75/APs/APm/APl
dari metrics.json (VAL split, periodic eval saat training). Itu bukan benar-benar lengkap:
- Keluarga AR (AR1/AR10/AR100/ARs/ARm/ARl) adalah pendamping standar AP di setiap laporan
  COCO-style, TIDAK pernah dicatat metrics.json (di-drop oleh training hook), tapi BISA
  dihitung ulang murah dari prediksi mentah yang sudah tersimpan (tanpa forward pass baru).
- Precision/Recall/F1 pada satu threshold skor tetap TIDAK PERNAH dihasilkan oleh COCOeval
  standar (AP/AR keduanya threshold-agnostic), padahal itu yang mencerminkan performa pada
  titik operasi yang benar-benar dipakai. Proyek ini sudah punya preseden untuk metrik itu
  (scripts/compute_maskrcnn_f1.py, scores>=0.35, IoU=0.50), dipakai ulang persis di sini.

Sumber: coco_instances_results.json tiap run (prediksi mentah, VAL split, sudah ada dari
eval periodik saat training) + instances_val.json (ground truth). Tidak ada forward pass GPU baru.

⚠️ VERIFIKASI SPLIT (31 Jul): sempat dicoba pakai instances_test.json dan GAGAL total
(overlap image_id = 0 dari 852). Dicek ulang: kedelapan config overlap 852/852 dengan
instances_val.json, 0/843 dengan instances_test.json. Cache "inference/" TERNYATA VAL split
di seluruh 8 config (konsisten dgn DATASETS.TEST=..._val di config.yaml, BUKAN nama foldernya
yang menyesatkan). Jangan asumsikan nama folder "inference/" = held-out test tanpa mengecek
overlap image_id-nya lebih dulu. Split VAL ini justru KONSISTEN dgn AP50 yang sudah dipakai
di seluruh Bagian 1 (val segm AP50), jadi tabel ini melengkapi tabel lama, bukan menggantinya
dgn split berbeda.

Jalankan:
    python scripts/compute_seg_full_metrics_8configs.py
"""
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_FILE = ("/scratch2/pr65/anur0018/tree_classification/data/combined/"
           "annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_val.json")
OUTPUT_ROOT = Path("/scratch2/pr65/anur0018/maskdino_output")
REPORT_DIR = Path("/scratch2/pr65/anur0018/tree_classification/reports")
SCORE_THR = 0.35
IOU_THR = 0.50

CONFIGS = [
    ("Plain155k", "FocalNet_L_combined_rgbd_scratch_v4_stratified_plain155k"),
    ("Plain135k", "FocalNet_L_combined_rgbd_scratch_v4_stratified_plain135k"),
    ("Plain_ext20k", "FocalNet_L_combined_rgbd_scratch_v4_stratified_plain_ext20k"),
    ("Joint 11-sp, basis plain135k", "FocalNet_L_trunk_roi_dbh_hybrid_joint_from135k_11species_20k"),
    ("Joint 7-sp, basis plain135k", "FocalNet_L_trunk_roi_dbh_hybrid_joint_from135k_7species_20k"),
    ("Joint 11-sp, basis plain_ext20k", "FocalNet_L_trunk_roi_dbh_hybrid_joint_plainext_20k"),
    ("Joint 7-sp, basis plain_ext20k", "FocalNet_L_trunk_roi_dbh_hybrid_joint_7species_20k"),
    ("Joint 11-sp, from-scratch 155k", "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k"),
]


def ap_ar_family(coco_gt, dt_path):
    """AP, AP50, AP75, APs/m/l, AR1/10/100, ARs/m/l -- threshold-agnostic, standar COCO."""
    dt = json.load(open(dt_path))
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dt)
        E = COCOeval(coco_gt, coco_dt, "segm")
        E.evaluate(); E.accumulate(); E.summarize()
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl",
            "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    return {k: round(float(v) * 100, 2) for k, v in zip(keys, E.stats)}


def precision_recall_f1(coco_gt, dt_path, score_thr=SCORE_THR, iou_thr=IOU_THR):
    """P/R/F1 pada satu threshold skor tetap -- metrik operasional, bukan threshold-agnostic.
    Metodologi identik scripts/compute_maskrcnn_f1.py."""
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
    for label, out_dir in CONFIGS:
        dt_path = OUTPUT_ROOT / out_dir / "inference" / "coco_instances_results.json"
        if not dt_path.exists():
            print(f"  {label:<34} TIDAK ADA prediksi tersimpan, dilewati")
            continue
        fam = ap_ar_family(coco_gt, dt_path)
        prf = precision_recall_f1(coco_gt, dt_path)
        row = {**fam, **prf, "output_dir": out_dir}
        results[label] = row
        print(f"  {label:<34} AP50={fam['AP50']:5.2f}  AP75={fam['AP75']:5.2f}  "
              f"AR100={fam['AR100']:5.2f}  P@.35={prf['Precision']:5.2f}  "
              f"R@.35={prf['Recall']:5.2f}  F1@.35={prf['F1']:5.2f}")

    out_path = REPORT_DIR / "seg_full_metrics_8configs_val.json"
    with open(out_path, "w") as f:
        json.dump({"split": "val", "score_thr": SCORE_THR, "iou_thr_prf": IOU_THR,
                   "gt_file": GT_FILE, "results": results}, f, indent=2)
    print(f"\nTersimpan: {out_path}")


if __name__ == "__main__":
    main()
