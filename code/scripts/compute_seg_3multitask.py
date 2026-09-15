#!/usr/bin/env python3
"""compute_seg_3multitask.py — metrik segmentasi ketiga run multi-task, untuk slide 3-5.

Ketiganya dibandingkan pada GT yang sama persis (combined val, 7.870 instance) dengan
metodologi yang sama persis, dengan mengimpor fungsi dari skrip yang sudah dipakai laporan
sebelumnya alih-alih menyalinnya. Kalau rumusnya kelak dikoreksi di sana, angka di sini ikut
terkoreksi, dan tidak ada versi kedua yang diam-diam menyimpang.

Sumber prediksi: `inference/coco_instances_results.json` tiap run. Itu cache eval periodik
saat training, dan splitnya VAL, bukan test; ini sudah diverifikasi 31 Juli dan dicatat di
compute_seg_full_metrics_8configs.py. Tidak ada forward pass GPU baru.

    python scripts/compute_seg_3multitask.py
"""
import json
import sys
from pathlib import Path

from pycocotools.coco import COCO

PROJ = Path("/scratch2/pr65/anur0018/tree_classification")
sys.path.insert(0, str(PROJ / "scripts"))

from compute_seg_full_metrics_8configs import (  # noqa: E402
    GT_FILE, ap_ar_family, precision_recall_f1)
from compute_literature_metrics_v4 import compute_per_class_ap_and_confusion  # noqa: E402

OUT = Path("/scratch2/pr65/anur0018/maskdino_output")
HASIL = PROJ / "reports/stratified_v4/seg_3multitask_val.json"

RUNS = [
    ("Multi-task 11-spesies", "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k"),
    ("Multi-task 7-spesies", "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_7species_155k"),
    ("Multi-task 7-spesies DBH_WEIGHT 5 ke 50",
     "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_7species_155k_wramp"),
]


def main():
    coco_gt = COCO(GT_FILE)
    print(f"GT: {len(coco_gt.getAnnIds())} anotasi, {len(coco_gt.getImgIds())} gambar (val)\n",
          flush=True)

    hasil = {}
    for label, d in RUNS:
        dt = OUT / d / "inference" / "coco_instances_results.json"
        if not dt.exists():
            print(f"  {label}: prediksi tidak ada, dilewati", flush=True)
            continue
        print(f"  {label} ...", flush=True)
        fam = ap_ar_family(coco_gt, dt)
        prf = precision_recall_f1(coco_gt, dt)
        per_kelas, konfusi = compute_per_class_ap_and_confusion(GT_FILE, dt)
        hasil[label] = {**fam, **prf, "per_class_AP50": per_kelas,
                        "confusion_matrix": konfusi, "output_dir": d}
        print(f"    AP50={fam['AP50']:5.2f}  AP={fam['AP']:5.2f}  AR100={fam['AR100']:5.2f}  "
              f"P={prf['Precision']:5.2f}  R={prf['Recall']:5.2f}  F1={prf['F1']:5.2f}",
              flush=True)

    HASIL.write_text(json.dumps({"split": "val", "gt_file": GT_FILE, "results": hasil},
                                indent=2), encoding="utf-8")
    print(f"\nTersimpan: {HASIL}", flush=True)


if __name__ == "__main__":
    main()
