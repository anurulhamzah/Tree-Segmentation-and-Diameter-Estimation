#!/usr/bin/env python3
"""recall_per_jarak.py — recall deteksi pohon dipecah menurut jarak kamera.

Menjawab: pada jarak berapa model kehilangan pohon, dan apakah dua model berbeda di
jarak yang berbeda. Ini melengkapi angka agregat yang sudah ada, yang menunjukkan 41,6%
GT tidak terdeteksi tanpa memberi tahu di mana kehilangan itu menumpuk.

Jarak tiap instance GT memakai `d_trunk`, depth pada baris batang setinggi 1,3 m,
dihitung lewat `_find_row_1p3m()` yang hanya bergantung pada mask GT, peta depth, dan
intrinsik kamera. TIDAK bergantung weights model, jadi bin berisi pohon yang sama persis
untuk model mana pun dan selisih recall antar model murni beda model. Replikasi model-free
ini sudah divalidasi terhadap berkas per-instance eval resmi: sebaran per bin cocok dalam
1 poin persen di kelima bin, p50 5,49 lawan 5,62 m.

Recall, bukan AP, yang dipecah per bin. Recall terdefinisi bersih (dari pohon GT di bin
ini, berapa yang ketemu), sedangkan prediksi false positive tidak punya depth GT
sehingga tidak bisa dimasukkan ke bin mana pun.

    # bangun tabel jarak GT sekali, lalu pakai berkali-kali
    python scripts/recall_per_jarak.py --dump-gt --split val
    python scripts/recall_per_jarak.py --split val \
        --pred <output_dir>/inference/coco_instances_results.json --label "joint 11-sp"
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJ = Path("/scratch2/pr65/anur0018/tree_classification")
sys.path.insert(0, str(PROJ / "scripts"))
sys.path.insert(0, "/scratch2/pr65/anur0018/MaskDINO/MaskDINO")
for _k, _v in [("COMBINED_ROOT", PROJ / "data" / "combined"),
               ("RAINFORESTS_ROOT", PROJ / "data" / "rainforests"),
               ("PLANTATIONS_ROOT", PROJ / "data" / "plantations")]:
    os.environ.setdefault(_k, str(_v))

DS = "combined_inst_rle_f1000_repaired_v4_stratified"
GT_DIR = PROJ / "reports" / "dbh_eval"
BIN = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 1e9)]


def label_bin(lo, hi):
    return f"{lo}-{hi} m" if hi < 1e8 else f">{lo} m"


def dump_gt(split):
    """Tabel GT: jarak, kelas, dan RLE tiap instance. Instance yang geometrinya gagal
    TETAP dicatat dengan d=None — itu pohon sungguhan yang seharusnya terdeteksi, dan
    membuangnya akan membuat recall tampak lebih baik daripada kenyataan."""
    import torch
    from detectron2.data import DatasetCatalog
    from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import (
        _read_pfm, _normalize_depth)
    from eval_joint_trunkroi_dbh import DEPTH_DIR, decode_mask
    import trunk_roi_dbh_head_hybrid as H
    from register_combined import register_all_combined
    from pycocotools import mask as mask_util

    register_all_combined(os.environ["COMBINED_ROOT"])
    recs = DatasetCatalog.get(f"{DS}_{split}")
    print(f"split {split}: {len(recs)} gambar", flush=True)

    out, t0, n_nodepth = [], time.time(), 0
    for i, rec in enumerate(recs):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(recs)}  ({time.time()-t0:.0f}s)", flush=True)
        Hh, W = rec["height"], rec["width"]
        stem = Path(rec["file_name"]).stem
        pfm = DEPTH_DIR / f"{stem}.pfm"
        depth01 = None
        if pfm.exists():
            d = _read_pfm(str(pfm))
            if d.shape != (Hh, W):
                import cv2
                d = cv2.resize(d, (W, Hh), interpolation=cv2.INTER_NEAREST)
            depth01 = torch.as_tensor(_normalize_depth(d), dtype=torch.float32)
        else:
            n_nodepth += 1

        for k, ann in enumerate(rec.get("annotations", [])):
            m = decode_mask(ann, Hh, W)
            dt = None
            if depth01 is not None:
                scale = Hh / H._ORIG_H
                res = H._find_row_1p3m(torch.as_tensor(m).bool(),
                                       H._denorm_depth(depth01),
                                       H._ORIG_CY * scale, H._ORIG_FY * scale, tol_wh=0.5)
                if res[0] is not None:
                    dt = round(float(res[2]), 4)
            rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            out.append(dict(image_id=rec["image_id"], idx=k,
                            cat=int(ann["category_id"]), d=dt, rle=rle))

    p = GT_DIR / f"gt_depth_{split}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(split=split, n_images=len(recs), n_gt=len(out), table=out), open(p, "w"))
    ada = sum(1 for r in out if r["d"] is not None)
    print(f"\ntersimpan: {p}")
    print(f"  {len(out)} GT instance, {ada} punya jarak ({100*ada/len(out):.1f}%), "
          f"{len(out)-ada} geometri gagal, {n_nodepth} gambar tanpa depth")


def recall(split, pred_path, label, iou_thr, score_thr):
    from pycocotools import mask as mask_util

    gt = json.load(open(GT_DIR / f"gt_depth_{split}.json"))["table"]
    preds = [p for p in json.load(open(pred_path)) if p.get("score", 1.0) >= score_thr]
    print(f"{label}: {len(preds)} prediksi (score>={score_thr}), {len(gt)} GT")

    per_img_pred = defaultdict(list)
    for p in preds:
        per_img_pred[p["image_id"]].append(p)
    per_img_gt = defaultdict(list)
    for g in gt:
        per_img_gt[g["image_id"]].append(g)

    for g in gt:
        g["hit"] = False
        g["iou"] = None
        g["benar_sp"] = None
    for img, gs in per_img_gt.items():
        ps = sorted(per_img_pred.get(img, []), key=lambda x: -x.get("score", 0))
        if not ps:
            continue
        ious = mask_util.iou([p["segmentation"] for p in ps],
                             [g["rle"] for g in gs], [0] * len(gs))
        dipakai = set()
        for pi in range(len(ps)):
            best, bi = iou_thr, None
            for gi in range(len(gs)):
                if gi in dipakai:
                    continue
                if ious[pi][gi] >= best:
                    best, bi = ious[pi][gi], gi
            if bi is not None:
                dipakai.add(bi)
                gs[bi]["hit"] = True
                gs[bi]["iou"] = float(ious[pi][bi])
                # category_id prediksi 1-based, cat GT 0-based. Pencocokan sengaja
                # class-agnostic supaya recall memisahkan "tidak ketemu" dari
                # "ketemu tapi salah spesies"; keduanya beda masalah.
                gs[bi]["benar_sp"] = (ps[pi]["category_id"] == gs[bi]["cat"] + 1)

    def ringkas(s, nama):
        """recall class-agnostic, lalu mutu deteksi yang berhasil: akurasi spesies dan IoU."""
        if not s:
            return None
        h = [g for g in s if g["hit"]]
        rec = 100 * len(h) / len(s)
        sp = 100 * sum(g["benar_sp"] for g in h) / len(h) if h else float("nan")
        iou = 100 * sum(g["iou"] for g in h) / len(h) if h else float("nan")
        print(f"  {nama:>9} {len(s):>7} {rec:>8.1f}% {sp:>9.1f}% {iou:>8.1f}%")
        return dict(bin=nama, n=len(s), recall=rec, akurasi_sp=sp, iou=iou)

    print(f"\n  {'bin':>9} {'N GT':>7} {'recall':>9} {'sp benar':>10} {'IoU':>9}")
    print(f"  {'':>9} {'':>7} {'':>9} {'(dari yg terdeteksi)':>21}")
    hasil = [r for lo, hi in BIN
             if (r := ringkas([g for g in gt if g["d"] is not None and lo <= g["d"] < hi],
                              label_bin(lo, hi)))]
    ringkas([g for g in gt if g["d"] is None], "tanpa d")
    ringkas(gt, "SEMUA")
    return hasil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--dump-gt", action="store_true")
    ap.add_argument("--pred")
    ap.add_argument("--label", default="model")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--score", type=float, default=0.5,
                    help="threshold skor; 0,5 mengikuti titik operasi yang dipakai laporan")
    a = ap.parse_args()
    if a.dump_gt:
        dump_gt(a.split)
    if a.pred:
        recall(a.split, a.pred, a.label, a.iou, a.score)


if __name__ == "__main__":
    main()
