#!/usr/bin/env python3
"""occlusion_robustness.py — robustness terhadap oklusi, per instance dan per scene.

Proposal menjanjikan presisi pada scene dengan lebih dari 30% batang tertutup. Yang
tersedia sebelumnya hanya stratifikasi menurut jarak dan ukuran, bukan rasio oklusi.
SPREAD menyediakan mask instance sehingga rasio itu dapat dihitung tanpa training ulang.

Definisi rasio oklusi. Untuk instance i, ambil convex hull dari mask-nya. Wilayah di dalam
hull yang BUKAN milik i, tetapi dimiliki instance lain yang lebih DEKAT ke kamera, dihitung
sebagai wilayah tertutup:

    oklusi(i) = |hull(i) \\ mask(i) yang ditempati instance j dgn depth(j) < depth(i)|
                --------------------------------------------------------------------
                                        |hull(i)|

Convex hull dipakai sebagai perkiraan bentuk utuh pohon. Urutan depth memakai median
depth pada mask, sehingga "tertutup oleh" punya arti fisik, bukan sekadar bertumpang tindih.

Presisi hanya terdefinisi pada tingkat Scene, karena prediksi yang tidak berpasangan dengan
GT mana pun tidak memiliki rasio oklusi sendiri. Karena itu recall dilaporkan per bin oklusi
instance, sedangkan presisi dan F1 dilaporkan per kelompok scene.

    python scripts/occlusion_robustness.py --split val --pred <coco_instances_results.json>
"""
import argparse
import json
import os
import sys
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

GT_DIR = PROJ / "reports" / "dbh_eval"
BIN = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 1.01)]

# Pakis (Fern) dan Liana (LeechVine), penutup yang disebut eksplisit di proposal.
# Indeks 0-based mengikuti category_id dataset.
CLUTTER = {9, 10}


def label_bin(lo, hi):
    return f"{lo*100:.0f}-{min(hi,1.0)*100:.0f}%"


def hitung_oklusi(split):
    """Rasio oklusi per instance GT, disimpan ke berkas agar tidak dihitung ulang."""
    import cv2
    from pycocotools import mask as mu
    from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import _read_pfm
    from eval_joint_trunkroi_dbh import DEPTH_DIR

    rows = json.load(open(GT_DIR / f"gt_depth_{split}.json"))["table"]
    per_img = defaultdict(list)
    for i, r in enumerate(rows):
        per_img[r["image_id"]].append(i)

    # nama berkas gambar diperlukan untuk membaca peta depth
    from detectron2.data import DatasetCatalog
    from register_combined import register_all_combined
    register_all_combined(os.environ["COMBINED_ROOT"])
    DS = "combined_inst_rle_f1000_repaired_v4_stratified"
    stem = {rec["image_id"]: Path(rec["file_name"]).stem
            for rec in DatasetCatalog.get(f"{DS}_{split}")}

    okl = [None] * len(rows)
    okl_clutter = [None] * len(rows)
    for n, (img, idxs) in enumerate(per_img.items(), 1):
        if n % 200 == 0:
            print(f"  {n}/{len(per_img)} gambar", flush=True)
        pfm = DEPTH_DIR / f"{stem[img]}.pfm"
        if not pfm.exists():
            continue
        dep = _read_pfm(str(pfm))
        masks = [mu.decode(rows[i]["rle"]).astype(bool) for i in idxs]
        if dep.shape != masks[0].shape:
            dep = cv2.resize(dep, (masks[0].shape[1], masks[0].shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        # depth wakil tiap instance; nilai sentinel langit dibuang
        med = []
        for m in masks:
            v = dep[m]
            v = v[np.isfinite(v) & (v > 0.1) & (v < 200)]
            med.append(float(np.median(v)) if v.size else np.inf)

        for a, ia in enumerate(idxs):
            m = masks[a]
            pts = cv2.findNonZero(m.astype(np.uint8))
            if pts is None or len(pts) < 3:
                continue
            # np.zeros baru, BUKAN zeros_like dari array bool: fillConvexPoly menolak
            # layout hasil zeros_like dengan dtype yang ditimpa.
            hull = np.zeros(m.shape, dtype=np.uint8)
            cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
            hull = hull.astype(bool)
            luar = hull & ~m                      # bagian hull yang bukan milik instance ini
            if luar.sum() == 0:
                okl[ia] = 0.0
                continue
            tertutup = np.zeros_like(m, dtype=bool)
            oleh_clutter = np.zeros_like(m, dtype=bool)
            for b, ib in enumerate(idxs):
                if b == a or med[b] >= med[a]:    # hanya yang LEBIH DEKAT
                    continue
                tumpang = luar & masks[b]
                tertutup |= tumpang
                # Proposal mendefinisikan oklusi sebagai tertutup Pakis atau Liana,
                # bukan tertutup apa pun. Keduanya dicatat terpisah agar definisi
                # proposal dapat dilaporkan apa adanya di samping definisi umum.
                if rows[ib]["cat"] in CLUTTER:
                    oleh_clutter |= tumpang
            luas = float(hull.sum())
            okl[ia] = float(tertutup.sum()) / luas
            okl_clutter[ia] = float(oleh_clutter.sum()) / luas

    p = GT_DIR / f"gt_occlusion_{split}.json"
    json.dump({"split": split, "okl": okl, "okl_clutter": okl_clutter}, open(p, "w"))
    ada = sum(1 for x in okl if x is not None)
    print(f"\ntersimpan: {p}")
    print(f"  {ada} dari {len(okl)} instance punya rasio oklusi")
    v = np.array([x for x in okl if x is not None])
    print(f"  rasio: median {np.median(v):.3f}, p90 {np.percentile(v,90):.3f}, "
          f"maks {v.max():.3f}")
    print(f"  instance dgn oklusi >30%: {(v>0.30).sum()} ({(v>0.30).mean()*100:.1f}%)")
    c = np.array([x for x in okl_clutter if x is not None])
    print(f"  oklusi khusus Pakis/Liana: median {np.median(c):.3f}, "
          f">30% pada {(c>0.30).sum()} instance ({(c>0.30).mean()*100:.1f}%)")


def analisis(split, pred_path, label, iou_thr, score_thr):
    from pycocotools import mask as mu
    rows = json.load(open(GT_DIR / f"gt_depth_{split}.json"))["table"]
    okl = json.load(open(GT_DIR / f"gt_occlusion_{split}.json"))["okl"]
    preds = [p for p in json.load(open(pred_path)) if p.get("score", 1.0) >= score_thr]
    print(f"{label}: {len(preds)} prediksi (score>={score_thr}), {len(rows)} GT\n")

    pi, gi = defaultdict(list), defaultdict(list)
    for p in preds:
        pi[p["image_id"]].append(p)
    for i, g in enumerate(rows):
        gi[g["image_id"]].append(i)

    hit = [False] * len(rows)
    tp_img, fp_img = defaultdict(int), defaultdict(int)
    for img, idxs in gi.items():
        ps = sorted(pi.get(img, []), key=lambda x: -x.get("score", 0))
        if not ps:
            continue
        ious = mu.iou([p["segmentation"] for p in ps], [rows[i]["rle"] for i in idxs],
                      [0] * len(idxs))
        used = set()
        for a in range(len(ps)):
            best, bi = iou_thr, None
            for b in range(len(idxs)):
                if b in used:
                    continue
                if ious[a][b] >= best:
                    best, bi = ious[a][b], b
            if bi is not None:
                used.add(bi); hit[idxs[bi]] = True; tp_img[img] += 1
            else:
                fp_img[img] += 1
    for img in pi:
        if img not in gi:
            fp_img[img] += len(pi[img])

    print("=== RECALL PER BIN OKLUSI (tingkat instance) ===")
    print(f"  {'oklusi':>10} {'N GT':>7} {'recall':>9}")
    for lo, hi in BIN:
        s = [i for i in range(len(rows)) if okl[i] is not None and lo <= okl[i] < hi]
        if not s:
            continue
        r = 100 * sum(hit[i] for i in s) / len(s)
        print(f"  {label_bin(lo,hi):>10} {len(s):>7} {r:>8.1f}%")
    tanpa = [i for i in range(len(rows)) if okl[i] is None]
    if tanpa:
        print(f"  {'tanpa nilai':>10} {len(tanpa):>7} "
              f"{100*sum(hit[i] for i in tanpa)/len(tanpa):>8.1f}%")

    print("\n=== PRESISI DAN F1 PER KELOMPOK Scene ===")
    print("  Scene dikelompokkan menurut porsi batangnya yang teroklusi lebih dari 30%.")
    frac = {}
    for img, idxs in gi.items():
        v = [okl[i] for i in idxs if okl[i] is not None]
        if v:
            frac[img] = sum(1 for x in v if x > 0.30) / len(v)
    grup = [("0%", 0.0, 1e-9), ("0-15%", 1e-9, 0.15), ("15-30%", 0.15, 0.30),
            (">30%", 0.30, 1.01)]
    print(f"  {'scene':>10} {'n img':>7} {'TP':>6} {'FP':>6} {'FN':>6} "
          f"{'presisi':>9} {'recall':>8} {'F1':>7}")
    for nm, lo, hi in grup:
        imgs = [i for i, f in frac.items() if lo <= f < hi] if nm != "0%" \
            else [i for i, f in frac.items() if f == 0.0]
        if not imgs:
            continue
        tp = sum(tp_img[i] for i in imgs); fp = sum(fp_img[i] for i in imgs)
        fn = sum(1 for i in imgs for j in gi[i] if not hit[j])
        pr = 100 * tp / (tp + fp) if tp + fp else 0
        rc = 100 * tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        print(f"  {nm:>10} {len(imgs):>7} {tp:>6} {fp:>6} {fn:>6} "
              f"{pr:>8.1f}% {rc:>7.1f}% {f1:>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--compute", action="store_true", help="hitung rasio oklusi dulu")
    ap.add_argument("--pred")
    ap.add_argument("--label", default="model")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--score", type=float, default=0.30)
    a = ap.parse_args()
    if a.compute:
        hitung_oklusi(a.split)
    if a.pred:
        analisis(a.split, a.pred, a.label, a.iou, a.score)


if __name__ == "__main__":
    main()
