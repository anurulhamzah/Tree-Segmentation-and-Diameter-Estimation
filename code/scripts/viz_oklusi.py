#!/usr/bin/env python3
"""viz_oklusi.py — gambar satu contoh nyata perhitungan rasio oklusi.

Menghasilkan figur tiga panel untuk satu instance GT terpilih:

  1. potongan image asli, batas instance target ditandai
  2. mask target dan convex hull-nya
  3. wilayah hull di luar mask, dipisah antara yang tertutup instance lebih dekat
     (masuk hitungan) dan yang memang cekungan bentuk pohonnya sendiri (tidak masuk)

Angka rasio pada judul dihitung ulang di sini dengan rumus yang sama persis seperti
`occlusion_robustness.py`, jadi kalau keduanya berbeda berarti ada yang salah, bukan
sekadar beda pembulatan.

    python scripts/viz_oklusi.py --split val --target 0.5
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
OUT = PROJ / "reports" / "stratified_v4" / "oklusi_contoh.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--target", type=float, default=0.30,
                    help="rasio oklusi yang dicari, contoh terdekat yang dipakai")
    # Contoh pertama yang dirender kebetulan pohon berdaun menjari: mask hanya 17% dari
    # hull-nya, sehingga hull membentang jauh melewati tubuh pohon dan gambarnya sulit
    # dibaca sebagai ilustrasi definisi. Soliditas menyaring kasus seperti itu.
    ap.add_argument("--min-soliditas", type=float, default=0.45,
                    help="batas bawah luas mask dibagi luas hull")
    ap.add_argument("--n", type=int, default=4, help="berapa kandidat dirender")
    a = ap.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from pycocotools import mask as mu
    from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import _read_pfm
    from eval_joint_trunkroi_dbh import DEPTH_DIR
    from detectron2.data import DatasetCatalog
    from register_combined import register_all_combined

    rows = json.load(open(GT_DIR / f"gt_depth_{a.split}.json"))["table"]
    okl = json.load(open(GT_DIR / f"gt_occlusion_{a.split}.json"))["okl"]

    register_all_combined(os.environ["COMBINED_ROOT"])
    DS = "combined_inst_rle_f1000_repaired_v4_stratified"
    rec = {r["image_id"]: r for r in DatasetCatalog.get(f"{DS}_{a.split}")}

    # Kandidat disaring tiga kali: rasio dekat target, mask cukup besar untuk terbaca,
    # dan bentuknya cukup padat sehingga hull-nya memeluk mask alih-alih membentang jauh.
    kand = []
    for i, v in enumerate(okl):
        if v is None or rows[i].get("rle") is None:
            continue
        if abs(v - a.target) > 0.12:
            continue
        m = mu.decode(rows[i]["rle"]).astype(bool)
        if m.sum() < 1500:
            continue
        pts = cv2.findNonZero(m.astype(np.uint8))
        if pts is None or len(pts) < 3:
            continue
        hl = np.zeros(m.shape, dtype=np.uint8)
        cv2.fillConvexPoly(hl, cv2.convexHull(pts), 1)
        sol = m.sum() / float(hl.sum())
        if sol < a.min_soliditas:
            continue
        ys, xs = np.where(hl.astype(bool))
        rasio_sisi = (ys.max() - ys.min() + 1) / float(xs.max() - xs.min() + 1)
        if not 0.45 < rasio_sisi < 2.4:
            continue
        kand.append((i, v, sol, m.sum()))
    if not kand:
        sys.exit("tidak ada kandidat yang lolos filter")
    kand.sort(key=lambda t: (-t[2], -t[3]))
    print(f"{len(kand)} kandidat lolos; {min(a.n, len(kand))} teratas dirender")
    for k, (i, v, sol, luas) in enumerate(kand[:a.n], 1):
        print(f"  {k}. instance {i}: rasio {v:.3f}, soliditas {sol:.2f}, mask {luas} px")
        gambar(i, k, rows, rec, okl, cv2, plt, Patch, mu, _read_pfm, DEPTH_DIR, np)
    return


def gambar(ia, urut, rows, rec, okl, cv2, plt, Patch, mu, _read_pfm, DEPTH_DIR, np):
    rasio = okl[ia]
    img_id = rows[ia]["image_id"]
    idxs = [i for i, r in enumerate(rows) if r["image_id"] == img_id]
    masks = [mu.decode(rows[i]["rle"]).astype(bool) for i in idxs]
    stem = Path(rec[img_id]["file_name"]).stem
    dep = _read_pfm(str(DEPTH_DIR / f"{stem}.pfm"))
    if dep.shape != masks[0].shape:
        dep = cv2.resize(dep, (masks[0].shape[1], masks[0].shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    med = []
    for m in masks:
        v = dep[m]
        v = v[np.isfinite(v) & (v > 0.1) & (v < 200)]
        med.append(float(np.median(v)) if v.size else np.inf)

    p = idxs.index(ia)
    m = masks[p]
    hull = np.zeros(m.shape, dtype=np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(m.astype(np.uint8))), 1)
    hull = hull.astype(bool)
    luar = hull & ~m
    tertutup = np.zeros_like(m)
    for b, ib in enumerate(idxs):
        if b == p or med[b] >= med[p]:
            continue
        tertutup |= luar & masks[b]
    ulang = tertutup.sum() / float(hull.sum())
    print(f"rasio dihitung ulang     {ulang:.3f}  (selisih {abs(ulang-rasio):.4f})")
    print(f"luas hull {hull.sum()}, mask {m.sum()}, hull di luar mask {luar.sum()}, "
          f"tertutup {tertutup.sum()}")

    rgb = cv2.cvtColor(cv2.imread(rec[img_id]["file_name"]), cv2.COLOR_BGR2RGB)
    ys, xs = np.where(hull)
    # Padding proporsional, bukan tetap 12 px: dengan padding tetap, instance kecil
    # tampil sesak sedangkan instance besar tampil terlalu longgar.
    pad = max(10, int(0.12 * max(ys.max() - ys.min(), xs.max() - xs.min())))
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad)
    pot = lambda A: A[y0:y1, x0:x1]

    # Instance yang benar-benar menutupi, untuk digambar batasnya di panel 1.
    penutup = np.zeros_like(m)
    for b, ib in enumerate(idxs):
        if b == p or med[b] >= med[p]:
            continue
        if (luar & masks[b]).sum() > 0:
            penutup |= masks[b]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.2))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].imshow(pot(rgb))
    axes[0].contour(pot(m), levels=[0.5], colors="#c9591f", linewidths=2.2)
    axes[0].contour(pot(penutup), levels=[0.5], colors="#1d6b4a", linewidths=1.4)
    axes[0].set_title("1. Image asli\njingga: mask GT target, hijau: instance yang lebih dekat",
                      fontsize=9.5)

    kanvas = pot(rgb).copy().astype(float) * 0.30 + 175 * 0.70
    kanvas[pot(m)] = [201, 89, 31]
    axes[1].imshow(kanvas.astype(np.uint8))
    axes[1].contour(pot(hull), levels=[0.5], colors="#14130f",
                    linewidths=2.0, linestyles="--")
    axes[1].set_title(f"2. Mask ({m.sum():,} px) dan convex hull ({hull.sum():,} px)\n"
                      f"soliditas {m.sum()/hull.sum():.2f}".replace(",", "."), fontsize=9.5)

    vis = np.full(pot(m).shape + (3,), 250, dtype=np.uint8)
    vis[pot(m)] = [201, 89, 31]
    vis[pot(luar & ~tertutup)] = [214, 210, 198]
    vis[pot(tertutup)] = [29, 107, 74]
    axes[2].imshow(vis)
    axes[2].set_title(f"3. Rasio oklusi = {tertutup.sum():,} / {hull.sum():,} = {ulang:.3f}"
                      .replace(",", "."), fontsize=9.5)
    axes[2].legend(handles=[
        Patch(facecolor="#c9591f", label="mask target"),
        Patch(facecolor="#1d6b4a", label="tertutup instance lebih dekat, DIHITUNG"),
        Patch(facecolor="#d6d2c6", label="cekungan bentuk sendiri, TIDAK dihitung")],
        loc="upper center", bbox_to_anchor=(0.5, -0.03), fontsize=8.5, frameon=False)

    fig.tight_layout()
    out = OUT.with_name(f"oklusi_contoh_{urut}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"     tersimpan: {out.name}  (hull {hull.sum()}, tertutup {tertutup.sum()}, "
          f"selisih vs tersimpan {abs(ulang-rasio):.4f})")


if __name__ == "__main__":
    main()
