#!/usr/bin/env python3
"""viz_oklusi_slide.py — figur dua contoh oklusi, dirender khusus untuk disematkan ke deck.

Berbeda dari viz_oklusi.py yang menyeleksi kandidat, skrip ini menerima id instance yang
sudah dipilih dan menyusun keduanya jadi SATU figur dua baris, dengan dpi rendah supaya
hasil base64-nya tidak membengkakkan deck.

Detectron2 sengaja TIDAK diimpor. `file_name` sudah absolut di dalam instances_val.json,
jadi DatasetCatalog tidak diperlukan dan waktu muatnya turun dari beberapa menit ke detik.

    python scripts/viz_oklusi_slide.py --ids 149 636
"""
import argparse
import json
from pathlib import Path

import numpy as np

PROJ = Path("/scratch2/pr65/anur0018/tree_classification")
GT_DIR = PROJ / "reports" / "dbh_eval"
DEPTH_DIR = PROJ / "data" / "combined" / "depth_pfm"
COCO = (PROJ / "data/combined/annotation_inst/"
        "filtered_rle_f1000_repaired_v4_stratified/instances_val.json")
OUT = PROJ / "reports" / "stratified_v4" / "oklusi_slide.png"


def read_pfm(path):
    with open(path, "rb") as f:
        if f.readline().rstrip() not in (b"PF", b"Pf"):
            raise ValueError("bukan PFM")
        w, h = map(int, f.readline().split())
        skala = float(f.readline().rstrip())
        data = np.fromfile(f, "<f" if skala < 0 else ">f")
    return np.flipud(data.reshape(h, w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs=2, required=True)
    a = ap.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from pycocotools import mask as mu

    rows = json.load(open(GT_DIR / "gt_depth_val.json"))["table"]
    okl = json.load(open(GT_DIR / "gt_occlusion_val.json"))["okl"]
    berkas = {im["id"]: im["file_name"] for im in json.load(open(COCO))["images"]}

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.4))
    for ia, baris in zip(a.ids, axes):
        img_id = rows[ia]["image_id"]
        idxs = [i for i, r in enumerate(rows) if r["image_id"] == img_id]
        masks = [mu.decode(rows[i]["rle"]).astype(bool) for i in idxs]
        dep = read_pfm(str(DEPTH_DIR / f"{Path(berkas[img_id]).stem}.pfm"))
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
        hull = np.zeros(m.shape, np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(m.astype(np.uint8))), 1)
        hull = hull.astype(bool)
        luar = hull & ~m
        tertutup = np.zeros_like(m)
        penutup = np.zeros_like(m)
        for b, ib in enumerate(idxs):
            if b == p or med[b] >= med[p]:
                continue
            t = luar & masks[b]
            tertutup |= t
            if t.sum():
                penutup |= masks[b]
        rasio = tertutup.sum() / float(hull.sum())
        assert abs(rasio - okl[ia]) < 1e-6, f"rasio tak cocok: {rasio} vs {okl[ia]}"

        rgb = cv2.cvtColor(cv2.imread(berkas[img_id]), cv2.COLOR_BGR2RGB)
        ys, xs = np.where(hull)
        pad = max(8, int(0.10 * max(ys.max() - ys.min(), xs.max() - xs.min())))
        y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad)
        pot = lambda A: A[y0:y1, x0:x1]

        for ax in baris:
            ax.set_xticks([]); ax.set_yticks([])
        baris[0].imshow(pot(rgb))
        baris[0].contour(pot(m), levels=[0.5], colors="#c9591f", linewidths=1.8)
        baris[0].contour(pot(penutup), levels=[0.5], colors="#1d6b4a", linewidths=1.1)
        baris[0].set_title("image asli, batas mask target dan penutupnya", fontsize=8.5)

        kanvas = pot(rgb).astype(float) * 0.28 + 178 * 0.72
        kanvas[pot(m)] = [201, 89, 31]
        baris[1].imshow(kanvas.astype(np.uint8))
        baris[1].contour(pot(hull), levels=[0.5], colors="#14130f",
                         linewidths=1.8, linestyles="--")
        baris[1].set_title(f"mask {m.sum():,} px, hull {hull.sum():,} px"
                           .replace(",", "."), fontsize=8.5)

        vis = np.full(pot(m).shape + (3,), 250, np.uint8)
        vis[pot(m)] = [201, 89, 31]
        vis[pot(luar & ~tertutup)] = [214, 210, 198]
        vis[pot(tertutup)] = [29, 107, 74]
        baris[2].imshow(vis)
        baris[2].set_title(f"rasio = {tertutup.sum():,} / {hull.sum():,} = {rasio:.3f}"
                           .replace(",", "."), fontsize=8.5)
        print(f"instance {ia}: rasio {rasio:.3f}, mask {m.sum()}, hull {hull.sum()}")

    fig.legend(handles=[
        Patch(facecolor="#c9591f", label="mask target"),
        Patch(facecolor="#1d6b4a", label="tertutup instance lebih dekat, dihitung"),
        Patch(facecolor="#d6d2c6", label="cekungan bentuk sendiri, tidak dihitung")],
        loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(OUT, dpi=96, bbox_inches="tight", facecolor="white")
    print(f"tersimpan: {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
