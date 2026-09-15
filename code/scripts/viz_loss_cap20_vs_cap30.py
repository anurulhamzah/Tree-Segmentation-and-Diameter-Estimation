#!/usr/bin/env python3
"""viz_loss_cap20_vs_cap30.py — loss segmentasi dan loss diameter kedua run berdampingan.

Membandingkan run 11-spesies `max_depth` 20 (asal) lawan 30 (berjalan). Dua panel:

  kiri  : loss segmentasi, komponen diameter DIKELUARKAN dari total_loss kedua run
  kanan : raw_loss diameter, yaitu loss diameter SEBELUM dikalikan DBH_WEIGHT

⚠️ Kedua panel tidak sama tafsirnya. Panel kiri membandingkan dua besaran yang identik
definisinya, jadi selisihnya langsung berarti. Panel kanan TIDAK: `max_depth` menentukan
instance mana yang boleh menyumbang loss, sehingga cap 30 merata-ratakan populasi yang lebih
besar dan lebih jauh. Kurvanya boleh dibandingkan bentuknya, bukan tingginya.

    python scripts/viz_loss_cap20_vs_cap30.py
"""
import json
from pathlib import Path

import numpy as np

OUT_ROOT = Path("/scratch2/pr65/anur0018/maskdino_output")
RUNS = [("max_depth 20 (asal)", "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k", "#8a877d"),
        ("max_depth 30 (berjalan)", "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k_cap30",
         "#c9591f")]
HASIL = Path("/scratch2/pr65/anur0018/tree_classification/reports/stratified_v4/"
             "loss_cap20_vs_cap30.png")


def baca(d):
    it, seg, dia, nclean = [], [], [], []
    for line in (OUT_ROOT / d / "metrics.json").open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("iteration") is None or "total_loss" not in r:
            continue
        if "segm/AP50" in r:          # baris eval juga punya total_loss, jangan ikut
            continue
        it.append(r["iteration"])
        seg.append(r["total_loss"] - (r.get("loss_dbh_trunkroi") or 0))
        dia.append(r.get("dbh_trunkroi/raw_loss"))
        nclean.append(r.get("dbh_trunkroi/n_clean"))
    return np.array(it), np.array(seg, float), np.array(dia, float), np.array(nclean, float)


def halus(x, y, w=25):
    """Rerata bergerak; kurva mentah terlalu berisik untuk dibandingkan secara visual."""
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < w:
        return x, y
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.5))
    data = {}
    for label, d, warna in RUNS:
        it, seg, dia, nc = baca(d)
        data[label] = (it, seg, dia, nc)
        for a, y, in ((ax[0], seg), (ax[1], dia)):
            xs, ys = halus(it, y)
            a.plot(xs, ys, color=warna, lw=1.8, label=label)

    batas = min(data[l][0].max() for l in data)
    for a, judul, ylab in ((ax[0], "Loss segmentasi (komponen diameter dikeluarkan)", "loss"),
                           (ax[1], "raw_loss diameter (sebelum dikali DBH_WEIGHT)", "loss")):
        a.set_title(judul, fontsize=10)
        a.set_xlabel("iterasi"); a.set_ylabel(ylab)
        a.set_xlim(0, batas)
        a.grid(alpha=.25, lw=.6)
        a.legend(fontsize=8.5, frameon=False)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylim(70, 200)
    ax[1].set_ylim(0.2, 1.2)
    ax[1].text(.98, .95, "populasi berbeda,\nbandingkan bentuknya bukan tingginya",
               transform=ax[1].transAxes, ha="right", va="top", fontsize=8, color="#8a877d")

    fig.suptitle(f"Multi-task 11 spesies, dua nilai max_depth  "
                 f"(dibandingkan sampai iterasi {batas:,})".replace(",", "."), fontsize=11)
    fig.tight_layout()
    fig.savefig(HASIL, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"tersimpan: {HASIL}  ({HASIL.stat().st_size/1024:.0f} KB)")

    # ringkasan angka pada rentang yang berpasangan
    import statistics as st
    for label in data:
        it, seg, dia, nc = data[label]
        m = it <= batas
        print(f"\n{label}")
        print(f"  loss seg 3k iterasi terakhir : {np.nanmean(seg[m][-30:]):.2f}")
        print(f"  raw_loss diameter idem       : {np.nanmean(dia[m][-30:]):.4f}")
        print(f"  n_clean rerata               : {np.nanmean(nc[m]):.2f}")


if __name__ == "__main__":
    main()
