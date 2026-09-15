#!/usr/bin/env python3
"""Ukur dampak koreksi tangent pada estimator geometris DBH.

Latar: Persamaan 2 naskah, D = w_px * d / f_x, identik dengan Equation (1) Holcomb dkk. 2023,
yang mereka nyatakan sendiri "consistently underestimate ... especially for large trees and when
the user stands close to the trunk". Dua koreksi terbitan tersedia. Skrip ini menghitung ulang
D_geom dengan ketiganya pada dump per-instance yang SUDAH ADA, jadi tanpa GPU dan tanpa training.

Yang diukur hanya estimator geometrisnya, bukan prediksi model, karena D_geom masuk ke MLP
sebagai input feature dan efeknya pada prediksi akhir hanya bisa diukur dengan retrain.
"""
import json, sys, numpy as np
from pathlib import Path

F_X = 240.0     # _ORIG_FY di trunk_roi_dbh_head_hybrid.py; diverifikasi ulang di bawah
DUMP = Path("/scratch2/pr65/anur0018/tree_classification/reports/dbh_eval/"
            "dbh_per_instance_FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k_cap30.json")

def statistik(pred_mm, gt_mm):
    e = np.asarray(pred_mm) - np.asarray(gt_mm)
    return (np.sqrt((e ** 2).mean()) / 10, np.abs(e).mean() / 10, e.mean() / 10)

d = json.load(open(DUMP))
for split in ("val", "test"):
    rows = [r for r in d["data"][split] if r.get("px") and r.get("d") and r.get("gt")]
    px = np.array([r["px"] for r in rows], float)
    dist = np.array([r["d"] for r in rows], float)
    gt = np.array([r["gt"] for r in rows], float)
    geom_tersimpan = np.array([r["geom"] for r in rows], float)

    kita = px * dist * 1000.0 / F_X
    beda = np.abs(kita - geom_tersimpan).max()
    holcomb = px * dist * 1000.0 / (F_X - px / 4.0)
    x = px / F_X
    feng = 2 * dist * 1000.0 / (np.sqrt(4 / x ** 2 + 1) - 1)

    print(f"\n===== split {split}, N = {len(rows)} =====")
    print(f"  cek rumus: selisih maks terhadap kolom 'geom' tersimpan = {beda:.4f} mm")
    print(f"  {'estimator':<26} {'RMSE':>8} {'MAE':>8} {'bias':>8}   (cm)")
    for nama, v in [("kita, Holcomb Eq.1", kita), ("Holcomb 2023 Eq.2", holcomb),
                    ("Feng 2024 Eq.1", feng)]:
        r, m, b = statistik(v, gt)
        print(f"  {nama:<26} {r:8.2f} {m:8.2f} {b:8.2f}")
    print(f"  {'prediksi head (acuan)':<26} "
          f"{statistik([r['pred'] for r in rows], gt)[0]:8.2f} "
          f"{statistik([r['pred'] for r in rows], gt)[1]:8.2f} "
          f"{statistik([r['pred'] for r in rows], gt)[2]:8.2f}")
    print(f"  sebaran px: median {np.median(px):.0f}, p90 {np.percentile(px,90):.0f}, "
          f"maks {px.max():.0f}  ->  koreksi maks "
          f"{100*(F_X/(F_X-px.max()/4)-1):.1f}%")
