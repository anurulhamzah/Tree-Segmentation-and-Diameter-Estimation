#!/usr/bin/env python3
"""eval_depth_sweep_locked.py - sweep max_depth frozen, dinilai pada POPULASI TERKUNCI.

MASALAH YANG DIPECAHKAN. Tiap run sweep menyimpan `best_metrics.json` yang dihitung pada
populasinya SENDIRI, sehingga N berbeda antar cap (cap 10 -> N kecil berisi pohon dekat yang
mudah, cap 40 -> N besar berisi pohon jauh yang sulit). Membandingkan RMSE lintas baris
seperti itu mengukur kesulitan populasi, bukan kualitas model, dan pernah menghasilkan
kesimpulan terbalik "cap 10 terbaik".

CARA YANG BENAR. Kunci satu himpunan instance (filter dasar, TANPA batas depth), jalankan
setiap head di atasnya, lalu simpan `d_trunk` per instance. Dengan begitu RMSE pada cap
evaluasi mana pun dapat dihitung ulang dari satu berkas, dan dua pertanyaan berbeda dapat
dijawab terpisah:
  - kualitas model : populasi dikunci, cap latih yang divariasikan
  - titik operasi  : tiap cap dinilai pada populasi yang benar-benar dilayaninya

Keluaran: reports/dbh_eval/depth_sweep_locked.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(HERE))

SWEEP = PROJECT_ROOT.parent / "maskdino_output" / "sweep_dbh_hyperparams"
CACHE = PROJECT_ROOT.parent / "maskdino_output" / "dbh_sandbox_cache_plain135k"
OUT = PROJECT_ROOT / "reports" / "dbh_eval" / "depth_sweep_locked.json"

SPECIES = (3, 5, 8, 9, 10, 12, 13)
SEEDS = (0, 1, 2)
# cap 20 adalah baku sweep, jadi direktorinya tanpa sufiks maxdepth
CAPS = {10: "maxdepth10", 15: "maxdepth15", 20: "", 25: "maxdepth25",
        30: "maxdepth30", 40: "maxdepth40"}


def main():
    from dbh_sandbox_head import FeatureCache, load_index, build_feature_vec
    from trunk_roi_dbh_head_hybrid import HybridTrunkROIDBHHead

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hasil = {}

    for split in ("val", "test"):
        df = load_index(CACHE, split)
        # Filter dasar SAMA dengan evaluator joint, kecuali batas depth sengaja dilepas
        # supaya himpunan ini menjadi superset semua cap yang diuji.
        m = (df.category_id.isin(SPECIES) & (df.gt_dbh_mm > 0) & (df.trunk_px >= 5)
             & ((df.world_h - 1.3).abs() <= 0.5) & (df.d_trunk >= 1.0))
        # RubberFig dibatasi 150 cm, mengikuti konvensi seluruh laporan
        m &= ~((df.category_id == 12) & (df.gt_dbh_mm >= 1500.0))
        locked = df[m].copy()
        print(f"[{split}] populasi terkunci: N={len(locked)} "
              f"(d_trunk {locked.d_trunk.min():.1f}-{locked.d_trunk.max():.1f} m)")

        cache = FeatureCache(CACHE, split)
        hasil[split] = {"d": [float(x) for x in locked.d_trunk],
                        "gt": [float(x) for x in locked.gt_dbh_mm],
                        "sp": [int(x) for x in locked.category_id],
                        "px": [float(x) for x in locked.trunk_px],
                        "pred": {}}

        for cap, suf in CAPS.items():
            for seed in SEEDS:
                nama = (f"sevenspecies_plain135k_{suf}_seed{seed}" if suf
                        else f"sevenspecies_plain135k_seed{seed}")
                ckpt = SWEEP / nama / "head_best.pth"
                if not ckpt.exists():
                    print(f"  LEWAT {nama}: head_best.pth tidak ada"); continue
                head = HybridTrunkROIDBHHead(in_channels=192, hidden=256, dropout=0.3,
                                             strip_rows=5).to(device)
                head.load_state_dict(torch.load(ckpt, map_location=device))
                head.eval()
                preds = []
                with torch.no_grad():
                    for row in locked.itertuples():
                        x = build_feature_vec(head, cache, row, device, use_geom=True)
                        preds.append(float(torch.expm1(head.mlp(x).squeeze(-1)).item())
                                     if x is not None else float("nan"))
                hasil[split]["pred"][f"{cap}_{seed}"] = [round(p, 3) for p in preds]
                arr = np.array(preds); gt = locked.gt_dbh_mm.values
                ok = ~np.isnan(arr)
                print(f"  cap {cap:>2} seed {seed}: RMSE(terkunci)="
                      f"{np.sqrt(np.mean((arr[ok]-gt[ok])**2)):.2f} mm  N={ok.sum()}")
                del head
                torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(hasil, open(OUT, "w"))
    print(f"\nTersimpan: {OUT}")


if __name__ == "__main__":
    main()
