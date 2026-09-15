"""
recompute_leaderboard.py — Rekonstruksi leaderboard.csv dari metrics.json tiap trial yang
SUDAH SELESAI, dengan metrik ranking BARU (mean/worst_species_R2 hanya dari spesies N>=min_species_n)
TANPA retrain apa pun -- metrics.json sudah punya per_species lengkap (N/R2/MAE) di tiap eval
checkpoint dari run yang lama, jadi cukup dihitung ulang dari data itu.

Alasan: ditemukan 12 Jul 2026 bahwa mean_per_species_R2 (metrik ranking utama sweep) sangat
tidak stabil utk spesies ber-N eval kecil (mis. Persimmon N=10 -> R2=-23.9 padahal MAE=36.5mm,
salah satu terbaik). Fix diterapkan di dbh_sandbox_head.py (compute_metrics, min_species_n=30
default) utk trial BARU; script ini menerapkan definisi yang sama secara retroaktif ke trial LAMA.

Jalankan: python scripts/dbh_sandbox/recompute_leaderboard.py
"""
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SWEEP_OUT = Path("/scratch2/pr65/anur0018/maskdino_output/sweep_dbh_hyperparams")
# SENGAJA nulis ke file terpisah (bukan leaderboard.csv langsung) -- Fase B kadang masih
# berjalan background dan terus append ke leaderboard.csv; nulis bareng-bareng bisa race
# condition / korup. Gabungkan manual ke leaderboard.csv setelah semua fase selesai.
LEADERBOARD = SWEEP_OUT / "leaderboard_recomputed.csv"
MIN_SPECIES_N = 30

RESULT_FIELDS = ["phase", "name", "iteration", "N", "R2", "mean_per_species_R2",
                  "worst_species_R2", "MAE", "bias", "w50mm", "w100mm",
                  "n_species_in_mean", "species_excluded_small_n"]


def infer_phase(name: str) -> str:
    if name.startswith("phase05"):
        return "0.5"
    if name.startswith("phaseA"):
        return "A"
    if name.startswith("phaseB"):
        return "B"
    if name.startswith("phaseC"):
        return "C"
    return "?"


def recompute_entry(metrics_entry: dict, min_species_n: int = MIN_SPECIES_N) -> dict:
    per_species = metrics_entry.get("per_species", {})
    included = {cid: s for cid, s in per_species.items()
                if not math.isnan(s["R2"]) and s["N"] >= min_species_n}
    excluded = sorted(int(cid) for cid, s in per_species.items() if cid not in included)
    sp_r2_vals = [s["R2"] for s in included.values()]
    mean_r2 = sum(sp_r2_vals) / len(sp_r2_vals) if sp_r2_vals else float("nan")
    worst_r2 = min(sp_r2_vals) if sp_r2_vals else float("nan")
    out = dict(metrics_entry)
    out["mean_per_species_R2"] = mean_r2
    out["worst_species_R2"] = worst_r2
    out["n_species_in_mean"] = len(sp_r2_vals)
    out["species_excluded_small_n"] = excluded
    return out


def main():
    trial_dirs = sorted(p for p in SWEEP_OUT.iterdir()
                         if p.is_dir() and (p / "metrics.json").exists())
    print(f"Ditemukan {len(trial_dirs)} trial dgn metrics.json")

    rows = []
    for d in trial_dirs:
        try:
            metrics_log = json.load(open(d / "metrics.json"))
        except Exception as e:
            print(f"  SKIP {d.name}: gagal baca metrics.json ({e})")
            continue
        valid = [m for m in metrics_log if "per_species" in m]
        if not valid:
            print(f"  SKIP {d.name}: tidak ada eval dgn per_species (run gagal/belum eval)")
            continue

        recomputed = [recompute_entry(m) for m in valid]
        best = max((r for r in recomputed if not math.isnan(r["mean_per_species_R2"])),
                   key=lambda r: r["mean_per_species_R2"], default=None)
        if best is None:
            print(f"  SKIP {d.name}: semua eval NaN stlh threshold (semua spesies < N={MIN_SPECIES_N})")
            continue

        row = {"phase": infer_phase(d.name), "name": d.name}
        for k in RESULT_FIELDS[2:]:
            row[k] = best.get(k)
        rows.append(row)

        # Simpan metrics_recomputed.json per trial (audit trail, tidak menimpa metrics.json asli)
        with open(d / "metrics_recomputed.json", "w") as f:
            json.dump(recomputed, f, indent=2)

    rows.sort(key=lambda r: -r["mean_per_species_R2"] if not math.isnan(r["mean_per_species_R2"]) else 1e9)

    with open(LEADERBOARD, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nLeaderboard direkonstruksi: {LEADERBOARD}  ({len(rows)} trial)")
    print(f"\nTop 10 (metrik BARU, min_species_n={MIN_SPECIES_N}):")
    print(f'{"name":<50}{"meanPerSpR2":>12}{"nSpIncl":>9}{"aggR2":>8}{"MAE":>7}')
    for r in rows[:10]:
        print(f'{r["name"]:<50}{r["mean_per_species_R2"]:>12.4f}{r["n_species_in_mean"]:>9}'
              f'{r["R2"]:>8.4f}{r["MAE"]:>7.1f}')


if __name__ == "__main__":
    main()
