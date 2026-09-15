"""
run_dbh_sweep.py — Driver DBH sandbox: Fase 0.5 (kalibrasi) -> A (OFAT screening)
-> B (kombinasi) -> C (konfirmasi top-3, test split, 3 seed).

Pola sama seperti scripts/run_sweep_phases1to4.py: sequential, idempoten (skip
kalau metrics.json sudah lengkap), subprocess via os.system(), hasil dikumpulkan
ke leaderboard CSV. Lihat rencana lengkap:
/home/anur0018/.claude/plans/melodic-mapping-journal.md

Fase D (sanity-check fidelitas via pipeline produksi asli, non-cache) SENGAJA
belum diimplementasi di sini -- baru bisa disusun setelah tahu kombinasi juara
dari Fase C (menghindari kerja spekulatif utk kombinasi yg belum tentu menang).

Jalankan:
  nohup /scratch2/pr65/anur0018/conda_envs/maskdino/bin/python -u \\
      scripts/dbh_sandbox/run_dbh_sweep.py --phase 0.5 \\
      > logs/dbh_sweep_phase05_nohup.log 2>&1 &
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

HERE         = Path(__file__).resolve().parent
SCRIPTS_ROOT = HERE.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
sys.path.insert(0, str(HERE))

import filters as F_

PYTHON      = "/scratch2/pr65/anur0018/conda_envs/maskdino/bin/python"
HEAD_SCRIPT = HERE / "dbh_sandbox_head.py"
CACHE_DIR   = PROJECT_ROOT.parent / "maskdino_output" / "dbh_sandbox_cache"
SWEEP_OUT   = PROJECT_ROOT.parent / "maskdino_output" / "sweep_dbh_hyperparams"
LEADERBOARD = SWEEP_OUT / "leaderboard.csv"
SCREEN_ITER_FILE = SWEEP_OUT / "screen_max_iter.json"

# Baseline: konfigurasi V6 utk axis non-spesies + universe 11-spesies baru
# (lihat plan bagian "Fase A"). Kunci di sini HARUS match nama --flag di
# dbh_sandbox_head.py (lihat CLI_FLAG di bawah).
BASELINE = dict(
    species_subset="universe11", min_trunk_px=5, min_depth=1.0, max_depth=20.0,
    wh_tolerance=0.5, ratio_bounds="off", rubberfig_cap_cm="150",
    strip_rows=5, species_weight_scheme="inverse_freq_capped", species_weight_cap=20.0,
    dropout=0.3, hidden_width=256, lr=1e-4, loss_type="smooth_l1", geometric_feature="on",
)

# Level alternatif per axis (tabel Fase A di plan). None-entry tidak perlu --
# baseline axis levels ITU SENDIRI adalah pembanding "level 0".
AXIS_LEVELS = {
    "species_subset":        ["v6_6", "drop_rubberfig", "drop_mango", "drop_both",
                               "rainforest", "plantation"],
    "min_trunk_px":           [3, 7, 10],
    "max_depth":               [10.0, 15.0, 25.0],
    "min_depth":               [0.5],
    "wh_tolerance":            [0.25, 0.75],
    "ratio_bounds":            ["0.3,3.0", "0.5,2.0"],
    "rubberfig_cap_cm":        ["off", "100"],
    "strip_rows":              [1, 3, 7],
    "species_weight_scheme":  ["none", "sqrt_inverse_freq"],
    "dropout":                 [0.1, 0.5],
    "hidden_width":            [128, 512],
    "lr":                      [5e-5, 2e-4],
    "loss_type":               ["mse"],
    "geometric_feature":       ["off"],
}

CLI_FLAG = {
    "species_subset": "--species-subset", "min_trunk_px": "--min-trunk-px",
    "min_depth": "--min-depth", "max_depth": "--max-depth", "wh_tolerance": "--wh-tolerance",
    "ratio_bounds": "--ratio-bounds", "rubberfig_cap_cm": "--rubberfig-cap-cm",
    "strip_rows": "--strip-rows", "species_weight_scheme": "--species-weight-scheme",
    "species_weight_cap": "--species-weight-cap", "dropout": "--dropout",
    "hidden_width": "--hidden-width", "lr": "--lr", "loss_type": "--loss-type",
    "geometric_feature": "--geometric-feature",
}

RESULT_FIELDS = ["name", "iteration", "N", "R2", "mean_per_species_R2",
                  "worst_species_R2", "MAE", "bias", "w50mm", "w100mm"]


def run_trial(name, combo, max_iter, eval_split="val", seed=0, eval_period=500, batch_size=64):
    out_dir = SWEEP_OUT / name
    metrics_path = out_dir / "metrics.json"

    if metrics_path.exists():
        try:
            metrics_log = json.load(open(metrics_path))
            last_iter = metrics_log[-1].get("iteration", -1) if metrics_log else -1
            if metrics_log and "per_species" in metrics_log[-1] and last_iter >= max_iter:
                print(f"  SKIP {name} (sudah selesai, iter {last_iter}/{max_iter})")
                return summarize(name, metrics_log)
            elif metrics_log:
                print(f"  RETRAIN {name} (sebelumnya terpotong di iter {last_iter}/{max_iter})")
        except Exception:
            pass

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "combo_requested.json", "w") as f:
        json.dump({"combo": combo, "max_iter": max_iter, "eval_split": eval_split,
                   "seed": seed}, f, indent=2)

    cmd_parts = [PYTHON, "-u", str(HEAD_SCRIPT),
                 "--cache-dir", str(CACHE_DIR), "--output-dir", str(out_dir),
                 "--eval-split", eval_split, "--max-iter", str(max_iter),
                 "--eval-period", str(eval_period), "--batch-size", str(batch_size),
                 "--seed", str(seed)]
    for k, v in combo.items():
        cmd_parts += [CLI_FLAG[k], str(v)]
    cmd = " ".join(cmd_parts) + f" > {out_dir}/run.log 2>&1"

    print(f"  RUN  {name}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"  FAIL {name} (exit={ret}) -- lihat {out_dir}/run.log")
        return None
    if not metrics_path.exists():
        print(f"  FAIL {name} (metrics.json tidak dihasilkan)")
        return None
    metrics_log = json.load(open(metrics_path))
    return summarize(name, metrics_log)


MIN_SPECIES_N = 30  # sinkron dgn default --min-species-n di dbh_sandbox_head.py


def _recompute_thresholded(entry):
    """R2 pada N kecil sangat tidak stabil (lihat temuan Persimmon N=10 R2=-23.9 tapi MAE
    bagus). Selalu hitung ulang mean/worst_species_R2 dari per_species mentah dgn threshold
    MIN_SPECIES_N -- JANGAN percaya nilai mean_per_species_R2 yang tersimpan di metrics.json,
    krn trial lama (sebelum fix ini) dihitung tanpa threshold."""
    per_species = entry.get("per_species", {})
    included = {cid: s for cid, s in per_species.items()
                if not math.isnan(s["R2"]) and s["N"] >= MIN_SPECIES_N}
    sp_r2_vals = [s["R2"] for s in included.values()]
    out = dict(entry)
    out["mean_per_species_R2"] = sum(sp_r2_vals) / len(sp_r2_vals) if sp_r2_vals else float("nan")
    out["worst_species_R2"] = min(sp_r2_vals) if sp_r2_vals else float("nan")
    out["n_species_in_mean"] = len(sp_r2_vals)
    return out


def summarize(name, metrics_log):
    with_species = [m for m in metrics_log if "per_species" in m]
    if not with_species:
        return None
    recomputed = [_recompute_thresholded(m) for m in with_species]
    valid = [m for m in recomputed if not math.isnan(m["mean_per_species_R2"])]
    if not valid:
        return None
    best = max(valid, key=lambda r: r["mean_per_species_R2"])
    row = {"name": name}
    for k in RESULT_FIELDS[1:]:
        row[k] = best.get(k)
    return row


def append_leaderboard(row, phase):
    if row is None:
        return
    SWEEP_OUT.mkdir(parents=True, exist_ok=True)
    row = {"phase": phase, **row}
    exists = LEADERBOARD.exists()
    if exists:
        # Idempoten: skip kalau (phase,name) ini sudah tercatat (mis. re-run setelah resume).
        with open(LEADERBOARD) as f:
            already = {(r["phase"], r["name"]) for r in csv.DictReader(f)}
        if (phase, row["name"]) in already:
            return
    with open(LEADERBOARD, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["phase"] + RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def determine_plateau(metrics_log, tol_abs=0.02, streak=5):
    """Iterasi pertama dgn AGGREGATE R2 (bukan mean_per_species_R2) dlm ±tol_abs (absolut,
    bukan relatif) dari nilai final utk >=streak eval berturut-turut, dibulatkan ke atas ke
    kelipatan 1000.

    Kenapa pakai aggregate R2 utk deteksi plateau (bukan mean_per_species_R2 yg jadi metrik
    ranking utama di Fase A/B/C): mean_per_species_R2 sangat noisy krn didominasi spesies
    ber-N kecil (mis. Pomegranate/Persimmon/Mango, N~100-300) -- R2 per-spesies utk N kecil
    bisa berayun besar (-0.2 s/d -0.9) antar-eval TANPA model benar2 masih belajar; itu noise
    pengukuran, bukan sinyal konvergensi. Aggregate R2 (didominasi BangaloPalm, N besar) jauh
    lebih stabil dan mencerminkan kapan MLP head berhenti membaik scr riil. Toleransi relatif
    (persen dari nilai final) jg tidak cocok krn nilai bisa kecil/negatif -- pakai absolut.
    """
    scores = [(m["iteration"], m["R2"]) for m in metrics_log
              if "R2" in m and not math.isnan(m["R2"])]
    if not scores:
        return 20000
    final = scores[-1][1]
    for i in range(len(scores)):
        window = scores[i:i + streak]
        if len(window) < streak:
            break
        if all(abs(v - final) <= tol_abs for _, v in window):
            plateau_iter = window[0][0]
            return int(math.ceil(plateau_iter / 1000.0) * 1000)
    print("  !! Tidak landai lebih awal -- pakai budget penuh 20000 utk Fase A/B")
    return scores[-1][0]


def get_screen_iter():
    if SCREEN_ITER_FILE.exists():
        return json.load(open(SCREEN_ITER_FILE))["screen_max_iter"]
    raise RuntimeError("Fase 0.5 belum jalan -- jalankan --phase 0.5 dulu.")


def phase_05():
    print("\n=== FASE 0.5: Kalibrasi budget iterasi ===")
    row = run_trial("phase05_calibration_baseline", BASELINE, max_iter=20000, eval_period=250)
    append_leaderboard(row, "0.5")

    metrics_log = json.load(open(SWEEP_OUT / "phase05_calibration_baseline" / "metrics.json"))
    screen_iter = determine_plateau(metrics_log)
    print(f"  -> SCREEN_MAX_ITER = {screen_iter}")
    with open(SCREEN_ITER_FILE, "w") as f:
        json.dump({"screen_max_iter": screen_iter}, f)
    return screen_iter


def phase_A(screen_iter):
    print(f"\n=== FASE A: OFAT screening (budget={screen_iter}) ===")
    results = {}
    baseline_row = summarize("phase05_calibration_baseline",
                              json.load(open(SWEEP_OUT / "phase05_calibration_baseline" / "metrics.json")))
    results["baseline"] = baseline_row
    for axis, levels in AXIS_LEVELS.items():
        for level in levels:
            combo = {**BASELINE, axis: level}
            name = f"phaseA_{axis}_{level}".replace(",", "-").replace(" ", "")
            row = run_trial(name, combo, max_iter=screen_iter)
            append_leaderboard(row, "A")
            results[(axis, level)] = row
    return results


def phase_B(screen_iter, phaseA_results):
    """
    Fase B — REVISI setelah lihat hasil Fase A (12 Jul 2026): axis species_subset ternyata
    MENDOMINASI seluruh axis lain -- HANYA "rainforest"/"v6_6" yg mengalahkan baseline
    (universe11), sedangkan SEMUA 13 axis non-spesies lain kalah dari baseline saat diuji di
    atas universe11. Ini kemungkinan besar krn universe11 punya spesies ber-N kecil
    (Persimmon/Pomegranate) yg bikin mean_per_species_R2 sangat noisy -- jadi hasil axis
    non-spesies dari Fase A TIDAK bisa dipercaya sbg "axis itu buruk", melainkan "diuji di
    atas basis yg buruk".

    Fix: kunci species_subset ke level terbaik dari Fase A dulu (anchor), BARU ulang OFAT
    ronde-2 utk 13 axis non-spesies DI ATAS anchor itu -- baru axis2 itu diuji scr adil.
    """
    print("\n=== FASE B: Anchor species_subset terbaik + ronde-2 OFAT axis lain ===")
    baseline_score = phaseA_results["baseline"]["mean_per_species_R2"] if phaseA_results["baseline"] else -1e9

    best_species_level, best_species_score = None, baseline_score
    for level in AXIS_LEVELS["species_subset"]:
        row = phaseA_results.get(("species_subset", level))
        if row and row["mean_per_species_R2"] > best_species_score:
            best_species_level, best_species_score = level, row["mean_per_species_R2"]

    if best_species_level is None:
        anchor, anchor_name = dict(BASELINE), "baseline"
        print("  species_subset baseline (universe11) tetap terbaik -- anchor = baseline.")
    else:
        anchor = {**BASELINE, "species_subset": best_species_level}
        anchor_name = f"species_{best_species_level}"
        print(f"  Anchor: species_subset={best_species_level}  "
              f"(mean_per_species_R2={best_species_score:.4f} vs baseline universe11 {baseline_score:.4f})")

    other_axes = {a: lv for a, lv in AXIS_LEVELS.items() if a != "species_subset"}

    if anchor_name == "baseline":
        # anchor == BASELINE persis -> kombinasi {**anchor, axis: level} utk axis non-spesies
        # IDENTIK dgn yg sudah diuji Fase A ({**BASELINE, axis: level}). Reuse hasilnya
        # langsung (skor sudah dihitung ulang dgn threshold N via summarize()), jangan retrain.
        print("  Anchor = baseline persis -> reuse hasil axis non-spesies dari Fase A (no retrain).")
        anchor_row = phaseA_results["baseline"]
        anchor_score = anchor_row["mean_per_species_R2"] if anchor_row else best_species_score
        winner_level, impact = {}, {}
        for axis, levels in other_axes.items():
            best_level, best_score = None, anchor_score
            for level in levels:
                row = phaseA_results.get((axis, level))
                if row and row["mean_per_species_R2"] > best_score:
                    best_level, best_score = level, row["mean_per_species_R2"]
            if best_level is not None:
                winner_level[axis] = best_level
                impact[axis] = best_score - anchor_score
        ranked_axes = sorted(impact, key=lambda a: -impact[a])
        print(f"  Axis (reuse Fase A, anchor=baseline) dgn improvement > anchor: {ranked_axes}")
        return _phase_B_finalize(anchor, anchor_name, winner_level, ranked_axes, screen_iter)

    anchor_row = run_trial(f"phaseB_anchor_{anchor_name}", anchor, max_iter=screen_iter)
    append_leaderboard(anchor_row, "B")
    anchor_score = anchor_row["mean_per_species_R2"] if anchor_row else best_species_score

    winner_level, impact = {}, {}
    for axis, levels in other_axes.items():
        best_level, best_score = None, anchor_score
        for level in levels:
            combo = {**anchor, axis: level}
            name = f"phaseB_{anchor_name}_{axis}_{level}".replace(",", "-").replace(" ", "")
            row = run_trial(name, combo, max_iter=screen_iter)
            append_leaderboard(row, "B")
            if row and row["mean_per_species_R2"] > best_score:
                best_level, best_score = level, row["mean_per_species_R2"]
        if best_level is not None:
            winner_level[axis] = best_level
            impact[axis] = best_score - anchor_score

    ranked_axes = sorted(impact, key=lambda a: -impact[a])
    print(f"  Axis (ronde-2, anchor={anchor_name}) dgn improvement > anchor: {ranked_axes}")
    return _phase_B_finalize(anchor, anchor_name, winner_level, ranked_axes, screen_iter)


def _phase_B_finalize(anchor, anchor_name, winner_level, ranked_axes, screen_iter):
    if not ranked_axes:
        candidates = {f"phaseB_final_{anchor_name}_anchor_only": anchor}
    else:
        all_winners = {**anchor, **{a: winner_level[a] for a in ranked_axes}}
        candidates = {f"phaseB_final_{anchor_name}_all_winners": all_winners}
        for axis in ranked_axes[:4]:
            combo = {**all_winners}
            combo[axis] = anchor[axis]  # leave-one-out: kembalikan axis ini ke level anchor
            candidates[f"phaseB_final_{anchor_name}_loo_{axis}"] = combo

    results = {}
    for name, combo in candidates.items():
        row = run_trial(name, combo, max_iter=screen_iter)
        append_leaderboard(row, "B")
        results[name] = (combo, row)
    return results


def phase_C(phaseB_results, top_k=3):
    print("\n=== FASE C: Konfirmasi top-3 (test split, 20k iter x 3 seed) ===")
    ranked = sorted(
        [(n, c, r) for n, (c, r) in phaseB_results.items() if r is not None],
        key=lambda t: -t[2]["mean_per_species_R2"])[:top_k]

    final_results = {}
    for name, combo, _ in ranked:
        seed_rows = []
        for seed in [0, 1, 2]:
            trial_name = f"phaseC_{name}_seed{seed}"
            row = run_trial(trial_name, combo, max_iter=20000, eval_split="test", seed=seed)
            append_leaderboard(row, "C")
            if row:
                seed_rows.append(row)
        if seed_rows:
            mean_r2 = sum(r["mean_per_species_R2"] for r in seed_rows) / len(seed_rows)
            print(f"  {name}: mean_per_species_R2 (3 seed) = {mean_r2:.4f}")
            final_results[name] = {"combo": combo, "seed_rows": seed_rows, "mean_r2": mean_r2}

    if final_results:
        winner = max(final_results, key=lambda n: final_results[n]["mean_r2"])
        print(f"\n  *** JUARA Fase C: {winner}  mean_per_species_R2={final_results[winner]['mean_r2']:.4f} ***")
        with open(SWEEP_OUT / "phaseC_winner.json", "w") as f:
            json.dump({"name": winner, **final_results[winner]}, f, indent=2)
    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["0.5", "A", "B", "C", "all"])
    args = parser.parse_args()

    SWEEP_OUT.mkdir(parents=True, exist_ok=True)

    if args.phase in ("0.5", "all"):
        screen_iter = phase_05()
    else:
        screen_iter = get_screen_iter()

    if args.phase in ("A", "all"):
        phaseA_results = phase_A(screen_iter)
    elif args.phase in ("B", "C"):
        # Re-derive dari leaderboard yg sudah ada (idempoten -- run_trial akan skip yg selesai)
        phaseA_results = phase_A(screen_iter)
    else:
        phaseA_results = None

    if args.phase in ("B", "all") or (args.phase == "C" and phaseA_results):
        phaseB_results = phase_B(screen_iter, phaseA_results)
    else:
        phaseB_results = None

    if args.phase in ("C", "all") and phaseB_results:
        phase_C(phaseB_results)

    print(f"\nLeaderboard: {LEADERBOARD}")


if __name__ == "__main__":
    main()
