#!/usr/bin/env python3
"""
eval_frozen_full_metrics.py — Evaluasi ulang run frozen/decoupled dengan metrik LENGKAP,
tanpa melatih ulang apa pun.

Kenapa ada. `dbh_sandbox_head.py` sudah menyimpan RMSE keseluruhan, tetapi blok `per_species`
di dalamnya hanya memuat N, R2, dan MAE. Sejak RMSE ditetapkan sebagai metrik UTAMA pelaporan
(18 dari 20 makalah DBH memakainya sebagai metrik utama), sisi frozen jadi tidak setara dengan
sisi joint, yang kini menghitung RMSE per spesies. Akibatnya dekomposisi efek komposisi versus
efek pelatihan hanya bisa dilakukan pada metrik pendukung. Skrip ini menutup ketimpangan itu.

Yang dihitung, keseluruhan MAUPUN per spesies:
    RMSE (utama) · MAE · bias · rRMSE · MRAE · R2 (pendukung) · within-50mm · within-100mm
plus rata-rata per-spesies untuk RMSE, MAE, bias, dan R2 (kriteria N>=min_species_n).

Head TIDAK dilatih ulang. Weights dibaca dari head_best.pth (dan head_final.pth kalau ada),
feature dibaca dari cache res2 yang sama, filter direkonstruksi persis dari combo.json milik run
tersebut. Jadi angkanya sebanding apple-to-apple dengan best_metrics.json yang lama.

Jalankan:
    # satu run
    python scripts/dbh_sandbox/eval_frozen_full_metrics.py \\
        --run-dir /scratch2/pr65/anur0018/maskdino_output/sweep_dbh_hyperparams/universe11_plain135k_seed0

    # banyak run sekaligus + agregasi antar-seed
    python scripts/dbh_sandbox/eval_frozen_full_metrics.py \\
        --run-glob '/scratch2/pr65/anur0018/maskdino_output/sweep_dbh_hyperparams/*_plain135k_seed*' \\
        --splits val,test
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import filters as F_                                            # noqa: E402
from dbh_sandbox_head import (FeatureCache, load_index,          # noqa: E402
                              build_feature_vec, NUM_SPECIES)
from trunk_roi_dbh_head_hybrid import HybridTrunkROIDBHHead      # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "dbh_eval"


def metrics_block(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Metrik untuk satu kelompok instance. Urutan kunci mengikuti urutan kepentingan
    pelaporan: metrik utama lebih dulu, pendukung belakangan."""
    err = pred - gt
    abs_err = np.abs(err)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    gt_mean = float(np.mean(gt))
    ss_tot = float(np.sum((gt - gt_mean) ** 2))
    return {
        "N": int(len(pred)),
        "RMSE": rmse,
        "MAE": float(np.mean(abs_err)),
        "bias": float(np.mean(err)),
        "rRMSE": float(rmse / gt_mean * 100) if gt_mean > 0 else float("nan"),
        "MRAE": float(np.mean(abs_err / np.maximum(gt, 1e-6)) * 100),
        "R2": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "w50mm": float(np.mean(abs_err <= 50) * 100),
        "w100mm": float(np.mean(abs_err <= 100) * 100),
    }


def full_metrics(pred, gt, cats, min_species_n: int) -> dict:
    pred, gt, cats = np.asarray(pred), np.asarray(gt), np.asarray(cats)
    out = metrics_block(pred, gt)

    per_species, scored = {}, []
    for cid in sorted(set(cats.tolist())):
        m = cats == cid
        blk = metrics_block(pred[m], gt[m])
        blk["name"] = F_.CAT_ID_TO_NAME.get(int(cid), str(cid))
        per_species[int(cid)] = blk
        if blk["N"] >= min_species_n and not math.isnan(blk["R2"]):
            scored.append(blk)

    # Rata-rata per-spesies dihitung untuk metrik utama juga, bukan hanya R2. Inilah yang
    # membuat dekomposisi komposisi-vs-pelatihan bisa dijalankan pada RMSE dan MAE.
    for key in ("RMSE", "MAE", "bias", "R2"):
        out[f"mean_per_species_{key}"] = (
            float(np.mean([s[key] for s in scored])) if scored else float("nan"))
    out["worst_species_R2"] = float(min(s["R2"] for s in scored)) if scored else float("nan")
    out["n_species_in_mean"] = len(scored)
    out["min_species_n_threshold"] = min_species_n
    out["species_excluded_small_n"] = sorted(
        cid for cid, s in per_species.items() if s["N"] < min_species_n)
    out["per_species"] = per_species
    return out


@torch.no_grad()
def eval_one(run_dir: Path, splits, ckpt_name: str, device: str) -> dict:
    combo = json.load(open(run_dir / "combo.json"))
    cache_dir = Path(combo["cache_dir"])

    species_subset = F_.resolve_species_subset(combo["species_subset"])
    rb = combo.get("ratio_bounds", "off")
    ratio_bounds = None if (rb is None or str(rb).lower() == "off") else tuple(
        float(x) for x in str(rb).split(","))
    cap = combo.get("rubberfig_cap_cm", "off")
    rubberfig_cap = None if (cap is None or str(cap).lower() == "off") else float(cap)
    use_geom = combo.get("geometric_feature", "on") == "on"

    head = HybridTrunkROIDBHHead(
        in_channels=192, hidden=combo["hidden_width"], strip_rows=combo["strip_rows"],
        num_species=NUM_SPECIES, dropout=combo["dropout"]).to(device)
    state = torch.load(run_dir / ckpt_name, map_location=device)
    head.load_state_dict(state)
    head.eval()

    result = {"run": run_dir.name, "ckpt": ckpt_name,
              "species_subset": combo["species_subset"], "max_depth": combo["max_depth"],
              "seed": combo.get("seed"), "splits": {}}

    for split in splits:
        df = F_.apply_filters(
            load_index(cache_dir, split), species_subset=species_subset,
            min_trunk_px=combo["min_trunk_px"], min_depth=combo["min_depth"],
            max_depth=combo["max_depth"], wh_tolerance=combo["wh_tolerance"],
            ratio_bounds=ratio_bounds, rubberfig_cap_cm=rubberfig_cap)
        cache = FeatureCache(cache_dir, split)

        preds, gts, cats = [], [], []
        for row in df.itertuples():
            x = build_feature_vec(head, cache, row, device, use_geom)
            if x is None:
                continue
            preds.append(torch.expm1(head.mlp(x).squeeze(-1)).item())
            gts.append(row.gt_dbh_mm)
            cats.append(int(row.category_id))

        if not preds:
            result["splits"][split] = {"N": 0}
            continue
        result["splits"][split] = full_metrics(preds, gts, cats, combo["min_species_n"])
    return result


def aggregate(results, splits):
    """Rata-rata antar-seed per (species_subset, max_depth), plus rentangnya."""
    groups = {}
    for r in results:
        groups.setdefault((r["species_subset"], r["max_depth"]), []).append(r)

    agg = []
    for (subset, depth), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        row = {"species_subset": subset, "max_depth": depth, "n_seed": len(rs),
               "seeds": sorted(x["seed"] for x in rs if x["seed"] is not None), "splits": {}}
        for s in splits:
            vals = {}
            for key in ("RMSE", "MAE", "bias", "R2", "mean_per_species_RMSE",
                        "mean_per_species_MAE", "mean_per_species_R2", "N"):
                v = [x["splits"][s][key] for x in rs
                     if s in x["splits"] and key in x["splits"][s]]
                if v:
                    vals[key] = {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                 "max": float(np.max(v)), "spread": float(np.max(v) - np.min(v))}
            row["splits"][s] = vals
        agg.append(row)
    return agg


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="satu direktori run sweep")
    g.add_argument("--run-glob", help="pola glob untuk banyak run")
    ap.add_argument("--splits", default="val,test",
                    help="val untuk memilih checkpoint, test untuk pelaporan")
    ap.add_argument("--ckpt", default="head_best.pth", choices=["head_best.pth", "head_final.pth"])
    ap.add_argument("--out-name", default="frozen_full_metrics",
                    help="nama dasar berkas output di reports/dbh_eval/")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    dirs = ([Path(args.run_dir)] if args.run_dir
            else sorted(Path(p) for p in glob.glob(args.run_glob) if Path(p).is_dir()))
    dirs = [d for d in dirs if (d / "combo.json").exists() and (d / args.ckpt).exists()]
    if not dirs:
        sys.exit("tidak ada run yang cocok (butuh combo.json dan " + args.ckpt + ")")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[frozen-eval] {len(dirs)} run, split={splits}, ckpt={args.ckpt}, device={device}\n")

    results = []
    for i, d in enumerate(dirs, 1):
        r = eval_one(d, splits, args.ckpt, device)
        results.append(r)
        parts = " | ".join(
            f"{s}: RMSE={r['splits'][s].get('RMSE', float('nan')):6.2f} "
            f"MAE={r['splits'][s].get('MAE', float('nan')):6.2f} "
            f"mSpRMSE={r['splits'][s].get('mean_per_species_RMSE', float('nan')):6.2f} "
            f"mSpR2={r['splits'][s].get('mean_per_species_R2', float('nan'))*100:6.2f}%"
            for s in splits if s in r["splits"])
        print(f"  [{i:>2}/{len(dirs)}] {d.name:<44} {parts}", flush=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{args.out_name}.json"
    with open(out, "w") as f:
        json.dump({"ckpt": args.ckpt, "splits": splits,
                   "runs": results, "aggregate": aggregate(results, splits)}, f, indent=2)
    print(f"\nTersimpan: {out}")

    print("\n=== rata-rata antar-seed (split terakhir) ===")
    s = splits[-1]
    print(f"{'subset':<14}{'depth':>6}{'seed':>5}{'RMSE':>9}{'MAE':>8}"
          f"{'mSpRMSE':>10}{'mSpR2':>9}{'rentang mSpR2':>15}")
    for row in aggregate(results, splits):
        v = row["splits"].get(s, {})
        if not v:
            continue
        print(f"{str(row['species_subset']):<14}{row['max_depth']:>6.0f}{row['n_seed']:>5}"
              f"{v['RMSE']['mean']:>9.2f}{v['MAE']['mean']:>8.2f}"
              f"{v['mean_per_species_RMSE']['mean']:>10.2f}"
              f"{v['mean_per_species_R2']['mean']*100:>8.2f}%"
              f"{v['mean_per_species_R2']['spread']*100:>14.2f}pp")


if __name__ == "__main__":
    main()
