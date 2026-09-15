#!/usr/bin/env python3
"""
eval_joint_trunkroi_dbh_percheckpoint.py — Eval DBH pada SEMUA checkpoint sebuah job joint
trunk-ROI, supaya kualitas diameter bisa diplot per iterasi berdampingan dengan AP50 segmentasi.

Bedanya dengan eval_dbh_percheckpoint.py: skrip itu untuk head `dbh_embed` bawaan decoder,
sedangkan job joint di laporan ini memakai HybridTrunkROIDBHHead (submodule eksternal), jadi
pipeline eval-nya mengikuti eval_joint_trunkroi_dbh.py (forward native-res, bukan LSJ 640x640).

Model dibangun SEKALI lalu weights-nya di-reload tiap checkpoint, jadi jauh lebih murah daripada
memanggil eval_joint_trunkroi_dbh.py berulang kali.

Output: reports/dbh_eval/dbh_percheckpoint_<output_dir>.json
    {"output_dir":..., "species_subset":[...], "table":[
        {"ckpt":"model_0002499.pth", "iteration":2499, "N":..., "R2":...,
         "mean_per_species_R2":..., "MAE":..., "bias":...}, ...]}

Jalankan (nohup di node VS Code, bukan SLURM):
    nohup /scratch2/pr65/anur0018/conda_envs/maskdino/bin/python -u \\
        scripts/eval_joint_trunkroi_dbh_percheckpoint.py \\
        --output-dir FocalNet_L_trunk_roi_dbh_hybrid_joint_from135k_11species_20k \\
        --species-subset 1,3,4,5,6,7,8,9,10,12,13 \\
        > logs/eval_percheckpoint_11sp.log 2>&1 &
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
OUTPUT_ROOT = PROJECT_ROOT.parent / "maskdino_output"
REPORT_DIR = PROJECT_ROOT / "reports" / "dbh_eval"

sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from detectron2.utils.logger import setup_logger  # noqa: E402

from eval_joint_trunkroi_dbh import build_model_with_head, evaluate, r2_mae  # noqa: E402
from register_combined import register_all_combined  # noqa: E402

MIN_N_PER_SPECIES = 30  # konsisten dengan kriteria di seluruh laporan


def iter_of(ckpt_name: str) -> int:
    """model_0002499.pth -> 2499 ; model_final.pth -> None (diisi belakangan)."""
    m = re.search(r"model_(\d+)\.pth", ckpt_name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="nama dir di maskdino_output/")
    ap.add_argument("--species-subset", required=True, help="mis. 3,5,8,9,10,12,13")
    ap.add_argument("--dataset-test", default="combined_inst_rle_f1000_repaired_v4_stratified_test")
    args = ap.parse_args()

    setup_logger()
    import os
    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_combined(os.environ["COMBINED_ROOT"])

    out_dir = OUTPUT_ROOT / args.output_dir
    subset = set(int(x) for x in args.species_subset.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Bangun model 1x (memuat model_final.pth), lalu reload weights tiap checkpoint.
    model, _cfg = build_model_with_head(out_dir, device)
    dataset_dicts = DatasetCatalog.get(args.dataset_test)

    ckpts = sorted(p.name for p in out_dir.glob("model_*.pth") if p.name != "model_final.pth")
    ckpts.append("model_final.pth")
    print(f"[percheckpoint] {len(ckpts)} checkpoint di {out_dir.name}")

    table = []
    for name in ckpts:
        DetectionCheckpointer(model).load(str(out_dir / name))
        model.eval()
        per_pred, per_gt = evaluate(model, dataset_dicts, subset, device)

        # metrik keseluruhan
        all_pred = [v for cid in per_pred for v in per_pred[cid]]
        all_gt = [v for cid in per_gt for v in per_gt[cid]]
        overall = r2_mae(all_pred, all_gt)

        # mean per-spesies, hanya spesies dengan N >= MIN_N_PER_SPECIES
        per_sp = []
        for cid in sorted(per_pred):
            if len(per_pred[cid]) >= MIN_N_PER_SPECIES:
                per_sp.append(r2_mae(per_pred[cid], per_gt[cid])["R2"])
        mean_sp = sum(per_sp) / len(per_sp) if per_sp else float("nan")

        row = {
            "ckpt": name,
            "iteration": iter_of(name),
            "N": overall["N"],
            "R2": overall["R2"],
            "mean_per_species_R2": mean_sp,
            "n_species_scored": len(per_sp),
            "MAE": overall["MAE"],
            "bias": overall["bias"],
        }
        table.append(row)
        it = row["iteration"] if row["iteration"] is not None else "final"
        print(f"  {name:<22} iter={str(it):>7}  N={row['N']:>5}  "
              f"R2={row['R2']*100:6.2f}%  meanSp={mean_sp*100:6.2f}%  MAE={row['MAE']:6.2f}mm")

    # model_final = iterasi terakhir; samakan nomornya dengan checkpoint numerik tertinggi + 1
    nums = [r["iteration"] for r in table if r["iteration"] is not None]
    if nums:
        for r in table:
            if r["iteration"] is None:
                r["iteration"] = max(nums) + 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"dbh_percheckpoint_{args.output_dir}.json"
    with open(out_path, "w") as f:
        json.dump({"output_dir": args.output_dir,
                   "species_subset": sorted(subset),
                   "min_n_per_species": MIN_N_PER_SPECIES,
                   "table": table}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
