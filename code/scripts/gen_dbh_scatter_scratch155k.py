#!/usr/bin/env python3
"""
gen_dbh_scatter_scratch155k.py -- Rekonstruksi prediksi per-instance (pred_mm, gt_mm, species)
untuk model Joint 11-spesies from-scratch 155k, pada checkpoint RMSE terbaik (iter 114.999) dan
checkpoint akhir (model_final.pth), test split. Dipakai untuk scatter plot pada laporan khusus
model ini.

Reuse langsung build_model_with_head()/evaluate() dari eval_joint_trunkroi_dbh.py, sama seperti
gen_dbh_scatter_frozen_vs_joint.py, tapi param init_ckpt diarahkan ke checkpoint yang dipilih.

Output: reports/dbh_eval/dbh_scatter_scratch155k.json
    { "best":  {"pred_mm":[...], "gt_mm":[...], "species":[...], "ckpt": "model_0114999.pth"},
      "final": {"pred_mm":[...], "gt_mm":[...], "species":[...], "ckpt": "model_final.pth"} }

Usage:
    python scripts/gen_dbh_scatter_scratch155k.py
"""
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
OUTPUT_ROOT = PROJECT_ROOT.parent / "maskdino_output"
REPORT_DIR = PROJECT_ROOT / "reports" / "dbh_eval"

sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

CAT_NAMES = {
    1: "Apple", 3: "Loquat", 4: "Mango", 5: "Orange", 6: "Persimmon", 7: "Pomegranate",
    8: "AliiFig", 9: "BangaloPalm", 10: "Fern", 12: "RubberFig", 13: "Umbrella",
}
SPECIES_11 = tuple(sorted(CAT_NAMES.keys()))

OUTPUT_DIR = OUTPUT_ROOT / "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k"
CHECKPOINTS = {
    "best": "model_0114999.pth",
    "final": "model_final.pth",
}


def gen_for_ckpt(ckpt_name, dataset_dicts, device):
    from eval_joint_trunkroi_dbh import build_model_with_head, evaluate
    model, _ = build_model_with_head(OUTPUT_DIR, device, init_ckpt=ckpt_name)
    per_species_pred, per_species_gt = evaluate(model, dataset_dicts, set(SPECIES_11), device)
    preds, gts, cats = [], [], []
    for cid in per_species_pred:
        preds.extend(per_species_pred[cid])
        gts.extend(per_species_gt[cid])
        cats.extend([cid] * len(per_species_pred[cid]))
    print(f"[{ckpt_name}] N predicted = {len(preds)}")
    del model
    torch.cuda.empty_cache()
    return {"pred_mm": preds, "gt_mm": gts, "species": cats, "ckpt": ckpt_name}


def main():
    from detectron2.data import DatasetCatalog
    from detectron2.utils.logger import setup_logger
    from register_combined import register_all_combined

    setup_logger()
    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_combined(os.environ["COMBINED_ROOT"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_dicts = DatasetCatalog.get("combined_inst_rle_f1000_repaired_v4_stratified_test")

    out = {}
    for label, ckpt_name in CHECKPOINTS.items():
        print(f"=== Generating {label} ({ckpt_name}) ===")
        out[label] = gen_for_ckpt(ckpt_name, dataset_dicts, device)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "dbh_scatter_scratch155k.json"
    out["species_names"] = CAT_NAMES
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
