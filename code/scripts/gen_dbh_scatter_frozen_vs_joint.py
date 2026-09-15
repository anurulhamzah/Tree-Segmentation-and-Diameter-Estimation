#!/usr/bin/env python3
"""
gen_dbh_scatter_frozen_vs_joint.py — Rekonstruksi prediksi per-instance (pred_mm, gt_mm,
species) untuk scatter plot Frozen vs Joint, subset 7-spesies, test split.

Frozen: reuse dbh_sandbox_head.py punya build_feature_vec/FeatureCache/load_index, checkpoint
        gapcheck_rf5_loquat_orange_seed0/head_best.pth (kombinasi = combo.json trial itu).
Joint:  reuse eval_joint_trunkroi_dbh.py punya evaluate()/build_model_with_head(), checkpoint
        FocalNet_L_trunk_roi_dbh_hybrid_joint_7species_20k/model_final.pth.

Output: reports/dbh_eval/dbh_scatter_frozen_vs_joint_7species.json
    { "frozen": {"pred_mm":[...], "gt_mm":[...], "species":[...]},
      "joint":  {"pred_mm":[...], "gt_mm":[...], "species":[...]} }

Usage:
    python scripts/gen_dbh_scatter_frozen_vs_joint.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
OUTPUT_ROOT = PROJECT_ROOT.parent / "maskdino_output"
REPORT_DIR = PROJECT_ROOT / "reports" / "dbh_eval"

sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "dbh_sandbox"))

CAT_NAMES = {
    1: "Apple", 3: "Loquat", 4: "Mango", 5: "Orange", 6: "Persimmon", 7: "Pomegranate",
    8: "AliiFig", 9: "BangaloPalm", 10: "Fern", 12: "RubberFig", 13: "Umbrella",
}
SPECIES_7 = (3, 5, 8, 9, 10, 12, 13)


def gen_frozen():
    import filters as F_
    from dbh_sandbox_head import FeatureCache, load_index, build_feature_vec
    from trunk_roi_dbh_head_hybrid import HybridTrunkROIDBHHead

    cache_dir = OUTPUT_ROOT / "dbh_sandbox_cache"
    ckpt = OUTPUT_ROOT / "sweep_dbh_hyperparams" / "gapcheck_rf5_loquat_orange_seed0" / "head_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df_raw = load_index(cache_dir, "test")
    test_df = F_.apply_filters(
        test_df_raw, species_subset=SPECIES_7, min_trunk_px=5, min_depth=1.0, max_depth=20.0,
        wh_tolerance=0.5, ratio_bounds=None, rubberfig_cap_cm=150.0,
    )
    print(f"[frozen] test N (post-filter) = {len(test_df)}")

    head = HybridTrunkROIDBHHead(in_channels=192, hidden=256, strip_rows=5,
                                  num_species=13, dropout=0.3).to(device)
    head.load_state_dict(torch.load(ckpt, map_location=device))
    head.eval()

    feat_cache = FeatureCache(cache_dir, "test")
    preds, gts, cats = [], [], []
    with torch.no_grad():
        for row in test_df.itertuples():
            x = build_feature_vec(head, feat_cache, row, device, use_geom=True)
            if x is None:
                continue
            pred_log = head.mlp(x).squeeze(-1)
            preds.append(torch.expm1(pred_log).item())
            gts.append(float(row.gt_dbh_mm))
            cats.append(int(row.category_id))
    print(f"[frozen] N predicted = {len(preds)}")
    return {"pred_mm": preds, "gt_mm": gts, "species": cats}


def gen_joint():
    import os
    from detectron2.data import DatasetCatalog
    from detectron2.utils.logger import setup_logger
    from eval_joint_trunkroi_dbh import build_model_with_head, evaluate
    from register_combined import register_all_combined

    setup_logger()
    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_combined(os.environ["COMBINED_ROOT"])

    output_dir = OUTPUT_ROOT / "FocalNet_L_trunk_roi_dbh_hybrid_joint_7species_20k"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = build_model_with_head(output_dir, device)
    dataset_dicts = DatasetCatalog.get("combined_inst_rle_f1000_repaired_v4_stratified_test")

    per_species_pred, per_species_gt = evaluate(model, dataset_dicts, set(SPECIES_7), device)
    preds, gts, cats = [], [], []
    for cid in per_species_pred:
        preds.extend(per_species_pred[cid])
        gts.extend(per_species_gt[cid])
        cats.extend([cid] * len(per_species_pred[cid]))
    print(f"[joint] N predicted = {len(preds)}")
    return {"pred_mm": preds, "gt_mm": gts, "species": cats}


def main():
    print("=== Generating FROZEN 7-species predictions ===")
    frozen = gen_frozen()
    print("=== Generating JOINT 7-species predictions ===")
    joint = gen_joint()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "dbh_scatter_frozen_vs_joint_7species.json"
    with open(out_path, "w") as f:
        json.dump({"frozen": frozen, "joint": joint, "species_names": CAT_NAMES}, f)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
