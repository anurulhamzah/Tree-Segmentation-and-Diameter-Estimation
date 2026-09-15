#!/usr/bin/env python3
"""Eval DBH untuk SEMUA checkpoint sebuah model dbh_embed (bukan trunk-roi), test split v4.

Kenapa ada: head DBH (dbh_embed) itu NON-MONOTON — val metric sering memuncak di tengah lalu
turun (overfit N-per-spesies kecil), jadi model_final BELUM TENTU checkpoint terbaik. Sebelum
trim checkpoint model DBH, jalankan ini untuk cari best sesungguhnya. Contoh nyata (29 Jul 2026):
dbh_joint_plainext_20k best = iter 7499 (MAE 14,7mm/R² 0,59), JAUH lebih baik dari final
(21,7mm/0,36). Lihat [[feedback-dbh-checkpoint-trim]].

Reuse fungsi dari eval_dbh_v4_test.py (load, infer_with_dbh, match_predictions, compute_metrics).
Build model 1x, reload weights tiap checkpoint (efisien). Metrik seleksi utama: MAE + R² overall
(robust); meanPerSpR2 juga dilaporkan tapi noisy (N-per-spesies kecil).

Jalankan (nohup di A40, BUKAN SLURM):
  cd /scratch2/pr65/anur0018/tree_classification
  nohup /scratch2/pr65/anur0018/conda_envs/maskdino/bin/python \
    scripts/eval_dbh_percheckpoint.py --output-dir FocalNet_L_dbh_joint_plainext_20k \
    > logs/eval_dbh_percheckpoint.log 2>&1 &
"""
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
OUTPUT_ROOT   = PROJECT_ROOT.parent / "maskdino_output"
sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import eval_dbh_v4_test as E
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from maskdino import add_maskdino_config
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import COCOInstanceDBHDatasetMapper
from register_combined import register_all_combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="nama dir di maskdino_output/")
    ap.add_argument("--dataset", default="combined_inst_rle_f1000_repaired_v4_stratified_test")
    ap.add_argument("--min-n", type=int, default=10, help="min pasangan per-spesies utk R2 spesies")
    args = ap.parse_args()

    setup_logger()
    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_combined(os.environ["COMBINED_ROOT"])

    out_dir = OUTPUT_ROOT / args.output_dir
    ckpts = sorted(f for f in os.listdir(out_dir) if f.startswith("model_") and f.endswith(".pth"))
    print(f"checkpoints ({len(ckpts)}): {ckpts}", flush=True)

    cfg = get_cfg(); add_deeplab_config(cfg); add_maskdino_config(cfg)
    cfg.merge_from_file(str(out_dir / "config.yaml"))
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR = str(out_dir); cfg.freeze()
    model = build_model(cfg)

    cat_names = MetadataCatalog.get(args.dataset).thing_classes
    loader = build_detection_test_loader(cfg, args.dataset,
                                         mapper=COCOInstanceDBHDatasetMapper(cfg, is_train=False))
    gt_index = E.build_gt_index(args.dataset)
    print(f"images: {len(loader)}  classes: {len(cat_names)}", flush=True)

    table = {}
    for ck in ckpts:
        DetectionCheckpointer(model).load(str(out_dir / ck)); model.eval()
        results = E.infer_with_dbh(model, loader, gt_index)
        pairs = E.match_predictions(results, cat_names)
        overall = E.compute_metrics(pairs)
        by = defaultdict(list)
        for p, g, c in pairs: by[c].append((p, g, c))
        persp = {c: E.compute_metrics(by[c]) for c in by}
        r2s = [persp[c]["R2"] for c in persp if persp[c]["N"] >= args.min_n and persp[c]["R2"] is not None]
        mean_sp_r2 = float(np.mean(r2s)) if r2s else None
        table[ck] = {"N": overall["N"], "R2_overall": overall["R2"], "MAE": overall["MAE"],
                     "meanPerSpR2": mean_sp_r2, "n_species_used": len(r2s)}
        msr = "None" if mean_sp_r2 is None else f"{mean_sp_r2:.4f}"
        print(f"[{ck}] N={overall['N']} R2_all={overall['R2']:.4f} MAE={overall['MAE']:.1f} "
              f"meanPerSpR2={msr} (nsp={len(r2s)})", flush=True)

    # ranking: MAE naik (primer robust), lalu R2_overall
    order = sorted(table, key=lambda k: (table[k]["MAE"] if table[k]["MAE"] is not None else 1e9))
    print("\n==== RANKING (MAE asc; R2_overall & meanPerSpR2 sbg konteks) ====", flush=True)
    for k in order:
        v = table[k]
        print(f"  {k:20s} MAE={v['MAE']:.1f}  R2_all={v['R2_overall']:.4f}  meanPerSpR2={v['meanPerSpR2']}")
    best = order[0] if order else None
    print(f"\nBEST (MAE) = {best}   final==best? {best=='model_final.pth'}")

    outp = PROJECT_ROOT / "reports" / "dbh_eval" / f"{args.output_dir}_percheckpoint.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"dir": args.output_dir, "min_n_per_species": args.min_n,
               "best_by_MAE": best, "table": table}, open(outp, "w"), indent=2)
    print(f"saved {outp}")


if __name__ == "__main__":
    main()
