#!/usr/bin/env python3
"""eval_dbh_maskrcnn_percheckpoint.py — Eval DBH pada SEMUA checkpoint Mask R-CNN DBHROIHeads,
val+test sekaligus. Analog dari eval_joint_trunkroi_dbh_percheckpoint_dualsplit.py (MaskDINO),
tapi memakai infrastruktur eval_dbh_maskrcnn.py (matching IoU + DBHROIHeads.pred_dbh langsung,
bukan lewat dbh_embed decoder).

Dibuat 31 Agt 2026 -- sebelum ini Mask R-CNN cuma pernah dievaluasi di satu checkpoint
(model_final.pth), padahal head DBH terbukti non-monoton di semua model lain (checkpoint
terbaik jarang di checkpoint terakhir). Perlu sweep penuh sebelum checkpoint boleh di-trim.

Model dibangun & data loader dibangun SEKALI, weights di-reload per checkpoint -- jauh lebih
murah daripada memanggil eval_dbh_maskrcnn.py 64x (32 checkpoint x 2 split) yang tiap kali
rebuild dataset loader dari nol.

Usage:
    python scripts/eval_dbh_maskrcnn_percheckpoint.py \\
        --output-dir maskrcnn_R50_combined_rgbd_dbh_scratch_v4_stratified_155k_cap30_detach \\
        --config configs/maskrcnn_R50_combined_rle_f1000_rgbd_dbh_scratch_v4_stratified_155k_cap30_detach.yaml \\
        --max-depth 30.0 \\
        --ckpt-stride 8 --ckpt-offset 0
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT.parent / "maskdino_output"
REPORT_DIR = PROJECT_ROOT / "reports" / "dbh_eval"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import build_detection_test_loader, MetadataCatalog  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.utils.logger import setup_logger  # noqa: E402

from register_rainforests import register_all_rainforests  # noqa: E402
from register_plantations import register_all_plantations  # noqa: E402
from register_combined import register_all_combined  # noqa: E402
from maskrcnn_rgbd_dbh import add_rgbd_dbh_config, RGBDDatasetMapper  # noqa: E402

from eval_dbh_maskrcnn import build_gt_index, infer_with_dbh, match_predictions, compute_metrics  # noqa: E402

MIN_N_PER_SPECIES = 30


def iter_of(ckpt_name: str):
    m = re.search(r"model_(\d+)\.pth", ckpt_name)
    return int(m.group(1)) if m else None


def merge_shards(output_dir_name: str):
    shards = sorted(REPORT_DIR.glob(f"dbh_percheckpoint_dualsplit_maskrcnn_{output_dir_name}__shard*.json"))
    if not shards:
        sys.exit(f"tidak ada shard untuk {output_dir_name} di {REPORT_DIR}")
    rows, meta = {}, None
    for p in shards:
        d = json.load(open(p))
        meta = meta or d
        for r in d["table"]:
            rows[r["ckpt"]] = r
        print(f"  {p.name}: {len(d['table'])} checkpoint")
    table = sorted(rows.values(), key=lambda r: (r["iteration"] is None, r["iteration"]))
    out = REPORT_DIR / f"dbh_percheckpoint_dualsplit_maskrcnn_{output_dir_name}.json"
    with open(out, "w") as f:
        json.dump({"output_dir": output_dir_name, "n_shards": len(shards), "table": table}, f, indent=2)
    print(f"\nGabungan {len(table)} checkpoint -> {out}")
    ok = [r for r in table if r["val"]["RMSE_mm"] is not None]
    if ok:
        best = min(ok, key=lambda r: r["val"]["RMSE_mm"])
        last = ok[-1]
        print(f"  val RMSE optimum {best['val']['RMSE_mm']:.2f}mm @iter {best['iteration']}"
              f"   akhir {last['val']['RMSE_mm']:.2f}mm @iter {last['iteration']}")
    ok_t = [r for r in table if r["test"]["RMSE_mm"] is not None]
    if ok_t:
        best = min(ok_t, key=lambda r: r["test"]["RMSE_mm"])
        last = ok_t[-1]
        print(f"  test RMSE optimum {best['test']['RMSE_mm']:.2f}mm @iter {best['iteration']}"
              f"   akhir {last['test']['RMSE_mm']:.2f}mm @iter {last['iteration']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="nama dir di maskdino_output/")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-depth", type=float, default=30.0)
    ap.add_argument("--ckpt-stride", type=int, default=1)
    ap.add_argument("--ckpt-offset", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge_shards(args.output_dir)
        return

    setup_logger()
    os.environ.setdefault("RAINFORESTS_ROOT", str(PROJECT_ROOT / "data" / "rainforests"))
    os.environ.setdefault("PLANTATIONS_ROOT", str(PROJECT_ROOT / "data" / "plantations"))
    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_rainforests(os.environ["RAINFORESTS_ROOT"])
    register_all_plantations(os.environ["PLANTATIONS_ROOT"])
    register_all_combined(os.environ["COMBINED_ROOT"])

    out_dir = OUTPUT_ROOT / args.output_dir
    ckpts = sorted(p.name for p in out_dir.glob("model_*.pth") if p.name != "model_final.pth")
    if (out_dir / "model_final.pth").exists():
        ckpts.append("model_final.pth")
    if not ckpts:
        sys.exit(f"tidak ada checkpoint di {out_dir}")

    cfg = get_cfg()
    add_rgbd_dbh_config(cfg)
    cfg.merge_from_file(str(args.config))
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()

    model = build_model(cfg)
    print(f"[model] dibangun sekali, device={cfg.MODEL.DEVICE}")

    gt_index_val, ds_val = build_gt_index(cfg, "val")
    gt_index_test, ds_test = build_gt_index(cfg, "test")
    mapper = RGBDDatasetMapper(cfg, is_train=False)
    loader_val = build_detection_test_loader(cfg, ds_val, mapper=mapper)
    loader_test = build_detection_test_loader(cfg, ds_test, mapper=mapper)
    print(f"[data] val: {len(gt_index_val)} gambar ber-anotasi, test: {len(gt_index_test)} gambar ber-anotasi")

    n_all = len(ckpts)
    if args.ckpt_stride > 1:
        ckpts = ckpts[args.ckpt_offset :: args.ckpt_stride]
    print(f"[percheckpoint] worker {args.ckpt_offset+1}/{args.ckpt_stride}: "
          f"{len(ckpts)} dari {n_all} checkpoint di {out_dir.name}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"__shard{args.ckpt_offset}of{args.ckpt_stride}" if args.ckpt_stride > 1 else ""
    out_path = REPORT_DIR / f"dbh_percheckpoint_dualsplit_maskrcnn_{args.output_dir}{suffix}.json"

    table, t0 = [], time.time()
    for k, name in enumerate(ckpts, 1):
        muat = DetectionCheckpointer(model).load(str(out_dir / name))
        model.eval()

        it_ckpt = iter_of(name)
        if it_ckpt is None and isinstance(muat, dict):
            it_ckpt = muat.get("iteration")

        results_val = infer_with_dbh(model, loader_val, gt_index_val, cfg.INPUT.DEPTH_DIR, args.max_depth)
        results_test = infer_with_dbh(model, loader_test, gt_index_test, cfg.INPUT.DEPTH_DIR, args.max_depth)
        pairs_val = match_predictions(results_val, max_depth=args.max_depth)
        pairs_test = match_predictions(results_test, max_depth=args.max_depth)
        m_val = compute_metrics(pairs_val)
        m_test = compute_metrics(pairs_test)

        row = {"ckpt": name, "iteration": it_ckpt, "val": m_val, "test": m_test}
        table.append(row)

        el = time.time() - t0
        it = row["iteration"] if row["iteration"] is not None else "final"
        print(f"  [{k:>2}/{len(ckpts)}] {name:<22} iter={str(it):>7}  "
              f"val RMSE={m_val.get('RMSE_mm', float('nan')):6.2f}mm N={m_val.get('N',0):4d} | "
              f"test RMSE={m_test.get('RMSE_mm', float('nan')):6.2f}mm N={m_test.get('N',0):4d}"
              f"   ({el/k:.0f}s/ckpt, sisa ~{(len(ckpts)-k)*el/k/60:.0f}m)", flush=True)

        nums = [r["iteration"] for r in table if r["iteration"] is not None]
        for r in table:
            if r["iteration"] is None and nums:
                r["iteration"] = max(nums) + 1
                r["iteration_ditebak"] = True
        with open(out_path, "w") as f:
            json.dump({"output_dir": args.output_dir, "table": table}, f, indent=2)

    print(f"\nSelesai dalam {(time.time()-t0)/60:.1f} menit. Tersimpan: {out_path}")


if __name__ == "__main__":
    main()
