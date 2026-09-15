#!/usr/bin/env python3
"""DBH evaluation untuk Mask R-CNN DBHROIHeads (scripts/maskrcnn_rgbd_dbh/).

Beda dari scripts/eval_dbh.py (khusus MaskDINO, baca outputs["pred_dbh"] dari decoder
query-based) — DBHROIHeads sudah menempelkan `pred_dbh` langsung ke Instances hasil
inference standar Mask R-CNN, jadi cukup panggil model secara normal lalu ambil field itu.
Logika IoU-matching & perhitungan metrik meniru eval_dbh.py supaya angka sebanding.

CATATAN JUJUR: populasi evaluasi TIDAK bisa disamakan persis dengan N=1790/1640 milik model
final (FocalNet-L, Table 4/6 naskah) — itu pakai filter geometris trunk-row breast-height 20m
via HybridTrunkROIDBHHead. Di sini jarak instance diperkirakan dari median depth mentah di
dalam gt_mask (proxy sederhana, sama seperti filter DBH_MAX_DEPTH saat training), bukan
estimasi geometris. N aktual dilaporkan apa adanya, jangan diasumsikan sama.

Usage (dari PROJECT_ROOT):
    python scripts/eval_dbh_maskrcnn.py \
        --config configs/maskrcnn_R50_combined_rle_f1000_rgbd_dbh_scratch_v4_stratified_155k_cap30.yaml \
        --checkpoint /scratch2/pr65/anur0018/maskdino_output/maskrcnn_R50_combined_rgbd_dbh_scratch_v4_stratified_155k_cap30/model_final.pth \
        --split test --max-depth 20.0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from pycocotools import mask as maskUtils

from register_rainforests import register_all_rainforests
from register_plantations import register_all_plantations
from register_combined import register_all_combined
from maskrcnn_rgbd_dbh import add_rgbd_dbh_config, RGBDDatasetMapper
from maskrcnn_rgbd_dbh.dataset_mapper import _read_pfm

SCORE_THRESH = 0.5
IOU_THRESH = 0.5


def load_model_and_cfg(config_path, checkpoint_path):
    cfg = get_cfg()
    add_rgbd_dbh_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(checkpoint_path))
    model.eval()
    print(f"[model] loaded from {checkpoint_path} on {cfg.MODEL.DEVICE}")
    return model, cfg


def build_gt_index(cfg, split):
    dataset_name = cfg.DATASETS.TEST[0]
    if split == "test":
        dataset_name = dataset_name.replace("_val", "_test")
    json_file = MetadataCatalog.get(dataset_name).json_file
    with open(json_file) as f:
        coco = json.load(f)
    cat_id_to_cls = {c["id"]: i for i, c in enumerate(coco["categories"])}
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    RUBBERFIG_CAT_ID = 12
    RUBBERFIG_CAP_MM = 1500.0
    gt_index = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        seg = ann.get("segmentation")
        if seg is None:
            continue
        img_info = next(im for im in coco["images"] if im["id"] == iid)
        if isinstance(seg, dict):
            rle = seg
        else:
            rle = maskUtils.merge(maskUtils.frPyObjects(seg, img_info["height"], img_info["width"]))
        # Field anotasi "dbh" dalam CM (lihat catatan di dataset_mapper.py) -> x10 utk mm.
        dbh_mm = float(ann.get("dbh", 0.0) or 0.0) * 10.0
        # RubberFig cap 150cm=1500mm, sama persis dgn filter dataset_mapper.py (training) --
        # tanpa ini populasi eval tidak konsisten dgn apa yg model pelajari, dan RMSE
        # dirusak oleh ~15 instance RubberFig ekstrem (GT sampai 754cm). Instance yg kena
        # cap diperlakukan sama seperti vine: dbh=0, di-skip match_predictions() (gt_dbh<=0).
        if ann["category_id"] == RUBBERFIG_CAT_ID and dbh_mm >= RUBBERFIG_CAP_MM:
            dbh_mm = 0.0
        gt_index.setdefault(iid, []).append({
            "class": cat_id_to_cls.get(ann["category_id"], -1),
            "cat_name": cat_id_to_name.get(ann["category_id"], "?"),
            "dbh": dbh_mm,
            "rle": rle,
        })
    return gt_index, dataset_name


@torch.no_grad()
def infer_with_dbh(model, data_loader, gt_index, depth_dir, max_depth):
    """Jalankan inference standar Mask R-CNN, ambil pred_dbh dari Instances, hitung jarak
    proxy GT (median depth mentah dlm gt_mask) utk subset opsional <= max_depth."""
    results = []
    for i, batch in enumerate(data_loader):
        if (i + 1) % 100 == 0:
            print(f"  {i+1} gambar diproses...")
        outputs = model(batch)
        inst = outputs[0]["instances"].to("cpu")

        pred_masks = inst.pred_masks.numpy().astype(np.uint8) if inst.has("pred_masks") else np.zeros((0,))
        pred_rle = [maskUtils.encode(np.asfortranarray(m)) for m in pred_masks]
        scores = inst.scores.numpy() if inst.has("scores") else np.array([])
        classes = inst.pred_classes.numpy() if inst.has("pred_classes") else np.array([])
        pred_dbh = inst.pred_dbh.numpy() if inst.has("pred_dbh") else np.zeros(len(inst))

        keep = scores >= SCORE_THRESH
        image_id = batch[0]["image_id"]
        gt_anns = gt_index.get(image_id, [])

        # Jarak proxy per GT instance (median depth mentah dlm mask), utk filter opsional
        gt_dist = [None] * len(gt_anns)
        if max_depth is not None and gt_anns:
            file_name = batch[0]["file_name"]
            stem = os.path.splitext(os.path.basename(file_name))[0]
            pfm_path = os.path.join(depth_dir, stem + ".pfm")
            if os.path.exists(pfm_path):
                depth_raw = _read_pfm(pfm_path)
                for j, a in enumerate(gt_anns):
                    m = maskUtils.decode(a["rle"]).astype(bool)
                    if m.shape != depth_raw.shape:
                        continue
                    region = depth_raw[m]
                    valid = region[region < 65000.0]
                    gt_dist[j] = float(np.median(valid)) if valid.size > 0 else None

        results.append({
            "image_id": image_id,
            "pred_scores": scores[keep],
            "pred_classes": classes[keep],
            "pred_rle": [r for r, k in zip(pred_rle, keep) if k],
            "pred_dbh_mm": pred_dbh[keep],
            "gt_classes": np.array([a["class"] for a in gt_anns], dtype=np.int64),
            "gt_dbh": np.array([a["dbh"] for a in gt_anns], dtype=np.float32),
            "gt_rle": [a["rle"] for a in gt_anns],
            "gt_cat_name": [a["cat_name"] for a in gt_anns],
            "gt_dist": gt_dist,
        })
    return results


def match_predictions(results, max_depth=None):
    """Greedy IoU match. Kembalikan list (pred_dbh_mm, gt_dbh_mm, cat_name, dist_or_None)."""
    pairs = []
    for r in results:
        n_gt, n_pred = len(r["gt_rle"]), len(r["pred_rle"])
        if n_gt == 0 or n_pred == 0:
            continue
        gt_matched = np.zeros(n_gt, dtype=bool)
        order = np.argsort(-r["pred_scores"])
        for p_idx in order:
            ious = maskUtils.iou([r["pred_rle"][p_idx]], r["gt_rle"], [0] * n_gt)[0]
            ious[gt_matched] = 0.0
            best_gt = int(np.argmax(ious))
            if ious[best_gt] >= IOU_THRESH:
                gt_matched[best_gt] = True
                gt_dbh = r["gt_dbh"][best_gt]
                if gt_dbh <= 0:
                    continue
                dist = r["gt_dist"][best_gt]
                if max_depth is not None and (dist is None or dist > max_depth):
                    continue
                pairs.append((
                    float(r["pred_dbh_mm"][p_idx]), float(gt_dbh),
                    r["gt_cat_name"][best_gt], dist,
                ))
    return pairs


def compute_metrics(pairs):
    if len(pairs) == 0:
        return {"N": 0, "MAE": None, "RMSE": None, "R2": None, "bias": None}
    pred = np.array([p[0] for p in pairs])
    gt = np.array([p[1] for p in pairs])
    err = pred - gt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    return {
        "N": len(pairs),
        "MAE_mm": float(np.mean(np.abs(err))),
        "RMSE_mm": float(np.sqrt(np.mean(err ** 2))),
        "bias_mm": float(np.mean(err)),
        "R2": 1.0 - ss_res / (ss_tot + 1e-10),
    }


def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--max-depth", type=float, default=None,
                     help="Kalau diisi, tambahkan subset laporan dgn jarak proxy GT <= nilai ini (m).")
    ap.add_argument("--out", default=None, help="Path JSON output (default: reports/dbh_eval/...)")
    args = ap.parse_args()

    os.environ.setdefault("RAINFORESTS_ROOT", str(PROJECT_ROOT / "data" / "rainforests"))
    os.environ.setdefault("PLANTATIONS_ROOT", str(PROJECT_ROOT / "data" / "plantations"))
    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_rainforests(os.environ["RAINFORESTS_ROOT"])
    register_all_plantations(os.environ["PLANTATIONS_ROOT"])
    register_all_combined(os.environ["COMBINED_ROOT"])

    model, cfg = load_model_and_cfg(args.config, args.checkpoint)
    gt_index, dataset_name = build_gt_index(cfg, args.split)
    meta = MetadataCatalog.get(dataset_name)
    print(f"[data] dataset: {dataset_name}, {len(gt_index)} gambar ber-anotasi")

    mapper = RGBDDatasetMapper(cfg, is_train=False)
    loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    print(f"[infer] menjalankan inference ({len(loader)} gambar)...")
    results = infer_with_dbh(model, loader, gt_index, cfg.INPUT.DEPTH_DIR, args.max_depth)

    pairs_all = match_predictions(results, max_depth=None)
    m_all = compute_metrics(pairs_all)
    print(f"\n=== METRIK DBH — SELURUH POPULASI (tanpa cap jarak) ===")
    print(f"N={m_all['N']}", m_all)

    report = {"config": str(args.config), "checkpoint": str(args.checkpoint),
              "split": args.split, "overall_uncapped": m_all}

    if args.max_depth is not None:
        pairs_capped = match_predictions(results, max_depth=args.max_depth)
        m_capped = compute_metrics(pairs_capped)
        print(f"\n=== METRIK DBH — subset jarak proxy <= {args.max_depth}m ===")
        print(f"N={m_capped['N']}", m_capped)
        report[f"overall_capped_{args.max_depth}m"] = m_capped

    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "reports" / "dbh_eval" /
        f"dbh_metrics_maskrcnn_{Path(args.checkpoint).parent.name}_{args.split}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
