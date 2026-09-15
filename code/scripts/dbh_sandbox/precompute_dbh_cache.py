"""
precompute_dbh_cache.py — Tier 0 dari DBH sandbox.

Jalan SEKALI: forward frozen FocalNet-L backbone atas semua image (train/val/test)
dengan transform DETERMINISTIK (tanpa LSJ scale-jitter/crop — gambar sumber sudah
native 480x270, cukup di-pad ke kelipatan 32, tidak perlu resize), simpan:
  1. Fitur res2 (fp16) per image -> features/{split}/{image_id}.pt
  2. Tabel index per-instance (Parquet) -> index_{split}.parquet, berisi geometri
     row-1.3m yang dihitung SEKALI dgn threshold paling longgar (tol_wh=inf) — semua
     varian min_trunk_px/max_depth/wh_tolerance/ratio_filter di Tier 1 tinggal
     filter boolean di tabel ini (lihat filters.py), tidak decode ulang mask/PFM.

Cache MENCAKUP SEMUA 13 SPESIES (bukan cuma 11 dlm DEFAULT_SPECIES_UNIVERSE) —
exclude Lemon/LeechVine terjadi di filters.py (Tier 1), bukan di sini, supaya
subset spesies tetap jadi sweep axis yg fleksibel.

Jalankan (uji subset dulu):
  nohup /scratch2/pr65/anur0018/conda_envs/maskdino/bin/python -u \\
      scripts/dbh_sandbox/precompute_dbh_cache.py \\
      --config-file configs/maskdino_FocalNet_L_combined_rle_f1000_rgbd_scratch_dbh_json_v2.yaml \\
      --model-weights /scratch2/pr65/anur0018/maskdino_output/FocalNet_L_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k/model_final.pth \\
      --cache-dir /scratch2/pr65/anur0018/maskdino_output/dbh_sandbox_cache \\
      --limit 50 \\
      > logs/dbh_sandbox_cache_smoketest_nohup.log 2>&1 &
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE          = Path(__file__).resolve().parent          # scripts/dbh_sandbox
SCRIPTS_ROOT  = HERE.parent                                # scripts/
PROJECT_ROOT  = SCRIPTS_ROOT.parent                         # tree_classification/
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(HERE))

from pycocotools import mask as maskUtils

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList

from maskdino import add_maskdino_config
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import (
    _read_pfm, _normalize_depth,
)

from register_combined import register_all_combined
from trunk_roi_dbh_head_hybrid import _find_row_1p3m, _ORIG_H, _ORIG_W, _ORIG_CY, _ORIG_FY, _H_CAM

logger = logging.getLogger("precompute_dbh_cache")

DEPTH_DIR = PROJECT_ROOT / "data" / "combined" / "depth_pfm"


def setup_cfg(config_file: str, model_weights: str):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    if model_weights:
        cfg.MODEL.WEIGHTS = model_weights
    cfg.freeze()
    return cfg


def decode_mask(ann: dict, H_img: int, W_img: int) -> np.ndarray:
    rle = ann["segmentation"]
    if isinstance(rle, list):
        rle = maskUtils.frPyObjects(rle, H_img, W_img)
        rle = maskUtils.merge(rle)
    return maskUtils.decode(rle).astype(bool)


def process_split(split: str, dataset_name: str, backbone, pixel_mean, pixel_std,
                   size_divisibility: int, device, cache_dir: Path, limit: int = 0):
    logger.info(f"=== Split: {split}  (dataset={dataset_name}) ===")
    feat_dir = cache_dir / "features" / split
    feat_dir.mkdir(parents=True, exist_ok=True)

    dataset_dicts = DatasetCatalog.get(dataset_name)
    if limit > 0:
        dataset_dicts = dataset_dicts[:limit]
    logger.info(f"  {len(dataset_dicts)} images")

    records = []
    n_no_depth = 0
    n_row_fail = 0
    n_decode_fail = 0
    n_dbh_null = 0
    t0 = time.time()
    Hp = Wp = None

    for i, rec in enumerate(dataset_dicts):
        file_name = rec["file_name"]
        image_id = rec["image_id"]
        H_img, W_img = rec["height"], rec["width"]

        stem = Path(file_name).stem
        pfm_path = DEPTH_DIR / f"{stem}.pfm"
        if not pfm_path.exists():
            n_no_depth += len(rec.get("annotations", []))
            continue

        image = utils.read_image(file_name, format="RGB")  # (H,W,3) uint8
        depth_raw = _read_pfm(str(pfm_path))                # (H,W) float32, meters
        if depth_raw.shape != (H_img, W_img):
            import cv2
            depth_raw = cv2.resize(depth_raw, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

        depth_norm255 = _normalize_depth(depth_raw) * 255.0  # matches mapper's 4th-channel scale

        image_t = torch.as_tensor(image.transpose(2, 0, 1).copy(), dtype=torch.float32)
        depth_t = torch.as_tensor(depth_norm255, dtype=torch.float32).unsqueeze(0)
        img4 = torch.cat([image_t, depth_t], dim=0)  # (4,H,W)

        img_list = ImageList.from_tensors([img4], size_divisibility)
        if Hp is None:
            Hp, Wp = img_list.tensor.shape[-2:]
            logger.info(f"  Padded size: {H_img}x{W_img} -> {Hp}x{Wp} (size_divisibility={size_divisibility})")

        imgs_norm = (img_list.tensor.to(device) - pixel_mean) / pixel_std
        with torch.no_grad():
            features = backbone(imgs_norm)
            res2 = features["res2"][0]  # (C, Hp/4, Wp/4)
        torch.save(res2.half().cpu(), feat_dir / f"{image_id}.pt")

        # Geometry: pad raw depth (meters) with zeros to (Hp,Wp), top-left aligned
        # (matches ImageList's top-left-origin padding convention).
        depth_pad = np.zeros((Hp, Wp), dtype=np.float32)
        depth_pad[:H_img, :W_img] = depth_raw
        depth_pad_t = torch.as_tensor(depth_pad)

        for ann_idx, ann in enumerate(rec.get("annotations", [])):
            dbh_cm = ann.get("dbh", None)
            if dbh_cm is None or dbh_cm <= 0:
                n_dbh_null += 1
                continue

            try:
                mask_bin = decode_mask(ann, H_img, W_img)
            except Exception:
                n_decode_fail += 1
                continue

            mask_pad = np.zeros((Hp, Wp), dtype=bool)
            mask_pad[:H_img, :W_img] = mask_bin
            mask_pad_t = torch.as_tensor(mask_pad)

            # tol_wh=inf: selalu ambil kandidat baris TERBAIK (paling dekat 1.3m),
            # filtering ketat dilakukan belakangan lewat filters.py (Tier 1).
            row_img, trunk_cols, d_trunk, world_h = _find_row_1p3m(
                mask_pad_t, depth_pad_t, cy=_ORIG_CY, fy=_ORIG_FY,
                h_cam=_H_CAM, tol_wh=float("inf"))

            if row_img is None:
                n_row_fail += 1
                continue

            trunk_px = len(trunk_cols)
            dbh_geom_mm = trunk_px * d_trunk * 1000.0 / _ORIG_FY

            # load_coco_json() remaps category_id ke contiguous 0-based (raw_id - 1)
            # sesuai urutan 'categories' di JSON (1..13, sama urutan dgn CLASS_NAMES).
            # Konversi balik ke 1-based supaya konsisten dgn filters.py/CAT_ID_TO_NAME
            # dan konvensi id di seluruh proyek (Mango=4, RubberFig=12, dst).
            category_id_1based = ann["category_id"] + 1

            records.append({
                "image_id": image_id,
                "ann_id": f"{image_id}_{ann_idx}",
                "split": split,
                "category_id": category_id_1based,
                "gt_dbh_mm": dbh_cm * 10.0,
                "row_img": int(row_img),
                "trunk_cols": trunk_cols.tolist(),
                "trunk_px": trunk_px,
                "d_trunk": float(d_trunk),
                "world_h": float(world_h),
                "dbh_geom_mm": float(dbh_geom_mm),
            })

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            logger.info(f"  [{i+1}/{len(dataset_dicts)}]  {elapsed:.0f}s elapsed  "
                        f"records so far: {len(records)}")

    df = pd.DataFrame.from_records(records)
    index_path = cache_dir / f"index_{split}.pkl"
    df.to_pickle(index_path)
    logger.info(f"  Saved {index_path}  ({len(df)} instances)")
    logger.info(f"  Skipped: no_depth_ann={n_no_depth}  dbh_null_or_0={n_dbh_null}  "
                f"decode_fail={n_decode_fail}  row_fail={n_row_fail}")

    # Ringkasan per-spesies (sanity check cepat vs create_clean_dbh_json_v2.py)
    if len(df):
        from filters import CAT_ID_TO_NAME
        logger.info("  Per-spesies (semua N di cache, BELUM difilter axis):")
        for cid, g in df.groupby("category_id"):
            name = CAT_ID_TO_NAME.get(int(cid), str(cid))
            logger.info(f"    {name:<14} N={len(g):>6}  "
                        f"trunk_px med={g['trunk_px'].median():.1f}  "
                        f"|wh-1.3| med={ (g['world_h']-1.3).abs().median():.3f}m")

    return df


def main(args):
    logging.basicConfig(level=logging.INFO,
                         format="[%(asctime)s %(name)s] %(message)s", datefmt="%m/%d %H:%M:%S")
    register_all_combined(os.environ.get("COMBINED_ROOT",
        str(PROJECT_ROOT / "data" / "combined")))

    cfg = setup_cfg(args.config_file, args.model_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    backbone = model.backbone
    pixel_mean = model.pixel_mean.to(device)
    pixel_std  = model.pixel_std.to(device)
    size_div   = model.size_divisibility
    logger.info(f"Backbone loaded & frozen from {cfg.MODEL.WEIGHTS}")
    logger.info(f"size_divisibility={size_div}  pixel_mean/std shape={tuple(pixel_mean.shape)}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        dataset_name = f"{args.dataset_prefix}_{split}"
        process_split(split, dataset_name, backbone, pixel_mean, pixel_std,
                      size_div, device, cache_dir, limit=args.limit)

    manifest = {
        "config_file": args.config_file,
        "model_weights": args.model_weights,
        "dataset_prefix": args.dataset_prefix,
        "size_divisibility": size_div,
        "limit": args.limit,
    }
    import json
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"DONE. Cache dir: {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataset-prefix", default="combined_inst_rle_f1000_repaired_v4_stratified")
    parser.add_argument("--limit", type=int, default=0, help="Batasi N image per split (0=semua) — utk smoke test")
    args = parser.parse_args()
    main(args)
