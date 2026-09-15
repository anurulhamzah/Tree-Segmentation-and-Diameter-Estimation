#!/usr/bin/env python3
"""dump_dbh_per_instance.py — simpan prediksi DBH PER INSTANCE beserta geometrinya.

KENAPA ADA. `evaluate()` di eval_joint_trunkroi_dbh.py hanya mengembalikan pred/gt yang
sudah dikelompokkan per spesies, lalu membuang `trunk_px` dan `d_trunk` yang sebenarnya
sudah dihitung `head.forward_single`. Akibatnya tidak ada cara memotong error menurut
ukuran diameter, ketebalan batang dalam piksel, atau jarak.

BATAS KEDALAMAN. Evaluator baku memakai MAX_DEPTH = 20,0 m, sehingga pohon 20-30 m TIDAK
PERNAH dinilai — termasuk saat membandingkan model yang dilatih dengan cap 30. Skrip ini
membuat batas itu jadi argumen (`--max-depth`), supaya potongan per jarak bisa dibuat dan
manfaat cap lebar bisa diukur pada populasi yang benar-benar mencakupnya.

KESETIAAN PADA EVALUATOR BAKU. Perulangan di bawah menyalin `evaluate()` baris demi baris,
hanya menambah pengumpulan field dan memparameterkan threshold. Verifikasinya: dump ini
disaring ke d_trunk < 20 harus menghasilkan RMSE yang SAMA PERSIS dengan hasil eval resmi.
Jalankan `--verifikasi` untuk memeriksanya.

Contoh:
    python scripts/dump_dbh_per_instance.py \
        --output-dir FocalNet_L_trunk_roi_dbh_joint_ft20k_7sp_cap30 \
        --ckpt model_0017499.pth --species-subset 3,5,8,9,10,12,13 \
        --max-depth 30 --splits val,test
"""
import argparse
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

DS_PREFIX = "combined_inst_rle_f1000_repaired_v4_stratified"


def dump_split(model, dataset_dicts, subset, device, threshold, img_stride, img_offset):
    """Salinan setia evaluate(), tetapi menyimpan tiap instance beserta geometrinya."""
    import numpy as np
    from detectron2.data import detection_utils as utils
    from detectron2.structures import ImageList
    from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import (
        _read_pfm, _normalize_depth)
    from eval_joint_trunkroi_dbh import (DEPTH_DIR, decode_mask, RUBBERFIG_CAT_ID,
                                         RUBBERFIG_CAP_CM)

    head = model.dbh_head_trunkroi
    head.eval()
    size_div = model.size_divisibility
    rows = []
    n_total = n_skip = 0

    recs = dataset_dicts[img_offset::img_stride] if img_stride > 1 else dataset_dicts
    for i, rec in enumerate(recs):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(recs)} images...", flush=True)
        H_img, W_img = rec["height"], rec["width"]
        stem = Path(rec["file_name"]).stem
        pfm = DEPTH_DIR / f"{stem}.pfm"
        if not pfm.exists():
            continue

        image = utils.read_image(rec["file_name"], format="RGB")
        depth_raw = _read_pfm(str(pfm))
        if depth_raw.shape != (H_img, W_img):
            import cv2
            depth_raw = cv2.resize(depth_raw, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        depth01 = _normalize_depth(depth_raw)

        image_t = torch.as_tensor(image.transpose(2, 0, 1).copy(), dtype=torch.float32)
        depth_t = torch.as_tensor(depth01 * 255.0, dtype=torch.float32).unsqueeze(0)
        img_list = ImageList.from_tensors([torch.cat([image_t, depth_t], dim=0)], size_div)
        imgs_norm = (img_list.tensor.to(device) - model.pixel_mean) / model.pixel_std
        res2 = model.backbone(imgs_norm)["res2"][0]
        depth_b = torch.as_tensor(depth01, dtype=torch.float32, device=device)

        for ann in rec.get("annotations", []):
            dbh_cm = float(ann.get("dbh") or 0.0)
            if dbh_cm <= 0:
                continue
            n_total += 1
            cat_id = int(ann["category_id"]) + 1
            if cat_id not in subset:
                n_skip += 1; continue
            if cat_id == RUBBERFIG_CAT_ID and dbh_cm * 10.0 >= RUBBERFIG_CAP_CM * 10.0:
                n_skip += 1; continue

            mask_t = torch.as_tensor(decode_mask(ann, H_img, W_img), device=device)
            result = head.forward_single(res2, depth_b, mask_t,
                                         species_idx=cat_id - 1, strict=True)
            if result is None:
                n_skip += 1; continue
            pred, dbh_geom_mm, d_trunk, trunk_px, world_h = result

            if trunk_px < threshold["min_trunk_px"]:
                n_skip += 1; continue
            if not (threshold["min_depth"] <= d_trunk < threshold["max_depth"]):
                n_skip += 1; continue
            if abs(world_h - 1.3) > threshold["max_wh_dev"]:
                n_skip += 1; continue

            rows.append(dict(
                sp=cat_id,
                pred=round(float(torch.expm1(pred).clamp(min=0).item()), 3),
                gt=round(dbh_cm * 10.0, 3),
                geom=round(float(dbh_geom_mm), 3),
                px=round(float(trunk_px), 3),
                d=round(float(d_trunk), 4),
                wh=round(float(world_h), 4),
            ))
    print(f"  n_total={n_total} n_skip={n_skip} tersimpan={len(rows)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ckpt", default="model_final.pth")
    ap.add_argument("--species-subset", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--max-depth", type=float, default=20.0,
                    help="Batas jarak evaluasi. Baku evaluator resmi 20,0. Naikkan untuk "
                         "menilai pohon jauh yang biasanya dibuang.")
    ap.add_argument("--min-depth", type=float, default=1.0)
    ap.add_argument("--min-trunk-px", type=float, default=5.0)
    ap.add_argument("--max-wh-dev", type=float, default=0.5)
    ap.add_argument("--img-stride", type=int, default=1)
    ap.add_argument("--img-offset", type=int, default=0)
    ap.add_argument("--tag", default="", help="tag tambahan pada nama berkas output")
    args = ap.parse_args()

    from detectron2.data import DatasetCatalog
    from detectron2.utils.logger import setup_logger
    from register_combined import register_all_combined
    from eval_joint_trunkroi_dbh import build_model_with_head

    setup_logger()
    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_combined(os.environ["COMBINED_ROOT"])

    subset = set(int(x) for x in args.species_subset.split(","))
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    threshold = dict(min_depth=args.min_depth, max_depth=args.max_depth,
                  min_trunk_px=args.min_trunk_px, max_wh_dev=args.max_wh_dev)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = OUTPUT_ROOT / args.output_dir
    model, _ = build_model_with_head(out_dir, device, init_ckpt=args.ckpt)

    hasil = {}
    for s in splits:
        print(f"=== {args.output_dir} / {args.ckpt} / {s} (max_depth={args.max_depth}) ===")
        hasil[s] = dump_split(model, DatasetCatalog.get(f"{DS_PREFIX}_{s}"),
                              subset, device, threshold, args.img_stride, args.img_offset)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    shard = f"__img{args.img_offset}of{args.img_stride}" if args.img_stride > 1 else ""
    p = REPORT_DIR / f"dbh_per_instance_{args.output_dir}{args.tag}{shard}.json"
    with open(p, "w") as f:
        json.dump(dict(output_dir=args.output_dir, ckpt=args.ckpt,
                       species_subset=sorted(subset), threshold=threshold,
                       n={s: len(v) for s, v in hasil.items()}, data=hasil), f)
    print(f"Tersimpan: {p}")


if __name__ == "__main__":
    main()
