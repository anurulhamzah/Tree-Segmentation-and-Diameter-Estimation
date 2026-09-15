#!/usr/bin/env python3
"""
eval_joint_trunkroi_dbh.py — Evaluasi DBH R2/MAE untuk model JOINT (HybridTrunkROIDBHHead
+ backbone unfrozen), dikondisikan pada GT mask.

PENTING (v2, diperbaiki 27 Jul 2026): v1 memakai COCOInstanceDBHDatasetMapper (LSJ,
ResizeScale ke kanvas PERSEGI 640x640 dari sumber 480x270/16:9) -- ini MENDISTORSI aspect
ratio, merusak asumsi "square pixels" (fx=fy) di formula geometris row-1.3m
(_find_row_1p3m/trunk_roi_dbh_head_hybrid.py), menghasilkan R2 sangat negatif (-215%
avg) yang TERNYATA bug preprocessing, bukan model gagal. v2 memakai pipeline NATIVE-RESOLUTION
(baca gambar asli 480x270 + PFM depth langsung, cuma di-pad ke kelipatan size_divisibility,
TANPA resize/crop) -- persis metodologi precompute_dbh_cache.py (Sandbox) yang terbukti benar
(R2=76% utk model frozen). Backbone FocalNet-L dilatih dgn LSJ 640x640, TAPI res2 (stage awal,
lokal/translation-equivariant) terbukti empiris tetap bekerja baik pd native-res di Sandbox.

Usage:
    python scripts/eval_joint_trunkroi_dbh.py \
        --output-dir FocalNet_L_trunk_roi_dbh_hybrid_joint_7species_20k \
        --species-subset 3,5,8,9,10,12,13
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
OUTPUT_ROOT   = PROJECT_ROOT.parent / "maskdino_output"
REPORT_DIR    = PROJECT_ROOT / "reports" / "dbh_eval"
DEPTH_DIR     = PROJECT_ROOT / "data" / "combined" / "depth_pfm"

sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "dbh_sandbox"))

from pycocotools import mask as maskUtils
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.modeling import build_model
from detectron2.structures import ImageList
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config

from maskdino import add_maskdino_config
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import _read_pfm, _normalize_depth
from register_combined import register_all_combined
from trunk_roi_dbh_head_hybrid import HybridTrunkROIDBHHead

IN_CHANNELS = 192
HIDDEN = 256
STRIP_ROWS = 5
DROPOUT = 0.3
NUM_SPECIES = 13

CAT_NAMES = {
    1: "Apple", 2: "Lemon", 3: "Loquat", 4: "Mango", 5: "Orange", 6: "Persimmon",
    7: "Pomegranate", 8: "AliiFig", 9: "BangaloPalm", 10: "Fern", 11: "LeechVine",
    12: "RubberFig", 13: "Umbrella",
}

MIN_TRUNK_PX = 5
MIN_DEPTH = 1.0
MAX_DEPTH = 20.0
MAX_WH_DEV = 0.5
RUBBERFIG_CAT_ID = 12
RUBBERFIG_CAP_CM = 150.0
MIN_SPECIES_N = 30


def build_model_with_head(output_dir: Path, device: str, init_ckpt: str = "model_final.pth"):
    """init_ckpt hanya menentukan weights awal saat model dibangun. Pemanggil yang me-reload
    weight per checkpoint (mis. eval per-checkpoint) boleh menunjuk checkpoint mana pun yang
    sudah ada, berguna selagi training masih berjalan dan model_final.pth belum ditulis."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(output_dir / "config.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()

    model = build_model(cfg)
    # DBH_HEAD_IN_CHANNELS baru ditulis ke config.yaml sejak --dbh-head-in-channels
    # ditambahkan (25 Agt 2026); run lama (mis. FocalNet-L detach) tidak punya key ini,
    # fallback ke IN_CHANNELS=192 (perilaku lama, benar utk FocalNet-L res2). Tanpa ini,
    # eval backbone lain (FocalNet-B res2=128) crash shape mismatch di LayerNorm pertama.
    in_channels = getattr(cfg, "DBH_HEAD_IN_CHANNELS", IN_CHANNELS)
    dbh_head = HybridTrunkROIDBHHead(
        in_channels=in_channels, hidden=HIDDEN, strip_rows=STRIP_ROWS,
        num_species=NUM_SPECIES, dropout=DROPOUT,
    ).to(device)
    model.dbh_head_trunkroi = dbh_head

    ckpt = output_dir / init_ckpt
    DetectionCheckpointer(model).load(str(ckpt))
    model.eval()
    print(f"Loaded {ckpt} on {device}")
    n_head = sum(p.numel() for p in dbh_head.parameters())
    print(f"dbh_head_trunkroi: {n_head:,} params")
    return model, cfg


def decode_mask(ann: dict, H: int, W: int) -> np.ndarray:
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rle = maskUtils.frPyObjects(seg, H, W)
        rle = maskUtils.merge(rle)
    else:
        rle = seg
    return maskUtils.decode(rle).astype(bool)


@torch.no_grad()
def evaluate(model, dataset_dicts, species_subset, device):
    head = model.dbh_head_trunkroi
    head.eval()
    size_div = model.size_divisibility

    per_species_pred = {}
    per_species_gt = {}
    n_total = n_skip_species = n_skip_filter = n_no_depth = 0

    for i, rec in enumerate(dataset_dicts):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(dataset_dicts)} images...", flush=True)

        file_name = rec["file_name"]
        H_img, W_img = rec["height"], rec["width"]
        stem = Path(file_name).stem
        pfm_path = DEPTH_DIR / f"{stem}.pfm"
        if not pfm_path.exists():
            n_no_depth += len(rec.get("annotations", []))
            continue

        image = utils.read_image(file_name, format="RGB")
        depth_raw = _read_pfm(str(pfm_path))
        if depth_raw.shape != (H_img, W_img):
            import cv2
            depth_raw = cv2.resize(depth_raw, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        depth_norm01 = _normalize_depth(depth_raw)              # [0,1], utk head geometri
        depth_norm255 = depth_norm01 * 255.0                     # utk channel ke-4 backbone

        image_t = torch.as_tensor(image.transpose(2, 0, 1).copy(), dtype=torch.float32)
        depth_t = torch.as_tensor(depth_norm255, dtype=torch.float32).unsqueeze(0)
        img4 = torch.cat([image_t, depth_t], dim=0)

        img_list = ImageList.from_tensors([img4], size_div)
        imgs_norm = (img_list.tensor.to(device) - model.pixel_mean) / model.pixel_std
        res2 = model.backbone(imgs_norm)["res2"][0]              # (C, Hf, Wf)

        depth_b = torch.as_tensor(depth_norm01, dtype=torch.float32, device=device)  # (H_img, W_img)

        for ann in rec.get("annotations", []):
            dbh_cm = float(ann.get("dbh") or 0.0)
            if dbh_cm <= 0:
                continue
            n_total += 1
            # load_coco_json() remaps category_id ke contiguous 0-based -- kembalikan ke
            # 1-based asli (sama persis fix di precompute_dbh_cache.py baris 181)
            cat_id = int(ann["category_id"]) + 1
            if cat_id not in species_subset:
                n_skip_species += 1
                continue
            if cat_id == RUBBERFIG_CAT_ID and dbh_cm * 10.0 >= RUBBERFIG_CAP_CM * 10.0:
                n_skip_species += 1
                continue

            mask_np = decode_mask(ann, H_img, W_img)
            mask_t = torch.as_tensor(mask_np, device=device)

            result = head.forward_single(
                res2, depth_b, mask_t, species_idx=cat_id - 1, strict=True)
            if result is None:
                n_skip_filter += 1
                continue
            pred, dbh_geom_mm, d_trunk, trunk_px, world_h = result

            if trunk_px < MIN_TRUNK_PX:
                n_skip_filter += 1; continue
            if not (MIN_DEPTH <= d_trunk < MAX_DEPTH):
                n_skip_filter += 1; continue
            if abs(world_h - 1.3) > MAX_WH_DEV:
                n_skip_filter += 1; continue

            pred_mm = torch.expm1(pred).clamp(min=0).item()
            gt_mm = dbh_cm * 10.0
            per_species_pred.setdefault(cat_id, []).append(pred_mm)
            per_species_gt.setdefault(cat_id, []).append(gt_mm)

    print(f"n_total(dbh>0)={n_total}  n_skip_species={n_skip_species}  "
          f"n_skip_filter={n_skip_filter}  n_no_depth={n_no_depth}")
    return per_species_pred, per_species_gt


def r2_mae(pred, gt):
    """RMSE adalah metrik UTAMA pelaporan (18 dari 20 makalah DBH memakainya sebagai metrik
    utama), MAE dan bias pendamping, R2 pendukung. Semuanya dalam milimeter kecuali R2.

    rRMSE (diverifikasi 31 Jul terhadap BANYAK rujukan independen, bukan cuma satu, setelah
    ditegur user "cari sumber lain, yakinkan rumus benar"):
      rRMSE = RMSE / mean(GT) x 100  -- INI yang dipakai di bawah. Didukung oleh:
        (a) Shao dkk. 2025, "A Comparative Analysis of Low-Cost Devices for High-Precision
            DBH Estimation" (Remote Sensing 17:3888), Eq.3 -- rumus sama persis, domain DBH.
        (b) Konvensi UMUM di statistik/remote sensing/biomassa hutan (dikonfirmasi via
            pencarian luas): paket R `metrica::RRMSE`, `ehaGoF::gofRRMSE`, `Fgmutils::rrmse`
            semua mengimplementasikan formula ini; literatur biomassa hutan & remote sensing
            umum juga memakainya sbg definisi standar/mayoritas.
      Satu rujukan DBH ditemukan memakai formula BERBEDA: Fan dkk. 2018, "Estimating Tree
      Position, DBH, and Tree Height ... RGB-D SLAM" (Remote Sensing 10:1845), Eq.22:
      relRMSE = sqrt(mean(((pred/gt)-1)^2)) x 100 (RMS error relatif PER INSTANCE, bukan
      RMSE dibagi rata-rata). SECARA MATEMATIS BERBEDA dari rumus di atas (tidak ekuivalen
      kecuali gt konstan); selisihnya materiil pada data proyek ini: 26,05% (formula dipakai)
      vs 34,14% (formula Fan dkk.) utk model frozen 7-spesies, N=1640. Karena formula di atas
      terbukti MAYORITAS/standar (bukan cuma 1 rujukan yg kebetulan cocok), TIDAK diganti ke
      varian Fan dkk. Konsekuensinya: rRMSE laporan ini TIDAK otomatis apple-to-apple dgn
      angka "%" self-reported milik Fan dkk. 2018 secara spesifik (rujukan lain yg formulanya
      belum dicek eksplisit juga berlaku hati-hati serupa). MRAE = mean(|err|/gt)*100,
      ekuivalen MAPE standar (konvensi tunggal, tidak ambigu di literatur manapun yg
      ditelusuri)."""
    pred = np.array(pred); gt = np.array(gt)
    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-10))
    bias = float(np.mean(err))
    w50 = float(np.mean(np.abs(err) <= 50)) * 100
    w100 = float(np.mean(np.abs(err) <= 100)) * 100
    gt_mean = float(np.mean(gt))
    rrmse = float(rmse / gt_mean * 100) if gt_mean > 0 else float("nan")
    mrae = float(np.mean(np.abs(err) / np.maximum(gt, 1e-6)) * 100)
    return {"N": len(pred), "RMSE": rmse, "MAE": mae, "bias": bias,
            "rRMSE": rrmse, "MRAE": mrae, "R2": r2, "w50mm": w50, "w100mm": w100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--species-subset", required=True)
    ap.add_argument("--dataset-test", default="combined_inst_rle_f1000_repaired_v4_stratified_test")
    args = ap.parse_args()

    species_subset = set(int(x) for x in args.species_subset.split(","))
    output_dir = OUTPUT_ROOT / args.output_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

    setup_logger()
    import os
    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_combined(os.environ["COMBINED_ROOT"])

    model, cfg = build_model_with_head(output_dir, device)
    dataset_dicts = DatasetCatalog.get(args.dataset_test)
    print(f"Evaluating on {len(dataset_dicts)} images (native-res pipeline), "
          f"species_subset={sorted(species_subset)}")

    per_species_pred, per_species_gt = evaluate(model, dataset_dicts, species_subset, device)

    all_pred = [p for v in per_species_pred.values() for p in v]
    all_gt = [g for v in per_species_gt.values() for g in v]
    overall = r2_mae(all_pred, all_gt)
    print(f"\nOVERALL: N={overall['N']}  R2={overall['R2']*100:.2f}%  MAE={overall['MAE']:.2f}mm  "
          f"bias={overall['bias']:.2f}  w50mm={overall['w50mm']:.2f}%  w100mm={overall['w100mm']:.2f}%")

    per_species = {}
    scored = []
    print("\nPER-SPECIES:")
    for cid in sorted(per_species_pred.keys()):
        m = r2_mae(per_species_pred[cid], per_species_gt[cid])
        per_species[cid] = m
        name = CAT_NAMES.get(cid, str(cid))
        flag = "" if m["N"] >= MIN_SPECIES_N else "  (N<30, excluded from mean)"
        print(f"  {name:<14} N={m['N']:4d}  R2={m['R2']*100:6.2f}%  MAE={m['MAE']:6.2f}mm{flag}")
        if m["N"] >= MIN_SPECIES_N:
            scored.append(m["R2"])

    mean_per_species_r2 = float(np.mean(scored)) if scored else None
    if mean_per_species_r2 is not None:
        print(f"\nmean_per_species_R2 (N>={MIN_SPECIES_N} only, {len(scored)} spesies) = "
              f"{mean_per_species_r2*100:.2f}%")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "output_dir": args.output_dir,
        "species_subset": sorted(species_subset),
        "overall": overall,
        "mean_per_species_R2": mean_per_species_r2,
        "n_species_scored": len(scored),
        "per_species": {CAT_NAMES.get(k, str(k)): v for k, v in per_species.items()},
    }
    out_path = REPORT_DIR / f"dbh_metrics_joint_{args.output_dir}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
