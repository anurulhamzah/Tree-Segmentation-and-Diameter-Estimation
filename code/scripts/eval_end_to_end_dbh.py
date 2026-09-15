#!/usr/bin/env python3
"""eval_end_to_end_dbh.py — Evaluasi DBH benar-benar end-to-end: mask dan spesies yang
dipakai head diameter adalah MILIK PREDIKSI MODEL SENDIRI (dicocokkan ke GT lewat IoU),
bukan GT langsung seperti eval_joint_trunkroi_dbh.py / dump_dbh_per_instance.py.

Latar belakang: seluruh angka diameter di naskah (RMSE headline 7.80cm dkk) dihitung dengan
head.forward_single(strict=True) diberi ann["segmentation"]/ann["category_id"] GT langsung --
bukan hasil deteksi model. Itu pilihan desain yang sah (mengisolasi kualitas regressor dari
noise deteksi, konsisten dengan narasi "deteksi adalah bottleneck" di §3.4), tapi belum pernah
diukur: seberapa besar bedanya kalau head dikasih mask/spesies HASIL PREDIKSI MODEL, seperti
yang sungguh terjadi saat deployment (persis pipeline gen_fig4_panels.py / predict_batch).

Protokol pencocokan: greedy IoU>=0.5, prediksi diurutkan skor menurun, satu GT dipasangkan
maksimal sekali (standar deteksi). Filter kelayakan GT dan filter geometri identik
eval_joint_trunkroi_dbh.py (species_subset, RUBBERFIG cap, MIN_TRUNK_PX, MIN_DEPTH/MAX_DEPTH,
MAX_WH_DEV) supaya populasi sebisa mungkin sepadan, hanya beda sumber mask/spesies.

Usage:
    python scripts/eval_end_to_end_dbh.py \
        --output-dir FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k_cap30_detach \
        --species-subset 1,3,4,5,6,7,8,9,10,12,13
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pycocotools import mask as maskUtils
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import ImageList
from detectron2.utils.logger import setup_logger

from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import build_transform_gen
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import _read_pfm, _normalize_depth
from register_combined import register_all_combined
from eval_joint_trunkroi_dbh import (
    build_model_with_head, decode_mask, r2_mae, DEPTH_DIR, CAT_NAMES,
    MIN_TRUNK_PX, MIN_DEPTH, MAX_DEPTH, MAX_WH_DEV, RUBBERFIG_CAT_ID, RUBBERFIG_CAP_CM,
    MIN_SPECIES_N,
)

SCORE_THR = 0.30
IOU_MATCH = 0.5


def iou_bool(a, b):
    inter = (a & b).sum()
    if inter == 0:
        return 0.0
    union = (a | b).sum()
    return float(inter) / float(union)


@torch.no_grad()
def evaluate(model, tfm, dataset_dicts, species_subset, device):
    head = model.dbh_head_trunkroi
    head.eval()
    size_div = model.size_divisibility

    per_species_pred, per_species_gt = {}, {}
    per_instance = []          # dump lengkap: (gt_id, cat_id, gt_mm, pred_mm, d_trunk, world_h) -- utk
                                 # analisis lanjutan (populasi matched vs constant, dsb.)
    n_gt_eligible = 0          # GT dbh>0, species_subset, lolos cap -- populasi yang SEHARUSNYA diukur
    n_gt_matched = 0           # dari situ, yang berhasil dicocokkan ke prediksi (IoU>=0.5)
    n_skip_filter = 0          # matched tapi gagal filter geometri head (MIN_TRUNK_PX dst.)
    n_no_depth = 0

    for i, rec in enumerate(dataset_dicts):
        if (i + 1) % 100 == 0:
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
        depth_norm01 = _normalize_depth(depth_raw)

        # (A) deteksi standar: pred_masks/classes/scores >= SCORE_THR
        img_t, tfms = T.apply_transform_gens(tfm, image.copy())
        dep_t = tfms.apply_image(depth_norm01[:, :, None])[:, :, 0]
        out = model([{"image": torch.as_tensor(np.ascontiguousarray(img_t.transpose(2, 0, 1))),
                      "depth": torch.as_tensor(np.ascontiguousarray(dep_t)),
                      "height": H_img, "width": W_img}])[0]["instances"].to("cpu")
        keep = out.scores >= SCORE_THR
        pred_masks = out.pred_masks[keep].numpy().astype(bool)
        pred_classes = out.pred_classes[keep].numpy()   # 0-based
        pred_scores = out.scores[keep].numpy()
        order = np.argsort(-pred_scores)                # skor menurun, utk matching greedy

        # (B) res2 utk head, backbone native-padded (identik eval_joint_trunkroi_dbh.py)
        depth_norm255 = depth_norm01 * 255.0
        image_t = torch.as_tensor(image.transpose(2, 0, 1).copy(), dtype=torch.float32)
        depth_t = torch.as_tensor(depth_norm255, dtype=torch.float32).unsqueeze(0)
        img4 = torch.cat([image_t, depth_t], dim=0)
        img_list = ImageList.from_tensors([img4], size_div)
        imgs_norm = (img_list.tensor.to(device) - model.pixel_mean) / model.pixel_std
        res2 = model.backbone(imgs_norm)["res2"][0]
        depth_b = torch.as_tensor(depth_norm01, dtype=torch.float32, device=device)

        # (C) GT anotasi -- utk kelayakan populasi & matching target, BUKAN utk head
        gts = []
        for local_idx, ann in enumerate(rec.get("annotations", [])):
            dbh_cm = float(ann.get("dbh") or 0.0)
            cat_id = int(ann["category_id"]) + 1
            eligible = (dbh_cm > 0 and cat_id in species_subset and
                        not (cat_id == RUBBERFIG_CAT_ID and dbh_cm >= RUBBERFIG_CAP_CM))
            gt_id = ann.get("id", f"{rec['image_id']}_{local_idx}")
            gts.append(dict(mask=decode_mask(ann, H_img, W_img), cat_id=cat_id,
                             dbh_mm=dbh_cm * 10.0, eligible=eligible, gt_id=gt_id))
        n_gt_eligible += sum(1 for g in gts if g["eligible"])

        # (D) matching greedy IoU>=0.5, satu GT dipasangkan maksimal sekali
        gt_used = [False] * len(gts)
        for pi in order:
            best_j, best_iou = -1, 0.0
            for j, g in enumerate(gts):
                if gt_used[j]:
                    continue
                v = iou_bool(pred_masks[pi], g["mask"])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j < 0 or best_iou < IOU_MATCH:
                continue
            gt_used[best_j] = True
            g = gts[best_j]
            if not g["eligible"]:
                continue
            n_gt_matched += 1

            result = head.forward_single(
                res2, depth_b, torch.as_tensor(pred_masks[pi], device=device),
                species_idx=int(pred_classes[pi]), strict=False)
            if result is None:
                n_skip_filter += 1
                continue
            pred, dbh_geom_mm, d_trunk, trunk_px, world_h = result
            if trunk_px < MIN_TRUNK_PX or not (MIN_DEPTH <= d_trunk < MAX_DEPTH) or \
               abs(world_h - 1.3) > MAX_WH_DEV:
                n_skip_filter += 1
                continue

            pred_mm = torch.expm1(pred).clamp(min=0).item()
            cid = g["cat_id"]        # tetap dikelompokkan per spesies GT, utk tabel per-spesies
            per_species_pred.setdefault(cid, []).append(pred_mm)
            per_species_gt.setdefault(cid, []).append(g["dbh_mm"])
            per_instance.append(dict(gt_id=g["gt_id"], cat_id=cid, gt_mm=g["dbh_mm"],
                                      pred_mm=pred_mm, d_trunk=d_trunk, world_h=world_h))

    print(f"\nn_gt_eligible={n_gt_eligible}  n_gt_matched(IoU>={IOU_MATCH})={n_gt_matched}  "
          f"n_skip_filter={n_skip_filter}  n_no_depth={n_no_depth}")
    print(f"cakupan (matched/eligible) = {100*n_gt_matched/max(n_gt_eligible,1):.1f}%")
    return per_species_pred, per_species_gt, n_gt_eligible, n_gt_matched, per_instance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--species-subset", required=True)
    ap.add_argument("--dataset-test", default="combined_inst_rle_f1000_repaired_v4_stratified_test")
    args = ap.parse_args()

    species_subset = set(int(x) for x in args.species_subset.split(","))
    output_dir = PROJECT_ROOT.parent / "maskdino_output" / args.output_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

    setup_logger()
    import os
    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_combined(os.environ["COMBINED_ROOT"])

    model, cfg = build_model_with_head(output_dir, device)
    tfm = build_transform_gen(cfg, is_train=False)
    dataset_dicts = DatasetCatalog.get(args.dataset_test)
    print(f"Evaluating END-TO-END on {len(dataset_dicts)} images, species_subset={sorted(species_subset)}")

    per_species_pred, per_species_gt, n_elig, n_match, per_instance = evaluate(
        model, tfm, dataset_dicts, species_subset, device)

    all_pred = [p for v in per_species_pred.values() for p in v]
    all_gt = [g for v in per_species_gt.values() for g in v]
    overall = r2_mae(all_pred, all_gt)
    print(f"\nOVERALL (end-to-end): N={overall['N']}  RMSE={overall['RMSE']:.2f}mm  "
          f"MAE={overall['MAE']:.2f}mm  bias={overall['bias']:.2f}mm  R2={overall['R2']*100:.2f}%")

    scored = []
    print("\nPER-SPECIES:")
    for cid in sorted(per_species_pred.keys()):
        m = r2_mae(per_species_pred[cid], per_species_gt[cid])
        name = CAT_NAMES.get(cid, str(cid))
        flag = "" if m["N"] >= MIN_SPECIES_N else "  (N<30, excluded from mean)"
        print(f"  {name:<14} N={m['N']:4d}  RMSE={m['RMSE']:6.2f}mm  R2={m['R2']*100:6.2f}%{flag}")
        if m["N"] >= MIN_SPECIES_N:
            scored.append(m["R2"])
    mean_r2 = float(np.mean(scored)) if scored else None
    print(f"\nmean_per_species_R2 (N>={MIN_SPECIES_N}, {len(scored)} spesies) = "
          f"{mean_r2*100:.2f}%" if mean_r2 is not None else "mean_per_species_R2 = n/a")

    REPORT_DIR = PROJECT_ROOT / "reports" / "dbh_eval"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"dbh_metrics_end2end_{args.output_dir}.json"
    json.dump({
        "output_dir": args.output_dir, "species_subset": sorted(species_subset),
        "n_gt_eligible": n_elig, "n_gt_matched": n_match,
        "overall": overall, "mean_per_species_R2": mean_r2, "n_species_scored": len(scored),
        "per_species": {CAT_NAMES.get(k, str(k)): r2_mae(per_species_pred[k], per_species_gt[k])
                        for k in per_species_pred},
    }, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")

    inst_path = REPORT_DIR / f"dbh_per_instance_end2end_{args.output_dir}.json"
    json.dump(per_instance, open(inst_path, "w"))
    print(f"Saved: {inst_path}  ({len(per_instance)} instances)")


if __name__ == "__main__":
    main()
