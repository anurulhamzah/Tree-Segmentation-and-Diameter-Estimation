#!/usr/bin/env python
"""
End-to-end DBH predictor: gambar -> per-pohon {jenis, DBH_mm}.

Pipeline (konsisten PERSIS dgn cara head Sandbox dilatih):
  1. MaskDINO FocalNet-L  -> instance mask + jenis (species) + skor  (inference standar, resize 640)
  2. Backbone forward native-padded (480x270 -> pad kelipatan 32) -> feature res2  (SAMA spt precompute cache)
  3. Untuk tiap instance: _find_row_1p3m (pinhole, H_cam=2m, fy=240) -> ambil strip res2 ->
     head MLP -> ln(1+DBH) -> DBH_mm = expm1(output)

Head-nya = Sandbox winner (phaseB_final_baseline_anchor_only/head_best.pth).
Backbone = FocalNet-L combined RGBD scratch 75k (checkpoint yg dipakai precompute cache).

Jalankan:
  python scripts/predict_dbh.py --n 3          # 3 gambar test acak
  python scripts/predict_dbh.py --stem Tree123_...   # gambar spesifik
"""
import os, sys, json, argparse, math, random
import numpy as np
import torch
import torch.nn.functional as F

USER = "/scratch2/pr65/anur0018"
MK = f"{USER}/MaskDINO/MaskDINO"
PROJ = f"{USER}/tree_classification"
sys.path.insert(0, MK)
sys.path.insert(0, f"{PROJ}/scripts")

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import maskdino  # noqa: F401
from maskdino import add_maskdino_config
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import build_transform_gen
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import _read_pfm, _normalize_depth
from trunk_roi_dbh_head_hybrid import (
    HybridTrunkROIDBHHead, _find_row_1p3m, _ORIG_CY, _ORIG_FY, _H_CAM,
)

CONFIG = f"{PROJ}/configs/maskdino_FocalNet_L_combined_rle_f1000_rgbd_scratch_dbh_json_v2.yaml"
BACKBONE_CKPT = f"{USER}/maskdino_output/FocalNet_L_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k/model_final.pth"
HEAD_CKPT = f"{USER}/maskdino_output/sweep_dbh_hyperparams/phaseB_final_baseline_anchor_only/head_best.pth"
DEPTH_DIR = f"{PROJ}/data/combined/depth_pfm"
RGB_DIR = f"{PROJ}/data/combined/rgb_resized"
GT_JSON = f"{PROJ}/data/combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json"

CLASS_NAMES = ["Apple", "Lemon", "Loquat", "Mango", "Orange", "Persimmon", "Pomegranate",
               "AliiFig", "BangaloPalm", "Fern", "LeechVine", "RubberFig", "Umbrella"]
SCORE_THR = 0.35
# LeechVine (liana) tidak punya konsep DBH -> lewati
SKIP_SPECIES = {"LeechVine"}


def build():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = BACKBONE_CKPT
    cfg.freeze()
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    dev = model.pixel_mean.device

    head = HybridTrunkROIDBHHead(in_channels=192, hidden=256, strip_rows=5,
                                 num_species=13, dropout=0.3).to(dev)
    sd = torch.load(HEAD_CKPT, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    head.load_state_dict(sd)
    head.eval()

    tfm = build_transform_gen(cfg, is_train=False)  # test resize (640) utk instance
    sizediv = model.backbone.size_divisibility or 32
    return cfg, model, head, tfm, dev, sizediv


@torch.no_grad()
def predict_image(model, head, tfm, dev, sizediv, rgb_path):
    # depth diturunkan dari path RGB (rgb_resized -> depth_pfm, .png -> .pfm)
    pfm_path = rgb_path.replace("/rgb_resized/", "/depth_pfm/").replace(".png", ".pfm")
    if not os.path.exists(pfm_path):
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        pfm_path = f"{DEPTH_DIR}/{stem}.pfm"
    if not os.path.exists(rgb_path) or not os.path.exists(pfm_path):
        return None
    image = utils.read_image(rgb_path, format="RGB")           # (H,W,3) uint8
    H_img, W_img = image.shape[:2]
    depth_raw = _read_pfm(pfm_path).astype(np.float32)         # (H,W) meters
    if depth_raw.shape != (H_img, W_img):
        import cv2
        depth_raw = cv2.resize(depth_raw, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

    # --- (A) instance prediction: MaskDINO inference standar (mapper test, resize 640) ---
    depth_norm01 = _normalize_depth(depth_raw)                 # [0,1]
    img_t, tfms = T.apply_transform_gens(tfm, image.copy())
    dep_t = tfms.apply_image(depth_norm01[:, :, None])[:, :, 0]
    inst_in = {"image": torch.as_tensor(np.ascontiguousarray(img_t.transpose(2, 0, 1))),
               "depth": torch.as_tensor(np.ascontiguousarray(dep_t)),
               "height": H_img, "width": W_img}
    out = model([inst_in])[0]["instances"].to("cpu")
    keep = out.scores >= SCORE_THR
    masks = out.pred_masks[keep].numpy().astype(bool)          # (N,H,W) di frame native
    classes = out.pred_classes[keep].numpy()
    scores = out.scores[keep].numpy()

    # --- (B) feature res2 dari backbone native-padded (identik precompute) ---
    depth255 = depth_norm01 * 255.0
    img4 = torch.cat([torch.as_tensor(image.transpose(2, 0, 1).copy(), dtype=torch.float32),
                      torch.as_tensor(depth255, dtype=torch.float32).unsqueeze(0)], dim=0)
    ilist = ImageList.from_tensors([img4], sizediv)
    Hp, Wp = ilist.tensor.shape[-2:]
    imgs_norm = (ilist.tensor.to(dev) - model.pixel_mean) / model.pixel_std
    res2 = model.backbone(imgs_norm)["res2"][0]                # (192, Hp/4, Wp/4)

    depth_pad = torch.zeros((Hp, Wp), dtype=torch.float32)
    depth_pad[:H_img, :W_img] = torch.as_tensor(depth_raw)

    # --- (C) per-instance: cari row 1.3m, ambil strip, head -> DBH ---
    results = []
    for m, c, s in zip(masks, classes, scores):
        sp = CLASS_NAMES[int(c)]
        if sp in SKIP_SPECIES:
            continue
        mask_pad = torch.zeros((Hp, Wp), dtype=torch.bool)
        mask_pad[:H_img, :W_img] = torch.as_tensor(m)
        row, trunk_cols, d_trunk, world_h = _find_row_1p3m(
            mask_pad, depth_pad, cy=_ORIG_CY, fy=_ORIG_FY, h_cam=_H_CAM, tol_wh=float("inf"))
        if row is None:
            continue
        strip = head._extract_strip_feat(res2, int(row), torch.as_tensor(trunk_cols, device=dev))
        if strip is None:
            continue
        trunk_px = len(trunk_cols)
        dbh_geom_mm = trunk_px * d_trunk * 1000.0 / _ORIG_FY
        geom = torch.tensor([math.log1p(dbh_geom_mm) / 10.0, d_trunk / 10.0], device=dev)
        onehot = F.one_hot(torch.tensor(int(c), device=dev), 13).float()
        x = torch.cat([strip, geom, onehot], dim=0)
        pred = head.mlp(x).squeeze(-1)
        dbh_mm = float(torch.expm1(pred))
        results.append(dict(species=sp, dbh_mm=dbh_mm, score=float(s),
                            depth_m=round(float(d_trunk), 1), trunk_px=trunk_px,
                            dbh_geom_mm=round(dbh_geom_mm, 1), world_h=round(float(world_h), 2)))
    return results


def gt_dbh_for(stem, gt):
    """Distribusi GT DBH per-species utk gambar ini (referensi sanity, bukan matching per-instance)."""
    img = next((im for im in gt["images"] if os.path.splitext(os.path.basename(im["file_name"]))[0] == stem), None)
    if img is None:
        return None
    anns = [a for a in gt["annotations"] if a["image_id"] == img["id"] and a.get("dbh", 0) > 0]
    by_sp = {}
    for a in anns:
        nm = CLASS_NAMES[a["category_id"] - 1]
        by_sp.setdefault(nm, []).append(a["dbh"] * 10.0)  # cm -> mm
    return {k: (len(v), round(np.mean(v), 1)) for k, v in by_sp.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--stem", type=str, default=None)
    args = ap.parse_args()

    gt = json.load(open(GT_JSON))
    stem2path = {os.path.splitext(os.path.basename(im["file_name"]))[0]: im["file_name"]
                 for im in gt["images"]}
    if args.stem:
        stems = [args.stem]
    else:
        random.seed(0)
        stems = [os.path.splitext(os.path.basename(im["file_name"]))[0]
                 for im in random.sample(gt["images"], args.n)]

    cfg, model, head, tfm, dev, sizediv = build()
    print(f"Backbone: {os.path.basename(BACKBONE_CKPT)}")
    print(f"DBH head: Sandbox winner (phaseB_final_baseline_anchor_only)\n")

    for stem in stems:
        res = predict_image(model, head, tfm, dev, sizediv, stem2path.get(stem, ""))
        gtd = gt_dbh_for(stem, gt)
        print("=" * 78)
        print(f"GAMBAR: {stem}")
        if gtd:
            print("  GT DBH per-species (referensi): " +
                  ", ".join(f"{k}: n={v[0]}, rata2={v[1]}mm" for k, v in gtd.items()))
        if not res:
            print("  (tidak ada pohon terprediksi / valid)"); continue
        print(f"  {'#':>2} {'jenis':<12} {'DBH_pred':>10} {'DBH_geom':>10} {'depth':>7} {'trunk_px':>8} {'skor':>6}")
        for i, r in enumerate(sorted(res, key=lambda x: -x["score"]), 1):
            print(f"  {i:>2} {r['species']:<12} {r['dbh_mm']:>8.1f}mm {r['dbh_geom_mm']:>8.1f}mm "
                  f"{r['depth_m']:>6.1f}m {r['trunk_px']:>8} {r['score']:>6.2f}")
        preds = [r["dbh_mm"] for r in res]
        print(f"  -> {len(res)} pohon, DBH prediksi: min={min(preds):.0f} median={np.median(preds):.0f} max={max(preds):.0f} mm")


if __name__ == "__main__":
    main()
