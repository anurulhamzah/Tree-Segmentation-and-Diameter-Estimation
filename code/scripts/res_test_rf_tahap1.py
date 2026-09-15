#!/usr/bin/env python
"""
Tahap 1 — Uji hipotesis "Bukti 3": recall objek kecil dibatasi RESOLUSI INPUT.

Desain (inference-only, tanpa retrain):
  Satu model (FocalNet-L RF RGBD scratch 75k) dievaluasi dalam dua arm yang
  HANYA berbeda pada detail input RGB, dengan resolusi PEMROSESAN network
  identik (mapper test me-resize keduanya ke IMAGE_SIZE=640):
    - Arm 480 : RGB dari rgb_resized (480x270, upsampled dari sumber)
    - Arm 960 : RGB dari rgb_960     (960x540, NATIVE render, detail asli 26x HF)
  Depth (480x270 native) sama untuk kedua arm (di-resize oleh transform yg sama).
  Prediksi arm 960 diskalakan x0.5 ke frame 480 agar dicocokkan ke GT 480 yg SAMA.

Metrik utama = recall per-bin-depth (invariant terhadap resolusi, karena
depth = besaran fisik). Hipotesis terkonfirmasi bila recall bin JAUH naik di arm
960 sementara bin DEKAT relatif datar.

Matching: greedy per-kategori, box-IoU >= 0.5, skor >= 0.35 (konvensi F1 proyek).
Box-IoU dipakai sbg proxy "terdeteksi atau tidak" (mask-IoU = penghalusan lanjutan).
"""
import os, sys, json, argparse, time
from collections import defaultdict

import numpy as np
import torch

USER_ROOT = "/scratch2/pr65/anur0018"
MK = f"{USER_ROOT}/MaskDINO/MaskDINO"
PROJECT = f"{USER_ROOT}/tree_classification"
sys.path.insert(0, MK)
sys.path.insert(0, f"{PROJECT}/scripts")

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import detection_utils as utils

import maskdino  # noqa: F401  (registers meta-arch + configs)
from maskdino import add_maskdino_config
from detectron2.data import transforms as T
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
    build_transform_gen,
)
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import (
    _read_pfm, _normalize_depth,
)

DEPTH_DIR = f"{PROJECT}/data/rainforests/depth_pfm"

RF_DIR = f"{USER_ROOT}/maskdino_output/FocalNet_L_rf_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k"
CONFIG = f"{RF_DIR}/config.yaml"
CKPT = f"{RF_DIR}/model_final.pth"
GT_JSON = f"{PROJECT}/data/rainforests/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/instances_test.json"
RGB_480 = f"{PROJECT}/data/rainforests/rgb_resized"
RGB_960 = f"{PROJECT}/data/rainforests/rgb_960"
OUT_JSON = f"{PROJECT}/reports/res_test_rf_tahap1.json"

SCORE_THR = 0.35
IOU_THR = 0.50
# Bin depth (meter). Median GT ~11.3m, q25=7.7, q75=17.9.
DEPTH_BINS = [(0, 8), (8, 12), (12, 18), (18, 30), (30, 1e9)]
DEPTH_LABELS = ["<8m (dekat)", "8-12m", "12-18m", "18-30m", ">30m (jauh)"]


def build():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = CKPT
    cfg.freeze()
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return cfg, model


def box_iou(a, b):
    # a: [N,4] xyxy, b: [M,4] xyxy
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    iou = np.zeros((len(a), len(b)), dtype=np.float32)
    for i in range(len(a)):
        xx1 = np.maximum(a[i, 0], b[:, 0]); yy1 = np.maximum(a[i, 1], b[:, 1])
        xx2 = np.minimum(a[i, 2], b[:, 2]); yy2 = np.minimum(a[i, 3], b[:, 3])
        w = (xx2 - xx1).clip(0); h = (yy2 - yy1).clip(0)
        inter = w * h
        iou[i] = inter / (area_a[i] + area_b - inter + 1e-9)
    return iou


def depth_bin(d):
    for i, (lo, hi) in enumerate(DEPTH_BINS):
        if lo <= d < hi:
            return i
    return len(DEPTH_BINS) - 1


def make_tfm(sz):
    # Deterministic test resize to sz x sz (longest edge), no pad — mirrors
    # build_transform_gen(is_train=False) but with configurable processing size.
    return [
        T.ResizeScale(min_scale=1.0, max_scale=1.0, target_height=sz, target_width=sz),
        T.FixedSizeCrop(crop_size=(sz, sz), pad=False),
    ]


@torch.no_grad()
def infer(model, tfm_gens, file_name, depth_stem, H, W):
    # Replicates COCOInstanceDBHDatasetMapper test-branch, but upsamples depth
    # to the RGB image size first (native depth is 480x270; arm960 RGB is 960x540).
    image = utils.read_image(file_name, format="RGB")
    depth = _normalize_depth(_read_pfm(os.path.join(DEPTH_DIR, depth_stem + ".pfm")))
    if depth.shape[:2] != image.shape[:2]:
        # exact 2x for 960x540; nearest-neighbour (adds NO new depth detail)
        fy = image.shape[0] // depth.shape[0]
        fx = image.shape[1] // depth.shape[1]
        depth = depth.repeat(fy, axis=0).repeat(fx, axis=1)
        if depth.shape[:2] != image.shape[:2]:
            depth = np.asarray(
                __import__("PIL.Image", fromlist=["Image"]).fromarray(depth).resize(
                    (image.shape[1], image.shape[0]))
            )
    image2, transforms = T.apply_transform_gens(tfm_gens, image)
    depth2 = transforms.apply_image(depth[:, :, np.newaxis])[:, :, 0]
    inp = {
        "image": torch.as_tensor(np.ascontiguousarray(image2.transpose(2, 0, 1))),
        "depth": torch.as_tensor(np.ascontiguousarray(depth2)),
        "height": H, "width": W,
    }
    out = model([inp])[0]["instances"].to("cpu")
    boxes = out.pred_boxes.tensor.numpy() if len(out) else np.zeros((0, 4))
    scores = out.scores.numpy() if len(out) else np.zeros((0,))
    classes = out.pred_classes.numpy() if len(out) else np.zeros((0,), dtype=int)
    return boxes, scores, classes


def match_recall(gt_list, boxes, scores, classes, scale):
    """gt_list: [(cat_id 1-6, bbox_xywh, depth)]. Preds classes 0-5 -> +1.
    Returns per-bin (matched, total) and overall (matched, total)."""
    boxes = boxes.copy()
    if len(boxes):
        boxes *= scale  # to 480 frame
    keep = scores >= SCORE_THR
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    pcat = classes + 1
    # GT boxes xywh -> xyxy
    per_bin = defaultdict(lambda: [0, 0])  # bin -> [matched, total]
    matched_flags = []
    by_cat_gt = defaultdict(list)
    for idx, (cid, bb, dep) in enumerate(gt_list):
        by_cat_gt[cid].append(idx)
    gt_matched = [False] * len(gt_list)
    for cid in by_cat_gt:
        gidx = by_cat_gt[cid]
        gboxes = np.array([[gt_list[i][1][0], gt_list[i][1][1],
                            gt_list[i][1][0] + gt_list[i][1][2],
                            gt_list[i][1][1] + gt_list[i][1][3]] for i in gidx])
        pmask = pcat == cid
        pidx = np.where(pmask)[0]
        if len(pidx) == 0:
            continue
        pboxes = boxes[pidx]
        psc = scores[pidx]
        order = np.argsort(-psc)  # high score first
        iou = box_iou(pboxes, gboxes)
        used_g = set()
        for oi in order:
            row = iou[oi].copy()
            for g in used_g:
                row[g] = 0
            best = np.argmax(row) if len(row) else -1
            if best >= 0 and row[best] >= IOU_THR:
                used_g.add(best)
                gt_matched[gidx[best]] = True
    # tally by bin
    for i, (cid, bb, dep) in enumerate(gt_list):
        b = depth_bin(dep)
        per_bin[b][1] += 1
        if gt_matched[i]:
            per_bin[b][0] += 1
    total = len(gt_list)
    matched = sum(gt_matched)
    return per_bin, matched, total


def main():
    t0 = time.time()
    gt = json.load(open(GT_JSON))
    imgs = {im["id"]: im for im in gt["images"]}
    anns_by_img = defaultdict(list)
    for a in gt["annotations"]:
        d = a.get("depth_mean")
        if not isinstance(d, (int, float)) or d <= 0:
            continue
        anns_by_img[a["image_id"]].append((a["category_id"], a["bbox"], float(d)))

    cfg, model = build()

    results = {}
    # (label, rgb_dir, scale_to_480, (H,W), processing_size)
    # proc=640 = model's trained resolution. proc>640 tests whether giving the
    # backbone more pixels on the trunk (native detail) recovers far-object recall.
    arms = [
        ("arm480_p640",  RGB_480, 1.0, (270, 480), 640),
        ("arm960_p640",  RGB_960, 0.5, (540, 960), 640),
        ("arm960_p960",  RGB_960, 0.5, (540, 960), 960),
        ("arm960_p1280", RGB_960, 0.5, (540, 960), 1280),
    ]
    for arm, rgb_dir, scale, dims, proc in arms:
        H, W = dims
        tfm_gens = make_tfm(proc)
        agg_bin = defaultdict(lambda: [0, 0])
        tot_m = tot_n = 0
        n_img = 0
        limit = int(os.environ.get("LIMIT", "0"))
        for iid, im in imgs.items():
            if limit and n_img >= limit:
                break
            base = os.path.basename(im["file_name"])
            fp = os.path.join(rgb_dir, base)
            if not os.path.exists(fp):
                continue
            gl = anns_by_img.get(iid, [])
            if not gl:
                continue
            stem = os.path.splitext(base)[0]
            boxes, scores, classes = infer(model, tfm_gens, fp, stem, H, W)
            pb, m, n = match_recall(gl, boxes, scores, classes, scale)
            for b, (mm, nn) in pb.items():
                agg_bin[b][0] += mm; agg_bin[b][1] += nn
            tot_m += m; tot_n += n; n_img += 1
        results[arm] = {
            "n_images": n_img,
            "overall_recall": round(tot_m / max(tot_n, 1), 4),
            "matched": tot_m, "total": tot_n,
            "per_depth_bin": {
                DEPTH_LABELS[b]: {
                    "recall": round(agg_bin[b][0] / max(agg_bin[b][1], 1), 4),
                    "matched": agg_bin[b][0], "total": agg_bin[b][1],
                } for b in range(len(DEPTH_BINS))
            },
        }
        print(f"[{time.time()-t0:6.0f}s] {arm}: overall recall={results[arm]['overall_recall']:.4f} "
              f"(N={tot_n}, imgs={n_img})")

    # comparison table: recall per depth bin across all arms
    arm_keys = [a[0] for a in arms]
    print("\n=== RECALL PER BIN Depth (proxy box-IoU>=0.5, skor>=0.35) ===")
    hdr = f"{'bin':16s} {'N':>6s}" + "".join(f"{k.replace('arm',''):>13s}" for k in arm_keys)
    print(hdr)
    for b in range(len(DEPTH_BINS)):
        lab = DEPTH_LABELS[b]
        row = f"{lab:16s} {results[arm_keys[0]]['per_depth_bin'][lab]['total']:>6d}"
        for k in arm_keys:
            row += f"{results[k]['per_depth_bin'][lab]['recall']:>13.4f}"
        print(row)
    row = f"{'OVERALL':16s} {results[arm_keys[0]]['total']:>6d}"
    for k in arm_keys:
        row += f"{results[k]['overall_recall']:>13.4f}"
    print(row)
    base = results["arm480_p640"]["overall_recall"]
    print("\ndelta overall vs arm480_p640 (baseline): " +
          ", ".join(f"{k.replace('arm','')}={results[k]['overall_recall']-base:+.4f}" for k in arm_keys))

    results["_meta"] = {
        "score_thr": SCORE_THR, "iou_thr": IOU_THR,
        "note": "Arm A recall harus ~mirip Recall@0.35 RF known=0.4595 (mask-based) sbg sanity.",
        "gt": GT_JSON, "ckpt": CKPT,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nSaved -> {OUT_JSON}  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
