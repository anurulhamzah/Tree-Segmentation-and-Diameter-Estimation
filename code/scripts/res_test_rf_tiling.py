#!/usr/bin/env python
"""
Tahap 1b — Uji TILING/SAHI: rute inference-only tersisa utk bottleneck resolusi.

Ide: potong gambar 960x540 native jadi tile berukuran 480x270 (PERSIS ukuran
training) dengan overlap, tiap tile diproses di IMAGE_SIZE=640 (resolusi terlatih).
Objek tetap pada skala apparent yg dipelajari model (tak ada train/test mismatch),
tapi trunk dapat ~2x piksel efektif dibanding whole-960->640. Prediksi antar-tile
digabung via NMS global per-kategori, lalu diskalakan x0.5 ke frame 480 dan
dicocokkan ke GT 480 yg sama. Bandingkan recall per-bin-depth vs baseline
whole-image (arm480_p640=0.4846 dari Tahap 1a).

Reuse harness: scripts/res_test_rf_tahap1.py.
"""
import os, sys, json, time
from collections import defaultdict

import numpy as np
import torch
from torchvision.ops import nms

USER_ROOT = "/scratch2/pr65/anur0018"
PROJECT = f"{USER_ROOT}/tree_classification"
sys.path.insert(0, f"{USER_ROOT}/MaskDINO/MaskDINO")
sys.path.insert(0, f"{PROJECT}/scripts")

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import res_test_rf_tahap1 as base  # build, match_recall, depth_bin, DEPTH_*, infer, make_tfm
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import (
    _read_pfm, _normalize_depth,
)

RGB_480 = f"{PROJECT}/data/rainforests/rgb_resized"
RGB_960 = f"{PROJECT}/data/rainforests/rgb_960"
DEPTH_DIR = f"{PROJECT}/data/rainforests/depth_pfm"
GT_JSON = base.GT_JSON
OUT_JSON = f"{PROJECT}/reports/res_test_rf_tiling.json"

TILE_W, TILE_H = 480, 270      # = ukuran gambar training (skala apparent terlatih)
OVERLAP = 0.25                  # fraksi tumpang-tindih antar tile
NMS_IOU = 0.5                   # dedup deteksi antar-tile
SCORE_KEEP = 0.05               # simpan utk NMS; threshold 0.35 diterapkan saat match


def tile_origins(full, tile, overlap):
    stride = max(1, int(round(tile * (1 - overlap))))
    xs = list(range(0, max(full - tile, 0) + 1, stride))
    if not xs or xs[-1] != full - tile:
        xs.append(max(full - tile, 0))
    return sorted(set(xs))


@torch.no_grad()
def infer_tiled(model, tfm_gens, rgb960_path, depth_stem):
    image = utils.read_image(rgb960_path, format="RGB")            # 540x960x3
    depth = _normalize_depth(_read_pfm(os.path.join(DEPTH_DIR, depth_stem + ".pfm")))  # 270x480
    if depth.shape[:2] != image.shape[:2]:
        fy = image.shape[0] // depth.shape[0]; fx = image.shape[1] // depth.shape[1]
        depth = depth.repeat(fy, axis=0).repeat(fx, axis=1)         # 540x960
    H, W = image.shape[:2]
    xs = tile_origins(W, TILE_W, OVERLAP)
    ys = tile_origins(H, TILE_H, OVERLAP)
    aboxes, ascores, aclasses = [], [], []
    for y0 in ys:
        for x0 in xs:
            crop_img = np.ascontiguousarray(image[y0:y0 + TILE_H, x0:x0 + TILE_W])
            crop_dep = np.ascontiguousarray(depth[y0:y0 + TILE_H, x0:x0 + TILE_W])
            img2, tfms = T.apply_transform_gens(tfm_gens, crop_img)
            dep2 = tfms.apply_image(crop_dep[:, :, np.newaxis])[:, :, 0]
            inp = {
                "image": torch.as_tensor(np.ascontiguousarray(img2.transpose(2, 0, 1))),
                "depth": torch.as_tensor(np.ascontiguousarray(dep2)),
                "height": TILE_H, "width": TILE_W,
            }
            out = model([inp])[0]["instances"].to("cpu")
            if len(out) == 0:
                continue
            b = out.pred_boxes.tensor.numpy().copy()
            b[:, [0, 2]] += x0; b[:, [1, 3]] += y0                 # -> 960 frame
            aboxes.append(b)
            ascores.append(out.scores.numpy())
            aclasses.append(out.pred_classes.numpy())
    if not aboxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.concatenate(aboxes); scores = np.concatenate(ascores)
    classes = np.concatenate(aclasses)
    keep_s = scores >= SCORE_KEEP
    boxes, scores, classes = boxes[keep_s], scores[keep_s], classes[keep_s]
    # global per-category NMS to merge duplicates across overlapping tiles
    fb, fs, fc = [], [], []
    for c in np.unique(classes):
        m = classes == c
        tb = torch.as_tensor(boxes[m], dtype=torch.float32)
        ts = torch.as_tensor(scores[m], dtype=torch.float32)
        k = nms(tb, ts, NMS_IOU).numpy()
        fb.append(boxes[m][k]); fs.append(scores[m][k]); fc.append(classes[m][k])
    return np.concatenate(fb), np.concatenate(fs), np.concatenate(fc)


def summarize(agg_bin, tot_m, tot_n):
    return {
        "overall_recall": round(tot_m / max(tot_n, 1), 4),
        "matched": tot_m, "total": tot_n,
        "per_depth_bin": {
            base.DEPTH_LABELS[b]: {
                "recall": round(agg_bin[b][0] / max(agg_bin[b][1], 1), 4),
                "matched": agg_bin[b][0], "total": agg_bin[b][1],
            } for b in range(len(base.DEPTH_BINS))
        },
    }


def main():
    t0 = time.time()
    gt = json.load(open(GT_JSON))
    imgs = {im["id"]: im for im in gt["images"]}
    anns_by_img = defaultdict(list)
    for a in gt["annotations"]:
        d = a.get("depth_mean")
        if isinstance(d, (int, float)) and d > 0:
            anns_by_img[a["image_id"]].append((a["category_id"], a["bbox"], float(d)))

    cfg, model = base.build()
    tfm640 = base.make_tfm(640)
    limit = int(os.environ.get("LIMIT", "0"))

    results = {}
    # baseline whole-image 480 @640 (reproduksi Tahap 1a utk harness identik)
    for arm in ("baseline480_p640", "tiling960"):
        agg = defaultdict(lambda: [0, 0]); tm = tn = ni = 0
        for iid, im in imgs.items():
            if limit and ni >= limit:
                break
            base_name = os.path.basename(im["file_name"])
            stem = os.path.splitext(base_name)[0]
            gl = anns_by_img.get(iid, [])
            if not gl:
                continue
            if arm == "baseline480_p640":
                fp = os.path.join(RGB_480, base_name)
                if not os.path.exists(fp):
                    continue
                boxes, scores, classes = base.infer(model, tfm640, fp, stem, 270, 480)
                scale = 1.0
            else:
                fp = os.path.join(RGB_960, base_name)
                if not os.path.exists(fp):
                    continue
                boxes, scores, classes = infer_tiled(model, tfm640, fp, stem)
                scale = 0.5
            pb, m, n = base.match_recall(gl, boxes, scores, classes, scale)
            for b, (mm, nn) in pb.items():
                agg[b][0] += mm; agg[b][1] += nn
            tm += m; tn += n; ni += 1
        results[arm] = summarize(agg, tm, tn)
        print(f"[{time.time()-t0:6.0f}s] {arm}: overall recall={results[arm]['overall_recall']:.4f} "
              f"(N={tn}, imgs={ni})")

    print("\n=== RECALL PER BIN Depth (box-IoU>=0.5, skor>=0.35) ===")
    print(f"{'bin':16s} {'N':>6s} {'whole480':>10s} {'tiling960':>10s} {'delta':>9s}")
    for b in range(len(base.DEPTH_BINS)):
        lab = base.DEPTH_LABELS[b]
        a = results["baseline480_p640"]["per_depth_bin"][lab]
        c = results["tiling960"]["per_depth_bin"][lab]
        print(f"{lab:16s} {a['total']:>6d} {a['recall']:>10.4f} {c['recall']:>10.4f} "
              f"{c['recall']-a['recall']:>+9.4f}")
    a = results["baseline480_p640"]["overall_recall"]
    c = results["tiling960"]["overall_recall"]
    print(f"{'OVERALL':16s} {results['baseline480_p640']['total']:>6d} "
          f"{a:>10.4f} {c:>10.4f} {c-a:>+9.4f}")

    results["_meta"] = {
        "tile": [TILE_W, TILE_H], "overlap": OVERLAP, "nms_iou": NMS_IOU,
        "score_thr_match": base.SCORE_THR, "iou_thr_match": base.IOU_THR,
        "note": "tile 480x270 = ukuran training; tiap tile diproses @640 (skala terlatih).",
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nSaved -> {OUT_JSON}  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
