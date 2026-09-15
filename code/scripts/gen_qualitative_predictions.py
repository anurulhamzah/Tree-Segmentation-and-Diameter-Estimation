#!/usr/bin/env python3
"""gen_qualitative_predictions.py — figur kualitatif tesis: prediksi model final (segmentasi
+ jenis + DBH) pada satu contoh plantation dan satu contoh rainforest.

Pipa identik predict_dbh.py (inferensi deteksi standar untuk mask/jenis/skor, backbone
native-padded terpisah untuk feature res2 head DBH), tapi memakai checkpoint JOINT tunggal
(backbone + dbh_head_trunkroi dilatih bersama) via build_model_with_head dari
eval_joint_trunkroi_dbh.py, bukan backbone beku + head Sandbox terpisah.

Jalankan:
    python scripts/gen_qualitative_predictions.py
"""
import sys, os, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

USER = "/scratch2/pr65/anur0018"
MK = f"{USER}/MaskDINO/MaskDINO"
PROJ = f"{USER}/tree_classification"
sys.path.insert(0, MK)
sys.path.insert(0, f"{PROJ}/scripts")

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import ImageList

import maskdino  # noqa: F401
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import build_transform_gen
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import _read_pfm, _normalize_depth
from trunk_roi_dbh_head_hybrid import _find_row_1p3m, _ORIG_CY, _ORIG_FY, _H_CAM
from eval_joint_trunkroi_dbh import build_model_with_head
from register_combined import register_all_combined

OUTPUT_ROOT = Path(f"{USER}/maskdino_output")
OUT_DIR = OUTPUT_ROOT / "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k_cap30_detach"
FIG_DIR = Path(f"{PROJ}/paper/tesis_final/gambar")

CLASS_NAMES = ["Apple", "Lemon", "Loquat", "Mango", "Orange", "Persimmon", "Pomegranate",
               "AliiFig", "BangaloPalm", "Fern", "LeechVine", "RubberFig", "Umbrella"]
SCORE_THR = 0.30
SKIP_SPECIES = {"LeechVine"}

TARGETS = [
    ("Plantation", f"{PROJ}/data/plantations/rgb_resized/Tree62_1721042591.png"),
    ("Rainforest", f"{PROJ}/data/rainforests/rgb_resized/Tree8447_1720691246.png"),
]


@torch.no_grad()
def predict_image(model, tfm, dev, sizediv, rgb_path):
    pfm_path = rgb_path.replace("/rgb_resized/", "/depth_pfm/").replace(".png", ".pfm")
    image = utils.read_image(rgb_path, format="RGB")
    H_img, W_img = image.shape[:2]
    depth_raw = _read_pfm(pfm_path).astype(np.float32)
    if depth_raw.shape != (H_img, W_img):
        import cv2
        depth_raw = cv2.resize(depth_raw, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
    depth_norm01 = _normalize_depth(depth_raw)

    # (A) deteksi instance standar (mapper test)
    img_t, tfms = T.apply_transform_gens(tfm, image.copy())
    dep_t = tfms.apply_image(depth_norm01[:, :, None])[:, :, 0]
    inst_in = {"image": torch.as_tensor(np.ascontiguousarray(img_t.transpose(2, 0, 1))),
               "depth": torch.as_tensor(np.ascontiguousarray(dep_t)),
               "height": H_img, "width": W_img}
    out = model([inst_in])[0]["instances"].to("cpu")
    keep = out.scores >= SCORE_THR
    masks = out.pred_masks[keep].numpy().astype(bool)
    classes = out.pred_classes[keep].numpy()
    scores = out.scores[keep].numpy()

    # (B) feature res2, backbone native-padded (identik precompute/head training)
    depth255 = depth_norm01 * 255.0
    img4 = torch.cat([torch.as_tensor(image.transpose(2, 0, 1).copy(), dtype=torch.float32),
                      torch.as_tensor(depth255, dtype=torch.float32).unsqueeze(0)], dim=0)
    ilist = ImageList.from_tensors([img4], sizediv)
    Hp, Wp = ilist.tensor.shape[-2:]
    imgs_norm = (ilist.tensor.to(dev) - model.pixel_mean) / model.pixel_std
    res2 = model.backbone(imgs_norm)["res2"][0]

    depth_pad = torch.zeros((Hp, Wp), dtype=torch.float32)
    depth_pad[:H_img, :W_img] = torch.as_tensor(depth_raw)

    # (C) per instance TERPREDIKSI (bukan GT): cari row 1.3m dari mask prediksi -> DBH
    head = model.dbh_head_trunkroi
    results = []
    for m, c, s in zip(masks, classes, scores):
        sp = CLASS_NAMES[int(c)]
        dbh_mm = None
        if sp not in SKIP_SPECIES:
            mask_pad = torch.zeros((Hp, Wp), dtype=torch.bool)
            mask_pad[:H_img, :W_img] = torch.as_tensor(m)
            row, trunk_cols, d_trunk, world_h = _find_row_1p3m(
                mask_pad, depth_pad, cy=_ORIG_CY, fy=_ORIG_FY, h_cam=_H_CAM, tol_wh=float("inf"))
            if row is not None:
                strip = head._extract_strip_feat(res2, int(row), torch.as_tensor(trunk_cols, device=dev))
                if strip is not None:
                    trunk_px = len(trunk_cols)
                    dbh_geom_mm = trunk_px * d_trunk * 1000.0 / _ORIG_FY
                    geom = torch.tensor([math.log1p(dbh_geom_mm) / 10.0, d_trunk / 10.0], device=dev)
                    onehot = F.one_hot(torch.tensor(int(c), device=dev), 13).float()
                    x = torch.cat([strip, geom, onehot], dim=0)
                    pred = head.mlp(x).squeeze(-1)
                    dbh_mm = float(torch.expm1(pred).clamp(min=0))
        results.append(dict(mask=m, species=sp, score=float(s), dbh_mm=dbh_mm))
    return image, results


def draw_panel(ax, image, results, title):
    import matplotlib.patches as mpatches
    rng = np.random.RandomState(0)
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22",
              "#e84393", "#00b894", "#0984e3", "#fdcb6e", "#6c5ce7", "#d35400"]
    ax.imshow(image)
    H, W = image.shape[:2]
    overlay = np.zeros((H, W, 4))
    for i, r in enumerate(sorted(results, key=lambda x: -x["score"])):
        col = colors[i % len(colors)]
        rgba = tuple(int(col[j:j+2], 16) / 255.0 for j in (1, 3, 5)) + (0.45,)
        overlay[r["mask"]] = rgba
        ys, xs = np.where(r["mask"])
        if len(xs) == 0:
            continue
        cx, cy = xs.mean(), ys.min()
        dbh_s = f"{r['dbh_mm']/10:.0f}cm" if r["dbh_mm"] is not None else "n/a"
        label = f"{r['species']}\n{dbh_s}"
        ax.text(cx, max(cy - 4, 6), label, fontsize=5.6, color="white", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=col, alpha=0.85, edgecolor="none"))
    ax.imshow(overlay)
    ax.set_title(title, fontsize=8.5)
    ax.axis("off")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    register_all_combined(os.environ.get("COMBINED_ROOT", f"{PROJ}/data/combined"))
    model, cfg = build_model_with_head(OUT_DIR, device, init_ckpt="model_final.pth")
    model.eval()
    tfm = build_transform_gen(cfg, is_train=False)
    size_div = model.size_divisibility

    panels = []
    for tag, path in TARGETS:
        image, results = predict_image(model, tfm, device, size_div, path)
        panels.append((tag, image, results))
        print(f"=== {tag}: {path} ===")
        for r in sorted(results, key=lambda x: -x["score"]):
            dbh_s = f"{r['dbh_mm']/10:.1f}cm" if r["dbh_mm"] is not None else "n/a"
            print(f"  {r['species']:<12} score={r['score']:.2f} DBH={dbh_s}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, (tag, image, results) in zip(axes, panels):
        draw_panel(ax, image, results, tag)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "fig6_qualitative.pdf"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
