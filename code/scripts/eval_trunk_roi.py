"""
Evaluasi manual trunk_roi_latest.pth dari job 58104286.
Fix Bug #6: pakai is_train=True di val mapper.
"""
import sys, os
from pathlib import Path

HERE          = Path(__file__).resolve().parent
PROJECT_ROOT  = HERE.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(HERE))

import argparse, json, math
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.structures import ImageList
from maskdino import add_maskdino_config
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import COCOInstanceDBHDatasetMapper
from register_combined import register_all_combined
from register_rainforests import register_all_rainforests
from register_plantations import register_all_plantations
from trunk_roi_dbh_head import TrunkROIDBHHead

import logging
logger = logging.getLogger("eval_trunk_roi")


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def evaluate(maskdino, trunk_head, val_loader, device, split_label="val"):
    trunk_head.eval()
    maskdino.eval()
    preds_all, gts_all = [], []
    cats_all = []
    n_skip, n_total = 0, 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            depths_raw = [x.get("depth") for x in batch]
            if any(d is None for d in depths_raw):
                continue
            depths_norm = [d.to(device) for d in depths_raw]
            depths_4ch  = [d.unsqueeze(0).mul(255.0) for d in depths_norm]
            imgs_4ch    = [torch.cat([x["image"].to(device), d], dim=0)
                           for x, d in zip(batch, depths_4ch)]
            imgs_norm   = [(img - maskdino.pixel_mean) / maskdino.pixel_std
                           for img in imgs_4ch]
            img_list  = ImageList.from_tensors(imgs_norm, maskdino.size_divisibility)
            features  = maskdino.backbone(img_list.tensor)
            res2      = features["res2"]

            instances = [x.get("instances") for x in batch]
            for b, inst in enumerate(instances):
                if inst is None or not inst.has("gt_dbh"):
                    continue
                feat_b    = res2[b]
                depth_b   = depths_norm[b]
                masks_b   = inst.gt_masks.to(device)
                dbhs_b    = inst.gt_dbh.to(device)
                classes_b = inst.gt_classes.to(device)

                for j in range(len(inst)):
                    dbh_cm = dbhs_b[j].item()
                    if dbh_cm <= 0:
                        continue
                    n_total += 1
                    gt_mm = dbh_cm * 10.0
                    pred  = trunk_head.forward_single(feat_b, depth_b, masks_b[j].bool())
                    if pred is None:
                        n_skip += 1
                        continue
                    preds_all.append(torch.expm1(pred).item())
                    gts_all.append(gt_mm)
                    cats_all.append(classes_b[j].item() + 1)   # 0-based → 1-based cat_id

    if not preds_all:
        print(f"[{split_label}] ALL SKIPPED: n_skip={n_skip}/{n_total}")
        return

    p  = np.array(preds_all)
    g  = np.array(gts_all)
    e  = np.abs(p - g)
    r2 = 1 - np.sum((g-p)**2) / np.sum((g-g.mean())**2)

    CAT_NAMES = {
        1:'Apple',2:'Lemon',3:'Loquat',4:'Mango',5:'Orange',6:'Persimmon',
        7:'Pomegranate',8:'AliiFig',9:'BangaloPalm',10:'Fern',11:'LeechVine',
        12:'RubberFig',13:'Umbrella'
    }

    print(f"\n{'='*55}")
    print(f"[{split_label}] TrunkROI (iter 18k) — FULL val set")
    print(f"{'='*55}")
    print(f"  N={len(p)}  skip={n_skip}/{n_total}  ({n_skip/max(n_total,1)*100:.1f}% skip)")
    print(f"  R²    = {r2:.4f}")
    print(f"  MAE   = {e.mean():.1f} mm")
    print(f"  RMSE  = {np.sqrt((e**2).mean()):.1f} mm")
    print(f"  Bias  = {(p-g).mean():.1f} mm")
    print(f"  ±50mm = {(e<50).mean()*100:.1f}%")
    print(f"  ±100mm= {(e<100).mean()*100:.1f}%")

    print(f"\n  Per-spesies R²:")
    for cid in sorted(set(cats_all)):
        idx = [k for k,c in enumerate(cats_all) if c==cid]
        gi = g[idx]; pi = p[idx]; ei = np.abs(gi-pi)
        ss_r = np.sum((gi-pi)**2); ss_t = np.sum((gi-gi.mean())**2)
        r2_c = 1-ss_r/ss_t if ss_t>0 else float('nan')
        print(f"    {CAT_NAMES.get(cid,cid):<14} N={len(idx):4d}  R²={r2_c:.3f}  MAE={ei.mean():.0f}mm")

    result = {"N": len(p), "n_skip": n_skip, "n_total": n_total,
              "R2": float(r2), "MAE": float(e.mean()),
              "RMSE": float(np.sqrt((e**2).mean())),
              "bias": float((p-g).mean()),
              "w50mm": float((e<50).mean()*100),
              "w100mm": float((e<100).mean()*100)}
    out_path = Path(args.checkpoint).parent / "eval_trunk_roi_iter18k.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")


def main(args):
    register_all_rainforests(os.environ.get("RAINFORESTS_ROOT",
        str(PROJECT_ROOT / "data" / "rainforests")))
    register_all_plantations(os.environ.get("PLANTATIONS_ROOT",
        str(PROJECT_ROOT / "data" / "plantations")))
    register_all_combined(os.environ.get("COMBINED_ROOT",
        str(PROJECT_ROOT / "data" / "combined")))

    cfg    = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    maskdino = build_model(cfg).to(device)
    DetectionCheckpointer(maskdino).load(cfg.MODEL.WEIGHTS)
    maskdino.eval()
    for p in maskdino.parameters():
        p.requires_grad = False
    logger.info("MaskDINO loaded.")

    trunk_head = TrunkROIDBHHead(in_channels=192, hidden=256, strip_rows=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    trunk_head.load_state_dict(ckpt)
    trunk_head.eval()
    logger.info(f"TrunkROIDBHHead loaded from {args.checkpoint}")

    # Bug #6 fix: is_train=True agar gt_dbh tersedia
    mapper_val = COCOInstanceDBHDatasetMapper(cfg, True)
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper_val)

    evaluate(maskdino, trunk_head, val_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",  required=True)
    parser.add_argument("--model-weights", required=True,
                        help="MaskDINO backbone model_final.pth")
    parser.add_argument("--checkpoint",   required=True,
                        help="trunk_roi_latest.pth dari job 58104286")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    setup_logger(name="eval_trunk_roi")
    main(args)
