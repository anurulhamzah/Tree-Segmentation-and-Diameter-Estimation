"""
Evaluasi hybrid_best.pth dengan breakdown:
  - Full val set (semua spesies)
  - Valid DBH species only (exclude plantation preset)
  - Per-spesies breakdown
"""
import sys, os, json, math
from pathlib import Path
from collections import defaultdict

HERE          = Path(__file__).resolve().parent
PROJECT_ROOT  = HERE.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(HERE))

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
from trunk_roi_dbh_head_hybrid import HybridTrunkROIDBHHead

import argparse, logging
logger = logging.getLogger("eval_hybrid_valid")

CAT_NAMES = {
    1:'Apple', 2:'Lemon', 3:'Loquat', 4:'Mango', 5:'Orange', 6:'Persimmon',
    7:'Pomegranate', 8:'AliiFig', 9:'BangaloPalm', 10:'Fern',
    11:'LeechVine', 12:'RubberFig', 13:'Umbrella'
}

# Spesies dengan DBH preset Unreal Engine — bukan diukur di 1.3m
PLANTATION_PRESET = {1, 2, 3, 5, 6, 7}   # Apple,Lemon,Loquat,Orange,Persimmon,Pomegranate


def r2_score(p, g):
    ss_res = np.sum((g - p) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def print_metrics(label, p, g, cats=None):
    e = np.abs(p - g)
    r2 = r2_score(p, g)
    bias = (p - g).mean()
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    print(f"  N={len(p)}  R²={r2:.4f}  MAE={e.mean():.1f}mm  RMSE={np.sqrt((e**2).mean()):.1f}mm")
    print(f"  Bias={bias:+.1f}mm  ±50mm={( e<50).mean()*100:.1f}%  ±100mm={(e<100).mean()*100:.1f}%")
    if cats is not None:
        print(f"\n  Per-spesies R²:")
        for cid in sorted(set(cats)):
            idx = [k for k,c in enumerate(cats) if c==cid]
            gi = g[idx]; pi = p[idx]; ei = np.abs(gi-pi)
            r2c = r2_score(pi, gi)
            tag = " [PLANTATION]" if cid in PLANTATION_PRESET else ""
            print(f"    {CAT_NAMES.get(cid,cid):<14}  N={len(idx):>5}  R²={r2c:>7.3f}  "
                  f"MAE={ei.mean():>6.0f}mm{tag}")
    return {"N": len(p), "R2": float(r2), "MAE": float(e.mean()),
            "RMSE": float(np.sqrt((e**2).mean())), "bias": float(bias),
            "w50mm": float((e<50).mean()*100), "w100mm": float((e<100).mean()*100)}


def main(args):
    register_all_rainforests(os.environ.get("RAINFORESTS_ROOT",
        str(PROJECT_ROOT / "data" / "rainforests")))
    register_all_plantations(os.environ.get("PLANTATIONS_ROOT",
        str(PROJECT_ROOT / "data" / "plantations")))
    register_all_combined(os.environ.get("COMBINED_ROOT",
        str(PROJECT_ROOT / "data" / "combined")))

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    default_setup(cfg, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maskdino = build_model(cfg).to(device)
    DetectionCheckpointer(maskdino).load(cfg.MODEL.WEIGHTS)
    maskdino.eval()
    for p in maskdino.parameters():
        p.requires_grad = False

    trunk_head = HybridTrunkROIDBHHead(in_channels=192, hidden=256,
                                        strip_rows=1, num_species=13).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    trunk_head.load_state_dict(ckpt)
    trunk_head.eval()
    logger.info(f"Loaded: {args.checkpoint}")

    mapper_val = COCOInstanceDBHDatasetMapper(cfg, True)   # Bug #6 fix
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper_val)

    preds_all, gts_all, cats_all = [], [], []
    n_skip = 0; n_total = 0

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
                    result = trunk_head.forward_single(
                        feat_b, depth_b, masks_b[j].bool(),
                        species_idx=classes_b[j].item(), strict=False)
                    if result is None:
                        n_skip += 1
                        continue
                    preds_all.append(torch.expm1(result[0]).item())
                    gts_all.append(dbh_cm * 10.0)
                    cats_all.append(classes_b[j].item() + 1)  # 0-based → 1-based

    if not preds_all:
        print(f"ALL SKIPPED: {n_skip}/{n_total}")
        return

    p_all = np.array(preds_all)
    g_all = np.array(gts_all)
    c_all = np.array(cats_all)

    print(f"\n  skip={n_skip}/{n_total} ({n_skip/max(n_total,1)*100:.1f}%)")

    # 1. Semua spesies
    res_all = print_metrics("FULL val set (semua spesies)", p_all, g_all, c_all.tolist())

    # 2. Valid DBH species saja (exclude plantation preset)
    mask_valid = np.array([c not in PLANTATION_PRESET for c in c_all])
    p_v = p_all[mask_valid]; g_v = g_all[mask_valid]; c_v = c_all[mask_valid]
    res_valid = print_metrics(
        f"VALID DBH species only (N={mask_valid.sum()}, excl Apple/Lemon/Loquat/Orange/Persimmon/Pomegranate)",
        p_v, g_v, c_v.tolist())

    # 3. Plantation saja (untuk referensi)
    mask_pl = ~mask_valid
    p_pl = p_all[mask_pl]; g_pl = g_all[mask_pl]; c_pl = c_all[mask_pl]
    if len(p_pl) > 0:
        res_pl = print_metrics("PLANTATION only (untuk referensi)", p_pl, g_pl, c_pl.tolist())

    # Simpan hasil
    out = {
        "checkpoint": args.checkpoint,
        "full": res_all,
        "valid_species": res_valid,
        "plantation_only": res_pl if len(p_pl) > 0 else None,
        "n_skip": n_skip, "n_total": n_total,
        "valid_species_excluded": ["Apple","Lemon","Loquat","Orange","Persimmon","Pomegranate"]
    }
    out_path = Path(args.checkpoint).parent / "eval_hybrid_best_valid_species.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    setup_logger(name="eval_hybrid_valid")
    logger = setup_logger(name="eval_hybrid_valid")
    main(args)
