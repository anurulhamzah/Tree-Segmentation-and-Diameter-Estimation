#!/usr/bin/env python3
"""
Stage-2 DBH Fine-tune: FocalNet-L RGBD Combined V4 Stratified
══════════════════════════════════════════════════════════════
Memuat checkpoint FocalNet-L RGBD Scratch (terbaik untuk segmentasi),
membekukan backbone + decoder, lalu melatih hanya dbh_embed selama 20k
iterasi dengan dua perbaikan kunci vs training joint sebelumnya:

  1. FREEZE backbone + pixel_decoder + transformer decoder
     → dbh_embed mendapat 100% sinyal gradient
  2. Per-species z-score normalization di log-space
     → RubberFig (372mm) tidak lagi mendominasi loss

Usage:
    cd /scratch2/pr65/anur0018/tree_classification
    python scripts/finetune_focal_dbh.py \\
        --config-file configs/maskdino_FocalNet_L_combined_rle_f1000_rgbd_scratch_dbh_finetune_v4_stratified_20k_lr1e3.yaml \\
        --model-weights /scratch2/pr65/anur0018/maskdino_output/FocalNet_L_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k/model_final.pth \\
        --num-gpus 1
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import copy
import itertools
from typing import Any, Dict, List, Set

import numpy as np
import torch
import torch.nn.functional as F

HERE          = Path(__file__).resolve().parent
PROJECT_ROOT  = HERE.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"

sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(HERE))

# ── Dataset registration ───────────────────────────────────────────────────────
from register_combined import register_all_combined
from register_rainforests import register_all_rainforests
from register_plantations import register_all_plantations

# ── Detectron2 + MaskDINO ─────────────────────────────────────────────────────
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, EventStorage

from maskdino import add_maskdino_config
from maskdino.modeling.criterion import SetCriterion
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import COCOInstanceDBHDatasetMapper
from detectron2.data import build_detection_train_loader, build_detection_test_loader

# ── Per-species DBH statistics in log-space (computed from V4 train set) ──────
# target = log1p(dbh_mm); normalize → z = (target - mu) / sigma per cat_id
CAT_LOG_MEAN = {
    1: 3.0991,   # Apple
    2: 2.8020,   # Lemon
    3: 2.3825,   # Loquat
    4: 3.2736,   # Mango
    5: 2.9299,   # Orange
    6: 2.7118,   # Persimmon
    7: 3.1230,   # Pomegranate
    8: 3.9572,   # AliiFig
    9: 3.1455,   # BangaloPalm
    10: 3.8777,  # Fern
    11: 3.3954,  # LeechVine
    12: 5.5931,  # RubberFig  ← was the main culprit (mean=372mm)
    13: 3.5910,  # Umbrella
}
CAT_LOG_STD = {
    1: 0.1347,   # Apple
    2: 0.0531,   # Lemon
    3: 0.3175,   # Loquat
    4: 0.2660,   # Mango
    5: 0.1826,   # Orange
    6: 0.0702,   # Persimmon
    7: 0.1995,   # Pomegranate
    8: 0.2657,   # AliiFig
    9: 0.4212,   # BangaloPalm
    10: 0.2726,  # Fern
    11: 0.1704,  # LeechVine
    12: 0.9506,  # RubberFig
    13: 0.3434,  # Umbrella
}

logger = logging.getLogger("finetune_focal_dbh")


# ══════════════════════════════════════════════════════════════════════════════
# Custom criterion: per-species z-score DBH loss
# ══════════════════════════════════════════════════════════════════════════════
class DBHNormCriterion(SetCriterion):
    """Replaces loss_dbh with per-species log-space z-score normalization."""

    def loss_dbh(self, outputs, targets, indices, num_masks):
        if "pred_dbh" not in outputs:
            device = next(iter(outputs.values())).device
            return {"loss_dbh": torch.as_tensor(0.0, device=device)}

        src_idx = self._get_src_permutation_idx(indices)
        src_dbh = outputs["pred_dbh"][src_idx]                               # (N,) predicted log1p(mm)
        tgt_dbh = torch.cat([t["dbh"][J] for t, (_, J) in zip(targets, indices)])    # (N,) raw mm
        tgt_cls = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # (N,) cat_id

        valid = tgt_dbh > 0
        if valid.sum() == 0:
            return {"loss_dbh": src_dbh.sum() * 0}

        pred_log = src_dbh[valid]                      # head output: log1p(mm) space
        gt_log   = torch.log1p(tgt_dbh[valid])         # GT in log1p(mm) space
        cat_ids  = tgt_cls[valid]

        # Normalize to z-score per species in log-space
        pred_norm = torch.zeros_like(pred_log)
        gt_norm   = torch.zeros_like(gt_log)

        for cid in cat_ids.unique():
            mask  = cat_ids == cid
            cid_i = cid.item()
            mu    = CAT_LOG_MEAN.get(cid_i, 3.5)
            sigma = CAT_LOG_STD.get(cid_i, 0.5)
            pred_norm[mask] = (pred_log[mask] - mu) / sigma
            gt_norm[mask]   = (gt_log[mask]   - mu) / sigma

        loss = F.smooth_l1_loss(pred_norm, gt_norm, reduction="mean", beta=1.0)
        return {"loss_dbh": loss}


# ══════════════════════════════════════════════════════════════════════════════
# Trainer with freeze + criterion replacement
# ══════════════════════════════════════════════════════════════════════════════
class DBHFineTuner(DefaultTrainer):

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """MaskDINO-compatible optimizer with full_model gradient clipping support."""
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}

        norm_module_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm, torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm, torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "relative_position_bias_table" in module_param_name or \
                        "absolute_pos_embed" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = COCOInstanceDBHDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = COCOInstanceDBHDatasetMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    def _replace_criterion(self):
        """Swap the model's SetCriterion with DBHNormCriterion (in-place)."""
        crit = self.model.criterion
        # DBHNormCriterion is a subclass — copy all attributes, override loss_dbh
        crit.__class__ = DBHNormCriterion
        logger.info("Criterion replaced with DBHNormCriterion (per-species normalization).")

    def _freeze_non_dbh(self):
        """Freeze all parameters except dbh_embed."""
        frozen, active = 0, 0
        for name, param in self.model.named_parameters():
            if "dbh_embed" in name:
                param.requires_grad = True
                active += param.numel()
                logger.info(f"  [trainable] {name}  shape={tuple(param.shape)}")
            else:
                param.requires_grad = False
                frozen += param.numel()
        total = frozen + active
        logger.info(
            f"Frozen {frozen/1e6:.1f}M params, training {active/1e3:.1f}K params "
            f"({active/total*100:.3f}% of model)."
        )

    def resume_or_load(self, resume=True):
        """Load base checkpoint then apply freezing."""
        result = super().resume_or_load(resume=resume)
        # Freeze + replace criterion after weights are loaded
        self._freeze_non_dbh()
        self._replace_criterion()
        return result

    def build_writers(self):
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]


# ══════════════════════════════════════════════════════════════════════════════
# Setup & launch
# ══════════════════════════════════════════════════════════════════════════════
def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Override weights with the base segmentation checkpoint
    if args.model_weights:
        cfg.MODEL.WEIGHTS = args.model_weights

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # Register datasets
    combined_root = os.environ.get("COMBINED_ROOT",
        str(PROJECT_ROOT / "data" / "combined"))
    register_all_rainforests(os.environ.get("RAINFORESTS_ROOT",
        str(PROJECT_ROOT / "data" / "rainforests")))
    register_all_plantations(os.environ.get("PLANTATIONS_ROOT",
        str(PROJECT_ROOT / "data" / "plantations")))
    register_all_combined(combined_root)

    cfg = setup(args)
    trainer = DBHFineTuner(cfg)
    trainer.resume_or_load(resume=False)   # always fresh from base checkpoint

    logger.info("=== Starting DBH fine-tune ===")
    logger.info(f"  Base checkpoint : {cfg.MODEL.WEIGHTS}")
    logger.info(f"  Output dir      : {cfg.OUTPUT_DIR}")
    logger.info(f"  Max iter        : {cfg.SOLVER.MAX_ITER}")
    logger.info(f"  DBH_WEIGHT      : {cfg.MODEL.MaskDINO.DBH_WEIGHT}")
    logger.info(f"  Normalization   : per-species z-score in log-space")

    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FocalNet-L DBH fine-tune")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-weights", default="",
                        help="Path to FocalNetL-sc model_final.pth")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    setup_logger(name="fvcore")
    logger = setup_logger(name="finetune_focal_dbh")
    logger.info(f"Args: {args}")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
