"""
finetune_focal_dbh_joint.py — JOINT multi-task fine-tune (segmentasi + DBH, backbone TIDAK dibekukan).

Kontras dengan finetune_focal_dbh.py (frozen): di sana backbone+decoder dibekukan dan hanya
dbh_embed dilatih (DBH_WEIGHT=30, loss segmentasi dimatikan). Di sini SELURUH model dilatih:
loss segmentasi (mask/dice/class/box/giou) DAN loss DBH aktif bersama, sehingga feature backbone
dapat dibentuk oleh sinyal DBH juga. Tujuan: menguji apakah joint training mengalahkan pendekatan
frozen/terpisah untuk estimasi DBH — dengan HEAD & KRITERION IDENTIK (DBHNormCriterion, per-species
z-score) sehingga satu-satunya variabel yang berbeda adalah freeze vs tidak.

Pelajaran dari kegagalan joint SwinB-DBH (R2~=0) yang dihindari di sini:
  1. Init dari checkpoint SEGMENTASI TERBAIK (plain_ext20k), bukan dari backbone pretrained/scratch
     — backbone sudah kompeten di segmentasi, tinggal diadaptasi; jauh lebih stabil.
  2. Loss DBH di-z-score (DBHNormCriterion) → skala ~O(1), sebanding dengan loss segmentasi ternormalisasi,
     jadi DBH_WEIGHT tidak perlu ekstrem. loss_dbh>0 di-assert (bug lama: register drop dbh → loss_dbh=0).
  3. dbh_embed (mulai dari weight ~nol) diberi LR lebih tinggi daripada backbone, agar head cepat belajar
     TANPA memaksa backbone bergerak agresif (lindungi segmentasi).
  4. AMP off + grad clip (pelajaran NaN scratch RGBD).

Jalankan:
  python scripts/finetune_focal_dbh_joint.py \\
      --config-file configs/maskdino_FocalNet_L_combined_rgbd_dbh_joint_plainext_20k.yaml \\
      --model-weights /scratch2/pr65/anur0018/maskdino_output/FocalNet_L_combined_rgbd_scratch_v4_stratified_plain_ext20k/model_final.pth \\
      --num-gpus 1 OUTPUT_DIR <dir>
"""
import argparse
import copy
import itertools
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse SEMUA komponen dari skrip frozen: DBHNormCriterion, DBHFineTuner, setup(), registrasi.
import finetune_focal_dbh as base
from finetune_focal_dbh import DBHFineTuner, setup
from detectron2.engine import launch
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("finetune_focal_dbh_joint")

# LR khusus untuk head DBH (dbh_embed) — mulai dari weight ~nol, perlu belajar lebih cepat
# daripada backbone yang hanya perlu adaptasi halus. Backbone/rest pakai SOLVER.BASE_LR.
DBH_HEAD_LR = 1e-3


class JointDBHTrainer(DBHFineTuner):
    """Sama seperti DBHFineTuner TAPI: (a) TIDAK membekukan apa pun (joint), (b) dbh_embed diberi
    LR lebih tinggi, (c) tetap memakai DBHNormCriterion (z-score per-species)."""

    @classmethod
    def build_optimizer(cls, cfg, model):
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
        n_head, n_backbone, n_rest = 0, 0, 0
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "dbh_embed" in module_name:
                    hyperparams["lr"] = DBH_HEAD_LR   # head baru → LR tinggi
                    hyperparams["weight_decay"] = 0.0
                    n_head += value.numel()
                elif "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    n_backbone += value.numel()
                else:
                    n_rest += value.numel()
                if "relative_position_bias_table" in module_param_name or \
                        "absolute_pos_embed" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
        logger.info(f"JOINT optimizer param groups: dbh_embed={n_head/1e3:.1f}K @lr={DBH_HEAD_LR}, "
                    f"backbone={n_backbone/1e6:.1f}M @lr={cfg.SOLVER.BASE_LR*cfg.SOLVER.BACKBONE_MULTIPLIER:.1e}, "
                    f"rest={n_rest/1e6:.1f}M @lr={cfg.SOLVER.BASE_LR:.1e}")

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                      and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                      and clip_norm_val > 0.0)

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        if cfg.SOLVER.OPTIMIZER == "ADAMW":
            return maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
        elif cfg.SOLVER.OPTIMIZER == "SGD":
            return maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        raise NotImplementedError(cfg.SOLVER.OPTIMIZER)

    def resume_or_load(self, resume=True):
        """Load base checkpoint, ganti criterion ke DBHNormCriterion, TAPI JANGAN freeze apa pun."""
        # Panggil DefaultTrainer.resume_or_load (lewati DBHFineTuner yang membekukan)
        from detectron2.engine import DefaultTrainer
        result = DefaultTrainer.resume_or_load(self, resume=resume)
        self._replace_criterion()   # DBHNormCriterion (z-score) — sama dengan frozen
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"JOINT mode: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params trainable "
                    f"({n_train/n_total*100:.1f}%) — TIDAK ADA yang dibekukan.")
        return result


def main(args):
    # registrasi dataset — reuse fungsi yang sudah diimpor di modul base
    combined_root = os.environ.get("COMBINED_ROOT", str(HERE.parent / "data" / "combined"))
    base.register_all_rainforests(os.environ.get("RAINFORESTS_ROOT", str(HERE.parent / "data" / "rainforests")))
    base.register_all_plantations(os.environ.get("PLANTATIONS_ROOT", str(HERE.parent / "data" / "plantations")))
    base.register_all_combined(combined_root)

    cfg = setup(args)
    assert cfg.MODEL.MaskDINO.DBH_WEIGHT > 0, "DBH_WEIGHT harus > 0 untuk joint training!"
    assert cfg.MODEL.MaskDINO.MASK_WEIGHT > 0, "Loss segmentasi harus AKTIF untuk joint (MASK_WEIGHT>0)!"
    trainer = JointDBHTrainer(cfg)
    trainer.resume_or_load(resume=False)
    logger.info("=== Starting JOINT (seg+DBH) fine-tune, backbone UNFROZEN ===")
    logger.info(f"  Base checkpoint : {cfg.MODEL.WEIGHTS}")
    logger.info(f"  Output dir      : {cfg.OUTPUT_DIR}")
    logger.info(f"  DBH_WEIGHT      : {cfg.MODEL.MaskDINO.DBH_WEIGHT}  "
                f"(seg: mask={cfg.MODEL.MaskDINO.MASK_WEIGHT} dice={cfg.MODEL.MaskDINO.DICE_WEIGHT} "
                f"class={cfg.MODEL.MaskDINO.CLASS_WEIGHT})")
    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FocalNet-L JOINT (seg+DBH) fine-tune")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-weights", default="")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger(name="finetune_focal_dbh_joint")
    logger.info(f"Args: {args}")
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
