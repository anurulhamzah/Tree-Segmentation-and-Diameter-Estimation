#!/usr/bin/env python3
"""
finetune_trunk_roi_dbh_hybrid_joint_7species.py — VARIAN dari
finetune_trunk_roi_dbh_hybrid_joint.py (universe11), HANYA beda spesies DBH: subset
7-spesies hasil sintesis literatur+data (lihat Report_DBH_Species_Selection.html §10) —
Loquat/Orange/AliiFig/BangaloPalm/Fern/RubberFig/Umbrella. Apple, Mango, Persimmon,
Pomegranate (di samping Lemon/LeechVine yg sudah exclude dari universe11) dikeluarkan
dari loss DBH (gt_dbh di-nol-kan) — TAPI dataset training tetap full 13-kelas utk
segmentasi, sama seperti skrip universe11 aslinya (lihat EXCLUDE_SPECIES_1BASED di bawah).

--- Dokumentasi asli (universe11), masih berlaku utk desain arsitektur joint ---

finetune_trunk_roi_dbh_hybrid_joint.py — JOINT multitask training: segmentasi MaskDINO
(mask/dice/class/box/giou AKTIF) + DBH Trunk-ROI Hybrid head (kombinasi PEMENANG DBH
Sandbox Fase C, R²=0.757 test/mean-per-species-R²=+0.136), backbone TIDAK dibekukan.

Latar belakang (lihat [[project-dbh-diameter-research]] Section 12-13 di memory):
  - Head DBH terbaik yang PERNAH ditemukan bukan dbh_embed lama (di dalam decoder,
    query_emb⊕class_probs → MLP kecil) — head itu GAGAL joint (SwinB, R²≈-0.004) dan
    GAGAL murni trunk-features (R²=-0.075). Head TERBAIK adalah HybridTrunkROIDBHHead
    (trunk_roi_dbh_head_hybrid.py): backbone res2 di baris geometris 1.3m + geometric
    prior + species one-hot → MLP, tapi SELALU dilatih dgn backbone DIBEKUKAN.
  - Eksperimen JOINT-vs-FROZEN 22 Jul (job 58470985/86) sudah membuktikan joint TIDAK
    merusak segmentasi drastis untuk head LAMA, tapi head LAMA arsitekturnya lemah
    (per-query, bukan trunk-localized) — belum menjawab "apakah head TERBAIK bisa jadi
    joint?". Skrip ini mengisi gap itu: head TERBAIK (Hybrid Trunk-ROI, hyperparameter
    sandbox pemenang) + backbone UNFROZEN + init dari model segmentasi TERBAIK
    (plain_ext20k, test AP50=62.08%).

Desain (mirip finetune_focal_dbh_joint.py tapi utk head EKSTERNAL, bukan built-in decoder):
  1. Head DBH TIDAK dimasukkan ke decoder — tetap modul terpisah (HybridTrunkROIDBHHead),
     dipasang sbg submodule baru pd instance model via `model.dbh_head_trunkroi = ...`
     (nn.Module.__setattr__ otomatis meregistrasi submodule, TANPA mengubah struktur
     checkpoint lama — kunci `backbone.*`/`sem_seg_head.*`/`criterion.*` tetap cocok
     persis dgn plain_ext20k/model_final.pth, hanya `dbh_head_trunkroi.*` yg baru/random).
  2. `model.backbone` di-hook (forward hook) utk menangkap res2 SEKALI per iterasi —
     res2 yg SAMA dipakai baik oleh segmentasi (via forward asli) MAUPUN oleh head DBH,
     jadi gradient DBH & gradient segmentasi mengalir lewat backbone yg SAMA (genuinely
     joint, bukan dua forward pass terpisah).
  3. `model.__class__` di-reassign ke subclass dinamis yg override `forward()`: panggil
     forward asli (dapat loss segmentasi + res2 ter-cache), lalu tambahkan
     `loss_dbh_trunkroi` ke dict loss SEBELUM dikembalikan — SimpleTrainer.run_step()
     bawaan detectron2 otomatis menjumlahkan SEMUA value di loss dict, jadi tidak perlu
     override run_step/optimizer.step manual (pola monkeypatch-class ini SAMA dgn yg
     sudah dipakai `_replace_criterion()` di finetune_focal_dbh.py — dikonfirmasi aman).
  4. dbh_embed BAWAAN (built-in decoder) DIMATIKAN via DBH_WEIGHT=0.0 di config (tetap
     forward, tapi weight 0 → tidak menyumbang gradient) — supaya tidak ada 2 head DBH
     yg saling berebut sinyal backbone.
  5. Filter species mengikuti kombinasi PEMENANG sandbox: universe11 (exclude Lemon
     cat_id=2, LeechVine cat_id=11 — blank-DBH >50%), RubberFig cap 150cm (exclude
     instance dgn dbh>=150cm, distribusi bimodal), ratio_bounds=off, min_trunk_px=5,
     depth∈[1,20)m, |world_height-1.3|<=0.5m, strip_rows=5, dropout=0.3, hidden=256,
     species_weight inverse-freq cap 20 (SAMA PERSIS dgn V7/leaderboard phaseC winner).
     Filter diterapkan PER-BATCH saat compute_loss (bukan pre-filter dataset) — dataset
     training TETAP full v4_stratified (semua 13 kelas, semua instance) supaya AP50 tidak
     terkorbankan oleh subset data DBH-eligible saja.

DBH_WEIGHT (multiplier eksternal, BUKAN cfg.MODEL.MaskDINO.DBH_WEIGHT) HARUS dikalibrasi
via smoke test sebelum submit job penuh — loss head ini SATU skalar (bukan deep-supervision
~11 komponen spt dbh_embed lama), jadi skala mentahnya beda jauh dari kalibrasi lama
(DBH_WEIGHT=1.0 lama ≈ 31% total loss). Jangan asumsikan angka yang sama berlaku di sini.

Jalankan:
  python scripts/finetune_trunk_roi_dbh_hybrid_joint.py \\
      --config-file configs/maskdino_FocalNet_L_trunk_roi_dbh_hybrid_joint_plainext_20k.yaml \\
      --model-weights /scratch2/pr65/anur0018/maskdino_output/FocalNet_L_combined_rgbd_scratch_v4_stratified_plain_ext20k/model_final.pth \\
      --dbh-weight <dikalibrasi> \\
      --num-gpus 1 OUTPUT_DIR <dir>
"""
import argparse
import copy
import itertools
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(HERE))

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, get_event_storage
from detectron2.utils.logger import setup_logger

from maskdino import add_maskdino_config
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import (
    COCOInstanceDBHDatasetMapper)

from register_combined import register_all_combined
from register_rainforests import register_all_rainforests
from register_plantations import register_all_plantations

from trunk_roi_dbh_head_hybrid import HybridTrunkROIDBHHead

logger = logging.getLogger("finetune_trunk_roi_dbh_hybrid_joint_7species")

# ── Kombinasi pemenang DBH Sandbox Fase C (leaderboard.csv, phaseC_..._seed0/1/2) ────
DBH_HEAD_LR = 1e-4
DBH_HEAD_WD = 5e-4
STRIP_ROWS = 5
DROPOUT = 0.3
HIDDEN = 256
IN_CHANNELS = 192  # FocalNet-L res2

# Subset 7-spesies (literatur+data, lihat §10 Report_DBH_Species_Selection.html):
# KEEP: Loquat=3, Orange=5, AliiFig=8, BangaloPalm=9, Fern=10, RubberFig=12, Umbrella=13
# EXCLUDE: Apple=1, Lemon=2, Mango=4, Persimmon=6, Pomegranate=7, LeechVine=11
# + RubberFig cap 150cm (sama seperti universe11).
EXCLUDE_SPECIES_1BASED = {1, 2, 4, 6, 7, 11}
RUBBERFIG_CAT_ID = 12
RUBBERFIG_CAP_CM = 150.0

# Per-species log1p(DBH_mm) stats + inverse-freq weights, dari dataset v3 universe11
# train (identik dgn scripts/finetune_trunk_roi_dbh_hybrid_v7_universe11.py).
CAT_LOG_MEAN = {
    1: 5.3710, 3: 4.6873, 4: 5.5160, 5: 5.2144, 6: 4.9466, 7: 5.3494,
    8: 6.2302, 9: 5.4537, 10: 6.1035, 12: 6.3724, 13: 5.9343,
}
CAT_LOG_STD = {
    1: 0.1327, 3: 0.2960, 4: 0.2803, 5: 0.1893, 6: 0.0843, 7: 0.2189,
    8: 0.2492, 9: 0.4202, 10: 0.2578, 12: 0.1407, 13: 0.2813,
}
SPECIES_WEIGHT = {
    1: 7.9, 3: 3.9, 4: 20.0, 5: 5.3, 6: 20.0, 7: 20.0,
    8: 4.7, 9: 1.0, 10: 3.9, 12: 14.7, 13: 3.6,
}
CAT_NAMES = {
    1: "Apple", 3: "Loquat", 4: "Mango", 5: "Orange", 6: "Persimmon", 7: "Pomegranate",
    8: "AliiFig", 9: "BangaloPalm", 10: "Fern", 12: "RubberFig", 13: "Umbrella",
}


def _prepare_dbh_targets(inst):
    """Clone gt_dbh dan nol-kan instance yang TIDAK eligible (species exclude / RubberFig
    cap) — HybridTrunkROIDBHHead.compute_loss() sudah skip dbh<=0 secara otomatis."""
    gt_dbh = inst.gt_dbh.clone()
    cat_ids = inst.gt_classes + 1
    excl = torch.zeros_like(gt_dbh, dtype=torch.bool)
    for e in EXCLUDE_SPECIES_1BASED:
        excl |= (cat_ids == e)
    excl |= (cat_ids == RUBBERFIG_CAT_ID) & (gt_dbh >= RUBBERFIG_CAP_CM)
    gt_dbh[excl] = 0.0
    inst.gt_dbh = gt_dbh
    return inst


def _make_joint_class(base_cls):
    """Subclass dinamis: forward() asli + loss_dbh_trunkroi ditambahkan ke dict loss.
    Sama pola dgn `crit.__class__ = DBHNormCriterion` di finetune_focal_dbh.py."""

    class _JointMaskDINOTrunkROI(base_cls):
        def forward(self, batched_inputs):
            losses = super().forward(batched_inputs)
            if not self.training:
                return losses

            device = next(self.parameters()).device
            res2 = self._res2_cache
            assert res2 is not None, "res2 hook tidak terpicu — cek registrasi forward hook"

            depths, instances = [], []
            for x in batched_inputs:
                depths.append(x["depth"].to(device))
                instances.append(_prepare_dbh_targets(x["instances"].to(device)))

            dbh_loss, n_clean, n_total = self.dbh_head_trunkroi.compute_loss(
                res2, depths, instances, CAT_LOG_MEAN, CAT_LOG_STD,
                min_trunk_px=5, min_depth=1.0, max_depth=20.0, max_wh_dev=0.5,
                min_ratio=0.0, max_ratio=float("inf"),
                species_weights=SPECIES_WEIGHT,
            )
            losses["loss_dbh_trunkroi"] = dbh_loss * self.dbh_weight

            try:
                storage = get_event_storage()
                storage.put_scalar("dbh_trunkroi/n_clean", float(n_clean), smoothing_hint=False)
                storage.put_scalar("dbh_trunkroi/n_total", float(n_total), smoothing_hint=False)
                storage.put_scalar("dbh_trunkroi/raw_loss", float(dbh_loss.item()), smoothing_hint=False)
            except AssertionError:
                pass  # no active EventStorage (e.g. smoke test outside trainer loop)

            return losses

    return _JointMaskDINOTrunkROI


class JointTrunkROITrainer(DefaultTrainer):

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        device = next(model.parameters()).device

        dbh_head = HybridTrunkROIDBHHead(
            in_channels=IN_CHANNELS, hidden=HIDDEN, strip_rows=STRIP_ROWS,
            num_species=13, dropout=DROPOUT,
        ).to(device)
        n_params = sum(p.numel() for p in dbh_head.parameters())
        logger.info(f"HybridTrunkROIDBHHead (JOINT, backbone unfrozen): {n_params:,} params, "
                    f"strip_rows={STRIP_ROWS} dropout={DROPOUT} hidden={HIDDEN}")

        model.dbh_head_trunkroi = dbh_head          # registrasi submodule otomatis
        model.dbh_weight = cfg.DBH_JOINT_WEIGHT      # lihat add ke cfg custom key di setup()
        model._res2_cache = None

        def _hook(module, inp, out):
            model._res2_cache = out["res2"]
        model.backbone.register_forward_hook(_hook)

        model.__class__ = _make_joint_class(model.__class__)
        logger.info(f"Model class reassigned -> {model.__class__.__name__} "
                    f"(dbh_weight={model.dbh_weight})")
        return model

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
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "dbh_head_trunkroi" in module_name:
                    hyperparams["lr"] = DBH_HEAD_LR
                    hyperparams["weight_decay"] = DBH_HEAD_WD
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
        logger.info(f"JOINT optimizer groups: dbh_head={n_head/1e3:.1f}K @lr={DBH_HEAD_LR}, "
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

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = COCOInstanceDBHDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = COCOInstanceDBHDatasetMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def resume_or_load(self, resume=False):
        """Load base checkpoint (backbone/decoder/criterion) — dbh_head_trunkroi TIDAK
        ada di checkpoint lama, tetap random-init (missing-key warning normal/expected,
        pola sama spt dbh_embed pertama kali dimuat dari checkpoint non-DBH)."""
        result = DefaultTrainer.resume_or_load(self, resume=resume)
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"JOINT mode: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params trainable "
                    f"({n_train/n_total*100:.1f}%) — backbone TIDAK dibekukan.")
        return result

    def build_writers(self):
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.set_new_allowed(True)
    cfg.DBH_JOINT_WEIGHT = args.dbh_weight  # custom top-level key, dibaca build_model()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.model_weights:
        cfg.MODEL.WEIGHTS = args.model_weights
    cfg.DBH_JOINT_WEIGHT = args.dbh_weight  # opts tidak boleh menimpa CLI --dbh-weight
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    combined_root = os.environ.get("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_rainforests(os.environ.get("RAINFORESTS_ROOT", str(PROJECT_ROOT / "data" / "rainforests")))
    register_all_plantations(os.environ.get("PLANTATIONS_ROOT", str(PROJECT_ROOT / "data" / "plantations")))
    register_all_combined(combined_root)

    cfg = setup(args)
    assert cfg.MODEL.MaskDINO.MASK_WEIGHT > 0, "Loss segmentasi harus AKTIF (joint)!"
    assert cfg.MODEL.MaskDINO.DBH_WEIGHT == 0.0, \
        "dbh_embed BAWAAN harus DBH_WEIGHT=0.0 (head eksternal dipakai, hindari 2 head bentrok)"

    trainer = JointTrunkROITrainer(cfg)
    trainer.resume_or_load(resume=False)

    logger.info("=== Starting JOINT (seg + Trunk-ROI DBH) fine-tune, backbone UNFROZEN ===")
    logger.info(f"  Base checkpoint : {cfg.MODEL.WEIGHTS}")
    logger.info(f"  Output dir      : {cfg.OUTPUT_DIR}")
    logger.info(f"  DBH_JOINT_WEIGHT: {cfg.DBH_JOINT_WEIGHT}")
    logger.info(f"  Seg weights     : mask={cfg.MODEL.MaskDINO.MASK_WEIGHT} "
                f"dice={cfg.MODEL.MaskDINO.DICE_WEIGHT} class={cfg.MODEL.MaskDINO.CLASS_WEIGHT} "
                f"box={cfg.MODEL.MaskDINO.BOX_WEIGHT} giou={cfg.MODEL.MaskDINO.GIOU_WEIGHT}")
    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FocalNet-L JOINT seg+TrunkROI-DBH fine-tune")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-weights", default="")
    parser.add_argument("--dbh-weight", type=float, required=True,
                         help="Multiplier eksternal utk loss_dbh_trunkroi (kalibrasi via smoke test)")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger(name="finetune_trunk_roi_dbh_hybrid_joint_7species")
    logger.info(f"Args: {args}")
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
