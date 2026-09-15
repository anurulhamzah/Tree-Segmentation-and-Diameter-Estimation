#!/usr/bin/env python3
"""
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

logger = logging.getLogger("finetune_trunk_roi_dbh_hybrid_joint")

# ── Kombinasi pemenang DBH Sandbox Fase C (leaderboard.csv, phaseC_..._seed0/1/2) ────
DBH_HEAD_LR = 1e-4
DBH_HEAD_WD = 5e-4
STRIP_ROWS = 5
DROPOUT = 0.3
HIDDEN = 256
IN_CHANNELS = 192  # FocalNet-L res2 (default; override via --dbh-head-in-channels utk backbone lain)

# Species universe11 (exclude Lemon=2, LeechVine=11) + RubberFig cap 150cm.
# Preset subset spesies untuk loss DBH. Dataset training tetap 13 kelas penuh untuk
# segmentasi; yang dikecualikan hanya kontribusinya ke loss DBH (gt_dbh di-nol-kan).
# Menyatukan kedua preset di SATU skrip supaya perbaikan (mis. jadwal LR head, max_depth)
# tidak perlu disalin ke varian terpisah lalu berisiko melenceng.
SPECIES_PRESET = {
    # universe11: buang Lemon=2 dan LeechVine=11 (blank-DBH di atas 50%)
    "universe11": {2, 11},
    # sevenspecies: buang juga Apple=1, Mango=4, Persimmon=6, Pomegranate=7
    # (konvensi pengukuran / N test di bawah 30). Lihat Report_DBH_Species_Selection.html
    "sevenspecies": {1, 2, 4, 6, 7, 11},
}
EXCLUDE_SPECIES_1BASED = SPECIES_PRESET["universe11"]   # ditimpa oleh --species-preset
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
        def _dbh_weight_now(self):
            """Loss weight DBH pada iterasi ini.

            Weight tetap (dbh_weight_end None) mengembalikan nilai konstan, sehingga seluruh
            run lama berperilaku persis sama. Jadwal naik dipakai untuk menguji usulan
            pembimbing: biarkan segmentasi terbentuk dengan weight rendah, lalu naikkan
            weight DBH.

            Ramp WAJIB selesai sebelum LR decay pertama. Sesudah `STEPS` pertama, LR
            tinggal sepersepuluh dan backbone nyaris tidak bergerak, sehingga kenaikan weight
            di sana akan terbaca sebagai "tidak berpengaruh" padahal yang habis adalah LR.
            """
            if self.dbh_weight_end is None:
                return self.dbh_weight
            try:
                it = get_event_storage().iter
            except Exception:
                # Di luar konteks training loop (mis. smoke test manual) pakai nilai awal.
                return self.dbh_weight
            a, b = self.dbh_ramp_start, self.dbh_ramp_end
            if it <= a:
                return self.dbh_weight
            if it >= b:
                return self.dbh_weight_end
            f = (it - a) / float(b - a)
            return self.dbh_weight + f * (self.dbh_weight_end - self.dbh_weight)

        def forward(self, batched_inputs):
            losses = super().forward(batched_inputs)
            if not self.training:
                return losses

            device = next(self.parameters()).device
            res2 = self._res2_cache
            assert res2 is not None, "res2 hook tidak terpicu — cek registrasi forward hook"
            if self.dbh_detach_backbone:
                # Putus gradien loss diameter ke backbone: head tetap membaca res2 yang
                # sama (jadi tetap satu forward pass, satu model), tapi tidak lagi ikut
                # membentuk feature bersama. Menguji apakah interferensi searah (§4.1: diameter
                # dirugikan, segmentasi tidak diuntungkan) hilang kalau jalur gradiennya diputus.
                res2 = res2.detach()

            depths, instances = [], []
            for x in batched_inputs:
                depths.append(x["depth"].to(device))
                instances.append(_prepare_dbh_targets(x["instances"].to(device)))

            dbh_loss, n_clean, n_total = self.dbh_head_trunkroi.compute_loss(
                res2, depths, instances, CAT_LOG_MEAN, CAT_LOG_STD,
                min_trunk_px=5, min_depth=1.0,
                max_depth=self.dbh_max_depth, max_wh_dev=0.5,
                min_ratio=0.0, max_ratio=float("inf"),
                species_weights=SPECIES_WEIGHT,
            )
            w = self._dbh_weight_now()
            losses["loss_dbh_trunkroi"] = dbh_loss * w

            try:
                storage = get_event_storage()
                storage.put_scalar("dbh_trunkroi/n_clean", float(n_clean), smoothing_hint=False)
                storage.put_scalar("dbh_trunkroi/n_total", float(n_total), smoothing_hint=False)
                storage.put_scalar("dbh_trunkroi/raw_loss", float(dbh_loss.item()), smoothing_hint=False)
                # Weight ikut dicatat supaya jadwalnya bisa diverifikasi dari metrics.json
                # tanpa menebak dari log teks, dan supaya kurvanya bisa digambar.
                storage.put_scalar("dbh_trunkroi/weight", float(w), smoothing_hint=False)
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
            in_channels=cfg.DBH_HEAD_IN_CHANNELS, hidden=HIDDEN, strip_rows=STRIP_ROWS,
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
        model.dbh_max_depth = float(cfg.DBH_MAX_DEPTH)
        model.dbh_weight_end = cfg.DBH_WEIGHT_END          # None = weight tetap
        model.dbh_ramp_start = int(cfg.DBH_RAMP_START)
        model.dbh_ramp_end = int(cfg.DBH_RAMP_END)
        model.dbh_detach_backbone = bool(cfg.DBH_DETACH_BACKBONE)
        jadwal = ("tetap" if model.dbh_weight_end is None else
                  f"{model.dbh_weight} -> {model.dbh_weight_end} "
                  f"linear iter {model.dbh_ramp_start}-{model.dbh_ramp_end}")
        logger.info(f"Model class reassigned -> {model.__class__.__name__} "
                    f"(dbh_weight={model.dbh_weight}, jadwal={jadwal}, "
                    f"dbh_max_depth={model.dbh_max_depth}m, "
                    f"dbh_detach_backbone={model.dbh_detach_backbone})")
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
                    hyperparams["is_dbh_head"] = True   # dibaca build_lr_scheduler()
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
    def build_lr_scheduler(cls, cfg, optimizer):
        """Opsi mengecualikan head DBH dari decay multi-step. MATI secara baku.

        MEKANISMENYA. `build_optimizer` di atas sudah memberi head grup sendiri
        (DBH_HEAD_LR=1e-4), tetapi detectron2 mengembalikan LRMultiplier yang memakai
        SATU multiplier skalar untuk SELURUH grup. Akibatnya gamma=0,1 tiap milestone
        ikut mengenai head, sehingga pada jadwal 155k LR head jatuh ke 1e-6 dan head
        praktis membeku pada ~10.333 iterasi terakhir. Jalur di bawah memberi grup head
        multiplier warmup-saja (tanpa milestone), grup lain tetap memakai jadwal penuh.

        HIPOTESISNYA SUDAH DIUJI DAN DITOLAK (3 Agt 2026). Dugaan semula: pembekuan itu
        yang membuat keempat run joint memuncak tepat sebelum decay terakhir lalu
        memburuk 2,26-2,54mm RMSE. Ujinya lewat cabang dari model_0119999.pth run
        from-scratch 155k, satu variabel berubah, dan log `[LR cek]` memastikan head
        memang bertahan di 1e-4 sementara grup lain turun ke 1e-6. Hasilnya kebalikan
        dari prediksi yang dinyatakan di muka. Test RMSE, asli lawan cabang:

            124.999  79,91 / 84,14      139.999  80,05 / 81,83
            129.999  81,83 / 87,10      144.999  80,37 / 81,11
            134.999  78,19 / 82,82      149.999  80,43 / 81,87

        Cabang kalah di 7 dari 7 titik test dan 6 dari 7 di val; tiga selisih teratas
        jauh di luar noise band 0,58mm. Pola "memuncak lalu turun" pun tetap muncul DI
        DALAM cabang (puncak 144.999, turun di 149.999), jadi decay LR head bukan
        penyebabnya. Segmentasi tidak berubah pada kedua sisi (AP50 64,02 lawan 63,88
        di 149.999), sesuai dugaan bahwa koplingnya lemah.

        Tafsiran sekarang: saat backbone sudah membeku di 1e-6, head ber-LR tinggi
        mengejar representasi yang sudah statis dan itu merugikan. Decay-nya menolong.
        Karena itu defaultnya dikembalikan ke perilaku lama; jalur pengecualian
        dipertahankan hanya agar cabang LRFIX tetap bisa direproduksi lewat
        --freeze-dbh-head-lr.
        """
        from detectron2.solver.lr_scheduler import LRMultiplier
        from fvcore.common.param_scheduler import (ConstantParamScheduler,
                                                   MultiStepParamScheduler,
                                                   ParamScheduler)
        from detectron2.solver.build import WarmupParamScheduler

        head_idx = {i for i, g in enumerate(optimizer.param_groups) if g.get("is_dbh_head")}
        if not head_idx or not cfg.SOLVER.FREEZE_DBH_HEAD_LR_DECAY:
            logger.info("LR scheduler: jadwal baku dipakai untuk SEMUA grup "
                        f"(head_groups={len(head_idx)}, "
                        f"FREEZE_DBH_HEAD_LR_DECAY={cfg.SOLVER.FREEZE_DBH_HEAD_LR_DECAY})")
            return super().build_lr_scheduler(cfg, optimizer)

        steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
        warm = lambda base: WarmupParamScheduler(
            base, cfg.SOLVER.WARMUP_FACTOR,
            min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
            cfg.SOLVER.WARMUP_METHOD, cfg.SOLVER.RESCALE_INTERVAL)
        full = warm(MultiStepParamScheduler(
            values=[cfg.SOLVER.GAMMA ** k for k in range(len(steps) + 1)],
            milestones=steps, num_updates=cfg.SOLVER.MAX_ITER))
        head_only = warm(ConstantParamScheduler(1.0))

        class _PerGroupLRMultiplier(LRMultiplier):
            """head_idx sengaja ditangkap sebagai SET INDEKS di closure, bukan dibaca ulang
            dari param_groups saat get_lr. Alasannya: `optimizer.load_state_dict()` pada saat
            resume menimpa isi param_groups dengan versi tersimpan, sehingga kunci
            `is_dbh_head` bisa hilang. Indeks grup tidak berubah, jadi cara ini tahan resume
            (diverifikasi 2 Agt 2026 dengan mensimulasikan load_state_dict)."""

            _log_every = 5000

            def get_lr(self):
                # Jepit tepat di bawah 1,0. Pada langkah TERAKHIR, last_epoch == max_iter
                # sehingga t = 1,0 persis. fvcore tidak konsisten di titik itu:
                # MultiStepParamScheduler memakai `if where > 1.0` (menerima 1,0) sedangkan
                # ConstantParamScheduler memakai `if where >= 1.0` (MENOLAK 1,0). Karena
                # jadwal head memakai Constant, run 2-3 Agt 2026 menyelesaikan seluruh
                # iterasi lalu crash di langkah scheduler terakhir, sehingga checkpoint
                # terakhir dan model_final TIDAK tersimpan. Jepitan ini tidak mengubah
                # nilai LR: pada MultiStep, t=1-1e-9 jatuh di bucket milestone yang sama.
                t = min(self.last_epoch / self._max_iter, 1.0 - 1e-9)
                m_full, m_head = full(t), head_only(t)
                lrs = [base * (m_head if i in head_idx else m_full)
                       for i, base in enumerate(self.base_lrs)]
                # Bukti berkala: detectron2 hanya mencetak SATU grup (yang paramnya
                # terbanyak), jadi tanpa baris ini LR head tidak pernah terlihat di log.
                if self.last_epoch % self._log_every == 0:
                    hd = next((lrs[i] for i in sorted(head_idx)), float("nan"))
                    ot = next((lrs[i] for i in range(len(lrs)) if i not in head_idx), float("nan"))
                    logger.info(f"[LR cek] iter={self.last_epoch}  head_dbh={hd:.2e}  "
                                f"grup_lain={ot:.2e}")
                return lrs

        logger.info(f"LR scheduler: {len(head_idx)} grup head DBH DIKECUALIKAN dari decay "
                    f"(tetap {DBH_HEAD_LR:.0e} setelah warmup); "
                    f"{len(optimizer.param_groups)-len(head_idx)} grup lain memakai "
                    f"WarmupMultiStepLR milestones={steps} gamma={cfg.SOLVER.GAMMA}")
        return _PerGroupLRMultiplier(optimizer, multiplier=full, max_iter=cfg.SOLVER.MAX_ITER)

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
    global EXCLUDE_SPECIES_1BASED
    EXCLUDE_SPECIES_1BASED = SPECIES_PRESET[args.species_preset]
    cfg.DBH_MAX_DEPTH = args.max_depth       # batas jarak instance yg dipakai loss DBH
    cfg.DBH_WEIGHT_END = args.dbh_weight_end   # None = weight tetap, perilaku run lama
    cfg.DBH_RAMP_START = args.dbh_ramp_start
    cfg.DBH_RAMP_END = args.dbh_ramp_end
    cfg.SOLVER.FREEZE_DBH_HEAD_LR_DECAY = args.freeze_dbh_head_lr
    cfg.DBH_DETACH_BACKBONE = args.detach_dbh_backbone
    cfg.DBH_HEAD_IN_CHANNELS = args.dbh_head_in_channels
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.model_weights:
        cfg.MODEL.WEIGHTS = args.model_weights
    cfg.DBH_JOINT_WEIGHT = args.dbh_weight  # opts tidak boleh menimpa CLI --dbh-weight
    cfg.DBH_MAX_DEPTH = args.max_depth
    cfg.DBH_WEIGHT_END = args.dbh_weight_end
    cfg.DBH_RAMP_START = args.dbh_ramp_start
    cfg.DBH_RAMP_END = args.dbh_ramp_end
    cfg.SOLVER.FREEZE_DBH_HEAD_LR_DECAY = args.freeze_dbh_head_lr
    cfg.DBH_DETACH_BACKBONE = args.detach_dbh_backbone
    cfg.DBH_HEAD_IN_CHANNELS = args.dbh_head_in_channels

    # Validasi jadwal DBH_WEIGHT SEBELUM training dimulai. Run ini memakan berhari-hari, dan
    # konfigurasi yang salah baru ketahuan setelah selesai kalau tidak dicek di sini.
    if args.dbh_weight_end is not None:
        if args.dbh_ramp_end <= args.dbh_ramp_start:
            raise SystemExit(f"--dbh-ramp-end ({args.dbh_ramp_end}) harus > "
                             f"--dbh-ramp-start ({args.dbh_ramp_start})")
        steps = list(cfg.SOLVER.STEPS) or [cfg.SOLVER.MAX_ITER]
        if args.dbh_ramp_end > steps[0]:
            raise SystemExit(
                f"--dbh-ramp-end ({args.dbh_ramp_end}) melewati LR decay pertama "
                f"(STEPS[0]={steps[0]}). Sesudah titik itu LR tinggal sepersepuluh dan "
                f"backbone nyaris tidak bergerak, sehingga kenaikan weight tidak akan "
                f"terbaca sebagai efek weight melainkan sebagai LR yang habis. "
                f"Selesaikan ramp sebelum {steps[0]}.")
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
    # resume=True dipakai untuk MENCABANG dari checkpoint tertentu: salin checkpoint ke
    # OUTPUT_DIR lalu tulis namanya ke berkas `last_checkpoint`, sehingga iteration dan
    # optimizer state ikut dipulihkan dan training melanjutkan, bukan mengulang dari 0.
    trainer.resume_or_load(resume=args.resume)

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
                         help="Multiplier eksternal utk loss_dbh_trunkroi (kalibrasi via smoke test). "
                              "Kalau --dbh-weight-end diisi, nilai ini menjadi weights AWAL.")
    parser.add_argument("--dbh-weight-end", type=float, default=None,
                        help="Weight akhir. Kosong = tetap sepanjang training, yaitu perilaku "
                             "seluruh run sebelum 9 Agt 2026. Diisi = naik linear dari "
                             "--dbh-weight ke nilai ini antara --dbh-ramp-start dan --dbh-ramp-end.")
    parser.add_argument("--dbh-ramp-start", type=int, default=0,
                        help="Iterasi awal ramp; sebelumnya weight tetap di --dbh-weight")
    parser.add_argument("--dbh-ramp-end", type=int, default=0,
                        help="Iterasi akhir ramp; sesudahnya weight tetap di --dbh-weight-end")
    parser.add_argument("--species-preset", default="universe11",
                        choices=sorted(SPECIES_PRESET),
                        help="Subset spesies yang menyumbang loss DBH. Dataset segmentasi "
                             "tetap 13 kelas penuh pada kedua preset.")
    parser.add_argument("--max-depth", type=float, default=20.0,
                        # % WAJIB ditulis %%: argparse memformat help string dengan operator %,
                        # dan satu % telanjang membuat --help melempar ValueError.
                        help="Batas jarak kamera-ke-pohon (m) untuk loss DBH. Dulu hardcoded "
                             "20,0 dan tidak pernah divariasikan di satu pun job joint. Uji "
                             "langsung di joint (3 Agt 2026, dua fine-tune 20k identik kecuali "
                             "argumen ini) memberi hasil SERI: selisih 0,37mm val dan 0,21mm "
                             "test, di bawah noise dan berlawanan arah. Dipilih 30,0 atas dasar "
                             "cakupan, 99,0%% distribusi depth lawan 94,1%%.")
    parser.add_argument("--freeze-dbh-head-lr", action="store_true",
                        help="Kecualikan LR head DBH dari decay multi-step. MATI secara baku: "
                             "uji cabang 3 Agt 2026 menolak hipotesis ini, cabang kalah di 7 "
                             "dari 7 titik test (sampai 5,27mm RMSE). Dipertahankan hanya agar "
                             "cabang LRFIX bisa direproduksi. Rincian di docstring "
                             "build_lr_scheduler.")
    parser.add_argument("--detach-dbh-backbone", action="store_true",
                        help="Putus gradien loss_dbh_trunkroi sebelum res2 (stop-gradient). "
                             "Head tetap membaca feature backbone yang sama (satu forward pass, "
                             "satu checkpoint), tapi tidak lagi ikut membentuknya. Diuji krn "
                             "§4.1 tesis membuktikan interferensi searah: diameter dirugikan "
                             "oleh training segmentasi berkelanjutan, sementara segmentasi tidak "
                             "diuntungkan sama sekali oleh loss diameter (costs nothing "
                             "measurable) -- jadi memutus gradien diameter seharusnya tidak "
                             "merugikan segmentasi. MATI secara baku (perilaku run lama).")
    parser.add_argument("--dbh-head-in-channels", type=int, default=IN_CHANNELS,
                        help="Channel res2 backbone yang dibaca HybridTrunkROIDBHHead. Baku "
                             "192 (FocalNet-L). WAJIB diisi manual utk backbone lain -- FocalNet-B "
                             "res2=128 (feat_dim head=143), berbeda dari FocalNet-L (feat_dim=207). "
                             "Salah isi -> RuntimeError shape mismatch di LayerNorm pertama.")
    parser.add_argument("--resume", action="store_true",
                        help="Lanjutkan dari `last_checkpoint` di OUTPUT_DIR (memulihkan\n"
                             "iterasi + optimizer state). Untuk fresh start, JANGAN dipakai.")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--dist-url", default="auto")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger(name="finetune_trunk_roi_dbh_hybrid_joint")
    logger.info(f"Args: {args}")
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
