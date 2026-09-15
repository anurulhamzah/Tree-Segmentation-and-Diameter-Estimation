"""DBHROIHeads — Mask R-CNN StandardROIHeads + head regresi DBH per-ROI.

Menambah satu head regresi (paralel dgn box & mask head) yang memprediksi log1p(dbh_mm)
per proposal/instance. Semantik DBH meniru MaskDINO:
  - target  = log1p(gt_dbh_mm)
  - gt_dbh == 0 (vine/tanpa trunk) → di-skip dari loss
  - prediksi mentah = log1p(dbh); inference dikonversi balik dgn expm1 → mm

Isolasi: hanya impor detectron2; tidak menyentuh repo MaskDINO.
"""
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals

__all__ = ["DBHROIHeads", "DBHHead"]


class DBHHead(nn.Module):
    """FC kecil: feature ROI ter-pool → skalar log1p(dbh_mm)."""

    def __init__(self, input_shape: ShapeSpec, num_fc: int = 2, fc_dim: int = 1024):
        super().__init__()
        in_dim = input_shape.channels * input_shape.height * input_shape.width
        self.fcs = nn.ModuleList()
        for _ in range(num_fc):
            self.fcs.append(nn.Linear(in_dim, fc_dim))
            in_dim = fc_dim
        self.predictor = nn.Linear(in_dim, 1)
        for layer in self.fcs:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.constant_(layer.bias, 0.0)
        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.constant_(self.predictor.bias, 0.0)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for fc in self.fcs:
            x = F.relu(fc(x))
        return self.predictor(x).squeeze(-1)   # (N,)


@ROI_HEADS_REGISTRY.register()
class DBHROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, *, dbh_in_features, dbh_pooler, dbh_head, dbh_loss_weight,
                 dbh_detach_backbone=False, **kwargs):
        super().__init__(**kwargs)
        self.dbh_in_features = dbh_in_features
        self.dbh_pooler = dbh_pooler
        self.dbh_head = dbh_head
        self.dbh_loss_weight = dbh_loss_weight
        self.dbh_detach_backbone = dbh_detach_backbone

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_res = cfg.MODEL.ROI_DBH_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        in_channels = [input_shape[f].channels for f in in_features][0]
        ret["dbh_in_features"] = in_features
        ret["dbh_pooler"] = ROIPooler(
            output_size=pooler_res,
            scales=pooler_scales,
            sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
            pooler_type=cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE,
        )
        ret["dbh_head"] = DBHHead(
            ShapeSpec(channels=in_channels, height=pooler_res, width=pooler_res),
            num_fc=cfg.MODEL.ROI_DBH_HEAD.NUM_FC,
            fc_dim=cfg.MODEL.ROI_DBH_HEAD.FC_DIM,
        )
        ret["dbh_loss_weight"] = cfg.MODEL.ROI_DBH_HEAD.LOSS_WEIGHT
        ret["dbh_detach_backbone"] = cfg.MODEL.ROI_DBH_HEAD.DETACH_BACKBONE
        return ret

    def forward(self, images, features, proposals, targets=None):
        # super() melakukan label_and_sample_proposals → mengembalikan proposals tersample
        # yang sudah membawa gt_dbh (field gt_* otomatis disalin ke proposal).
        instances, losses = super().forward(images, features, proposals, targets)
        if self.training:
            losses.update(self._forward_dbh(features, instances))
            return instances, losses
        instances = self._forward_dbh(features, instances)
        return instances, {}

    def _forward_dbh(self, features, instances):
        feats = [features[f] for f in self.dbh_in_features]
        if self.dbh_detach_backbone:
            # Stop-gradient loss_dbh sebelum backbone/FPN (meniru --detach-dbh-backbone
            # pipeline FocalNet-L/Swin-L). Backbone tetap dilatih penuh oleh loss
            # segmentasi/box/mask; hanya gradien loss_dbh yang diputus di sini.
            feats = [f.detach() for f in feats]

        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            boxes = [x.proposal_boxes for x in proposals]
            n = sum(len(b) for b in boxes)
            if n == 0:
                return {"loss_dbh": self.dbh_head.predictor.weight.sum() * 0.0}
            pred = self.dbh_head(self.dbh_pooler(feats, boxes))          # (N,)
            gt = cat([p.gt_dbh for p in proposals])                      # (N,) mm
            valid = gt > 0                                               # skip vine (dbh=0)
            if valid.sum() == 0:
                return {"loss_dbh": pred.sum() * 0.0}
            loss = F.smooth_l1_loss(pred[valid], torch.log1p(gt[valid]), reduction="mean")
            return {"loss_dbh": loss * self.dbh_loss_weight}

        # inference
        boxes = [x.pred_boxes for x in instances]
        if sum(len(b) for b in boxes) == 0:
            for inst in instances:
                inst.pred_dbh = inst.pred_boxes.tensor.new_zeros((0,))
            return instances
        pred = self.dbh_head(self.dbh_pooler(feats, boxes))
        dbh_mm = torch.expm1(pred).clamp_(min=0.0)                       # balik ke mm
        for inst, d in zip(instances, dbh_mm.split([len(b) for b in boxes])):
            inst.pred_dbh = d
        return instances
