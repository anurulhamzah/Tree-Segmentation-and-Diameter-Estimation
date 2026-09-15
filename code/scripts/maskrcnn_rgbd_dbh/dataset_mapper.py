"""DatasetMapper untuk Mask R-CNN RGBD + DBH.

Menghasilkan image 4-channel (RGB 0-255 + depth*255) langsung di "image", sehingga
GeneralizedRCNN.preprocess_image cukup melakukan (x-mean)/std dengan PIXEL_MEAN/STD 4-dim —
TANPA perlu subclass meta-arch. Skala depth (×255) & normalisasi log1p meniru persis
implementasi RGBD MaskDINO (coco_instance_dbh_dataset_mapper.py) agar konsisten.

Menempelkan `gt_dbh` per-instance dari field anotasi "dbh" (mm; 0 = vine/tanpa trunk → di-skip di loss).
"""
import copy
import os

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

__all__ = ["RGBDDatasetMapper"]

# log1p(237.6) — sama dengan _LOG_DEPTH_MAX MaskDINO (max depth efektif dataset)
_LOG_DEPTH_MAX = float(np.log1p(237.6))


def _read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").strip()
        assert header in ("PF", "Pf"), f"Not a PFM file: {path}"
        w, h = (int(v) for v in f.readline().decode("latin-1").split())
        scale = float(f.readline().decode("latin-1").strip())
        endian = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(f.read(), dtype=endian).reshape(h, w)
    return np.flipud(data).copy()


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth < 65000
    out = np.where(valid, np.log1p(depth), 0.0) / _LOG_DEPTH_MAX
    return out.astype(np.float32)          # [0, 1]


class RGBDDatasetMapper(DatasetMapper):
    @configurable
    def __init__(self, *, use_depth=False, depth_dir="", dbh_max_depth=None, **kwargs):
        super().__init__(**kwargs)
        self.use_depth = use_depth
        self.depth_dir = depth_dir
        self.dbh_max_depth = dbh_max_depth   # metres; None = filter nonaktif

    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = super().from_config(cfg, is_train)
        ret["use_depth"] = cfg.INPUT.USE_DEPTH
        ret["depth_dir"] = cfg.INPUT.DEPTH_DIR
        max_depth = float(getattr(cfg.INPUT, "DBH_MAX_DEPTH", 0.0))
        ret["dbh_max_depth"] = max_depth if max_depth > 0 else None
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)  # HWC, RGB, 0-255
        utils.check_image_size(dataset_dict, image)

        depth_raw = None
        if self.use_depth:
            stem = os.path.splitext(os.path.basename(dataset_dict["file_name"]))[0]
            depth_raw = _read_pfm(os.path.join(self.depth_dir, stem + ".pfm"))  # HxW, metres (>=65000 = invalid)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]

        img_t = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).float()  # (3,H,W)
        depth_raw_t = None
        if depth_raw is not None:
            depth_raw_t = transforms.apply_image(depth_raw)             # ikut transform geometris yg sama
            depth_norm = _normalize_depth(depth_raw_t)
            depth_t = torch.as_tensor(np.ascontiguousarray(depth_norm)).unsqueeze(0).float().mul_(255.0)  # (1,H,W) 0-255
            img_t = torch.cat([img_t, depth_t], dim=0)                  # (4,H,W)
        dataset_dict["image"] = img_t

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            # gt_dbh sebelum filter_empty agar indexing konsisten (di-index ulang oleh filter).
            # Field anotasi "dbh" dalam CM (dikonfirmasi 22 Agt 2026: median mentah 29.69,
            # max 754.4 -- masuk akal sbg cm, absurd sbg mm; konsisten dgn konvensi
            # trunk_roi_dbh_head_hybrid.py: `gt_mm = dbh_cm * 10.0`). DBHROIHeads/roi_heads.py
            # mengasumsikan gt_dbh SUDAH dalam mm (docstring: "target = log1p(gt_dbh_mm)"),
            # jadi WAJIB dikali 10 di sini -- sebelum baris ini ditambahkan, seluruh training
            # DBH Mask R-CNN (termasuk run 155k cap30 yg baru selesai) belajar pada skala 10x
            # terlalu kecil.
            gt_dbh = torch.tensor(
                [float(a.get("dbh", 0.0)) * 10.0 for a in annos], dtype=torch.float32
            )
            # RubberFig (category_id=12) cap 150cm=1500mm, meniru RUBBERFIG_CAP_CM di
            # trunk_roi_dbh_head_hybrid.py -- distribusi bimodal, sebagian anotasi RubberFig
            # ekstrem (GT sampai 754cm, jelas keliru). Ditemukan 25 Agt 2026: tanpa filter ini,
            # 50/1537 instance (3.3%) merusak RMSE test dari ~81mm ke 404mm. Instance yang
            # kena cap diperlakukan sama seperti vine: gt_dbh=0, di-skip dari loss.
            for i, a in enumerate(annos):
                if a.get("category_id") == 12 and gt_dbh[i] >= 1500.0:
                    gt_dbh[i] = 0.0
            if (self.dbh_max_depth is not None and depth_raw_t is not None
                    and instances.has("gt_masks") and len(instances) > 0):
                # Proxy jarak kamera-ke-pohon: median depth mentah di dalam gt_mask (bukan
                # estimasi geometris trunk-row breast-height seperti pipeline FocalNet-L —
                # DBHROIHeads meregresi dari feature ROI, tidak punya konsep trunk row).
                # Instance yang jaraknya tak terukur (mask jatuh di piksel background/invalid)
                # atau > dbh_max_depth diperlakukan sama seperti vine: gt_dbh=0, di-skip loss.
                masks_np = instances.gt_masks.tensor.numpy().astype(bool)  # (N,H,W)
                for i in range(masks_np.shape[0]):
                    region = depth_raw_t[masks_np[i]]
                    valid_px = region[region < 65000.0]
                    if valid_px.size == 0 or float(np.median(valid_px)) > self.dbh_max_depth:
                        gt_dbh[i] = 0.0
            instances.gt_dbh = gt_dbh
            if self.recompute_boxes and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
