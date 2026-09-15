# Mosaic (crowd) augmentation mapper — extends COCOInstanceDBHDatasetMapper.
# Tujuan: perbanyak instance per gambar (2x2 mosaic) agar model belajar mendeteksi
# lebih banyak pohon per scene (atasi under-detection scene padat). Depth & DBH ikut.
# Register lewat DATASET_MAPPER_NAME: "coco_instance_lsj_dbh_mosaic" + monkey-patch build_train_loader.
import copy
import os
import random

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode, BitMasks, PolygonMasks
from pycocotools import mask as maskUtils

from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import (
    COCOInstanceDBHDatasetMapper, _read_pfm, _normalize_depth,
)
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
    convert_coco_poly_to_mask,
)

__all__ = ["MosaicDBHDatasetMapper"]


class MosaicDBHDatasetMapper(COCOInstanceDBHDatasetMapper):
    """2x2 mosaic crowd-augmentation. Dengan probabilitas `mosaic_prob`, gabung 4 gambar
    (current + 3 acak) jadi satu kanvas, lalu pipeline transform/LSJ yang sama dijalankan.
    Jika tidak mosaic, identik dengan parent (single image)."""

    @configurable
    def __init__(self, is_train=True, *, tfm_gens, image_format, mask_format="polygon",
                 use_depth=False, depth_dir="", mosaic_prob=0.5, train_dataset=""):
        super().__init__(is_train=is_train, tfm_gens=tfm_gens, image_format=image_format,
                         mask_format=mask_format, use_depth=use_depth, depth_dir=depth_dir)
        self.mosaic_prob = mosaic_prob
        self._train_dataset = train_dataset
        self._pool = None  # lazy

    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = super().from_config(cfg, is_train)
        ret["mosaic_prob"] = cfg.INPUT.get("MOSAIC_PROB", 0.5)
        ret["train_dataset"] = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else ""
        return ret

    def _get_pool(self):
        if self._pool is None:
            self._pool = DatasetCatalog.get(self._train_dataset)
        return self._pool

    # ---- bangun mosaic image+depth+annotations (semua di koordinat kanvas) ----
    def _build_mosaic(self, base_dict):
        pool = self._get_pool()
        dicts = [base_dict] + [copy.deepcopy(random.choice(pool)) for _ in range(3)]
        # baca semua dulu untuk tentukan ukuran kuadran (pakai maksimum agar muat)
        imgs, deps = [], []
        for d in dicts:
            im = utils.read_image(d["file_name"], format=self.img_format)
            imgs.append(im)
            if self.use_depth:
                stem = os.path.splitext(os.path.basename(d["file_name"]))[0]
                deps.append(_normalize_depth(_read_pfm(os.path.join(self.depth_dir, stem + ".pfm"))))
            else:
                deps.append(None)
        qh = max(im.shape[0] for im in imgs)
        qw = max(im.shape[1] for im in imgs)
        H, W = 2 * qh, 2 * qw
        canvas = np.zeros((H, W, imgs[0].shape[2]), dtype=imgs[0].dtype)
        dcanvas = np.zeros((H, W), dtype=np.float32) if self.use_depth else None
        offsets = [(0, 0), (0, qw), (qh, 0), (qh, qw)]  # (oy, ox)
        annos = []
        for (im, dep, d, (oy, ox)) in zip(imgs, deps, dicts, offsets):
            ih, iw = im.shape[:2]
            canvas[oy:oy + ih, ox:ox + iw] = im
            if self.use_depth and dep is not None:
                dcanvas[oy:oy + ih, ox:ox + iw] = dep
            for ann in d.get("annotations", []):
                if ann.get("iscrowd", 0) != 0:
                    continue
                seg = ann["segmentation"]
                # decode -> tempel ke kanvas -> re-encode RLE (ukuran kanvas)
                m = maskUtils.decode(seg) if isinstance(seg, dict) else None
                if m is None:
                    continue
                cm = np.zeros((H, W), dtype=np.uint8, order="F")
                cm[oy:oy + ih, ox:ox + iw] = m[:ih, :iw]
                if cm.sum() == 0:
                    continue
                rle = maskUtils.encode(np.asfortranarray(cm))
                rle["counts"] = rle["counts"].decode("ascii")
                na = copy.deepcopy(ann)
                na["segmentation"] = rle
                bx = ann["bbox"]
                bm = ann.get("bbox_mode", BoxMode.XYWH_ABS)
                xywh = BoxMode.convert(bx, bm, BoxMode.XYWH_ABS)
                na["bbox"] = [xywh[0] + ox, xywh[1] + oy, xywh[2], xywh[3]]
                na["bbox_mode"] = BoxMode.XYWH_ABS
                annos.append(na)
        return canvas, dcanvas, annos, H, W

    def __call__(self, dataset_dict):
        if not (self.is_train and random.random() < self.mosaic_prob):
            return super().__call__(dataset_dict)

        dataset_dict = copy.deepcopy(dataset_dict)
        image, depth, annos, H, W = self._build_mosaic(dataset_dict)
        dataset_dict["height"], dataset_dict["width"] = H, W

        padding_mask = np.ones((H, W))
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)
        if depth is not None:
            depth = transforms.apply_image(depth[:, :, np.newaxis])[:, :, 0]

        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        if depth is not None:
            dataset_dict["depth"] = torch.as_tensor(np.ascontiguousarray(depth))

        for anno in annos:
            anno.pop("keypoints", None)
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in annos if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.mask_format)
        instances.gt_dbh = torch.tensor([a.get("dbh", 0.0) for a in annos], dtype=torch.float32)
        if not instances.has("gt_masks"):
            if self.mask_format == "bitmask":
                instances.gt_masks = BitMasks(torch.zeros((0, *image_shape), dtype=torch.uint8))
            else:
                instances.gt_masks = PolygonMasks([])
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        instances = utils.filter_empty_instances(instances)
        h, w = instances.image_size
        if hasattr(instances, "gt_masks"):
            if self.mask_format == "bitmask":
                instances.gt_masks = instances.gt_masks.tensor
            else:
                instances.gt_masks = convert_coco_poly_to_mask(instances.gt_masks.polygons, h, w)
        dataset_dict["instances"] = instances
        return dataset_dict
