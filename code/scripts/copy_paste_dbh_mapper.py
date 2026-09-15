# Copy-paste (crowd) augmentation mapper — extends COCOInstanceDBHDatasetMapper.
# Tujuan: perbanyak instance per gambar DENGAN MEMPERTAHANKAN SKALA ASLI pohon
# (beda dgn mosaic 2x2 yang menyusutkan tiap gambar jadi 1/4). Instance dari
# 1-2 gambar donor (domain sama: RF<->RF / plantation<->plantation) ditempel ke
# kanvas base; mask base yang tertimpa di-okklusi. Depth & DBH ikut tiap instance.
# Picu lewat INPUT.COPY_PASTE_PROB > 0 + monkey-patch build_train_loader.
import copy
import os
import random

import cv2
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

__all__ = ["CopyPasteDBHDatasetMapper"]

_MIN_AREA = 16  # px; instance lebih kecil dibuang (sisa setelah okklusi/resize)


def _domain_of(file_name):
    """Tag domain kasar dari path agar donor sejenis (RF padat <-> RF padat)."""
    fn = file_name.lower()
    if "ainforest" in fn or "rainforest" in fn:
        return "rainforest"
    if "lantation" in fn:
        return "plantation"
    return "other"


class CopyPasteDBHDatasetMapper(COCOInstanceDBHDatasetMapper):
    """Copy-paste crowd-augmentation. Dengan probabilitas `copy_paste_prob`, ambil
    subset instance dari 1-`max_donors` gambar donor (domain sama), tempel ke kanvas
    base pada skala asli, lalu jalankan pipeline transform/LSJ yang sama.
    Jika tidak copy-paste, identik dengan parent (single image)."""

    @configurable
    def __init__(self, is_train=True, *, tfm_gens, image_format, mask_format="polygon",
                 use_depth=False, depth_dir="", copy_paste_prob=0.5, max_donors=2,
                 train_dataset=""):
        super().__init__(is_train=is_train, tfm_gens=tfm_gens, image_format=image_format,
                         mask_format=mask_format, use_depth=use_depth, depth_dir=depth_dir)
        self.copy_paste_prob = copy_paste_prob
        self.max_donors = max_donors
        self._train_dataset = train_dataset
        self._pool = None       # lazy: list semua dict
        self._pool_by_dom = None  # lazy: dict domain -> list

    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = super().from_config(cfg, is_train)
        ret["copy_paste_prob"] = cfg.INPUT.get("COPY_PASTE_PROB", 0.5)
        ret["max_donors"] = cfg.INPUT.get("COPY_PASTE_MAX_DONORS", 2)
        ret["train_dataset"] = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else ""
        return ret

    def _get_pool(self, domain):
        if self._pool is None:
            self._pool = DatasetCatalog.get(self._train_dataset)
            self._pool_by_dom = {}
            for d in self._pool:
                self._pool_by_dom.setdefault(_domain_of(d["file_name"]), []).append(d)
        # donor sejenis bila ada; jika tidak, pakai seluruh pool
        cand = self._pool_by_dom.get(domain)
        return cand if cand and len(cand) > 1 else self._pool

    def _read_one(self, d):
        im = utils.read_image(d["file_name"], format=self.img_format)
        dep = None
        if self.use_depth:
            stem = os.path.splitext(os.path.basename(d["file_name"]))[0]
            dep = _normalize_depth(_read_pfm(os.path.join(self.depth_dir, stem + ".pfm")))
        return im, dep

    @staticmethod
    def _decode_mask(ann):
        seg = ann.get("segmentation")
        if not isinstance(seg, dict):
            return None
        return maskUtils.decode(seg).astype(np.uint8)

    @staticmethod
    def _mask_to_anno(base_ann, m):
        """Re-encode mask uint8 (Fortran) -> RLE + bbox dari mask. Return None bila kosong."""
        if m.sum() < _MIN_AREA:
            return None
        rle = maskUtils.encode(np.asfortranarray(m))
        rle["counts"] = rle["counts"].decode("ascii")
        ys, xs = np.where(m > 0)
        na = copy.deepcopy(base_ann)
        na["segmentation"] = rle
        na["bbox"] = [float(xs.min()), float(ys.min()),
                      float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
        na["bbox_mode"] = BoxMode.XYWH_ABS
        return na

    # ---- bangun kanvas copy-paste (image+depth+annotations di koordinat base) ----
    def _build_copy_paste(self, base_dict):
        base_im, base_dep = self._read_one(base_dict)
        H, W = base_im.shape[:2]
        image = base_im.copy()
        depth = base_dep.copy() if base_dep is not None else None

        # mask base (akan di-okklusi oleh instance yang ditempel di atasnya)
        base_masks = []
        for a in base_dict.get("annotations", []):
            if a.get("iscrowd", 0) != 0:
                continue
            m = self._decode_mask(a)
            if m is not None and m.sum() > 0:
                base_masks.append([copy.deepcopy(a), m])

        domain = _domain_of(base_dict["file_name"])
        pool = self._get_pool(domain)
        paste_union = np.zeros((H, W), dtype=np.uint8)
        pasted_annos = []

        n_donors = random.randint(1, max(1, self.max_donors))
        for _ in range(n_donors):
            dd = copy.deepcopy(random.choice(pool))
            if dd["file_name"] == base_dict["file_name"]:
                continue
            try:
                dim, ddep = self._read_one(dd)
            except Exception:
                continue
            # samakan ukuran donor -> base (pohon tetap ~skala asli, bukan 1/4 spt mosaic)
            if dim.shape[:2] != (H, W):
                dim = cv2.resize(dim, (W, H), interpolation=cv2.INTER_LINEAR)
                if ddep is not None:
                    ddep = cv2.resize(ddep, (W, H), interpolation=cv2.INTER_NEAREST)

            danns = [a for a in dd.get("annotations", []) if a.get("iscrowd", 0) == 0]
            random.shuffle(danns)
            k = max(1, int(len(danns) * random.uniform(0.3, 0.8)))
            for a in danns[:k]:
                m = self._decode_mask(a)
                if m is None:
                    continue
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                if m.sum() < _MIN_AREA:
                    continue
                sel = m > 0
                image[sel] = dim[sel]
                if depth is not None and ddep is not None:
                    depth[sel] = ddep[sel]
                paste_union[sel] = 1
                na = self._mask_to_anno(a, m)
                if na is not None:
                    pasted_annos.append(na)

        # okklusi mask base oleh union instance yang ditempel
        final_annos = []
        for a, m in base_masks:
            if paste_union.any():
                m = m.copy()
                m[paste_union > 0] = 0
            na = self._mask_to_anno(a, m)
            if na is not None:
                final_annos.append(na)
        final_annos.extend(pasted_annos)
        return image, depth, final_annos, H, W

    def __call__(self, dataset_dict):
        if not (self.is_train and random.random() < self.copy_paste_prob):
            return super().__call__(dataset_dict)

        dataset_dict = copy.deepcopy(dataset_dict)
        image, depth, annos, H, W = self._build_copy_paste(dataset_dict)
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
