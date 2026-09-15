# DBH/RGBD extension of COCOInstanceNewBaselineDatasetMapper.
# Use DATASET_MAPPER_NAME: "coco_instance_lsj_dbh" in configs that need
# depth loading (RGBD) or DBH regression targets.  Non-DBH configs continue
# to use "coco_instance_lsj" and the unmodified base mapper.
import copy
import logging
import os

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, PolygonMasks

from .coco_instance_new_baseline_dataset_mapper import (
    COCOInstanceNewBaselineDatasetMapper,
    build_transform_gen,
    convert_coco_poly_to_mask,
)

__all__ = ["COCOInstanceDBHDatasetMapper"]

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
    return out.astype(np.float32)


class COCOInstanceDBHDatasetMapper(COCOInstanceNewBaselineDatasetMapper):
    """
    Extends the base mapper with:
      - Optional RGBD depth loading from .pfm files (USE_DEPTH=True in config).
      - Attaches gt_dbh per instance for DBH regression (dbh=0 for vine species).

    Register as DATASET_MAPPER_NAME: "coco_instance_lsj_dbh".
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        mask_format="polygon",
        use_depth=False,
        depth_dir="",
    ):
        super().__init__(
            is_train=is_train,
            tfm_gens=tfm_gens,
            image_format=image_format,
            mask_format=mask_format,
        )
        self.use_depth = use_depth
        self.depth_dir = depth_dir

    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = super().from_config(cfg, is_train)
        ret["use_depth"] = cfg.INPUT.get("USE_DEPTH", False)
        ret["depth_dir"] = cfg.INPUT.get("DEPTH_DIR", "")
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Load depth before spatial transforms so the same ops apply to both.
        depth = None
        if self.use_depth:
            stem = os.path.splitext(os.path.basename(dataset_dict["file_name"]))[0]
            pfm_path = os.path.join(self.depth_dir, stem + ".pfm")
            depth = _normalize_depth(_read_pfm(pfm_path))  # float32 (H, W)

        padding_mask = np.ones(image.shape[:2])
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
            instances = utils.annotations_to_instances(annos, image_shape,
                                                       mask_format=self.mask_format)
            # Attach gt_dbh before filter_empty_instances so indexing stays consistent.
            # dbh=0 marks vine species (no trunk) — skipped in loss computation.
            instances.gt_dbh = torch.tensor(
                [ann.get("dbh", 0.0) for ann in annos], dtype=torch.float32
            )
            if not instances.has("gt_masks"):
                if self.mask_format == "bitmask":
                    instances.gt_masks = BitMasks(
                        torch.zeros((0, *image_shape), dtype=torch.uint8))
                else:
                    instances.gt_masks = PolygonMasks([])
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)
            h, w = instances.image_size
            if hasattr(instances, "gt_masks"):
                if self.mask_format == "bitmask":
                    instances.gt_masks = instances.gt_masks.tensor
                else:
                    gt_masks = instances.gt_masks
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                    instances.gt_masks = gt_masks

            dataset_dict["instances"] = instances

        return dataset_dict