"""train_combined_mosaic.py + per-class segmentation loss reweighting.
Setelah model dibangun, timpa criterion.empty_weight[:num_classes] dengan
cfg.MODEL.MaskDINO.PER_CLASS_WEIGHT (list of float, len == num_classes).
Index -1 (no-object/background) tetap dikontrol terpisah oleh NO_OBJECT_WEIGHT
(tidak disentuh script ini). Selain itu identik dengan train_combined_mosaic.
"""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(HERE))

from register_rainforests import register_all_rainforests  # noqa: E402
from register_plantations import register_all_plantations  # noqa: E402
from register_combined import register_all_combined, CLASS_NAMES  # noqa: E402

import train_net  # noqa: E402
from detectron2.engine import launch  # noqa: E402
from detectron2.data import build_detection_train_loader  # noqa: E402
from detectron2.utils.events import CommonMetricPrinter, JSONWriter  # noqa: E402
from mosaic_dbh_mapper import MosaicDBHDatasetMapper  # noqa: E402

_orig_build_loader = train_net.Trainer.build_train_loader.__func__
_orig_build_model = train_net.Trainer.build_model.__func__


def _build_train_loader(cls, cfg):
    if (cfg.INPUT.get("MOSAIC_PROB", 0.0) > 0
            and cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj_dbh"):
        mapper = MosaicDBHDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    return _orig_build_loader(cls, cfg)


def _build_model(cls, cfg):
    model = _orig_build_model(cls, cfg)
    per_class_weight = list(cfg.MODEL.MaskDINO.get("PER_CLASS_WEIGHT", []))
    if not per_class_weight:
        return model
    num_classes = len(CLASS_NAMES)
    assert len(per_class_weight) == num_classes, (
        f"PER_CLASS_WEIGHT len={len(per_class_weight)} != num_classes={num_classes}"
    )
    criterion = model.criterion
    with torch.no_grad():
        for i, w in enumerate(per_class_weight):
            criterion.empty_weight[i] = w
    print(f"[train_combined_classweight] Applied PER_CLASS_WEIGHT to criterion.empty_weight[:{num_classes}]:")
    for name, w in zip(CLASS_NAMES, per_class_weight):
        print(f"  {name:12s} weight={w:.3f}")
    print(f"  [no-object, unchanged] weight={criterion.empty_weight[-1].item():.4f}")
    return model


def _main(args):
    register_all_rainforests(os.environ.get("RAINFORESTS_ROOT"))
    register_all_plantations(os.environ.get("PLANTATIONS_ROOT"))
    register_all_combined(os.environ.get("COMBINED_ROOT"))

    def _safe_build_writers(self):
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]
    train_net.Trainer.build_writers = _safe_build_writers
    train_net.Trainer.build_train_loader = classmethod(_build_train_loader)
    train_net.Trainer.build_model = classmethod(_build_model)
    return train_net.main(args)


if __name__ == "__main__":
    args = train_net.default_argument_parser().parse_args()
    launch(_main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
