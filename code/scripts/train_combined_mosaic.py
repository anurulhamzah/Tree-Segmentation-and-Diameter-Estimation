"""train_combined.py + mosaic crowd-aug loader.
Patch Trainer.build_train_loader agar memakai MosaicDBHDatasetMapper saat
DATASET_MAPPER_NAME == 'coco_instance_lsj_dbh_mosaic'. Selain itu identik dgn train_combined.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(HERE))

from register_rainforests import register_all_rainforests  # noqa: E402
from register_plantations import register_all_plantations  # noqa: E402
from register_combined import register_all_combined        # noqa: E402

import train_net  # noqa: E402
from detectron2.engine import launch  # noqa: E402
from detectron2.data import build_detection_train_loader  # noqa: E402
from detectron2.utils.events import CommonMetricPrinter, JSONWriter  # noqa: E402
from mosaic_dbh_mapper import MosaicDBHDatasetMapper  # noqa: E402

_orig_build = train_net.Trainer.build_train_loader.__func__


def _build_train_loader(cls, cfg):
    # Picu mosaic crowd-aug HANYA di train loader bila MOSAIC_PROB>0 (mapper depth/DBH).
    # DATASET_MAPPER_NAME tetap 'coco_instance_lsj_dbh' → test loader tetap pakai depth (eval utuh).
    if (cfg.INPUT.get("MOSAIC_PROB", 0.0) > 0
            and cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj_dbh"):
        mapper = MosaicDBHDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)
    return _orig_build(cls, cfg)


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
    return train_net.main(args)


if __name__ == "__main__":
    args = train_net.default_argument_parser().parse_args()
    launch(_main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
