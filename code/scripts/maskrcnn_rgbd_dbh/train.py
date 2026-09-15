"""Entry-point training Mask R-CNN R50-FPN + RGBD + DBH pada dataset Combined.

Self-contained: hanya detectron2 + register_*.py proyek + paket maskrcnn_rgbd_dbh.
Tidak menyentuh repo MaskDINO.

Jalankan:
    python scripts/maskrcnn_rgbd_dbh/train.py \
        --config-file configs/maskrcnn_R50_combined_rle_f1000_rgbd_dbh_75k.yaml \
        --num-gpus 1 OUTPUT_DIR <dir>
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))          # untuk register_* dan paket maskrcnn_rgbd_dbh

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import CommonMetricPrinter, JSONWriter

from register_rainforests import register_all_rainforests
from register_plantations import register_all_plantations
from register_combined import register_all_combined

from maskrcnn_rgbd_dbh import add_rgbd_dbh_config, RGBDDatasetMapper  # noqa: E402


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=RGBDDatasetMapper(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name,
                                           mapper=RGBDDatasetMapper(cfg, is_train=False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_writers(self):
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]


def setup(args):
    cfg = get_cfg()
    add_rgbd_dbh_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    register_all_rainforests(os.environ.get("RAINFORESTS_ROOT"))
    register_all_plantations(os.environ.get("PLANTATIONS_ROOT"))
    register_all_combined(os.environ.get("COMBINED_ROOT"))

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return Trainer.test(cfg, model)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
