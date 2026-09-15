"""
Mask R-CNN baseline training for rainforests dataset.

Run locally or via SLURM:
    python scripts/train_maskrcnn_rainforests.py \
        --config-file configs/maskrcnn_R50_rf_rle_f1000.yaml \
        --num-gpus 1

For eval-only:
    python scripts/train_maskrcnn_rainforests.py \
        --config-file configs/maskrcnn_R50_rf_rle_f1000.yaml \
        --eval-only MODEL.WEIGHTS <path/to/model_final.pth>
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from register_rainforests import register_all_rainforests  # noqa: E402

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.engine import (  # noqa: E402
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import COCOEvaluator  # noqa: E402
from detectron2.utils.events import CommonMetricPrinter, JSONWriter  # noqa: E402


class Trainer(DefaultTrainer):
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


def main(args):
    register_all_rainforests(os.environ.get("RAINFORESTS_ROOT"))

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

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