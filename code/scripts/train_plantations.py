"""
Thin wrapper around MaskDINO/train_net.py that registers the plantations
dataset before delegating to the upstream main().

Run via SLURM or locally:
    python scripts/train_plantations.py \
        --config-file configs/maskdino_SwinB_pl_rle_f1000.yaml \
        --num-gpus 1
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"

sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(HERE))

from register_plantations import register_all_plantations  # noqa: E402
from split_validator import check_stratified_datasets      # noqa: E402

import train_net  # noqa: E402  (from MaskDINO repo)
from detectron2.engine import launch  # noqa: E402
from detectron2.utils.events import CommonMetricPrinter, JSONWriter


def _main(args):
    register_all_plantations(os.environ.get("PLANTATIONS_ROOT"))

    cfg = train_net.setup(args)
    check_stratified_datasets(cfg)

    def _safe_build_writers(self):
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]
    train_net.Trainer.build_writers = _safe_build_writers
    return train_net.main(args)


if __name__ == "__main__":
    args = train_net.default_argument_parser().parse_args()
    launch(
        _main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
