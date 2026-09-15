"""
Thin wrapper around MaskDINO/train_net.py that registers the rainforests
dataset before delegating to the upstream main().

Run via slurm/train.slurm; or locally:
    python scripts/train_rainforests.py \
        --config-file configs/maskdino_R50_rainforests.yaml \
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

from register_rainforests import register_all_rainforests  # noqa: E402

import train_net  # noqa: E402  (from MaskDINO repo)
from detectron2.engine import launch  # noqa: E402
from detectron2.utils.events import CommonMetricPrinter, JSONWriter


def _main(args):
    register_all_rainforests(os.environ.get("RAINFORESTS_ROOT"))
    # Patch build_writers to exclude TensorboardXWriter — scratch filesystem
    # can delete the TFEvents file mid-training, causing a background thread
    # crash that kills the entire job (FileNotFoundError in EventFileWriter).
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
