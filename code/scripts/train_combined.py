"""
Thin wrapper around MaskDINO/train_net.py that registers the combined
plantations+rainforests dataset before delegating to the upstream main().

Run via slurm/train_R50_combined_rle_f1000_50k.slurm; or locally:
    python scripts/train_combined.py \
        --config-file configs/maskdino_R50_combined_rle_f1000_50k.yaml \
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
from register_plantations import register_all_plantations  # noqa: E402
from register_combined import register_all_combined        # noqa: E402
from split_validator import check_stratified_datasets      # noqa: E402

import train_net  # noqa: E402  (from MaskDINO repo)
from detectron2.engine import launch  # noqa: E402
from detectron2.utils.events import CommonMetricPrinter, JSONWriter


def _main(args):
    register_all_rainforests(os.environ.get("RAINFORESTS_ROOT"))
    register_all_plantations(os.environ.get("PLANTATIONS_ROOT"))
    register_all_combined(os.environ.get("COMBINED_ROOT"))

    cfg = train_net.setup(args)
    # eval-only: warn only (non-stratified test sets are valid for held-out evaluation)
    check_stratified_datasets(cfg, error=not args.eval_only)

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
