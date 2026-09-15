"""
Register the plantations dataset (train/val/test) with Detectron2's DatasetCatalog.

Annotations source: <PLANTATIONS_ROOT>/annotations_inst/instances_{split}_rle.json
Images:             <PLANTATIONS_ROOT>/rgb_resized/

Dataset names registered:
  plantations_inst_rle_{split}          — full RLE dataset (no area filter)
  plantations_inst_rle_f0_{split}       — alias f0 = no filter (same as above, for sweep)
  plantations_inst_rle_f{N}_{split}     — RLE filtered by min area N (e.g. filtered_rle_f300/)

Env var (optional): PLANTATIONS_ROOT
    Default: <project_root>/data/plantations
"""
import os
from pathlib import Path
from typing import Optional, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

CLASS_NAMES = ["Apple", "Lemon", "Loquat", "Mango", "Orange", "Persimmon", "Pomegranate"]

# Extra per-annotation fields to preserve from the COCO JSON (beyond Detectron2 defaults).
_EXTRA_ANN_KEYS = ["dbh", "height_ue", "crown_diameter", "depth_mean", "depth_min", "depth_max"]


def _register(name: str, ann_file: Path, img_dir: Path) -> None:
    if name in MetadataCatalog.list():
        return
    ann_file_str = str(ann_file)
    img_dir_str = str(img_dir)
    DatasetCatalog.register(
        name,
        lambda f=ann_file_str, d=img_dir_str: load_coco_json(
            f, d, name, extra_annotation_keys=_EXTRA_ANN_KEYS
        ),
    )
    MetadataCatalog.get(name).set(
        json_file=ann_file_str,
        image_root=img_dir_str,
        thing_classes=CLASS_NAMES,
        evaluator_type="coco",
    )


def register_all_plantations(root: Optional[Union[str, os.PathLike]] = None) -> None:
    if root is None:
        root = os.environ.get(
            "PLANTATIONS_ROOT",
            str(Path(__file__).resolve().parents[1] / "data" / "plantations"),
        )
    root = Path(root)
    img_dir = root / "rgb_resized"
    ann_dir = root / "annotations_inst"

    if not ann_dir.exists():
        raise FileNotFoundError(f"annotations_inst not found: {ann_dir}")

    import re

    # Primary RLE datasets: instances_{split}_rle.json
    # Also registered as f0 (no area filter) for sweep compatibility.
    for split in ("train", "val", "test"):
        ann_file = ann_dir / f"instances_{split}_rle.json"
        if ann_file.exists():
            _register(f"plantations_inst_rle_{split}", ann_file, img_dir)
            _register(f"plantations_inst_rle_f0_{split}", ann_file, img_dir)

    # Filtered RLE subdirs: annotations_inst/filtered_rle_f300/, filtered_rle_f1000_repaired_v4/, etc.
    for filt_dir in sorted(ann_dir.glob("filtered_rle_f*")):
        if not filt_dir.is_dir():
            continue
        m = re.match(r"filtered_rle_f(\d+)((?:_[A-Za-z0-9]+)*)$", filt_dir.name)
        if not m:
            continue
        fnum = m.group(1)
        suffix = m.group(2) or ""
        for split in ("train", "val", "test"):
            ann_file = filt_dir / f"instances_{split}.json"
            if ann_file.exists():
                _register(f"plantations_inst_rle_f{fnum}{suffix}_{split}", ann_file, img_dir)


if __name__ == "__main__":
    register_all_plantations()
    print("Classes:", CLASS_NAMES)
    registered = sorted(n for n in MetadataCatalog.list() if n.startswith("plantations"))
    for n in registered:
        print(f"Registered: {n}")
