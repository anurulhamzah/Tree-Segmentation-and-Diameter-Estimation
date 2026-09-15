"""
Register the rainforests dataset (train/val/test) with Detectron2's
DatasetCatalog.

Annotation source priority:
    1. <RAINFORESTS_ROOT>/splits/instances_{train,val,test}.json   (preferred, scene-aware split)
    2. <RAINFORESTS_ROOT>/annotations/instances_{train,val}.json   (legacy split, no test)

All splits share the same image folder: <RAINFORESTS_ROOT>/rgb_resized/.
Detectron2 separates them via image_id inside each JSON.

Env var (optional): RAINFORESTS_ROOT
    Default: <project_root>/data/rainforests
"""
import os
from pathlib import Path
from typing import Optional, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

CLASS_NAMES = ["AliiFig", "BangaloPalm", "Fern", "LeechVine", "RubberFig", "Umbrella"]

# Extra per-annotation fields to preserve from the COCO JSON (beyond Detectron2 defaults).
# Detectron2's load_coco_json only keeps ["iscrowd","bbox","keypoints","category_id"] by default;
# all other fields are dropped unless listed here.
_EXTRA_ANN_KEYS = ["dbh", "height_ue", "crown_diameter", "depth_mean", "depth_min", "depth_max"]


def _resolve_ann(root: Path, split: str) -> Optional[Path]:
    candidates = [
        root / "splits"      / f"instances_{split}.json",
        root / "annotations" / f"instances_{split}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


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


def register_all_rainforests(root: Optional[Union[str, os.PathLike]] = None) -> None:
    if root is None:
        root = os.environ.get(
            "RAINFORESTS_ROOT",
            str(Path(__file__).resolve().parents[1] / "data" / "rainforests"),
        )
    root = Path(root)
    img_dir = root / "rgb_resized"

    # Full splits
    for split in ("train", "val", "test"):
        ann_file = _resolve_ann(root, split)
        if ann_file is not None:
            _register(f"rainforests_{split}", ann_file, img_dir)

    # 5% sandbox splits (jika tersedia)
    sample_dir = root / "splits" / "sample"
    for split in ("train", "val", "test"):
        ann_file = sample_dir / f"instances_sample_{split}.json"
        if ann_file.exists():
            _register(f"rainforests_sample_{split}", ann_file, img_dir)

    # pctXX splits: splits/pct20/, splits/pct50/, etc.
    import re
    for pct_dir in sorted((root / "splits").glob("pct*")):
        if not pct_dir.is_dir():
            continue
        m = re.match(r"pct(\d+)$", pct_dir.name)
        if not m:
            continue
        pct_num = m.group(1)
        for split in ("train", "val", "test"):
            ann_file = pct_dir / f"instances_{split}.json"
            if ann_file.exists():
                _register(f"rainforests_{pct_num}pct_{split}", ann_file, img_dir)

    # annotations_480 datasets
    register_rainforests480(root)

    # annotations_960 datasets
    register_rainforests960(root)

    # annotation_inst datasets (480x270, 70/15/15 split)
    register_rainforests_inst(root)


def register_rainforests480(root: Optional[Union[str, os.PathLike]] = None) -> None:
    """Register datasets sourced from annotations_480/ (480x270, denser annotations)."""
    if root is None:
        root = os.environ.get(
            "RAINFORESTS_ROOT",
            str(Path(__file__).resolve().parents[1] / "data" / "rainforests"),
        )
    root = Path(root)
    img_dir = root / "rgb_resized"
    ann480_dir = root / "annotations_480"

    if not ann480_dir.exists():
        return

    # Full train/val
    for split in ("train", "val"):
        ann_file = ann480_dir / f"instances_{split}.json"
        if ann_file.exists():
            _register(f"rainforests480_{split}", ann_file, img_dir)

    # Sample splits (e.g. 5%)
    sample_dir = ann480_dir / "sample"
    for split in ("train", "val"):
        ann_file = sample_dir / f"instances_sample_{split}.json"
        if ann_file.exists():
            _register(f"rainforests480_sample_{split}", ann_file, img_dir)

    # filtered/ subdirectory (area-filtered annotations)
    filtered_dir = ann480_dir / "filtered"
    if filtered_dir.exists():
        for split in ("train", "val"):
            ann_file = filtered_dir / f"instances_{split}.json"
            if ann_file.exists():
                _register(f"rainforests480_filtered_{split}", ann_file, img_dir)

        filtered_sample_dir = filtered_dir / "sample"
        for split in ("train", "val"):
            ann_file = filtered_sample_dir / f"instances_sample_{split}.json"
            if ann_file.exists():
                _register(f"rainforests480_filtered_sample_{split}", ann_file, img_dir)

        import re
        for pct_dir in sorted(filtered_dir.glob("pct*")):
            if not pct_dir.is_dir():
                continue
            m = re.match(r"pct(\d+)$", pct_dir.name)
            if not m:
                continue
            pct_num = m.group(1)
            for split in ("train", "val"):
                ann_file = pct_dir / f"instances_{split}.json"
                if ann_file.exists():
                    _register(f"rainforests480_filtered_{pct_num}pct_{split}", ann_file, img_dir)

    # filtered_f2000/ subdirectory (area > 2000, 480px)
    filtered_f2000_dir = ann480_dir / "filtered_f2000"
    if filtered_f2000_dir.exists():
        for split in ("train", "val"):
            ann_file = filtered_f2000_dir / f"instances_{split}.json"
            if ann_file.exists():
                _register(f"rainforests480_f2000_{split}", ann_file, img_dir)

        filtered_f2000_sample_dir = filtered_f2000_dir / "sample"
        for split in ("train", "val"):
            ann_file = filtered_f2000_sample_dir / f"instances_sample_{split}.json"
            if ann_file.exists():
                _register(f"rainforests480_f2000_sample_{split}", ann_file, img_dir)

        import re
        for pct_dir in sorted(filtered_f2000_dir.glob("pct*")):
            if not pct_dir.is_dir():
                continue
            m = re.match(r"pct(\d+)$", pct_dir.name)
            if not m:
                continue
            pct_num = m.group(1)
            for split in ("train", "val"):
                ann_file = pct_dir / f"instances_{split}.json"
                if ann_file.exists():
                    _register(f"rainforests480_f2000_{pct_num}pct_{split}", ann_file, img_dir)

    # pctXX splits inside annotations_480/
    import re
    for pct_dir in sorted(ann480_dir.glob("pct*")):
        if not pct_dir.is_dir():
            continue
        m = re.match(r"pct(\d+)$", pct_dir.name)
        if not m:
            continue
        pct_num = m.group(1)
        for split in ("train", "val"):
            ann_file = pct_dir / f"instances_{split}.json"
            if ann_file.exists():
                _register(f"rainforests480_{pct_num}pct_{split}", ann_file, img_dir)


def register_rainforests960(root: Optional[Union[str, os.PathLike]] = None) -> None:
    """Register datasets sourced from annotations_960/ (960x540, higher resolution)."""
    if root is None:
        root = os.environ.get(
            "RAINFORESTS_ROOT",
            str(Path(__file__).resolve().parents[1] / "data" / "rainforests"),
        )
    root = Path(root)
    img_dir = root / "rgb_960"
    ann960_dir = root / "annotations_960"

    if not ann960_dir.exists():
        return

    # Full train/val
    for split in ("train", "val"):
        ann_file = ann960_dir / f"instances_{split}.json"
        if ann_file.exists():
            _register(f"rainforests960_{split}", ann_file, img_dir)

    # filtered/ subdirectory (area-filtered annotations)
    filtered_dir = ann960_dir / "filtered"
    if filtered_dir.exists():
        for split in ("train", "val"):
            ann_file = filtered_dir / f"instances_{split}.json"
            if ann_file.exists():
                _register(f"rainforests960_filtered_{split}", ann_file, img_dir)

        filtered_sample_dir = filtered_dir / "sample"
        for split in ("train", "val"):
            ann_file = filtered_sample_dir / f"instances_sample_{split}.json"
            if ann_file.exists():
                _register(f"rainforests960_filtered_sample_{split}", ann_file, img_dir)

        import re
        for pct_dir in sorted(filtered_dir.glob("pct*")):
            if not pct_dir.is_dir():
                continue
            m = re.match(r"pct(\d+)$", pct_dir.name)
            if not m:
                continue
            pct_num = m.group(1)
            for split in ("train", "val"):
                ann_file = pct_dir / f"instances_{split}.json"
                if ann_file.exists():
                    _register(f"rainforests960_filtered_{pct_num}pct_{split}", ann_file, img_dir)

    # filtered_f1000/ subdirectory (area > 1000, 960px)
    filtered_f1000_dir = ann960_dir / "filtered_f1000"
    if filtered_f1000_dir.exists():
        for split in ("train", "val"):
            ann_file = filtered_f1000_dir / f"instances_{split}.json"
            if ann_file.exists():
                _register(f"rainforests960_f1000_{split}", ann_file, img_dir)

        filtered_f1000_sample_dir = filtered_f1000_dir / "sample"
        for split in ("train", "val"):
            ann_file = filtered_f1000_sample_dir / f"instances_sample_{split}.json"
            if ann_file.exists():
                _register(f"rainforests960_f1000_sample_{split}", ann_file, img_dir)

        import re
        for pct_dir in sorted(filtered_f1000_dir.glob("pct*")):
            if not pct_dir.is_dir():
                continue
            m = re.match(r"pct(\d+)$", pct_dir.name)
            if not m:
                continue
            pct_num = m.group(1)
            for split in ("train", "val"):
                ann_file = pct_dir / f"instances_{split}.json"
                if ann_file.exists():
                    _register(f"rainforests960_f1000_{pct_num}pct_{split}", ann_file, img_dir)


def register_rainforests_inst(root: Optional[Union[str, os.PathLike]] = None) -> None:
    """Register datasets sourced from annotation_inst/ (480x270, 70/15/15 scene-aware split)."""
    if root is None:
        root = os.environ.get(
            "RAINFORESTS_ROOT",
            str(Path(__file__).resolve().parents[1] / "data" / "rainforests"),
        )
    root = Path(root)
    img_dir = root / "rgb_resized"
    inst_dir = root / "annotation_inst"

    if not inst_dir.exists():
        return

    import re

    # Primary: _rebuilt files sit directly in annotation_inst/
    for split in ("train", "val", "test"):
        ann_file = inst_dir / f"instances_{split}_rebuilt.json"
        if ann_file.exists():
            _register(f"rainforests_inst_{split}", ann_file, img_dir)

    # RLE full dataset: instances_{split}_rle.json
    # Also registered as f0 (no area filter) for sweep compatibility.
    for split in ("train", "val", "test"):
        ann_file = inst_dir / f"instances_{split}_rle.json"
        if ann_file.exists():
            _register(f"rainforests_inst_rle_{split}", ann_file, img_dir)
            _register(f"rainforests_inst_rle_f0_{split}", ann_file, img_dir)

    # Legacy fallback: split701515/ subdir (kept for backward compat)
    split_dir = inst_dir / "split701515"
    if split_dir.exists():
        for split in ("train", "val", "test"):
            ann_file = split_dir / f"instances_{split}.json"
            if ann_file.exists():
                _register(f"rainforests_inst_{split}", ann_file, img_dir)

        sample_dir = split_dir / "sample"
        for split in ("train", "val"):
            ann_file = sample_dir / f"instances_sample_{split}.json"
            if ann_file.exists():
                _register(f"rainforests_inst_sample_{split}", ann_file, img_dir)

        for pct_dir in sorted(split_dir.glob("pct*")):
            if not pct_dir.is_dir():
                continue
            m = re.match(r"pct(\d+)$", pct_dir.name)
            if not m:
                continue
            pct_num = m.group(1)
            for split in ("train", "val"):
                ann_file = pct_dir / f"instances_{split}.json"
                if ann_file.exists():
                    _register(f"rainforests_inst_{pct_num}pct_{split}", ann_file, img_dir)

        for filt_dir in sorted(split_dir.glob("filtered_f*")):
            if not filt_dir.is_dir():
                continue
            m = re.match(r"filtered_f(\d+)$", filt_dir.name)
            if not m:
                continue
            fnum = m.group(1)
            for split in ("train", "val", "test"):
                ann_file = filt_dir / f"instances_{split}.json"
                if ann_file.exists():
                    _register(f"rainforests_inst_f{fnum}_{split}", ann_file, img_dir)
            fsample_dir = filt_dir / "sample"
            for split in ("train", "val"):
                ann_file = fsample_dir / f"instances_sample_{split}.json"
                if ann_file.exists():
                    _register(f"rainforests_inst_f{fnum}_sample_{split}", ann_file, img_dir)
            for pct_dir in sorted(filt_dir.glob("pct*")):
                if not pct_dir.is_dir():
                    continue
                mp = re.match(r"pct(\d+)$", pct_dir.name)
                if not mp:
                    continue
                pct_num = mp.group(1)
                for split in ("train", "val"):
                    ann_file = pct_dir / f"instances_{split}.json"
                    if ann_file.exists():
                        _register(f"rainforests_inst_f{fnum}_{pct_num}pct_{split}", ann_file, img_dir)

    # filtered_fXXXX subdirs directly under annotation_inst/ (e.g. filtered_f1000/)
    for filt_dir in sorted(inst_dir.glob("filtered_f*")):
        if not filt_dir.is_dir():
            continue
        m = re.match(r"filtered_f(\d+)$", filt_dir.name)
        if not m:
            continue
        fnum = m.group(1)
        for split in ("train", "val", "test"):
            ann_file = filt_dir / f"instances_{split}.json"
            if ann_file.exists():
                _register(f"rainforests_inst_f{fnum}_{split}", ann_file, img_dir)

    # filtered_rle_fXXXX[_suffix] subdirs — _rle annotation variants
    # Matches: filtered_rle_f1000, filtered_rle_f1000_repaired_v4_stratified, etc.
    for filt_dir in sorted(inst_dir.glob("filtered_rle_f*")):
        if not filt_dir.is_dir():
            continue
        m = re.match(r"filtered_rle_f(\d+)((?:_[A-Za-z0-9]+)*)$", filt_dir.name)
        if not m or filt_dir.name.endswith("_960"):
            continue
        fnum = m.group(1)
        suffix = m.group(2) or ""
        for split in ("train", "val", "test"):
            ann_file = filt_dir / f"instances_{split}.json"
            if ann_file.exists():
                _register(f"rainforests_inst_rle_f{fnum}{suffix}_{split}", ann_file, img_dir)

    # filtered_rle_fXXXX_960 subdirs — 960px variant (e.g. filtered_rle_f1000_960/)
    img_dir_960 = root / "rgb_960"
    for filt_dir in sorted(inst_dir.glob("filtered_rle_f*_960")):
        if not filt_dir.is_dir():
            continue
        m = re.match(r"filtered_rle_f(\d+)_960$", filt_dir.name)
        if not m:
            continue
        fnum = m.group(1)
        for split in ("train", "val", "test"):
            ann_file = filt_dir / f"instances_{split}.json"
            if ann_file.exists():
                _register(f"rainforests_inst_rle_f{fnum}_960_{split}", ann_file, img_dir_960)


if __name__ == "__main__":
    register_all_rainforests()
    print("Classes:", CLASS_NAMES)
    for n in sorted(MetadataCatalog.list()):
        if n.startswith("rainforests"):
            print(f"Registered: {n}")
