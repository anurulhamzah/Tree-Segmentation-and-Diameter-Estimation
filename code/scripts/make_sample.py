"""
Generate a 5% scene-aware subsample of train/val/test for fast sanity-check
training. Outputs to data/rainforests/splits/sample/.

The full splits in data/rainforests/splits/ are not modified.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR   = PROJECT_ROOT / "data" / "rainforests" / "splits"
SAMPLE_DIR   = SPLITS_DIR / "sample"

SAMPLE_FRACTION = 0.05
SEED            = 42
SPLITS          = ("train", "val", "test")


def scene_of(stem: str) -> str:
    return stem.split("_")[0]


def subsample_scenes(scenes: list[str], frac: float, rng: random.Random) -> list[str]:
    n = max(1, int(round(frac * len(scenes))))
    return rng.sample(scenes, n)


def write_split(orig_json: Path, sample_stems: set[str], out_json: Path) -> tuple[int, int]:
    with orig_json.open() as f:
        coco = json.load(f)
    keep_imgs, keep_ann = [], []
    next_iid, next_aid = 1, 1
    img_id_remap: dict[int, int] = {}
    for im in coco["images"]:
        if Path(im["file_name"]).stem not in sample_stems:
            continue
        new_im = {**im, "id": next_iid}
        keep_imgs.append(new_im)
        img_id_remap[im["id"]] = next_iid
        next_iid += 1
    for an in coco["annotations"]:
        if an["image_id"] not in img_id_remap:
            continue
        keep_ann.append({
            **an,
            "id": next_aid,
            "image_id": img_id_remap[an["image_id"]],
        })
        next_aid += 1
    out = {"images": keep_imgs, "annotations": keep_ann,
           "categories": coco["categories"]}
    with out_json.open("w") as f:
        json.dump(out, f)
    return len(keep_imgs), len(keep_ann)


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print(f"Sampling {SAMPLE_FRACTION:.0%} per split (scene-aware, seed={SEED})")
    for split in SPLITS:
        scenes_file = SPLITS_DIR / f"{split}_scenes.txt"
        stems_file  = SPLITS_DIR / f"{split}_stems.txt"
        all_scenes = scenes_file.read_text().split()
        all_stems  = stems_file.read_text().split()

        sampled_scenes = set(subsample_scenes(all_scenes, SAMPLE_FRACTION, rng))
        sampled_stems = [s for s in all_stems if scene_of(s) in sampled_scenes]

        # Write listings
        (SAMPLE_DIR / f"sample_{split}_scenes.txt").write_text(
            "\n".join(sorted(sampled_scenes)) + "\n")
        (SAMPLE_DIR / f"sample_{split}_stems.txt").write_text(
            "\n".join(sorted(sampled_stems)) + "\n")

        # Write COCO subset
        n_imgs, n_anns = write_split(
            SPLITS_DIR / f"instances_{split}.json",
            set(sampled_stems),
            SAMPLE_DIR / f"instances_sample_{split}.json",
        )
        print(f"  {split:5s}: {len(sampled_scenes):4d} scenes  /  "
              f"{n_imgs:4d} images  /  {n_anns:6d} anns")

    print("\nDone. Outputs in:", SAMPLE_DIR)


if __name__ == "__main__":
    main()
