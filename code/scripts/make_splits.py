"""
Generate scene-aware 70/15/15 train/val/test splits without touching the
existing files. Outputs everything under `data/rainforests/splits/`.

Outputs:
    splits/
    ├── train_scenes.txt, val_scenes.txt, test_scenes.txt   # unique scene prefixes
    ├── train_stems.txt,  val_stems.txt,  test_stems.txt    # full file stems
    ├── instances_train.json, instances_val.json, instances_test.json
    ├── manifest.csv          # 1 row per stem: paths to every modality
    └── class_distribution.csv

Usage:
    python scripts/make_splits.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "rainforests"
SPLITS_DIR   = DATA_DIR / "splits"
EXISTING_TRAIN_JSON = DATA_DIR / "annotations" / "instances_train.json"
EXISTING_VAL_JSON   = DATA_DIR / "annotations" / "instances_val.json"
RGB_RESIZED_DIR     = DATA_DIR / "rgb_resized"

# Original raw-data folders (Windows local path). For manifest only.
RAW_ROOT = Path("D:/9. Thesis Plan/Analysis/Datasets/rainforests")
RAW_MODALITIES = {
    "rgb":              ("rgb",                   ".png"),
    "rgb_meta":         ("rgb",                   ".txt"),
    "rgb_resized":      ("rgb_resized",           ".png"),
    "depth_pfm":        ("depth_pfm",             ".pfm"),
    "instance_seg_png": ("instance_segmentation", ".png"),
    "instance_seg_txt": ("instance_segmentation", ".txt"),
    "semantic_seg":     ("semantic_segmentation", ".png"),
    "coco_per_image":   ("coco_annotation",       ".json"),
}

# --- Config --------------------------------------------------------------
SEED        = 42
SPLIT_RATIO = (0.70, 0.15, 0.15)   # train, val, test
SPLIT_NAMES = ("train", "val", "test")


def scene_of(stem: str) -> str:
    return stem.split("_")[0]


# ---------------------------------------------------------------------- #
# 1. Enumerate stems and group by scene
# ---------------------------------------------------------------------- #
all_stems = sorted(p.stem for p in RGB_RESIZED_DIR.glob("*.png"))
scene_to_stems: dict[str, list[str]] = defaultdict(list)
for s in all_stems:
    scene_to_stems[scene_of(s)].append(s)

scenes = sorted(scene_to_stems.keys())
print(f"Total stems: {len(all_stems)}  |  unique scenes: {len(scenes)}")

# ---------------------------------------------------------------------- #
# 2. Scene-level shuffle + 70/15/15 split
# ---------------------------------------------------------------------- #
rng = random.Random(SEED)
rng.shuffle(scenes)

n_total = len(scenes)
n_train = int(round(SPLIT_RATIO[0] * n_total))
n_val   = int(round(SPLIT_RATIO[1] * n_total))
n_test  = n_total - n_train - n_val
scene_splits = {
    "train": scenes[:n_train],
    "val":   scenes[n_train:n_train + n_val],
    "test":  scenes[n_train + n_val:],
}
stem_to_split: dict[str, str] = {}
stems_in_split: dict[str, list[str]] = {k: [] for k in SPLIT_NAMES}
for split, scn_list in scene_splits.items():
    for scn in scn_list:
        for st in scene_to_stems[scn]:
            stem_to_split[st] = split
            stems_in_split[split].append(st)

for k in SPLIT_NAMES:
    print(f"  {k:5s}: {len(scene_splits[k]):4d} scenes  /  "
          f"{len(stems_in_split[k]):4d} images")

# ---------------------------------------------------------------------- #
# 3. Write scene + stem listings
# ---------------------------------------------------------------------- #
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
for k in SPLIT_NAMES:
    (SPLITS_DIR / f"{k}_scenes.txt").write_text(
        "\n".join(sorted(scene_splits[k])) + "\n"
    )
    (SPLITS_DIR / f"{k}_stems.txt").write_text(
        "\n".join(sorted(stems_in_split[k])) + "\n"
    )

# ---------------------------------------------------------------------- #
# 4. Re-split the existing merged COCO JSONs by stem -> split
# ---------------------------------------------------------------------- #
def load_coco(p: Path) -> dict:
    print(f"  loading {p.name} ...")
    with p.open() as f:
        return json.load(f)


def merge_coco(a: dict, b: dict) -> dict:
    """Merge b into a; a's image_ids and ann_ids are kept, b's are offset."""
    out_imgs = list(a["images"])
    out_anns = list(a["annotations"])
    used_img_ids = {im["id"] for im in out_imgs}
    used_ann_ids = {an["id"] for an in out_anns}
    img_id_offset = max(used_img_ids) + 1 if used_img_ids else 0
    ann_id_offset = max(used_ann_ids) + 1 if used_ann_ids else 0

    remap_img = {}
    for im in b["images"]:
        new_id = im["id"] + img_id_offset if im["id"] in used_img_ids else im["id"]
        remap_img[im["id"]] = new_id
        out_imgs.append({**im, "id": new_id})
    for an in b["annotations"]:
        new_id = an["id"] + ann_id_offset if an["id"] in used_ann_ids else an["id"]
        out_anns.append({**an, "id": new_id, "image_id": remap_img[an["image_id"]]})
    return {"images": out_imgs, "annotations": out_anns,
            "categories": a.get("categories", b.get("categories", []))}


print("Re-building unified COCO from existing train/val ...")
merged = merge_coco(load_coco(EXISTING_TRAIN_JSON), load_coco(EXISTING_VAL_JSON))
print(f"  unified: {len(merged['images'])} images  {len(merged['annotations'])} anns")

# Map stem -> image record(s) in merged set
stem_to_imgs: dict[str, list[dict]] = defaultdict(list)
for im in merged["images"]:
    stem = Path(im["file_name"]).stem
    stem_to_imgs[stem].append(im)

# Group annotations by image_id
img_to_anns: dict[int, list[dict]] = defaultdict(list)
for an in merged["annotations"]:
    img_to_anns[an["image_id"]].append(an)

# Build per-split COCO with renumbered ids (0..N)
def build_split_coco(stems: list[str]) -> dict:
    images, annotations = [], []
    next_img_id = 1
    next_ann_id = 1
    for st in sorted(stems):
        for im in stem_to_imgs.get(st, []):
            new_im = {**im, "id": next_img_id}
            images.append(new_im)
            for an in img_to_anns.get(im["id"], []):
                annotations.append({
                    **an,
                    "id": next_ann_id,
                    "image_id": next_img_id,
                })
                next_ann_id += 1
            next_img_id += 1
    return {"images": images, "annotations": annotations,
            "categories": merged["categories"]}


for k in SPLIT_NAMES:
    coco = build_split_coco(stems_in_split[k])
    out = SPLITS_DIR / f"instances_{k}.json"
    with out.open("w") as f:
        json.dump(coco, f)
    print(f"  {out.name:24s}  images={len(coco['images']):5d}  "
          f"anns={len(coco['annotations']):7d}")

# ---------------------------------------------------------------------- #
# 5. Manifest CSV: every stem, every modality, with absolute path or ''
# ---------------------------------------------------------------------- #
print("Building manifest.csv ...")
manifest_rows = []
for st in sorted(all_stems):
    row = {"stem": st, "scene": scene_of(st),
           "split": stem_to_split.get(st, "")}
    for col, (folder, ext) in RAW_MODALITIES.items():
        f = RAW_ROOT / folder / f"{st}{ext}"
        row[col] = str(f) if f.exists() else ""
    manifest_rows.append(row)

manifest_path = SPLITS_DIR / "manifest.csv"
with manifest_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
    writer.writeheader()
    writer.writerows(manifest_rows)
print(f"  {manifest_path.name}  ({len(manifest_rows)} rows)")

# ---------------------------------------------------------------------- #
# 6. Class distribution per split
# ---------------------------------------------------------------------- #
print("Computing class distribution ...")
cat_id_to_name = {c["id"]: c["name"] for c in merged["categories"]}
img_id_to_split: dict[int, str] = {}
for k in SPLIT_NAMES:
    coco = json.loads((SPLITS_DIR / f"instances_{k}.json").read_text())
    for im in coco["images"]:
        img_id_to_split[(k, im["id"])] = k

dist: dict[tuple[str, str], int] = Counter()
for k in SPLIT_NAMES:
    coco = json.loads((SPLITS_DIR / f"instances_{k}.json").read_text())
    for an in coco["annotations"]:
        dist[(k, cat_id_to_name[an["category_id"]])] += 1

dist_path = SPLITS_DIR / "class_distribution.csv"
with dist_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class"] + list(SPLIT_NAMES) + ["total"])
    classes = [c["name"] for c in merged["categories"]]
    for cls in classes:
        row_counts = [dist[(k, cls)] for k in SPLIT_NAMES]
        w.writerow([cls] + row_counts + [sum(row_counts)])
    totals = [sum(dist[(k, c)] for c in classes) for k in SPLIT_NAMES]
    w.writerow(["TOTAL"] + totals + [sum(totals)])
print(f"  {dist_path.name}")

print("\nDone. All outputs under:", SPLITS_DIR)
