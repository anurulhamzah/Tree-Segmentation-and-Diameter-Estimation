#!/usr/bin/env python3
"""
Buat RF dan PL stratified splits yang konsisten dengan combined_stratified.

Strategi:
  - Combined stratified (sudah ada, 0 scene overlap, proportional species) → SOURCE OF TRUTH
  - RF_stratified  = extract RF images dari combined_stratified + remap category IDs
  - PL_stratified  = extract PL images dari combined_stratified

Hasilnya: PL_train ⊂ Combined_train, RF_train ⊂ Combined_train → konsisten untuk perbandingan.

Output:
  RF  → data/rainforests/annotation_inst/filtered_rle_f1000_repaired_v4_stratified/
  PL  → data/plantations/annotations_inst/filtered_rle_f1000_repaired_v4_stratified/
"""

import json
from pathlib import Path
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/scratch2/pr65/anur0018/tree_classification/data")

CB_STRAT  = DATA_ROOT / "combined/annotation_inst/filtered_rle_f1000_repaired_v4_stratified"
RF_OUT    = DATA_ROOT / "rainforests/annotation_inst/filtered_rle_f1000_repaired_v4_stratified"
PL_OUT    = DATA_ROOT / "plantations/annotations_inst/filtered_rle_f1000_repaired_v4_stratified"

# Category remapping: combined id → domain-specific id
# PL: ids 1-7 same in both combined and PL standalone → no remap needed
# RF: combined 8=AliiFig→1, 9=BangaloPalm→2, 10=Fern→3, 11=LeechVine→4, 12=RubberFig→5, 13=Umbrella→6
RF_CATMAP = {8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6}

RF_CATEGORIES = [
    {"id": 1, "name": "AliiFig",     "supercategory": "tree"},
    {"id": 2, "name": "BangaloPalm", "supercategory": "tree"},
    {"id": 3, "name": "Fern",        "supercategory": "tree"},
    {"id": 4, "name": "LeechVine",   "supercategory": "tree"},
    {"id": 5, "name": "RubberFig",   "supercategory": "tree"},
    {"id": 6, "name": "Umbrella",    "supercategory": "tree"},
]

PL_CATEGORIES = [
    {"id": 1, "name": "Apple",       "supercategory": "tree"},
    {"id": 2, "name": "Lemon",       "supercategory": "tree"},
    {"id": 3, "name": "Loquat",      "supercategory": "tree"},
    {"id": 4, "name": "Mango",       "supercategory": "tree"},
    {"id": 5, "name": "Orange",      "supercategory": "tree"},
    {"id": 6, "name": "Persimmon",   "supercategory": "tree"},
    {"id": 7, "name": "Pomegranate", "supercategory": "tree"},
]


def get_scene(fname: str) -> str:
    return fname.split("/")[-1].split("_")[0]


def filter_and_remap(cb_data: dict, domain: str) -> dict:
    """Extract domain images/annotations from combined split, remap IDs."""
    # Filter images by domain
    imgs = [img for img in cb_data["images"] if domain in img["file_name"].lower()]
    img_id_set = {img["id"] for img in imgs}

    # Build annotation index per image for fast lookup
    ann_by_img = {}
    for a in cb_data["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    # Collect and remap annotations
    if domain == "rainforest":
        valid_cb_cats = set(RF_CATMAP.keys())  # 8-13
        catmap = RF_CATMAP
        categories = RF_CATEGORIES
    else:  # plantation
        valid_cb_cats = set(range(1, 8))  # 1-7
        catmap = {i: i for i in range(1, 8)}
        categories = PL_CATEGORIES

    # Remap image IDs to sequential starting at 1
    old_to_new_img = {img["id"]: new_id for new_id, img in enumerate(imgs, start=1)}

    new_imgs = []
    for img in imgs:
        new_img = dict(img)
        new_img["id"] = old_to_new_img[img["id"]]
        new_imgs.append(new_img)

    new_anns = []
    ann_id = 1
    for old_img_id in img_id_set:
        for a in ann_by_img.get(old_img_id, []):
            if a["category_id"] not in valid_cb_cats:
                continue
            new_a = dict(a)
            new_a["id"] = ann_id
            new_a["image_id"] = old_to_new_img[old_img_id]
            new_a["category_id"] = catmap[a["category_id"]]
            new_anns.append(new_a)
            ann_id += 1

    info = cb_data.get("info", {"description": "stratified split derived from combined_repaired_v4_stratified"})
    licenses = cb_data.get("licenses", [])

    return {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": new_imgs,
        "annotations": new_anns,
    }


def print_stats(name: str, data: dict):
    cat_dict = {c["id"]: c["name"] for c in data["categories"]}
    counts = Counter(a["category_id"] for a in data["annotations"])
    print(f"  {name}: {len(data['images'])} imgs, {len(data['annotations'])} anns")
    for cid, cname in sorted(cat_dict.items()):
        n = counts.get(cid, 0)
        flag = " ⚠" if n < 50 else ""
        print(f"    {cname:<12} {n:>5}{flag}")


def verify_no_scene_overlap(splits: dict):
    scene_sets = {}
    for split_name, data in splits.items():
        scene_sets[split_name] = set(get_scene(img["file_name"]) for img in data["images"])
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        overlap = scene_sets[a] & scene_sets[b]
        status = "✓ 0 overlap" if not overlap else f"✗ {len(overlap)} overlap!"
        print(f"    Scene {a}∩{b}: {status}")


def main():
    RF_OUT.mkdir(parents=True, exist_ok=True)
    PL_OUT.mkdir(parents=True, exist_ok=True)

    print("Loading combined stratified splits...")
    cb_splits = {}
    for split in ("train", "val", "test"):
        with open(CB_STRAT / f"instances_{split}.json") as f:
            cb_splits[split] = json.load(f)
    print(f"  Loaded train/val/test: "
          f"{len(cb_splits['train']['images'])}/"
          f"{len(cb_splits['val']['images'])}/"
          f"{len(cb_splits['test']['images'])} images")

    # ── RF ───────────────────────────────────────────────────────────────────
    print("\n=== Creating RF stratified splits ===")
    rf_splits = {}
    for split in ("train", "val", "test"):
        rf_splits[split] = filter_and_remap(cb_splits[split], "rainforest")
        out_path = RF_OUT / f"instances_{split}.json"
        with open(out_path, "w") as f:
            json.dump(rf_splits[split], f)
        print_stats(split, rf_splits[split])
    print("  Scene overlap check:")
    verify_no_scene_overlap(rf_splits)

    # ── PL ───────────────────────────────────────────────────────────────────
    print("\n=== Creating PL stratified splits ===")
    pl_splits = {}
    for split in ("train", "val", "test"):
        pl_splits[split] = filter_and_remap(cb_splits[split], "plantation")
        out_path = PL_OUT / f"instances_{split}.json"
        with open(out_path, "w") as f:
            json.dump(pl_splits[split], f)
        print_stats(split, pl_splits[split])
    print("  Scene overlap check:")
    verify_no_scene_overlap(pl_splits)

    print("\n=== Summary ===")
    print(f"RF stratified → {RF_OUT}")
    print(f"PL stratified → {PL_OUT}")
    print("Combined stratified unchanged (source of truth).")
    print("\nDataset names yang akan ter-register:")
    print("  RF:       rainforests_inst_rle_f1000_repaired_v4_stratified_{train,val,test}")
    print("  PL:       plantations_inst_rle_f1000_repaired_v4_stratified_{train,val,test}")
    print("  Combined: combined_inst_rle_f1000_repaired_v4_stratified_{train,val,test} (sudah ada)")


if __name__ == "__main__":
    main()
