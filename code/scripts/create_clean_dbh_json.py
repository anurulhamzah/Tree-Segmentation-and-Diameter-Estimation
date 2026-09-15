"""
Membuat COCO JSON bersih untuk training DBH head.
Dua level pembersihan:

  Level 1 — GT-based (tanpa load file):
    - Exclude plantation species (DBH bukan diukur di 1.3m, preset asset Unreal)
    - Exclude RubberFig DBH > 150cm (bimodal gap 75–250cm, group besar tidak terukur)
    - Exclude DBH = 0

  Level 2 — Geometric feasibility (load PFM + decode RLE):
    - depth di row 1.3m ∈ [1, 10m)
    - |world_height - 1.3| ≤ 0.5m
    - trunk_px ≥ 5
    - trunk_px / W_img < 0.5  (trunk tidak terpotong image)

Output: instances_train_dbh_clean.json, instances_val_dbh_clean.json
"""

import sys, os, json, math, struct, time
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from pycocotools import mask as maskUtils
except ImportError:
    print("ERROR: pycocotools not found")
    sys.exit(1)

# ── Constants (kamera original 480×270) ──────────────────────────────────────
_ORIG_H   = 270.0
_ORIG_W   = 480.0
_ORIG_CY  = 135.0
_ORIG_FY  = 240.0
_H_CAM    = 2.0

# ── Level 1 filter params ────────────────────────────────────────────────────
PLANTATION_SPECIES = {1, 2, 3, 5, 6, 7}   # Apple,Lemon,Loquat,Orange,Persimmon,Pomegranate
RF_CAT_ID          = 12                    # RubberFig
RF_DBH_MAX_CM      = 150.0                 # batas atas DBH RubberFig (bimodal gap 75–250cm)
MIN_DBH_CM         = 0.0                   # annotasi dbh=0 dikesampingkan

# ── Level 2 filter params ────────────────────────────────────────────────────
MIN_DEPTH      = 1.0    # meter
MAX_DEPTH      = 10.0   # meter
TOL_WH         = 0.5    # |world_height - 1.3| ≤ TOL_WH
MIN_TRUNK_PX   = 5      # pixel
MAX_TRUNK_FRAC = 0.5    # trunk_px / W_img < 0.5


def load_pfm(path: str) -> np.ndarray:
    """Load PFM depth file → float32 array (H, W) dalam meter."""
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").strip()
        assert header in ("PF", "Pf"), f"Bukan PFM: {header}"
        dims   = f.readline().decode("latin-1").strip()
        scale  = float(f.readline().decode("latin-1").strip())
        W, H   = map(int, dims.split())
        endian = "<" if scale < 0 else ">"
        scale  = abs(scale)
        data   = np.frombuffer(f.read(), dtype=np.float32).copy()
        data   = data.reshape((H, W))
        if scale < 0 or endian == "<":
            data = data[::-1, :]   # PFM tersimpan bottom-up
    return data


def find_row_1p3m(mask: np.ndarray, depth_raw: np.ndarray,
                  cy: float, fy: float, fx: float,
                  h_cam: float = _H_CAM,
                  tol_wh: float = TOL_WH,
                  depth_tol: float = 0.3):
    """
    Cari row gambar yang paling dekat ke world height = 1.3m.
    Returns: (trunk_px, d_trunk, world_h) atau None jika gagal.
    """
    rows_with_mask = np.where(mask.sum(axis=1) > 0)[0]
    if len(rows_with_mask) < 3:
        return None

    best_r, best_wh = None, None
    for r in rows_with_mask:
        cols = np.where(mask[r])[0]
        if len(cols) < 2:
            continue
        d_vals = depth_raw[r][cols]
        valid  = (d_vals > 0.1) & (d_vals < 200.0)
        if valid.sum() < 2:
            continue
        d  = float(np.median(d_vals[valid]))
        wh = h_cam - (r - cy) * d / fy
        if best_r is None or abs(wh - 1.3) < abs(best_wh - 1.3):
            best_r, best_wh = int(r), wh

    if best_r is None or abs(best_wh - 1.3) > tol_wh:
        return None

    row_cols = np.where(mask[best_r])[0]
    row_deps = depth_raw[best_r][row_cols]
    valid    = (row_deps > 0.1) & (row_deps < 200.0)
    row_cols = row_cols[valid]
    row_deps = row_deps[valid]
    if len(row_cols) < 2:
        return None

    d_min    = row_deps.min()
    trunk_m  = row_deps <= d_min + depth_tol
    trunk_cols = row_cols[trunk_m]
    d_trunk  = float(row_deps[trunk_m].mean())
    if len(trunk_cols) < 2:
        return None

    trunk_px = len(trunk_cols)
    return trunk_px, d_trunk, best_wh


def depth_path_for_image(img_file_name: str, depth_dir: str) -> str:
    stem = Path(img_file_name).stem
    return str(Path(depth_dir) / f"{stem}.pfm")


def process_split(json_path: str, depth_dir: str, out_path: str, split_name: str):
    print(f"\n{'='*60}")
    print(f"Processing: {split_name}  ({json_path})")
    print(f"{'='*60}")
    t0 = time.time()

    with open(json_path) as f:
        data = json.load(f)

    img_map = {img["id"]: img for img in data["images"]}
    cat_map = {c["id"]: c["name"] for c in data["categories"]}

    # ── Level 1 filter ────────────────────────────────────────────────────────
    l1_pass = []
    n_excl_species = 0; n_excl_rf = 0; n_excl_zero = 0
    for ann in data["annotations"]:
        dbh = ann.get("dbh", None)
        if dbh is None or dbh <= MIN_DBH_CM:
            n_excl_zero += 1
            continue
        cid = ann["category_id"]
        if cid in PLANTATION_SPECIES:
            n_excl_species += 1
            continue
        if cid == RF_CAT_ID and dbh > RF_DBH_MAX_CM:
            n_excl_rf += 1
            continue
        l1_pass.append(ann)

    total_ann = len(data["annotations"])
    print(f"\nLevel 1 (GT-based):")
    print(f"  Total ann:              {total_ann:>7}")
    print(f"  Excl dbh=0:             {n_excl_zero:>7}")
    print(f"  Excl plantation:        {n_excl_species:>7}  {','.join(cat_map.get(c,'?') for c in sorted(PLANTATION_SPECIES))}")
    print(f"  Excl RubberFig>150cm:   {n_excl_rf:>7}")
    print(f"  → Level 1 pass:         {len(l1_pass):>7}")

    # ── Level 2 filter (geometric feasibility) ────────────────────────────────
    # Group annotations by image_id to load each PFM once
    by_img = defaultdict(list)
    for ann in l1_pass:
        by_img[ann["image_id"]].append(ann)

    l2_pass = []
    n_no_depth = 0; n_geom_fail = 0; n_depth_range = 0
    n_wh_fail = 0; n_px_fail = 0; n_clip_fail = 0
    n_imgs_processed = 0

    for img_id, anns in by_img.items():
        img_info = img_map[img_id]
        W_img = img_info["width"]
        H_img = img_info["height"]
        scale = H_img / _ORIG_H
        cy    = _ORIG_CY * scale
        fy    = _ORIG_FY * scale
        fx    = _ORIG_FY * scale  # square pixels

        dfn = depth_path_for_image(img_info["file_name"], depth_dir)
        if not os.path.exists(dfn):
            n_no_depth += len(anns)
            continue

        try:
            depth_raw = load_pfm(dfn)
            # Scale depth to match image size jika perlu
            if depth_raw.shape != (H_img, W_img):
                import cv2
                depth_raw = cv2.resize(depth_raw, (W_img, H_img),
                                       interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            n_no_depth += len(anns)
            continue

        n_imgs_processed += 1
        for ann in anns:
            # Decode RLE mask
            try:
                rle = ann["segmentation"]
                if isinstance(rle, list):
                    # Polygon → RLE
                    rle = maskUtils.frPyObjects(rle, H_img, W_img)
                    rle = maskUtils.merge(rle)
                mask_bin = maskUtils.decode(rle).astype(bool)
            except Exception:
                n_geom_fail += 1
                continue

            result = find_row_1p3m(mask_bin, depth_raw, cy, fy, fx)
            if result is None:
                n_wh_fail += 1
                continue

            trunk_px, d_trunk, world_h = result

            # Depth range
            if not (MIN_DEPTH <= d_trunk < MAX_DEPTH):
                n_depth_range += 1
                continue
            # Trunk pixel size
            if trunk_px < MIN_TRUNK_PX:
                n_px_fail += 1
                continue
            # Trunk tidak terpotong gambar (< 50% lebar image)
            if trunk_px / W_img >= MAX_TRUNK_FRAC:
                n_clip_fail += 1
                continue

            l2_pass.append(ann)

        if n_imgs_processed % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_imgs_processed}/{len(by_img)} imgs, {elapsed:.0f}s]  "
                  f"pass so far: {len(l2_pass)}")

    print(f"\nLevel 2 (Geometric feasibility):")
    print(f"  No depth map:           {n_no_depth:>7}  (plantation images excluded)")
    print(f"  wh fail (>±0.5m):       {n_wh_fail:>7}")
    print(f"  Depth out of range:     {n_depth_range:>7}")
    print(f"  trunk_px < 5:           {n_px_fail:>7}")
    print(f"  trunk clips image:      {n_clip_fail:>7}")
    print(f"  Geom decode error:      {n_geom_fail:>7}")
    print(f"  → Level 2 pass:         {len(l2_pass):>7}  ({len(l2_pass)/len(l1_pass)*100:.1f}% of L1)")

    # Per-spesies breakdown
    by_cat = defaultdict(list)
    for ann in l2_pass:
        by_cat[ann["category_id"]].append(ann["dbh"])

    print(f"\n  Per-spesies (clean):")
    for cid in sorted(by_cat.keys()):
        vals = np.array(by_cat[cid])
        print(f"    {cat_map.get(cid,cid):<14}  N={len(vals):>5}  "
              f"DBH=[{vals.min():.0f},{vals.max():.0f}]cm  std={vals.std():.1f}")

    # ── Build output JSON ─────────────────────────────────────────────────────
    keep_img_ids = {ann["image_id"] for ann in l2_pass}
    out_data = {
        "info":        data.get("info", {}),
        "licenses":    data.get("licenses", []),
        "categories":  data["categories"],
        "images":      [img for img in data["images"] if img["id"] in keep_img_ids],
        "annotations": l2_pass,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f)

    elapsed = time.time() - t0
    print(f"\n  Images in clean JSON: {len(out_data['images'])}")
    print(f"  Annotations:          {len(l2_pass)}")
    print(f"  Saved: {out_path}  ({elapsed:.1f}s)")
    return len(l2_pass)


if __name__ == "__main__":
    BASE = "/scratch2/pr65/anur0018/tree_classification/data/combined"
    ANN_DIR  = f"{BASE}/annotation_inst/filtered_rle_f1000_repaired_v4_stratified"
    DEPTH_DIR = f"{BASE}/depth_pfm"
    OUT_DIR  = f"{BASE}/annotation_inst/filtered_rle_f1000_repaired_v4_stratified_dbh_clean"

    os.makedirs(OUT_DIR, exist_ok=True)

    process_split(
        json_path  = f"{ANN_DIR}/instances_train.json",
        depth_dir  = DEPTH_DIR,
        out_path   = f"{OUT_DIR}/instances_train.json",
        split_name = "TRAIN",
    )
    process_split(
        json_path  = f"{ANN_DIR}/instances_val.json",
        depth_dir  = DEPTH_DIR,
        out_path   = f"{OUT_DIR}/instances_val.json",
        split_name = "VAL",
    )

    print("\nDone. Clean JSONs ready untuk training DBH head.")
