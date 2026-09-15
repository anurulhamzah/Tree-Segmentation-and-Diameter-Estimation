"""
Membuat COCO JSON bersih untuk training DBH head — versi 3 (universe 11 spesies).

Perbedaan dari v2 (6 spesies): Level 0 seleksi spesies diperluas dari asumsi lama "1-7 preset
GT perkebunan" ke kriteria yang diverifikasi langsung dari data (lihat sesi DBH Sandbox,
12 Jul 2026): hanya spesies dgn blank-DBH (dbh=null) >50% yang dikecualikan permanen —
Lemon (86,9% blank) dan LeechVine (66,4% blank, liana). 5 spesies plantation yang dulu
dikecualikan tanpa diuji (Apple/Loquat/Orange/Persimmon/Pomegranate) sekarang MASUK.

Level 1/2 params (RF_DBH_MAX_CM, MIN_DEPTH, MAX_DEPTH, TOL_WH, MIN_TRUNK_PX) TIDAK berubah
dari v2 -- dipilih SAMA PERSIS dgn kombinasi baseline yg menang di DBH Sandbox
(scripts/dbh_sandbox/, Fase C, mean_per_species_R2=0.136 test set) supaya Fase D
(sanity-check fidelitas via pipeline produksi asli, LSJ augmentation nyata) bisa
dibandingkan apple-to-apple dgn hasil cache-based sandbox.

Level 0 — Seleksi spesies:
  Exclude: 2 (Lemon, blank-DBH 86.9%), 11 (LeechVine, blank-DBH 66.4%, liana)
  Include: 9 spesies lain (1 Apple, 3 Loquat, 4 Mango, 5 Orange, 6 Persimmon,
           7 Pomegranate, 8 AliiFig, 9 BangaloPalm, 10 Fern, 12 RubberFig, 13 Umbrella)
           = 11 spesies total

Level 1 — GT-based: exclude DBH=0, exclude RubberFig DBH>=150cm (bimodal)
Level 2 — Geometric feasibility: depth di row 1.3m dlm [1,20)m, |world_height-1.3|<=0.5m, trunk_px>=5

Output: instances_{train,val,test}.json di OUT_DIR
"""

import sys, os, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from pycocotools import mask as maskUtils
except ImportError:
    print("ERROR: pycocotools not found — jalankan di ml-env atau maskdino env")
    sys.exit(1)

# ── Camera constants (480×270 original) ──────────────────────────────────────
_ORIG_H  = 270.0
_ORIG_W  = 480.0
_ORIG_CY = 135.0
_ORIG_FY = 240.0
_H_CAM   = 2.0

# ── Level 0: spesies yang dieksklusikan (HANYA blank-DBH >50%, lihat docstring) ──
EXCLUDE_SPECIES = {2, 11}

# ── Level 1 params (sama dgn v2 / baseline sandbox) ──────────────────────────
RF_CAT_ID     = 12
RF_DBH_MAX_CM = 150.0
MIN_DBH_CM    = 0.0

# ── Level 2 params (sama dgn v2 / baseline sandbox) ──────────────────────────
MIN_DEPTH    = 1.0
MAX_DEPTH    = 20.0
TOL_WH       = 0.5
MIN_TRUNK_PX = 5


def load_pfm(path: str) -> np.ndarray:
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
        if endian == "<":
            data = data[::-1, :]
    return data


def find_row_1p3m(mask: np.ndarray, depth_raw: np.ndarray,
                  cy: float, fy: float, fx: float,
                  h_cam: float = _H_CAM,
                  tol_wh: float = TOL_WH,
                  depth_tol: float = 0.3):
    rows_with_mask = np.where(mask.sum(axis=1) > 0)[0]
    if len(rows_with_mask) < 3:
        return None

    best_r, best_wh = None, None
    for r in rows_with_mask:
        cols  = np.where(mask[r])[0]
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

    d_min      = row_deps.min()
    trunk_m    = row_deps <= d_min + depth_tol
    trunk_cols = row_cols[trunk_m]
    d_trunk    = float(row_deps[trunk_m].mean())
    if len(trunk_cols) < 2:
        return None

    return len(trunk_cols), d_trunk, best_wh


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

    l1_pass = []
    n_excl_species = n_excl_rf = n_excl_zero = 0
    for ann in data["annotations"]:
        dbh = ann.get("dbh", None)
        if dbh is None or dbh <= MIN_DBH_CM:
            n_excl_zero += 1
            continue
        cid = ann["category_id"]
        if cid in EXCLUDE_SPECIES:
            n_excl_species += 1
            continue
        if cid == RF_CAT_ID and dbh >= RF_DBH_MAX_CM:
            n_excl_rf += 1
            continue
        l1_pass.append(ann)

    total_ann = len(data["annotations"])
    excl_sp_names = sorted(cat_map.get(c, str(c)) for c in EXCLUDE_SPECIES)
    print(f"\nLevel 0+1 (GT-based):")
    print(f"  Total ann:              {total_ann:>7}")
    print(f"  Excl dbh=0:             {n_excl_zero:>7}")
    print(f"  Excl species:           {n_excl_species:>7}  ({', '.join(excl_sp_names)})")
    print(f"  Excl RubberFig>=150cm:  {n_excl_rf:>7}")
    print(f"  → Level 1 pass:         {len(l1_pass):>7}")

    by_img = defaultdict(list)
    for ann in l1_pass:
        by_img[ann["image_id"]].append(ann)

    l2_pass = []
    n_no_depth = n_geom_fail = n_depth_range = n_wh_fail = n_px_fail = 0
    n_imgs_processed = 0

    for img_id, anns in by_img.items():
        img_info = img_map[img_id]
        W_img = img_info["width"]
        H_img = img_info["height"]
        scale = H_img / _ORIG_H
        cy    = _ORIG_CY * scale
        fy    = _ORIG_FY * scale
        fx    = _ORIG_FY * scale

        dfn = depth_path_for_image(img_info["file_name"], depth_dir)
        if not os.path.exists(dfn):
            n_no_depth += len(anns)
            continue

        try:
            depth_raw = load_pfm(dfn)
            if depth_raw.shape != (H_img, W_img):
                import cv2
                depth_raw = cv2.resize(depth_raw, (W_img, H_img),
                                       interpolation=cv2.INTER_NEAREST)
        except Exception:
            n_no_depth += len(anns)
            continue

        n_imgs_processed += 1
        for ann in anns:
            try:
                rle = ann["segmentation"]
                if isinstance(rle, list):
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

            if not (MIN_DEPTH <= d_trunk < MAX_DEPTH):
                n_depth_range += 1
                continue
            if trunk_px < MIN_TRUNK_PX:
                n_px_fail += 1
                continue

            l2_pass.append(ann)

        if n_imgs_processed % 300 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_imgs_processed}/{len(by_img)} imgs, {elapsed:.0f}s]  "
                  f"pass so far: {len(l2_pass)}")

    print(f"\nLevel 2 (Geometric feasibility, max_depth={MAX_DEPTH}m):")
    print(f"  No depth map:           {n_no_depth:>7}")
    print(f"  wh fail (>±{TOL_WH}m):      {n_wh_fail:>7}")
    print(f"  Depth out of range:     {n_depth_range:>7}")
    print(f"  trunk_px < {MIN_TRUNK_PX}:           {n_px_fail:>7}")
    print(f"  Geom decode error:      {n_geom_fail:>7}")
    pct = len(l2_pass) / len(l1_pass) * 100 if l1_pass else 0
    print(f"  → Level 2 pass:         {len(l2_pass):>7}  ({pct:.1f}% of L1)")

    by_cat = defaultdict(list)
    for ann in l2_pass:
        by_cat[ann["category_id"]].append(ann["dbh"])

    print(f"\n  Per-spesies (clean):")
    for cid in sorted(by_cat.keys()):
        vals = np.array(by_cat[cid])
        print(f"    {cat_map.get(cid, cid):<14}  N={len(vals):>5}  "
              f"DBH=[{vals.min():.0f},{vals.max():.0f}]cm  "
              f"med={np.median(vals):.0f}  std={vals.std():.1f}")

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
    BASE      = "/scratch2/pr65/anur0018/tree_classification/data/combined"
    ANN_DIR   = f"{BASE}/annotation_inst/filtered_rle_f1000_repaired_v4_stratified"
    DEPTH_DIR = f"{BASE}/depth_pfm"
    OUT_DIR   = f"{BASE}/annotation_inst/filtered_rle_f1000_repaired_v4_stratified_dbh_clean_v3_universe11"

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")
    print(f"EXCLUDE    = {sorted(EXCLUDE_SPECIES)}  (Lemon, LeechVine -- blank-DBH >50% only)")

    n_train = process_split(
        json_path  = f"{ANN_DIR}/instances_train.json",
        depth_dir  = DEPTH_DIR,
        out_path   = f"{OUT_DIR}/instances_train.json",
        split_name = "TRAIN",
    )
    n_val = process_split(
        json_path  = f"{ANN_DIR}/instances_val.json",
        depth_dir  = DEPTH_DIR,
        out_path   = f"{OUT_DIR}/instances_val.json",
        split_name = "VAL",
    )
    n_test = process_split(
        json_path  = f"{ANN_DIR}/instances_test.json",
        depth_dir  = DEPTH_DIR,
        out_path   = f"{OUT_DIR}/instances_test.json",
        split_name = "TEST",
    )

    print(f"\n{'='*60}")
    print(f"DONE — clean JSONs v3 (universe11) siap untuk training DBH head V7")
    print(f"  Train: {n_train}  Val: {n_val}  Test: {n_test}  Total: {n_train+n_val+n_test}")
    print(f"{'='*60}")
