#!/usr/bin/env python3
"""ablasi_depth_tol.py — jalankan dump_dbh_per_instance.py dengan depth_tol berbeda,
via monkeypatch default kwarg _find_row_1p3m di trunk_roi_dbh_head_hybrid.

Uji kausal: apakah menaikkan depth_tol (toleransi jarak kolom dari titik terdekat saat
mengisolasi batang di baris breast-height) mengurangi bias diameter pada batang besar
(radius > 0,3 m), sesuai dugaan dari analisis observasional (rasio px_obs/px_expected
turun tajam persis di radius 30cm = depth_tol default).

Tidak mengubah kode inti — patch dilakukan di memori proses ini saja, sebelum
dump_dbh_per_instance.main() dipanggil. Output ditulis dengan --tag terpisah, TIDAK
menimpa dump kanonik yang dipakai naskah.

Jalankan:
    python scripts/ablasi_depth_tol.py --depth-tol 0.6
"""
import argparse, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

ap = argparse.ArgumentParser()
ap.add_argument("--depth-tol", type=float, required=True)
ap.add_argument("--output-dir", default="FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k_cap30_detach")
ap.add_argument("--species-subset", default="1,3,4,5,6,7,8,9,10,12,13")
ap.add_argument("--max-depth", default="30.0")
ap.add_argument("--splits", default="test")
args, _ = ap.parse_known_args()

import trunk_roi_dbh_head_hybrid as H
_orig = H._find_row_1p3m
def _patched(mask, depth_raw, cy, fy, h_cam=H._H_CAM, tol_wh=0.5, depth_tol=args.depth_tol):
    return _orig(mask, depth_raw, cy, fy, h_cam=h_cam, tol_wh=tol_wh, depth_tol=depth_tol)
H._find_row_1p3m = _patched
print(f"[ablasi] depth_tol default dipatch ke {args.depth_tol} m (semula 0.3 m)", flush=True)

import dump_dbh_per_instance as D
tag = f"_depthtol{str(args.depth_tol).replace('.', 'p')}"
sys.argv = ["dump_dbh_per_instance.py",
            "--output-dir", args.output_dir,
            "--species-subset", args.species_subset,
            "--max-depth", args.max_depth,
            "--splits", args.splits,
            "--tag", tag]
D.main()
