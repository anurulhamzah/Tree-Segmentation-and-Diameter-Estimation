#!/usr/bin/env python3
"""Uji ketergantungan model final pada channel depth, dengan menurunkan kualitas depth saat
inferensi lalu mengukur AP50 segmentasi.

Latar: naskah mengklaim depth menyumbang 14,71 poin AP50, tetapi klaim itu diukur dari dua run
training terpisah. Yang belum pernah diuji: apakah model YANG SUDAH DILATIH benar-benar membaca
geometri depth saat inferensi, atau depth hanya berperan sebagai regularisasi selama training.
Kalau merusak depth saat inferensi tidak mengubah apa pun, klaim kontribusi depth perlu dibaca
ulang.

Cara kerja: monkeypatch `_normalize_depth` di mapper MaskDINO, jadi sumber bersama tidak disentuh.
Weights dan pipeline lain identik di semua kondisi; satu-satunya yang berubah adalah isi channel ke-4.

Kondisi:
  intact     kontrol, harus mereproduksi angka yang dilaporkan naskah
  noise05    gaussian noise sigma 0,05 pada skala depth ternormalisasi [0,1]
  noise15    sigma 0,15
  shuffle    depth diacak antar baris, statistiknya sama tetapi geometrinya rusak
  zero       channel depth dinolkan, hadir tetapi tanpa informasi

Jalankan:
    python scripts/eval_depth_degradation.py --output-dir <nama_run> [--dataset <nama>]
"""
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
OUTPUT_ROOT = PROJECT_ROOT.parent / "maskdino_output"
sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from detectron2.checkpoint import DetectionCheckpointer          # noqa: E402
from detectron2.config import get_cfg                            # noqa: E402
from detectron2.data import DatasetCatalog, build_detection_test_loader  # noqa: E402
from detectron2.evaluation import COCOEvaluator, inference_on_dataset    # noqa: E402
from detectron2.modeling import build_model                      # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config       # noqa: E402
from maskdino import add_maskdino_config                         # noqa: E402
from maskdino.data.dataset_mappers import coco_instance_dbh_dataset_mapper as M  # noqa: E402
from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import (      # noqa: E402
    COCOInstanceDBHDatasetMapper)
from register_combined import register_all_combined              # noqa: E402

RNG = np.random.default_rng(0)
_ASLI = M._normalize_depth


def pasang_degradasi(mode: str):
    """Ganti _normalize_depth dengan versi yang merusak depth sesudah normalisasi."""
    def diganti(depth: np.ndarray) -> np.ndarray:
        d = _ASLI(depth)                       # float32 (H, W) di [0, 1]
        if mode == "intact":
            return d
        if mode == "zero":
            return np.zeros_like(d)
        if mode == "shuffle":                  # statistik sama, geometri hancur
            idx = RNG.permutation(d.shape[0])
            return d[idx]
        sigma = {"noise05": 0.05, "noise15": 0.15}[mode]
        return np.clip(d + RNG.normal(0, sigma, d.shape).astype(np.float32), 0, 1)
    M._normalize_depth = diganti


def bangun(output_dir: Path, device: str):
    cfg = get_cfg()
    add_deeplab_config(cfg); add_maskdino_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(str(output_dir / "config.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()
    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(output_dir / "model_final.pth"))
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset", default="combined_inst_rle_f1000_repaired_v4_stratified_val")
    ap.add_argument("--modes", default="intact,noise05,noise15,shuffle,zero")
    args = ap.parse_args()

    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_combined(os.environ["COMBINED_ROOT"])
    out_dir = OUTPUT_ROOT / args.output_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hasil = {}
    for mode in args.modes.split(","):
        pasang_degradasi(mode)
        model, cfg = bangun(out_dir, device)          # dibangun ulang tiap mode, weights sama
        kerja = out_dir / f"eval_depth_{mode}"
        ev = COCOEvaluator(args.dataset, output_dir=str(kerja))
        # WAJIB mapper DBH: loader default tidak memuat depth, model lalu menerima 3 channel
        # sementara PIXEL_MEAN-nya 4, dan inferensi gagal di normalisasi.
        mapper = COCOInstanceDBHDatasetMapper(cfg, is_train=False)
        loader = build_detection_test_loader(cfg, args.dataset, mapper=mapper)
        with contextlib.redirect_stdout(io.StringIO()):
            r = inference_on_dataset(model, loader, ev)
        segm = r["segm"]
        hasil[mode] = {k: round(float(segm[k]), 2) for k in ("AP", "AP50", "AP75")}
        print(f"  {mode:9s} AP50={hasil[mode]['AP50']:6.2f}  AP={hasil[mode]['AP']:6.2f}  "
              f"AP75={hasil[mode]['AP75']:6.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    M._normalize_depth = _ASLI
    dasar = hasil.get("intact", {}).get("AP50")
    if dasar:
        print("\n  selisih AP50 terhadap depth utuh:")
        for m, v in hasil.items():
            if m != "intact":
                print(f"    {m:9s} {v['AP50'] - dasar:+6.2f} poin")
    simpan = PROJECT_ROOT / "reports" / f"depth_degradation_{args.output_dir}.json"
    simpan.parent.mkdir(exist_ok=True)
    json.dump({"run": args.output_dir, "dataset": args.dataset, "hasil": hasil},
              open(simpan, "w"), indent=2)
    print(f"\n  tersimpan: {simpan}")


if __name__ == "__main__":
    main()
