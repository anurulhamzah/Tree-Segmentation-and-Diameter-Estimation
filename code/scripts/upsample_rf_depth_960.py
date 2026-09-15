#!/usr/bin/env python
"""Upsample rainforest depth PFM 480x270 -> 960x540 (2x nearest) untuk training @960.

Model RGB-D butuh depth ber-resolusi sama dgn RGB (mapper menerapkan ResizeTransform
gambar ke depth; dimensi harus cocok). rgb_960 = 960x540 tapi depth_pfm hanya 480x270.
Nearest 2x (bukan bilinear) dipilih agar diskontinuitas depth di tepi pohon/latar tidak
diinterpolasi jadi nilai antara yang palsu.

Output: data/rainforests/depth_960/<stem>.pfm  (idempotent — skip yg sudah ada).
"""
import glob
import os
import sys

import numpy as np

SRC = "/scratch2/pr65/anur0018/tree_classification/data/rainforests/depth_pfm"
DST = "/scratch2/pr65/anur0018/tree_classification/data/rainforests/depth_960"


def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").strip()
        assert header in ("PF", "Pf"), f"Not a PFM file: {path}"
        color = header == "PF"
        w, h = (int(v) for v in f.readline().decode("latin-1").split())
        scale = float(f.readline().decode("latin-1").strip())
        endian = "<f4" if scale < 0 else ">f4"
        n = w * h * (3 if color else 1)
        data = np.frombuffer(f.read(n * 4), dtype=endian)
        data = data.reshape((h, w, 3) if color else (h, w))
    return np.flipud(data).copy(), color


def write_pfm(path, data, color):
    # invers dari read_pfm: simpan flipud(data) little-endian, scale=-1.0
    data = np.flipud(data).astype("<f4")
    with open(path, "wb") as f:
        f.write((b"PF\n" if color else b"Pf\n"))
        h, w = data.shape[:2]
        f.write(f"{w} {h}\n".encode("latin-1"))
        f.write(b"-1.0\n")
        f.write(data.tobytes())


def main():
    os.makedirs(DST, exist_ok=True)
    src_files = sorted(glob.glob(os.path.join(SRC, "*.pfm")))
    print(f"[upsample] {len(src_files)} file depth di {SRC}")
    done = skipped = 0
    for i, sp in enumerate(src_files):
        stem = os.path.splitext(os.path.basename(sp))[0]
        dp = os.path.join(DST, stem + ".pfm")
        if os.path.exists(dp):
            skipped += 1
            continue
        arr, color = read_pfm(sp)
        assert not color, f"depth harus grayscale: {sp}"
        assert arr.shape == (270, 480), f"dim depth tak terduga {arr.shape}: {sp}"
        up = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)  # -> (540, 960) nearest
        assert up.shape == (540, 960)
        write_pfm(dp, up, color=False)
        done += 1
        if (i + 1) % 500 == 0:
            print(f"[upsample] {i + 1}/{len(src_files)} (baru={done} skip={skipped})")
    print(f"[upsample] SELESAI. baru={done} skip={skipped} total={len(src_files)} -> {DST}")

    # verifikasi roundtrip 1 file
    if src_files:
        chk, _ = read_pfm(os.path.join(DST, os.path.splitext(os.path.basename(src_files[0]))[0] + ".pfm"))
        print(f"[verify] contoh output shape={chk.shape} (harus (540, 960))  min={chk.min():.2f} max={chk.max():.2f}")


if __name__ == "__main__":
    sys.exit(main())
