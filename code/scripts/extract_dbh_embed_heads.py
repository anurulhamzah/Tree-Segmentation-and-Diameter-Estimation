#!/usr/bin/env python3
"""Ekstrak head dbh_embed dari SEMUA checkpoint model DBH FROZEN, verifikasi, lalu (opsional)
hapus checkpoint besar.

Berlaku HANYA untuk model dbh_embed dgn backbone+decoder DIBEKUKAN (mis. dbh_frozen_plainext_20k):
di sana tiap checkpoint 5,3G = (backbone+decoder identik --base) + head dbh_embed 0,14MB. Jadi
menyimpan head mungil tiap iterasi (arsip ~1,3MB) = menjaga SEMUA kandidat sekaligus membebaskan
belasan GB. Rekonstruksi full-model: load --base, lalu model.load_state_dict(head, strict=False).

JANGAN pakai untuk model JOINT (backbone ikut berubah tiap iter → head tak bisa dipisah dari
backbone-nya). Untuk joint pakai eval_dbh_percheckpoint.py lalu keep best+final.

Script memverifikasi (a) tiap checkpoint punya 4 key dbh_embed, (b) backbone identik --base,
(c) head hasil ekstrak cocok bit-per-bit — dan HANYA menghapus jika semua lolos.

Jalankan:
  python scripts/extract_dbh_embed_heads.py --output-dir FocalNet_L_dbh_frozen_plainext_20k \
      --base FocalNet_L_combined_rgbd_scratch_v4_stratified_plain_ext20k/model_final.pth [--apply]
"""
import argparse, os, sys
from pathlib import Path
import torch

OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent / "maskdino_output"


def load_sd(path):
    c = torch.load(path, map_location="cpu")
    return c["model"] if "model" in c else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base", required=True, help="checkpoint backbone frozen (relatif ke maskdino_output/)")
    ap.add_argument("--apply", action="store_true", help="hapus checkpoint setelah verifikasi lolos")
    args = ap.parse_args()

    D = OUTPUT_ROOT / args.output_dir
    base_path = OUTPUT_ROOT / args.base
    arch = D / "dbh_embed_heads_all.pth"
    ckpts = sorted(f for f in os.listdir(D) if f.startswith("model_") and f.endswith(".pth"))
    print(f"checkpoints: {ckpts}")

    sb = load_sd(base_path)
    heads = {}
    for f in ckpts:
        sd = load_sd(D / f)
        hk = [k for k in sd if "dbh_embed" in k]
        assert len(hk) == 4, f"{f}: dbh_embed keys={len(hk)} (harusnya 4) — model ini frozen dbh_embed?"
        # verifikasi backbone identik base (sanity: pastikan benar-benar frozen)
        bk = [k for k in sd if k.startswith("backbone") and k in sb][:100]
        diff = sum(1 for k in bk if not (sd[k].shape == sb[k].shape and torch.equal(sd[k], sb[k])))
        assert diff == 0, f"{f}: backbone BEDA dari --base ({diff}/{len(bk)}) — BUKAN frozen! Batalkan."
        heads[f] = {k: sd[k].clone() for k in hk}
        print(f"  {f}: 4 head keys, backbone identik base ✓")

    torch.save({"heads": heads, "base_checkpoint": str(base_path),
                "note": "Rekonstruksi: load base, lalu model.load_state_dict(head, strict=False)."}, arch)
    print(f"\narsip: {arch} ({os.path.getsize(arch)/1e6:.2f} MB)")

    arc = torch.load(arch, map_location="cpu")["heads"]
    for f in ckpts:
        sd = load_sd(D / f)
        for k, v in arc[f].items():
            assert k in sd and sd[k].shape == v.shape and torch.equal(sd[k], v), f"MISMATCH {f}:{k}"
    print("✓ verifikasi bit-per-bit lolos untuk semua head")

    freed = sum(os.path.getsize(D / f) for f in ckpts)
    print(f"\n{'APPLY' if args.apply else 'DRY-RUN'}: {len(ckpts)} checkpoint = {freed/1e9:.1f} G")
    if args.apply:
        for f in ckpts:
            os.remove(D / f)
        print(f">>> DIHAPUS. Tersisa: {sorted(os.listdir(D))}")
    else:
        print("(tambah --apply untuk hapus)")


if __name__ == "__main__":
    main()
