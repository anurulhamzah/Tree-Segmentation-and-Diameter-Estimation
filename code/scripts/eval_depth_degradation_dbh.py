#!/usr/bin/env python3
"""Uji ketergantungan pendugaan diameter pada channel depth, dengan menurunkan kualitas depth
saat inferensi lalu mengukur RMSE.

Pelengkap `eval_depth_degradation.py`, yang mengukur hal yang sama untuk segmentasi (AP50).
Naskah mengukur sumbangan depth ke segmentasi lewat 6 pasang run RGB lawan RGB-D, tetapi tidak
pernah mengukur sumbangannya ke diameter sama sekali.

Ablasi biner RGB lawan RGB-D TIDAK terdefinisi untuk diameter: `_find_row_1p3m()` memerlukan
depth untuk menemukan baris setinggi 1,3 m, jadi tanpa depth head mengembalikan None, bukan
prediksi yang lebih buruk. Yang bisa diukur adalah sensitivitas terhadap KUALITAS depth.

Depth masuk ke diameter lewat dua jalur terpisah, dan script ini bisa merusaknya sendiri-sendiri:

  feat  depth sebagai channel ke-4 masuk backbone, ikut membentuk res2 yang dibaca MLP head
  geom  depth dipakai head untuk mencari baris 1,3 m dan menyaring kolom trunk
  both  keduanya, sebanding dengan uji segmentasi

Memisahkan keduanya menjawab pertanyaan yang berbeda dari sekadar "apakah depth penting":
apakah depth berguna karena membentuk fitur, atau karena menunjukkan DI MANA batang berada.

Noise dibangkitkan dengan seed per gambar, bukan satu aliran berurutan, supaya kondisi feat,
geom, dan both menerima perusakan yang identik dan selisihnya murni soal jalur.

Jalankan:
    python scripts/eval_depth_degradation_dbh.py --output-dir <run> --species-subset 1,3,4,...
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from detectron2.data import DatasetCatalog, detection_utils as utils
from detectron2.structures import ImageList
from detectron2.utils.logger import setup_logger
from register_combined import register_all_combined
import eval_joint_trunkroi_dbh as E


def rusak(d: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """d sudah ternormalisasi [0,1]. Semantiknya sama persis dengan eval_depth_degradation.py."""
    if mode == "intact":
        return d
    if mode == "zero":
        return np.zeros_like(d)
    rng = np.random.default_rng(seed)
    if mode == "shuffle":
        return d[rng.permutation(d.shape[0])]
    sigma = {"noise05": 0.05, "noise15": 0.15}[mode]
    return np.clip(d + rng.normal(0, sigma, d.shape).astype(np.float32), 0, 1)


@torch.no_grad()
def evaluate(model, dataset_dicts, species_subset, device, mode, path):
    head = model.dbh_head_trunkroi; head.eval()
    size_div = model.size_divisibility
    pred_by_sp, gt_by_sp = {}, {}
    n_filter = 0

    for i, rec in enumerate(dataset_dicts):
        if (i + 1) % 300 == 0:
            print(f"    {i+1}/{len(dataset_dicts)} images", flush=True)
        stem = Path(rec["file_name"]).stem
        pfm = E.DEPTH_DIR / f"{stem}.pfm"
        if not pfm.exists():
            continue
        H_img, W_img = rec["height"], rec["width"]
        image = utils.read_image(rec["file_name"], format="RGB")
        depth_raw = E._read_pfm(str(pfm))
        if depth_raw.shape != (H_img, W_img):
            import cv2
            depth_raw = cv2.resize(depth_raw, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        d01 = E._normalize_depth(depth_raw)

        d_feat = rusak(d01, mode, i) if path in ("feat", "both") else d01
        d_geom = rusak(d01, mode, i) if path in ("geom", "both") else d01

        image_t = torch.as_tensor(image.transpose(2, 0, 1).copy(), dtype=torch.float32)
        depth_t = torch.as_tensor(d_feat * 255.0, dtype=torch.float32).unsqueeze(0)
        img_list = ImageList.from_tensors([torch.cat([image_t, depth_t], 0)], size_div)
        imgs_norm = (img_list.tensor.to(device) - model.pixel_mean) / model.pixel_std
        res2 = model.backbone(imgs_norm)["res2"][0]
        depth_b = torch.as_tensor(d_geom, dtype=torch.float32, device=device)

        for ann in rec.get("annotations", []):
            dbh_cm = float(ann.get("dbh") or 0.0)
            if dbh_cm <= 0:
                continue
            cat_id = int(ann["category_id"]) + 1
            if cat_id not in species_subset:
                continue
            if cat_id == E.RUBBERFIG_CAT_ID and dbh_cm >= E.RUBBERFIG_CAP_CM:
                continue
            mask_t = torch.as_tensor(E.decode_mask(ann, H_img, W_img), device=device)
            result = head.forward_single(res2, depth_b, mask_t, species_idx=cat_id - 1, strict=True)
            if result is None:
                n_filter += 1; continue
            pred, _, d_trunk, trunk_px, world_h = result
            if trunk_px < E.MIN_TRUNK_PX or not (E.MIN_DEPTH <= d_trunk < E.MAX_DEPTH) \
               or abs(world_h - 1.3) > E.MAX_WH_DEV:
                n_filter += 1; continue
            pred_by_sp.setdefault(cat_id, []).append(torch.expm1(pred).clamp(min=0).item())
            gt_by_sp.setdefault(cat_id, []).append(dbh_cm * 10.0)
    return pred_by_sp, gt_by_sp, n_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--species-subset", required=True)
    ap.add_argument("--dataset", default="combined_inst_rle_f1000_repaired_v4_stratified_test")
    ap.add_argument("--modes", default="intact,noise05,noise15,shuffle,zero")
    ap.add_argument("--paths", default="both,feat,geom")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: batasi jumlah gambar")
    args = ap.parse_args()

    setup_logger()
    import os
    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_combined(os.environ["COMBINED_ROOT"])
    subset = set(int(x) for x in args.species_subset.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = E.build_model_with_head(E.OUTPUT_ROOT / args.output_dir, device)
    dd = DatasetCatalog.get(args.dataset)
    if args.limit: dd = dd[:args.limit]
    print(f"{len(dd)} images, subset={sorted(subset)}, device={device}\n")

    hasil = {}
    for path in args.paths.split(","):
        for mode in args.modes.split(","):
            if mode == "intact" and path != "both":
                continue                      # intact identik di ketiga jalur
            key = f"{path}/{mode}"
            print(f"  {key} ...", flush=True)
            p, g, nf = evaluate(model, dd, subset, device, mode, path)
            allp = [x for v in p.values() for x in v]
            allg = [x for v in g.values() for x in v]
            if not allp:
                hasil[key] = {"N": 0, "n_filter": nf}
                print(f"    N=0, semua tersaring (n_filter={nf})", flush=True); continue
            m = E.r2_mae(allp, allg)
            scored = [E.r2_mae(p[c], g[c])["R2"] for c in p if len(p[c]) >= E.MIN_SPECIES_N]
            m["mean_per_species_R2"] = float(np.mean(scored)) if scored else None
            m["n_filter"] = nf
            hasil[key] = m
            print(f"    N={m['N']}  RMSE={m['RMSE']:.2f}mm  MAE={m['MAE']:.2f}mm  "
                  f"bias={m['bias']:.2f}mm  R2={m['R2']*100:.2f}%", flush=True)

    out = PROJECT_ROOT / "reports" / f"depth_degradation_dbh_{args.output_dir}.json"
    json.dump({"run": args.output_dir, "dataset": args.dataset, "hasil": hasil},
              open(out, "w"), indent=2)
    print(f"\ntersimpan: {out}")


if __name__ == "__main__":
    main()
