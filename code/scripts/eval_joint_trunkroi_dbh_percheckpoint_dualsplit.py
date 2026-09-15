#!/usr/bin/env python3
"""
eval_joint_trunkroi_dbh_percheckpoint_dualsplit.py — Eval DBH pada SEMUA checkpoint sebuah job
joint trunk-ROI, untuk val DAN test sekaligus dalam satu lintasan.

Kenapa dua split. Kurva test dipakai untuk PELAPORAN (protokol Sandbox memakai test), sedangkan
kurva val dipakai untuk MEMILIH titik berhenti dan checkpoint terbaik. Memilih keduanya dari test
membuat angka yang dilaporkan bias optimistik, karena titik pengambilannya ikut dioptimalkan pada
split yang sama. Model dibangun sekali lalu weights-nya di-reload tiap checkpoint, dan kedua split
dievaluasi selagi weights itu termuat, sehingga biayanya jauh lebih murah daripada dua lintasan
terpisah.

Beda dengan eval_joint_trunkroi_dbh_percheckpoint.py: skrip itu hanya test split.

Output: reports/dbh_eval/dbh_percheckpoint_dualsplit_<output_dir>.json
    {"output_dir":..., "species_subset":[...], "splits":{"val":..., "test":...}, "table":[
        {"ckpt":..., "iteration":..., "val":{...}, "test":{...}}, ...]}

Jalankan (nohup di node VS Code, bukan SLURM):
    nohup /scratch2/pr65/anur0018/conda_envs/maskdino/bin/python -u \\
        scripts/eval_joint_trunkroi_dbh_percheckpoint_dualsplit.py \\
        --output-dir FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k \\
        --species-subset 1,3,4,5,6,7,8,9,10,12,13 \\
        > logs/eval_percheckpoint_dualsplit_fromscratch.log 2>&1 &
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
OUTPUT_ROOT = PROJECT_ROOT.parent / "maskdino_output"
REPORT_DIR = PROJECT_ROOT / "reports" / "dbh_eval"

sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from detectron2.utils.logger import setup_logger  # noqa: E402

from eval_joint_trunkroi_dbh import build_model_with_head, evaluate, r2_mae  # noqa: E402
from register_combined import register_all_combined  # noqa: E402

MIN_N_PER_SPECIES = 30  # konsisten dengan kriteria di seluruh laporan
DS_PREFIX = "combined_inst_rle_f1000_repaired_v4_stratified"


def iter_of(ckpt_name: str):
    """model_0002499.pth -> 2499 ; model_final.pth -> None (diisi belakangan)."""
    m = re.search(r"model_(\d+)\.pth", ckpt_name)
    return int(m.group(1)) if m else None


def metrics_for(model, dataset_dicts, subset, device):
    """Satu lintasan eval -> dict metrik agregat + mean per-spesies + rincian per spesies."""
    per_pred, per_gt = evaluate(model, dataset_dicts, subset, device)

    all_pred = [v for cid in per_pred for v in per_pred[cid]]
    all_gt = [v for cid in per_gt for v in per_gt[cid]]
    overall = r2_mae(all_pred, all_gt)

    per_species, scored = {}, []
    for cid in sorted(per_pred):
        m = r2_mae(per_pred[cid], per_gt[cid])
        m["N"] = len(per_pred[cid])
        per_species[str(cid)] = m
        if m["N"] >= MIN_N_PER_SPECIES:
            scored.append(m["R2"])

    # mean per-spesies dihitung untuk metrik UTAMA (RMSE, MAE) maupun pendukung (R2),
    # supaya efek komposisi bisa dipisahkan pada metrik mana pun
    mean_sp = lambda key: (sum(per_species[c][key] for c in per_species
                               if per_species[c]["N"] >= MIN_N_PER_SPECIES) / len(scored)
                           ) if scored else float("nan")
    return {
        "N": overall["N"],
        "RMSE": overall["RMSE"],          # metrik utama
        "MAE": overall["MAE"],
        "bias": overall["bias"],
        "rRMSE": overall["rRMSE"],
        "MRAE": overall["MRAE"],
        "R2": overall["R2"],              # metrik pendukung
        "mean_per_species_RMSE": mean_sp("RMSE"),
        "mean_per_species_MAE": mean_sp("MAE"),
        "mean_per_species_R2": (sum(scored) / len(scored)) if scored else float("nan"),
        "n_species_scored": len(scored),
        "per_species": per_species,
    }


def _subset_tag(subset):
    """Tag subset untuk nama berkas. TANPA ini, mengevaluasi model yang sama dengan
    subset spesies berbeda menghasilkan nama berkas identik dan hasil lama TERTIMPA
    (terjadi 2 Agt 2026: hasil 11-spesies 32 checkpoint tertimpa 7-spesies 4 checkpoint,
    dipulihkan dari shard). universe11 memakai nama kosong demi kompatibilitas mundur."""
    ids = sorted(int(x) for x in subset)
    return "" if ids == [1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13] else "_sp" + "-".join(map(str, ids))


def merge_shards(output_dir_name: str, tag: str = ""):
    """Gabungkan berkas shard jadi satu tabel terurut iterasi, lalu ringkas puncaknya."""
    shards = sorted(REPORT_DIR.glob(f"dbh_percheckpoint_dualsplit_{output_dir_name}{tag}__shard*.json"))
    if not shards:
        sys.exit(f"tidak ada shard untuk {output_dir_name} di {REPORT_DIR}")

    rows, meta = {}, None
    for p in shards:
        d = json.load(open(p))
        meta = meta or d
        for r in d["table"]:
            rows[r["ckpt"]] = r          # ckpt unik, aman kalau ada shard tumpang tindih
        print(f"  {p.name}: {len(d['table'])} checkpoint")

    table = sorted(rows.values(), key=lambda r: (r["iteration"] is None, r["iteration"]))
    out = REPORT_DIR / f"dbh_percheckpoint_dualsplit_{output_dir_name}{tag}.json"
    with open(out, "w") as f:
        json.dump({"output_dir": output_dir_name,
                   "species_subset": meta["species_subset"],
                   "min_n_per_species": meta["min_n_per_species"],
                   "splits": meta["splits"],
                   "n_shards": len(shards),
                   "table": table}, f, indent=2)
    print(f"\nGabungan {len(table)} checkpoint -> {out}")

    # Letak optimum dilaporkan untuk metrik UTAMA lebih dulu, baru pendukung. Keduanya
    # belum tentu di iterasi yang sama, dan justru selisih itu yang perlu dilihat.
    for s in meta["splits"]:
        ok = [r for r in table if r[s]["RMSE"] == r[s]["RMSE"]]
        if not ok:
            continue
        last = ok[-1]
        print(f"\n  [{s}]")
        for key, label, lower_better in [("RMSE", "RMSE (utama)", True),
                                         ("MAE", "MAE", True),
                                         ("mean_per_species_R2", "mean per-sp R2 (pendukung)", False)]:
            pick = min if lower_better else max
            best = pick(ok, key=lambda r: r[s][key])
            unit = "mm" if lower_better else "%"
            scale = 1.0 if lower_better else 100.0
            # model_final.pth tidak punya nomor iterasi (iteration=None). Kalau ia yang
            # terpilih sebagai optimum -- mudah terjadi saat checkpoint tersisa sedikit --
            # format ",d" akan crash. Tampilkan namanya sebagai gantinya.
            def _it(r):
                v = r.get("iteration")
                return f"{v:>7,}" if isinstance(v, int) else f"{r.get('ckpt', '?'):>7}"
            print(f"    {label:<28} optimum {best[s][key]*scale:7.2f}{unit} @iter {_it(best)}"
                  f"   akhir {last[s][key]*scale:7.2f}{unit} @iter {_it(last)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="nama dir di maskdino_output/")
    ap.add_argument("--species-subset", required=True, help="mis. 1,3,4,5,6,7,8,9,10,12,13")
    ap.add_argument("--splits", default="val,test", help="urutan split, dipisah koma")
    ap.add_argument("--limit-ckpt", type=int, default=0,
                    help="hanya N checkpoint pertama, untuk uji cepat (0 = semua)")
    ap.add_argument("--ckpt-stride", type=int, default=1,
                    help="jumlah worker paralel; tiap worker mengambil checkpoint berselang-seling")
    ap.add_argument("--ckpt-offset", type=int, default=0,
                    help="indeks worker ini, 0 sampai stride-1")
    ap.add_argument("--merge", action="store_true",
                    help="jangan eval, cukup gabungkan seluruh berkas shard jadi satu")
    args = ap.parse_args()

    if args.merge:
        # tag subset WAJIB ikut, kalau tidak hasil subset berbeda saling menimpa
        merge_shards(args.output_dir,
                     _subset_tag(set(int(x) for x in args.species_subset.split(","))))
        return

    setup_logger()
    os.environ.setdefault("COMBINED_ROOT", str(PROJECT_ROOT / "data" / "combined"))
    register_all_combined(os.environ["COMBINED_ROOT"])

    out_dir = OUTPUT_ROOT / args.output_dir
    subset = set(int(x) for x in args.species_subset.split(","))
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = sorted(p.name for p in out_dir.glob("model_*.pth") if p.name != "model_final.pth")
    if (out_dir / "model_final.pth").exists():
        ckpts.append("model_final.pth")
    if not ckpts:
        sys.exit(f"tidak ada checkpoint di {out_dir}")

    # model dibangun dari checkpoint mana pun yang ada; weights-nya di-reload per checkpoint
    # di bawah, jadi tidak harus model_final.pth (yang belum ada selagi training berjalan)
    model, _cfg = build_model_with_head(out_dir, device, init_ckpt=ckpts[0])
    dicts = {s: DatasetCatalog.get(f"{DS_PREFIX}_{s}") for s in splits}
    for s in splits:
        print(f"[dualsplit] split {s}: {len(dicts[s])} gambar")

    if args.limit_ckpt:
        ckpts = ckpts[: args.limit_ckpt]
    n_all = len(ckpts)
    if args.ckpt_stride > 1:
        # berselang-seling, bukan blok, supaya hasil parsial tiap worker tetap
        # tersebar merata di seluruh rentang iterasi
        ckpts = ckpts[args.ckpt_offset :: args.ckpt_stride]
    print(f"[dualsplit] worker {args.ckpt_offset+1}/{args.ckpt_stride}: "
          f"{len(ckpts)} dari {n_all} checkpoint di {out_dir.name}, split: {splits}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = (f"__shard{args.ckpt_offset}of{args.ckpt_stride}"
              if args.ckpt_stride > 1 else "")
    out_path = REPORT_DIR / f"dbh_percheckpoint_dualsplit_{args.output_dir}{_subset_tag(subset)}{suffix}.json"

    table, t0 = [], time.time()
    for k, name in enumerate(ckpts, 1):
        muat = DetectionCheckpointer(model).load(str(out_dir / name))
        model.eval()

        # Nomor iterasi model_final.pth dibaca dari ISI checkpoint, bukan ditebak.
        # Versi lama mengisinya dengan max(iterasi shard ini) + 1, sehingga labelnya
        # bergantung pada worker mana yang kebagian: pada eval cap20/cap30 (stride 4)
        # model_final jatuh ke worker pemegang 9.999 dan tercatat "10000", padahal
        # checkpoint-nya berisi iteration=19999. Salah label itu menempatkan checkpoint
        # TERAKHIR di tengah trayektori dan membuat tabel menyesatkan.
        it_ckpt = iter_of(name)
        if it_ckpt is None and isinstance(muat, dict):
            it_ckpt = muat.get("iteration")
        row = {"ckpt": name, "iteration": it_ckpt}
        for s in splits:
            row[s] = metrics_for(model, dicts[s], subset, device)
        table.append(row)

        el = time.time() - t0
        it = row["iteration"] if row["iteration"] is not None else "final"
        parts = " | ".join(
            f"{s}: RMSE={row[s]['RMSE']:6.2f} MAE={row[s]['MAE']:6.2f}mm "
            f"bias={row[s]['bias']:+6.2f} R2={row[s]['R2']*100:5.2f}% "
            f"mSpR2={row[s]['mean_per_species_R2']*100:6.2f}%"
            for s in splits
        )
        print(f"  [{k:>2}/{len(ckpts)}] {name:<22} iter={str(it):>7}  {parts}"
              f"   ({el/k:.0f}s/ckpt, sisa ~{(len(ckpts)-k)*el/k/60:.0f}m)", flush=True)

        # tulis inkremental supaya hasil parsial tetap terpakai kalau dihentikan.
        # Cadangan terakhir kalau checkpoint tidak menyimpan nomor iterasi sama sekali:
        # taruh sesudah titik tertinggi shard ini. Ditandai lewat `iteration_ditebak`
        # supaya pembaca tahu angka itu bukan berasal dari checkpoint.
        nums = [r["iteration"] for r in table if r["iteration"] is not None]
        for r in table:
            if r["iteration"] is None and nums:
                r["iteration"] = max(nums) + 1
                r["iteration_ditebak"] = True
        with open(out_path, "w") as f:
            json.dump({"output_dir": args.output_dir,
                       "species_subset": sorted(subset),
                       "min_n_per_species": MIN_N_PER_SPECIES,
                       "splits": splits,
                       "table": table}, f, indent=2)

    print(f"\nSelesai dalam {(time.time()-t0)/60:.1f} menit. Tersimpan: {out_path}")

    # ringkasan: di mana puncaknya menurut masing-masing split
    for s in splits:
        best = max(table, key=lambda r: r[s]["mean_per_species_R2"])
        last = table[-1]
        print(f"  {s:<5} puncak mean-per-spesies {best[s]['mean_per_species_R2']*100:.2f}% "
              f"@iter {best['iteration']:,}  |  akhir {last[s]['mean_per_species_R2']*100:.2f}% "
              f"@iter {last['iteration']:,}")


if __name__ == "__main__":
    main()
