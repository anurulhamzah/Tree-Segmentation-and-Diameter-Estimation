#!/usr/bin/env python3
"""Prune checkpoint intermediate setelah training SELESAI.

Simpan hanya `model_final.pth` + checkpoint dengan segm/AP50 terbaik.
Aman untuk dijalankan berkali-kali; melewati run yang masih berjalan.

Contoh:
  # dry-run semua run yang sudah selesai (default, tidak menghapus):
  python prune_checkpoints.py
  # eksekusi untuk satu run:
  python prune_checkpoints.py --apply SwinB_combined_rle_f1000_75k_dbh
  # eksekusi untuk semua run yang sudah selesai:
  python prune_checkpoints.py --apply --all
"""
import argparse, json, os, re, sys

OUT_ROOT = "/fs04/scratch2/pr65/anur0018/maskdino_output"

def cfg_max_iter(d):
    c = os.path.join(d, "config.yaml")
    if not os.path.exists(c): return None
    for line in open(c):
        m = re.match(r"\s*MAX_ITER:\s*(\d+)", line)
        if m: return int(m.group(1))
    return None

def best_ap50_iter(d):
    """Iterasi dengan segm/AP50 tertinggi, atau None kalau belum ada eval."""
    mf = os.path.join(d, "metrics.json")
    best_it, best_v = None, -1.0
    for line in open(mf):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if "segm/AP50" in r and "iteration" in r and r["segm/AP50"] > best_v:
            best_v, best_it = r["segm/AP50"], int(r["iteration"])
    return best_it, best_v

def last_iter(d):
    mf = os.path.join(d, "metrics.json")
    last = -1
    for line in open(mf):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if "total_loss" in r and "iteration" in r:
            last = max(last, int(r["iteration"]))
    return last

def is_finished(d):
    """Selesai jika model_final.pth ada DAN iterasi terakhir mencapai MAX_ITER-buffer."""
    if not os.path.exists(os.path.join(d, "model_final.pth")):
        return False, "tidak ada model_final.pth (masih berjalan / gagal)"
    tgt = cfg_max_iter(d)
    li = last_iter(d)
    if tgt and li < tgt - 200:
        return False, f"iterasi terakhir {li} < MAX_ITER {tgt} (masih berjalan)"
    return True, "selesai"

def keep_set(d):
    keep = {"model_final.pth"}
    bi, bv = best_ap50_iter(d)
    if bi is not None:
        for cand in (f"model_{bi-1:07d}.pth", f"model_{bi:07d}.pth"):
            if os.path.exists(os.path.join(d, cand)): keep.add(cand)
    return keep, bi, bv

def prune_dir(d, apply):
    ok, reason = is_finished(d)
    name = os.path.basename(d.rstrip("/"))
    if not ok:
        print(f"SKIP {name:44s} — {reason}")
        return 0, 0
    keep, bi, bv = keep_set(d)
    ckpts = [f for f in os.listdir(d) if f.startswith("model_") and f.endswith(".pth")]
    todel = [f for f in ckpts if f not in keep]
    freed = sum(os.path.getsize(os.path.join(d, f)) for f in todel)
    tag = "APPLY" if apply else "DRY "
    print(f"{tag} {name:44s} best=AP50 {bv:.1f}@{bi}  keep={sorted(keep)}  "
          f"hapus {len(todel)} ckpt ({freed/1e9:.1f}G)")
    if apply:
        for f in todel: os.remove(os.path.join(d, f))
    return len(todel), freed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="nama dir di maskdino_output (kosong = semua)")
    ap.add_argument("--all", action="store_true", help="proses semua run")
    ap.add_argument("--apply", action="store_true", help="eksekusi hapus (default: dry-run)")
    a = ap.parse_args()

    if a.dirs:
        targets = [os.path.join(OUT_ROOT, x) for x in a.dirs]
    elif a.all or True:  # default: scan semua
        targets = [os.path.join(OUT_ROOT, x) for x in sorted(os.listdir(OUT_ROOT))]
    targets = [t for t in targets if os.path.isdir(t) and os.path.exists(os.path.join(t, "metrics.json"))]

    if not a.apply:
        print(">>> DRY-RUN (tidak menghapus). Tambahkan --apply untuk eksekusi.\n")
    n, f = 0, 0
    for d in targets:
        dn, df = prune_dir(d, a.apply)
        n += dn; f += df
    print(f"\n{'DIHAPUS' if a.apply else 'AKAN DIHAPUS'}: {n} checkpoint, {f/1e9:.1f} G")

if __name__ == "__main__":
    main()
