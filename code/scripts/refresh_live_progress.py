#!/usr/bin/env python3
"""
refresh_live_progress.py — Sinkronkan angka "live" di lr_restart_comparison_interactive.html
dengan kondisi terkini job training 58544358 (joint from-scratch 155k).

Laporan menampilkan 4 nilai live yang selama ini di-update manual dan cepat basi:
  LIVE_ITER, LIVE_AP50, LIVE_AP50_ITER, LIVE_DBH_LOSS  (blok JS di HTML)
  + deret `live` pada chart progres  (x/y per checkpoint eval 5000 iter)
  + kalimat "posisi terakhir iter N, AP50=X%" di footnote chart

Skrip ini membaca angka itu langsung dari log training lalu menulis ulang HTML-nya.

Usage:
    python scripts/refresh_live_progress.py            # tulis perubahan
    python scripts/refresh_live_progress.py --dry-run  # tampilkan saja
"""
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = PROJECT_ROOT / "logs" / "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k_58544358.out"
HTML = PROJECT_ROOT / "reports" / "stratified_v4" / "lr_restart_comparison_interactive.html"
EVAL_PERIOD = 5000


def parse_log(log_path):
    """Return (last_iter, dbh_loss, [(iter, segm_ap50), ...])."""
    text = log_path.read_text(errors="replace")

    iters = re.findall(r"iter: (\d+)", text)
    if not iters:
        sys.exit(f"[ERROR] tidak ada baris 'iter:' di {log_path}")
    last_iter = int(iters[-1])

    losses = re.findall(r"dbh_trunkroi/raw_loss: ([\d.]+)", text)
    dbh_loss = float(losses[-1]) if losses else None

    # Blok evaluasi: baris "Task: segm" lalu header lalu baris angka.
    ap50s = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "copypaste: Task: segm" in line:
            for j in range(i + 1, min(i + 4, len(lines))):
                m = re.search(r"copypaste: ([\d.]+),([\d.]+),", lines[j])
                if m:
                    ap50s.append(float(m.group(2)))
                    break

    points = [((k + 1) * EVAL_PERIOD - 1, ap) for k, ap in enumerate(ap50s)]
    return last_iter, dbh_loss, points


def fmt_series(points, indent="        "):
    xs = [str(p[0]) for p in points]
    ys = [f"{p[1]:.2f}" for p in points]

    def wrap(vals):
        out, line = [], ""
        for v in vals:
            piece = (v + ",")
            if len(line) + len(piece) > 72:
                out.append(line.rstrip())
                line = ""
            line += piece
        out.append(line.rstrip().rstrip(","))
        return ("\n" + indent).join(out)

    return wrap(xs), wrap(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    last_iter, dbh_loss, points = parse_log(LOG)
    if not points:
        sys.exit("[ERROR] belum ada checkpoint eval di log")
    last_eval_iter, last_ap50 = points[-1]

    print(f"iter terakhir     : {last_iter:,}")
    print(f"eval terakhir     : iter {last_eval_iter:,}  AP50={last_ap50:.2f}%")
    print(f"loss DBH terakhir : {dbh_loss}")
    print(f"jumlah titik eval : {len(points)}")

    html = HTML.read_text()
    orig = html

    html = re.sub(r"var LIVE_ITER = \d+;", f"var LIVE_ITER = {last_iter};", html)
    html = re.sub(r"var LIVE_AP50 = [\d.]+;", f"var LIVE_AP50 = {last_ap50:.2f};", html)
    html = re.sub(r"var LIVE_AP50_ITER = \d+;", f"var LIVE_AP50_ITER = {last_eval_iter};", html)
    if dbh_loss is not None:
        html = re.sub(r"var LIVE_DBH_LOSS = [\d.]+;", f"var LIVE_DBH_LOSS = {dbh_loss};", html)

    xs, ys = fmt_series(points)
    new_series = (
        "  var live = { // job 58544358 — di-refresh oleh scripts/refresh_live_progress.py\n"
        f"    x: [{xs}],\n"
        f"    y: [{ys}]\n"
        "  };"
    )
    html = re.sub(
        r"  var live = \{.*?\n  \};",
        lambda _: new_series,
        html,
        count=1,
        flags=re.S,
    )

    iter_id = f"{last_eval_iter:,}".replace(",", ".")      # 99.999
    ap50_id = f"{last_ap50:.2f}".replace(".", ",")          # 57,42
    html = re.sub(
        r"berjalan \(posisi terakhir iter [\d.]+, AP50=[\d,]+%\)",
        f"berjalan (posisi terakhir iter {iter_id}, AP50={ap50_id}%)",
        html,
    )

    if html == orig:
        print("\nTidak ada perubahan — HTML sudah sinkron.")
        return
    if args.dry_run:
        print("\n[dry-run] HTML TIDAK ditulis.")
        return
    HTML.write_text(html)
    print(f"\nHTML diperbarui: {HTML}")


if __name__ == "__main__":
    main()
