#!/usr/bin/env python3
"""refresh_joint_report.py — segarkan blok data pada laporan joint from-scratch 155k.

Menggantikan refresh_joint_report_run7.py, yang hanya menangani dua blok saat laporan
belum memuat run jadwal DBH_WEIGHT.

Bagian 1 sampai 11 laporan itu membahas model 11-spesies yang sudah selesai dan TIDAK
disentuh. Yang diganti hanya dua blok data JavaScript:

    const RUN7 = {...};   status + kurva run final 7-spesies
    const CMP3 = {...};   perbandingan tiga run (tanpa head DBH / joint 11-sp / joint 7-sp)

Narasi di sekitarnya dihitung dari data ini oleh JavaScript, jadi ikut menyesuaikan
sendiri. Jalankan ulang kapan saja:

    python scripts/refresh_joint_report_run7.py
"""
import json
import re
from pathlib import Path

ROOT = Path("/scratch2/pr65/anur0018")
OUT = ROOT / "maskdino_output"
HTML = (ROOT / "tree_classification/reports/stratified_v4" /
        "Report_Joint_FromScratch_155k_Varian_dan_Analisis_Error.html")

RUNS = {
    "plain": "FocalNet_L_combined_rgbd_scratch_v4_stratified_plain155k",
    "j11": "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k",
    "j7": "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_7species_155k",
    "wramp": "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_7species_155k_wramp",
}
# Jadwal DBH_WEIGHT run wramp; dipakai untuk menggambar kurva DBH_WEIGHT dan menandai fasenya.
RAMP = dict(w0=5.0, w1=50.0, start=60000, end=85000)
TARGET_ITER = 155000


def baca(nama):
    """Pisahkan baris eval dari baris loss. Baris eval JUGA memuat total_loss, jadi
    pemisahannya harus lewat kehadiran segm/AP50, bukan elif berantai."""
    seg, loss = [], []
    for line in (OUT / nama / "metrics.json").open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        it = r.get("iteration")
        if it is None:
            continue
        if "segm/AP50" in r:
            seg.append(dict(iteration=it, AP50=round(r["segm/AP50"], 3),
                            AP=round(r.get("segm/AP", 0), 3), AP75=round(r.get("segm/AP75", 0), 3),
                            APs=round(r.get("segm/APs", 0), 3), APm=round(r.get("segm/APm", 0), 3),
                            APl=round(r.get("segm/APl", 0), 3)))
        elif "total_loss" in r:
            loss.append(dict(iteration=it, total=round(r["total_loss"], 3),
                             dbh_w=round(r.get("loss_dbh_trunkroi", 0) or 0, 4),
                             dbh_raw=round(r.get("dbh_trunkroi/raw_loss", 0) or 0, 4),
                             n_clean=r.get("dbh_trunkroi/n_clean"),
                             n_total=r.get("dbh_trunkroi/n_total")))
    return seg, loss


def sub(html, nama, obj):
    """Ganti satu blok `const NAMA = {...};`. Wajib tepat satu kecocokan, supaya
    kesalahan pola tidak diam-diam menghasilkan berkas yang tidak berubah."""
    pola = re.compile(r"const " + nama + r" = \{.*?\};\n", re.S)
    baru = f"const {nama} = " + json.dumps(obj, separators=(",", ":")) + ";\n"
    html, n = pola.subn(lambda _m: baru, html, count=1)
    if n != 1:
        raise SystemExit(f"pola `const {nama}` cocok {n} kali, seharusnya 1")
    return html


def weight_pada(it):
    """Loss weight DBH pada satu iterasi, mereplikasi _dbh_weight_now() di skrip training."""
    if it <= RAMP["start"]:
        return RAMP["w0"]
    if it >= RAMP["end"]:
        return RAMP["w1"]
    f = (it - RAMP["start"]) / (RAMP["end"] - RAMP["start"])
    return RAMP["w0"] + f * (RAMP["w1"] - RAMP["w0"])


def blok_wramp(seg7, loss7):
    """Data run jadwal DBH_WEIGHT, berpasangan dengan baseline pada iterasi yang SAMA.

    Selisih AP50 dilaporkan hanya pada titik yang ada di KEDUA run. Menjajarkan lewat
    interpolasi akan mengarang presisi, dan pada metrik yang ayunannya beberapa poin
    persen itu menyesatkan.

    Rentang sebelum RAMP["start"] adalah kontrol alami: di situ kedua run dikonfigurasi
    identik, sehingga selisihnya mengukur noise antar-run, bukan efek jadwal DBH_WEIGHT.
    """
    segw, lossw = baca(RUNS["wramp"])
    if not segw:
        return dict(ada=False)

    base = {s["iteration"]: s["AP50"] for s in seg7}
    pas = [dict(iteration=s["iteration"], w=s["AP50"], b=base[s["iteration"]],
                d=round(s["AP50"] - base[s["iteration"]], 3))
           for s in segw if s["iteration"] in base]
    pra = [p["d"] for p in pas if p["iteration"] < RAMP["start"]]

    it_now = lossw[-1]["iteration"]
    step = max(1, len(lossw) // 70)
    nc = [l["n_clean"] for l in lossw if l["n_clean"] is not None]
    nt = [l["n_total"] for l in lossw if l["n_total"] is not None]

    # Loss segmentasi kedua run pada iterasi yang sama, komponen DBH dikeluarkan.
    bl = {l["iteration"]: l["total"] - l["dbh_w"] for l in loss7}
    kurva = [dict(x=l["iteration"], w=round(l["total"] - l["dbh_w"], 3),
                  b=round(bl[l["iteration"]], 3))
             for l in lossw[::step] if l["iteration"] in bl]

    return dict(
        ada=True, iter_now=it_now, target=TARGET_ITER, ramp=RAMP,
        w_now=round(weight_pada(it_now), 2),
        segw=segw, pas=pas, kurva=kurva,
        n_clean_avg=round(sum(nc) / len(nc), 2) if nc else None,
        n_total_avg=round(sum(nt) / len(nt), 2) if nt else None,
        # noise floor: sebaran selisih AP50 pada rentang yang konfigurasinya identik.
        pra_n=len(pra),
        pra_mean=round(sum(pra) / len(pra), 2) if pra else None,
        pra_min=round(min(pra), 2) if pra else None,
        pra_max=round(max(pra), 2) if pra else None,
        kurva_weight=[dict(x=x, y=round(weight_pada(x), 2))
                     for x in range(0, TARGET_ITER + 1, 2500)],
    )


def main():
    seg7, loss7 = baca(RUNS["j7"])
    if not seg7:
        raise SystemExit("run final belum punya titik eval")
    batas = max(s["iteration"] for s in seg7) + 1

    step = max(1, len(loss7) // 70)
    nc = [l["n_clean"] for l in loss7 if l["n_clean"] is not None]
    nt = [l["n_total"] for l in loss7 if l["n_total"] is not None]
    run7 = dict(seg7=seg7,
                seg11=[s for s in baca(RUNS["j11"])[0] if s["iteration"] < batas],
                loss7=loss7[::step], iter_now=loss7[-1]["iteration"], target=TARGET_ITER,
                n_clean_avg=round(sum(nc) / len(nc), 1) if nc else None,
                n_total_avg=round(sum(nt) / len(nt), 1) if nt else None)

    cmp3 = {}
    for tag, nama in RUNS.items():
        seg, loss = baca(nama)
        st = max(1, len(loss) // 80)
        # Loss DBH dikeluarkan supaya SEMUA kurva mengukur hal yang SAMA: run tanpa
        # head DBH memang tidak memilikinya.
        cmp3[tag] = dict(seg=[dict(x=s["iteration"], y=s["AP50"]) for s in seg],
                         loss=[dict(x=l["iteration"], y=round(l["total"] - l["dbh_w"], 3))
                               for l in loss[::st]])

    runw = blok_wramp(seg7, loss7)

    html = HTML.read_text(encoding="utf-8")
    html = sub(html, "RUN7", run7)
    html = sub(html, "CMP3", cmp3)
    html = sub(html, "RUNW", runw)
    HTML.write_text(html, encoding="utf-8")

    pct = run7["iter_now"] / TARGET_ITER * 100
    print(f"tersimpan: {HTML}")
    print(f"  run final : iterasi {run7['iter_now']:,} ({pct:.1f}%), "
          f"{len(seg7)} titik eval, AP50 terakhir {seg7[-1]['AP50']:.2f}%".replace(",", "."))
    print(f"  pembanding: 11-spesies {len(run7['seg11'])} titik dalam rentang yang sama")
    if runw["ada"]:
        print(f"  wramp     : iterasi {runw['iter_now']:,} "
              f"({runw['iter_now']/TARGET_ITER*100:.1f}%), weight {runw['w_now']}, "
              f"{len(runw['pas'])} titik berpasangan".replace(",", "."))
        if runw["pra_n"]:
            print(f"  noise floor: {runw['pra_n']} titik pra-ramp, selisih AP50 "
                  f"{runw['pra_min']:+.2f} s/d {runw['pra_max']:+.2f} pp "
                  f"(rerata {runw['pra_mean']:+.2f})")
    else:
        print("  wramp     : belum ada titik eval")
    print(f"  ukuran    : {len(html):,} byte".replace(",", "."))


if __name__ == "__main__":
    main()
