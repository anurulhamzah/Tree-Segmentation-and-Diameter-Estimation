#!/usr/bin/env python3
"""refresh_progress_slides.py — segarkan angka run final pada deck progres.

Deck `reports/stratified_v4/progress_agustus2026_slides.html` memuat teks kemajuan DAN blok data chart yang
bergerak selama run final berjalan: iterasi, persentase, lebar bar kemajuan, laju, perkiraan
selesai, margin batas SLURM, AP50 terakhir, dan dua kurva loss. Menyuntingnya satu per satu
mengundang angka yang saling bertentangan di dalam satu halaman, dan itu sudah pernah terjadi
pada laporan HTML.

Skrip ini membaca ulang semuanya dari sumber (metrics.json, log SLURM, squeue), lalu mengganti
setiap angka lewat pola yang WAJIB cocok tepat satu kali. Kalau ada pola yang tidak cocok,
skrip berhenti tanpa menulis apa pun, sehingga deck tidak pernah setengah diperbarui.

Semua angka memakai SATU stempel waktu, karena perkiraan selesai yang dihitung dari acuan lama
pernah membuat "sisa waktu" meleset satu jam penuh.

Dua keadaan ditangani. Selama run BERJALAN, sumbernya `squeue` dan kolom waktu berisi
perkiraan. Setelah run SELESAI job hilang dari antrean, jadi sumbernya pindah ke `sacct`
dan kolom yang sama berisi waktu selesai sebenarnya, laju rata-rata sebenarnya, serta sisa
margin batas SLURM yang tidak terpakai. Setiap pola ditulis supaya cocok pada KEDUA bentuk
teks itu, sehingga skrip tetap bisa dijalankan ulang berapa kali pun tanpa merusak deck.

    python scripts/refresh_progress_slides.py
"""
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJ = Path("/scratch2/pr65/anur0018/tree_classification")
DECK = PROJ / "reports" / "stratified_v4" / "progress_agustus2026_slides.html"
OUT = Path("/scratch2/pr65/anur0018/maskdino_output/"
           "FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_7species_155k")
JOB = "58750955"
TARGET = 155000

# Hasil eval DBH per-checkpoint kedua run joint. Yang 7-spesies dinilai pada subset
# 7 spesies, yang 11-spesies pada 11 spesies, masing-masing populasi asalnya sendiri.
EVAL7 = (PROJ / "reports" / "dbh_eval" / "dbh_percheckpoint_dualsplit_FocalNet_L_trunk_roi_"
         "dbh_hybrid_joint_scratch_7species_155k_sp3-5-8-9-10-12-13.json")
EVAL11 = (PROJ / "reports" / "dbh_eval" /
          "dbh_percheckpoint_dualsplit_FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_155k.json")

# Nilai frozen (cm, rerata 3 seed) berasal dari pipeline sandbox frozen, bukan dari berkas
# eval per-checkpoint di atas, jadi ditulis sebagai konstanta. Dipakai hanya untuk menentukan
# sel mana yang ditebalkan sebagai nilai terbaik dalam tiap pasangan subset.
FROZEN = {
    "11": dict(RMSE=7.86, MAE=5.21, R2=14.09, bias=-0.59, ap="64,14%", iter="135k + 20k"),
    "7": dict(RMSE=8.03, MAE=5.35, R2=20.00, bias=-0.53, ap="64,14%", iter="135k + 20k"),
}


def final_test_metrics(path):
    """Metrik test pada model_final.pth, atau None kalau evalnya belum ada.

    Dipilih eksplisit lewat nama checkpoint, bukan lewat baris terakhir tabel: tabel
    diurutkan menurut iterasi dan model_final.pth punya iteration=None, sehingga posisinya
    di ujung tidak dijamin oleh urutan itu."""
    if not path.exists():
        return None
    tabel = json.load(open(path))["table"]
    for r in tabel:
        if r["ckpt"] == "model_final.pth":
            return r["test"]
    return None



def baris_model(dbh7, dbh11, ap7):
    """Array `model` pada blok data chart slide 1.

    Blok ini SEBELUMNYA tidak didaftarkan ke skrip, sehingga chart dan tabel pada slide
    yang sama pernah bertentangan: tabel menampilkan 8,32 cm untuk run final sementara
    chart masih menggambar bar putus-putus "belum ada", dan baris 11-spesies memakai
    checkpoint terbaik di chart tetapi checkpoint akhir di tabel. Keduanya kini diturunkan
    dari berkas eval yang sama.
    """
    def isi(m):
        if m is None:
            return dict(rmse=None, mae=None, r2=None, bias=None)
        return dict(rmse=round(m["RMSE"] / 10, 3), mae=round(m["MAE"] / 10, 3),
                    r2=round(m["mean_per_species_R2"] * 100, 2),
                    bias=round(m["bias"] / 10, 3))
    m7, m11 = isi(dbh7), isi(dbh11)
    return [
        dict(n="Tanpa head DBH", color="#8a877d", it="155k seg", ap=64.48,
             rmse=None, mae=None, r2=None, bias=None, k="ref"),
        dict(n="Multi-task 11-sp", color="#c9591f", it="155k bersama", ap=64.02, k="joint", **m11),
        dict(n="Two-stage 11-sp", color="#7a55c9", it="135k + 20k", ap=64.14,
             rmse=FROZEN["11"]["RMSE"], mae=FROZEN["11"]["MAE"], r2=FROZEN["11"]["R2"],
             bias=FROZEN["11"]["bias"], k="frozen"),
        dict(n="Multi-task 7-sp", color="#2a78d6", it="155k bersama", ap=ap7, k="p7", **m7),
        dict(n="Two-stage 7-sp", color="#1d6b4a", it="135k + 20k", ap=64.14,
             rmse=FROZEN["7"]["RMSE"], mae=FROZEN["7"]["MAE"], r2=FROZEN["7"]["R2"],
             bias=FROZEN["7"]["bias"], k="f7"),
    ]



def sel(v, n=2, tebal=False, minus=False):
    """Satu sel angka. `minus` memakai &minus; supaya tanda negatif tampil sebagai tanda
    matematis, mengikuti sel bias yang sudah ada di deck."""
    isi = ("&minus;" + dec(abs(v), n)) if minus else dec(v, n)
    return f"<td><strong>{isi}</strong></td>" if tebal else f"<td>{isi}</td>"


def baris_dbh(nama, iterasi, ap, joint, frozen, hi=False):
    """Dua baris tabel untuk satu subset: joint lalu frozen pasangannya.

    Penebalan ditentukan PER PASANGAN, bukan lintas seluruh tabel, mengikuti konvensi yang
    sudah dipakai deck. RMSE/MAE makin kecil makin baik, R2 makin besar makin baik, bias
    dinilai dari jarak ke nol sehingga tanda tidak ikut menentukan pemenang."""
    f = frozen
    if joint is None:
        # Evalnya belum ada. Baris frozen sengaja TIDAK ditebalkan: tanpa pembanding,
        # penebalan "nilai terbaik" tidak punya arti.
        kosong = '<td class="muted">menunggu eval</td>'
        tr = '<tr class="hi">' if hi else "<tr>"
        return (
            f'{tr}<td class="l">Multi-task {nama}</td><td class="l">{iterasi}</td><td>{ap}</td>\n'
            f'        {kosong}{kosong}<td class="muted">&mdash;</td>\n'
            f'        <td class="muted">&mdash;</td></tr>\n'
            f'    <tr><td class="l">Two-stage {nama}</td><td class="l">{f["iter"]}</td>'
            f'<td>{f["ap"]}</td>\n'
            f'        {sel(f["RMSE"], 2)}{sel(f["MAE"], 2)}{persen(f["R2"], False)}\n'
            f'        {sel(f["bias"], 2, minus=True)}</tr>'
        )

    j = dict(RMSE=joint["RMSE"] / 10, MAE=joint["MAE"] / 10, bias=joint["bias"] / 10,
             R2=joint["mean_per_species_R2"] * 100)
    menang = dict(RMSE=j["RMSE"] < f["RMSE"], MAE=j["MAE"] < f["MAE"],
                  R2=j["R2"] > f["R2"], bias=abs(j["bias"]) < abs(f["bias"]))
    tr = '<tr class="hi">' if hi else "<tr>"
    return (
        f'{tr}<td class="l">Multi-task {nama}</td><td class="l">{iterasi}</td><td>{ap}</td>\n'
        f'        {sel(j["RMSE"], 2, menang["RMSE"])}{sel(j["MAE"], 2, menang["MAE"])}'
        f'{persen(j["R2"], menang["R2"])}\n'
        f'        {sel(j["bias"], 2, menang["bias"], minus=True)}</tr>\n'
        f'    <tr><td class="l">Two-stage {nama}</td><td class="l">{f["iter"]}</td>'
        f'<td>{f["ap"]}</td>\n'
        f'        {sel(f["RMSE"], 2, not menang["RMSE"])}{sel(f["MAE"], 2, not menang["MAE"])}'
        f'{persen(f["R2"], not menang["R2"])}\n'
        f'        {sel(f["bias"], 2, not menang["bias"], minus=True)}</tr>'
    )


def persen(v, tebal):
    isi = f"+{dec(v, 2)}%" if v >= 0 else f"&minus;{dec(abs(v), 2)}%"
    return f"<td><strong>{isi}</strong></td>" if tebal else f"<td>{isi}</td>"


def dec(v, n=1):
    """Desimal gaya Indonesia: koma sebagai pemisah pecahan, titik sebagai ribuan."""
    s = f"{v:,.{n}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


HARI = {"Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu",
        "Thursday": "Kamis", "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu"}
BULAN = {"January": "Januari", "February": "Februari", "March": "Maret", "April": "April",
         "May": "Mei", "June": "Juni", "July": "Juli", "August": "Agustus",
         "September": "September", "October": "Oktober", "November": "November",
         "December": "Desember"}


def id_tanggal(d, fmt):
    """Tanggal berbahasa Indonesia. strftime memakai locale sistem yang di cluster ini
    berbahasa Inggris, dan setlocale ke id_ID tidak tersedia, jadi diganti manual."""
    s = d.strftime(fmt)
    for en, idn in {**HARI, **BULAN}.items():
        s = s.replace(en, idn)
    return s


def jam(s):
    hari, _, sisa = s.rpartition("-")
    p = [float(x) for x in sisa.split(":")]
    while len(p) < 3:
        p = [0] + p
    return (int(hari) if hari else 0) * 24 + p[0] + p[1] / 60 + p[2] / 3600


def main():
    now = datetime.now()

    seg, loss = [], []
    for line in (OUT / "metrics.json").open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("iteration") is None:
            continue
        if "segm/AP50" in r:
            seg.append([r["iteration"], round(r["segm/AP50"], 2)])
        elif "total_loss" in r:
            # Komponen DBH dikeluarkan supaya kurva ini sebanding dengan run 11-spesies
            # pada slide status, yang juga sudah dikurangi komponen yang sama.
            loss.append([r["iteration"],
                         round(r["total_loss"] - (r.get("loss_dbh_trunkroi", 0) or 0), 1)])
    if not seg or not loss:
        sys.exit("metrics.json belum memuat titik eval atau loss")

    q = subprocess.run(["squeue", "-j", JOB, "-h", "-o", "%M|%L"],
                       capture_output=True, text=True).stdout.strip()
    if q:
        el, lf = q.split("|")
        it = loss[-1][0]
        pct = it / TARGET * 100
        laju = it / jam(el)
        sisa = (TARGET - it) / laju
        selesai = now + timedelta(hours=sisa) - timedelta(hours=3)   # AEST -> WIB
        margin = jam(lf) - sisa
        status = f"berjalan, {dec(pct)}%"
        judul = "Perkiraan selesai"
        rencana = f"berjalan, {dec(pct)}%"
        ket = f"selesai {id_tanggal(selesai, '%A')}"
        stat_rmse = '<td class="muted">menunggu eval</td>'
        blok4 = (baris_dbh("11-spesies", "155k bersama", "64,02%",
                           final_test_metrics(EVAL11), FROZEN["11"], hi=True) + "\n    " +
                 baris_dbh("7-spesies", "155k bersama", f"{dec(seg[-1][1], 2)}%",
                           None, FROZEN["7"]))
        fn_bar = "Bar putus-putus bertanda tanya berarti run masih berjalan."
        fn_lin = "Run 7-spesies berhenti di iterasi berjalan; run 135k berakhir di 135.000 sesuai jadwalnya."
        leg7 = "berjalan"
    else:
        # Job sudah keluar dari antrean. Bedakan selesai wajar dari mati di tengah jalan:
        # menandai deck "selesai" untuk run yang di-cancel atau kehabisan waktu akan
        # menampilkan model yang tidak pernah ada.
        s = subprocess.run(["sacct", "-j", JOB, "-X", "-n", "-o", "State,Elapsed,End,Timelimit"],
                           capture_output=True, text=True).stdout.split()
        if not s:
            sys.exit(f"job {JOB} tidak ada di antrean maupun di sacct; deck tidak diperbarui")
        if s[0] != "COMPLETED":
            sys.exit(f"job {JOB} berstatus {s[0]}, bukan COMPLETED; deck tidak diperbarui")
        lama = jam(s[1])
        it = TARGET
        pct = 100.0
        laju = loss[-1][0] / lama
        selesai = datetime.strptime(s[2], "%Y-%m-%dT%H:%M:%S") - timedelta(hours=3)
        margin = jam(s[3]) - lama
        status = f"selesai, {int(lama // 24)} hari {dec(lama % 24)} jam"
        judul = "Selesai"
        rencana = "selesai"
        # Baris run final di slide 1. Sebelum eval per-checkpoint selesai, kolom diameter
        # berbunyi "menunggu eval"; sesudahnya diisi dari berkas eval. Checkpoint yang
        # dipakai adalah model_final.pth, mengikuti keputusan memprioritaskan AP, dan
        # SELURUH kolom pada baris ini berasal dari checkpoint yang sama itu — baris
        # Joint 11-spesies pernah mencampur AP50 dari 155k dengan RMSE dari 114.999.
        # TIDAK ada baris yang disorot di tabel ini. `tr.hi` menebalkan seluruh sel pada satu
        # baris, sedangkan penebalan di tabel ini berarti "nilai terbaik dalam pasangannya".
        # Dua arti pada satu tanda visual membuat sel yang kalah ikut tampak menang, jadi
        # sorotan baris dilepas dan penebalan diserahkan sepenuhnya ke `sel()` dan `persen()`.
        dbh = final_test_metrics(EVAL7)
        blok4 = (baris_dbh("11-spesies", "155k bersama", "64,02%",
                           final_test_metrics(EVAL11), FROZEN["11"]) + "\n    " +
                 baris_dbh("7-spesies", "155k bersama", f"{dec(seg[-1][1], 2)}%",
                           dbh, FROZEN["7"]))
        if dbh is None:
            fn_bar = ("Bar putus-putus bertanda tanya berarti angka diameter menunggu "
                      "eval per-checkpoint.")
            stat_rmse = '<td class="muted">menunggu eval</td>'
        else:
            fn_bar = ("Bar putus-putus bertanda tanya berarti model itu tidak menghasilkan "
                      "angka diameter.")
            stat_rmse = f'<td>{dec(dbh["RMSE"] / 10, 2)}</td>'
        fn_lin = "Run 7-spesies berakhir di 155.000, run 135k di 135.000, keduanya sesuai jadwal."
        leg7 = "selesai"
        # %B, bukan %b: peta nama bulan hanya memuat bentuk penuh, jadi %b lolos
        # sebagai "Aug" berbahasa Inggris.
        ket = f"selesai {id_tanggal(selesai, '%-d %B %H:%M')} WIB"
    # Kurva loss disubsample supaya blok data tetap ringan; AP50 sudah jarang.
    step = max(1, len(loss) // 60)
    run = dict(seg=seg, loss=loss[::step], it=it, target=TARGET)

    G = [
        # Slide status dihapus 14 Agt, digantikan slide lintasan yang datanya statis di
        # blok `const K`. Sepuluh pola yang dulu menyegarkan isinya ikut dilepas: bilah
        # kemajuan, laju iterasi/jam, waktu selesai, margin SLURM, sel iterasi dan AP50,
        # sel RMSE, dan blok data "run". Semuanya melaporkan kemajuan run yang sedang
        # berjalan, dan tidak ada lagi run yang berjalan.
        #
        # Slide rencana menyebut status yang sama; ini satu-satunya sisa yang masih hidup.
        (r'<td class="l">(?:berjalan, [\d,]+%|selesai)</td>', f'<td class="l">{rencana}</td>'),
        # Keterangan di sebelahnya ikut, kalau tidak slide rencana berbunyi "selesai" dan
        # "selesai Sabtu" berdampingan padahal harinya sudah lewat.
        (r'<td class="l muted">selesai [^<]*</td>', f'<td class="l muted">{ket}</td>'),
        # Baris run final pada tabel lima model di slide 1, plus dua footnote dan satu
        # label legend yang sama-sama menyebut keadaan run. Kalau ini tertinggal, deck
        # menyatakan "selesai" di tabel status sementara chart di slide lain masih
        # berlabel "berjalan".
        # Keempat baris DBH di slide 1 (joint/frozen x 11-sp/7-sp) ditulis ulang sekaligus.
        # Penebalan "nilai terbaik" bersifat lintas-baris di dalam tiap pasangan, jadi
        # mengganti satu baris saja bisa meninggalkan dua sel yang sama-sama tebal atau
        # sama-sama tidak. Baris "Tanpa head DBH" di atasnya tidak ikut, ia tidak punya
        # angka diameter sama sekali.
        # Pola menerima nama LAMA maupun BARU. Istilah diganti 14 Agt dari "Joint"/"Frozen"
        # ke "Multi-task"/"Two-stage"; tanpa alternasi ini skrip berhenti begitu penggantian
        # pertama tertulis, karena polanya tidak lagi menemukan teks yang dicarinya.
        (r'<tr[^>]*><td class="l">(?:Joint|Multi-task) 11-spesies</td>.*?'
         r'<td class="l">(?:Frozen|Two-stage) 7-spesies</td>.*?</tr>', blok4),
        (r'Bar putus-putus bertanda tanya berarti [^<]*\.', fn_bar),
        # Provenance checkpoint. Tanpa kalimat ini pembaca tidak bisa tahu bahwa kolom AP50
        # dan kolom diameter pada satu baris berasal dari weight yang sama — justru itu yang
        # dulu TIDAK berlaku pada baris Joint 11-spesies.
        # ⚠️ POLA INI WAJIB MENCAKUP SELURUH TEKS YANG DITULISNYA SENDIRI. Versi sebelumnya
        # hanya mencocokkan dua kalimat pertama sementara penggantinya menyisipkan kalimat
        # ketiga di tengah, sehingga tiap refresh menambah satu salinan lagi dan footnote
        # sempat memuat 28 salinan kalimat yang sama. Penjaga "cocok tepat sekali" tidak
        # menolongnya, karena polanya memang cocok sekali setiap kali dijalankan.
        # Jangkarnya kini kalimat berikutnya yang TIDAK ikut diganti, sehingga berapa pun
        # salinan yang sudah telanjur ada akan tersapu jadi satu.
        # `\s+` di jangkarnya, bukan spasi harfiah: teks berkas dipatah antar baris, dan
        # pola dengan spasi harfiah gagal cocok karena di situ ada pergantian baris.
        (r'Baris (?:frozen|two-stage) rerata 3 seed.*?(?=Populasi\s+identik\s+per\s+subset)',
         'Baris two-stage rerata 3 seed, multi-task 1 seed. Angka yang ditebalkan adalah yang '
         'lebih baik di dalam pasangannya, yaitu multi-task lawan two-stage pada subset spesies '
         'yang sama; kolom AP50 tidak ditebalkan karena bukan pokok perbandingan slide ini. '
         'Seluruh baris multi-task memakai checkpoint '
         'akhir 155k, weight yang sama dengan sumber angka AP50 di kolom sebelahnya. '),
        (r'<p class="footnote">Run 7-spesies [^<]*</p>', f'<p class="footnote">{fn_lin}</p>'),
        # `var L7` ikut hilang bersama slide status; legenda slide lintasan tidak lagi
        # menempelkan keterangan berjalan atau selesai pada nama run.
        # Array `model` pada blok data chart slide 1, diturunkan dari berkas eval yang SAMA
        # dengan tabel di slide itu. Sebelum didaftarkan di sini, chart dan tabel pernah
        # menampilkan angka berbeda untuk baris yang sama.
        (r'"model":\[.*?\}\],"',
         '"model":' + json.dumps(baris_model(dbh, final_test_metrics(EVAL11),
                                             round(seg[-1][1], 2)),
                                 separators=(",", ":")) + ',"'),
    ]

    h = DECK.read_text(encoding="utf-8")
    for pola, baru in G:
        h, n = re.subn(pola, lambda _m, b=baru: b, h, count=1, flags=re.S)
        if n != 1:
            sys.exit(f"pola cocok {n} kali, seharusnya 1: {pola[:60]}")

    DECK.write_text(h, encoding="utf-8")
    print(f"tersimpan: {DECK}")
    print(f"  iterasi {dec(it,0)} / 155.000 ({dec(pct)}%)")
    print(f"  status {status}, laju {dec(laju,0)} iterasi/jam")
    print(f"  selesai {selesai.strftime('%a %d %b %H:%M')} WIB, margin +{margin:.0f} jam")
    print(f"  chart: {len(run['loss'])} titik loss, {len(seg)} titik AP50 "
          f"(terakhir {dec(seg[-1][1],2)}% @{dec(seg[-1][0],0)})")


if __name__ == "__main__":
    main()
