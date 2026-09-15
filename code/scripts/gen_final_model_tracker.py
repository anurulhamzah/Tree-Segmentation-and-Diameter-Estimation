#!/usr/bin/env python3
"""gen_final_model_tracker.py — bangun ulang laporan pemantauan rencana model final.

Laporan menampilkan rencana bertahap menuju FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch
7-spesies, beserta status tiap tahap yang DIBACA LANGSUNG dari filesystem, bukan diketik
tangan. Jalankan ulang kapan saja untuk menyegarkan:

    python scripts/gen_final_model_tracker.py

Sumber status: direktori output di maskdino_output/ (checkpoint + metrics.json),
antrean SLURM, dan berkas hasil eval di reports/dbh_eval/.
"""
import json, os, re, subprocess
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path("/scratch2/pr65/anur0018")
OUT = ROOT / "maskdino_output"
PROJ = ROOT / "tree_classification"
REPORTS = PROJ / "reports"
TARGET = REPORTS / "stratified_v4" / "final_model_tracker.html"

# ── tahap yang dipantau ────────────────────────────────────────────────────────
STAGES = [
    dict(id="s1", nama="Tahap 1 · Uji hipotesis jadwal LR head",
         dir="FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_LRFIX_branch",
         job_match=["LRFIX"],
         iter_target=150000, iter_mulai=119999, biaya="23,4 jam (selesai)",
         slurm="slurm/train_joint_scratch_LRFIX_branch119999.slurm",
         tujuan="Cabang dari model_0119999 dengan head DBH dikecualikan dari decay. "
                "Mekanismenya TERBUKTI bekerja: baris [LR cek] menunjukkan head bertahan "
                "1e-04 melewati kedua decay sementara grup lain turun ke 1e-05 lalu 1e-06.",
         vonis=("tolak", "Hipotesis ditolak", """
Prediksi yang dinyatakan di muka: checkpoint akhir akan menyamai atau melampaui puncaknya.
Yang terjadi kebalikannya. Cabang kalah di <strong>7 dari 7</strong> titik test dan 6 dari 7
di val. Tiga selisih teratas jauh di luar noise band 0,58mm.
<table class="mini"><thead><tr><th>Test RMSE (mm)</th><th>124.999</th><th>129.999</th>
<th>134.999</th><th>139.999</th><th>144.999</th><th>149.999</th></tr></thead><tbody>
<tr><td class="lbl">Jadwal asli</td><td class="win">79,91</td><td class="win">81,83</td>
<td class="win">78,19</td><td class="win">80,05</td><td class="win">80,37</td>
<td class="win">80,43</td></tr>
<tr><td class="lbl">Head dikecualikan</td><td>84,14</td><td>87,10</td><td>82,82</td>
<td>81,83</td><td>81,11</td><td>81,87</td></tr></tbody></table>
Pola &quot;memuncak lalu turun&quot; tetap muncul <em>di dalam</em> cabang (puncak val
144.999, turun di 149.999), jadi decay LR head bukan penyebabnya. Segmentasi tidak berubah
pada kedua sisi (AP50 64,02 lawan 63,88 di 149.999), sesuai dugaan bahwa koplingnya lemah.
Tafsiran sekarang: saat backbone sudah membeku di 1e-6, head ber-LR tinggi mengejar
representasi yang sudah statis, dan itu merugikan. Decay-nya justru menolong."""),
         gate="SELESAI. Default script training dikembalikan ke decay penuh; jalur pengecualian "
              "dipertahankan di balik --freeze-dbh-head-lr agar cabang ini tetap reproducible. "
              "Checkpoint 154.999 hilang karena crash scheduler, tidak dipulihkan sebab "
              "kesimpulannya sudah tegas dari enam titik lain."),
    dict(id="s1b", nama="Tahap 1b · max_depth 20 vs 30 di joint",
         dir="FocalNet_L_trunk_roi_dbh_joint_ft20k_7sp_depth30",
         job_match=["jft20k_d20", "jft20k_d30", "ft20k_7sp"],
         iter_target=17499, iter_mulai=0, biaya="13 jam (selesai)",
         slurm="slurm/train_joint_ft20k_7sp_depth{20,30}.slurm",
         tujuan="Dimaksudkan sebagai dua fine-tune 20k 7-spesies yang hanya berbeda max_depth. "
                "Ternyata KEDUANYA tidak pernah memuat weights basis, jadi bukan fine-tune.",
         vonis=("cacat", "Uji cacat, bukti lemah", """
<strong>Kedua run ini tidak pernah memuat weights basis.</strong> Config
<code>..._from135k_20k.yaml</code> menyetel <code>WEIGHTS: &quot;&quot;</code> dan menunggu
diisi lewat <code>--model-weights</code>, tetapi kedua skrip SLURM tidak pernah meneruskan
argumen itu. Log menyatakannya terang-terangan:
<code>No checkpoint found. Initializing model from scratch</code>. Jadi keduanya model acak
yang dilatih 20k iterasi, bukan fine-tune. Terlihat jelas di AP50:
<table class="mini"><thead><tr><th>iter</th><th>depth20</th>
<th>from135k_7species_20k (fine-tune benar)</th></tr></thead><tbody>
<tr><td class="lbl">2.499</td><td>2,35%</td><td class="win">62,58%</td></tr>
<tr><td class="lbl">20k</td><td>19,44%</td><td class="win">64,22%</td></tr>
</tbody></table>
Dua cacat lain: keduanya memakai jadwal LR freeze-head yang kemudian ditolak Tahap 1, dan
keduanya crash di langkah terakhir oleh bug fvcore sehingga checkpoint 19.999 hilang.
<br><br>
Perbandingan A/B-nya sendiri tetap terkontrol (identik kecuali satu argumen) dan hasilnya
seri: val 91,26 lawan 90,89, test 94,73 lawan 94,94, di bawah noise 0,58mm dan berlawanan
arah. Tetapi itu seri di rezim jauh dari konvergen dan di bawah jadwal LR yang salah, jadi
statusnya <strong>indikasi lemah, bukan bukti</strong>. Jangan tulis di tesis sebagai hasil
uji."""),
         gate="Keputusan max_depth=30 pada run final TETAP BERLAKU, karena dasarnya cakupan "
              "(99,0% lawan 94,1% distribusi depth) dan bukan akurasi, sehingga tidak "
              "bergantung pada uji ini. Kalau mau diulang benar: tambahkan --model-weights dan "
              "pastikan AP50 titik pertama ~62%, bukan ~2%. Arsip di reports/run_archive/."),
    dict(id="s2", nama="Tahap 2 · Run final 7-spesies from-scratch",
         dir="FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_7species_155k",
         job_match=["FocL_joint_7sp_155k_final"],
         iter_target=155000, iter_mulai=0, biaya="≈4,4 hari",
         slurm="slurm/train_joint_scratch_7sp_155k_final.slurm (job 58750955)",
         tujuan="Model final. 7 spesies, from-scratch, max_depth=30, jadwal LR baku "
                "(head IKUT decay), DBH_WEIGHT=5,0, seleksi checkpoint pakai val.",
         gate="BERJALAN sejak 4 Agt 02:36 di m3g100. Kesehatan terpantau: raw_loss DBH turun "
              "8,69 ke 0,47 dan total_loss 1232 ke 128 pada 16k iterasi pertama, tanpa NaN. "
              "DISK: sisa 28 checkpoint × 2,7 GB = 76 GB. Kuota proyek 312130 terpakai 2,957T "
              "dari soft 3T dan hard 3,3T, jadi batas KERASNYA masih 350 GB dan tidak akan "
              "tersentuh. Yang terlampaui hanya soft quota, sekitar iterasi 95.000, yang "
              "memulai masa tenggang. Trimming tetap dilakukan agar tidak masuk masa tenggang "
              "sama sekali, bukan karena training terancam mati."),
    dict(id="s3", nama="Tahap 3 · Dua seed tambahan (konfigurasi sama)",
         dir="FocalNet_L_trunk_roi_dbh_hybrid_joint_scratch_7species_155k_seed1",
         job_match=["7sp_155k_seed"],
         iter_target=155000, iter_mulai=0, biaya="2 × 4,4 hari, dapat paralel",
         slurm="(dibuat setelah Tahap 2 selesai)",
         tujuan="Mengulang konfigurasi pemenang dengan seed 1 dan 2. Seluruh angka joint saat "
                "ini berasal dari 1 seed, sedangkan pembanding frozen memakai 3 seed.",
         gate="Menunggu Tahap 2. Prioritas DI ATAS uji weight, karena mengubah status hasil "
              "dari 'terukur sekali' menjadi dapat dipertahankan."),
    dict(id="s4", nama="Tahap 4 · Uji DBH_WEIGHT (opsional)",
         dir="FocalNet_L_trunk_roi_dbh_hybrid_joint_from135k_7species_W50",
         job_match=["7species_W"],
         iter_target=20000, iter_mulai=0, biaya="≈13,7 jam/run",
         slurm="(belum dibuat)",
         tujuan="Hanya kalau anggaran tersisa. Lewat fine-tune 20k, bukan from-scratch. "
                "Diturunkan prioritasnya karena bukti internal justru memperkirakan hasil "
                "negatif, sedangkan multi-seed pasti memperkuat setiap klaim yang sudah ada.",
         gate="Menunggu Tahap 3."),
]

def baca_iter(d: Path):
    """Iterasi terakhir + jumlah checkpoint, dibaca dari direktori output."""
    if not d.is_dir():
        return None
    ckpts = sorted(d.glob("model_0*.pth"))
    it = None
    mj = d / "metrics.json"
    if mj.exists():
        try:
            for line in mj.open():
                r = json.loads(line)
                if "iteration" in r:
                    it = max(it or 0, int(r["iteration"]))
        except Exception:
            pass
    if it is None and ckpts:
        m = re.search(r"model_0*(\d+)\.pth", ckpts[-1].name)
        it = int(m.group(1)) if m else None
    return dict(ada=True, iterasi=it, n_ckpt=len(ckpts),
                ukuran=sum(f.stat().st_size for f in d.glob("*.pth")) / 2**30)

def laju_dan_iter(st):
    """Baca laju detik/iterasi + iterasi terakhir dari log SLURM tahap ini.
    Dipakai memproyeksikan jadwal dari data yang benar-benar terukur, bukan angka katalog."""
    pola = {"s1": "joint_LRFIX_branch_*.out", "s2": "*7species_155k*.out", "s3": "*W50*.out"}
    logs = sorted((PROJ / "logs").glob(pola.get(st["id"], "___")), key=os.path.getmtime)
    if not logs:
        return None, None
    txt = logs[-1].read_text(errors="ignore")
    its = re.findall(r"iter: (\d+)", txt)
    tms = re.findall(r"time: ([0-9.]+)  last_time", txt)
    return (int(its[-1]) if its else None), (float(tms[-1]) if tms else None)


def jadwal():
    """Proyeksi jadwal seluruh tahap, berbasis laju terukur tahap yang sedang jalan."""
    it, sec = laju_dan_iter(STAGES[0])
    if not (it and sec):
        return None
    now = datetime.now()
    t1 = now + timedelta(seconds=(STAGES[0]["iter_target"] - it) * sec)
    ev1 = t1 + timedelta(hours=3.5)
    t2 = ev1 + timedelta(days=4.4)
    ev2 = t2 + timedelta(hours=7)
    t3 = ev2 + timedelta(hours=14)
    f = lambda d: d.strftime("%a %d %b, %H:%M")
    return dict(laju=sec, iter=it, rows=[
        ("Sekarang", f(now), f"Tahap 1 di iterasi {it:,}".replace(",", "."), "berjalan"),
        ("Iterasi 125.000", f(now + timedelta(seconds=(125000 - it) * sec)),
         "Baris [LR cek] pertama pasca-decay muncul", "GO/NO-GO, cukup baca log"),
        ("Tahap 1 selesai", f(t1), "155.000 iterasi tuntas", "otomatis"),
        ("Eval Tahap 1", f(ev1), "8 checkpoint, dual-split, via SLURM", "≈3,5 jam"),
        ("Keputusan jadwal LR", f(ev1), "Menentukan konfigurasi Tahap 2 dan jumlah run Tahap 3",
         "PERLU KEPUTUSAN"),
        ("Tahap 2 selesai", f(t2), "Run final 7-spesies from-scratch", "≈4,4 hari"),
        ("Eval Tahap 2", f(ev2), "32 checkpoint", "≈7 jam"),
        ("Tahap 3: seed 1 & 2", f(ev2 + timedelta(days=4.4)),
         "Ulang konfigurasi pemenang, 2 seed paralel", "≈4,4 hari"),
        ("Eval Tahap 3", f(ev2 + timedelta(days=4.4, hours=7)),
         "Sebaran antar-seed untuk RMSE dan AP50", "≈7 jam"),
        ("Tahap 4 opsional", f(ev2 + timedelta(days=5, hours=7)),
         "3 run weight DBH (w=5/50/125), paralel", "≈14 jam"),
        ("Keputusan weight DBH", f(ev2 + timedelta(days=5, hours=14)),
         "Menentukan DBH_WEIGHT, memakai aturan threshold di bawah", "PERLU KEPUTUSAN"),
    ])


def squeue():
    try:
        o = subprocess.run(["squeue", "-u", "anur0018", "-h", "-o", "%i|%j|%T|%M"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        return [l.split("|") for l in o.splitlines() if l.strip()]
    except Exception:
        return []

def status(st):
    info = baca_iter(OUT / st["dir"])
    # Cocokkan job lewat tag EKSPLISIT yang ditulis di STAGES, bukan potongan nama
    # direktori. Heuristik lama (`dir[:20] in nama_job`) diam-diam gagal untuk Tahap 2:
    # potongannya "FocalNet_L_trunk_roi" sedangkan nama job "FocL_joint_7sp_155k_final",
    # sehingga run yang sedang berjalan dilaporkan "Terhenti sebagian".
    pola = [p.lower() for p in st.get("job_match", [])]
    jobs = [j for j in squeue() if any(p in j[1].lower() for p in pola)]
    if info and info["iterasi"] and info["iterasi"] >= st["iter_target"] - 1001:
        return "selesai", info, jobs
    if jobs:
        return "jalan", info, jobs
    if info:
        return "sebagian", info, jobs
    return "belum", None, jobs

BADGE = {"selesai": ("Selesai", "ok"), "jalan": ("Sedang berjalan", "run"),
         "sebagian": ("Terhenti sebagian", "warn"), "belum": ("Belum dimulai", "idle")}

rows = []
for st in STAGES:
    s, info, jobs = status(st)
    lab, cls = BADGE[s]
    if info and info["iterasi"] is not None:
        span = st["iter_target"] - st["iter_mulai"]
        prog = max(0.0, min(1.0, (info["iterasi"] - st["iter_mulai"]) / span))
        det = (f"iterasi {info['iterasi']:,} dari {st['iter_target']:,}".replace(",", ".")
               + f" · {info['n_ckpt']} checkpoint · {info['ukuran']:.1f} GB")
    else:
        prog, det = 0.0, "direktori output belum ada"
    jd = "; ".join(f"job {j[0]} ({j[2]}, {j[3]})" for j in jobs) if jobs else ""
    # Vonis ditulis tangan setelah eval dibaca; ia menyatakan APA ARTINYA hasil itu,
    # sesuatu yang tidak bisa disimpulkan dari status filesystem.
    v = st.get("vonis")
    vonis_html = ("" if not v else
                  f'<div class="vonis {v[0]}"><span class="vonis-tag">{v[1]}</span>'
                  f'<div class="vonis-body">{v[2].strip()}</div></div>')
    rows.append(dict(st=st, kelas=cls, label=lab, prog=prog, detail=det, job=jd,
                     vonis=vonis_html))

now = datetime.now().strftime("%d %B %Y, %H:%M AEST")
prog_rows = "\n".join(f'''
      <div class="stage {r["kelas"]}">
        <div class="stage-head">
          <span class="stage-name">{r["st"]["nama"]}</span>
          <span class="badge {r["kelas"]}">{r["label"]}</span>
        </div>
        <p class="stage-goal">{r["st"]["tujuan"]}</p>
        <div class="bar"><div class="fill" style="width:{r["prog"]*100:.1f}%"></div></div>
        <p class="stage-det">{r["detail"]} · biaya {r["st"]["biaya"]}
           {"· " + r["job"] if r["job"] else ""}</p>
        {r["vonis"]}
        <p class="stage-gate"><strong>Syarat lanjut:</strong> {r["st"]["gate"]}</p>
        <p class="stage-src">Skrip: <code>{r["st"]["slurm"]}</code> · output:
           <code>{r["st"]["dir"]}</code></p>
      </div>''' for r in rows)

jd = jadwal()
if jd:
    baris = "\n".join(
        f'        <tr><td class="lbl">{a}</td><td class="lbl strong">{b}</td>'
        f'<td class="lbl">{c}</td><td class="lbl muted">{d}</td></tr>'
        for a, b, c, d in jd["rows"])
    jadwal_html = f'''<table>
      <thead><tr><th class="lbl">Tonggak</th><th class="lbl">Perkiraan</th><th class="lbl">Kegiatan</th><th class="lbl">Sifat</th></tr></thead>
      <tbody>
{baris}
      </tbody>
    </table>
    <p class="foot">Proyeksi memakai laju <strong>{jd["laju"]:.2f} detik per iterasi</strong> yang
      terukur dari run Tahap 1 yang sedang berjalan, bukan angka katalog. Kalau node Tahap 2 lebih
      lambat atau berbagi beban, angka 4,4 hari dapat molor.</p>'''
else:
    jadwal_html = ('<p class="foot">Jadwal belum dapat diproyeksikan: belum ada log Tahap 1 '
                   'yang memuat laju iterasi.</p>')

HTML = open(PROJ / "scripts" / "_tracker_template.html", encoding="utf-8").read()
HTML = (HTML.replace("__STAGES__", prog_rows).replace("__UPDATED__", now)
            .replace("__JADWAL__", jadwal_html))
TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(HTML, encoding="utf-8")
print(f"tersimpan: {TARGET}")
for r in rows:
    print(f"  {r['st']['nama'][:44]:<46} {r['label']:<18} {r['detail']}")
