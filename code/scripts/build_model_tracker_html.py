#!/usr/bin/env python
"""Build the standalone Model Selection & Fine-Tuning Tracker HTML document.
Self-contained: embeds 5 figures as base64 PNG. No external deps beyond stdlib."""
import base64
import os

FIGDIR = "/scratch2/pr65/anur0018/tree_classification/reports/figures"
OUT = "/scratch2/pr65/anur0018/tree_classification/paper/model_finetune_tracker.html"


def b64(name):
    with open(os.path.join(FIGDIR, name), "rb") as f:
        return base64.b64encode(f.read()).decode()


IMG_LADDER = b64("fig_seg_ladder.png")
IMG_DBH_PROG = b64("fig_dbh_progression.png")
IMG_RES_TEST = b64("fig_resolution_test.png")
IMG_SIZE_OCC = b64("fig_size_occlusion.png")
IMG_THR_SWEEP = b64("threshold_sweep_focalnetl.png")

HTML = f"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Selection &amp; Fine-Tuning Tracker</title>
<style>
:root{{
  --ink:#1a1a1a; --muted:#5a5a5a; --rule:#dcdcdc; --bg:#ffffff; --card:#f7f8fa;
  --blue:#0072B2; --orange:#D55E00; --green:#009E73; --sky:#56B4E9; --amber:#E69F00; --grey:#999999;
  --blue-bg:#eaf3fb; --orange-bg:#fdf0ea; --green-bg:#e8f6f1; --grey-bg:#f0f0f0;
}}
*{{box-sizing:border-box}}
html,body{{background:var(--bg)}}
body{{
  color:var(--ink); max-width:1080px; margin:0 auto; padding:2rem 1.2rem 5rem;
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
}}
h1{{font-size:1.85rem; margin:.1em 0 .2em}}
h1 .sub{{display:block; font-size:1rem; font-weight:400; color:var(--muted); margin-top:.3em}}
h2{{font-size:1.35rem; border-bottom:2px solid var(--rule); padding-bottom:.3em; margin-top:2.6em}}
h3{{font-size:1.08rem; margin-top:1.6em; color:var(--ink)}}
p{{margin:.6em 0}}
.meta{{color:var(--muted); font-size:.88rem; margin-bottom:1.6em}}
a{{color:var(--blue)}}
code{{background:var(--card); padding:.1em .4em; border-radius:4px; font-size:.85em; word-break:break-all}}

/* TL;DR cards */
.tldr{{display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1.4em 0 2em}}
@media (max-width:720px){{.tldr{{grid-template-columns:1fr}}}}
.card{{background:var(--card); border:1px solid var(--rule); border-radius:4px; padding:1.1rem 1.3rem}}
.card.seg{{border-top:3px solid var(--blue)}}
.card.dbh{{border-top:3px solid var(--green)}}
.card h3{{margin:0 0 .5em; font-size:1.02rem}}
.card .metric{{font-size:1.5rem; font-weight:700}}
.card .metric small{{font-size:.6em; font-weight:400; color:var(--muted)}}
.card .rows{{font-size:.86rem; color:var(--muted); margin-top:.5em; line-height:1.7}}
.card .rows b{{color:var(--ink); font-weight:600}}
.badge{{display:block; font-size:.72rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); border-bottom:1px solid var(--rule); padding-bottom:.4em; margin-bottom:.6em}}

/* pipeline flow diagram */
.flow{{margin:1.6em 0 2.2em; overflow-x:auto; padding-bottom:.4em}}
.flow-row{{display:flex; align-items:stretch; gap:0; min-width:max-content}}
.flow-box{{
  border-radius:4px; padding:.7rem .85rem; min-width:132px; max-width:160px;
  font-size:.8rem; line-height:1.35; display:flex; flex-direction:column; justify-content:center;
  border:1.5px solid; text-align:center;
}}
.flow-box b{{display:block; font-size:.84rem; margin-bottom:.15em}}
.flow-box .tag{{display:block; font-size:.68rem; color:var(--muted); margin-top:.3em}}
.flow-arrow{{display:flex; align-items:center; justify-content:center; min-width:28px; color:var(--muted); font-size:1.1rem}}
.fb-data{{background:var(--grey-bg); border-color:var(--grey)}}
.fb-arch{{background:var(--blue-bg); border-color:var(--blue)}}
.fb-ft{{background:var(--orange-bg); border-color:var(--orange)}}
.fb-final{{background:var(--green-bg); border-color:var(--green); border-width:2.5px}}
.fb-dbh{{background:var(--green-bg); border-color:var(--green)}}
.fb-diag{{background:var(--orange-bg); border-color:var(--orange)}}
.flow-branch-label{{font-size:.72rem; color:var(--muted); font-weight:600; margin:.9em 0 .3em; padding-left:.2em}}

/* tables */
table{{border-collapse:collapse; width:100%; font-size:.85rem; margin:1em 0}}
th,td{{padding:.4em .6em; text-align:right; border-bottom:1px solid var(--rule)}}
th{{border-bottom:2px solid var(--ink); font-weight:600; text-align:right; color:var(--muted); font-size:.78rem;
   text-transform:uppercase; letter-spacing:.02em}}
td:first-child,th:first-child{{text-align:left}}
tbody tr:hover{{background:var(--card)}}
tr.best td{{font-weight:700; color:var(--green)}}
tr.best td:first-child::before{{content:"\\2605  "; color:var(--green)}}
tr.control td{{font-style:italic; color:var(--muted)}}
tr.group-head td{{background:var(--card); font-weight:700; color:var(--muted); font-size:.78rem;
  text-transform:uppercase; letter-spacing:.03em; padding-top:.7em}}
.twrap{{overflow-x:auto}}

/* figures */
figure{{margin:1.6em 0; text-align:center}}
figure img{{max-width:100%; height:auto; border:1px solid var(--rule); border-radius:4px}}
figcaption{{font-size:.82rem; color:var(--muted); margin-top:.5em; text-align:left}}

/* timeline */
.timeline{{border-left:2px solid var(--rule); margin:1.2em 0 1.2em .4em; padding-left:1.3em}}
.tl-item{{position:relative; margin-bottom:1.1em}}
.tl-item::before{{content:""; position:absolute; left:-1.75em; top:.3em; width:9px; height:9px;
  border-radius:50%; background:var(--blue)}}
.tl-item.milestone::before{{background:var(--green); width:12px; height:12px; left:-1.9em}}
.tl-date{{font-size:.76rem; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.03em}}
.tl-title{{font-weight:600; margin:.1em 0}}
.tl-desc{{font-size:.88rem; color:var(--muted)}}

.note{{background:var(--orange-bg); border-left:3px solid var(--orange); padding:.6em 1em; border-radius:2px;
  font-size:.87rem; margin:1em 0}}
.footer{{margin-top:3.5em; padding-top:1.2em; border-top:1px solid var(--rule); font-size:.8rem; color:var(--muted)}}
.footer code{{font-size:.9em}}
ul.src{{font-size:.83rem; color:var(--muted); line-height:1.8}}
</style>
</head>
<body>

<h1>Model Selection &amp; Fine-Tuning Tracker
<span class="sub">Tree instance segmentation (MaskDINO) + diameter estimation (DBH) &mdash; SPREAD synthetic dataset</span></h1>
<div class="meta">Terakhir diperbarui: 15 Juli 2026 &middot; Proyek: <code>tree_classification</code> &middot;
Dokumen ini melacak seluruh proses pemilihan model final dan fine-tuning yang telah dilakukan, dari ablasi arsitektur awal hingga diagnosis bottleneck terbaru.</div>

<div class="tldr">
  <div class="card seg">
    <span class="badge">MODEL SEGMENTASI FINAL</span>
    <h3>FocalNet-L (RGBD scratch) + Mosaic + NoObj + Class-reweighting</h3>
    <div class="metric">61.17% <small>Test AP50</small></div>
    <div class="rows">
      <b>F1@0.35:</b> 63.85% &nbsp;|&nbsp; <b>Recall:</b> 57.39% &nbsp;|&nbsp; <b>Precision:</b> 71.95%<br>
      <b>Checkpoint:</b><br><code>maskdino_output/FocalNet_L_combined_rgbd_scratch_v4_stratified_mosaic_classweight_20k/model_final.pth</code>
    </div>
  </div>
  <div class="card dbh">
    <span class="badge">MODEL DBH FINAL</span>
    <h3>Hybrid Trunk-ROI Head &mdash; DBH Sandbox winner (11 spesies)</h3>
    <div class="metric">+0.136 <small>mean per-species R&sup2; (3-seed)</small></div>
    <div class="rows">
      <b>Aggregate R&sup2;:</b> 0.757 &nbsp;|&nbsp; <b>MAE:</b> ~52mm &nbsp;|&nbsp; <b>Within 100mm:</b> ~87%<br>
      <b>Config:</b> <code>phaseB_final_baseline_anchor_only</code> &mdash; strip_rows=5, dropout=0.3, hidden=256, lr=1e-4
    </div>
  </div>
</div>

<div class="note">
<b>Temuan terbaru (14&ndash;15 Jul):</b> ceiling recall model segmentasi (~57&ndash;67%) BUKAN disebabkan resolusi
input &mdash; tiga uji inference-only (native 960px, naikkan resolusi proses, tiling) semua GAGAL menaikkan
recall. Analisis per-instance menunjukkan bottleneck sebenarnya adalah <b>under-detection akibat oklusi</b>
pada kanopi ukuran medium di scene padat, bukan objek kecil (yang cuma 1,7% instance). Lihat
<a href="#diagnosis">Bagian 5</a>.
</div>

<h2>1. Gambaran Umum Pipeline</h2>
<p>Diagram berikut merangkum seluruh alur kegiatan, dari persiapan dataset hingga model final dan
investigasi diagnostik terbaru. Setiap kotak berwarna menunjukkan tahap; kotak hijau tebal adalah
output akhir (model final) yang dipilih untuk deployment/pelaporan.</p>

<div class="flow">
  <div class="flow-row">
    <div class="flow-box fb-data"><b>SPREAD Dataset</b>13 spesies, RGB-D, whole-tree mask<span class="tag">V4 stratified split, repair 416 ann.</span></div>
    <div class="flow-arrow">&#8594;</div>
    <div class="flow-box fb-arch"><b>Ablasi Arsitektur</b>Mask R-CNN vs MaskDINO<span class="tag">R50 backbone, RGB-pt: 15.89% vs 40.20%</span></div>
    <div class="flow-arrow">&#8594;</div>
    <div class="flow-box fb-arch"><b>Ablasi Backbone &amp; Modalitas</b>R50/SwinT/B/L/FocalNet-B/T/L &times; RGB/RGBD &times; pretrain/scratch<span class="tag">24 run, PL/RF/Combined domain</span></div>
    <div class="flow-arrow">&#8594;</div>
    <div class="flow-box fb-ft"><b>Base Terpilih</b>FocalNet-L RGBD Scratch<span class="tag">Test AP50 = 57.18%</span></div>
    <div class="flow-arrow">&#8594;</div>
    <div class="flow-box fb-ft"><b>Fine-tune Chain</b>Mosaic &rarr; NoObj &rarr; ClassWeight<span class="tag">+ kontrol atribusi kausal</span></div>
    <div class="flow-arrow">&#8594;</div>
    <div class="flow-box fb-final"><b>&#9733; MODEL SEGMENTASI FINAL</b>ClassWeight 20k<span class="tag">Test AP50 = 61.17%</span></div>
  </div>

  <div class="flow-branch-label">&#9492;&#9472; cabang DBH (dari model final di atas, backbone dibekukan)</div>
  <div class="flow-row">
    <div class="flow-box fb-dbh"><b>DBH Head V1&ndash;V2</b>Geometric &rarr; gagal (R&sup2;=&minus;5.28)<span class="tag">Neural head awal: R&sup2;=0.149</span></div>
    <div class="flow-arrow">&#8594;</div>
    <div class="flow-box fb-dbh"><b>Hybrid V2&ndash;V6</b>strip_rows sweep + data cleaning<span class="tag">Per-sp R&sup2;: &minus;0.42 &rarr; &minus;0.13</span></div>
    <div class="flow-arrow">&#8594;</div>
    <div class="flow-box fb-dbh"><b>DBH Sandbox</b>Cache-based 14-axis search, 11 spesies<span class="tag">Fase 0.5&rarr;A&rarr;B&rarr;C</span></div>
    <div class="flow-arrow">&#8594;</div>
    <div class="flow-box fb-final"><b>&#9733; MODEL DBH FINAL</b>phaseB_final_baseline_anchor_only<span class="tag">Per-sp R&sup2; = +0.136</span></div>
  </div>

  <div class="flow-branch-label">&#9492;&#9472; cabang diagnosis bottleneck (dari model segmentasi final, 14&ndash;15 Jul)</div>
  <div class="flow-row">
    <div class="flow-box fb-diag"><b>Uji Resolusi</b>Native 960px, naikkan proc-res, tiling<span class="tag">Ketiganya GAGAL/rusak</span></div>
    <div class="flow-arrow">&#8594;</div>
    <div class="flow-box fb-diag"><b>Analisis Near-miss</b>Mask-IoU per ukuran objek<span class="tag">93% true-miss = kanopi medium</span></div>
    <div class="flow-arrow">&#8594;</div>
    <div class="flow-box fb-diag"><b>Threshold Sweep</b>F1 vs skor, IoU=0.5<span class="tag">F1 optimal @0.35, ceiling recall ~0.67</span></div>
    <div class="flow-arrow">&#8594;</div>
    <div class="flow-box fb-ft"><b>Kesimpulan</b>Bottleneck = oklusi, bukan resolusi<span class="tag">Arah riset berikutnya</span></div>
  </div>
</div>

<h2>2. Timeline Kegiatan</h2>
<div class="timeline">
  <div class="tl-item"><div class="tl-date">29 Jun &ndash; 1 Jul 2026</div>
    <div class="tl-title">Ablasi arsitektur &amp; backbone/modalitas (matriks V4)</div>
    <div class="tl-desc">24 konfigurasi dilatih: Mask R-CNN vs MaskDINO; backbone R50/SwinT/B/L/FocalNet-B/T/L; RGB vs RGBD; pretrained vs scratch; domain Plantation/Rainforest/Combined. Base terbaik: FocalNet-L RGBD Scratch, test AP50=57.18% (job 57963254).</div></div>
  <div class="tl-item"><div class="tl-date">3&ndash;4 Jul 2026</div>
    <div class="tl-title">DBH percobaan awal + pelengkap ablasi</div>
    <div class="tl-desc">DBH fine-tune head pertama (R&sup2;=0.149) dan Trunk ROI murni (R&sup2;=&minus;0.075, gagal). Model backbone tambahan (FocalNet-B/T, RGB pretrain/scratch varian) melengkapi ladder.</div></div>
  <div class="tl-item"><div class="tl-date">4&ndash;8 Jul 2026</div>
    <div class="tl-title">Progresi DBH Hybrid V2&rarr;V6</div>
    <div class="tl-desc">Sweep strip_rows (1&rarr;3&rarr;5) + dataset cleaning bertahap. Mean per-spesies R&sup2; naik dari &minus;0.416 (V2) ke &minus;0.125 (V6), tapi tetap negatif (6 spesies saja).</div></div>
  <div class="tl-item"><div class="tl-date">4&ndash;5 Jul 2026</div>
    <div class="tl-title">Fine-tune tahap 1&ndash;2: Mosaic augmentation</div>
    <div class="tl-desc">ft20k &rarr; ext20k (mosaic 4-gambar, expose model ke oklusi lebih berat). Base AP50 57.18% belum berubah signifikan di tahap ini (masih warmup augmentasi).</div></div>
  <div class="tl-item milestone"><div class="tl-date">5 Jul 2026</div>
    <div class="tl-title">Fine-tune tahap 3: NoObj weight turun 0.1&rarr;0.05</div>
    <div class="tl-desc">Test AP50 naik ke 60.02%, F1 63.01%, Recall 56.42% &mdash; gain pertama yang jelas dari fine-tuning.</div></div>
  <div class="tl-item milestone"><div class="tl-date">7 Jul 2026</div>
    <div class="tl-title">Fine-tune tahap 4: Class reweighting &mdash; MODEL FINAL</div>
    <div class="tl-desc">Test AP50 naik ke <b>61.17%</b>, F1 63.85%, Recall 57.39%. Ditetapkan sebagai model segmentasi deployment final.</div></div>
  <div class="tl-item"><div class="tl-date">8 Jul 2026</div>
    <div class="tl-title">Kontrol atribusi kausal + baseline Mask R-CNN</div>
    <div class="tl-desc">Run kontrol (config identik tanpa class-weight) untuk memisahkan efek reweighting vs iterasi tambahan. Baseline Mask R-CNN RGB-pt (15.89%) dan RGBD-scratch (6.86%) ditambahkan untuk melengkapi ladder.</div></div>
  <div class="tl-item"><div class="tl-date">11&ndash;12 Jul 2026</div>
    <div class="tl-title">DBH Sandbox dibangun &amp; dijalankan</div>
    <div class="tl-desc">Infrastruktur cache-based sweep 14 axis, universe 11 spesies. Fase 0.5 (kalibrasi) &rarr; Fase A (OFAT screening) &rarr; Fase B (kombinasi). Bug metrik ranking ditemukan &amp; diperbaiki di tengah jalan (equal-weight R&sup2; tak stabil utk N kecil).</div></div>
  <div class="tl-item milestone"><div class="tl-date">12 Jul 2026</div>
    <div class="tl-title">DBH Sandbox Fase C: juara terkonfirmasi &mdash; MODEL DBH FINAL</div>
    <div class="tl-desc">Konfirmasi 3-seed pada test split: <b>mean per-species R&sup2;=+0.136</b>, aggregate R&sup2;=0.757, MAE~52mm. Menggantikan V6 sebagai model DBH terbaik.</div></div>
  <div class="tl-item"><div class="tl-date">14 Jul 2026</div>
    <div class="tl-title">Investigasi bottleneck resolusi &mdash; 3 eksperimen, semua negatif</div>
    <div class="tl-desc">(a) Native 960px pada resolusi proses terlatih: +0.6pp saja. (b) Naikkan resolusi proses tanpa retrain: recall turun (mismatch skala train/test). (c) Tiling/SAHI: recall anjlok &minus;44.5pp (memfragmentasi mask whole-tree).</div></div>
  <div class="tl-item milestone"><div class="tl-date">14 Jul 2026</div>
    <div class="tl-title">Analisis near-miss/oklusi &mdash; akar masalah ditemukan</div>
    <div class="tl-desc">Breakdown mask-IoU per ukuran objek: objek kecil hanya 1,7% instance, kanopi besar 90,8% terdeteksi. True-miss 93% adalah kanopi <b>medium</b> yang under-detected di scene padat &mdash; bottleneck = oklusi, bukan resolusi.</div></div>
  <div class="tl-item"><div class="tl-date">15 Jul 2026</div>
    <div class="tl-title">Threshold sweep + penulisan tesis + verifikasi literatur</div>
    <div class="tl-desc">Kurva F1 vs skor: optimal di threshold 0.35 (sudah jadi operating point saat ini), recall ceiling ~0.67. Tesis lengkap ditulis, seluruh data diaudit ke file sumber, literatur diverifikasi via perpustakaan PDF lokal.</div></div>
</div>

<h2>3. Pemilihan Model Segmentasi</h2>
<p>Ablasi menyeluruh menguji dua arsitektur, tiga skala backbone, dan empat kombinasi
modalitas-inisialisasi. FocalNet-L dengan input RGB-D yang dilatih dari awal (scratch) terpilih
sebagai basis terbaik sebelum fine-tuning, mengalahkan seluruh varian Swin dan ResNet-50.</p>

<figure>
  <img src="data:image/png;base64,{IMG_LADDER}" alt="Segmentation performance ladder">
  <figcaption><b>Gambar 1.</b> Ladder performa AP50 pada test set (843 gambar), dari Mask R-CNN
  hingga model final. Lompatan terbesar terjadi saat transisi arsitektur (Mask R-CNN&rarr;MaskDINO,
  +24,3pp pada kondisi setara).</figcaption>
</figure>

<div class="twrap">
<table>
<thead><tr><th>Model</th><th>Init</th><th>Val AP50</th><th>Test AP50</th><th>F1@0.35</th><th>Recall</th><th>Job ID</th></tr></thead>
<tbody>
<tr class="group-head"><td colspan="7">Baseline arsitektur (R50, kondisi setara)</td></tr>
<tr><td>Mask R-CNN R50</td><td>RGBD scratch</td><td>&mdash;</td><td>6.86%</td><td>11.39%</td><td>14.11%</td><td>58199916</td></tr>
<tr><td>Mask R-CNN R50</td><td>RGB pretrained</td><td>&mdash;</td><td>15.89%</td><td>22.81%</td><td>22.61%</td><td>58199915</td></tr>
<tr><td>MaskDINO R50</td><td>RGB scratch</td><td>33.55%</td><td>32.00%</td><td>40.01%</td><td>29.97%</td><td>N/A</td></tr>
<tr><td>MaskDINO R50</td><td>RGB pretrained</td><td>42.25%</td><td>40.20%</td><td>47.95%</td><td>41.05%</td><td>57920933</td></tr>
<tr class="group-head"><td colspan="7">Skala backbone &amp; modalitas (Combined domain)</td></tr>
<tr><td>MaskDINO SwinB</td><td>RGB scratch</td><td>41.19%</td><td>38.95%</td><td>46.13%</td><td>35.85%</td><td>57921562</td></tr>
<tr><td>MaskDINO SwinB</td><td>RGBD pretrained</td><td>40.72%</td><td>40.16%</td><td>46.90%</td><td>36.87%</td><td>57950142</td></tr>
<tr><td>MaskDINO SwinB</td><td>RGB pretrained</td><td>46.80%</td><td>44.23%</td><td>51.67%</td><td>43.38%</td><td>57921028</td></tr>
<tr><td>MaskDINO SwinB</td><td>RGBD scratch</td><td>49.81%</td><td>48.63%</td><td>51.62%</td><td>41.03%</td><td>57939427</td></tr>
<tr><td>MaskDINO SwinT</td><td>RGB pretrained</td><td>45.99%</td><td>43.47%</td><td>&mdash;</td><td>&mdash;</td><td>57986170</td></tr>
<tr><td>MaskDINO SwinL</td><td>RGB pretrained</td><td>44.90%</td><td>43.77%</td><td>51.22%</td><td>41.91%</td><td>57950207</td></tr>
<tr><td>MaskDINO SwinL</td><td>RGBD scratch</td><td>52.38%</td><td>50.82%</td><td>53.33%</td><td>43.13%</td><td>57973055</td></tr>
<tr><td>MaskDINO FocalNet-B</td><td>RGB pretrained</td><td>43.83%</td><td>41.74%</td><td>50.08%</td><td>40.83%</td><td>N/A</td></tr>
<tr><td>MaskDINO FocalNet-T</td><td>RGB pretrained</td><td>44.45%</td><td>41.70%</td><td>49.60%</td><td>40.20%</td><td>N/A</td></tr>
<tr><td>MaskDINO FocalNet-L</td><td>RGB scratch</td><td>45.98%</td><td>44.25%</td><td>52.05%</td><td>44.23%</td><td>58073777</td></tr>
<tr><td>MaskDINO FocalNet-L</td><td>RGBD pretrained</td><td>52.69%</td><td>49.45%</td><td>55.11%</td><td>49.41%</td><td>58008966</td></tr>
<tr><td>MaskDINO FocalNet-L</td><td>RGB pretrained</td><td>50.43%</td><td>48.54%</td><td>55.19%</td><td>50.88%</td><td>58045233</td></tr>
<tr class="best"><td>MaskDINO FocalNet-L</td><td>RGBD scratch (base)</td><td>60.69%</td><td>57.18%</td><td>61.03%</td><td>52.82%</td><td>57963254</td></tr>
<tr class="group-head"><td colspan="7">Domain-spesifik (referensi, bukan kandidat deployment combined)</td></tr>
<tr><td>MaskDINO SwinB</td><td>Plantation, RGB pretrained</td><td>56.28%</td><td>53.36%</td><td>61.79%</td><td>59.34%</td><td>57920735</td></tr>
<tr><td>MaskDINO FocalNet-L</td><td>Plantation, RGB pretrained</td><td>59.57%</td><td>57.87%</td><td>65.32%</td><td>64.30%</td><td>58074012</td></tr>
<tr><td>MaskDINO FocalNet-L</td><td>Plantation, RGBD scratch</td><td>78.14%</td><td>74.06%</td><td>81.29%</td><td>81.32%</td><td>58008967</td></tr>
<tr><td>MaskDINO FocalNet-L</td><td>Rainforest, RGBD scratch</td><td>45.73%</td><td>42.23%</td><td>53.86%</td><td>45.95%</td><td>58008968</td></tr>
<tr><td>MaskDINO SwinB</td><td>Combined, RGBD+DBH pretrained</td><td>41.04%</td><td>40.09%</td><td>46.20%</td><td>36.76%</td><td>57951925</td></tr>
</tbody>
</table>
</div>

<h2>4. Rantai Fine-tuning Model Segmentasi</h2>
<p>Base model (FocalNet-L RGBD scratch, 57,18%) diperbaiki melalui empat tahap fine-tuning
sekuensial, masing-masing melanjutkan checkpoint tahap sebelumnya:</p>

<div class="twrap">
<table>
<thead><tr><th>Tahap</th><th>Intervensi</th><th>Test AP50</th><th>F1@0.35</th><th>Recall</th><th>Delta AP50</th></tr></thead>
<tbody>
<tr><td>0 (base)</td><td>&mdash;</td><td>57.18%</td><td>61.03%</td><td>52.82%</td><td>&mdash;</td></tr>
<tr><td>1&ndash;2</td><td>Mosaic augmentation (ft20k &rarr; ext20k)</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>warmup</td></tr>
<tr><td>3</td><td>+ NO_OBJECT_WEIGHT 0.1&rarr;0.05</td><td>60.02%</td><td>63.01%</td><td>56.42%</td><td>+2.84pp</td></tr>
<tr class="best"><td>4 (final)</td><td>+ Per-class loss reweighting</td><td>61.17%</td><td>63.85%</td><td>57.39%</td><td>+1.15pp</td></tr>
<tr class="control"><td>4 (kontrol)</td><td>Config identik TANPA reweighting</td><td>60.35%</td><td>63.80%</td><td>57.21%</td><td>+0.33pp*</td></tr>
</tbody>
</table>
</div>
<p style="font-size:.85rem;color:var(--muted)">*Kontrol dijalankan paralel dgn tahap final utk isolasi kausal &mdash; iterasi tambahan sama, hanya reweighting yang beda.</p>

<h3>Atribusi kausal (final vs kontrol)</h3>
<p>Karena kontrol memakai konfigurasi identik minus reweighting, gain dari tahap final bisa
dipisah menjadi efek "iterasi tambahan" vs efek "reweighting murni":</p>
<div class="twrap">
<table>
<thead><tr><th>Metrik</th><th>% dari iterasi tambahan</th><th>% dari reweighting</th></tr></thead>
<tbody>
<tr><td>F1@0.35 (operasional)</td><td>94%</td><td>6%</td></tr>
<tr><td>Recall@0.35</td><td>81%</td><td>19%</td></tr>
<tr><td>AP50 (ranking COCO)</td><td>29%</td><td>71%</td></tr>
</tbody>
</table>
</div>
<p>Kesimpulan: reweighting efektif menaikkan AP50 (metrik ranking), tapi sebagian besar gain F1
operasional berasal dari iterasi tambahan, bukan reweighting itu sendiri. Keempat spesies rainforest
yang di-upweight (RubberFig, AliiFig, LeechVine, Umbrella) semuanya untung (rata-rata +0.80pp),
setara dengan 9 kelas lain (+0.83pp) &mdash; reweighting tidak mengorbankan kelas lain.</p>

<h2>5. Pengembangan Model DBH</h2>
<p>Setelah model segmentasi final ditetapkan, backbone dibekukan dan head estimasi DBH dilatih
terpisah. Pendekatan geometris murni gagal total (R&sup2;=&minus;5.28) karena lebar batang di
480&times;270px cuma beberapa piksel. Pendekatan neural bertahap (Hybrid Trunk-ROI) menaikkan skor
secara konsisten:</p>

<figure>
  <img src="data:image/png;base64,{IMG_DBH_PROG}" alt="DBH head progression chart">
  <figcaption><b>Gambar 2.</b> Progresi mean per-species R&sup2; dari V2 (hand-tuned, 6 spesies) ke
  Sandbox winner (cache-search, 11 spesies) &mdash; pertama kalinya skor menyeberang ke positif.</figcaption>
</figure>

<div class="twrap">
<table>
<thead><tr><th>Versi</th><th>Spesies</th><th>Perubahan kunci</th><th>Aggregate R&sup2;</th><th>MAE</th><th>Mean per-sp R&sup2;</th></tr></thead>
<tbody>
<tr><td>Geometric baseline</td><td>&mdash;</td><td>Pixel-width &times; depth / focal length</td><td>&minus;5.28</td><td>96mm</td><td>&mdash;</td></tr>
<tr><td>V2</td><td>6</td><td>Hybrid head, clean data, strip_rows=1</td><td>0.622</td><td>~65mm</td><td>&minus;0.416</td></tr>
<tr><td>V3</td><td>6</td><td>+ per-species weighted loss</td><td>0.626</td><td>63.6mm</td><td>&minus;0.375</td></tr>
<tr><td>V4</td><td>6</td><td>strip_rows=3</td><td>0.637</td><td>62.5mm</td><td>&minus;0.248</td></tr>
<tr><td>V5</td><td>6</td><td>strip_rows=5</td><td>0.646</td><td>61.8mm</td><td>&minus;0.222</td></tr>
<tr><td>V6</td><td>6</td><td>Dataset v2 (exclude LeechVine, depth&le;20m)</td><td>0.622</td><td>62.9mm</td><td>&minus;0.125</td></tr>
<tr class="best"><td>Sandbox (final)</td><td>11</td><td>Cache-based 14-axis search, 3-seed confirm</td><td>0.757</td><td>~52mm</td><td>+0.136</td></tr>
</tbody>
</table>
</div>

<h3>DBH Sandbox &mdash; ringkasan proses pencarian</h3>
<p>Infrastruktur baru (<code>scripts/dbh_sandbox/</code>) dibangun untuk sweep sistematis 14 axis
(7 filter data + 7 hyperparameter model) tanpa mengubah pipeline V1&ndash;V6 lama. Arsitektur:
backbone forward SEKALI &rarr; feature di-cache &rarr; head dilatih dari cache (~10&ndash;50&times;
lebih cepat dari retrain penuh).</p>
<div class="twrap">
<table>
<thead><tr><th>Fase</th><th>Tujuan</th><th>Hasil</th></tr></thead>
<tbody>
<tr><td>0.5 &mdash; Kalibrasi</td><td>Tentukan iterasi screening optimal</td><td>7000 iter (plateau aggR&sup2;~0.76 sejak iter 5000&ndash;6000)</td></tr>
<tr><td>A &mdash; OFAT Screening</td><td>~45 trial, 1 axis diubah per trial</td><td>Bug metrik ranking ditemukan &amp; diperbaiki (equal-weight R&sup2; tak stabil di N kecil)</td></tr>
<tr><td>B &mdash; Kombinasi</td><td>Gabungkan axis terbaik dari Fase A</td><td>Baseline (universe11, semua default) tetap juara &mdash; tak ada kombinasi yang mengalahkannya</td></tr>
<tr class="best"><td>C &mdash; Konfirmasi</td><td>3 seed pada test split, model juara Fase B</td><td>Mean per-species R&sup2;=+0.136 (rata2 3 seed: 0.154/0.119/0.135)</td></tr>
</tbody>
</table>
</div>
<div class="note">
<b>Pelajaran metodologi penting:</b> metrik ranking awal (equal-weight rata-rata R&sup2; semua
spesies) rapuh untuk spesies dengan N evaluasi kecil &mdash; satu kasus nyata: Persimmon N=10 &rarr;
R&sup2;=&minus;23,9 padahal MAE=36,5mm (salah satu TERBAIK). Ini artefak statistik, bukan model gagal.
Setelah difilter (hanya spesies N&ge;30 masuk skor ranking), kesimpulan Fase A/B yang sempat terbalik
("buang semua spesies plantation") dikoreksi balik ke baseline universe-11 penuh.
</div>

<h2 id="diagnosis">6. Diagnosis Bottleneck Recall (Investigasi Terbaru)</h2>
<p>Metrik size-stratified pada model final menunjukkan recall objek kecil sangat rendah (APs=6,33%,
ARs=7,53% vs objek besar APl=45,73%, ARl=67,76%) &mdash; pola yang tampak seperti masalah resolusi.
Tiga eksperimen inference-only menguji hipotesis ini secara langsung, semuanya pada domain
Rainforest (satu-satunya domain dengan render native 960&times;540):</p>

<figure>
  <img src="data:image/png;base64,{IMG_RES_TEST}" alt="Resolution test results">
  <figcaption><b>Gambar 3.</b> Recall pada test set Rainforest lintas rute resolusi. Detail native
  pada resolusi proses terlatih nyaris tak menolong (+0,6pp); menaikkan resolusi proses atau tiling
  justru merusak recall.</figcaption>
</figure>

<p>Analisis lanjutan memakai mask-IoU pada prediksi tersimpan model terbaik (Combined test, 7.925
instance) mengungkap akar masalah sebenarnya:</p>

<figure>
  <img src="data:image/png;base64,{IMG_SIZE_OCC}" alt="Size and occlusion breakdown">
  <figcaption><b>Gambar 4.</b> Komposisi hasil deteksi per ukuran objek. Kanopi besar praktis
  selesai (90,8% hit); objek kecil mayoritas gagal tapi cuma 1,7% instance; true-miss didominasi
  (93%) oleh kanopi <b>medium</b> yang under-detected di scene padat.</figcaption>
</figure>

<p>Sweep threshold pada model final mengonfirmasi operating point saat ini (skor 0,35) sudah
F1-optimal, dan mengungkap recall ceiling keras di sekitar 0,67 yang tak bisa ditembus threshold
berapa pun:</p>

<figure>
  <img src="data:image/png;base64,{IMG_THR_SWEEP}" alt="Threshold sweep curve">
  <figcaption><b>Gambar 5.</b> Precision/Recall/F1 vs threshold skor (IoU&ge;0,5). F1 puncak di
  threshold 0,35 (operating point saat ini); recall mentok ~0,67 pada threshold berapapun &mdash;
  batas kapasitas model, bukan pilihan threshold.</figcaption>
</figure>

<p><b>Kesimpulan gabungan:</b> ceiling recall model segmentasi ditentukan oleh under-detection
akibat oklusi kanopi padat (utamanya kelas ukuran medium), bukan oleh resolusi piksel maupun pilihan
operating point. Retrain resolusi-tinggi (960px) hanya akan menyasar segmen kecil (~1,7% instance);
arah riset yang lebih berdampak adalah metode occlusion-aware/amodal atau kapasitas query yang lebih
besar untuk scene padat.</p>

<h2>7. Kartu Model Final</h2>
<div class="tldr">
  <div class="card seg">
    <h3>Segmentasi &mdash; FocalNet-L Combined RGBD Scratch + Mosaic/NoObj/ClassWeight</h3>
    <div class="rows">
      <b>Arsitektur:</b> MaskDINO, backbone FocalNet-L, 300 query, 9 decoder layer<br>
      <b>Input:</b> RGB-D 4-channel, 640&times;640 processing size<br>
      <b>Training:</b> 75k iter base (from-scratch, AMP off, warmup 5000, backbone_mult 1.0) + 4-tahap fine-tune 20k iter/tahap<br>
      <b>Test AP50 / F1@0.35 / Recall / Precision:</b> 61.17% / 63.85% / 57.39% / 71.95%<br>
      <b>Checkpoint:</b><br><code>/scratch2/pr65/anur0018/maskdino_output/FocalNet_L_combined_rgbd_scratch_v4_stratified_mosaic_classweight_20k/model_final.pth</code><br>
      <b>Config:</b> <code>configs/maskdino_FocalNet_L_combined_rgbd_scratch_v4_stratified_mosaic_classweight_20k.yaml</code>
    </div>
  </div>
  <div class="card dbh">
    <h3>DBH &mdash; Hybrid Trunk-ROI Head (Sandbox winner)</h3>
    <div class="rows">
      <b>Arsitektur:</b> 2-layer MLP (207&rarr;256&rarr;128&rarr;1) di atas backbone beku, strip_rows=5<br>
      <b>Feature:</b> res2 (stride 4, 192ch) + one-hot spesies (11) + 2 feature geometris<br>
      <b>Training:</b> head-only dari cache, dropout=0.3, hidden=256, lr=1e-4, smooth-L1 per-species z-score<br>
      <b>Test (3-seed):</b> Aggregate R&sup2;=0.757, Mean per-sp R&sup2;=+0.136, MAE~52mm, within-100mm~87%<br>
      <b>Checkpoint:</b><br><code>/scratch2/pr65/anur0018/maskdino_output/sweep_dbh_hyperparams/phaseB_final_baseline_anchor_only/head_best.pth</code><br>
      <b>Config:</b> species_subset=universe11, min_trunk_px=5, max_depth=20.0, rubberfig_cap_cm=150, species_weight_scheme=inverse_freq_capped
    </div>
  </div>
</div>

<div class="footer">
<p><b>Sumber data &amp; skrip terkait</b> (semua path relatif thd <code>/scratch2/pr65/anur0018/tree_classification/</code>
kecuali disebutkan lain):</p>
<ul class="src">
<li>Tabel evaluasi model: <code>reports/eval_test_stratified_v4.csv</code>, <code>reports/eval_literature_metrics_v4.json</code></li>
<li>Dokumentasi parameter training: <code>reports/stratified_v4/Dokumentasi_Training_V4.html</code></li>
<li>Riset DBH V1&ndash;V6: <code>reports/stratified_v4/Report_FocalNetL_RGBD_Scratch_DBH_Research.html</code></li>
<li>DBH Sandbox: <code>scripts/dbh_sandbox/</code>, leaderboard di <code>/scratch2/pr65/anur0018/maskdino_output/sweep_dbh_hyperparams/leaderboard_recomputed.csv</code></li>
<li>Investigasi resolusi/oklusi: <code>scripts/res_test_rf_tahap1.py</code>, <code>scripts/res_test_rf_tiling.py</code>, <code>scripts/nearmiss_occlusion_analysis.py</code>, <code>scripts/threshold_sweep_figure.py</code></li>
<li>Tesis lengkap (naratif detail seluruh temuan): <code>paper/thesis_full.{{tex,pdf,html}}</code></li>
<li>Memory sistem: <code>project_context_quickref.md</code>, <code>project_training_progress.md</code>,
<code>project_dbh_diameter_research.md</code>, <code>project_resolution_bottleneck_test.md</code>,
<code>project_operating_threshold_dense_trees.md</code></li>
</ul>
<p>Dokumen ini dibuat sebagai tracker khusus proses pemilihan model &amp; fine-tuning &mdash; untuk
narasi akademik lengkap dengan tinjauan pustaka dan diskusi, lihat <a href="thesis_full.html">thesis_full.html</a>.</p>
</div>

</body>
</html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"wrote {OUT} ({len(HTML)//1024} KB)")
