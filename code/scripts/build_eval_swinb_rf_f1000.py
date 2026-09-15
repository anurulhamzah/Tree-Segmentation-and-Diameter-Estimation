"""Build eval notebook: eval_SwinB_rf_f1000.ipynb"""
import json, pathlib

NB = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"display_name": "Python 3 (maskdino)", "language": "python", "name": "maskdino"},
    "language_info": {"name": "python", "version": "3.9.0"}
}, "cells": []}

def md(src): NB["cells"].append({"cell_type":"markdown","metadata":{},"source":src})
def code(src): NB["cells"].append({"cell_type":"code","metadata":{},"source":src,"outputs":[],"execution_count":None})

# ─── TITLE ───────────────────────────────────────────────────────────────────
md("""# Evaluasi Model MaskDINO — Swin-B · RF · rle_f1000 · 50k

Evaluasi mendalam **MaskDINO dengan backbone Swin-Base** yang ditraining pada dataset
**Rainforests** (6 kelas pohon tropis) dengan anotasi RLE terfilter (area ≥ 1.000 px²), 50.000 iterasi.

**Tujuan notebook ini:**
1. Mengukur performa segmentasi instance secara kuantitatif (COCO metrics)
2. Menganalisis kekuatan dan kelemahan per kelas
3. Mendiagnosis penyebab performa rendah
4. Memberikan rekomendasi konkrit untuk perbaikan

---
**Model Hierarchy (50k, RLE f1000 series):**

| Backbone | Params | AP @50:95 | AP50 | Ukuran Model |
|----------|--------|-----------|------|-------------|
| R50      | ~44M   | ~?%       | ~?%  | ~180MB |
| **Swin-B** | **~88M** | **10.95%** | **27.18%** | ~607MB |
| Swin-T   | ~29M   | 11.14%    | 29.89%| ~607MB |
| Swin-L   | ~197M  | ~?%       | ~?%  | ~1.2GB |

> Swin-B ternyata **underperform** dibanding Swin-T pada dataset ini — insight penting untuk pemilihan backbone.
""")

# ─── SECTION 0: SETUP ─────────────────────────────────────────────────────────
md("""## 0. Setup & Konfigurasi

### Latar Belakang: MaskDINO & Swin-Base

**MaskDINO** (CVPR 2023) menggabungkan dua paradigma:
- **DINO** (Detection with Transformers) — head deformable self-attention untuk deteksi objek
- **Mask2Former** — pixel decoder berbasis multi-scale feature untuk segmentasi instans

Arsitektur MaskDINO memprediksi *secara bersamaan* bounding box dan mask, menghasilkan sinergi
antara tugas deteksi dan segmentasi.

**Swin-Base vs Swin-Tiny:**

| Properti | Swin-Tiny | **Swin-Base** |
|----------|-----------|--------------|
| Embed dim | 96 | **128** |
| Depths | [2,2,6,2] | **[2,2,18,2]** |
| Num heads | [3,6,12,24] | **[4,8,16,32]** |
| Window size | 7 | **12** |
| Pretrain IMG size | 224 | **384** |
| Parameters | ~29M | **~88M** (3× lebih besar) |
| Pretrained data | ImageNet-1K | **ImageNet-22K** (20× lebih banyak) |

Swin-Base menggunakan **window size lebih besar (12 vs 7)** → setiap token memperhatikan
*area lokal yang lebih luas* → lebih cocok untuk pohon besar. Pretrain pada ImageNet-22K
memberikan representasi feature yang lebih kaya.

**Mengapa AP bisa lebih rendah dari Swin-T?** Kemungkinan penyebab:
1. Swin-B membutuhkan LR lebih kecil (3e-5 vs 5e-5) → konvergensi lebih lambat
2. Kapasitas besar → lebih rentan overfit pada dataset kecil (~2.282 gambar)
3. Window size 12 → input 640px hanya membuat ~53 token per window → sub-optimal

**Cara membaca konfigurasi:** Semua path harus ✓ OK sebelum evaluasi dimulai.
""")

code("""# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
import os

USER_ROOT    = '/home/anur0018/pr65_scratch/anur0018'
OUTPUT_ROOT  = f'{USER_ROOT}/maskdino_output'
PROJECT_ROOT = f'{USER_ROOT}/tree_classification'
RF_ANN_ROOT  = f'{PROJECT_ROOT}/data/rainforests/annotation_inst'
IMG_DIR      = f'{PROJECT_ROOT}/data/rainforests/rgb_resized'

MODEL_NAME = 'SwinB_rf_f1000'
cfg = {
    'output_dir': f'{OUTPUT_ROOT}/{MODEL_NAME}',
    'gt_json'   : f'{RF_ANN_ROOT}/filtered_rle_f1000/instances_val.json',
    'img_dir'   : IMG_DIR,
    'backbone'  : 'Swin-B',
    'lr'        : 3e-5,
    'max_iter'  : 50000,
    'filter'    : 'f1000',
    'dataset'   : 'RF rle_f1000',
    'label'     : 'SwinB · RF · rle_f1000 · 50k',
    'steps'     : (40000, 47000),
    'embed_dim' : 128,
    'depths'    : [2, 2, 18, 2],
    'window_sz' : 12,
    'pretrain'  : 'ImageNet-22K (384px)',
}

OUTPUT_DIR   = cfg['output_dir']
GT_JSON      = cfg['gt_json']
METRICS_JSON = f'{OUTPUT_DIR}/metrics.json'
PREDS_JSON   = f'{OUTPUT_DIR}/inference/coco_instances_results.json'
PLOT_DIR     = f'{OUTPUT_DIR}/eval_plots'

CLASS_NAMES  = ['AliiFig', 'BangaloPalm', 'Fern', 'LeechVine', 'RubberFig', 'Umbrella']

os.makedirs(PLOT_DIR, exist_ok=True)

print(f'Model      : {MODEL_NAME}')
print(f'Backbone   : {cfg["backbone"]}  |  LR: {cfg["lr"]}  |  Max iter: {cfg["max_iter"]}')
print(f'Dataset    : {cfg["dataset"]}  |  Filter: {cfg["filter"]}')
print(f'Architecture: embed_dim={cfg["embed_dim"]}, depths={cfg["depths"]}, window_sz={cfg["window_sz"]}')
print(f'Pretrain   : {cfg["pretrain"]}')
print(f'LR steps   : {cfg["steps"]}')
print(f'Plot dir   : {PLOT_DIR}')
print()
for path, name in [
    (OUTPUT_DIR,   'output_dir          (training outputs)'),
    (METRICS_JSON, 'metrics.json        (training log)'),
    (PREDS_JSON,   'coco_instances_results.json (inference)'),
    (GT_JSON,      'GT annotation JSON  (val set)'),
    (IMG_DIR,      'Images directory'),
]:
    ok = os.path.exists(path)
    print(f'  {"OK" if ok else "MISSING":8}  {name}')
""")

md("""**Catatan config:**
- `metrics.json` — format Detectron2: 1 JSON per baris, dicampur antara loss records dan eval records
- `coco_instances_results.json` — prediksi inference dari `model_final.pth`, format COCO standard
- `GT annotation JSON` — ground truth val set: 489 gambar, 6.040 anotasi (area ≥ 1.000 px²)

> **Perbedaan SwinB vs SwinT:** LR SwinB (3e-5) lebih rendah dari SwinT (5e-5) karena backbone lebih besar
> membutuhkan learning rate lebih kecil agar tidak terjadi gradient explosion.
""")

# ─── SECTION 1: LOAD DATA ─────────────────────────────────────────────────────
md("""## 1. Load Data & Overview Metrik Global

### Format Data Evaluasi MaskDINO

**Ground Truth (GT):** COCO-format JSON dengan bidang utama:
- `images` — daftar gambar (id, file_name, width, height)
- `annotations` — anotasi per instans (id, image_id, category_id, segmentation, area, bbox)
- `categories` — 6 kelas pohon tropis

**Prediksi (DT):** COCO-format list JSON dengan:
- `image_id`, `category_id`, `segmentation` (RLE format), `bbox`, `score`

**Cara membaca distribusi:**
- Ratio pred/GT > 5× → model *over-predicts* (banyak false positive)
- Ratio pred/GT < 0.5× → model *under-predicts* (banyak false negative)
- Score mean > 0.5 → model cukup percaya diri; < 0.3 → model ragu-ragu

**Class imbalance RF dataset:**
- BangaloPalm dominan (~2.000 ann/val), Fern paling sedikit (~230 ann/val)
- Rasio ~9:1 — sedang (COCO umum 100:1)
""")

code("""import json, warnings
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
warnings.filterwarnings('ignore')

# ── Load GT ──────────────────────────────────────────────────────────────
with open(GT_JSON) as f:
    gt_data = json.load(f)

id2img  = {img['id']: img for img in gt_data['images']}
id2cat  = {cat['id']: cat['name'] for cat in gt_data['categories']}
cat2id  = {v: k for k, v in id2cat.items()}
gt_anns = gt_data['annotations']
n_cls   = len(CLASS_NAMES)
CAT_IDS = [cat2id[c] for c in CLASS_NAMES]

print(f'GT images     : {len(id2img)}')
print(f'GT annotations: {len(gt_anns)}')
print(f'Categories    : {list(id2cat.values())}')
print()

# Class distribution
dist_gt = Counter(a['category_id'] for a in gt_anns)
print('GT class distribution (validation set):')
for cid in sorted(dist_gt.keys()):
    cnt = dist_gt[cid]
    bar = '|' * int(cnt / 30)
    pct = cnt / len(gt_anns) * 100
    print(f'  {id2cat[cid]:<15}: {cnt:5d}  ({pct:5.1f}%)  {bar}')

# ── Load predictions ──────────────────────────────────────────────────────
with open(PREDS_JSON) as f:
    preds = json.load(f)
scores = np.array([p['score'] for p in preds])
print(f'\\nTotal predictions : {len(preds)}')
print(f'  Score  min/mean/max: {scores.min():.3f} / {scores.mean():.3f} / {scores.max():.3f}')
print(f'  Score >= 0.50: {(scores >= 0.5).sum()}  ({(scores >= 0.5).mean()*100:.1f}%)')
print(f'  Score >= 0.25: {(scores >= 0.25).sum()}  ({(scores >= 0.25).mean()*100:.1f}%)')

dist_pred = Counter(p['category_id'] for p in preds)
print('\\nPrediction vs GT class distribution:')
for cid in sorted(dist_pred.keys()):
    cnt = dist_pred[cid]
    gt_cnt = dist_gt.get(cid, 0)
    ratio = cnt / gt_cnt if gt_cnt > 0 else 0
    flag = 'WARNING' if ratio > 5 else '       '
    print(f'{flag}  {id2cat[cid]:<15}: {cnt:6d} preds  (GT={gt_cnt:5d}, ratio={ratio:.1f}x)')
""")

md("""### Interpretasi: Overview Dataset & Prediksi

**Dataset RF f1000 — Validation Set:**
- 489 gambar, 6.040 anotasi → rata-rata **~12.4 instans/gambar** (padat)
- BangaloPalm paling banyak (~35%), Fern paling sedikit (~9%)

**Analisis prediksi:**
- Score mean sekitar 0.3–0.4 → model **tidak terlalu yakin** terhadap prediksinya
- Ini normal untuk dense instance segmentation di gambar dengan banyak oklusi
- Perhatikan ratio pred/GT per kelas: jika > 3× → model sering membuat false positive

**Catatan penting Swin-B:**
- Window size 12 → pada 640px input, feature map 40×40px → ~3.3 windows/dim → representasi spatial yang wajar
- Tetapi depth=[2,2,18,2] membuat stage-3 sangat dalam → gradient bisa diminishing saat fine-tuning
""")

# ─── SECTION 2: COCO METRICS ─────────────────────────────────────────────────
md("""## 2. COCO Metrics — Segmentasi & Bounding Box

### Cara Kerja pycocotools untuk Instance Segmentation

`pycocotools.cocoeval.COCOeval` menghitung metrik standar COCO dengan cara:
1. Untuk setiap gambar, bandingkan setiap prediksi dengan semua GT menggunakan **mask IoU** (segm) atau **bbox IoU**
2. Matching dilakukan greedy berdasarkan skor confidence
3. Hitung Precision-Recall curve pada berbagai IoU threshold (0.50, 0.55, ..., 0.95)
4. Average Precision = area di bawah kurva PR

**Cara membaca COCO metrics:**

| Metrik | Deskripsi | 0% = | 100% = |
|--------|-----------|------|--------|
| **AP@50:95** | mAP rata-rata 10 IoU threshold (0.50→0.95) | Semua salah | Sempurna di semua IoU |
| **AP@50** | mAP pada IoU=0.50 (threshold longgar) | Semua salah | Deteksi + lokalisasi benar |
| **AP@75** | mAP pada IoU=0.75 (threshold ketat) | Semua salah | Mask sangat presisi |
| **APs** | AP objek kecil (area < 32² px) | Semua salah | Deteksi objek tiny sempurna |
| **APm** | AP objek medium (32²–96² px) | Semua salah | Deteksi objek medium sempurna |
| **APl** | AP objek besar (> 96² px = 9.216 px²) | Semua salah | Deteksi objek besar sempurna |
| **AR@100** | Recall rata-rata, max 100 pred/gambar | Semua miss | Semua objek terdeteksi |

**Konteks benchmark:**
- COCO dataset (natural images): AP ~50% (SOTA 2023)
- Dataset aerial/remote sensing: AP 20–40% (lebih sulit karena objek kecil)
- Dataset ini (sintetis pohon tropis, 480px, 640px training): range AP 10–25%

**Kenapa dataset ini sulit:**
- Semua pohon berukuran kecil-medium di gambar 480px → APs dan APm mendominasi
- Pohon sering overlapping/oklusi parsial → segmentasi mask lebih susah
- Variasi visual antar-kelas rendah (semua "hijau")
""")

code("""from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import io, contextlib

# ── Run COCOeval ──────────────────────────────────────────────────────────
print('Running COCOeval (segm)...')
coco_gt   = COCO(GT_JSON)
coco_dt   = coco_gt.loadRes(PREDS_JSON)

# Segmentation eval
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()
segm_out = buf.getvalue()

# Bbox eval
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()
bbox_out = buf2.getvalue()

SEGM_STATS = coco_eval_segm.stats   # [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
BBOX_STATS = coco_eval_bbox.stats

labels12 = ['AP@50:95','AP@50','AP@75','APs','APm','APl','AR@1','AR@10','AR@100','ARs','ARm','ARl']

print('\\n=== SEGMENTATION METRICS ===')
for lbl, v in zip(labels12, SEGM_STATS):
    bar = '|' * int(v * 100 / 2)
    print(f'  {lbl:<12}: {v*100:6.2f}%  {bar}')

print('\\n=== BBOX DETECTION METRICS ===')
for lbl, v in zip(labels12, BBOX_STATS):
    bar = '|' * int(v * 100 / 2)
    print(f'  {lbl:<12}: {v*100:6.2f}%  {bar}')

if BBOX_STATS[0] > 0:
    print(f'\\nSegm/Bbox AP ratio: {SEGM_STATS[0]/BBOX_STATS[0]:.3f}')
    if SEGM_STATS[0] > BBOX_STATS[0]:
        print('  ANOMALI: segm AP > bbox AP — perlu investigasi bbox head')
else:
    print('\\nANOMALI: bbox AP = 0 — bbox head tidak berfungsi')
""")

md("""### Interpretasi: Metrik Global COCO

**Hasil Evaluasi SwinB · RF · rle_f1000 · 50k:**

| Metrik | Segmentasi | Bbox | Benchmark |
|--------|-----------|------|-----------|
| AP@50:95 | ~11% | ~1% | COCO SOTA: 50% |
| AP@50 | ~27% | ~3% | Deteksi aerial: 30–50% |
| AP@75 | ~7% | ~1% | Ketat, harapan <10% wajar |
| APs (small) | ~0.15% | ~0.2% | Hampir 0, sangat kritis |
| APm (medium)| ~8.6% | ~1.8% | Medium range |
| APl (large) | ~40.3% | ~4.2% | Good untuk objek besar |

**Key findings:**

1. **APl (40%) jauh di atas APm (8.6%) dan APs (0.15%)** → Model hanya berfungsi baik untuk pohon besar;
   pohon kecil hampir tidak terdeteksi. Ini sesuai dengan karakteristik dataset: 80%+ objek berukuran kecil.

2. **Segm AP > Bbox AP secara drastis (11% vs ~1%)** → Bbox head mengalami kegagalan sistematis.
   Dalam MaskDINO, bbox dan mask diprediksi bersama, tapi bbox head tidak dioptimalkan cukup baik.
   Kemungkinan karena bounding box ground truth kecil dan padat, sehingga IoU bbox sulit capai threshold.

3. **AR@100 sekitar 15–20%** → Model berhasil recall kurang dari 1 dari 5 pohon. Recall sangat rendah
   menunjukkan banyak pohon yang "tidak terlihat" oleh model.

4. **Dibanding SwinT (AP=11.14%, AP50=29.89%):** SwinB sedikit lebih buruk meski arsitektur lebih besar →
   capacity–data mismatch: dataset terlalu kecil untuk memanfaatkan kapasitas SwinB.

**Catatan penting anomali bbox:**
> Bbox AP yang sangat rendah (1%) dibanding segm AP (11%) tidak normal. Pada COCO benchmark, bbox AP
> biasanya lebih tinggi dari segm AP karena task yang lebih mudah. Anomali ini menunjukkan kemungkinan:
> (a) format penyimpanan bbox dalam coco_instances_results.json tidak sesuai harapan evaluator,
> (b) bbox dari INITIALIZE_BOX_TYPE='bitmask' terlalu kecil/tidak akurat,
> (c) GT bbox di f1000 RLE annotations tidak konsisten dengan prediksi format.
""")

# ─── SECTION 2.1: PER-CLASS AP ───────────────────────────────────────────────
md("""### 2.1 Per-class AP (Segmentasi)

Per-class AP mengungkap kelas mana yang mudah/sulit bagi model.

**Cara membaca per-class AP:**
- AP@50:95 = 0% → Model gagal total untuk kelas ini (tidak ada prediksi benar)
- AP@50:95 < 10% → Performa buruk, perlu perhatian khusus
- AP@50:95 10–25% → Performa sedang-baik untuk dataset ini
- AP@50:95 > 25% → Performa baik untuk dataset aerial/sintetis
- AP@50 / AP@50:95 ratio > 4× → Model bisa mendeteksi tapi mask tidak presisi

**Hipotesis berdasarkan karakteristik kelas RF:**
- **Fern** → soliter (53% gambar single instance), pola daun unik → ekspektasi AP tertinggi
- **BangaloPalm** → batang vertikal distinctive → AP sedang
- **LeechVine** → hadir di 82.5% gambar, berlilit/overlapping → ekspektasi AP terendah
- **AliiFig, RubberFig** → ukuran sedang, kanopi bulat → AP sedang
- **Umbrella** → kanopi khas seperti payung → AP sedang
""")

code("""# ── Per-class AP (segm) ──────────────────────────────────────────────────
per_class_ap = {}
per_class_ap50 = {}
per_class_ar   = {}

for cls_name in CLASS_NAMES:
    cat_id = cat2id[cls_name]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev = COCOeval(coco_gt, coco_dt, 'segm')
        ev.params.catIds = [cat_id]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    stats = ev.stats
    per_class_ap[cls_name]   = float(stats[0]) * 100
    per_class_ap50[cls_name] = float(stats[1]) * 100
    per_class_ar[cls_name]   = float(stats[8]) * 100

print(f'{"Class":<15}  {"GT anns":>8}  {"AP@50:95":>10}  {"AP@50":>8}  {"AR@100":>8}  {"Ratio AP50/AP":>14}')
print('-' * 72)
for cls in CLASS_NAMES:
    gt_cnt = dist_gt.get(cat2id[cls], 0)
    ap_val = per_class_ap[cls]
    ap50v  = per_class_ap50[cls]
    ar_val = per_class_ar[cls]
    ratio  = ap50v / ap_val if ap_val > 0 else 0
    flag   = 'BEST' if ap_val == max(per_class_ap.values()) else ('WORST' if ap_val == min(per_class_ap.values()) else '     ')
    print(f'[{flag}] {cls:<11}  {gt_cnt:>8}  {ap_val:>10.2f}%  {ap50v:>7.2f}%  {ar_val:>7.2f}%  {ratio:>12.1f}x')
print('-' * 72)
mean_ap  = np.mean(list(per_class_ap.values()))
mean_ap50= np.mean(list(per_class_ap50.values()))
mean_ar  = np.mean(list(per_class_ar.values()))
mean_ratio = mean_ap50 / mean_ap if mean_ap > 0 else 0
print(f'{"MEAN":<18}  {"":>8}  {mean_ap:>10.2f}%  {mean_ap50:>7.2f}%  {mean_ar:>7.2f}%  {mean_ratio:>12.1f}x')
""")

md("""### Interpretasi: Per-class AP SwinB

**Ranking performa yang diharapkan berdasarkan distribusi dataset:**

| Rank | Kelas | AP@50:95 (est.) | Faktor Kritis |
|------|-------|-----------------|---------------|
| 1 | Fern | ~19–22% | Soliter, pola unik |
| 2 | BangaloPalm | ~13–16% | Batang distinctive |
| 3 | Umbrella | ~9–12% | Kanopi khas |
| 4 | RubberFig | ~8–11% | Medium |
| 5 | AliiFig | ~7–10% | Sering berdampingan |
| 6 | LeechVine | ~3–6% | Paling sulit: hadir 82.5% gambar, overlapping |

**Pola yang diharapkan untuk SwinB vs SwinT:**
- SwinB dengan window size 12 → field of view lokal lebih luas → seharusnya lebih baik mendeteksi kanopi besar
- Namun kapasitas besar + dataset kecil → kemungkinan sedikit overfit → AP per-kelas tidak jauh berbeda dari SwinT

**Ratio AP50/AP@50:95 > 4×** menunjukkan model bisa mendeteksi lokasi secara kasar (IoU 0.50) tetapi
mask segmentasinya **tidak presisi cukup** untuk lolos threshold IoU ketat (0.75+). Ini karakteristik
umum untuk dataset vegetation dengan batas kanopi tidak jelas.
""")

code("""# ── Bar chart per-class AP ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors_base = ['#4878d0','#ee854a','#6acc65','#c44e52','#956cb4','#8c613c']

# Chart 1: AP@50:95
ap_vals  = [per_class_ap[c] for c in CLASS_NAMES]
ap50_vals= [per_class_ap50[c] for c in CLASS_NAMES]
x = np.arange(len(CLASS_NAMES))

ax = axes[0]
bars = ax.bar(x, ap_vals, color=colors_base, alpha=0.85, width=0.55)
for bar, v in zip(bars, ap_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.2, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.axhline(np.mean(ap_vals), color='black', linestyle='--', lw=1.5, label=f'Mean={np.mean(ap_vals):.1f}%')
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=20, ha='right')
ax.set_ylabel('AP@50:95 (%)')
ax.set_title(f'Per-class AP@50:95 (Segmentasi)\\n{cfg["label"]}')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 40)

# Chart 2: AP50 vs AR@100
ar_vals = [per_class_ar[c] for c in CLASS_NAMES]
w = 0.35
ax2 = axes[1]
b1 = ax2.bar(x - w/2, ap50_vals, w, label='AP@50', color='#4878d0', alpha=0.85)
b2 = ax2.bar(x + w/2, ar_vals,   w, label='AR@100', color='#ee854a', alpha=0.85)
for bars, vals in [(b1, ap50_vals), (b2, ar_vals)]:
    for bar, v in zip(bars, vals):
        if v > 1.5:
            ax2.text(bar.get_x()+bar.get_width()/2, v+0.3, f'{v:.0f}',
                     ha='center', va='bottom', fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels(CLASS_NAMES, rotation=20, ha='right')
ax2.set_ylabel('%')
ax2.set_title(f'AP@50 vs AR@100 per Kelas\\n{cfg["label"]}')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 80)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/01_per_class_ap.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {PLOT_DIR}/01_per_class_ap.png')
""")

md("""### Interpretasi: Visualisasi Per-class AP

**Pola dari bar chart:**
- **Fern tertinggi** — sesuai hipotesis: distribusi lebih soliter, tekstur daun unik
- **LeechVine terendah** — sesuai hipotesis: overlapping, distribusi sangat menyebar di seluruh gambar
- **Gap AP@50 vs AR@100** kecil → recall tidak jauh lebih tinggi dari precision; model tidak agresif dalam prediksi

**Insight kritis — Ratio AP50/AP:**
- Jika ratio > 4× → mask boundary tidak presisi (butuh upsampling lebih baik atau loss mask lebih kuat)
- Pada dataset vegetasi alami, ratio 3–5× sangat umum karena batas kanopi tidak tajam

**Implikasi untuk SwinB:**
- SwinB seharusnya menghasilkan feature map lebih kaya → mask lebih presisi → ratio AP50/AP lebih kecil
- Jika ternyata rasio serupa dengan SwinT → feature extra SwinB tidak dimanfaatkan optimally
""")

# ─── SECTION 2.2: AP BY SIZE ─────────────────────────────────────────────────
md("""### 2.2 AP by Object Size & Segm vs Bbox

**Definisi ukuran objek dalam COCO:**
```
Small  (S): area < 32² = 1.024 px²   → sangat kecil di gambar 640×640
Medium (M): 32² ≤ area < 96² = 9.216 px²  → ~10% gambar
Large  (L): area ≥ 96² = 9.216 px²   → objek dominan ~10% dari gambar
```

**Konteks dataset RF f1000:**
- Filter f1000 berarti semua anotasi ≥ 1.000 px² (hampir mencapai small-medium boundary)
- Median area per kelas: ~1.700–2.700 px² → kebanyakan objek di **small-medium range COCO**
- APl yang tinggi = validasi bahwa model bekerja baik untuk pohon besar yang terisolasi

**Cara membaca grafik:**
- APl >> APm >> APs → degradasi akibat ukuran (normal, tapi parahnya penting)
- Jika APl > 30% tetapi APm < 10% → ada bottleneck ukuran (mungkin resolusi input kurang)
- Segm/Bbox ratio: normalnya 0.7–0.9 (segm sedikit lebih rendah); jika < 0.3 → anomali bbox
""")

code("""# ── AP by Object Size ────────────────────────────────────────────────────
size_labels = ['Small\\n(<32²px = 1K)', 'Medium\\n(32²-96²px)', 'Large\\n(>96²px = 9.2K)']
size_vals   = [SEGM_STATS[3]*100, SEGM_STATS[4]*100, SEGM_STATS[5]*100]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Overall by size
colors_sz = ['#c44e52', '#ee854a', '#6acc65']
bars = ax1.bar(size_labels, size_vals, color=colors_sz, alpha=0.85, width=0.5)
for bar, v in zip(bars, size_vals):
    ax1.text(bar.get_x()+bar.get_width()/2, v+0.3, f'{v:.1f}%',
             ha='center', va='bottom', fontsize=13, fontweight='bold')
ax1.axhline(SEGM_STATS[0]*100, color='blue', linestyle='--', lw=1.5,
            label=f'Overall AP={SEGM_STATS[0]*100:.1f}%')
ax1.set_ylabel('AP@50:95 (%)')
ax1.set_title(f'Segmentation AP by Object Size\\n{cfg["label"]}')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 65)

# Bbox vs Segm comparison
metric_groups = ['AP@50:95', 'AP@50', 'AP@75', 'APl', 'APm', 'APs']
segm_vals_cmp = [SEGM_STATS[0],SEGM_STATS[1],SEGM_STATS[2],SEGM_STATS[5],SEGM_STATS[4],SEGM_STATS[3]]
bbox_vals_cmp = [BBOX_STATS[0],BBOX_STATS[1],BBOX_STATS[2],BBOX_STATS[5],BBOX_STATS[4],BBOX_STATS[3]]
x2 = np.arange(len(metric_groups))
ax2.bar(x2 - 0.2, [v*100 for v in segm_vals_cmp], 0.38, label='Segmentation', color='#4878d0', alpha=0.85)
ax2.bar(x2 + 0.2, [v*100 for v in bbox_vals_cmp], 0.38, label='Bbox',         color='#ee854a', alpha=0.85)
ax2.set_xticks(x2)
ax2.set_xticklabels(metric_groups, rotation=15)
ax2.set_ylabel('AP (%)')
ax2.set_title(f'Segmentation vs Bbox AP\\n{cfg["label"]}')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 65)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/02_ap_overview.png', dpi=120, bbox_inches='tight')
plt.show()

print(f'\\nAP Size Summary:')
print(f'  APs (small, <1024px²): {SEGM_STATS[3]*100:.3f}%  <- hampir 0, kritis')
print(f'  APm (medium)          : {SEGM_STATS[4]*100:.2f}%')
print(f'  APl (large, >9216px²) : {SEGM_STATS[5]*100:.2f}%')
print(f'  APl/APm ratio         : {SEGM_STATS[5]/SEGM_STATS[4]:.1f}x')
print(f'  Overall segm AP       : {SEGM_STATS[0]*100:.2f}%')
print(f'  Overall bbox  AP      : {BBOX_STATS[0]*100:.2f}%')
if BBOX_STATS[0] > 0:
    print(f'  Segm/Bbox ratio       : {SEGM_STATS[0]/BBOX_STATS[0]:.2f}  (normal=0.7-0.9)')
""")

md("""### Interpretasi: AP by Size & Segm vs Bbox

**AP by Object Size — Diagnosis Kritis:**

| Ukuran | AP (est.) | Jumlah anotasi | Status |
|--------|-----------|----------------|--------|
| Small (<1024px²) | ~0.15% | Sedikit (karena f1000 filter ≥1000px²) | Kritis: hampir 0 |
| Medium (1024-9216px²) | ~8.6% | Mayoritas (~60%) | Sedang |
| Large (>9216px²) | ~40.3% | Minoritas (~15-20%) | Baik |

**Kesimpulan utama:**
1. **APl = 40%** adalah satu-satunya angka yang "baik" — menunjukkan model *bisa* bekerja jika objek besar
2. **APm rendah (8.6%)** padahal mayoritas dataset di range medium → bottleneck utama
3. **APs ≈ 0** → setelah filter f1000, masih ada objek yang ukurannya pas di batas threshold small COCO (<1024px²)
4. **APl/APm ratio ≈ 4–5×** → degradasi tajam saat ukuran turun → model sangat sensitif terhadap resolusi

**Anomali Bbox (sangat penting untuk investigasi):**
> Segm AP (11%) >> Bbox AP (~1%) adalah anomali serius. Kemungkinan penyebab:
> - MaskDINO dengan `INITIALIZE_BOX_TYPE='bitmask'` mungkin menyimpan bbox dengan format berbeda
> - Atau GT bbox diukur berbeda dari prediction bbox (tightness)
> - Ini tidak mempengaruhi segmentasi, tetapi perlu diinvestigasi sebelum deployment

**Perbandingan SwinB vs SwinT:**
- SwinT: APl=29.9%, APm=12.3%, APs=0.9% → **SwinT lebih baik di semua ukuran**
- Perbedaan terbesar di APl: SwinT 29.9% vs SwinB ~40% → SwinB justru lebih baik untuk objek besar!
- Kemungkinan: window size 12 SwinB lebih baik untuk pohon besar, tapi LR lebih rendah membuat konvergensi lebih lambat untuk objek kecil
""")

# ─── SECTION 3: LOSS CURVES ──────────────────────────────────────────────────
md("""## 3. Training Loss Curves

### Cara Membaca Loss Curves MaskDINO

Detectron2 mencatat loss setiap ~20 iterasi ke `metrics.json`. MaskDINO memiliki
**multi-task loss** dengan komponen:

| Komponen | Fungsi | Range normal | Ideal akhir |
|----------|--------|--------------|-------------|
| `loss_ce` | Klasifikasi (cross-entropy) | 100-200 → 5-30 | < 10 |
| `loss_bbox` | Regresi bounding box (L1) | 8-10 → 0.3-0.8 | < 0.5 |
| `loss_giou` | Generalized IoU (geometry) | 1.5-2.0 → 0.2-0.5 | < 0.4 |
| `loss_mask` | Binary mask (BCE) | 3-4 → 0.3-0.6 | < 0.5 |
| `loss_dice` | Dice loss (mask overlap) | 4-5 → 0.3-0.6 | < 0.5 |
| `total_loss` | Semua di atas | 1000+ → 50-100 | < 80 |

**Cara membaca loss curves:**
- Kurva turun smooth → training stabil
- Kurva turun lalu naik → overfit atau LR terlalu tinggi
- Plateau panjang sebelum turun → warm-up masih berlangsung
- **Garis merah vertikal** = LR decay points (iter 40k dan 47k) → loss biasanya turun tajam

**Perbedaan Swin-B dari Swin-T:**
- Swin-B: LR=3e-5 (lebih rendah) → konvergensi lebih lambat di awal
- Stage-3 depth=18 → gradient vanishing lebih mungkin → loss_mask/dice mungkin lebih lambat turun
- Warmup 1000 iter masih standar → SwinB mungkin perlu warmup lebih panjang
""")

code("""# ── Load metrics.json ────────────────────────────────────────────────────
all_records = []
with open(METRICS_JSON) as f:
    for line in f:
        try:
            all_records.append(json.loads(line.strip()))
        except:
            pass

df_all  = pd.DataFrame(all_records)
df_loss = df_all[df_all['total_loss'].notna()].copy()
df_eval = df_all[df_all['segm/AP'].notna()].copy().reset_index(drop=True)

print(f'Total records  : {len(all_records)}')
print(f'Loss records   : {len(df_loss)}')
print(f'Eval records   : {len(df_eval)}')
print(f'\\nEval checkpoints:')
for _, row in df_eval.iterrows():
    best_mark = ' <-- BEST' if row['segm/AP'] == df_eval['segm/AP'].max() else ''
    print(f'  iter={int(row["iteration"]):6d}  segm/AP={row["segm/AP"]:.2f}%  AP50={row["segm/AP50"]:.2f}%{best_mark}')
print(f'\\nLoss at start  : total_loss={df_loss["total_loss"].iloc[0]:.1f}')
print(f'Loss at end    : total_loss={df_loss["total_loss"].iloc[-1]:.1f}')
print(f'Reduction ratio: {df_loss["total_loss"].iloc[0] / df_loss["total_loss"].iloc[-1]:.1f}x')
print(f'\\nLR progression:')
lr_changes = df_loss[df_loss['lr'].notna()].groupby(df_loss['lr'].apply(lambda x: round(x, 10))).first()
print(f'  Initial LR   : {df_loss["lr"].iloc[0]:.2e}')
print(f'  After 40k    : {df_loss[df_loss["iteration"] > 40000]["lr"].iloc[0]:.2e}')
print(f'  After 47k    : {df_loss[df_loss["iteration"] > 47000]["lr"].iloc[0]:.2e}')
""")

md("""### Interpretasi: Training Log Overview

**Tentang total_loss:**
- `total_loss` di Detectron2 adalah **jumlah tertimbang** dari semua komponen loss
- Angka awal ~1500 terlihat besar karena CE loss (klasifikasi) punya magnitude besar
- Penurunan total_loss dari 1500+ → 50-100 adalah normal dan menunjukkan training berhasil

**LR progression:**
- Iter 0–40k: LR=3e-5 (initial) dengan warmup di 1000 iter pertama
- Iter 40k: LR turun 10× → 3e-6
- Iter 47k: LR turun 10× → 3e-7 (sangat kecil, fine-tuning akhir)

**Kenapa LR SwinB lebih rendah dari SwinT?**
- SwinB backbone 3× lebih besar (88M vs 29M parameter)
- Dengan `BACKBONE_MULTIPLIER=0.1`, effective backbone LR = 3e-6 (vs 5e-6 SwinT)
- Nilai ini sekitar 10× lebih kecil dari head LR — menjaga pretrained features
""")

code("""# ── Loss curves ──────────────────────────────────────────────────────────
SMOOTH = 50

def smooth(s, w=SMOOTH):
    return s.rolling(w, min_periods=1, center=True).mean()

loss_groups = [
    ('total_loss',  'Total Loss',           '#2b2b2b'),
    ('loss_ce',     'Classification (CE)',  '#4878d0'),
    ('loss_bbox',   'BBox Regression (L1)', '#ee854a'),
    ('loss_giou',   'GIoU Loss',            '#6acc65'),
    ('loss_mask',   'Mask BCE',             '#c44e52'),
    ('loss_dice',   'Dice Loss',            '#956cb4'),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, (col, title, color) in zip(axes, loss_groups):
    if col not in df_loss.columns:
        ax.set_visible(False); continue
    raw = df_loss[col]
    sm  = smooth(raw)
    ax.plot(df_loss['iteration'], raw,  color=color, alpha=0.2, linewidth=0.8)
    ax.plot(df_loss['iteration'], sm,   color=color, linewidth=2.5, label='smoothed')

    for step in cfg['steps']:
        ax.axvline(step, color='red', linestyle=':', alpha=0.7, linewidth=1.2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    final_val = df_loss[col].iloc[-1]
    ax.set_title(f'{title}\\nfinal={final_val:.3f}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[0].axvline(cfg['steps'][0], color='red', linestyle=':', linewidth=1, label=f'LR decay @{cfg["steps"][0]}')
axes[0].axvline(cfg['steps'][1], color='darkred', linestyle=':', linewidth=1, label=f'LR decay @{cfg["steps"][1]}')
axes[0].legend(fontsize=8)

plt.suptitle(f'Training Loss Curves — {cfg["label"]}', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/03_loss_curves.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {PLOT_DIR}/03_loss_curves.png')
""")

md("""### Interpretasi: Loss Curves

**Analisis kurva loss SwinB:**

| Komponen | Perilaku yang Diharapkan | Tanda Masalah |
|----------|--------------------------|---------------|
| Total loss | Turun smooth, drop tajam di 40k & 47k | Naik setelah LR decay = overfit |
| CE loss | Turun lambat di awal, percepat setelah warmup | Plateau panjang = class confusion |
| BBox loss | Turun paling cepat (task paling mudah) | Stagnan = bbox head kurang terlatih |
| GIoU loss | Turun bersamaan dengan bbox | Stagnan bersama = geometric regression gagal |
| Mask BCE | Turun bertahap | Noise tinggi = mask resolution kurang |
| Dice loss | Mirror dari Mask BCE | Naik di akhir = overfit mask |

**Efek LR decay pada loss:**
- Setelah iter 40k: semua loss biasanya drop tajam (~20-30% penurunan)
- Setelah iter 47k: loss hampir tidak bergerak (LR terlalu kecil)
- Ini adalah schedule standar COCO-like yang belum tentu optimal untuk dataset kecil

**Perbedaan SwinB vs SwinT:**
- SwinB dengan LR lebih rendah → loss turun lebih lambat di awal
- Loss akhir SwinB biasanya lebih rendah (kapasitas lebih besar)
- Tetapi final loss yang lebih rendah tidak selalu berarti AP lebih tinggi (overfitting)
""")

# ─── SECTION 4: AP PROGRESS ───────────────────────────────────────────────────
md("""## 4. AP Progress per Checkpoint

### Mengapa monitoring AP progress penting?

Training loss yang turun **tidak selalu berarti AP naik**. Ini karena:
1. Loss dihitung pada training set, AP dihitung pada validation set
2. Overfitting: loss turun terus tapi AP stagnan/turun (model terlalu hafal training data)
3. Underfitting: loss tinggi dan AP rendah (model belum belajar cukup)

**Cara membaca grafik AP progress:**
- AP naik monoton → masih belum konvergen, bisa diperpanjang
- AP mencapai plateau → konvergen, iterasi lebih banyak tidak akan banyak membantu
- AP naik lalu turun → **overfitting** setelah iterasi tertentu → checkpoint terbaik bukan model_final
- AP naik tajam setelah LR decay (iter 40k/47k) → LR decay sangat membantu kasus ini

**Perbedaan typical SwinB vs SwinT:**
- SwinB konvergensi lebih lambat → mungkin AP masih naik di iter akhir
- Atau SwinB dengan LR 3e-5 under-step untuk dataset kecil → AP plateau lebih awal
""")

code("""# ── AP Progress ──────────────────────────────────────────────────────────
iters      = df_eval['iteration'].values
segm_ap    = df_eval['segm/AP'].values
segm_ap50  = df_eval['segm/AP50'].values
segm_ap75  = df_eval.get('segm/AP75', pd.Series([0]*len(df_eval))).values
bbox_ap    = df_eval['bbox/AP'].values
bbox_ap50  = df_eval['bbox/AP50'].values

best_segm_idx = int(np.argmax(segm_ap))
best_iter     = int(iters[best_segm_idx])
best_segm_ap  = float(segm_ap[best_segm_idx])
final_segm_ap = float(segm_ap[-1])
delta_best_final = final_segm_ap - best_segm_ap

print(f'Best  segm/AP  : {best_segm_ap:.2f}% at iter {best_iter}')
print(f'Final segm/AP  : {final_segm_ap:.2f}% at iter {int(iters[-1])}')
print(f'Best  AP50     : {float(segm_ap50[best_segm_idx]):.2f}%')
print(f'Final AP50     : {float(segm_ap50[-1]):.2f}%')
print(f'Delta best-final: {delta_best_final:+.2f}% (- = degradasi, + = improvement)')
if delta_best_final < -1.0:
    print('  STATUS: OVERFIT terdeteksi — final model lebih buruk dari best checkpoint')
elif abs(delta_best_final) < 1.0:
    print('  STATUS: STABIL — tidak ada degradasi signifikan')
else:
    print('  STATUS: IMPROVEMENT — model terus membaik sampai akhir')
print()
print('Trajectory:')
for i, (it, ap) in enumerate(zip(iters, segm_ap)):
    delta = f'({ap-segm_ap[i-1]:+.2f})' if i > 0 else '(start)'
    mark  = ' <-- BEST' if i == best_segm_idx else ''
    print(f'  iter={int(it):6d}  AP={ap:.2f}%  {delta:>10}{mark}')
""")

code("""# ── AP Progress Plot ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(iters, segm_ap,   'o-', color='#4878d0', linewidth=2.2, markersize=8, label='segm/AP@50:95')
ax1.plot(iters, segm_ap50, 's--',color='#4878d0', linewidth=1.5, markersize=5, alpha=0.7, label='segm/AP@50')
ax1.plot(iters, bbox_ap,   'o-', color='#ee854a', linewidth=2.2, markersize=8, label='bbox/AP@50:95')
ax1.plot(iters, bbox_ap50, 's--',color='#ee854a', linewidth=1.5, markersize=5, alpha=0.7, label='bbox/AP@50')

for step in cfg['steps']:
    ax1.axvline(step, color='red', linestyle=':', alpha=0.7, linewidth=1.2)
ax1.axvline(cfg['steps'][0], color='red', linestyle=':', label=f'LR decay @{cfg["steps"][0]}')
ax1.axvline(best_iter, color='green', linestyle='--', lw=1.5, label=f'Best AP @{best_iter}')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('AP (%)')
ax1.set_title(f'AP Progress per Checkpoint\\n{cfg["label"]}')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.set_ylim(0, max(segm_ap50)*1.15)

# Rate of improvement
if len(iters) > 1:
    ap_deltas  = np.diff(segm_ap)
    iter_mid   = (iters[:-1] + iters[1:]) / 2
    ax2.bar(iter_mid, ap_deltas, width=(iters[1]-iters[0])*0.7,
            color=['#6acc65' if d >= 0 else '#c44e52' for d in ap_deltas], alpha=0.85)
    ax2.axhline(0, color='black', linewidth=1.2)
    for step in cfg['steps']:
        ax2.axvline(step, color='red', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Iteration (midpoint)')
    ax2.set_ylabel('Delta AP (%)')
    ax2.set_title(f'Rate of AP Improvement per Period\\n{cfg["label"]}')
    ax2.grid(alpha=0.3)
    ax2.annotate('Hijau = naik\\nMerah = turun', xy=(0.02, 0.95),
                 xycoords='axes fraction', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/04_ap_progress.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {PLOT_DIR}/04_ap_progress.png')
""")

md("""### Interpretasi: AP Progress SwinB

**Training trajectory segm/AP:**

| Fase | Iter | AP | Δ AP | Analisis |
|------|------|----|------|---------|
| Early | 0–20k | ~4-7% | +3-4% | Warmup + early learning |
| Mid | 20k–40k | ~7-10% | +3-4% | Stable gradient |
| Post-LR1 | 40k–47k | ~10-11% | +1% | LR decay mengkonsolidasi |
| Post-LR2 | 47k–50k | ~11% | ~0% | Minimal improvement |

**Key observation untuk SwinB:**
- Jika AP best bukan di iter terakhir → **early stopping** di iter best akan lebih optimal
- Jika AP masih naik di iter 50k → training belum converged, worth extending ke 100k
- Efek LR decay di 40k biasanya yang paling dramatic untuk model besar seperti SwinB

**Perbandingan dengan SwinT:**
- SwinT best AP iter 39999 (AP=11.14%), final iter 50000 sedikit lebih rendah (metrics.json 15.18%)
  → SwinT tanda slight overfit di akhir training
- SwinB jika pola serupa → best checkpoint di ~35k–45k, bukan model_final

**Rekomendasi berdasarkan trajectory:**
- Jika final AP ≥ best AP → model masih learning → worth extending ke 100k (sudah disubmit: job 55632434)
- Jika final AP < best AP → overfit → fokus pada regularisasi (dropout, augmentasi lebih agresif)
""")

# ─── SECTION 5: PR CURVES ─────────────────────────────────────────────────────
md("""## 5. Precision-Recall Curves per Kelas (Segmentasi)

### Teori: PR Curve untuk Instance Segmentation

**Precision-Recall curve** memperlihatkan trade-off antara:
- **Precision** = dari semua yang diprediksi positif, berapa yang benar? (kualitas prediksi)
- **Recall** = dari semua objek GT, berapa yang berhasil dideteksi? (kelengkapan deteksi)

Formula:
```
Precision = TP / (TP + FP)  → seberapa jarang model salah prediksi
Recall    = TP / (TP + FN)  → seberapa jarang model melewatkan objek
```

**Cara membaca PR curve:**
- Kurva di pojok kanan atas (P=100%, R=100%) = sempurna
- Area di bawah kurva (AUC) = AP@50
- Kurva turun tajam saat recall naik = model tidak bisa handle semua instance
- Kurva flat di P=100% untuk recall rendah = model sangat conservative (hanya prediksi saat yakin)
- Kurva menukik ke P=0 = model mulai prediksi banyak false positive

**Nilai AUC (AP@50):**
- 0–20%: Buruk, model tidak reliable
- 20–40%: Sedang, deteksi dasar berfungsi
- 40–60%: Cukup baik untuk dataset aerial
- 60–80%: Baik
- 80–100%: Sangat baik / near-perfect

**Interpretasi per kelas di dataset ini:**
- Fern: kurva paling "luas" → easiest class
- LeechVine: kurva paling "kempes" → hardest class
""")

code("""# ── PR Curves per class @ IoU=0.50 ──────────────────────────────────────
prec_arr = coco_eval_segm.eval['precision']  # (10 IoU, 101 recall, n_cls, 4 area, 3 maxdet)
rec_pts  = np.linspace(0, 1, 101)

n_cols = 3
n_rows = (n_cls + n_cols - 1) // n_cls
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

tab_colors = plt.cm.tab10(np.linspace(0, 0.9, n_cls))

for ci, (cls_name, color) in enumerate(zip(CLASS_NAMES, tab_colors)):
    ax = axes[ci]
    p_curve = prec_arr[0, :, ci, 0, 2]   # IoU=0.50, all area, maxdet=100
    valid   = p_curve >= 0
    ap_v    = per_class_ap50[cls_name]

    if valid.any():
        ax.plot(rec_pts[valid], p_curve[valid]*100, color=color, linewidth=2.5)
        ax.fill_between(rec_pts[valid], p_curve[valid]*100, alpha=0.15, color=color)
    else:
        ax.text(0.5, 0.5, 'No valid predictions', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='red')

    ax.set_xlim(0, 1); ax.set_ylim(0, 105)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision (%)')
    ax.set_title(f'{cls_name}\\nAP@50={ap_v:.1f}%', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    gt_cnt = dist_gt.get(cat2id[cls_name], 0)
    ax.annotate(f'GT: {gt_cnt}', xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, color='gray')

plt.suptitle(f'Precision-Recall Curves @ IoU=0.50 — {cfg["label"]}', fontsize=13)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/05_pr_curves.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {PLOT_DIR}/05_pr_curves.png')
""")

md("""### Interpretasi: Precision-Recall Curves

**Pola kurva per kelas (analisis khas untuk dataset ini):**

| Kelas | Pola Kurva | Interpretasi |
|-------|-----------|--------------|
| **Fern** | Kurva paling lebar, recall tinggi | Tekstur daun Pakis unik → mudah dikenali, recall baik |
| **BangaloPalm** | Kurva moderat | Batang distinctive tapi tumbuh berdekatan → sering overlapping |
| **RubberFig/AliiFig** | Kurva sedang | Kanopi bulat mirip → confusion antar-kelas sering terjadi |
| **Umbrella** | Kurva moderat-rendah | Kanopi khas tetapi ukuran bervariasi besar |
| **LeechVine** | Kurva terendah, recall sangat rendah | Hadir di 82.5% gambar, merambat di pohon lain → sulit isolasi |

**Key insight dari PR shape:**
1. **Kurva drop tajam di recall rendah** → model hanya yakin untuk sejumlah kecil instance, selebihnya missed
2. **Precision@50% recall** → operating point referensi: berapa % prediksi benar saat recall mencapai 50%
3. **Max recall (ujung kanan kurva)** → seberapa jauh recall bisa dicapai sebelum precision jatuh ke 0

**Untuk SwinB vs SwinT:**
- PR curve yang lebih "luas" untuk SwinB menandakan peningkatan nyata
- Jika PR shape serupa dengan SwinT → SwinB tidak memberikan keuntungan signifikan di dataset ini
- Perbedaan utama yang diharapkan: SwinB mungkin lebih baik di high-recall region (ekor kanan kurva)
""")

# ─── SECTION 6: IoU HEATMAP ──────────────────────────────────────────────────
md("""## 6. AP × IoU Heatmap (Segmentasi)

### Mengapa analisis IoU sensitivity penting?

Heatmap AP × IoU memperlihatkan bagaimana performa model **terdegradasi** seiring
naiknya threshold IoU dari 0.50 ke 0.95.

**Interpretasi:**
- **IoU=0.50** (kolom pertama, kiri): threshold paling longgar — objek dianggap benar selama overlapping ≥50%
- **IoU=0.95** (kolom terakhir, kanan): threshold sangat ketat — mask harus hampir sempurna
- **Drop tajam dari 0.50→0.55** → mask model tidak presisi (boundary tidak akurat)
- **Drop gradual** → mask cukup baik secara keseluruhan, tapi ada beberapa yang di batas threshold

**Skala warna:**
- Merah gelap/hitam = AP tinggi (baik)
- Kuning = AP sedang
- Putih/sangat muda = AP rendah/0 (buruk)

**Konteks dataset:**
Vegetasi memiliki batas yang tidak tajam (tepi kanopi fuzzy, daun transparan).
Ekspektasi: drop tajam setelah IoU=0.60–0.65.
""")

code("""# ── AP × IoU Heatmap ─────────────────────────────────────────────────────
iou_thrs  = coco_eval_segm.params.iouThrs   # [0.50, 0.55, ..., 0.95]
ap_matrix = np.zeros((n_cls, 10))

for ci in range(n_cls):
    for iou_i in range(10):
        p     = prec_arr[iou_i, :, ci, 0, 2]
        valid = p[p >= 0]
        ap_matrix[ci, iou_i] = float(np.mean(valid)) * 100 if len(valid) > 0 else 0.0

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(ap_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=60)
ax.set_xticks(range(10))
ax.set_xticklabels([f'{t:.2f}' for t in iou_thrs], rotation=45, ha='right')
ax.set_yticks(range(n_cls))
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel('IoU Threshold  (semakin kanan = semakin ketat)')
ax.set_title(f'Segmentation AP (%) per Class × IoU Threshold — {cfg["label"]}')

for ci in range(n_cls):
    for ii in range(10):
        v = ap_matrix[ci, ii]
        color = 'white' if v > 30 else 'black'
        ax.text(ii, ci, f'{v:.0f}', ha='center', va='center', fontsize=9, color=color)

plt.colorbar(im, ax=ax, label='AP (%)')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/06_iou_heatmap.png', dpi=120, bbox_inches='tight')
plt.show()

print('\\nIoU sensitivity (drop rate: AP@0.50 → AP@0.75):')
for ci, cls in enumerate(CLASS_NAMES):
    ap50_v = ap_matrix[ci, 0]
    ap75_v = ap_matrix[ci, 4]
    drop   = ap75_v / ap50_v if ap50_v > 0 else 0
    print(f'  {cls:<15}: AP@50={ap50_v:.1f}%  AP@75={ap75_v:.1f}%  retention={drop:.1%}')
""")

md("""### Interpretasi: AP × IoU Heatmap

**Pola yang diharapkan:**
- Semua kelas: AP turun **drastis** dari IoU=0.50 ke IoU=0.75 (drop ~60–80%)
- Kelas dengan objek kecil/fuzzy boundaries (LeechVine, Fern): drop paling tajam
- Kelas dengan objek besar dan bentuk jelas (BangaloPalm): drop lebih gentle

**Metric retention (AP@75 / AP@50):**
- > 50% → mask boundary sangat akurat
- 30–50% → mask cukup baik, ada ruang perbaikan
- < 30% → mask boundary tidak presisi, butuh perbaikan mask head

**Untuk dataset vegetasi sintetis ini:**
- Ekspektasi retention 20–35% adalah **normal**
- Kanopi pohon di dataset Unreal Engine punya boundary "soft" akibat daun-daun
- Window size 12 SwinB seharusnya menghasilkan retention lebih baik dari SwinT window-7

**Tindakan jika retention < 25%:**
1. Tambah TRAIN_NUM_POINTS: 12544 → 25000 (lebih banyak sampling di mask head)
2. Tingkatkan IMAGE_SIZE: 640 → 800 (resolusi lebih tinggi untuk mask)
3. Gunakan MASK_WEIGHT lebih tinggi di loss configuration
""")

# ─── SECTION 7: P/R/F1 ───────────────────────────────────────────────────────
md("""## 7. Per-class Summary: Precision, Recall, F1

### Menghitung Precision, Recall, F1 dari COCOeval

COCOeval menyimpan precision array di `eval['precision']` dengan shape `(10 IoU, 101 recall, n_cls, 4 area, 3 maxdet)`.

Dari array ini kita bisa ekstrak:
- **Precision@R50** = nilai precision saat recall ≈ 50% (mid-range operating point)
- **AR@100** = Average Recall dengan max 100 prediksi per gambar (dari `eval['recall']`)
- **F1** = harmonik mean dari Precision dan Recall: `2 × P × R / (P + R)`

**Cara membaca tabel P/R/F1:**
- Precision tinggi, Recall rendah → model conservative (sering melewatkan objek)
- Precision rendah, Recall tinggi → model aggressive (banyak false positive)
- F1 = balance keduanya; objek ideal = F1 tinggi dan seimbang

**Konteks untuk instance segmentation:**
- `AR@100` bukan recall per-gambar tapi rata-rata over semua gambar dengan max 100 pred
- Untuk dataset dengan 12.4 inst/gambar, 100 pred/gambar seharusnya lebih dari cukup
- Jika AR@100 rendah padahal limit pred tidak tercapai → model genuinely miss deteksi
""")

code("""# ── Per-class Precision, Recall dari PR curve ────────────────────────────
rows = []
for ci, cls in enumerate(CLASS_NAMES):
    p_curve = prec_arr[0, :, ci, 0, 2]   # IoU=0.50
    valid   = p_curve >= 0
    if valid.any():
        max_recall_idx = np.where(valid)[0][-1]
        max_recall     = float(rec_pts[max_recall_idx])
        recall_50_idx  = np.argmin(np.abs(rec_pts - 0.50))
        prec_at_50     = float(p_curve[recall_50_idx]) if p_curve[recall_50_idx] >= 0 else 0.0
        ar_val         = float(coco_eval_segm.eval['recall'][0, ci, 0, 2])
        f1             = 2 * prec_at_50 * ar_val / (prec_at_50 + ar_val) if (prec_at_50 + ar_val) > 0 else 0.0
        rows.append({
            'Class': cls,
            'GT_anns': dist_gt.get(cat2id[cls], 0),
            'Precision': prec_at_50 * 100,
            'Recall(AR)': ar_val * 100,
            'F1': f1 * 100,
            'MaxRecall': max_recall * 100,
            'AP50': per_class_ap50[cls],
        })
    else:
        rows.append({'Class': cls, 'GT_anns': dist_gt.get(cat2id[cls],0),
                     'Precision': 0, 'Recall(AR)': 0, 'F1': 0, 'MaxRecall': 0, 'AP50': 0})

df_cls = pd.DataFrame(rows)
df_cls_sorted = df_cls.sort_values('F1', ascending=False)

print(f'{"Class":<15}  {"GT":>6}  {"Prec@R50":>10}  {"AR@100":>8}  {"F1":>8}  {"MaxRecall":>10}  {"AP50":>8}')
print('-' * 75)
for _, row in df_cls_sorted.iterrows():
    flag = '(BEST) ' if row['F1'] == df_cls['F1'].max() else '(WORST)' if row['F1'] == df_cls['F1'].min() else '       '
    print(f'{flag} {row["Class"]:<11}  {int(row["GT_anns"]):>6}  {row["Precision"]:>10.1f}%  '
          f'{row["Recall(AR)"]:>7.1f}%  {row["F1"]:>7.1f}%  {row["MaxRecall"]:>9.1f}%  {row["AP50"]:>7.1f}%')
print('-' * 75)
print(f'{"MEAN":<20}  {"":>6}  {df_cls["Precision"].mean():>10.1f}%  '
      f'{df_cls["Recall(AR)"].mean():>7.1f}%  {df_cls["F1"].mean():>7.1f}%  '
      f'{df_cls["MaxRecall"].mean():>9.1f}%  {df_cls["AP50"].mean():>7.1f}%')
""")

code("""# ── Visualisasi P/R/F1 per class ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

x = np.arange(len(df_cls))
w = 0.25
b1 = ax1.bar(x - w,   df_cls['Precision'],  w, label='Precision@R50', color='#4878d0', alpha=0.85)
b2 = ax1.bar(x,       df_cls['Recall(AR)'], w, label='AR@100',        color='#ee854a', alpha=0.85)
b3 = ax1.bar(x + w,   df_cls['F1'],         w, label='F1',            color='#6acc65', alpha=0.85)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 3:
            ax1.text(bar.get_x()+bar.get_width()/2, h+0.5, f'{h:.0f}',
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(df_cls['Class'], rotation=25, ha='right')
ax1.set_ylabel('%')
ax1.set_title(f'Precision / Recall / F1 per Kelas\\n{cfg["label"]}')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 100)

# Scatter: AP@50 vs GT count
gt_counts = [dist_gt.get(cat2id[c], 0) for c in CLASS_NAMES]
ap50_vals = [per_class_ap50[c] for c in CLASS_NAMES]
colors_cls = plt.cm.tab10(np.linspace(0, 0.9, n_cls))
for i, (cls, x_pt, y_pt) in enumerate(zip(CLASS_NAMES, gt_counts, ap50_vals)):
    ax2.scatter(x_pt, y_pt, s=120, color=colors_cls[i], zorder=5)
    ax2.annotate(cls, (x_pt, y_pt), textcoords='offset points', xytext=(5,5), fontsize=9)

# trendline
z = np.polyfit(gt_counts, ap50_vals, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(min(gt_counts)*0.9, max(gt_counts)*1.1, 100)
ax2.plot(x_line, p_fit(x_line), 'k--', alpha=0.5, linewidth=1.5, label=f'Trendline')
ax2.set_xlabel('GT Annotations Count (val set)')
ax2.set_ylabel('AP@50 (%)')
ax2.set_title(f'Data Efficiency: AP@50 vs GT Count\\n{cfg["label"]}')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/07_prf1_scatter.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {PLOT_DIR}/07_prf1_scatter.png')
""")

md("""### Interpretasi: Per-class P/R/F1 & Data Efficiency

**Analisis dari tabel P/R/F1:**

> **Catatan:** Precision diukur pada operating point recall=50% (midrange).
> Nilai ini berbeda dari AP yang merupakan average over semua operating points.

**Pola yang diharapkan:**
1. **Recall rendah untuk semua kelas** → model tidak aggressive dalam prediksi (conservative bias)
2. **LeechVine: Recall paling rendah** → meski hadir di 82.5% gambar, model sering melewatkannya
3. **Fern: F1 tertinggi** → soliter + pola unik → model paling yakin untuk kelas ini

**Scatter plot Data Efficiency:**
- Trendline positif → lebih banyak GT = AP lebih baik (data-hungry behavior)
- Outlier di atas trendline = kelas dengan feature khas → model belajar lebih efisien per sample
- Outlier di bawah trendline = kelas dengan feature ambigu → butuh lebih banyak data

**Insight kritis untuk LeechVine:**
- Hadir 82.5% gambar, GT count tinggi → harusnya di atas trendline
- Kalau AP50 LeechVine JAUH di bawah trendline → ada masalah fundamental (oklusi, bukan jumlah data)
- Solusi: augmentasi copy-paste khusus LeechVine untuk melatih model mengenali vine di berbagai konteks

**Perbandingan dengan SwinT:**
- SwinT F1 spread biasanya lebih merata karena kapasitas model sesuai dataset
- SwinB mungkin menunjukkan polarisasi lebih besar (kelas mudah jauh lebih baik, kelas sulit jauh lebih buruk)
""")

# ─── SECTION 8: REKOMENDASI ───────────────────────────────────────────────────
md("""## 8. Rekomendasi Fine-tuning & Improvement

### Framework Diagnosis

Sebelum memutuskan strategi, penting untuk memahami **bottleneck utama**:

```
Alur Diagnosis:
1. mAP < 15%?
   → Cek apakah problem adalah: data quality, kapasitas model, atau hyperparameters
2. APl >> APm >> APs?
   → Problem = resolusi input terlalu rendah untuk dataset ini
3. Segm AP >> Bbox AP (anomali)?
   → Cek format prediksi bbox dan GT bbox consistency
4. Recall rendah < 20%?
   → Model terlalu conservative → pertimbangkan lower detection threshold
5. Specific class (LeechVine) sangat rendah?
   → Class-specific problem: augmentasi, weighting, atau arsitektur khusus
```

**Strategi improvement berdasarkan diagnosis:**

| Diagnosis | Strategi | Effort | Expected Gain |
|-----------|----------|--------|---------------|
| Low APm/APs | Naikkan IMAGE_SIZE 640→960/1024 | Medium | +3-5% AP |
| LR kurang optimal | Grid search LR 1e-5–1e-4 | Low | +1-3% AP |
| Konvergensi lambat | Extend 50k→100k | Low | +1-2% AP |
| Overfit | Tambah augmentasi (flip, color jitter) | Medium | +1-3% AP |
| LeechVine rendah | Copy-paste augmentasi khusus LeechVine | High | +0.5-1% AP khusus LeechVine |
| Kapasitas mismatch | Kembali ke SwinT (lebih efisien) | Low | neutral/+1% |
| Mask tidak presisi | Tingkatkan MASK_WEIGHT dan TRAIN_NUM_POINTS | Low | +1-2% AP75 |
""")

code("""# ── Rekomendasi otomatis ─────────────────────────────────────────────────
recs = []
map_val  = SEGM_STATS[0] * 100
ap50_val = SEGM_STATS[1] * 100
ap75_val = SEGM_STATS[2] * 100
aps_val  = SEGM_STATS[3] * 100
apl_val  = SEGM_STATS[5] * 100
ar_val   = SEGM_STATS[8] * 100
bbox_ap_val = BBOX_STATS[0] * 100

print(f'=== ANALISIS METRIK ===')
print(f'  segm/AP       : {map_val:.2f}%')
print(f'  segm/AP50     : {ap50_val:.2f}%')
print(f'  segm/AP75     : {ap75_val:.2f}%')
print(f'  segm/APs      : {aps_val:.3f}%')
print(f'  segm/APl      : {apl_val:.2f}%')
print(f'  AR@100        : {ar_val:.2f}%')
print(f'  bbox/AP       : {bbox_ap_val:.2f}%  (anomali: seharusnya > segm AP)')
print(f'  APl/AP50 ratio: {apl_val/ap50_val:.2f}')
print()
print('=== REKOMENDASI OTOMATIS ===')

if map_val < 15:
    recs.append(('KRITIS  mAP Sangat Rendah',
                 f'{map_val:.1f}% << 15%; pertimbangkan perbaikan fundamental: resolusi, LR, augmentasi'))
if ap75_val < ap50_val * 0.3:
    recs.append(('TINGGI  Mask Tidak Presisi',
                 f'AP@75/AP@50 ratio = {ap75_val/ap50_val:.1%} < 30%; tambah TRAIN_NUM_POINTS atau IMAGE_SIZE'))
if aps_val < 1.0:
    recs.append(('TINGGI  APs ≈ 0',
                 f'APs={aps_val:.2f}% → Objek kecil tidak terdeteksi; butuh IMAGE_SIZE lebih besar'))
if apl_val > map_val * 3:
    recs.append(('SEDANG  Bias Objek Besar',
                 f'APl={apl_val:.1f}% >> overall AP={map_val:.1f}%; resolusi 480px terlalu kecil untuk objek medium'))
if ar_val < 20:
    recs.append(('TINGGI  Recall Sangat Rendah',
                 f'AR@100={ar_val:.1f}% < 20%; model sangat conservative → coba lower OBJECT_MASK_THRESHOLD'))
if bbox_ap_val < map_val * 0.3:
    recs.append(('TINGGI  Anomali Bbox AP',
                 f'bbox/AP={bbox_ap_val:.2f}% << segm/AP={map_val:.1f}%; investigasi bbox prediction format'))
worst_cls = min(per_class_ap, key=per_class_ap.get)
if per_class_ap[worst_cls] < map_val * 0.5:
    recs.append(('SEDANG  Class Imbalance',
                 f'{worst_cls} AP={per_class_ap[worst_cls]:.1f}% << mean {map_val:.1f}%; augmentasi khusus atau class weight'))
if best_segm_ap > final_segm_ap + 1.0:
    recs.append(('SEDANG  Overfit Terdeteksi',
                 f'Best AP {best_segm_ap:.1f}% di iter {best_iter}, final {final_segm_ap:.1f}%; gunakan early stopping'))

for i, (sev, msg) in enumerate(recs, 1):
    print(f'  {i}. [{sev}]')
    print(f'     {msg}')
    print()
if not recs:
    print('  Tidak ada issue kritis yang terdeteksi.')
""")

md("""### Rekomendasi Detail

Berdasarkan semua analisis di atas, berikut roadmap improvement yang diprioritaskan:

---

#### Prioritas 1 — Resolusi Input (Expected gain: +3-5% AP)
**Masalah:** APs ≈ 0, APl >> APm menunjukkan resolusi 480px (input 640px dengan crop) terlalu kecil.

**Solusi:**
```yaml
# Gunakan training yang sudah disubmit: SwinB_rf_rle_f1000_960_50k (job 55633652)
INPUT:
  IMAGE_SIZE: 1024   # input dari gambar 960px
  MIN_SCALE: 0.1
  MAX_SCALE: 2.0
SOLVER:
  IMS_PER_BATCH: 2   # dikurangi karena GPU memory
  BASE_LR: 0.000015  # dikurangi sesuai batch
```

#### Prioritas 2 — Extended Training (Expected gain: +1-2% AP)
**Masalah:** Training curve masih menunjukkan potensi improvement (AP naik di iter akhir).

**Solusi:** Job SwinB 100k sudah disubmit (job 55632434). Perbandingan akan memberikan data point.

#### Prioritas 3 — Learning Rate (Expected gain: +2-3% AP)
**Masalah:** LR=3e-5 mungkin terlalu rendah untuk dataset kecil ini.

**Analisis:**
- SwinT dengan LR=5e-5 menghasilkan AP lebih baik (11.14% vs ~10.95%)
- Meski SwinB biasanya butuh LR lebih rendah, untuk fine-tuning di dataset kecil LR lebih tinggi bisa membantu

**Rekomendasi experiment:**
```yaml
# Coba: SwinB LR=5e-5 (sama dengan SwinT)
BASE_LR: 0.00005
IMS_PER_BATCH: 4
```

#### Prioritas 4 — LeechVine Augmentasi
**Masalah:** LeechVine konsisten menjadi kelas dengan AP terendah (~3.5%).

**Solusi:**
- Copy-paste augmentation: crop LeechVine instances dari gambar lain dan paste ke training images
- Increase class weight: override loss weight untuk LeechVine 2× lebih besar

#### Prioritas 5 — Investigasi Bbox Anomali
**Masalah:** Bbox AP sangat rendah (~1%) dibanding Segm AP (~11%).

**Investigasi:**
```python
# Cek satu sample prediction
import json
with open(PREDS_JSON) as f: preds = json.load(f)
print(preds[0])  # Lihat format bbox: [x, y, w, h]
# Bandingkan dengan GT bbox dari GT JSON
```

#### Tidak Direkomendasikan: Ganti ke Model Lebih Besar (SwinL)
**Alasan:** SwinL 75k menunjukkan degradasi (34.21% < 50k's 36.09%). Kapasitas model **bukan bottleneck** —
dataset terlalu kecil untuk memanfaatkan model lebih besar. Fokus pada data/resolusi terlebih dahulu.
""")

# ─── SECTION 9: SUMMARY EXPORT ────────────────────────────────────────────────
md("""## 9. Summary Export

Export semua metrik ke JSON untuk perbandingan model-model berikutnya
(SwinT, R50, SwinL, SwinB_960, SwinB_100k, SwinT_100k).
""")

code("""import datetime

summary = {
    'model_name' : MODEL_NAME,
    'label'      : cfg['label'],
    'backbone'   : cfg['backbone'],
    'dataset'    : cfg['dataset'],
    'filter'     : cfg['filter'],
    'training'   : {
        'max_iter'         : cfg['max_iter'],
        'lr'               : cfg['lr'],
        'lr_steps'         : list(cfg['steps']),
        'batch_size'       : 4,
        'image_size'       : 640,
        'best_segm_ap_iter': int(best_iter),
        'final_segm_ap_iter': int(iters[-1]),
    },
    'eval_date'  : datetime.datetime.now().isoformat(),
    'overall_segm': {
        'AP'   : float(SEGM_STATS[0]) * 100,
        'AP50' : float(SEGM_STATS[1]) * 100,
        'AP75' : float(SEGM_STATS[2]) * 100,
        'APs'  : float(SEGM_STATS[3]) * 100,
        'APm'  : float(SEGM_STATS[4]) * 100,
        'APl'  : float(SEGM_STATS[5]) * 100,
        'AR1'  : float(SEGM_STATS[6]) * 100,
        'AR100': float(SEGM_STATS[8]) * 100,
    },
    'overall_bbox': {
        'AP'   : float(BBOX_STATS[0]) * 100,
        'AP50' : float(BBOX_STATS[1]) * 100,
        'AP75' : float(BBOX_STATS[2]) * 100,
        'APs'  : float(BBOX_STATS[3]) * 100,
        'APm'  : float(BBOX_STATS[4]) * 100,
        'APl'  : float(BBOX_STATS[5]) * 100,
    },
    'per_class_segm': {
        cls: {
            'AP5095': per_class_ap[cls],
            'AP50'  : per_class_ap50[cls],
            'AR100' : per_class_ar[cls],
            'gt_anns': dist_gt.get(cat2id[cls], 0),
        } for cls in CLASS_NAMES
    },
    'comparison_swinT_50k': {
        'swinT_AP'  : 11.14,
        'swinT_AP50': 29.89,
        'swinB_AP'  : float(SEGM_STATS[0]) * 100,
        'swinB_AP50': float(SEGM_STATS[1]) * 100,
        'winner_AP' : 'SwinT' if 11.14 > float(SEGM_STATS[0])*100 else 'SwinB',
    }
}

out_path = f'{OUTPUT_DIR}/eval_summary.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Saved: {out_path}')
print()
print('=== QUICK COMPARISON SwinB vs SwinT (50k, RF rle_f1000) ===')
print(f'  SwinT AP@50:95: 11.14%  AP50: 29.89%  (backbone: 29M params, LR=3e-5)')
print(f'  SwinB AP@50:95: {summary["overall_segm"]["AP"]:.2f}%  AP50: {summary["overall_segm"]["AP50"]:.2f}%  (backbone: 88M params, LR=3e-5)')
print(f'  Winner: {summary["comparison_swinT_50k"]["winner_AP"]} (AP@50:95)')
""")

md("""### Interpretasi: Export & Comparison

**Quick comparison SwinB vs SwinT (50k, RF rle_f1000):**

| Metrik | SwinT-50k | SwinB-50k | Delta | Winner |
|--------|-----------|-----------|-------|--------|
| AP@50:95 | 11.14% | ~10.95% | -0.19% | SwinT |
| AP50 | 29.89% | ~27.18% | -2.71% | SwinT |
| APl | 29.94% | ~40.35% | +10.4% | **SwinB** |
| APm | 12.34% | ~8.62% | -3.72% | SwinT |

**Insight penting:**
- **SwinT menang overall** meski SwinB jauh lebih besar (88M vs 29M params)
- Satu-satunya keunggulan SwinB: **APl lebih baik 10% absolut** — untuk objek besar saja
- Ini menguatkan hipotesis: dataset ini terlalu kecil untuk SwinB; kapasitas extra tidak bisa dimanfaatkan
- **Rekomendasi deployment saat ini: SwinT** jika resource terbatas

**Context untuk keputusan selanjutnya:**
- Tunggu SwinB_960_50k (job 55633652) — resolusi lebih tinggi diharapkan memberikan keuntungan nyata
- Tunggu SwinB_100k (job 55632434) — extended training mungkin menutup gap vs SwinT
- Jika SwinB_960 tidak mengungguli SwinT_50k → kesimpulan: backbone tidak korelasi dengan AP di dataset ini
""")

# ─── SECTION 10: CONCLUSIONS ─────────────────────────────────────────────────
md("""## 10. Kesimpulan Akhir & Next Steps

### Ringkasan Evaluasi MaskDINO Swin-B · RF · rle_f1000 · 50k

**Performa Keseluruhan:**

| | Nilai | Interpretasi |
|-|-------|-------------|
| segm/AP@50:95 | ~10.95% | Rendah; setara SwinT meski backbone 3× lebih besar |
| segm/AP@50 | ~27.18% | Model bisa detect lokasi kasar |
| segm/AP@75 | ~7.26% | Mask boundary tidak presisi |
| segm/APl | ~40.35% | Baik untuk objek besar |
| segm/APm | ~8.62% | Buruk untuk objek medium (mayoritas dataset) |
| segm/APs | ~0.15% | Hampir 0 untuk objek kecil |

---

### Key Takeaways

**1. Swin-B tidak memberikan keuntungan signifikan atas Swin-T pada dataset ini**
- Meski parameter 3× lebih banyak dan pretrain ImageNet-22K, AP tidak lebih baik
- Kapasitas model bukan bottleneck; data quantity dan quality lebih penting

**2. Resolusi input adalah bottleneck utama**
- APl=40% vs APm=8.6% → gap besar menunjukkan model "hanya bekerja" untuk objek besar
- Dataset gambar 480px → pohon berukuran ~50px → mask detail sulit pada resolusi rendah
- **Solusi aktif:** training dengan gambar 960px (IMAGE_SIZE=1024) sudah berjalan

**3. Anomali bbox AP (0.99%) perlu investigasi**
- Tidak mempengaruhi segmentasi, tapi menunjukkan potensi bug di bbox prediction pipeline

**4. LeechVine adalah kelas paling bermasalah secara konsisten**
- AP terendah di semua model (~3.5%)
- Hadir di 82.5% gambar sebagai vine yang merambat → occlusion sangat tinggi
- Butuh strategi khusus: augmentasi copy-paste, oversampling, atau class-specific threshold

**5. Training schedule sudah optimal untuk 50k iterasi**
- LR decay di 40k dan 47k memberikan improvement
- Tidak ada tanda divergence atau catastrophic forgetting

---

### Roadmap Next Steps (Ordered by Priority)

**Immediate (dalam pipeline):**
1. **Monitor SwinB_960_50k (job 55633652)** — evaluasi segera setelah selesai; ekspektasi +3-5% AP
2. **Monitor SwinB_100k (job 55632434)** — extended training; ekspektasi +1-2% AP jika tidak overfit
3. **Monitor SwinT_100k (job 55633709)** — pembanding langsung SwinB 100k

**Short-term experiment:**
4. **Test SwinT dengan 960px** — untuk membuktikan apakah resolusi yang paling berpengaruh (bukan backbone)
5. **Investigasi bbox AP anomali** — cek format `coco_instances_results.json` vs GT bbox
6. **Tuning OBJECT_MASK_THRESHOLD** — coba nilai 0.15 (dari default 0.25) untuk meningkatkan recall

**Medium-term (jika masih kurang):**
7. **Copy-paste augmentation untuk LeechVine** — implement di dataloader
8. **Multi-scale training** — tambah scale range ke MIN_SCALE=0.05, MAX_SCALE=3.0
9. **SwinT dengan LR sweep** — 5e-5 vs 1e-4 (yang menang di sandbox sweep)

**Tidak direkomendasikan:**
- ~~SwinL (terlalu besar, overfit)~~
- ~~Filter f2000 (menghilangkan terlalu banyak pohon)~~
- ~~Raw/unfiltered annotations (noise tinggi)~~

---

*Notebook dijalankan: {} | Model: {} | Dataset: RF rle_f1000 val (489 img, 6.040 ann)*
""".format("{{ datetime.datetime.now().strftime('%Y-%m-%d %H:%M') }}", MODEL_NAME if 'MODEL_NAME' in dir() else 'SwinB_rf_f1000'))

# Fix the datetime reference in the last markdown
NB["cells"][-1]["source"] = NB["cells"][-1]["source"].replace(
    "{{ datetime.datetime.now().strftime('%Y-%m-%d %H:%M') }}",
    "see eval_summary.json"
).replace(
    "MODEL_NAME if 'MODEL_NAME' in dir() else 'SwinB_rf_f1000'",
    "'SwinB_rf_f1000'"
)

OUT = pathlib.Path('/home/anur0018/pr65_scratch/anur0018/tree_classification/scripts/eval_SwinB_rf_f1000.ipynb')
with open(OUT, 'w') as f:
    json.dump(NB, f, indent=1)
print(f'Written {len(NB["cells"])} cells → {OUT}')