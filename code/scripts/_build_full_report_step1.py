#!/usr/bin/env python3
"""Step 1: edit struktural notebook evaluasi (results-independent).
- Tambah panduan teori/SOTA + cara baca angka/grafik sebelum bagian analisis per-model.
- Tambah slot Model #20..#26 supaya semua model ter-discover dievaluasi.
"""
import json, sys

SRC = '/home/anur0018/pr65_scratch2/anur0018/tree_classification/scripts/evaluate_model_template.ipynb'
nb = json.load(open(SRC))

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}

def find_md_idx(substr, start=0):
    for i in range(start, len(nb['cells'])):
        c = nb['cells'][i]
        if c['cell_type'] == 'markdown' and substr in ''.join(c['source']):
            return i
    raise ValueError(f'not found: {substr}')

def find_code_idx(substr, start=0):
    for i in range(start, len(nb['cells'])):
        c = nb['cells'][i]
        if c['cell_type'] == 'code' and substr in ''.join(c['source']):
            return i
    raise ValueError(f'not found: {substr}')

# ─────────────────────────────────────────────────────────────────────────────
# 1) PANDUAN TEORI / SOTA / CARA BACA  — disisipkan tepat sebelum "Model #1"
# ─────────────────────────────────────────────────────────────────────────────
GUIDE = r"""---
# 📖 Panduan Membaca Laporan (teori, SOTA & cara baca angka)

> **Baca bagian ini lebih dulu.** Ke-26 model di bawah dievaluasi dengan *blok analisis 12-section yang identik*, jadi penjelasan teori & cara membaca setiap grafik/angka dikumpulkan di sini sekali. Setiap section model juga mencetak baris `Interpretasi:` otomatis.

## Konteks tugas & ekspektasi angka
Ini **instance segmentation** (deteksi + mask per individu pohon) memakai **MaskDINO**. Metrik utama mengikuti protokol **COCO**. Penting: ini dataset sintetik *below-canopy* yang padat, banyak oklusi, kelas tak seimbang, dan sebagian GT *under-annotated* — jadi **angka absolut wajar lebih rendah** daripada benchmark COCO umum.

**Patokan SOTA (untuk kalibrasi ekspektasi, bukan target langsung):**
| Acuan | mask AP | mask AP50 |
|---|---|---|
| Mask R-CNN R50 (COCO, mudah) | ~34–37% | ~55–58% |
| SOTA instance-seg COCO | ~50%+ | ~70%+ |
| Domain padat/oklusif (dataset ini) | **AP 12–22% sudah wajar** | **AP50 35–55% bagus** |

## Aturan emas membaca semua metrik di laporan ini
- **Semua dalam persen, arah "makin tinggi makin baik".** `0%` = model gagal total (tak ada prediksi benar); `100%` = sempurna.
- **AP50 ≠ AP.** `AP50` = rata-rata presisi pada threshold IoU 0.50 (longgar — "kena kira-kira"). `AP` (metrik utama COCO) = rata-rata pada IoU 0.50→0.95 (ketat — menuntut mask presisi). `AP75` lebih ketat lagi.
- **Lintas-dataset TIDAK apple-to-apple.** AP50 Combined vs RF vs PL tidak bisa diadu langsung (val set & jumlah kelas berbeda). Bandingkan hanya **dalam grup yang sama**.

## Cara baca tiap section per-model
1. **Overview** — jumlah GT vs prediksi, sebaran skor. *Baca:* prediksi ≫ GT atau kelas "over-predicted >5×" → model spam FP.
2. **COCO metrics (segm/bbox)** — AP, AP50, AP75, AP by size (APs/m/l), AR@1/10/100. *Baca:* **gap AP50−AP besar (>18pp)** = deteksi OK tapi *mask kasar*. **APs≈0** = objek kecil nyaris gagal (perlu resolusi lebih tinggi). **AR@100** = batas atas recall.
3. **Per-class AP & AP by size** — bar per kelas + AP per ukuran objek + segm vs bbox. *Baca:* kelas di bawah garis mean = lemah. segm < bbox → bottleneck di kualitas mask, bukan deteksi.
4. **Loss curves** — total/CE/bbox/GIoU/mask/dice vs iterasi (garis merah putus = LR decay). *Baca:* loss turun & datar = konvergen; masih menurun tajam di akhir = *under-trained* (tambah iterasi).
5. **AP progress per checkpoint** — kurva AP & Δ-gain antar checkpoint. *Baca:* gain akhir >0.3pp = **belum konvergen**; bila best checkpoint ≠ checkpoint inference → re-run inference dari best.
6. **PR curves per kelas (IoU 0.50)** — 🟢≥30% / 🟡15–30% / 🔴<15% AP50. *Baca:* kurva "menggembung" ke kanan-atas = kelas kuat; jatuh di recall rendah = hanya instance mudah yang tertangkap.
7. **AP × IoU heatmap** — AP per kelas pada 10 threshold IoU (0.50→0.95). *Baca:* warna pudar ke kanan = mask makin meleset saat dituntut presisi; **drop AP50→AP75 = ukuran ketidakpresisian mask**.
8. **Per-class Precision/Recall/F1 + data efficiency** — *Baca:* Precision tinggi & Recall rendah = model "pemalu" (banyak miss, sedikit FP) → turunkan threshold. Scatter AP50-vs-jumlah-GT: tren naik = kelas minoritas lemah karena kurang data.
9. **Per-image F1 distribution** — histogram F1 per gambar (greedy IoU≥0.5, score≥0.5). *Baca:* **% gambar F1=0** tinggi (>40%) = banyak scene gagal total; median F1 = performa "gambar tipikal".
10. **Visualisasi mask best/worst** — kiri=prediksi, kanan=GT. **Hijau=TP, Merah=FP, Kuning=salah-kelas, Oranye dashed=FN (terlewat).**
11. **Confusion matrix (IoU-matched)** — kiri absolut, kanan row-norm (≈recall). *Baca:* diagonal kuat = klasifikasi benar; sel off-diagonal panas = pasangan kelas yang sering tertukar.
12. **Rekomendasi fine-tuning** — checklist otomatis 🔴/🟡/🟢 berbasis threshold metrik di atas.

## Glosarius singkat
**TP** benar; **FP** prediksi salah/berlebih; **FN** GT terlewat; **WrongClass** lokasi benar tapi label salah. **Precision** = TP/(TP+FP) (seberapa bersih prediksi); **Recall** = TP/(TP+FN) (seberapa lengkap menangkap); **F1** = harmonik keduanya.
"""

idx_model1 = find_md_idx('Model #1')
nb['cells'].insert(idx_model1, md(GUIDE))
print(f'Guide disisipkan di idx {idx_model1}')

# ─────────────────────────────────────────────────────────────────────────────
# 2) SLOT MODEL #20..#26  — disisipkan tepat sebelum "Tabel Resume"
# ─────────────────────────────────────────────────────────────────────────────
idx_resume = find_md_idx('Tabel Resume Perbandingan')
new_cells = []
for n in range(20, 27):          # #20..#26
    slot = n - 1                 # MODELS index
    new_cells.append(md(f"### \U0001F333 Model #{n}\n"))
    new_cells.append(code(
        f"if {slot} < len(MODELS):\n"
        f"    deep_eval_model(MODELS[{slot}])\n"
        f"else:\n"
        f"    pass  # tidak ada model di slot ini\n"))
nb['cells'][idx_resume:idx_resume] = new_cells
print(f'7 slot (Model #20..#26) disisipkan sebelum idx {idx_resume}')

json.dump(nb, open(SRC, 'w'), indent=1)
print(f'TOTAL cells sekarang: {len(nb["cells"])}')