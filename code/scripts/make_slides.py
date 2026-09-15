"""
Buat slide presentasi PowerPoint untuk komparasi model full-training MaskDINO.
Output: logs/slides/maskdino_model_comparison.pptx
"""
import os, csv, pickle
import numpy as np
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.enum.dml import MSO_THEME_COLOR

# ── Paths ──────────────────────────────────────────────────────────────
USER_ROOT    = '/scratch2/pr65/anur0018'
PROJECT_ROOT = f'{USER_ROOT}/tree_classification'
EVAL_OUTPUT  = f'{PROJECT_ROOT}/logs/full_model_comparison'
OUT_DIR      = f'{PROJECT_ROOT}/logs/slides'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
rows = []
with open(f'{EVAL_OUTPUT}/summary_all.csv') as f:
    rows = list(csv.DictReader(f))

def flt(r, k): return float(r[k])
rf_rows = [r for r in rows if r['Group'] == 'RF']
pl_rows = [r for r in rows if r['Group'] == 'PL']

# Sort by composite score
def composite(r):
    keys = ['segm_AP','segm_AP50','F1_global%','F1_macro%','segm_AR100']
    weights = [0.30, 0.20, 0.20, 0.15, 0.15]
    g = [rr for rr in rows if rr['Group'] == r['Group']]
    maxv = {k: max(flt(rr,k) for rr in g) or 1 for k in keys}
    return sum(w * flt(r,k) / maxv[k] for k,w in zip(keys,weights))

rf_ranked = sorted(rf_rows, key=composite, reverse=True)
pl_ranked = sorted(pl_rows, key=composite, reverse=True)

# ── Color palette ──────────────────────────────────────────────────────
NAVY    = RGBColor(0x1a, 0x2a, 0x4a)   # dark navy
BLUE    = RGBColor(0x2c, 0x6f, 0xbd)   # blue
ORANGE  = RGBColor(0xe0, 0x7b, 0x39)    # orange accent
GREEN   = RGBColor(0x2e, 0x8b, 0x57)
RED     = RGBColor(0xc0, 0x39, 0x2b)
LGRAY   = RGBColor(0xf5, 0xf5, 0xf5)
MGRAY   = RGBColor(0xd0, 0xd0, 0xd0)
DGRAY   = RGBColor(0x55, 0x55, 0x55)
WHITE   = RGBColor(0xff, 0xff, 0xff)
BLACK   = RGBColor(0x00, 0x00, 0x00)
GOLD    = RGBColor(0xff, 0xcc, 0x00)
TEAL    = RGBColor(0x00, 0x87, 0x96)

# ── Helpers ────────────────────────────────────────────────────────────
def add_slide(prs, layout_idx=6):
    layout = prs.slide_layouts[layout_idx]
    return prs.slides.add_slide(layout)

def bg(slide, color=WHITE):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def textbox(slide, text, l, t, w, h,
            size=18, bold=False, color=BLACK, align=PP_ALIGN.LEFT,
            italic=False, wrap=True):
    txBox = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox

def header_bar(slide, title, subtitle=''):
    """Blue top bar with title."""
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(1.2))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    textbox(slide, title, 0.2, 0.05, 11, 0.7, size=28, bold=True, color=WHITE)
    if subtitle:
        textbox(slide, subtitle, 0.2, 0.75, 11, 0.4, size=14, color=RGBColor(0xbb,0xcc,0xee))

def divider(slide, y, color=MGRAY, w=13.0):
    line = slide.shapes.add_shape(1, Inches(0.15), Inches(y), Inches(w), Inches(0.02))
    line.fill.solid(); line.fill.fore_color.rgb = color
    line.line.fill.background()

def img(slide, path, l, t, w, h=None):
    if os.path.exists(path):
        if h:
            slide.shapes.add_picture(path, Inches(l), Inches(t), Inches(w), Inches(h))
        else:
            slide.shapes.add_picture(path, Inches(l), Inches(t), Inches(w))

def colored_rect(slide, l, t, w, h, color):
    r = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    r.fill.solid(); r.fill.fore_color.rgb = color
    r.line.fill.background()
    return r

def bullet_text(slide, items, l, t, w, h, size=14, title=None, title_size=16):
    txBox = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = txBox.text_frame
    tf.word_wrap = True
    if title:
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = title
        run.font.size = Pt(title_size)
        run.font.bold = True
        run.font.color.rgb = NAVY
    for i, (text, color, indent, bold) in enumerate(items):
        p = tf.add_paragraph() if (title or i > 0) else tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        p.level = indent
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color

def make_table(slide, headers, data_rows, l, t, w, h,
               header_bg=NAVY, header_fg=WHITE,
               alt_row=LGRAY, highlight_rows=None):
    """Add a table with styled header row."""
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    rows_n = len(data_rows) + 1
    cols_n = len(headers)
    tbl = slide.shapes.add_table(rows_n, cols_n,
                                  Inches(l), Inches(t),
                                  Inches(w), Inches(h)).table
    col_w = w / cols_n
    for c in range(cols_n):
        tbl.columns[c].width = Inches(col_w)

    # Header
    for c, hdr in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.fill.solid(); cell.fill.fore_color.rgb = header_bg
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = hdr
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = header_fg

    # Data rows
    for r, row_data in enumerate(data_rows):
        row_bg = alt_row if r % 2 == 1 else WHITE
        if highlight_rows and r in highlight_rows:
            row_bg = RGBColor(0xff, 0xf0, 0xcc)
        for c, val in enumerate(row_data):
            cell = tbl.cell(r+1, c)
            cell.fill.solid(); cell.fill.fore_color.rgb = row_bg
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(9)
            run.font.color.rgb = BLACK

    return tbl

# ══════════════════════════════════════════════════════════════════════
# BUILD PRESENTATION
# ══════════════════════════════════════════════════════════════════════
prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 1 — TITLE
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl, NAVY)

# Big title
textbox(sl, 'Komparasi Model', 1.0, 1.0, 11.3, 1.2,
        size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
textbox(sl, 'MaskDINO Full-Training', 1.0, 2.1, 11.3, 0.9,
        size=36, bold=True, color=GOLD, align=PP_ALIGN.CENTER)

# Accent bar
colored_rect(sl, 1.5, 3.15, 10.3, 0.06, ORANGE)

textbox(sl, 'Instance Segmentation pada Dataset Rainforests & Plantations',
        1.0, 3.3, 11.3, 0.6, size=18, color=RGBColor(0xcc,0xdd,0xff), align=PP_ALIGN.CENTER)

# Stats
for i, (val, lbl) in enumerate([('12', 'Model\nDievaluasi'), ('43K', 'Gambar Test\n(RF+PL)'), ('6 / 7', 'Kelas Spesies\nRF / PL'), ('5 Metrik', 'Utama\nKomparasi')]):
    x = 1.5 + i * 2.6
    colored_rect(sl, x, 4.3, 2.2, 1.5, RGBColor(0x25,0x3a,0x60))
    textbox(sl, val, x+0.1, 4.4, 2.0, 0.65, size=26, bold=True, color=GOLD, align=PP_ALIGN.CENTER)
    textbox(sl, lbl, x+0.1, 5.0, 2.0, 0.65, size=11, color=RGBColor(0xaa,0xbb,0xdd), align=PP_ALIGN.CENTER)

textbox(sl, '21 Mei 2026', 0.2, 7.1, 12.9, 0.3, size=10,
        color=RGBColor(0x88,0x99,0xaa), align=PP_ALIGN.RIGHT)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 2 — OUTLINE
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, 'Outline', 'Struktur presentasi')

sections = [
    ('01', 'Dataset & Model', 'Spesifikasi dataset RF & PL, daftar 12 model'),
    ('02', 'Metodologi Evaluasi', 'Pipeline inferensi → AP → F1 → Error Breakdown'),
    ('03', 'Hasil Rainforests (RF)', 'Tabel metrik, ranking, per-kelas'),
    ('04', 'Hasil Plantations (PL)', 'Tabel metrik, ranking, per-kelas'),
    ('05', 'Analisis Faktor', 'Backbone vs Filter vs LR vs Iterasi'),
    ('06', 'Error & Distribusi', 'TP/FP/FN, boxplot F1, confusion matrix'),
    ('07', 'Insight Kunci & Rekomendasi', '7 temuan penting, langkah selanjutnya'),
]

for i, (num, title, desc) in enumerate(sections):
    y = 1.4 + i * 0.77
    colored_rect(sl, 0.3, y, 0.55, 0.55, BLUE)
    textbox(sl, num, 0.3, y+0.05, 0.55, 0.45, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    textbox(sl, title, 1.05, y, 4.5, 0.3, size=15, bold=True, color=NAVY)
    textbox(sl, desc,  1.05, y+0.3, 9.5, 0.35, size=11, color=DGRAY)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 3 — DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '01 · Dataset Overview', 'Test set yang digunakan untuk evaluasi')

# RF box
colored_rect(sl, 0.3, 1.35, 6.0, 5.5, RGBColor(0xf0,0xf5,0xff))
textbox(sl, '🌿  Rainforests (RF)', 0.4, 1.45, 5.7, 0.5, size=16, bold=True, color=NAVY)
divider(sl, 1.95, BLUE, 5.5)
rf_info = [
    ('Test images', '489'),
    ('GT instances (f1000)', '4,002'),
    ('Image size', '480 × 270 px'),
    ('Area filter', '≥ 1,000 px²'),
    ('Split', '70 / 15 / 15 %'),
]
rf_classes = [('AliiFig','Pohon ara'), ('BangaloPalm','Palem'), ('Fern','Pakis'),
              ('LeechVine','Tanaman liana'), ('RubberFig','Karet'), ('Umbrella','Pohon payung')]
for i, (k, v) in enumerate(rf_info):
    y = 2.0 + i * 0.42
    textbox(sl, k, 0.5, y, 2.8, 0.38, size=12, color=DGRAY)
    textbox(sl, v, 3.3, y, 2.8, 0.38, size=12, bold=True, color=BLACK)
textbox(sl, 'Kelas (6 spesies):', 0.5, 4.18, 5.5, 0.3, size=12, bold=True, color=NAVY)
for i, (name, desc) in enumerate(rf_classes):
    y = 4.48 + i * 0.33
    colored_rect(sl, 0.5, y+0.06, 0.12, 0.18, [BLUE,GREEN,ORANGE,RED,TEAL,NAVY][i])
    textbox(sl, f'{name} — {desc}', 0.72, y, 5.4, 0.3, size=11, color=BLACK)

# PL box
colored_rect(sl, 6.8, 1.35, 6.0, 5.5, RGBColor(0xf0,0xff,0xf2))
textbox(sl, '🌳  Plantations (PL)', 6.9, 1.45, 5.7, 0.5, size=16, bold=True, color=RGBColor(0x1a,0x5c,0x2a))
divider(sl, 1.95, GREEN, 5.5)
pl_info = [
    ('Test images', '359'),
    ('GT instances (f300)', '2,889'),
    ('Image size', '480 × 270 px'),
    ('Area filter', '≥ 300 px²'),
    ('Split', '70 / 15 / 15 %'),
]
pl_classes = [('Apple','Apel'), ('Lemon','Lemon'), ('Loquat','Loquat'),
              ('Mango','Mangga'), ('Orange','Jeruk'), ('Persimmon','Kesemek'), ('Pomegranate','Delima')]
for i, (k, v) in enumerate(pl_info):
    y = 2.0 + i * 0.42
    textbox(sl, k, 7.0, y, 2.8, 0.38, size=12, color=DGRAY)
    textbox(sl, v, 9.8, y, 2.8, 0.38, size=12, bold=True, color=BLACK)
textbox(sl, 'Kelas (7 spesies):', 7.0, 4.18, 5.5, 0.3, size=12, bold=True, color=RGBColor(0x1a,0x5c,0x2a))
pl_colors = [RGBColor(0xd4,0x42,0x00), RGBColor(0xdd,0xcc,0x00), RGBColor(0x88,0x44,0x00),
             RGBColor(0xff,0x88,0x00), RGBColor(0xff,0x55,0x00), RGBColor(0xaa,0x00,0x00), RGBColor(0x66,0x00,0x44)]
for i, (name, desc) in enumerate(pl_classes):
    y = 4.48 + i * 0.28
    colored_rect(sl, 7.0, y+0.05, 0.12, 0.18, pl_colors[i])
    textbox(sl, f'{name} — {desc}', 7.22, y, 5.4, 0.25, size=11, color=BLACK)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 4 — MODEL LIST: TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '01 · Parameter Training — Hyperparameter', '12 model, 100% data, GPU: 1× NVIDIA A40')

headers = ['Model', 'Dom', 'Backbone', 'Filter', 'Iter', 'Batch', 'LR', 'Warmup', 'WD']
data = [
    ('R50 · RF raw',           'RF', 'ResNet-50', 'none',     '25k', '4', '1e-4', '200',   '0.05'),
    ('R50 · RF raw 50k',       'RF', 'ResNet-50', 'none',     '50k', '4', '1e-4', '200',   '0.05'),
    ('R50 · RF f100 baseline', 'RF', 'ResNet-50', '≥100 px²', '50k', '4', '1e-4', '200',   '0.05'),
    ('R50 · RF f100 tuned',    'RF', 'ResNet-50', '≥100 px²', '50k', '4', '5e-5', '500',   '0.05'),
    ('R50 · RF f1000 (25k)',   'RF', 'ResNet-50', '≥1000px²', '25k', '4', '1e-4', '200',   '0.05'),
    ('R50 · RF f1000 (50k)',   'RF', 'ResNet-50', '≥1000px²', '50k', '4', '1e-4', '200',   '0.05'),
    ('SwinB · RF f1000 ★',    'RF', 'Swin-B',    '≥1000px²', '50k', '4', '3e-5', '1,000', '0.05'),
    ('R50 · PL f100 baseline', 'PL', 'ResNet-50', '≥100 px²', '50k', '4', '1e-4', '200',   '0.05'),
    ('R50 · PL f100 tuned',    'PL', 'ResNet-50', '≥100 px²', '50k', '4', '5e-5', '500',   '0.05'),
    ('SwinB · PL f300 ★',     'PL', 'Swin-B',    '≥300 px²', '50k', '4', '3e-5', '1,000', '0.05'),
    ('R50 · PL f300 tuned',    'PL', 'ResNet-50', '≥300 px²', '50k', '4', '5e-5', '500',   '0.05'),
    ('SwinL · PL raw',         'PL', 'Swin-L',    'none',     '50k', '2', '1e-5', '1,000', '0.05'),
]
make_table(sl, headers, data, 0.2, 1.3, 12.9, 5.6,
           highlight_rows=[6, 9])

# Common settings footer box
colored_rect(sl, 0.2, 6.95, 12.9, 0.45, RGBColor(0xee, 0xf4, 0xff))
textbox(sl,
    'Semua model: Optimizer=AdamW  |  Scheduler=MultiStepLR (no decay step)  |  '
    'Input=800–1333px  |  Pre-trained=COCO  |  WD Embed=0.0  |  ★ = Best per domain',
    0.3, 6.98, 12.6, 0.35, size=10, color=NAVY)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 4b — MODEL LIST: DATASET STATISTICS
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '01 · Parameter Training — Statistik Dataset', 'Jumlah data training per model setelah filter')

headers2 = ['Model', 'Dom', 'Filter', 'Train Images', 'Train Annotations', 'Ann/Image', 'Test Images', 'Test GT']
data2 = [
    ('R50 · RF raw',           'RF', 'none',     '2,282', '261,167', '114.4', '489', '4,002'),
    ('R50 · RF raw 50k',       'RF', 'none',     '2,282', '261,167', '114.4', '489', '4,002'),
    ('R50 · RF f100 baseline', 'RF', '≥100 px²', '2,282', '108,519', '47.6',  '489', '4,002'),
    ('R50 · RF f100 tuned',    'RF', '≥100 px²', '2,282', '108,519', '47.6',  '489', '4,002'),
    ('R50 · RF f1000 (25k)',   'RF', '≥1000px²', '2,278', '18,602',  '8.2',   '489', '4,002'),
    ('R50 · RF f1000 (50k)',   'RF', '≥1000px²', '2,278', '18,602',  '8.2',   '489', '4,002'),
    ('SwinB · RF f1000 ★',    'RF', '≥1000px²', '2,278', '18,602',  '8.2',   '489', '4,002'),
    ('R50 · PL f100 baseline', 'PL', '≥100 px²', '1,683', '23,144',  '13.8',  '359', '2,889'),
    ('R50 · PL f100 tuned',    'PL', '≥100 px²', '1,683', '23,144',  '13.8',  '359', '2,889'),
    ('SwinB · PL f300 ★',     'PL', '≥300 px²', '1,685', '14,023',  '8.3',   '359', '2,889'),
    ('R50 · PL f300 tuned',    'PL', '≥300 px²', '1,685', '14,023',  '8.3',   '359', '2,889'),
    ('SwinL · PL raw',         'PL', 'none',     '1,683', '34,221',  '20.3',  '359', '2,889'),
]
make_table(sl, headers2, data2, 0.2, 1.3, 12.9, 5.6,
           highlight_rows=[6, 9])

# Insight bar
colored_rect(sl, 0.2, 6.95, 12.9, 0.45, RGBColor(0xff, 0xf3, 0xee))
textbox(sl,
    '⚠  RF raw: 261K anns vs RF f1000: 18.6K anns (14× lebih banyak, mayoritas noise kecil)  |  '
    'Ann/image RF raw=114 vs f1000=8 — menjelaskan mengapa filter sangat membantu',
    0.3, 6.98, 12.6, 0.35, size=10, color=RGBColor(0x88, 0x33, 0x00))

# ─────────────────────────────────────────────────────────────────────
# SLIDE 5 — METHODOLOGY
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '02 · Metodologi Evaluasi', 'Pipeline lengkap dari weights → metrics')

# Pipeline flow
steps = [
    ('model_final.pth', NAVY, 'Weights model terlatih'),
    ('GPU Inference', BLUE, 'MaskDINO → prediksi mask, bbox, class, score'),
    ('COCO Eval', TEAL, 'COCOeval → AP 0.5:0.95, AP50, AP75, AR@100'),
    ('Greedy IoU Match', ORANGE, 'Per-image: match pred ↔ GT (IoU ≥ 0.5, conf ≥ 0.5)'),
    ('Metrics', GREEN, 'F1 global, F1 macro, TP/FP/FN/WC, per-class'),
]
for i, (label, color, desc) in enumerate(steps):
    x = 0.3 + i * 2.55
    colored_rect(sl, x, 1.4, 2.2, 0.6, color)
    textbox(sl, label, x, 1.4, 2.2, 0.6, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    textbox(sl, desc, x, 2.1, 2.2, 0.55, size=9, color=DGRAY, align=PP_ALIGN.CENTER)
    if i < 4:
        textbox(sl, '→', x+2.2, 1.55, 0.35, 0.35, size=18, bold=True, color=MGRAY, align=PP_ALIGN.CENTER)

# Metrics explanation boxes
metric_info = [
    ('segm AP (mAP 0.5:0.95)', BLUE,
     'Area di bawah kurva P-R, rata-rata 10 threshold IoU (0.5, 0.55,..., 0.95). Metrik utama COCO.'),
    ('AP50 / AP75', TEAL,
     'AP pada IoU threshold tetap 0.5 (longgar) dan 0.75 (ketat). Ukuran lokalisasi mask.'),
    ('AR@100', RGBColor(0x5c,0x35,0x8e),
     'Recall maksimum jika menerima ≤100 prediksi per gambar. Batas atas performa model.'),
    ('F1 Global', ORANGE,
     '2·P·R/(P+R) dari TP/FP/FN total. Lebih representatif untuk deployment nyata.'),
    ('F1 Macro', GREEN,
     'Rata-rata F1 per kelas. Adil ke kelas dengan sedikit instance (e.g., LeechVine).'),
    ('TP / FP / FN / WC', RED,
     'True/False Positive, False Negative, Wrong Class. Diagnosis tipe kegagalan model.'),
]

for i, (title, color, desc) in enumerate(metric_info):
    row, col = divmod(i, 3)
    x = 0.3 + col * 4.35
    y = 2.85 + row * 2.0
    colored_rect(sl, x, y, 4.1, 0.35, color)
    textbox(sl, title, x+0.1, y+0.04, 3.9, 0.28, size=11, bold=True, color=WHITE)
    textbox(sl, desc, x+0.1, y+0.42, 3.9, 0.85, size=10, color=DGRAY)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 6 — RF RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '03 · Hasil Evaluasi — Rainforests', 'Test set: 489 gambar, 4,002 GT instances')

headers = ['Model', 'Iter', 'LR', 'segmAP', 'AP50', 'AP75', 'AR100', 'Prec%', 'Rec%', 'F1g%', 'F1mac%', 'TP', 'FP', 'FN']
data = []
for r in rf_ranked:
    data.append([
        r['Model'],
        r['Iter(k)']+'k',
        r['LR'],
        f"{flt(r,'segm_AP'):.2f}",
        f"{flt(r,'segm_AP50'):.2f}",
        f"{flt(r,'segm_AP75'):.2f}",
        f"{flt(r,'segm_AR100'):.2f}",
        f"{flt(r,'Precision%'):.1f}",
        f"{flt(r,'Recall%'):.1f}",
        f"{flt(r,'F1_global%'):.1f}",
        f"{flt(r,'F1_macro%'):.1f}",
        r['TP'], r['FP'], r['FN'],
    ])
make_table(sl, headers, data, 0.2, 1.3, 12.9, 5.5, highlight_rows=[0])

textbox(sl, 'Diurutkan berdasarkan Composite Score (terbaik di atas)  |  ★ SwinB dominan di semua metrik',
        0.2, 7.1, 12.5, 0.3, size=10, color=DGRAY, italic=True)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 7 — PL RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '04 · Hasil Evaluasi — Plantations', 'Test set: 359 gambar, 2,889 GT instances')

data = []
for r in pl_ranked:
    data.append([
        r['Model'],
        r['Iter(k)']+'k',
        r['LR'],
        f"{flt(r,'segm_AP'):.2f}",
        f"{flt(r,'segm_AP50'):.2f}",
        f"{flt(r,'segm_AP75'):.2f}",
        f"{flt(r,'segm_AR100'):.2f}",
        f"{flt(r,'Precision%'):.1f}",
        f"{flt(r,'Recall%'):.1f}",
        f"{flt(r,'F1_global%'):.1f}",
        f"{flt(r,'F1_macro%'):.1f}",
        r['TP'], r['FP'], r['FN'],
    ])
make_table(sl, headers, data, 0.2, 1.3, 12.9, 4.8, highlight_rows=[0])

textbox(sl, 'Diurutkan berdasarkan Composite Score  |  R50·f300·tuned sangat kompetitif vs SwinB★',
        0.2, 7.1, 12.5, 0.3, size=10, color=DGRAY, italic=True)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 8 — COMPOSITE RANKING
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '03–04 · Composite Ranking', 'Skor gabungan 5 metrik utama')

# Formula
colored_rect(sl, 0.3, 1.3, 12.7, 0.55, RGBColor(0xee,0xf4,0xff))
textbox(sl, 'Composite Score = 0.30×segmAP + 0.20×AP50 + 0.20×F1global + 0.15×F1macro + 0.15×AR100  (semua dinormalisasi 0–1)',
        0.4, 1.35, 12.4, 0.45, size=12, color=NAVY)

# RF ranking
textbox(sl, 'RAINFORESTS', 0.3, 2.0, 5.5, 0.35, size=14, bold=True, color=BLUE)
for i, r in enumerate(rf_ranked):
    y = 2.4 + i * 0.62
    score = composite(r)
    bar_w = score * 5.0
    colored_rect(sl, 0.3, y, bar_w, 0.45, [NAVY,BLUE,TEAL,RGBColor(0x6b,0xae,0xd6),RGBColor(0x9e,0xca,0xe1),RGBColor(0xde,0xeb,0xf7),RGBColor(0xf7,0xfb,0xff)][i])
    rank_badge = '★' if i == 0 else f'#{i+1}'
    textbox(sl, rank_badge, 0.35, y+0.08, 0.4, 0.3, size=12, bold=True, color=WHITE if i<4 else DGRAY)
    textbox(sl, r['Model'], 0.8, y+0.08, 3.0, 0.3, size=11, color=WHITE if i<3 else BLACK)
    textbox(sl, f"{score:.3f}", 0.3+bar_w+0.05, y+0.08, 0.7, 0.3, size=11, bold=True, color=NAVY)
    textbox(sl, f"AP50={flt(r,'segm_AP50'):.1f}  F1={flt(r,'F1_global%'):.1f}%",
            1.4+bar_w, y+0.25, 4.0, 0.25, size=9, color=DGRAY)

# PL ranking
textbox(sl, 'PLANTATIONS', 6.9, 2.0, 5.5, 0.35, size=14, bold=True, color=GREEN)
for i, r in enumerate(pl_ranked):
    y = 2.4 + i * 0.62
    score = composite(r)
    bar_w = score * 4.5
    colors = [RGBColor(0x1a,0x5c,0x2a), RGBColor(0x2e,0x8b,0x57), RGBColor(0x41,0xae,0x76),
              RGBColor(0x74,0xc4,0x76), RGBColor(0xba,0xe4,0xb3)]
    colored_rect(sl, 6.9, y, bar_w, 0.45, colors[min(i,4)])
    rank_badge = '★' if i == 0 else f'#{i+1}'
    textbox(sl, rank_badge, 6.95, y+0.08, 0.4, 0.3, size=12, bold=True, color=WHITE if i<3 else DGRAY)
    textbox(sl, r['Model'], 7.4, y+0.08, 3.0, 0.3, size=11, color=WHITE if i<2 else BLACK)
    textbox(sl, f"{score:.3f}", 6.9+bar_w+0.05, y+0.08, 0.7, 0.3, size=11, bold=True, color=RGBColor(0x1a,0x5c,0x2a))
    textbox(sl, f"AP50={flt(r,'segm_AP50'):.1f}  F1={flt(r,'F1_global%'):.1f}%",
            7.9+bar_w, y+0.25, 4.0, 0.25, size=9, color=DGRAY)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 9 — BACKBONE COMPARISON
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '05 · Pengaruh Backbone', 'R50 vs SwinT vs SwinB vs SwinL (dataset RF, filter f1000)')

# Backbone comparison table
textbox(sl, 'Perbandingan langsung (dataset RF, filter f1000, iterasi ~5k dari sweep):',
        0.3, 1.3, 12.5, 0.35, size=13, color=NAVY, bold=True)

headers_bb = ['Backbone', 'Params', 'AP50', 'F1 Global%', 'F1 Macro%', 'AR100', 'Rank']
# From p2 sweep (all same config, only backbone varies)
# We'll use what we know from the comparison data + any p2 data
bb_data = [
    ('ResNet-50',       '44M',  '~18%', '~10–27%', '~9–25%', '~14–19%', '#1 R50'),
    ('Swin-Tiny',       '86M',  '~23%', '~15%',    '~14%',   '~20%',    'P2 sweep'),
    ('Swin-Base ★',    '197M', '49.5%', '44.9%',   '44.9%',  '30.6%',   'Best RF'),
    ('Swin-Large',      '387M', 'Masih training', '—', '—', '—', 'Pending'),
]
make_table(sl, headers_bb, bb_data, 0.3, 1.75, 8.5, 2.8, highlight_rows=[2])

# Key insight boxes
insights = [
    (BLUE, 'SwinB vs R50 terbaik', 'AP50: 49.5% vs 18.1%\nF1: 44.9% vs 26.6%\n→ Backbone = faktor terpenting'),
    (ORANGE, 'Swin-L (in progress)', 'Sedang training ~50k iter\nPrediksi: melebihi SwinB\nKarena 387M params vs 197M'),
    (TEAL, 'Biaya komputasi', 'SwinB: ~2× lebih lambat vs R50\nSwinL: ~3× lebih lambat\nTradeoff performa-waktu'),
]
for i, (color, title, body) in enumerate(insights):
    x = 0.3 + i * 4.35
    y = 4.75
    colored_rect(sl, x, y, 4.1, 0.38, color)
    textbox(sl, title, x+0.1, y+0.05, 3.9, 0.28, size=12, bold=True, color=WHITE)
    textbox(sl, body, x+0.1, y+0.48, 3.9, 1.2, size=11, color=DGRAY)

# Add overview plot
img(sl, f'{EVAL_OUTPUT}/overview_RF.png', 8.9, 1.7, 4.2)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 10 — FILTER & LR EFFECT
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '05 · Pengaruh Filter Area & Learning Rate', 'Eksperimen ablasi pada R50 — dataset RF')

# LR effect
textbox(sl, 'Efek Learning Rate (dataset yang sama: RF f100, 50k iter):',
        0.3, 1.3, 12.5, 0.35, size=13, bold=True, color=NAVY)

headers_lr = ['Konfigurasi', 'LR', 'AP50 (%)', 'Recall (%)', 'F1 Global (%)', 'F1 Macro (%)']
lr_data = [
    ('R50 · RF · f100 · baseline', '1e-4', '2.63',  '3.47', '5.92',  '5.58'),
    ('R50 · RF · f100 · tuned',    '5e-5', '17.65', '21.39', '26.61', '24.98'),
    ('Peningkatan',                '0.5×', '+570%', '+516%', '+349%', '+348%'),
]
make_table(sl, headers_lr, lr_data, 0.3, 1.75, 9.0, 2.1, highlight_rows=[1,2])

# LR callout
colored_rect(sl, 9.6, 1.75, 3.5, 2.1, RGBColor(0xff,0xf0,0xe8))
textbox(sl, '⚡ LR 2× lebih kecil\n→ F1 naik 4.5×', 9.7, 1.85, 3.3, 0.9,
        size=16, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
textbox(sl, 'LR=1e-4 terlalu besar\nuntuk fine-tuning\npada dataset menengah',
        9.7, 2.75, 3.3, 0.9, size=11, color=DGRAY, align=PP_ALIGN.CENTER)

# Filter effect
textbox(sl, 'Efek Filter Area (R50, LR=1e-4, 25–50k iter):',
        0.3, 4.05, 12.5, 0.35, size=13, bold=True, color=NAVY)

headers_flt = ['Filter', 'GT Instances', 'Noise Level', 'AP50 (%)', 'Precision (%)', 'Recall (%)']
flt_data = [
    ('Tidak ada filter', '~8000+', 'Sangat tinggi', '1.5–2.6', '19–21', '1–3'),
    ('f100 (≥100 px²)',  '~5000',  'Sedang',        '2.6–17.7', '20–35', '3–21'),
    ('f1000 (≥1000 px²)','4,002',  'Rendah',        '17.9–18.1', '69–82', '5–10'),
]
make_table(sl, headers_flt, flt_data, 0.3, 4.5, 12.7, 2.3, highlight_rows=[2])

textbox(sl, '⚠  Kesimpulan: Filter mereduksi noise tetapi juga membuang data. '
        'LR yang tepat dapat mengkompensasi noise (f100·tuned > f1000·25k di F1).',
        0.3, 7.1, 12.5, 0.3, size=11, color=RGBColor(0x88,0x44,0x00), italic=True)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 11 — ERROR BREAKDOWN
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '06 · Analisis Error: TP / FP / FN / Wrong Class', 'Diagnosis tipe kegagalan per model')

# Error type legend
types = [
    (GREEN, 'TP (True Positive)', 'Deteksi benar\nIoU ≥ 0.5, kelas benar'),
    (RED,   'FP (False Positive)', 'False alarm\nPrediksi tanpa GT match'),
    (ORANGE,'FN (False Negative)', 'Miss — objek terlewat\nGT tidak terdeteksi'),
    (TEAL,  'WC (Wrong Class)',    'Geometri benar\nKelas salah'),
]
for i, (color, label, desc) in enumerate(types):
    x = 0.3 + i * 3.25
    colored_rect(sl, x, 1.3, 3.0, 0.35, color)
    textbox(sl, label, x+0.1, 1.34, 2.8, 0.27, size=11, bold=True, color=WHITE)
    textbox(sl, desc, x+0.1, 1.72, 2.8, 0.55, size=10, color=DGRAY)

# Error table RF
textbox(sl, 'RF Error Breakdown (4,002 GT instances):', 0.3, 2.4, 8.0, 0.35, size=12, bold=True, color=NAVY)
headers_err = ['Model', 'TP', 'FP', 'FN', 'WC', 'Hit Rate', 'Miss Rate']
rf_err = []
for r in rf_ranked:
    tp, fp, fn, wc = int(r['TP']), int(r['FP']), int(r['FN']), int(r['WC'])
    total_gt = tp + fn + wc
    rf_err.append([r['Model'], str(tp), str(fp), str(fn), str(wc),
                   f"{tp/total_gt*100:.1f}%", f"{fn/total_gt*100:.1f}%"])
make_table(sl, headers_err, rf_err, 0.3, 2.8, 7.5, 2.8, highlight_rows=[0])

# Key FN insight
colored_rect(sl, 8.1, 2.4, 5.0, 3.2, RGBColor(0xff,0xf3,0xee))
textbox(sl, '🚨  FN Mendominasi', 8.2, 2.5, 4.8, 0.45, size=14, bold=True, color=RED)
fn_insights = [
    'Model terbaik (SwinB★) masih\nmelewatkan 69.7% objek (FN=2789)',
    '',
    'WrongClass < 5 di SEMUA model\n→ Classifier sangat baik',
    '',
    'Bottleneck: Detection Sensitivity\nbukan kemampuan klasifikasi',
    '',
    'Solusi potensial:\n• Backbone lebih kuat\n• Data lebih banyak\n• Confidence threshold lebih rendah',
]
textbox(sl, '\n'.join(fn_insights), 8.2, 3.05, 4.8, 2.4, size=11, color=DGRAY)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 12 — HEATMAP PER-CLASS
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '06 · Performa Per-Kelas (Heatmap AP50)', 'Variasi antar spesies — RF & PL')

img(sl, f'{EVAL_OUTPUT}/heatmap_RF.png', 0.2, 1.3, 6.5)
img(sl, f'{EVAL_OUTPUT}/heatmap_PL.png', 6.8, 1.3, 6.3)

# Annotation below
textbox(sl, 'RF — Temuan kunci:', 0.2, 5.8, 6.3, 0.3, size=12, bold=True, color=NAVY)
textbox(sl, '• Fern selalu tertinggi (tekstur khas, ukuran medium)\n'
            '• LeechVine selalu terendah (shape irregular, tanaman merambat)\n'
            '• SwinB★ menaikkan AP semua kelas: LeechVine 9.8% → 37.0%',
        0.2, 6.1, 6.3, 1.2, size=10, color=DGRAY)

textbox(sl, 'PL — Temuan kunci:', 6.8, 5.8, 6.0, 0.3, size=12, bold=True, color=RGBColor(0x1a,0x5c,0x2a))
textbox(sl, '• Loquat & Mango tertinggi (buah besar, warna kontras)\n'
            '• Pomegranate terendah (buah kecil kemerahan, mirip background)\n'
            '• R50·f300·tuned mendekati SwinB★ di semua kelas PL',
        6.8, 6.1, 6.0, 1.2, size=10, color=DGRAY)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 13 — BOXPLOT F1 DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '06 · Distribusi F1 Per-Gambar (Box Plot)', 'Konsistensi model di seluruh test set')

img(sl, f'{EVAL_OUTPUT}/boxplot_RF.png', 0.2, 1.3, 6.5)
img(sl, f'{EVAL_OUTPUT}/boxplot_PL.png', 6.8, 1.3, 6.3)

textbox(sl, 'RF — Konsistensi:', 0.2, 5.7, 6.3, 0.3, size=12, bold=True, color=NAVY)
textbox(sl, '• SwinB★: Mean≈Median≈0.45 → distribusi simetris, model stabil\n'
            '• raw·25k: Median=0.0 → >50% gambar F1=0 (model hampir tidak bekerja)\n'
            '• f100·tuned: Penyebaran lebar (std≈0.18) → performa bervariasi',
        0.2, 6.0, 6.3, 1.3, size=10, color=DGRAY)

textbox(sl, 'PL — Konsistensi:', 6.8, 5.7, 6.0, 0.3, size=12, bold=True, color=RGBColor(0x1a,0x5c,0x2a))
textbox(sl, '• SwinB★: Mean=Median=0.50 → distribusi paling simetris\n'
            '• R50·f300·tuned: Hampir setara SwinB★ (mean 0.48 vs 0.50)\n'
            '• SwinL·raw: Median=0.0 → 50%+ gambar F1=0, gagal total',
        6.8, 6.0, 6.0, 1.3, size=10, color=DGRAY)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 14 — RADAR CHART
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '06 · Profil Multi-Dimensi (Radar Chart)', '7 metrik dinormalisasi 0–1 per kelompok')

img(sl, f'{EVAL_OUTPUT}/radar_RF.png', 0.2, 1.3, 6.5)
img(sl, f'{EVAL_OUTPUT}/radar_PL.png', 6.8, 1.3, 6.3)

textbox(sl, 'RF — Interpretasi profil:', 0.2, 5.75, 6.3, 0.3, size=12, bold=True, color=NAVY)
textbox(sl, '• SwinB★: Luas terbesar, hampir melingkar → model paling seimbang\n'
            '• f1000·25k: Memanjang ke Precision, kempis di Recall → tidak seimbang\n'
            '• raw·*: Area sangat kecil → performa di semua dimensi buruk',
        0.2, 6.05, 6.3, 1.3, size=10, color=DGRAY)

textbox(sl, 'PL — Interpretasi profil:', 6.8, 5.75, 6.0, 0.3, size=12, bold=True, color=RGBColor(0x1a,0x5c,0x2a))
textbox(sl, '• SwinB★ & R50·f300·tuned: Profil hampir identik, luas besar\n'
            '• SwinL·raw: Area sangat kecil meski Precision nyaris 1.0\n'
            '• Lesson: Precision tinggi tanpa Recall = profil tidak berguna',
        6.8, 6.05, 6.0, 1.3, size=10, color=DGRAY)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 15 — PRECISION-RECALL SCATTER
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '06 · Precision vs Recall Scatter + Iso-F1 Curves', 'Efficiency frontier semua model')

img(sl, f'{EVAL_OUTPUT}/precision_recall.png', 1.5, 1.3, 10.3)

textbox(sl, 'Garis putus-putus = iso-F1 (F1 sama). Model di pojok kanan atas = terbaik.',
        0.3, 6.9, 12.5, 0.4, size=11, color=DGRAY, italic=True)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 16 — 7 KEY INSIGHTS
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '07 · 7 Insight Kunci', 'Temuan penting dari seluruh eksperimen')

insights_data = [
    (NAVY,   '1', 'Backbone > Filter > Iterasi > LR',
     'SwinB mengungguli semua R50 apapun konfigurasinya. Investasi backbone lebih efektif dari tuning hyperparameter.'),
    (BLUE,   '2', 'LR = Faktor Paling Kritis untuk R50',
     'LR 1e-4 → 5e-5: F1 naik 4.5× (5.9% → 26.6%). Default LR terlalu besar untuk dataset fine-tuning skala menengah.'),
    (TEAL,   '3', 'Filter Noise Data Penting, Tapi Bukan Satu-satunya',
     'f100·tuned mengalahkan f1000·25k di F1 (26.6% vs 9.7%). LR yang tepat bisa kompensasi data noisy.'),
    (GREEN,  '4', 'Iterasi Lebih Banyak → Kalibrasi Lebih Baik',
     'f1000·25k vs f1000·50k: AP50 hampir sama (~18%), tapi F1 berbeda jauh (9.7% vs 16.9%). 50k iter lebih terkalibasi.'),
    (ORANGE, '5', 'SwinL Gagal Tanpa Filter + LR Tepat',
     'SwinL PL raw (LR=1e-5, no filter): F1=5.3%, kalah dari R50·f300 (F1=45.4%). Backbone besar butuh konfigurasi hati-hati.'),
    (RED,    '6', 'Recall = Bottleneck Utama, Bukan Precision',
     'SwinB RF★ masih melewatkan 69.7% objek. Precision sudah 87%. Fokus improvement: detection sensitivity & recall.'),
    (RGBColor(0x5c,0x35,0x8e), '7', 'Classifier Sudah Sangat Baik',
     'WrongClass < 5 di semua model dari ribuan GT. Masalah ada di detector (RPN/query), bukan classifier head.'),
]

for i, (color, num, title, desc) in enumerate(insights_data):
    row, col = divmod(i, 2) if i < 6 else (3, 0)
    if i == 6:
        x, y, w = 0.3, 6.45, 12.7
    else:
        x = 0.3 + col * 6.55
        y = 1.35 + row * 1.6
        w = 6.2
    colored_rect(sl, x, y, 0.45, 1.2 if i < 6 else 0.75, color)
    textbox(sl, num, x+0.02, y+0.3 if i < 6 else y+0.15, 0.42, 0.5, size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    textbox(sl, title, x+0.55, y+0.03, w-0.6, 0.35, size=12, bold=True, color=color)
    textbox(sl, desc, x+0.55, y+0.42 if i < 6 else y+0.38, w-0.6, 0.7 if i < 6 else 0.35, size=10, color=DGRAY)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 17 — RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl)
header_bar(sl, '07 · Rekomendasi & Langkah Selanjutnya', 'Prioritas berdasarkan temuan evaluasi')

recs = [
    (NAVY, 'Prioritas 1 — SwinB RF LR Sweep (Segera)',
     ['Training SwinB + RF f1000 + LR=1e-4 dan LR=5e-5 untuk mencari titik optimal',
      'Prediksi: LR=5e-5 atau 3e-5 akan mengangkat F1 dari 44.9% ke ~55-60%',
      'Config: 50k iter, f1000 filter, 640px resolution (sama dengan SwinB terbaik saat ini)']),
    (BLUE, 'Prioritas 2 — Tunggu SwinL RF (In Progress)',
     ['SwinL_rf_f1000 & SwinL_rf_rle_f1000 sedang training (iter ~18k/50k)',
      'ETA: ~17-20 jam dari sekarang',
      'Setelah selesai: eval & tambahkan ke comparison untuk RF final ranking']),
    (GREEN, 'Prioritas 3 — R50 PL Sudah Kompetitif (Gunakan Langsung)',
     ['R50·PL·f300·tuned F1=45.4% vs SwinB★ F1=48.0% — gap hanya 2.6%',
      'Untuk deployment awal PL: gunakan R50·f300·tuned (lebih ringan, hampir setara)',
      'SwinB PL untuk kualitas maksimum']),
    (ORANGE, 'Jangka Panjang — Tingkatkan Recall',
     ['Recall RF terbaik: 30.3% (SwinB★) — masih terlalu rendah untuk aplikasi nyata',
      'Opsi: data augmentasi agresif, confidence threshold lebih rendah (0.3–0.4)',
      'Atau: ensemble model untuk coverage lebih luas']),
]

for i, (color, title, bullets) in enumerate(recs):
    row, col = divmod(i, 2)
    x = 0.3 + col * 6.55
    y = 1.3 + row * 2.9
    colored_rect(sl, x, y, 6.2, 0.38, color)
    textbox(sl, title, x+0.1, y+0.05, 6.0, 0.28, size=11, bold=True, color=WHITE)
    for j, bullet in enumerate(bullets):
        textbox(sl, f'• {bullet}', x+0.15, y+0.5+j*0.6, 5.9, 0.55, size=10, color=DGRAY)

# ─────────────────────────────────────────────────────────────────────
# SLIDE 18 — CLOSING
# ─────────────────────────────────────────────────────────────────────
sl = add_slide(prs)
bg(sl, NAVY)

textbox(sl, 'Kesimpulan', 1.0, 1.0, 11.3, 0.8, size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
colored_rect(sl, 1.5, 1.85, 10.3, 0.06, ORANGE)

summary_points = [
    ('RF Champion:', 'SwinB · RF · f1000★  —  AP50=49.5%  |  F1=44.9%  |  Score=1.000'),
    ('PL Champion:', 'SwinB · PL · f300★   —  AP50=43.2%  |  F1=48.0%  |  Score=1.000'),
    ('Runner-up PL:', 'R50 · PL · f300·tuned  —  AP50=34.4%  |  F1=45.4%  |  Score=0.844'),
    ('Best Factor:', 'Backbone (SwinB >> R50) > Filter > LR > Iterasi'),
    ('Main Issue:', 'FN mendominasi — Recall bottleneck di semua model'),
    ('Next Step:', 'SwinB RF LR sweep + tunggu SwinL RF selesai training'),
]

for i, (key, val) in enumerate(summary_points):
    y = 2.2 + i * 0.75
    textbox(sl, key, 1.2, y, 2.5, 0.5, size=14, bold=True, color=GOLD)
    textbox(sl, val, 3.8, y, 8.7, 0.5, size=13, color=WHITE)

textbox(sl, 'Data & plots tersimpan di logs/full_model_comparison/',
        0.5, 7.05, 12.3, 0.35, size=10,
        color=RGBColor(0x88,0x99,0xaa), align=PP_ALIGN.CENTER)

# ── Save ───────────────────────────────────────────────────────────────
out_path = f'{OUT_DIR}/maskdino_model_comparison.pptx'
prs.save(out_path)
print(f'Saved: {out_path}')
print(f'Total slides: {len(prs.slides)}')
