#!/usr/bin/env python3
"""Step 3: sisipkan markdown takeaways data-driven (per-model + agregat + summary/rekomendasi)."""
import json, re, glob, os, yaml, pandas as pd, numpy as np

NB='/home/anur0018/pr65_scratch2/anur0018/tree_classification/scripts/evaluate_model_template.ipynb'
CSV='/fs04/scratch2/pr65/anur0018/tree_classification/reports/evaluation/all_models/all_models_resume.csv'
OUTPUT_ROOT='/fs04/scratch2/pr65/anur0018/maskdino_output'
nb=json.load(open(NB))
df=pd.read_csv(CSV)
row_by_name={r['Model']:r for _,r in df.iterrows()}

def md(t): return {"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)}

# ── rederive MODELS discovery order (sorted glob, same filter) ────────────────
GT_KEYS={'RF640','RF960','Combined','PL'}
def gtkey(t):
    n=(t or '').lower()
    if n.startswith('combined'):return 'Combined'
    if n.startswith('plantations'):return 'PL'
    if n.startswith('rainforests'):return 'RF960' if '_960' in n else 'RF640'
    return None
order=[]
for d in sorted(glob.glob(f'{OUTPUT_ROOT}/*')):
    name=os.path.basename(d)
    if name in {'sweep_hyperparams'}: continue
    if not (os.path.exists(f'{d}/model_final.pth') and os.path.exists(f'{d}/inference/coco_instances_results.json') and os.path.exists(f'{d}/config.yaml')): continue
    cfg=yaml.safe_load(open(f'{d}/config.yaml'))
    test=(cfg.get('DATASETS',{}).get('TEST') or [None])[0]
    if gtkey(test) not in GT_KEYS: continue
    order.append(name)
assert len(order)==26, f'expected 26 got {len(order)}'

def f(v):
    try: return float(v)
    except: return float('nan')

def model_takeaway(name):
    r=row_by_name.get(name)
    if r is None: return md(f"> _Takeaways tidak tersedia untuk {name} (tidak ada di resume)._\n")
    ap50,ap,ap75=f(r['AP50']),f(r['AP']),f(r['AP75'])
    aps,ar100=f(r['APs']),f(r['AR100'])
    prec,rec,f1g,mf1=f(r['Prec']),f(r['Rec']),f(r['F1g']),f(r['mF1'])
    best=f(r['AP50*']); rank=int(r['Rank']); grp=r['Group']
    gap=ap50-ap
    bullets=[]
    # mask quality
    if gap>=22: bullets.append(f"**Mask kasar**: gap AP50−AP = {gap:.1f}pp → deteksi jauh lebih baik daripada presisi kontur mask.")
    elif gap>=15: bullets.append(f"Gap AP50−AP = {gap:.1f}pp → kualitas mask sedang; masih ada ruang menaikkan presisi.")
    else: bullets.append(f"Gap AP50−AP = {gap:.1f}pp → mask cukup konsisten dengan deteksi.")
    # small objects
    if aps<0: bullets.append("Objek kecil **APs = N/A** (kategori kecil tak ada/<min di val) — abaikan, bukan indikasi buruk.")
    elif aps<2: bullets.append(f"**Objek kecil nyaris gagal** (APs={aps:.1f}%) → butuh resolusi input lebih tinggi / multi-scale.")
    else: bullets.append(f"Objek kecil APs={aps:.1f}% (objek besar APl jauh lebih kuat — bias ke pohon besar).")
    # recall bottleneck
    if rec<25: bullets.append(f"**Recall sangat rendah** ({rec:.1f}%) → banyak pohon terlewat (FN). Pertimbangkan turunkan score-threshold / tambah query / atasi GT under-annotation.")
    elif rec<40: bullets.append(f"Recall {rec:.1f}% (precision {prec:.1f}%) → model cenderung 'pemalu' (sedikit FP, banyak miss).")
    else: bullets.append(f"Recall {rec:.1f}% & precision {prec:.1f}% relatif seimbang.")
    # checkpoint
    if not np.isnan(best) and best-ap50>2:
        bullets.append(f"**Inference bukan best checkpoint**: best AP50\\*={best:.1f}% > {ap50:.1f}% → re-run inference dari best checkpoint bisa +{best-ap50:.1f}pp gratis.")
    txt=(f"#### 📌 Takeaways — `{name}`\n"
         f"Peringkat **#{rank} di grup {grp}** · AP50 **{ap50:.1f}%** · AP {ap:.1f}% · AP75 {ap75:.1f}% · "
         f"AR@100 {ar100:.1f}% · Prec {prec:.1f}% · Rec {rec:.1f}% · F1 {f1g:.1f}% · macroF1 {mf1:.1f}%\n\n"
         + "\n".join(f"- {b}" for b in bullets) + "\n")
    return md(txt)

# ── sisipkan: after tiap model code cell ─────────────────────────────────────
new=[]
slot_pat=re.compile(r'deep_eval_model\(MODELS\[(\d+)\]\)')
for c in nb['cells']:
    new.append(c)
    if c['cell_type']=='code':
        m=slot_pat.search(''.join(c['source']))
        if m:
            i=int(m.group(1))
            if i<len(order):
                new.append(model_takeaway(order[i]))
nb['cells']=new

# ── aggregate "cara baca" sebelum discovery + takeaway setelah resume/benchmark
def insert_before_code(substr, cell):
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code' and substr in ''.join(c['source']):
            nb['cells'].insert(i,cell); return
def insert_after_code(substr, cell):
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code' and substr in ''.join(c['source']):
            nb['cells'].insert(i+1,cell); return

insert_before_code('Auto-discovery model selesai', md(
"### 🔎 Cara baca tabel discovery di bawah\n"
"Sel ini memindai `maskdino_output/` dan **hanya** menampilkan model yang sudah lengkap "
"(`model_final.pth` + prediksi inference + `config.yaml`). Kolom: **Ch** = jumlah channel input "
"(RGB=3 / RGBD=4 dengan depth), **DBH** = apakah ada head estimasi diameter batang, **GT** = val set "
"acuan (RF640/RF960 rainforest, PL plantation, Combined gabungan). Bila jumlah < total training, sisanya "
"masih berjalan atau belum di-inference.\n"))

# resume table takeaways (computed)
best={g:df[df.Group==g].sort_values('AP50',ascending=False).iloc[0] for g in df.Group.unique()}
res_take=("### 🧭 Cara baca tabel resume\n"
"Diurut per grup lalu AP50↓. Hijau = makin tinggi makin baik (AP/F1); oranye = recall/AR. **AP50\\*** = best "
"checkpoint dari `metrics.json` (kalau jauh di atas AP50 inference → checkpoint terakhir bukan yang terbaik). "
"**Ingat: jangan bandingkan AP50 lintas grup** (val set berbeda).\n\n"
"**Pemenang per grup:** "
f"RF → `{best['RF']['Model']}` ({best['RF']['AP50']:.1f}%), "
f"PL → `{best['PL']['Model']}` ({best['PL']['AP50']:.1f}%), "
f"Combined → `{best['Combined']['Model']}` ({best['Combined']['AP50']:.1f}%).\n")
insert_after_code("df.to_csv(csv_path, index=False)", md(res_take))

insert_after_code("all_models_benchmark.png", md(
"### 📊 Cara baca benchmark\n"
"Bar horizontal kiri = ranking AP50 (warna = grup dataset). Kanan = AP50 vs AP vs AP75 berdampingan: "
"makin lebar gap AP50→AP75, makin kasar mask-nya. Bar tinggi di AP50 tapi pendek di AP75 = 'kelihatan bagus "
"di metrik longgar, lemah saat dituntut presisi'.\n"))

json.dump(nb,open(NB,'w'),indent=1)
print(f'OK. total cells={len(nb["cells"])}, per-model takeaways=26')