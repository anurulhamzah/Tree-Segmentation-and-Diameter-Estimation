#!/usr/bin/env python3
"""Deteksi under-annotation (COCO buang batang) untuk SELURUH dataset: plantation + rainforest, semua split.
Bandingkan COCO mask-union vs semantic-seg tree mask per gambar.
Flag: ratio(sem/coco) >= 1.5 & missing >= 3% frame.
Output: scripts/coco_vs_semantic_full.csv (semua), ringkasan per scene x split."""
import numpy as np, json, csv
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
from pycocotools import mask as maskUtils

COCODIR='/home/anur0018/pr65_scratch/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000'
SEM={'plantation':'/home/anur0018/pr65_scratch2/anur0018/raw_data/plantations/semantic_segmentation',
     'rainforest':'/home/anur0018/pr65_scratch2/anur0018/raw_data/rainforests/semantic_segmentation'}
def scene(fn): return 'plantation' if '/plantations/' in fn else ('rainforest' if '/rainforests/' in fn else 'unknown')

def rle_mask(seg,h,w):
    if isinstance(seg,dict): r=seg
    else:
        o=maskUtils.frPyObjects(seg,h,w); r=o[0] if len(o)==1 else maskUtils.merge(o)
    return maskUtils.decode(r).astype(bool)

allrows=[]; no_sem=0; bad=0
for split in ['train','val','test']:
    d=json.load(open(f'{COCODIR}/instances_{split}.json'))
    img2ann=defaultdict(list)
    for a in d['annotations']: img2ann[a['image_id']].append(a)
    for im in d['images']:
        sc=scene(im['file_name']); nm=Path(im['file_name']).stem; H,W=im['height'],im['width']; imgA=H*W
        sp=Path(SEM[sc])/f'{nm}.png'
        if not sp.exists(): no_sem+=1; continue
        try: sem=np.array(Image.open(sp).convert('L'))
        except Exception: bad+=1; continue
        sem=np.array(Image.fromarray(sem).resize((W,H),Image.NEAREST))>0
        cu=np.zeros((H,W),bool)
        for a in img2ann[im['id']]: cu|=rle_mask(a['segmentation'],H,W)
        sa=int(sem.sum()); ca=int(cu.sum())
        if sa<800: continue
        missing=int((sem&~cu).sum())
        allrows.append({'split':split,'scene':sc,'file':nm,
            'coco_pct':round(100*ca/imgA,1),'sem_pct':round(100*sa/imgA,1),
            'ratio':round(sa/max(ca,1),2),'missing_pct':round(100*missing/imgA,1)})

def flag(r): return r['ratio']>=1.5 and r['missing_pct']>=3.0
for r in allrows: r['flagged']=flag(r)

OUT='/home/anur0018/pr65_scratch2/anur0018/tree_classification/scripts/coco_vs_semantic_full.csv'
with open(OUT,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0].keys())); w.writeheader()
    for r in sorted(allrows,key=lambda r:-r['missing_pct']): w.writerow(r)

print(f'dinilai {len(allrows)} gambar | tanpa semantic {no_sem} | rusak {bad}')
print(f'\n{"scene":<12}{"split":<7}{"dinilai":>9}{"flagged":>9}{"%":>7}')
agg=defaultdict(lambda:[0,0])
for r in allrows:
    k=(r['scene'],r['split']); agg[k][0]+=1; agg[k][1]+=int(r['flagged'])
for sc in ['plantation','rainforest']:
    for sp in ['train','val','test']:
        n,fl=agg[(sc,sp)]
        if n: print(f'{sc:<12}{sp:<7}{n:>9}{fl:>9}{100*fl/n:>6.1f}%')
flagged=[r for r in allrows if r['flagged']]
print(f'\nTOTAL flagged: {len(flagged)}  ->  plantation {sum(r["scene"]=="plantation" for r in flagged)} | rainforest {sum(r["scene"]=="rainforest" for r in flagged)}')
for thr in [1.3,1.5,2.0]:
    n=sum(1 for r in allrows if r['ratio']>=thr and r['missing_pct']>=3.0)
    print(f'  ratio>={thr} & missing>=3%: {n}')
print(f'\nCSV -> {OUT}')
print('\n--- TOP 15 (missing terbesar, semua scene) ---')
for r in sorted(flagged,key=lambda r:-r['missing_pct'])[:15]:
    print(f"  [{r['scene'][:5]}/{r['split']:<5}] {r['file']:<26} COCO={r['coco_pct']:4.1f}% sem={r['sem_pct']:4.1f}% ratio={r['ratio']:.1f}x missing={r['missing_pct']:4.1f}%")
