#!/usr/bin/env python3
"""Bandingkan COCO vs RAW instance-seg (non-putih) untuk SELURUH dataset.
Acuan 'seharusnya' = raw_data instance_segmentation non-putih (semua piksel pohon).
Hitung berapa gambar yang COCO-nya jauh di bawah raw."""
import numpy as np, json, csv
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
from pycocotools import mask as maskUtils

COCODIR='/home/anur0018/pr65_scratch/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000'
INST={'plantation':'/home/anur0018/pr65_scratch2/anur0018/raw_data/plantations/instance_segmentation',
      'rainforest':'/home/anur0018/pr65_scratch2/anur0018/raw_data/rainforests/instance_segmentation'}
def scene(fn): return 'plantation' if '/plantations/' in fn else ('rainforest' if '/rainforests/' in fn else '?')
def rle_mask(seg,h,w):
    if isinstance(seg,dict): r=seg
    else:
        o=maskUtils.frPyObjects(seg,h,w); r=o[0] if len(o)==1 else maskUtils.merge(o)
    return maskUtils.decode(r).astype(bool)

rows=[]; bad=0
for split in ['train','val','test']:
    d=json.load(open(f'{COCODIR}/instances_{split}.json'))
    img2ann=defaultdict(list)
    for a in d['annotations']: img2ann[a['image_id']].append(a)
    for im in d['images']:
        sc=scene(im['file_name']); nm=Path(im['file_name']).stem; H,W=im['height'],im['width']; imgA=H*W
        p=Path(INST[sc])/f'{nm}.png'
        if not p.exists(): continue
        try: inst=np.array(Image.open(p).convert('RGB'))
        except Exception: bad+=1; continue
        if inst.shape[:2]!=(H,W):
            inst=np.array(Image.fromarray(inst).resize((W,H),Image.NEAREST))
        raw=~np.all(inst==255,axis=-1)
        cu=np.zeros((H,W),bool)
        for a in img2ann[im['id']]: cu|=rle_mask(a['segmentation'],H,W)
        ra=int(raw.sum()); ca=int(cu.sum())
        if ra<800: continue
        missing=int((raw&~cu).sum())
        rows.append({'split':split,'scene':sc,'file':nm,
            'coco_pct':round(100*ca/imgA,1),'raw_pct':round(100*ra/imgA,1),
            'ratio':round(ra/max(ca,1),2),'missing_pct':round(100*missing/imgA,1)})

OUT='/home/anur0018/pr65_scratch2/anur0018/tree_classification/scripts/coco_vs_rawinstance_full.csv'
with open(OUT,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader()
    for r in sorted(rows,key=lambda r:-r['missing_pct']): w.writerow(r)

import numpy as np
print(f'dinilai {len(rows)} gambar | rusak {bad}')
for sc in ['plantation','rainforest']:
    sub=[r for r in rows if r['scene']==sc]
    rt=np.array([r['ratio'] for r in sub]); ms=np.array([r['missing_pct'] for r in sub])
    print(f'\n=== {sc} ({len(sub)}) ===')
    print('  ratio raw/coco pct:', {p:round(float(np.percentile(rt,p)),2) for p in [50,75,90,95]})
    print('  missing_pct pct   :', {p:round(float(np.percentile(ms,p)),1) for p in [50,75,90,95]})
    for thr in [3,5,8,12]:
        print(f'    missing_pct >= {thr}%: {int((ms>=thr).sum())} ({100*(ms>=thr).mean():.1f}%)')
print(f'\nCSV -> {OUT}')
