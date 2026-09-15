#!/usr/bin/env python3
"""Regenerasi mask COCO dari raw instance-seg (toleran per-instance). NON-DESTRUKTIF.
Asli: filtered_rle_f1000/  ->  Output BARU: filtered_rle_f1000_repaired/
- PLANTATION: regenerasi toleran per-instance (recover batang + tepi daun antialiasing).
- RAINFOREST: disalin APA ADANYA (toleran≈exact & 0 bermasalah -> tak perlu diubah, hindari risiko + hemat waktu).
Metode plantation (proto tervalidasi): tiap anotasi COCO dicocokkan ke warna instance-nya;
mask baru = piksel non-putih yg warna-palette-present-TERDEKAT-nya = warna itu & lebih dekat ke warna drpd putih.
Jamin new ⊇ old. Pertahankan id/category_id/image_id + metadata; hitung ulang bbox/area/RLE.
"""
import numpy as np, json, csv, os, time
from pathlib import Path
from collections import defaultdict
from PIL import Image
from pycocotools import mask as maskUtils

SRC='/home/anur0018/pr65_scratch/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000'
DST='/home/anur0018/pr65_scratch/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000_repaired'
RAW='/home/anur0018/pr65_scratch2/anur0018/raw_data'
REPORT='/home/anur0018/pr65_scratch2/anur0018/tree_classification/scripts/repair_report.csv'
os.makedirs(DST, exist_ok=True)

def scene(fn): return 'plantation' if '/plantations/' in fn else 'rainforest'
INSTDIR=f'{RAW}/plantations/instance_segmentation'
PAL=np.array([(int(r['R']),int(r['G']),int(r['B'])) for r in csv.DictReader(open(f'{RAW}/plantations/color_palette_Part I.csv'))])
PALSET=set(tuple(int(x) for x in c) for c in PAL)

def rle_decode(seg,h,w):
    if isinstance(seg,dict): r=seg
    else:
        o=maskUtils.frPyObjects(seg,h,w); r=o[0] if len(o)==1 else maskUtils.merge(o)
    return maskUtils.decode(r).astype(bool)
def rle_encode(m):
    r=maskUtils.encode(np.asfortranarray(m.astype(np.uint8))); r['counts']=r['counts'].decode('ascii'); return r

report=open(REPORT,'w',newline=''); rw=csv.writer(report)
rw.writerow(['split','file','ann_id','scene','category_id','old_area','new_area','ratio','status'])
stats=defaultdict(int); t0=time.time()

for split in ['val','train','test']:
    d=json.load(open(f'{SRC}/instances_{split}.json'))
    img2ann=defaultdict(list)
    for a in d['annotations']: img2ann[a['image_id']].append(a)
    out_anns=[]; n_img=len(d['images']); done=0; ov_px=0
    for im in d['images']:
        done+=1
        if done%300==0:
            print(f'[{split}] {done}/{n_img} ({time.time()-t0:.0f}s)', flush=True); report.flush()
        anns=img2ann[im['id']]
        if not anns: continue
        sc=scene(im['file_name'])
        if sc=='rainforest':                                    # SALIN apa adanya
            for a in anns:
                out_anns.append(a); stats['rainforest_copy']+=1
            continue
        H,W=im['height'],im['width']; nm=Path(im['file_name']).stem
        p=Path(INSTDIR)/f'{nm}.png'; inst=None
        if p.exists():
            try:
                inst=np.array(Image.open(p).convert('RGB'))
                if inst.shape[:2]!=(H,W): inst=np.array(Image.fromarray(inst).resize((W,H),Image.NEAREST))
            except Exception: inst=None
        if inst is None:                                        # fallback: pertahankan asli
            for a in anns:
                out_anns.append(a); stats['FALLBACK_no_png']+=1
                rw.writerow([split,nm,a['id'],sc,a['category_id'],a['area'],a['area'],1.0,'FALLBACK_no_png'])
            continue
        flat=inst.reshape(-1,3)
        uimg=np.unique(flat,axis=0)
        present=np.array([c for c in uimg if tuple(int(x) for x in c) in PALSET])
        present_t=[tuple(int(x) for x in c) for c in present]
        nonwhite=~np.all(flat==255,axis=1); idx_nw=np.where(nonwhite)[0]
        nn=valid=None
        if len(present) and len(idx_nw):
            pix=flat[idx_nw].astype(np.int32)
            d2=((pix[:,None,:]-present.astype(np.int32)[None,:,:])**2).sum(-1)
            nn=d2.argmin(1); valid=d2.min(1)<((pix-255)**2).sum(1)
        union_seen=np.zeros(H*W,bool)
        for a in anns:
            old=rle_decode(a['segmentation'],H,W).reshape(-1); oa=int(old.sum())
            status='ok'; newm=None
            if nn is not None:
                sub=flat[old]; uc,cc=np.unique(sub,axis=0,return_counts=True); bidx=-1
                for j in np.argsort(-cc):
                    col=tuple(int(x) for x in uc[j])
                    if col==(255,255,255): continue
                    if col in present_t: bidx=present_t.index(col); break
                if bidx>=0:
                    sel=idx_nw[(nn==bidx)&valid]
                    mm=np.zeros(H*W,bool); mm[sel]=True; mm|=old
                    newm=mm
                else: status='FALLBACK_no_color'
            else: status='FALLBACK_empty'
            if newm is None: newm=old.copy()
            na=int(newm.sum())
            if na<oa: status='WARN_shrink'; newm=old.copy(); na=oa
            ov_px+=int((newm&union_seen).sum()); union_seen|=newm
            a2=dict(a); a2['segmentation']=rle_encode(newm.reshape(H,W)); a2['area']=na
            a2['bbox']=[float(x) for x in maskUtils.toBbox(a2['segmentation'])]
            out_anns.append(a2); stats[status]+=1
            rw.writerow([split,nm,a['id'],sc,a['category_id'],oa,na,round(na/max(oa,1),2),status])
    assert len(out_anns)==len(d['annotations']), f"{split}: {len(out_anns)} != {len(d['annotations'])}"
    d['annotations']=out_anns
    json.dump(d, open(f'{DST}/instances_{split}.json','w'))
    print(f'== {split}: {n_img} img, {len(out_anns)} ann | inter-instance overlap_px={ov_px} ==', flush=True); report.flush()

report.close()
print('STATUS:', dict(stats))
print(f'SELESAI -> {DST} ({time.time()-t0:.0f}s) | report {REPORT}')
