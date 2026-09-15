import json, csv, numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from PIL import Image
from pycocotools import mask as maskUtils
from skimage import measure

DIR='/home/anur0018/pr65_scratch/anur0018/tree_classification/data/combined/annotation_inst/filtered_rle_f1000'
OUT='/home/anur0018/pr65_scratch2/anur0018/tree_classification/scripts'
rows=list(csv.DictReader(open(f'{OUT}/gt_audit_files_to_fix.csv')))
rows=[r for r in rows if r['scene']=='plantation']
rows.sort(key=lambda r: float(r['closest_fill']))
sel=rows[:12]
# load needed splits
data={s:json.load(open(f'{DIR}/instances_{s}.json')) for s in set(r['split'] for r in sel)}
def ensure_rle(seg,h,w):
    if isinstance(seg,dict): return seg
    o=maskUtils.frPyObjects(seg,h,w); return o[0] if len(o)==1 else maskUtils.merge(o)
fig,axes=plt.subplots(3,4,figsize=(22,11)); axes=axes.reshape(-1)
for ax,r in zip(axes,sel):
    d=data[r['split']]; iid=int(r['image_id'])
    im=next(x for x in d['images'] if x['id']==iid)
    a2=defaultdict(list)
    for a in d['annotations']: a2[a['image_id']].append(a)
    h,w=im['height'],im['width']
    ax.imshow(np.array(Image.open(im['file_name']).convert('RGB')))
    for b in a2[iid]:
        rle=ensure_rle(b['segmentation'],h,w); bn=maskUtils.decode(rle).astype(bool)
        ov=np.zeros((*bn.shape,4)); ov[bn]=[1,0,0,0.5]; ax.imshow(ov)
        for c in measure.find_contours(bn.astype(float),0.5):
            ax.plot(c[:,1],c[:,0],color='red',lw=1.0)
        bx=b['bbox']; ax.add_patch(mp.Rectangle((bx[0],bx[1]),bx[2],bx[3],fill=False,edgecolor='yellow',lw=1.0))
    ax.set_title(f"[{r['split']}] {r['file_name']}\nclosest: bbox={r['closest_bbox_pct']}% mask={r['closest_mask_pct']}% fill={r['closest_fill']} d={r['closest_depth_m']}m",fontsize=8)
    ax.axis('off')
plt.tight_layout(); out=f'{OUT}/gt_audit_montage.png'; plt.savefig(out,dpi=80,bbox_inches='tight'); print('saved',out)
