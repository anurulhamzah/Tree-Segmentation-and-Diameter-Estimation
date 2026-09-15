#!/usr/bin/env python3
"""Inflate stem conv pretrained 3-channel -> N-channel (default 4: RGB+Depth) untuk warm-start RGBD.

Masalah: load checkpoint Swin RGB (patch_embed.proj.weight [C,3,k,k]) ke model IN_CHANS=4
membuat Detectron2 DROP conv input (shape mismatch) -> stem random -> prior ImageNet di body
rusak oleh noise (lihat run pretrained RGBD AP50=41.7 < scratch 50.4).

Solusi: bikin checkpoint baru dengan stem 4-channel: 3 channel RGB di-copy apa adanya,
channel ke-4 (depth) di-init dari rata-rata RGB (default), zero, atau copy channel terakhir.
Setelah ini tidak ada lagi warning "patch_embed.proj.weight will not be loaded".

Pakai:
  python inflate_stem_rgbd.py IN.pkl OUT.pkl [--key patch_embed.proj.weight] [--init mean|zero|copy] [--in-chans 4]
Mendukung .pkl format Detectron2 ({'model':{...}, '__author__', 'matching_heuristics'}).
"""
import argparse, pickle, sys
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--key', default='patch_embed.proj.weight',
                    help='nama param stem conv (Swin: patch_embed.proj.weight; R50: stem.conv1.weight)')
    ap.add_argument('--init', choices=['mean', 'zero', 'copy'], default='mean',
                    help='inisialisasi channel depth: mean(RGB) [default], zero, atau copy channel terakhir')
    ap.add_argument('--in-chans', type=int, default=4)
    a = ap.parse_args()

    obj = pickle.load(open(a.inp, 'rb'))
    model = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
    if a.key not in model:
        sys.exit(f'ERROR: key {a.key!r} tidak ada. Contoh keys: {list(model)[:5]}')

    w = model[a.key]
    if not torch.is_tensor(w):
        w = torch.as_tensor(w)
    if w.dim() != 4:
        sys.exit(f'ERROR: {a.key} bukan conv 4D (shape={tuple(w.shape)})')
    out_c, in_c, kh, kw = w.shape
    add = a.in_chans - in_c
    if add <= 0:
        sys.exit(f'Tidak perlu inflate: in_chans checkpoint={in_c} >= target {a.in_chans}')

    if a.init == 'mean':
        extra = w.mean(dim=1, keepdim=True).repeat(1, add, 1, 1)
    elif a.init == 'zero':
        extra = torch.zeros(out_c, add, kh, kw, dtype=w.dtype)
    else:  # copy
        extra = w[:, -1:, :, :].repeat(1, add, 1, 1)
    w_new = torch.cat([w, extra], dim=1).contiguous().to(w.dtype)
    model[a.key] = w_new

    out_obj = obj if isinstance(obj, dict) and 'model' in obj else {'model': model}
    if isinstance(out_obj, dict):
        out_obj.setdefault('__author__', 'inflate_stem_rgbd')
        out_obj.setdefault('matching_heuristics', True)
    pickle.dump(out_obj, open(a.out, 'wb'))

    print(f'OK  {a.key}: {tuple(w.shape)} -> {tuple(w_new.shape)}  (depth init={a.init})')
    print(f'    L2(extra)={extra.norm().item():.4f}  L2(rgb)={w.norm().item():.4f}')
    print(f'    SAVED {a.out}')

if __name__ == '__main__':
    main()