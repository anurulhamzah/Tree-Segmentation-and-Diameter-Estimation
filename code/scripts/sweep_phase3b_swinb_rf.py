"""Phase 3b — LR sweep SwinB RF (second-best backbone from Phase 2 RF).

FocalNet_L menang Phase 2 RF dengan margin kecil (3.70% vs SwinB 3.19%).
Script ini sweep LR yang sama agar bisa dibandingkan secara fair.

Jalankan setelah Phase 4 sweep selesai (GPU bebas):
    python scripts/sweep_phase3b_swinb_rf.py

Output: sweep_hyperparams/p3b_rf_SwinB_lr*/
Idempoten: skip jika model_final.pth sudah ada.
"""

import os, sys, json
from pathlib import Path

USER_ROOT    = '/scratch2/pr65/anur0018'
PROJECT_ROOT = Path(USER_ROOT) / 'tree_classification'
MASKDINO_ROOT = Path(USER_ROOT) / 'MaskDINO' / 'MaskDINO'
OUTPUT_ROOT  = Path(USER_ROOT) / 'maskdino_output'
SANDBOX_OUT  = OUTPUT_ROOT / 'sweep_hyperparams'
SANDBOX_CFG  = PROJECT_ROOT / 'configs' / '_sweep'

os.environ.setdefault('MKL_INTERFACE_LAYER', '')
os.environ['RAINFORESTS_ROOT'] = str(
    Path('/home/anur0018/pr65_scratch/anur0018/tree_classification/data/rainforests'))
os.environ['LD_LIBRARY_PATH'] = (
    '/scratch/pr65/anur0018/conda_envs/maskdino/lib:'
    '/scratch/pr65/anur0018/conda_envs/maskdino/lib/python3.9/site-packages/torch/lib:'
    + os.environ.get('LD_LIBRARY_PATH', ''))
os.environ['PYTHONPATH'] = (
    f"{MASKDINO_ROOT}:{PROJECT_ROOT / 'scripts'}:{os.environ.get('PYTHONPATH', '')}")

os.chdir(PROJECT_ROOT)

BASE_CFG = str(PROJECT_ROOT / 'configs' / 'maskdino_SwinB_rf_f1000.yaml')
DATASET_TRAIN = 'rainforests_inst_rle_f1000_train'
DATASET_VAL   = 'rainforests_inst_rle_f1000_val'
IMG_SIZE = 640
MAX_ITER = 5000
LR_VALUES = [1e-5, 3e-5, 5e-5, 1e-4]


def make_config(name, lr):
    steps = f'({int(MAX_ITER * 0.70)}, {int(MAX_ITER * 0.95)})'
    content = f"""_BASE_: {BASE_CFG}

MODEL:
  SEM_SEG_HEAD:
    NUM_CLASSES: 6

DATASETS:
  TRAIN: ("{DATASET_TRAIN}",)
  TEST:  ("{DATASET_VAL}",)

INPUT:
  IMAGE_SIZE: {IMG_SIZE}
  MASK_FORMAT: "bitmask"

SOLVER:
  BASE_LR: {lr}
  MAX_ITER: {MAX_ITER}
  STEPS: {steps}
  WARMUP_ITERS: 200
  CHECKPOINT_PERIOD: {MAX_ITER}

TEST:
  EVAL_PERIOD: {MAX_ITER}
"""
    cfg_path = SANDBOX_CFG / f'{name}.yaml'
    cfg_path.write_text(content)
    return cfg_path


def best_ap50(out_dir):
    p = Path(out_dir) / 'metrics.json'
    if not p.exists():
        return None
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    aps = [r.get('segm/AP50', 0) for r in recs if 'segm/AP50' in r]
    return max(aps) if aps else None


def run(name, lr):
    out_dir = SANDBOX_OUT / name
    ckpt = out_dir / 'model_final.pth'
    if ckpt.exists():
        ap = best_ap50(out_dir)
        print(f'  SKIP {name}  AP50={ap:.2f}%' if ap else f'  SKIP {name}')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = make_config(name, lr)
    cmd = (f'{sys.executable} {PROJECT_ROOT}/scripts/train_rainforests.py '
           f'--config-file {cfg_path} --num-gpus 1 OUTPUT_DIR {out_dir}')
    print(f'\n  RUN  {name}  lr={lr}')
    ret = os.system(cmd)
    ap = best_ap50(out_dir)
    if ret == 0 and ap:
        print(f'  DONE {name}  AP50={ap:.2f}%')
    else:
        print(f'  FAIL {name}  exit={ret}')


if __name__ == '__main__':
    print('=== Phase 3b: SwinB RF LR Sweep ===')
    print(f'Base config : {BASE_CFG}')
    print(f'Dataset     : {DATASET_TRAIN}')
    print(f'Iter        : {MAX_ITER} | img_size={IMG_SIZE}')
    print(f'LR values   : {LR_VALUES}\n')

    results = {}
    for lr in LR_VALUES:
        lr_str = f'{lr:.0e}'
        name = f'p3b_rf_SwinB_lr{lr_str}'
        run(name, lr)
        ap = best_ap50(SANDBOX_OUT / name)
        results[lr_str] = ap

    print('\n=== Phase 3b Results ===')
    print(f'{"LR":<10}  {"AP50":>8}')
    print('-' * 22)
    for lr_str, ap in sorted(results.items()):
        print(f'{lr_str:<10}  {(ap or 0):>8.2f}%')

    best_lr = max(results, key=lambda k: results[k] or 0)
    print(f'\nBest SwinB RF LR : {best_lr}  (AP50={results[best_lr]:.2f}%)')

    print('\n=== Comparison: FocalNet_L vs SwinB (RF, 5k iter) ===')
    focal_results = {
        '1e-05': 0.6673, '3e-05': 3.8614, '5e-05': 5.8937, '1e-04': 9.1036
    }
    print(f'{"LR":<10}  {"FocalNet_L":>12}  {"SwinB":>8}  {"Winner":>10}')
    print('-' * 48)
    for lr_str in ['1e-05', '3e-05', '5e-05', '1e-04']:
        f_ap = focal_results.get(lr_str, 0)
        s_ap = results.get(lr_str) or 0
        winner = 'FocalNet_L' if f_ap >= s_ap else 'SwinB'
        print(f'{lr_str:<10}  {f_ap:>12.2f}%  {s_ap:>8.2f}%  {winner:>10}')
