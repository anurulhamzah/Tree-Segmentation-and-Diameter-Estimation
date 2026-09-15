"""
Phases 1-4 sweep — filter validation, backbone, LR, image size (headless for SLURM batch).

Phase 1: Filter sweep (R50 baseline, f0/f300/f500/f1000)
  RF: R50 × {f0,f300,f500,f1000}  |  PL: R50 × {f0,f300,f500,f1000}  → 8 runs
Phase 2: Backbone sweep (best filter dari Phase 1)
  RF: R50, SwinT, SwinB, FocalNet_L  |  PL: R50, SwinT, SwinB, FocalNet_L  → 8 runs
Phase 3: LR sweep (auto-detect best backbone dari Phase 2)
  LRs: 1e-5, 3e-5, 5e-5, 1e-4  → 8 runs
Phase 4: Image size (auto-detect best LR dari Phase 3)
  480px vs 640px  → 4 runs

Total: 28 experiments × ~1.2h ≈ 34h
Idempoten: skip jika model_final.pth sudah ada.
"""

import os, sys, json, re
from pathlib import Path

USER_ROOT     = '/scratch2/pr65/anur0018'
PROJECT_ROOT  = Path(USER_ROOT) / 'tree_classification'
MASKDINO_ROOT = Path(USER_ROOT) / 'MaskDINO' / 'MaskDINO'
OUTPUT_ROOT   = Path(USER_ROOT) / 'maskdino_output'
SANDBOX_OUT   = OUTPUT_ROOT / 'sweep_hyperparams'
SANDBOX_CFG   = PROJECT_ROOT / 'configs' / '_sweep'

PL_ROOT = Path('/home/anur0018/pr65_scratch/anur0018/tree_classification/data/plantations')
RF_ROOT = Path('/home/anur0018/pr65_scratch/anur0018/tree_classification/data/rainforests')

SANDBOX_OUT.mkdir(parents=True, exist_ok=True)
SANDBOX_CFG.mkdir(parents=True, exist_ok=True)

os.environ['PLANTATIONS_ROOT']    = str(PL_ROOT)
os.environ['RAINFORESTS_ROOT']    = str(RF_ROOT)
os.environ['MKL_INTERFACE_LAYER'] = os.environ.get('MKL_INTERFACE_LAYER', '')
os.environ['LD_LIBRARY_PATH'] = (
    '/scratch/pr65/anur0018/conda_envs/maskdino/lib:'
    '/scratch/pr65/anur0018/conda_envs/maskdino/lib/python3.9/site-packages/torch/lib:'
    + os.environ.get('LD_LIBRARY_PATH', ''))
os.environ['PYTHONPATH'] = (
    f"{MASKDINO_ROOT}:{PROJECT_ROOT / 'scripts'}:{os.environ.get('PYTHONPATH', '')}")

for p in [str(MASKDINO_ROOT), str(PROJECT_ROOT / 'scripts')]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(PROJECT_ROOT)

# ── Base configs ──────────────────────────────────────────────────────────
CFG = {
    'R50_pl':        str(PROJECT_ROOT / 'configs' / 'maskdino_R50_pl_f300_tuned.yaml'),
    'R50_rf':        str(PROJECT_ROOT / 'configs' / 'maskdino_R50_full_inst_f1000_50k.yaml'),
    'SwinT':         str(PROJECT_ROOT / 'configs' / 'maskdino_SwinT_plantations.yaml'),
    'SwinB_pl':      str(PROJECT_ROOT / 'configs' / 'maskdino_SwinB_plantations.yaml'),
    'SwinB_rf':      str(PROJECT_ROOT / 'configs' / 'maskdino_SwinB_rf_f1000.yaml'),
    'FocalNet_L_rf': str(PROJECT_ROOT / 'configs' / 'maskdino_FocalNet_L_rf_rle.yaml'),
    'FocalNet_L_pl': str(PROJECT_ROOT / 'configs' / 'maskdino_FocalNet_L_pl_rle.yaml'),
}

_bb_cfg = {
    ('R50',        'rf'): CFG['R50_rf'],        ('R50',        'pl'): CFG['R50_pl'],
    ('SwinT',      'rf'): CFG['SwinT'],          ('SwinT',      'pl'): CFG['SwinT'],
    ('SwinB',      'rf'): CFG['SwinB_rf'],       ('SwinB',      'pl'): CFG['SwinB_pl'],
    ('FocalNet_L', 'rf'): CFG['FocalNet_L_rf'],  ('FocalNet_L', 'pl'): CFG['FocalNet_L_pl'],
}

# Default LR per backbone (FocalNet-L: same scale as SwinL → 3e-5)
_bb_lr = {
    'R50':        {'rf': 1e-4,  'pl': 5e-5},
    'SwinT':      {'rf': 5e-5,  'pl': 5e-5},
    'SwinB':      {'rf': 3e-5,  'pl': 3e-5},
    'FocalNet_L': {'rf': 3e-5,  'pl': 3e-5},
}

def pl_ds(f): return (f'plantations_inst_rle_f{f}_train', f'plantations_inst_rle_f{f}_val')
def rf_ds(f): return (f'rainforests_inst_rle_f{f}_train', f'rainforests_inst_rle_f{f}_val')


def generate_filter_rle(root, ann_subpath, threshold):
    """Generate filtered_rle_f{threshold}/ from instances_*_rle.json. Idempotent."""
    raw_dir = root / ann_subpath
    out_dir = raw_dir / f'filtered_rle_f{threshold}'
    if (out_dir / 'instances_train.json').exists():
        d = json.load(open(out_dir / 'instances_train.json'))
        print(f'  filter rle_f{threshold}: already exists ({len(d["annotations"])} anns) — skip')
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        src_file = raw_dir / f'instances_{split}_rle.json'
        if not src_file.exists():
            continue
        src = json.load(open(src_file))
        filtered_anns = [a for a in src['annotations'] if a['area'] >= threshold]
        valid_ids     = {a['image_id'] for a in filtered_anns}
        filtered_imgs = [i for i in src['images'] if i['id'] in valid_ids]
        out = {'info': {}, 'licenses': [], 'categories': src['categories'],
               'images': filtered_imgs, 'annotations': filtered_anns}
        (out_dir / f'instances_{split}.json').write_text(json.dumps(out))
    d = json.load(open(out_dir / 'instances_train.json'))
    print(f'  filter rle_f{threshold}: {len(d["images"])} imgs, {len(d["annotations"])} anns ✓')


# ── Helpers ───────────────────────────────────────────────────────────────

def make_override_config(name, base_cfg, dataset_train, dataset_val,
                         num_classes, lr, img_size, max_iter=5000, warmup=200):
    steps_str = f'({int(max_iter*0.70)}, {int(max_iter*0.95)})'
    content = f"""_BASE_: {base_cfg}

MODEL:
  SEM_SEG_HEAD:
    NUM_CLASSES: {num_classes}

DATASETS:
  TRAIN: ("{dataset_train}",)
  TEST:  ("{dataset_val}",)

INPUT:
  IMAGE_SIZE: {img_size}
  MASK_FORMAT: "bitmask"

SOLVER:
  BASE_LR: {lr}
  MAX_ITER: {max_iter}
  STEPS: {steps_str}
  WARMUP_ITERS: {warmup}
  CHECKPOINT_PERIOD: {max_iter}

TEST:
  EVAL_PERIOD: {max_iter}
"""
    cfg_path = SANDBOX_CFG / f'{name}.yaml'
    cfg_path.write_text(content)
    return cfg_path


def best_ap(output_dir, metric='segm/AP'):
    path = Path(output_dir) / 'metrics.json'
    if not path.exists():
        return None
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    vals = [r.get(metric, 0) for r in records if metric in r]
    return max(vals) if vals else None


def run_experiment(exp):
    name    = exp['name']
    out_dir = SANDBOX_OUT / name
    ckpt    = out_dir / 'model_final.pth'

    if ckpt.exists():
        ap = best_ap(out_dir)
        print(f'  SKIP {name:<50}  segm/AP={ap:.2f}%' if ap is not None else f'  SKIP {name}')
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = make_override_config(
        name=name, base_cfg=exp['base_cfg'],
        dataset_train=exp['dataset_train'], dataset_val=exp['dataset_val'],
        num_classes=exp['num_classes'], lr=exp['lr'], img_size=exp['img_size'],
        max_iter=exp.get('max_iter', 5000), warmup=exp.get('warmup', 200))

    train_script = PROJECT_ROOT / 'scripts' / (
        'train_plantations.py' if exp['dataset'] == 'pl' else 'train_rainforests.py')
    cmd = (f'{sys.executable} {train_script} --config-file {cfg_path} '
           f'--num-gpus 1 OUTPUT_DIR {out_dir}')

    print(f'  RUN  {name}')
    ret = os.system(cmd)
    ap = best_ap(out_dir)
    if ret != 0:
        print(f'  FAIL {name} (exit={ret})')
    else:
        print(f'  DONE {name}  segm/AP={ap:.2f}%' if ap is not None else f'  DONE {name}')
    return out_dir


def extract_backbone(name):
    for bb in ['FocalNet_L', 'SwinB', 'SwinT', 'R50']:
        if bb in name:
            return bb
    return 'R50'


def extract_lr(name):
    m = re.search(r'_lr([\de.+-]+)$', name)
    return float(m.group(1)) if m else None


def run_phase(experiments):
    results = {}
    for i, exp in enumerate(experiments):
        print(f'\n[{i+1}/{len(experiments)}] {exp["name"]}')
        out = run_experiment(exp)
        results[exp['name']] = best_ap(out)
    return results


def print_phase_summary(label, results):
    rf = {k: v for k, v in results.items() if '_rf_' in k}
    pl = {k: v for k, v in results.items() if '_pl_' in k}
    print(f'\n{label}:')
    for tag, data in [('RF', rf), ('PL', pl)]:
        best_v = max((v for v in data.values() if v), default=0)
        for name, ap in sorted(data.items(), key=lambda x: -(x[1] or 0)):
            marker = ' ★' if ap == best_v and best_v > 0 else ''
            print(f'  {tag}  {name:<55}  segm/AP={ap:.2f}%{marker}' if ap is not None
                  else f'  {tag}  {name:<55}  no AP')


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Filter sweep (R50 baseline, f0/f300/f500/f1000)
# ══════════════════════════════════════════════════════════════════════════

FILTER_SWEEP = [0, 300, 500, 1000]  # f0 = no filter (full dataset)

# Generate missing filtered annotation dirs (idempotent, skips if already exists)
print('\n=== Pre-check: generating missing filter dirs ===')
for f in FILTER_SWEEP:
    if f == 0:
        continue  # f0 = unfiltered, no dir needed
    generate_filter_rle(RF_ROOT, 'annotation_inst',  f)
    generate_filter_rle(PL_ROOT, 'annotations_inst', f)

PHASE1 = []
for f in FILTER_SWEEP:
    PHASE1.append(dict(
        name=f'p1_rf_f{f}_R50',
        dataset='rf', base_cfg=CFG['R50_rf'],
        dataset_train=rf_ds(f)[0], dataset_val=rf_ds(f)[1],
        num_classes=6, lr=_bb_lr['R50']['rf'], img_size=640))
    PHASE1.append(dict(
        name=f'p1_pl_f{f}_R50',
        dataset='pl', base_cfg=CFG['R50_pl'],
        dataset_train=pl_ds(f)[0], dataset_val=pl_ds(f)[1],
        num_classes=7, lr=_bb_lr['R50']['pl'], img_size=640))

print(f'\n=== PHASE 1: Filter Sweep ({len(PHASE1)} experiments) ===')
print(f'Filters: {FILTER_SWEEP} | Backbone: R50 (baseline)\n')
results_p1 = run_phase(PHASE1)

rf_p1 = {k: v for k, v in results_p1.items() if '_rf_' in k}
pl_p1 = {k: v for k, v in results_p1.items() if '_pl_' in k}
best_rf_f_name = max(rf_p1, key=lambda k: rf_p1[k] or 0)
best_pl_f_name = max(pl_p1, key=lambda k: pl_p1[k] or 0)
BEST_FILTER_RF = int(re.search(r'_f(\d+)_', best_rf_f_name).group(1))
BEST_FILTER_PL = int(re.search(r'_f(\d+)_', best_pl_f_name).group(1))

print_phase_summary('PHASE 1 RESULTS', results_p1)
print(f'\n>>> BEST_FILTER_RF = {BEST_FILTER_RF}  (segm/AP={rf_p1[best_rf_f_name]:.2f}%)')
print(f'>>> BEST_FILTER_PL = {BEST_FILTER_PL}  (segm/AP={pl_p1[best_pl_f_name]:.2f}%)')


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Backbone sweep (best filter dari Phase 1)
# ══════════════════════════════════════════════════════════════════════════

PHASE2 = [
    dict(name=f'p2_rf_f{BEST_FILTER_RF}_R50',
         dataset='rf', base_cfg=CFG['R50_rf'],
         dataset_train=rf_ds(BEST_FILTER_RF)[0], dataset_val=rf_ds(BEST_FILTER_RF)[1],
         num_classes=6, lr=_bb_lr['R50']['rf'], img_size=640),
    dict(name=f'p2_rf_f{BEST_FILTER_RF}_SwinT',
         dataset='rf', base_cfg=CFG['SwinT'],
         dataset_train=rf_ds(BEST_FILTER_RF)[0], dataset_val=rf_ds(BEST_FILTER_RF)[1],
         num_classes=6, lr=_bb_lr['SwinT']['rf'], img_size=640),
    dict(name=f'p2_rf_f{BEST_FILTER_RF}_SwinB',
         dataset='rf', base_cfg=CFG['SwinB_rf'],
         dataset_train=rf_ds(BEST_FILTER_RF)[0], dataset_val=rf_ds(BEST_FILTER_RF)[1],
         num_classes=6, lr=_bb_lr['SwinB']['rf'], img_size=640),
    dict(name=f'p2_rf_f{BEST_FILTER_RF}_FocalNet_L',
         dataset='rf', base_cfg=CFG['FocalNet_L_rf'],
         dataset_train=rf_ds(BEST_FILTER_RF)[0], dataset_val=rf_ds(BEST_FILTER_RF)[1],
         num_classes=6, lr=_bb_lr['FocalNet_L']['rf'], img_size=640),
    dict(name=f'p2_pl_f{BEST_FILTER_PL}_R50',
         dataset='pl', base_cfg=CFG['R50_pl'],
         dataset_train=pl_ds(BEST_FILTER_PL)[0], dataset_val=pl_ds(BEST_FILTER_PL)[1],
         num_classes=7, lr=_bb_lr['R50']['pl'], img_size=640),
    dict(name=f'p2_pl_f{BEST_FILTER_PL}_SwinT',
         dataset='pl', base_cfg=CFG['SwinT'],
         dataset_train=pl_ds(BEST_FILTER_PL)[0], dataset_val=pl_ds(BEST_FILTER_PL)[1],
         num_classes=7, lr=_bb_lr['SwinT']['pl'], img_size=640),
    dict(name=f'p2_pl_f{BEST_FILTER_PL}_SwinB',
         dataset='pl', base_cfg=CFG['SwinB_pl'],
         dataset_train=pl_ds(BEST_FILTER_PL)[0], dataset_val=pl_ds(BEST_FILTER_PL)[1],
         num_classes=7, lr=_bb_lr['SwinB']['pl'], img_size=640),
    dict(name=f'p2_pl_f{BEST_FILTER_PL}_FocalNet_L',
         dataset='pl', base_cfg=CFG['FocalNet_L_pl'],
         dataset_train=pl_ds(BEST_FILTER_PL)[0], dataset_val=pl_ds(BEST_FILTER_PL)[1],
         num_classes=7, lr=_bb_lr['FocalNet_L']['pl'], img_size=640),
]

print(f'\n=== PHASE 2: Backbone Sweep ({len(PHASE2)} experiments) ===')
print(f'RF filter: f{BEST_FILTER_RF}, PL filter: f{BEST_FILTER_PL}')
print(f'Backbones: R50 | SwinT | SwinB | FocalNet_L\n')
results_p2 = run_phase(PHASE2)

# Auto-detect best backbone
rf_p2 = {k: v for k, v in results_p2.items() if '_rf_' in k}
pl_p2 = {k: v for k, v in results_p2.items() if '_pl_' in k}
best_rf_bb_name = max(rf_p2, key=lambda k: rf_p2[k] or 0)
best_pl_bb_name = max(pl_p2, key=lambda k: pl_p2[k] or 0)
BEST_BACKBONE_RF = extract_backbone(best_rf_bb_name)
BEST_BACKBONE_PL = extract_backbone(best_pl_bb_name)

print_phase_summary('PHASE 2 RESULTS', results_p2)
print(f'\n>>> BEST_BACKBONE_RF = \'{BEST_BACKBONE_RF}\'  (segm/AP={rf_p2[best_rf_bb_name]:.2f}%)')
print(f'>>> BEST_BACKBONE_PL = \'{BEST_BACKBONE_PL}\'  (segm/AP={pl_p2[best_pl_bb_name]:.2f}%)')


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3 — LR sweep (auto-detect backbone from Phase 2)
# ══════════════════════════════════════════════════════════════════════════

PHASE3 = []
for lr_val in [1e-5, 3e-5, 5e-5, 1e-4]:
    lr_str = f'{lr_val:.0e}'
    PHASE3.append(dict(
        name=f'p3_rf_{BEST_BACKBONE_RF}_lr{lr_str}',
        dataset='rf', base_cfg=_bb_cfg[(BEST_BACKBONE_RF, 'rf')],
        dataset_train=rf_ds(BEST_FILTER_RF)[0], dataset_val=rf_ds(BEST_FILTER_RF)[1],
        num_classes=6, lr=lr_val, img_size=640))
    PHASE3.append(dict(
        name=f'p3_pl_{BEST_BACKBONE_PL}_lr{lr_str}',
        dataset='pl', base_cfg=_bb_cfg[(BEST_BACKBONE_PL, 'pl')],
        dataset_train=pl_ds(BEST_FILTER_PL)[0], dataset_val=pl_ds(BEST_FILTER_PL)[1],
        num_classes=7, lr=lr_val, img_size=640))

print(f'\n\n=== PHASE 3: LR Sweep ({len(PHASE3)} experiments) ===')
print(f'RF: {BEST_BACKBONE_RF} f{BEST_FILTER_RF} | PL: {BEST_BACKBONE_PL} f{BEST_FILTER_PL}')
print(f'LRs: 1e-5, 3e-5, 5e-5, 1e-4\n')
results_p3 = run_phase(PHASE3)

# Auto-detect best LR
rf_p3 = {k: v for k, v in results_p3.items() if '_rf_' in k}
pl_p3 = {k: v for k, v in results_p3.items() if '_pl_' in k}
best_rf_lr_name = max(rf_p3, key=lambda k: rf_p3[k] or 0)
best_pl_lr_name = max(pl_p3, key=lambda k: pl_p3[k] or 0)
BEST_LR_RF = extract_lr(best_rf_lr_name)
BEST_LR_PL = extract_lr(best_pl_lr_name)

print_phase_summary('PHASE 3 RESULTS', results_p3)
print(f'\n>>> BEST_LR_RF = {BEST_LR_RF}  (segm/AP={rf_p3[best_rf_lr_name]:.2f}%)')
print(f'>>> BEST_LR_PL = {BEST_LR_PL}  (segm/AP={pl_p3[best_pl_lr_name]:.2f}%)')


# ══════════════════════════════════════════════════════════════════════════
# PHASE 4 — Image size (auto-detect LR from Phase 3)
# ══════════════════════════════════════════════════════════════════════════

PHASE4 = [
    dict(name=f'p4_rf_{BEST_BACKBONE_RF}_480px',
         dataset='rf', base_cfg=_bb_cfg[(BEST_BACKBONE_RF, 'rf')],
         dataset_train=rf_ds(BEST_FILTER_RF)[0], dataset_val=rf_ds(BEST_FILTER_RF)[1],
         num_classes=6, lr=BEST_LR_RF, img_size=480),
    dict(name=f'p4_rf_{BEST_BACKBONE_RF}_640px',
         dataset='rf', base_cfg=_bb_cfg[(BEST_BACKBONE_RF, 'rf')],
         dataset_train=rf_ds(BEST_FILTER_RF)[0], dataset_val=rf_ds(BEST_FILTER_RF)[1],
         num_classes=6, lr=BEST_LR_RF, img_size=640),
    dict(name=f'p4_pl_{BEST_BACKBONE_PL}_480px',
         dataset='pl', base_cfg=_bb_cfg[(BEST_BACKBONE_PL, 'pl')],
         dataset_train=pl_ds(BEST_FILTER_PL)[0], dataset_val=pl_ds(BEST_FILTER_PL)[1],
         num_classes=7, lr=BEST_LR_PL, img_size=480),
    dict(name=f'p4_pl_{BEST_BACKBONE_PL}_640px',
         dataset='pl', base_cfg=_bb_cfg[(BEST_BACKBONE_PL, 'pl')],
         dataset_train=pl_ds(BEST_FILTER_PL)[0], dataset_val=pl_ds(BEST_FILTER_PL)[1],
         num_classes=7, lr=BEST_LR_PL, img_size=640),
]

print(f'\n\n=== PHASE 4: Image Size ({len(PHASE4)} experiments) ===')
print(f'RF: {BEST_BACKBONE_RF} f{BEST_FILTER_RF} lr={BEST_LR_RF} | PL: {BEST_BACKBONE_PL} f{BEST_FILTER_PL} lr={BEST_LR_PL}')
print(f'Sizes: 480px vs 640px\n')
results_p4 = run_phase(PHASE4)

print_phase_summary('PHASE 4 RESULTS', results_p4)


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════

all_results = {**results_p1, **results_p2, **results_p3, **results_p4}
rf_all = {k: v for k, v in all_results.items() if '_rf_' in k}
pl_all = {k: v for k, v in all_results.items() if '_pl_' in k}

print('\n' + '='*65)
print('FINAL SUMMARY — PHASES 1-4')
print('='*65)

for label, data in [('RAINFORESTS', rf_all), ('PLANTATIONS', pl_all)]:
    best_v = max((v for v in data.values() if v), default=0)
    print(f'\n{label}:')
    for name, ap in sorted(data.items(), key=lambda x: -(x[1] or 0)):
        marker = ' ★' if ap == best_v and best_v > 0 else ''
        print(f'  {(ap or 0):>6.2f}%  {name}{marker}')

best_rf = max(rf_all.items(), key=lambda x: x[1] or 0)
best_pl = max(pl_all.items(), key=lambda x: x[1] or 0)

print(f'\n>>> Recommended config for RF full training:')
print(f'    Filter   : f{BEST_FILTER_RF}')
print(f'    Backbone : {BEST_BACKBONE_RF}')
print(f'    LR       : {BEST_LR_RF}')
m = re.search(r'_(\d+)px$', best_rf[0])
print(f'    ImgSize  : {m.group(1) if m else 640}px')
print(f'    Sandbox AP : {best_rf[1]:.2f}%')

print(f'\n>>> Recommended config for PL full training:')
print(f'    Filter   : f{BEST_FILTER_PL}')
print(f'    Backbone : {BEST_BACKBONE_PL}')
print(f'    LR       : {BEST_LR_PL}')
m = re.search(r'_(\d+)px$', best_pl[0])
print(f'    ImgSize  : {m.group(1) if m else 640}px')
print(f'    Sandbox AP : {best_pl[1]:.2f}%')

print('\nPhases 1-4 selesai.')
