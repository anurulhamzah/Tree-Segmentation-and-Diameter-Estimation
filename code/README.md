# Individual Tree Instance Segmentation and Diameter Estimation from Under-Canopy RGB-D Imagery

Code accompanying the thesis *"Individual Tree Instance Segmentation and Diameter Estimation from
Under-Canopy RGB-D Imagery for Indonesian Agroforestry and Rainforest Species"* (Akhmad Fauzi
Nurulhamzah, Monash University Indonesia). A single MaskDINO network with a four-channel RGB-D
input jointly segments trees, identifies species, and regresses diameter at breast height (DBH)
from a trunk-region head that reads backbone features directly.

This repository contains the training, evaluation, and figure-generation code. It does **not**
contain the SPREAD dataset (third-party, see below) or trained model checkpoints (see
*Checkpoints* below).

## Repository structure

```
configs/            MaskDINO configuration files for every reported run (ablations + final model)
scripts/            Training, evaluation, and figure-generation scripts (see "Reproducing results")
maskdino_patches/   Files that differ from the upstream MaskDINO implementation, with the same
                     relative paths they occupy inside the MaskDINO source tree (see "Setup")
env/                Conda environment specification
```

## Setup

1. Clone [MaskDINO](https://github.com/IDEA-Research/MaskDINO) (commit pinned in
   `maskdino_patches/COMMIT`) and follow its own installation instructions (Detectron2 + the
   deformable-attention CUDA extension).
2. Overlay the contents of `maskdino_patches/` onto the MaskDINO source tree, preserving the
   relative paths (e.g. `maskdino_patches/maskdino/modeling/backbone/focal.py` replaces
   `MaskDINO/maskdino/modeling/backbone/focal.py`). These are the only files that depart from the
   reference implementation: the four-channel input stem (`focal.py`, `swin.py`), the depth-aware
   dataset mapper (`coco_instance_dbh_dataset_mapper.py`, new), the trunk-region diameter head's
   integration into the training loop (`maskdino.py`, `criterion.py`, `config.py`,
   `train_net.py`), and the corresponding decoder/mapper hooks.
3. Create the environment: `conda env create -f env/environment.yml`, or install
   `env/requirements.txt` into an existing Python 3.9 environment. PyTorch 2.1.0 / CUDA 12.1 is
   what the reported runs used; the deformable-attention extension must be compiled against the
   same CUDA version.
4. Set `COMBINED_ROOT` (and `PLANTATIONS_ROOT`, `RAINFORESTS_ROOT` where a script expects the
   domain-split subsets) to point at your local copy of the dataset described below.

Scripts assume a project layout of `<root>/tree_classification/{data,scripts,configs,reports}` and
`<root>/MaskDINO/MaskDINO` for the backbone, mirroring the structure this code was developed in;
adjust the path constants at the top of a script if your layout differs.

## Data

Training and evaluation use SPREAD (Feng, She, and Keshav, 2025, *Ecological Informatics* 87:103085,
[doi:10.1016/j.ecoinf.2025.103085](https://doi.org/10.1016/j.ecoinf.2025.103085)), a synthetic
below-canopy dataset rendered in Unreal Engine 5. SPREAD is not redistributed here; obtain it from
the original authors and register it with `scripts/register_combined.py` (or
`register_plantations.py` / `register_rainforests.py` for the domain-split subsets).

## Checkpoints

Trained model checkpoints are not included in this repository because of their size. They are
available from the corresponding author upon reasonable request.

## Reproducing results

The scripts are organised by what they measure rather than by table number; the ones a reader
following the thesis is most likely to need are:

- `scripts/eval_joint_trunkroi_dbh.py` — GT-conditioned diameter evaluation (isolates the
  regression head from detection noise); used for the backbone comparison and the loss-weighting
  ablation.
- `scripts/eval_end_to_end_dbh.py` — end-to-end diameter evaluation, on the model's own detections
  and predicted masks, matched to ground truth at IoU $\geq 0.5$; this is the headline diameter
  accuracy reported in the thesis.
- `scripts/dump_dbh_per_instance.py` — per-instance diameter predictions with geometry, the
  canonical source for every diameter table.
- `scripts/recall_per_jarak.py` — detection recall, species accuracy, and mask IoU stratified by
  camera distance.
- `scripts/occlusion_robustness.py` — detection recall stratified by occlusion ratio, and the
  convex-hull occlusion definition.
- `paper/tesis_final/scripts_figur.py` (in the thesis directory, not this repository) — regenerates
  every figure from the numbers and cached predictions the scripts above produce.

Each ablation table in the thesis (depth channel, backbone selection) is trained with the
matching config in `configs/`, at the fixed budget stated in the config filename, and evaluated
with standard Detectron2 COCO instance evaluation.

## Citation

```bibtex
@article{nurulhamzah2026treesegdbh,
  title   = {Individual Tree Instance Segmentation and Diameter Estimation from
             Under-Canopy {RGB-D} Imagery for {I}ndonesian Agroforestry and Rainforest Species},
  author  = {Nurulhamzah, Akhmad Fauzi and Saputra, Muhamad Risqi Utama},
  journal = {Ecological Informatics},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

See `LICENSE`.
