# RGB-D Multitask Tree Instance Segmentation and DBH Estimation

Click https://claude.ai/code/artifact/b9360df3-0b0f-42de-bc88-667a56f85406 for more info

This thesis investigates whether a single RGB-D deep learning model can simultaneously identify individual trees, classify their species, and estimate their diameter at breast height (DBH) from below-canopy imagery.

The proposed approach extends MaskDINO with depth information and a multitask DBH estimation branch. Experiments evaluate the contribution of depth across multiple backbones, the cost of multitask learning on segmentation performance, and the factors limiting performance under distance and occlusion.

The best model achieved **62.29% AP50** for species-aware tree instance segmentation and **7.62 cm DBH RMSE**, with no measurable segmentation cost from adding the DBH task. Adding depth improved segmentation by **9.7–16.9 AP50 points** across the tested backbones.

The results show that depth provides important geometric information for both segmentation and diameter estimation. The main remaining challenge is tree detection, particularly at longer distances and under occlusion, while species recognition remains highly accurate for successfully detected trees.

## Contents

- `code/` — training, evaluation, and figure-generation scripts, MaskDINO patches, and configs
  for every reported run. See `code/README.md` for setup instructions.
- `Multitask Learning for Forestry.pdf` — extended reading summary of the full research paper.
- `Katalog_Kamera_Handphone_Depth_Sensing.pdf`, `Species_and_Depth_Devices_Tables.pdf` —
  supporting reference material.

The SPREAD dataset used to train and evaluate the models is not included here (third-party); see
[github.com/FrankFeng-23/SPREAD](https://github.com/FrankFeng-23/SPREAD). Trained model
checkpoints are available from the corresponding author upon reasonable request.
