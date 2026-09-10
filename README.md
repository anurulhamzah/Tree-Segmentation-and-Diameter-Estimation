# RGB-D Multitask Tree Instance Segmentation and DBH Estimation

This thesis investigates whether a single RGB-D deep learning model can simultaneously identify individual trees, classify their species, and estimate their diameter at breast height (DBH) from below-canopy imagery.

The proposed approach extends MaskDINO with depth information and a multitask DBH estimation branch. Experiments evaluate the contribution of depth across multiple backbones, the cost of multitask learning on segmentation performance, and the factors limiting performance under distance and occlusion.

The best model achieved **62.29% AP50** for species-aware tree instance segmentation and **7.62 cm DBH RMSE**, with no measurable segmentation cost from adding the DBH task. Adding depth improved segmentation by **9.7–16.9 AP50 points** across the tested backbones.

The results show that depth provides important geometric information for both segmentation and diameter estimation. The main remaining challenge is tree detection, particularly at longer distances and under occlusion, while species recognition remains highly accurate for successfully detected trees.
