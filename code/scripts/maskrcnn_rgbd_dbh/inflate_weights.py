"""One-off: inflate stem conv1 R50 dari 3→4 channel agar pretrained RGB tetap dipakai.

detectron2 otomatis membangun stem 4-channel (karena len(PIXEL_MEAN)==4), tapi weights COCO
pretrained punya conv1 (64,3,7,7) → mismatch → conv1 ke-reinit acak (kehilangan pretrain stem).
Script ini menyalin 3 channel RGB + mengisi channel depth ke-4 = rata-rata RGB, lalu menyimpan
.pkl baru untuk dipakai sebagai MODEL.WEIGHTS.

Jalankan sekali:
    python scripts/maskrcnn_rgbd_dbh/inflate_weights.py \
        --out /scratch2/pr65/anur0018/pretrained_weights/mask_rcnn_R_50_FPN_3x_rgbd4ch.pkl
"""
import argparse
import pickle

import numpy as np

from detectron2 import model_zoo
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer  # noqa: F401
from fvcore.common.file_io import PathManager  # type: ignore

CONV1_KEY = "backbone.bottom_up.stem.conv1.weight"
ZOO_CFG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", choices=["mean", "zero"], default="mean",
                    help="cara isi channel ke-4 (depth): rata-rata RGB atau nol")
    args = ap.parse_args()

    url = model_zoo.get_checkpoint_url(ZOO_CFG)
    local = PathManager.get_local_path(url)
    with open(local, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    model = obj["model"]
    w = np.asarray(model[CONV1_KEY])          # (64, 3, 7, 7)
    assert w.shape[1] == 3, f"conv1 sudah bukan 3-channel: {w.shape}"
    if args.init == "mean":
        extra = w.mean(axis=1, keepdims=True)
    else:
        extra = np.zeros((w.shape[0], 1, w.shape[2], w.shape[3]), dtype=w.dtype)
    model[CONV1_KEY] = np.concatenate([w, extra], axis=1)   # (64, 4, 7, 7)
    print(f"conv1: {w.shape} → {model[CONV1_KEY].shape}  (init ch4 = {args.init})")

    obj["model"] = model
    obj["__author__"] = "rgbd-inflate"
    with open(args.out, "wb") as f:
        pickle.dump(obj, f)
    print("saved →", args.out)


if __name__ == "__main__":
    main()
