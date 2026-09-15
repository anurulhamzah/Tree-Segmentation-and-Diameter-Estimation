"""
Ukur FLOPs (fvcore) dan FPS (wall-clock timing) per arsitektur backbone.
FLOPs = properti arsitektur+resolusi, TIDAK tergantung weights — cukup 1 checkpoint
representatif per arsitektur (7 arsitektur: R50, SwinT, SwinB, SwinL, FocalNet-T/B/L).

Usage:
    python measure_flops_fps_v4.py
"""
import os
import sys
import time
import json
from pathlib import Path

import torch

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
OUTPUT_ROOT   = PROJECT_ROOT.parent / "maskdino_output"
REPORT_DIR    = PROJECT_ROOT / "reports"

sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config

# Satu representative checkpoint per arsitektur (checkpoint apapun valid utk FLOPs count)
ARCHS = {
    "ResNet-50":  "R50_combined_rle_f1000_repaired_v4_stratified_75k",
    "Swin-T":     "SwinT_combined_rle_f1000_repaired_v4_stratified_75k",
    "Swin-B":     "SwinB_combined_rle_f1000_repaired_v4_stratified_75k",
    "Swin-L":     "SwinL_combined_rle_f1000_repaired_v4_stratified_75k",
    "FocalNet-T": "FocalNet_T_combined_rle_f1000_repaired_v4_stratified_75k_lr1e4",
    "FocalNet-B": "FocalNet_B_combined_rle_f1000_repaired_v4_stratified_75k_lr1e4",
    "FocalNet-L": "FocalNet_L_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k",
}

INPUT_SIZE = (640, 640)  # sesuai IMAGE_SIZE training


def load_model(out_dir):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(OUTPUT_ROOT / out_dir / "config.yaml"))
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    model = build_model(cfg)
    ckpt = OUTPUT_ROOT / out_dir / "model_final.pth"
    DetectionCheckpointer(model).load(str(ckpt))
    model.eval()
    n_ch = cfg.MODEL.FOCAL.get("in_chans", 3) if "FOCAL" in cfg.MODEL else 3
    return model, cfg, n_ch


def make_dummy_input(n_ch, device):
    h, w = INPUT_SIZE
    img = torch.rand(n_ch, h, w, device=device) * 255.0
    return [{"image": img, "height": h, "width": w}]


def measure_flops(model, dummy_input):
    from detectron2.utils.analysis import FlopCountAnalysis  # wrapper: handle output Instances via TracingAdapter
    try:
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy_input)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            total = flops.total()
        return total / 1e9  # GFLOPs
    except Exception as e:
        return f"ERROR: {e}"


def measure_fps(model, dummy_input, n_warmup=5, n_iter=20):
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    return n_iter / elapsed  # FPS


def main():
    results = {}
    for arch_name, out_dir in ARCHS.items():
        print(f"=== {arch_name} ({out_dir}) ===", flush=True)
        try:
            model, cfg, n_ch = load_model(out_dir)
            device = next(model.parameters()).device
            dummy = make_dummy_input(n_ch, device)

            gflops = measure_flops(model, dummy)
            fps = measure_fps(model, dummy)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6

            print(f"  Params: {n_params:.1f}M  GFLOPs: {gflops}  FPS: {fps:.2f}")
            results[arch_name] = {"params_m": round(n_params, 1), "gflops": gflops,
                                   "fps": round(fps, 2), "input_size": INPUT_SIZE, "n_channels": n_ch}
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}")
            results[arch_name] = {"error": str(e)}

    out_path = REPORT_DIR / "flops_fps_v4.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
