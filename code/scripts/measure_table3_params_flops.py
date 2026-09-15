"""
measure_table3_params_flops.py — Params (M) dan GFLOPs untuk 6 backbone MaskDINO + referensi
Mask R-CNN, RGB-D from-scratch 75k, dipakai di tab:backbone tesis.

FLOPs adalah properti arsitektur+resolusi, TIDAK tergantung weights terlatih -- checkpoint
apapun dari arsitektur yang sama valid untuk dihitung (lihat measure_flops_fps_v4.py,
preseden proyek ini). Input RGB-D (4ch), 640x640, sesuai training semua run yang dibandingkan.

Jalankan:
    python scripts/measure_table3_params_flops.py
"""
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASKDINO_ROOT = PROJECT_ROOT.parent / "MaskDINO" / "MaskDINO"
OUTPUT_ROOT = PROJECT_ROOT.parent / "maskdino_output"
REPORT_DIR = PROJECT_ROOT / "reports"

sys.path.insert(0, str(MASKDINO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "maskrcnn_rgbd_dbh"))

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config

INPUT_SIZE = (640, 640)

# arch_name -> (kind, output_dir, checkpoint_filename)
ARCHS = {
    "FocalNet-L": ("maskdino", "FocalNet_L_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "model_final.pth"),
    "FocalNet-B": ("maskdino", "FocalNet_B_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "model_final.pth"),
    "Swin-T": ("maskdino", "SwinT_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "model_final.pth"),
    "Swin-L": ("maskdino", "SwinL_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "model_final.pth"),
    "Swin-B": ("maskdino", "SwinB_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "model_final.pth"),
    "ResNet-50 (MaskDINO)": ("maskdino", "R50_combined_rle_f1000_rgbd_scratch_repaired_v4_stratified_75k", "model_final.pth"),
    "Mask R-CNN, ResNet-50": ("maskrcnn", "maskrcnn_R50_combined_rgbd_scratch_v4_stratified_75k", "model_final.pth"),
}


def load_maskdino_model(out_dir, ckpt_name):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(OUTPUT_ROOT / out_dir / "config.yaml"))
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    model = build_model(cfg)
    ckpt = OUTPUT_ROOT / out_dir / ckpt_name
    DetectionCheckpointer(model).load(str(ckpt))
    model.eval()
    # NB: cfg.MODEL.FOCAL always exists (add_maskdino_config registers both SWIN and FOCAL
    # sub-namespaces regardless of which backbone is actually selected via BACKBONE.NAME), so
    # "FOCAL" in cfg.MODEL is true even for Swin/R50 runs and its in_chans default (3) is unused
    # -- PIXEL_MEAN length is the one field guaranteed to match the actual input channel count.
    n_ch = len(cfg.MODEL.PIXEL_MEAN)
    return model, n_ch


def load_maskrcnn_model(out_dir, ckpt_name):
    from maskrcnn_rgbd_dbh import add_rgbd_dbh_config
    cfg = get_cfg()
    add_rgbd_dbh_config(cfg)
    cfg.merge_from_file(str(OUTPUT_ROOT / out_dir / "config.yaml"))
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    model = build_model(cfg)
    ckpt = OUTPUT_ROOT / out_dir / ckpt_name
    DetectionCheckpointer(model).load(str(ckpt))
    model.eval()
    n_ch = len(cfg.MODEL.PIXEL_MEAN)
    return model, n_ch


def make_dummy_input(n_ch, device):
    h, w = INPUT_SIZE
    img = torch.rand(n_ch, h, w, device=device) * 255.0
    return [{"image": img, "height": h, "width": w}]


def measure_flops(model, dummy_input):
    from detectron2.utils.analysis import FlopCountAnalysis
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy_input)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
    return total / 1e9  # GFLOPs


def measure_fps(model, dummy_input, n_warmup=5, n_iter=20):
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        import time
        t0 = time.time()
        for _ in range(n_iter):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    return n_iter / elapsed  # FPS, batch size 1


def main():
    results = {}
    for arch_name, (kind, out_dir, ckpt_name) in ARCHS.items():
        print(f"=== {arch_name} ({out_dir}) ===", flush=True)
        try:
            if kind == "maskdino":
                model, n_ch = load_maskdino_model(out_dir, ckpt_name)
            else:
                model, n_ch = load_maskrcnn_model(out_dir, ckpt_name)
            device = next(model.parameters()).device
            dummy = make_dummy_input(n_ch, device)

            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            gflops = measure_flops(model, dummy)
            fps = measure_fps(model, dummy)

            print(f"  Params: {n_params:.2f}M  GFLOPs: {gflops:.2f}  FPS: {fps:.2f}  n_channels: {n_ch}")
            results[arch_name] = {"params_m": round(n_params, 2), "gflops": round(gflops, 2),
                                   "fps": round(fps, 2),
                                   "input_size": INPUT_SIZE, "n_channels": n_ch}
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}")
            results[arch_name] = {"error": str(e)}

    out_path = REPORT_DIR / "table3_params_flops.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
