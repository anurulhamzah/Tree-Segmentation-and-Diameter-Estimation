#!/usr/bin/env python3
"""Figur 4 naskah: empat panel berdampingan pada satu scene test.

  (a) image input      RGB apa adanya
  (b) ground truth     mask instance beranotasi
  (c) predicted boxes  pred_boxes + pred_classes
  (d) predicted masks  pred_masks + spesies + DBH

Sebelumnya figur ini hanya memperlihatkan mask prediksi, sehingga box yang juga diprediksi
model tidak pernah terlihat, dan tidak ada pembanding ground truth di sebelahnya.

Memakai pipeline dan checkpoint yang identik dengan gen_qualitative_predictions.py.
Dijalankan di CPU, sekitar tiga menit untuk satu gambar.

Jalankan:  python scripts/gen_fig4_panels.py
"""
import json, os, pickle, sys, importlib.util
import numpy as np, torch

PROJ = "/scratch2/pr65/anur0018/tree_classification"
sys.path.insert(0, "/scratch2/pr65/anur0018/MaskDINO/MaskDINO")
sys.path.insert(0, f"{PROJ}/scripts")

spec = importlib.util.spec_from_file_location("gq", f"{PROJ}/scripts/gen_qualitative_predictions.py")
gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from pycocotools import mask as maskutil
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import build_transform_gen
from eval_joint_trunkroi_dbh import build_model_with_head
from register_combined import register_all_combined

STEM = "Tree8447_1720691246"
RGB  = f"{PROJ}/data/rainforests/rgb_resized/{STEM}.png"
ANN  = (f"{PROJ}/data/combined/annotation_inst/"
        "filtered_rle_f1000_repaired_v4_stratified/instances_test.json")
OUT  = f"{PROJ}/paper/tesis_final/gambar/fig4_panels.pdf"
# Inferensi CPU makan ~3 menit; hasilnya di-cache supaya penataan panel bisa diulang cepat.
CACHE = f"{PROJ}/reports/fig4_panels_cache.pkl"
WARNA = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22",
         "#e84393", "#00b894", "#0984e3", "#fdcb6e", "#6c5ce7", "#d35400"]


def ground_truth():
    """Mask GT untuk gambar target, di-decode dari RLE."""
    d = json.load(open(ANN))
    im = next(i for i in d["images"] if STEM in i["file_name"])
    kat = {c["id"]: c["name"] for c in d["categories"]}
    keluar = []
    for a in d["annotations"]:
        if a["image_id"] != im["id"]:
            continue
        keluar.append(dict(mask=maskutil.decode(a["segmentation"]).astype(bool),
                           species=kat[a["category_id"]], bbox=a["bbox"]))
    return keluar



def rapikan_label(fig, ax, label, W, H):
    """Tata label ke dalam jalur horizontal supaya dijamin tidak bertumpuk.

    Pendekatan geser-turun-lalu-menyamping sebelumnya menyerah setelah 40 percobaan dan
    meninggalkan label yang masih bertindih di baris atas. Di sini label diurutkan menurut
    posisi x, lalu tiap label ditaruh di jalur terendah yang sisi kanannya masih lowong.
    Karena jalur hanya menampung label yang terpisah secara horizontal, tumpang tindih
    tidak mungkin terjadi menurut konstruksi.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def kotak(t):
        bb = t.get_window_extent(rend)
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)

    ukur = []
    for t in label:
        x0, x1, y0, y1 = kotak(t)
        ukur.append((t, x1 - x0, y1 - y0))
    tinggi = max(h for _, _, h in ukur) if ukur else 0
    ukur.sort(key=lambda u: u[0].get_position()[0])

    jalur = []                       # tepi kanan terpakai per jalur
    for t, lebar, _ in ukur:
        xa = t.get_position()[0]
        xa = min(max(xa, lebar / 2 + 2), W - lebar / 2 - 2)
        kiri = xa - lebar / 2
        for j in range(len(jalur) + 1):
            if j == len(jalur):
                jalur.append(-1e9)
            if kiri > jalur[j] + 1.5:
                jalur[j] = xa + lebar / 2
                t.set_position((xa, 3 + j * (tinggi + 2)))
                break


def main():
    if os.path.exists(CACHE):
        image, hasil, boxes, kelas = pickle.load(open(CACHE, "rb"))
        H, W = image.shape[:2]
        print("memakai cache:", CACHE)
        return gambar(image, hasil, boxes, kelas, H, W)

    register_all_combined(os.environ.get("COMBINED_ROOT", f"{PROJ}/data/combined"))
    model, cfg = build_model_with_head(gq.OUT_DIR, "cpu", init_ckpt="model_final.pth")
    model.eval()
    tfm = build_transform_gen(cfg, is_train=False)

    image, hasil = gq.predict_image(model, tfm, "cpu", model.size_divisibility, RGB)

    # box dan class dari forward yang sama, tidak dikeluarkan predict_image
    H, W = image.shape[:2]
    from maskdino.data.dataset_mappers.coco_instance_dbh_dataset_mapper import _read_pfm, _normalize_depth
    depth = _normalize_depth(_read_pfm(RGB.replace("/rgb_resized/", "/depth_pfm/")
                                          .replace(".png", ".pfm")).astype(np.float32))
    img_t, tfms = T.apply_transform_gens(tfm, image.copy())
    dep_t = tfms.apply_image(depth[:, :, None])[:, :, 0]
    with torch.no_grad():
        out = model([{"image": torch.as_tensor(np.ascontiguousarray(img_t.transpose(2, 0, 1))),
                      "depth": torch.as_tensor(np.ascontiguousarray(dep_t)),
                      "height": H, "width": W}])[0]["instances"].to("cpu")
    simpan = out.scores >= gq.SCORE_THR
    boxes = out.pred_boxes[simpan].tensor.numpy()
    kelas = out.pred_classes[simpan].numpy()
    skor = out.scores[simpan].numpy()
    pickle.dump((image, hasil, boxes, kelas), open(CACHE, "wb"))
    return gambar(image, hasil, boxes, kelas, H, W)


def gambar(image, hasil, boxes, kelas, H, W):
    gt = ground_truth()
    print(f"GT {len(gt)} instance | prediksi {len(boxes)} instance di atas {gq.SCORE_THR}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.size": 8.5})
    fig, ax2 = plt.subplots(2, 2, figsize=(7.4, 4.7))
    axes = ax2.ravel()

    dasar = image.astype(float) / 255.0

    def bingkai(ax, judul):
        ax.imshow(dasar); ax.axis("off")
        ax.set_title(judul, fontsize=8.8, pad=4)

    bingkai(axes[0], "(a) input RGB-D scene")

    bingkai(axes[1], f"(b) ground truth, {len(gt)} instances")
    lap_gt = np.zeros((H, W, 4))
    for i, g in enumerate(gt):
        c = WARNA[i % len(WARNA)]
        lap_gt[g["mask"]] = tuple(int(c[j:j+2], 16) / 255 for j in (1, 3, 5)) + (0.45,)
    axes[1].imshow(lap_gt)

    bingkai(axes[2], f"(c) predicted boxes, {len(boxes)}")
    for i, b in enumerate(boxes):
        c = WARNA[i % len(WARNA)]
        axes[2].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                    fill=False, edgecolor=c, linewidth=1.2))

    bingkai(axes[3], "(d) predicted masks, species, DBH")
    lap = np.zeros((H, W, 4))
    label = []
    # boxes dan hasil berbagi urutan (diverifikasi lewat IoU box lawan bbox mask, 0,50-0,92),
    # jadi warna diikat ke indeks ASLI supaya panel (c) dan (d) sepadan. Penggambaran tetap
    # menurut skor, supaya label berskor tinggi mendapat posisi lebih dulu.
    for i in sorted(range(len(hasil)), key=lambda k: -hasil[k]["score"]):
        r = hasil[i]
        c = WARNA[i % len(WARNA)]
        lap[r["mask"]] = tuple(int(c[j:j+2], 16) / 255 for j in (1, 3, 5)) + (0.45,)
        ys, xs = np.where(r["mask"])
        if len(xs) == 0:
            continue
        dbh = f"{r['dbh_mm']/10:.0f}cm" if r["dbh_mm"] is not None else "n/a"
        t = axes[3].text(xs.mean(), max(ys.min() + 3, 3), f"{r['species']}\n{dbh}",
                         fontsize=6.0, color="white", ha="center", va="top", clip_on=True,
                         bbox=dict(boxstyle="round,pad=0.12", facecolor=c, alpha=.85,
                                   edgecolor="none"))
        label.append(t)
    axes[3].imshow(lap)


    fig.tight_layout(w_pad=0.8, h_pad=1.2)
    rapikan_label(fig, axes[3], label, W, H)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print("tersimpan:", OUT)


if __name__ == "__main__":
    main()
