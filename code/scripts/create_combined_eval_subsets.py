"""
Buat dua subset annotation JSON dari combined val untuk eval terpisah:

  combined_val_pl.json  — hanya gambar plantation (cat 1-7),
                          semua 13 kategori tetap ada di JSON
                          (agar combined model bisa eval tanpa KeyError cat-id mapping)

  combined_val_rf.json  — hanya gambar rainforest (cat 8-13),
                          semua 13 kategori tetap ada di JSON

Output ke:
  data/combined/annotation_inst/filtered_rle_f1000/instances_val_pl.json
  data/combined/annotation_inst/filtered_rle_f1000/instances_val_rf.json

Jalankan sekali sebelum submit SLURM eval:
  python scripts/create_combined_eval_subsets.py
"""
import json
from pathlib import Path

SRC = (
    Path(__file__).resolve().parents[1]
    / "data/combined/annotation_inst/filtered_rle_f1000/instances_val.json"
)

with open(SRC) as f:
    co = json.load(f)

pl_img_ids = {i["id"] for i in co["images"] if "plantation" in i["file_name"].lower()}
rf_img_ids = {i["id"] for i in co["images"] if "rainforest" in i["file_name"].lower()}

assert len(pl_img_ids) + len(rf_img_ids) == len(co["images"]), \
    "Ada gambar yang tidak terklasifikasi sebagai PL atau RF — periksa file_name"

print(f"PL images: {len(pl_img_ids)}, RF images: {len(rf_img_ids)}")

def make_subset(all_data: dict, keep_img_ids: set) -> dict:
    imgs = [i for i in all_data["images"] if i["id"] in keep_img_ids]
    anns = [a for a in all_data["annotations"] if a["image_id"] in keep_img_ids]
    return {
        "info":        all_data.get("info", {}),
        "licenses":    all_data.get("licenses", []),
        "categories":  all_data["categories"],   # semua 13 kategori
        "images":      imgs,
        "annotations": anns,
    }

out_dir = SRC.parent
for tag, ids in [("pl", pl_img_ids), ("rf", rf_img_ids)]:
    subset = make_subset(co, ids)
    out = out_dir / f"instances_val_{tag}.json"
    with open(out, "w") as f:
        json.dump(subset, f)
    print(f"Wrote {out}  ({len(subset['images'])} images, {len(subset['annotations'])} annotations)")
