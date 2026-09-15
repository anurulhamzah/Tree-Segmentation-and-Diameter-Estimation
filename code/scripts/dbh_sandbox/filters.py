"""
filters.py — Filter tabel index DBH sandbox (dipakai oleh precompute_dbh_cache.py
dan dbh_sandbox_head.py). Semua kombinasi axis di sweep dinyatakan sebagai filter
boolean di atas SATU tabel index yang dibangun sekali (Tier 0) dengan threshold
paling longgar — bukan regenerasi data per kombinasi.

Lihat rencana: /home/anur0018/.claude/plans/melodic-mapping-journal.md
"""

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── Nama kelas COCO (1-based category_id -> nama), sinkron dgn register_combined.py
CLASS_NAMES = [
    "Apple", "Lemon", "Loquat", "Mango", "Orange", "Persimmon", "Pomegranate",
    "AliiFig", "BangaloPalm", "Fern", "LeechVine", "RubberFig", "Umbrella",
]
CAT_ID_TO_NAME = {i + 1: name for i, name in enumerate(CLASS_NAMES)}
NAME_TO_CAT_ID = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}

# ── Universe spesies default: SEMUA kecuali blank-DBH > 50%
# Lemon (cat_id=2): 86.9% blank | LeechVine (cat_id=11): 66.4% blank (liana)
# Dihitung langsung dari data combined_inst_rle_f1000_repaired_v4_stratified (lihat plan).
EXCLUDED_BLANK_SPECIES = {2, 11}
DEFAULT_SPECIES_UNIVERSE = tuple(
    cid for cid in CAT_ID_TO_NAME if cid not in EXCLUDED_BLANK_SPECIES
)  # 11 spesies: 1,3,4,5,6,7,8,9,10,12,13

RUBBERFIG_CAT_ID = NAME_TO_CAT_ID["RubberFig"]

# Subset yang dipakai V6 (utk perbandingan historis / reproduksi validasi)
V6_SPECIES_SUBSET = (
    NAME_TO_CAT_ID["Mango"], NAME_TO_CAT_ID["AliiFig"], NAME_TO_CAT_ID["BangaloPalm"],
    NAME_TO_CAT_ID["Fern"], NAME_TO_CAT_ID["RubberFig"], NAME_TO_CAT_ID["Umbrella"],
)

RAINFOREST_SPECIES = tuple(
    NAME_TO_CAT_ID[n] for n in
    ["AliiFig", "BangaloPalm", "Fern", "RubberFig", "Umbrella"]  # LeechVine excluded (blank>50%)
)
PLANTATION_SPECIES = tuple(
    NAME_TO_CAT_ID[n] for n in
    ["Apple", "Loquat", "Orange", "Persimmon", "Pomegranate"]  # Lemon excluded (blank>50%)
)

# Rainforest + 2 spesies plantation yg SECARA TEORITIS paling layak diukur DBH@1.3m:
# Mango (pohon batang tunggal besar, ada tradisi ukur trunk diameter di literatur agroforestri)
# dan Persimmon (batang tunggal non-suckering) -- BEDA dari Apple/Citrus (industri ukur di
# ~15-30cm, bukan 1.3m -- rawan sudah masuk zona percabangan) dan Pomegranate (secara alami
# shrub multi-batang, "DBH tunggal" scr struktural tidak terdefinisi baik). Lihat diskusi sesi
# 12 Jul 2026 (WebSearch TCSA apple/citrus, growth habit pomegranate/persimmon/loquat/mango).
RAINFOREST_MANGO_PERSIMMON = tuple(sorted(set(
    RAINFOREST_SPECIES) | {NAME_TO_CAT_ID["Mango"], NAME_TO_CAT_ID["Persimmon"]}))

SPECIES_PRESETS = {
    "universe11": DEFAULT_SPECIES_UNIVERSE,
    "v6_6": V6_SPECIES_SUBSET,
    "rainforest_mango_persimmon": RAINFOREST_MANGO_PERSIMMON,
    "rainforest": RAINFOREST_SPECIES,
    "plantation": PLANTATION_SPECIES,
    "drop_rubberfig": tuple(c for c in DEFAULT_SPECIES_UNIVERSE if c != RUBBERFIG_CAT_ID),
    "drop_mango": tuple(c for c in DEFAULT_SPECIES_UNIVERSE if c != NAME_TO_CAT_ID["Mango"]),
    "drop_both": tuple(c for c in DEFAULT_SPECIES_UNIVERSE
                        if c not in (RUBBERFIG_CAT_ID, NAME_TO_CAT_ID["Mango"])),
}


def resolve_species_subset(spec: str) -> Tuple[int, ...]:
    """'universe11' / 'v6_6' / ... (lihat SPECIES_PRESETS) ATAU '1,3,4,5' (comma-separated cat_id)."""
    if spec in SPECIES_PRESETS:
        return SPECIES_PRESETS[spec]
    return tuple(int(x) for x in spec.split(","))


def apply_filters(
    df: pd.DataFrame,
    species_subset: Optional[Sequence[int]] = None,
    min_trunk_px: int = 5,
    min_depth: float = 1.0,
    max_depth: float = 20.0,
    wh_tolerance: float = 0.5,
    ratio_bounds: Optional[Tuple[float, float]] = None,
    rubberfig_cap_cm: Optional[float] = 150.0,
) -> pd.DataFrame:
    """
    Terapkan kombinasi filter axis ke tabel index (satu baris = satu annotation).

    Tabel index (dibangun oleh precompute_dbh_cache.py) diasumsikan berisi kolom:
      image_id, ann_id, split, category_id, gt_dbh_mm,
      row_feat, trunk_cols (list[int], indeks kolom feature map),
      trunk_px, d_trunk, world_h, dbh_geom_mm, feat_key

    Semua bound di sini adalah operasi boolean murni (tidak decode ulang
    mask/PFM) -> instan walau dipanggil ratusan kali dalam satu sweep.

    species_subset : None -> DEFAULT_SPECIES_UNIVERSE (11 spesies)
    ratio_bounds    : None -> filter rasio dbh_geom/gt dimatikan
    rubberfig_cap_cm: None -> tanpa cap; float -> exclude RubberFig dgn gt_dbh_mm >= cap*10
    """
    if species_subset is None:
        species_subset = DEFAULT_SPECIES_UNIVERSE

    mask = df["category_id"].isin(species_subset)
    mask &= df["trunk_px"] >= min_trunk_px
    mask &= (df["d_trunk"] >= min_depth) & (df["d_trunk"] < max_depth)
    mask &= (df["world_h"] - 1.3).abs() <= wh_tolerance

    if ratio_bounds is not None:
        lo, hi = ratio_bounds
        ratio = df["dbh_geom_mm"] / df["gt_dbh_mm"].clip(lower=1e-6)
        mask &= (ratio >= lo) & (ratio <= hi)

    if rubberfig_cap_cm is not None:
        is_rf = df["category_id"] == RUBBERFIG_CAT_ID
        over_cap = is_rf & (df["gt_dbh_mm"] >= rubberfig_cap_cm * 10.0)
        mask &= ~over_cap

    return df[mask].reset_index(drop=True)


def species_weight_scheme(
    df: pd.DataFrame, scheme: str = "inverse_freq_capped", cap: float = 20.0
) -> dict:
    """
    Hitung {category_id: weight} dari distribusi N di df (SUDAH difilter).
    scheme: "none" | "inverse_freq_capped" | "sqrt_inverse_freq"
    """
    counts = df["category_id"].value_counts()
    if counts.empty:
        return {}
    n_max = counts.max()

    if scheme == "none":
        return {int(cid): 1.0 for cid in counts.index}
    if scheme == "sqrt_inverse_freq":
        return {int(cid): float(np.sqrt(n_max / n)) for cid, n in counts.items()}
    if scheme == "inverse_freq_capped":
        return {int(cid): float(min(n_max / n, cap)) for cid, n in counts.items()}
    raise ValueError(f"Unknown species_weight scheme: {scheme}")


def cat_log_stats(df: pd.DataFrame) -> Tuple[dict, dict]:
    """Hitung {category_id: log1p(dbh_mm) mean/std} dari df (SUDAH difilter)."""
    log_dbh = np.log1p(df["gt_dbh_mm"].to_numpy())
    means, stds = {}, {}
    for cid in df["category_id"].unique():
        vals = log_dbh[df["category_id"].to_numpy() == cid]
        means[int(cid)] = float(vals.mean())
        stds[int(cid)] = float(vals.std()) if len(vals) > 1 else 1.0
    return means, stds
