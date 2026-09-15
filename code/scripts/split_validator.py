"""
Validator: pastikan training selalu menggunakan stratified split.

Import dan panggil check_stratified_datasets(cfg) di awal _main()
pada setiap train script.
"""
import sys
from typing import List


_REQUIRED_KEYWORD = "stratified"

_HINT = (
    "\n  Gunakan dataset dengan suffix '_stratified', contoh:"
    "\n    combined_inst_rle_f1000_repaired_v4_stratified_train"
    "\n    plantations_inst_rle_f1000_repaired_v4_stratified_train"
    "\n  Non-stratified split memiliki scene overlap antara train/val/test"
    "\n  yang dapat menyebabkan inflasi metrics — tidak valid untuk thesis/publikasi."
)


def check_stratified_datasets(cfg, *, error: bool = True) -> bool:
    """
    Periksa bahwa semua DATASETS.TRAIN dan DATASETS.TEST mengandung 'stratified'.

    Parameters
    ----------
    cfg  : detectron2 CfgNode (sudah di-merge dengan config file dan args)
    error: True  → raise SystemExit jika ada non-stratified dataset
           False → hanya print warning

    Returns True jika semua stratified, False jika ada yang tidak.
    """
    bad: List[str] = []

    for ds in list(cfg.DATASETS.TRAIN) + list(cfg.DATASETS.TEST):
        if _REQUIRED_KEYWORD not in ds:
            bad.append(ds)

    if not bad:
        return True

    msg = (
        "\n" + "=" * 70
        + "\n[SPLIT VALIDATOR] ✗  Dataset NON-STRATIFIED terdeteksi:\n"
        + "".join(f"    • {d}\n" for d in bad)
        + _HINT
        + "\n" + "=" * 70
    )

    if error:
        print(msg, file=sys.stderr)
        sys.exit(1)
    else:
        print(msg, file=sys.stderr)
        return False
