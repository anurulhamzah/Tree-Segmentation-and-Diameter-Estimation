"""Paket Mask R-CNN RGBD + DBH (self-contained, hanya detectron2).

Mengimpor modul di sini agar registrasi ROI head (DBHROIHeads) ke ROI_HEADS_REGISTRY
ter-trigger saat paket di-import. Tidak ada ketergantungan ke repo MaskDINO.
"""
from .config import add_rgbd_dbh_config
from .dataset_mapper import RGBDDatasetMapper
from . import roi_heads  # noqa: F401  (registrasi DBHROIHeads)

__all__ = ["add_rgbd_dbh_config", "RGBDDatasetMapper"]
