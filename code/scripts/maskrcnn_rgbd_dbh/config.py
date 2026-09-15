"""Config keys tambahan untuk Mask R-CNN RGBD + DBH.

Dipanggil SESUDAH get_cfg() (detectron2 default) dan SEBELUM merge_from_file.
Tidak menyentuh konfigurasi MaskDINO sama sekali.
"""
from detectron2.config import CfgNode as CN


def add_rgbd_dbh_config(cfg):
    # ── RGBD: depth sebagai channel ke-4 ────────────────────────────────────
    cfg.INPUT.USE_DEPTH = False
    cfg.INPUT.DEPTH_DIR = ""           # folder berisi <stem>.pfm
    # Channel ke-4 ditangani otomatis oleh detectron2 karena len(PIXEL_MEAN)==4.

    # ── Filter jarak kamera-ke-pohon utk supervisi DBH (meniru --max-depth di
    # pipeline FocalNet-L, TAPI versi sederhana: median depth mentah di dalam
    # gt_mask, bukan estimasi geometris trunk-row breast-height. DBHROIHeads
    # meregresi DBH langsung dari feature ROI, tidak punya konsep "trunk row" —
    # jadi proxy jarak median-per-mask ini yang paling sepadan level presisinya
    # dgn pendekatan regresi ROI-based ini, bukan usaha meniru geometri exact.
    # 0.0 = nonaktif (perilaku lama, semua instance dgn dbh>0 dipakai).
    cfg.INPUT.DBH_MAX_DEPTH = 0.0

    # ── DBH regression ROI head ─────────────────────────────────────────────
    cfg.MODEL.ROI_DBH_HEAD = CN()
    cfg.MODEL.ROI_DBH_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_DBH_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_DBH_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_DBH_HEAD.LOSS_WEIGHT = 0.5   # samakan dgn DBH_WEIGHT MaskDINO (0.5)
    # Stop-gradient loss_dbh sebelum backbone/FPN (meniru --detach-dbh-backbone di
    # finetune_trunk_roi_dbh_hybrid_joint.py). Ditambahkan 25 Agt 2026 setelah temuan detach
    # menang di FocalNet-L (seg +1.17pp, RMSE -3.81mm vs joint-biasa) -- backbone tetap dilatih
    # penuh oleh loss segmentasi/box/mask, cuma gradien loss_dbh yang diputus.
    cfg.MODEL.ROI_DBH_HEAD.DETACH_BACKBONE = False
