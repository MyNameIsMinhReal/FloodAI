# -*- coding: utf-8 -*-
"""
depth_analysis/reference_objects.py
-------------------------------------
Dinh nghia tat ca vat the tham chieu de do muc nuoc lu.

Moi vat the co:
  - height_cm:      chieu cao thuc te (cm)
  - width_cm:       chieu rong thuc te (cm) - dung de tinh ty le anh
  - priority:       uu tien su dung (1=cao, 5=thap)
  - color:          mau ve bbox (BGR)
  - pose_sensitive: True = can kiem tra pose truoc khi do
  - waterproof:     True = vat the nay khong bi ngap (bo qua khi tinh muc nuoc)
  - flood_indicator:True = su xuat hien cua vat the nay = co lu (bo, phao, v.v.)
  - measure_point:  "bottom" hoac "waterline" - diem do tren vat the
  - notes:          ghi chu them
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ReferenceObject:
    name:            str
    height_cm:       float
    width_cm:        float
    priority:        int              # 1 = cao nhat
    color_bgr:       Tuple[int,...]  # BGR
    yolo_class:      Optional[str]   # ten class trong YOLO COCO (None = custom)
    pose_sensitive:  bool  = False   # can check pose truoc khi do
    waterproof:      bool  = False   # vat the nay khong bao gio bi "ngap" that
    flood_indicator: bool  = False   # co vat the nay = co lu
    measure_point:   str   = "bottom"# "bottom"=chan, "full"=toan than
    notes:           str   = ""


# ==============================================================================
# DANH SACH VAT THE THAM CHIEU
# ==============================================================================

REFERENCE_OBJECTS: dict[str, ReferenceObject] = {

    # =========================================================================
    # NGUOI (pose-sensitive: phan biet dung/ngoi/boi)
    # =========================================================================
    "person": ReferenceObject(
        name          = "Nguoi (dung thang)",
        height_cm     = 170,
        width_cm      = 50,
        priority      = 1,
        color_bgr     = (0, 255, 0),
        yolo_class    = "person",
        pose_sensitive= True,    # QUAN TRONG: can check pose
        notes         = "Chieu cao trung binh nguoi Viet Nam"
    ),

    # =========================================================================
    # XE CO GIOI
    # =========================================================================
    "motorcycle": ReferenceObject(
        name       = "Xe may",
        height_cm  = 110,
        width_cm   = 65,
        priority   = 2,
        color_bgr  = (255, 128, 0),
        yolo_class = "motorcycle",
        notes      = "Tinh tu mat dat den yen xe"
    ),
    "bicycle": ReferenceObject(
        name       = "Xe dap",
        height_cm  = 95,
        width_cm   = 60,
        priority   = 2,
        color_bgr  = (255, 200, 0),
        yolo_class = "bicycle",
    ),
    "car": ReferenceObject(
        name       = "O to con",
        height_cm  = 145,
        width_cm   = 180,
        priority   = 2,
        color_bgr  = (0, 165, 255),
        yolo_class = "car",
        notes      = "Tinh den noc xe, nguong ngap an toan < 30cm"
    ),
    "truck": ReferenceObject(
        name       = "Xe tai",
        height_cm  = 280,
        width_cm   = 240,
        priority   = 3,
        color_bgr  = (0, 0, 255),
        yolo_class = "truck",
    ),
    "bus": ReferenceObject(
        name       = "Xe buyt",
        height_cm  = 320,
        width_cm   = 250,
        priority   = 3,
        color_bgr  = (128, 0, 255),
        yolo_class = "bus",
    ),

    # =========================================================================
    # TOA NHA / KET CAU XAY DUNG
    # =========================================================================
    "stop_sign": ReferenceObject(
        name       = "Bien dung (biet chieu cao)",
        height_cm  = 220,   # chieu cao cot bien thuong 2.2m
        width_cm   = 75,
        priority   = 3,
        color_bgr  = (0, 0, 200),
        yolo_class = "stop sign",
        notes      = "Cot bien duong o VN: 2.0-2.5m"
    ),
    "fire_hydrant": ReferenceObject(
        name       = "Tru nuoc chua chay",
        height_cm  = 70,
        width_cm   = 30,
        priority   = 3,
        color_bgr  = (0, 60, 200),
        yolo_class = "fire hydrant",
        notes      = "Chieu cao tieu chuan ~70cm, rat tot de do nuoc thap"
    ),

    # =========================================================================
    # DO VAT / NOI THAT (do muc nuoc thap)
    # =========================================================================
    "chair": ReferenceObject(
        name       = "Ghe",
        height_cm  = 80,
        width_cm   = 45,
        priority   = 4,
        color_bgr  = (180, 180, 0),
        yolo_class = "chair",
        notes      = "Ghe thuong ~80cm, tot de do nuoc thap trong nha"
    ),
    "dining_table": ReferenceObject(
        name       = "Ban an",
        height_cm  = 75,
        width_cm   = 120,
        priority   = 4,
        color_bgr  = (140, 180, 0),
        yolo_class = "dining table",
    ),
    "refrigerator": ReferenceObject(
        name       = "Tu lanh",
        height_cm  = 170,
        width_cm   = 60,
        priority   = 3,
        color_bgr  = (200, 200, 200),
        yolo_class = "refrigerator",
    ),

    # =========================================================================
    # VAT THE CHI BAO LU (su ton tai = co lu)
    # =========================================================================
    "boat": ReferenceObject(
        name          = "Thuyen / Bo",
        height_cm     = 80,    # chieu cao thanh bo
        width_cm      = 200,
        priority      = 1,
        color_bgr     = (255, 0, 200),
        yolo_class    = "boat",
        flood_indicator= True,  # Co thuyen = co lu chac chan
        waterproof    = True,   # Thuyen KHONG bi ngap
        notes         = "Su co mat cua thuyen = xac nhan co lu"
    ),

    # =========================================================================
    # VAT THE KHONG DUNG DE DO (waterproof / misleading)
    # =========================================================================
    "umbrella": ReferenceObject(
        name          = "O (du)",
        height_cm     = 160,
        width_cm      = 100,
        priority      = 5,
        color_bgr     = (100, 100, 200),
        yolo_class    = "umbrella",
        waterproof    = True,   # Nguoi cam o khong phai la dang bi ngap
        pose_sensitive= True,
        notes         = "Nguoi cam o co the dang di trong mua, khong phai ngap"
    ),
}


# ==============================================================================
# POSE STATES - cac trang thai tu the cua nguoi
# ==============================================================================

class PoseState:
    STANDING   = "standing"     # dung thang: do chieu cao day du
    SITTING    = "sitting"      # ngoi: giam ~40% chieu cao
    CROUCHING  = "crouching"   # ngoi xom: giam ~55% chieu cao
    BENDING    = "bending"     # cuoi nguoi
    HUNCHBACK  = "hunchback"   # gu lung
    SWIMMING   = "swimming"     # dang boi: khong do duoc
    WADING     = "wading"       # loi nuoc: than duoi bi ngap that
    UNKNOWN    = "unknown"


# Ty le chieu cao theo tung pose (so voi chieu cao dung = 1.0)
POSE_HEIGHT_RATIO = {
    PoseState.STANDING:  1.00,
    PoseState.SITTING:   0.60,   # ngoi ghe: ~60% chieu cao
    PoseState.CROUCHING: 0.45,
    PoseState.BENDING:   0.85,   # cuoi nguoi nhiet
    PoseState.HUNCHBACK: 0.90,   # gu lung, trai nguoi thap hon mot chut
    PoseState.SWIMMING:  None,   # khong ap dung
    PoseState.WADING:    1.00,   # dung trong nuoc: chieu cao khong doi
    PoseState.UNKNOWN:   1.00,   # an toan: dung chieu cao day du
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_yolo_class_list() -> list:
    """Tra ve list cac YOLO class can detect."""
    classes = set()
    for obj in REFERENCE_OBJECTS.values():
        if obj.yolo_class:
            classes.add(obj.yolo_class)
    return sorted(classes)


def get_by_yolo_class(yolo_class: str) -> Optional[ReferenceObject]:
    """Tim ReferenceObject theo yolo_class name."""
    for obj in REFERENCE_OBJECTS.values():
        if obj.yolo_class == yolo_class:
            return obj
    return None


def get_flood_indicators() -> list:
    """Tra ve list vat the la chi bao lu (flood_indicator=True)."""
    return [obj for obj in REFERENCE_OBJECTS.values() if obj.flood_indicator]


def get_waterproof_objects() -> list:
    """Tra ve list vat the khong bi ngap (waterproof=True)."""
    return [obj for obj in REFERENCE_OBJECTS.values() if obj.waterproof]


def get_infrastructure_objects() -> list:
    """Tra ve list vat the co dinh (cot, bien bao...) dung lam nguong do nuoc."""
    infra_keys = {
        "traffic_light", "stop_sign", "fire_hydrant",
        "utility_pole", "street_lamp", "tree_trunk",
    }
    return [obj for k, obj in REFERENCE_OBJECTS.items() if k in infra_keys]
