# -*- coding: utf-8 -*-
"""
VehicleDetector — phát hiện xe ô tô, xe máy và xe đạp trong ảnh lũ.

Sử dụng YOLO (COCO) để detect:
  - car        (COCO class 2)
  - truck      (COCO class 7)
  - bus        (COCO class 5)
  - motorcycle (COCO class 3)
  - bicycle    (COCO class 1)  ← FIX: đã thêm lại

FIX font: dùng PIL/Pillow với DejaVu Sans thay vì cv2.FONT_HERSHEY_SIMPLEX
          để hiển thị tiếng Việt đúng trên ảnh overlay.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)

# ── COCO class names phát hiện ────────────────────────────────────────────────
# FIX: thêm "bicycle" — trước đây bị bỏ sót nên xe đạp bị nhận nhầm là motorcycle
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

# Tên tiếng Việt hiển thị
VEHICLE_VI = {
    "car":        "Ô tô",
    "truck":      "Xe tải",
    "bus":        "Xe buýt",
    "motorcycle": "Xe máy",
    "bicycle":    "Xe đạp",
}

# Màu BGR cho từng loại xe
VEHICLE_COLORS_BGR = {
    "car":        (0,   165, 255),
    "truck":      (0,   0,   200),
    "bus":        (0,   200, 50),
    "motorcycle": (200, 50,  255),
    "bicycle":    (255, 200, 0),
}

# Chiều cao thực của xe (cm)
VEHICLE_HEIGHT_CM = {
    "car":        145,
    "truck":      280,
    "bus":        310,
    "motorcycle": 110,
    "bicycle":    100,
}

SUBMERGE_PARTIAL  = 0.20
SUBMERGE_CRITICAL = 0.60

# ── Font hỗ trợ tiếng Việt ────────────────────────────────────────────────────
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def _get_pil_font(size: int = 14):
    try:
        from PIL import ImageFont
        for path in _FONT_PATHS:
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
        return ImageFont.load_default()
    except ImportError:
        return None


def _pil_put_text(img_bgr, text: str, pos, font_size: int,
                color_bgr: tuple, bg_color_bgr: tuple | None = None, padding: int = 3):
    """Vẽ text tiếng Việt lên ảnh OpenCV dùng PIL."""
    try:
        from PIL import Image, ImageDraw
        import numpy as np

        font    = _get_pil_font(font_size)
        img_rgb = img_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(img_rgb)
        draw    = ImageDraw.Draw(pil_img)

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        x, y = pos
        if bg_color_bgr is not None:
            bg_rgb = (bg_color_bgr[2], bg_color_bgr[1], bg_color_bgr[0])
            draw.rectangle(
                [x - padding, y - padding, x + tw + padding, y + th + padding],
                fill=bg_rgb
            )

        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text((x, y), text, font=font, fill=color_rgb)

        img_bgr[:] = np.array(pil_img)[:, :, ::-1]
        return img_bgr, tw, th

    except Exception as e:
        import cv2
        fs = max(0.3, font_size / 30.0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        if bg_color_bgr is not None:
            cv2.rectangle(img_bgr,
                          (pos[0] - padding, pos[1] - padding),
                          (pos[0] + tw + padding, pos[1] + th + padding),
                          bg_color_bgr, -1)
        cv2.putText(img_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    fs, color_bgr, 1, cv2.LINE_AA)
        return img_bgr, tw, th


@dataclass
class VehicleDetection:
    vehicle_type:   str
    bbox:           List[int]
    confidence:     float
    submerged_pct:  float = 0.0
    flood_status:   str   = "safe"
    label:          str   = ""

    def to_dict(self) -> dict:
        return {
            "vehicle_type":  self.vehicle_type,
            "vehicle_vi":    VEHICLE_VI.get(self.vehicle_type, self.vehicle_type),
            "bbox":          self.bbox,
            "confidence":    round(self.confidence, 3),
            "submerged_pct": round(self.submerged_pct * 100, 1),
            "flood_status":  self.flood_status,
            "label":         self.label,
        }


class VehicleDetector:
    def __init__(self, conf_thresh: float = 0.35):
        self.conf_thresh = conf_thresh

    def detect(self, img_rgb, yolo_model,
               water_line_y: Optional[int] = None,
               img_h: int = 0, img_w: int = 0) -> List[VehicleDetection]:
        if img_h == 0: img_h = img_rgb.shape[0]
        if img_w == 0: img_w = img_rgb.shape[1]
        vehicles: List[VehicleDetection] = []
        try:
            for res in yolo_model(img_rgb, conf=self.conf_thresh, verbose=False):
                for box in res.boxes:
                    name = res.names[int(box.cls[0])]
                    if name not in VEHICLE_CLASSES:
                        continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(img_w, x2); y2 = min(img_h, y2)
                    bh = y2 - y1; bw = x2 - x1
                    if bh < img_h * 0.02 or bw < img_w * 0.02:
                        continue
                    veh = VehicleDetection(vehicle_type=name, bbox=[x1,y1,x2,y2], confidence=conf)
                    if water_line_y is not None and water_line_y < img_h:
                        veh.submerged_pct = self._calc_submerged(y1, y2, water_line_y)
                    veh.flood_status = self._flood_status(veh.submerged_pct)
                    veh.label        = self._make_label(veh)
                    vehicles.append(veh)
        except Exception as e:
            log.warning(f"[VehicleDetector] YOLO error: {e}")
        log.debug(f"[VehicleDetector] {len(vehicles)} xe: {[v.vehicle_type for v in vehicles]}")
        return vehicles

    @staticmethod
    def _calc_submerged(y1, y2, wl_y) -> float:
        bh = max(y2 - y1, 1)
        if wl_y <= y1: return 1.0
        if wl_y >= y2: return 0.0
        return (y2 - wl_y) / bh

    @staticmethod
    def _flood_status(pct: float) -> str:
        if pct >= SUBMERGE_CRITICAL: return "submerged"
        if pct >= SUBMERGE_PARTIAL:  return "partial"
        return "safe"

    @staticmethod
    def _make_label(v: "VehicleDetection") -> str:
        vi_name = VEHICLE_VI.get(v.vehicle_type, v.vehicle_type)
        # FIX: không dùng emoji unicode (✓ ⚠ ✘) tránh lỗi font fallback
        status_str = {
            "safe":      "An toan",
            "partial":   "Ngap mot phan",
            "submerged": "Ngap nang",
        }[v.flood_status]
        if v.submerged_pct > 0:
            return f"{vi_name} | {v.submerged_pct*100:.0f}% ngap | {status_str}"
        return f"{vi_name} | {status_str}"

    @staticmethod
    def draw_vehicles(img_bgr, vehicles: List[VehicleDetection], scale: float = 1.0):
        """
        Vẽ bbox + label xe lên ảnh.
        FIX: dùng PIL thay cv2.putText để hiển thị tiếng Việt đúng dấu.
        """
        import cv2
        thick   = max(1, round(scale * 2.0))
        font_px = max(11, round(scale * 13))

        STATUS_COLORS_BGR = {
            "safe":      (50,  200,  50),
            "partial":   (0,   170, 255),
            "submerged": (0,   0,   220),
        }

        for v in vehicles:
            x1, y1, x2, y2 = v.bbox
            base_color   = VEHICLE_COLORS_BGR.get(v.vehicle_type, (200, 200, 200))
            status_color = STATUS_COLORS_BGR[v.flood_status]

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 0),  thick + 2)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), base_color, thick)

            bar_h = max(4, round((y2 - y1) * 0.05))
            cv2.rectangle(img_bgr, (x1, y2 - bar_h), (x2, y2), status_color, -1)

            ly = max(y1 - font_px - 6, 2)
            _pil_put_text(img_bgr, v.label,
                          pos=(x1 + 2, ly),
                          font_size=font_px,
                          color_bgr=base_color,
                          bg_color_bgr=(20, 20, 20),
                          padding=3)

        return img_bgr

    @staticmethod
    def summarize(vehicles: List[VehicleDetection]) -> dict:
        if not vehicles:
            return {"total": 0, "by_type": {}, "by_status": {}}
        by_type: dict = {}
        by_status = {"safe": 0, "partial": 0, "submerged": 0}
        for v in vehicles:
            by_type[v.vehicle_type] = by_type.get(v.vehicle_type, 0) + 1
            by_status[v.flood_status] += 1
        return {
            "total":     len(vehicles),
            "by_type":   by_type,
            "by_status": by_status,
            "details":   [v.to_dict() for v in vehicles],
        }
