# -*- coding: utf-8 -*-
"""
depth_analysis/door_detector.py
================================
Phát hiện cửa / cổng / cửa cuốn — thay heuristic contour đơn thuần bằng
validated detection.

Nguyên tắc mới:
  1. Dùng YOLO custom (nếu có) hoặc fallback heuristic
  2. Cửa PHẢI nằm trong building/wall mask (validate_door)
  3. Cửa không thể hoàn toàn bị nước che
  4. Aspect ratio hợp lý (không phải biển quảng cáo ngang)

Classes nhận biết:
  door, gate, window, garage_door, shop_shutter

Đặc biệt thêm shop_shutter vì cửa cuốn/cửa sắt rất phổ biến ở Việt Nam.

Dùng:
    dd = DoorDetector(cfg)
    doors = dd.detect(img_bgr, building_mask=bseg.building_mask, water_mask=water_mask)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("depth_analysis.door")

# Các class cửa
DOOR_CLASSES = {"door", "gate", "window", "garage_door", "shop_shutter",
                "cửa", "cổng", "cửa_cuốn"}

# Aspect ratio hợp lệ (w/h): cửa thường cao hơn rộng
DOOR_ASPECT_MIN = 0.25   # cửa hẹp
DOOR_ASPECT_MAX = 2.5    # cổng rộng / garage_door


@dataclass
class DetectedDoor:
    bbox:        List[int]         # [x1, y1, x2, y2]
    door_class:  str = "door"      # door/gate/window/garage_door/shop_shutter
    confidence:  float = 0.5
    water_depth_cm: Optional[float] = None   # chiều sâu nước ở chân cửa
    is_valid:    bool = True
    invalid_reason: str = ""


class DoorDetector:
    """
    Phát hiện cửa với multi-step validation.

    Dùng:
        dd = DoorDetector(cfg)
        doors = dd.detect(img_bgr, building_mask, water_mask)
        valid_doors = [d for d in doors if d.is_valid]
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.model_path    = cfg.get("door_model", "")
        self.min_height_pct= cfg.get("door_min_height_pct", 0.12)  # ≥12% chiều cao ảnh
        self._model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        img_bgr: np.ndarray,
        building_mask: Optional[np.ndarray] = None,
        water_mask: Optional[np.ndarray] = None,
    ) -> List[DetectedDoor]:
        """
        Detect cửa trong ảnh và validate từng cửa.

        Args:
            img_bgr:       ảnh BGR
            building_mask: mask nhà/tường từ BuildingSegmentor
            water_mask:    mask nước từ WaterDetector

        Returns:
            List[DetectedDoor] — bao gồm cả invalid, kiểm tra is_valid
        """
        h, w = img_bgr.shape[:2]
        raw_doors = self._detect_raw(img_bgr)

        validated = []
        for door in raw_doors:
            ok, reason = validate_door(
                door_bbox=door.bbox,
                building_mask=building_mask,
                water_mask=water_mask,
                img_h=h,
                min_height_pct=self.min_height_pct,
            )
            door.is_valid = ok
            door.invalid_reason = reason
            if not ok:
                log.debug("[Door] Rejected %s at %s: %s", door.door_class, door.bbox, reason)
            validated.append(door)

        valid_count = sum(1 for d in validated if d.is_valid)
        log.debug("[Door] %d raw → %d valid doors", len(validated), valid_count)
        return validated

    def estimate_water_depth(
        self,
        door: DetectedDoor,
        water_mask: np.ndarray,
        door_height_cm: float = 200.0,
    ) -> Optional[float]:
        """
        Ước tính chiều sâu nước tại vị trí cửa dựa trên phần cửa bị nước che.

        Args:
            door:           DetectedDoor đã validate
            water_mask:     mask nước (0/255)
            door_height_cm: chiều cao cửa thật (mặc định 200cm cửa Việt Nam)

        Returns:
            depth_cm hoặc None nếu không xác định được
        """
        if not door.is_valid:
            return None
        x1, y1, x2, y2 = door.bbox
        door_h_px = y2 - y1
        if door_h_px < 1:
            return None

        # Phần cửa bị nước che (từ đáy lên)
        door_water = water_mask[y1:y2, x1:x2]
        if door_water.size == 0:
            return None

        # Tìm y cao nhất (gần đỉnh cửa nhất) có nước
        water_rows = np.where((door_water > 0).any(axis=1))[0]
        if len(water_rows) == 0:
            return None

        highest_water_row = water_rows.min()  # row 0 = đỉnh cửa
        covered_pct = 1.0 - (highest_water_row / door_h_px)
        depth_cm = covered_pct * door_height_cm

        door.water_depth_cm = round(depth_cm, 1)
        return door.water_depth_cm

    # ── Raw detection ─────────────────────────────────────────────────────────

    def _detect_raw(self, img_bgr: np.ndarray) -> List[DetectedDoor]:
        # Thử YOLO custom nếu có
        if self.model_path and Path(self.model_path).exists():
            results = self._run_yolo(img_bgr)
            if results:
                return results

        # Fallback heuristic
        return self._heuristic_detect(img_bgr)

    def _run_yolo(self, img_bgr: np.ndarray) -> Optional[List[DetectedDoor]]:
        try:
            from ultralytics import YOLO
            if self._model is None:
                self._model = YOLO(self.model_path)
            results = self._model(img_bgr, verbose=False)
            doors = []
            for r in results:
                # Ultralytics is imported dynamically, so type checkers may infer
                # each result as a Tensor instead of a Results instance.
                boxes: Any = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    # Ultralytics is imported dynamically; keep the result
                    # metadata opaque to static type checkers.
                    names: Any = getattr(r, "names", {})
                    cls_name = names.get(int(box.cls), "door").lower()
                    if cls_name not in DOOR_CLASSES:
                        continue
                    bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    conf = float(box.conf)
                    doors.append(DetectedDoor(bbox=bbox, door_class=cls_name, confidence=conf))
            return doors
        except Exception as exc:
            log.debug("[Door] YOLO failed: %s", exc)
            return None

    def _heuristic_detect(self, img_bgr: np.ndarray) -> List[DetectedDoor]:
        """
        Heuristic fallback:
        - Tìm hình chữ nhật đứng rõ nét qua gradient + contour
        - Chỉ chấp nhận nếu có edge dọc rõ ở 2 bên
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Cạnh Canny
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate để nối liền cạnh
        k = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, k, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        doors = []
        min_area = h * w * 0.008

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 20 or bh < 30:
                continue

            aspect = bw / max(bh, 1)
            if not (DOOR_ASPECT_MIN <= aspect <= DOOR_ASPECT_MAX):
                continue

            # Phải có edge dọc rõ ở 2 bên bbox
            left_strip  = edges[y:y+bh, max(0, x-4):x+4]
            right_strip = edges[y:y+bh, x+bw-4:min(w, x+bw+4)]
            if left_strip.mean() < 15 or right_strip.mean() < 15:
                continue

            door_class = "shop_shutter" if aspect > 1.5 else ("window" if bh < h * 0.15 else "door")
            doors.append(DetectedDoor(
                bbox=[x, y, x+bw, y+bh],
                door_class=door_class,
                confidence=0.35,  # heuristic → thấp
            ))

        return doors


# ── Validation function ────────────────────────────────────────────────────────

def validate_door(
    door_bbox: List[int],
    building_mask: Optional[np.ndarray],
    water_mask: Optional[np.ndarray],
    img_h: int,
    min_height_pct: float = 0.12,
) -> Tuple[bool, str]:
    """
    Kiểm tra một door bbox có hợp lệ không.

    Rules:
      1. Nằm trong building/wall mask (≥45% overlap)
      2. Không bị nước che hoàn toàn (≤85%)
      3. Chiều cao ≥ min_height_pct * img_h
      4. Aspect ratio hợp lý

    Returns:
        (is_valid: bool, reason: str)
    """
    x1, y1, x2, y2 = door_bbox
    dw = x2 - x1
    dh = y2 - y1

    if dw < 1 or dh < 1:
        return False, "zero size"

    # Rule: chiều cao tối thiểu
    if dh < img_h * min_height_pct:
        return False, f"too short ({dh}px < {img_h * min_height_pct:.0f}px)"

    # Rule: aspect ratio
    aspect = dw / dh
    if not (DOOR_ASPECT_MIN <= aspect <= DOOR_ASPECT_MAX):
        return False, f"bad aspect ratio {aspect:.2f}"

    # Rule: nằm trong building mask
    if building_mask is not None:
        region = building_mask[y1:y2, x1:x2]
        if region.size > 0:
            overlap = (region > 0).mean()
            if overlap < 0.45:
                return False, f"not in building mask (overlap={overlap:.2f})"

    # Rule: không bị nước che hoàn toàn
    if water_mask is not None:
        water_region = water_mask[y1:y2, x1:x2]
        if water_region.size > 0:
            water_pct = (water_region > 0).mean()
            if water_pct > 0.85:
                return False, f"mostly submerged ({water_pct:.0%})"

    return True, ""
