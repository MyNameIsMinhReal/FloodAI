# -*- coding: utf-8 -*-
"""
depth_analysis/building_segmentor.py
======================================
Phát hiện và segment nhà / tường / mặt tiền — dùng cho:
  - Phát hiện vết mực nước trên tường (tideline)
  - Validate vị trí cửa (door phải nằm trong building mask)
  - Loại false positive nước trên tường/biển quảng cáo
  - Ước lượng phối cảnh tốt hơn

Output:
    BuildingResult(
        building_mask  : np.ndarray (0/255)
        wall_mask      : np.ndarray (0/255)
        facade_bboxes  : List[[x1,y1,x2,y2]]
        confidence     : float
        tideline_y     : Optional[int]   ← vết nước trên tường
    )

Hai chế độ:
  1. YOLO-seg custom (nếu có model)  → mask đẹp
  2. Fallback heuristic  (gradient + structural cues) → nhanh, không cần GPU
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("depth_analysis.building")


@dataclass
class BuildingResult:
    building_mask: Optional[np.ndarray] = None   # toàn bộ vùng nhà
    wall_mask:     Optional[np.ndarray] = None   # chỉ tường (không cửa/cửa sổ)
    facade_bboxes: List[List[int]] = field(default_factory=list)
    confidence:    float = 0.0
    tideline_y:    Optional[int] = None   # y-coordinate của vết mực nước trên tường


class BuildingSegmentor:
    """
    Segment nhà/tường trong ảnh ngập.

    Dùng:
        seg = BuildingSegmentor(cfg)
        result = seg.segment(img_bgr)
        # result.building_mask — mask nhà
        # result.wall_mask     — mask tường thuần túy
        # result.tideline_y    — vị trí vết mực nước (nếu tìm thấy)
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.model_path   = cfg.get("building_model", "")
        self.min_area_pct = cfg.get("building_min_area", 0.04)   # ≥4% ảnh
        self._model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def segment(self, img_bgr: np.ndarray) -> BuildingResult:
        """
        Segment building/wall từ ảnh BGR.

        Thử YOLO-seg → fallback heuristic nếu không có model.
        """
        h, w = img_bgr.shape[:2]

        # Thử model custom nếu có
        if self.model_path and Path(self.model_path).exists():
            result = self._run_yolo_seg(img_bgr)
            if result and result.confidence > 0.4:
                result.tideline_y = self._find_tideline(img_bgr, result.wall_mask)
                return result

        # Fallback: heuristic
        return self._heuristic_segment(img_bgr)

    # ── YOLO-seg backend ──────────────────────────────────────────────────────

    def _run_yolo_seg(self, img_bgr: np.ndarray) -> Optional[BuildingResult]:
        try:
            from ultralytics import YOLO
            if self._model is None:
                self._model = YOLO(self.model_path)
            # Ultralytics' type stubs may infer the model output as a Tensor,
            # although inference returns an iterable of Results objects.
            results: Any = self._model(img_bgr, verbose=False)
            h, w = img_bgr.shape[:2]
            building_mask = np.zeros((h, w), dtype=np.uint8)
            wall_mask     = np.zeros((h, w), dtype=np.uint8)
            bboxes = []
            for r in results:
                if r.masks is None:
                    continue
                names = r.names
                for i, (cls_id, mask_xy) in enumerate(zip(r.boxes.cls, r.masks.xy)):
                    cls = names.get(int(cls_id), "").lower()
                    if cls in ("building", "house", "facade", "wall", "nhà", "tường"):
                        pts = np.array(mask_xy, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.fillPoly(building_mask, [pts], 255)
                        if cls in ("wall", "tường"):
                            cv2.fillPoly(wall_mask, [pts], 255)
                        box = r.boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                        bboxes.append(box)

            if building_mask.sum() == 0:
                return None
            if wall_mask.sum() == 0:
                wall_mask = building_mask.copy()

            conf = float(building_mask.sum()) / (h * w * 255)
            return BuildingResult(
                building_mask=building_mask,
                wall_mask=wall_mask,
                facade_bboxes=bboxes,
                confidence=min(conf * 5, 0.95),
            )
        except Exception as exc:
            log.debug("[BuildingSeg] YOLO-seg failed: %s", exc)
            return None

    # ── Heuristic fallback ────────────────────────────────────────────────────

    def _heuristic_segment(self, img_bgr: np.ndarray) -> BuildingResult:
        """
        Heuristic building detection dựa trên:
          - Gradient dọc mạnh (cạnh tường)
          - Màu sắc đồng nhất theo chiều ngang (texture tường)
          - Vị trí trong frame (thường chiếm nửa trên)
          - Line detection (Hough) → tường có nhiều đường ngang
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Gradient ngang → cạnh đứng (tường)
        sobelx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        vert_edge = (sobelx > np.percentile(sobelx, 80)).astype(np.uint8) * 255

        # Hough lines để tìm cạnh đứng
        lines = cv2.HoughLinesP(vert_edge, 1, np.pi / 180, 60,
                                minLineLength=h * 0.15, maxLineGap=20)
        facade_mask = np.zeros((h, w), dtype=np.uint8)
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1 + 1e-9)))
                if 60 < angle < 120:  # gần đứng
                    cv2.line(facade_mask, (x1, y1), (x2, y2), 255, 8)

        # Mở rộng và fill
        k = np.ones((25, 25), np.uint8)
        facade_mask = cv2.dilate(facade_mask, k)

        # Giới hạn vùng building ở nửa trên (không phải mặt đất)
        facade_mask[int(h * 0.85):] = 0

        if facade_mask.sum() < 255 * 500:
            # Fallback mạnh hơn: coi nửa trên ảnh là building nếu không detect được
            facade_mask = np.zeros((h, w), dtype=np.uint8)
            facade_mask[:int(h * 0.70), :] = 128  # uncertain, low confidence

        # Tìm bounding boxes
        contours, _ = cv2.findContours(facade_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        min_area = h * w * self.min_area_pct
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            bboxes.append([x, y, x + bw, y + bh])

        conf = 0.35 if bboxes else 0.15  # heuristic → low confidence
        result = BuildingResult(
            building_mask=facade_mask,
            wall_mask=facade_mask.copy(),
            facade_bboxes=bboxes,
            confidence=conf,
        )
        result.tideline_y = self._find_tideline(img_bgr, facade_mask)
        return result

    # ── Tideline detection ────────────────────────────────────────────────────

    def _find_tideline(
        self,
        img_bgr: np.ndarray,
        wall_mask: Optional[np.ndarray],
    ) -> Optional[int]:
        """
        Tìm vết mực nước trên tường (tideline):
        - Đường ngang đổi màu đột ngột trên vùng wall
        - Thường là dải đậm/nhạt đơn ngang

        Returns:
            y-coordinate của tideline, hoặc None nếu không tìm thấy
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Chỉ xét vùng tường
        if wall_mask is not None and wall_mask.sum() > 0:
            region = cv2.bitwise_and(gray, gray, mask=(wall_mask > 0).astype(np.uint8))
        else:
            region = gray.copy()
            region[int(h * 0.80):] = 0  # bỏ phần dưới

        # Tính độ biến đổi màu theo hàng (row variance)
        row_var = np.var(region.astype(np.float32), axis=1)

        # Tìm hàng có biến đổi cao nhất trong nửa dưới của wall region
        search_top    = int(h * 0.15)
        search_bottom = int(h * 0.80)
        if search_bottom <= search_top:
            return None

        search_var = row_var[search_top:search_bottom]
        if search_var.max() < 50:  # quá thấp → không có tideline rõ
            return None

        tideline_y = int(np.argmax(search_var)) + search_top
        log.debug("[BuildingSeg] Tideline detected at y=%d", tideline_y)
        return tideline_y
