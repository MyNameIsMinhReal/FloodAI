# -*- coding: utf-8 -*-
"""
depth_analysis/junk_classifier.py
=================================
Phân loại nhanh ảnh TRƯỚC khi chạy pipeline để tránh ngốn tài nguyên.

Mục đích:
  - Junk  : ảnh không liên quan lũ lụt (chart, screenshot, meme, indoor, selfie,
            document, bản đồ...). → Reject ngay, không chạy analysis.
  - Dry   : ảnh outdoor khô ráo, không nước. → Skip depth estimation.
  - Flood : ảnh có dấu hiệu nước/ngập. → Chạy pipeline bình thường.
  - Uncertain: không chắc → cho pipeline quyết định (an toàn, tránh false reject).

Chỉ dùng rule-based + color histogram, KHÔNG dùng model nặng (GPU-free, ~10ms/ảnh).
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class ImageType(Enum):
    """Loại ảnh sau khi pre-validate."""
    FLOOD     = "flood"      # có nước/ngập → chạy pipeline
    DRY       = "dry"        # outdoor khô ráo → skip depth
    JUNK      = "junk"       # không liên quan → reject
    UNCERTAIN = "uncertain"  # không chắc → pipeline quyết định

    @property
    def vietnamese(self) -> str:
        return {
            "flood": "ảnh có dấu hiệu ngập nước",
            "dry": "ảnh outdoor khô ráo, không phát hiện ngập",
            "junk": "ảnh không liên quan đến lũ lụt",
            "uncertain": "không xác định được nội dung",
        }[self.value]


@dataclass
class JunkClassification:
    """Kết quả pre-validate cho 1 ảnh."""
    image_type: ImageType
    confidence: float          # 0-1
    reason:     str            # key máy đọc được (VD: "chart_detected")
    sub_type:   str = ""       # chi tiết human-readable (VD: "screenshot", "meme")
    text_px_ratio: float = 0.0
    water_ratio: float = 0.0
    edge_density: float = 0.0

    @property
    def skip_pipeline(self) -> bool:
        """Junk hoặc Dry đều không cần chạy depth estimation."""
        return self.image_type in (ImageType.JUNK, ImageType.DRY)

    def to_dict(self) -> dict:
        return {
            "image_type":  self.image_type.value,
            "confidence":  round(self.confidence, 3),
            "reason":      self.reason,
            "sub_type":    self.sub_type,
            "skip_pipeline": self.skip_pipeline,
            "message":     self.vietnamese_message(),
        }

    def vietnamese_message(self) -> str:
        """Message Vietnamese cho user/app."""
        if self.image_type == ImageType.JUNK:
            return (f"Ảnh này được xác định là {self.sub_type or 'ảnh rác'} "
                    f"không liên quan đến lũ lụt (độ tin cậy {self.confidence:.0%}).")
        if self.image_type == ImageType.DRY:
            return (f"Ảnh này có vẻ khô ráo, không phát hiện dấu hiệu ngập nước "
                    f"(độ tin cậy {self.confidence:.0%}).")
        if self.image_type == ImageType.FLOOD:
            return f"Ảnh này có dấu hiệu ngập nước (độ tin cậy {self.confidence:.0%})."
        return "Không xác định chắc chắn nội dung ảnh, cần kiểm tra thêm."


class JunkClassifier:
    """
    Rule-based + color histogram image pre-classifier.

    Flow (layer cascade — layer sau chỉ chạy nếu layer trước chưa đủ tự tin):
      L1: geometric rules (resolution, aspect ratio, solid color)
      L2: text/edge density → chart/screenshot/document detection
      L3: color histogram + water-ratio → flood / dry detection
    """

    MIN_RESOLUTION = 64          # < này → quá nhỏ, không dùng được
    MAX_ASPECT_RATIO = 8.0       # panorama cực dài → junk
    SOLID_COLOR_VAR = 60.0       # độ lệch chuẩn thấp → screenshot/solid
    TEXT_EDGE_DENSITY = 0.42     # mật độ cạnh cao > này → văn bản/chart
    LOW_EDGE_DENSITY = 0.02      # quá ít cạnh → mờ/nhiễu
    WATER_HSV_BLUE = ((95, 60, 40), (140, 255, 255))   # nước xanh
    WATER_HSV_BROWN = ((0, 60, 60), (25, 255, 255))    # nước bùn nâu

    def __init__(self, junk_threshold: float = 0.80,
                 dry_threshold: float = 0.55,
                 dry_max_water: float = 5.0):
        self.junk_threshold = junk_threshold
        self.dry_threshold = dry_threshold
        self.dry_max_water = dry_max_water

    # ── API công khai ────────────────────────────────────────────
    def classify(self, image) -> JunkClassification:
        """
        Nhận đường dẫn ảnh (str/Path) hoặc ndarray BGR → trả JunkClassification.

        Không bao giờ raise — mọi lỗi đều fallback về UNCERTAIN để pipeline
        xử lý an toàn thay vì làm sập flow.
        """
        try:
            if isinstance(image, (str, Path)):
                arr = cv2.imread(str(image))
                if arr is None:
                    arr = self._load_pil_fallback(image)
            else:
                arr = image
            if arr is None or arr.size == 0:
                return JunkClassification(ImageType.UNCERTAIN, 0.0,
                                          "cannot_read",
                                          "không đọc được ảnh")
            return self._cascade(arr, str(image))
        except Exception as e:
            log.exception("[JunkClassifier] lỗi khi phân loại")
            return JunkClassification(ImageType.UNCERTAIN, 0.0,
                                      f"error:{type(e).__name__}", "lỗi xử lý")

    # ── Cascade ──────────────────────────────────────────────────
    def _cascade(self, bgr: np.ndarray, name: str) -> JunkClassification:
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # L1: geometric rules
        l1 = self._level1(w, h)
        if l1 is not None:
            return l1

        # L2: edge density → chart/screenshot/document/text
        edges = self._edge_stats(gray)
        l2 = self._level2(edges)
        if l2 is not None and l2.image_type != ImageType.UNCERTAIN:
            return l2

        # L3: color histogram + water ratio → flood/dry
        water_ratio = self._water_ratio(bgr)
        l3 = self._level3(bgr, edges, water_ratio)
        if l3.image_type in (ImageType.FLOOD, ImageType.DRY, ImageType.JUNK):
            return l3

        # Cả L2 lẫn L3 đều uncertain → dùng kết quả L2 nếu có (mờ) khác không
        if l2 is not None:
            return l2
        return l3

    # ── L1: geometric ────────────────────────────────────────────
    def _level1(self, w: int, h: int) -> Optional[JunkClassification]:
        if min(w, h) < self.MIN_RESOLUTION:
            return JunkClassification(ImageType.JUNK, 0.95, "too_small",
                                      "ảnh quá nhỏ, không rõ")
        ratio = max(w, h) / max(1, min(w, h))
        if ratio > self.MAX_ASPECT_RATIO:
            return JunkClassification(ImageType.JUNK, 0.90, "extreme_aspect_ratio",
                                      "tỷ lệ khung hình bất thường (panorama/scan)")
        return None

    # ── L2: text / edge density ──────────────────────────────────
    def _level2(self, edges: dict) -> Optional[JunkClassification]:
        density = edges["density"]
        if density >= self.TEXT_EDGE_DENSITY:
            # Rất nhiều cạnh → chart/bảng số/document/screenshot UI
            return JunkClassification(ImageType.JUNK, 0.85, "very_high_edge",
                                      "biểu đồ/bảng/screenshot",
                                      edge_density=density,
                                      text_px_ratio=density)
        if density <= self.LOW_EDGE_DENSITY:
            # Quá ít cạnh → background trơn / mờ / bị lỗi
            return JunkClassification(ImageType.UNCERTAIN, 0.5, "low_edge",
                                      "ảnh quá mờ hoặc trơn", edge_density=density)
        return None

    # ── L3: color / water ─────────────────────────────────────────
    def _level3(self, bgr: np.ndarray, edges: dict,
                water_ratio: float) -> JunkClassification:
        # Water ratio cao → flood (kể cả khi tương đối đồng màu, v.d. ảnh mặt nước)
        if water_ratio >= self.dry_max_water:
            return JunkClassification(ImageType.FLOOD, self._clip(0.5 + water_ratio / 100.0),
                                      "water_detected",
                                      "phát hiện vùng nước/ngập",
                                      water_ratio=water_ratio,
                                      edge_density=edges["density"])

        # Solid color → screenshot/UI
        if self._is_solid_color(bgr):
            return JunkClassification(ImageType.JUNK, 0.80, "solid_color",
                                      "màu đồng nhất, có thể là screenshot",
                                      edge_density=edges["density"])

        # Nước ít → có thể dry outdoor hoặc flood nhẹ
        # Không thấy nước đáng kể → dry outdoor (nếu có cảnh outdoor)
        if self._looks_outdoor(bgr):
            return JunkClassification(ImageType.DRY, self._clip(self.dry_threshold),
                                      "dry_outdoor",
                                      "outdoor khô ráo, không nước",
                                      water_ratio=water_ratio,
                                      edge_density=edges["density"])

        return JunkClassification(ImageType.UNCERTAIN, 0.5, "uncertain",
                                  "không xác định", water_ratio=water_ratio,
                                  edge_density=edges["density"])

    # ── Helpers ──────────────────────────────────────────────────
    def _edge_stats(self, gray: np.ndarray) -> dict:
        canny = cv2.Canny(gray, 60, 160)
        density = float(canny.mean() / 255.0)
        return {"density": density}

    def _is_solid_color(self, bgr: np.ndarray) -> bool:
        """UI/screenshot thường có vùng lớn đồng màu; ảnh chụp thật có texture khắp nơi."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Gradien nội địa: ảnh tự nhiên có biến thiên độ sáng khắp nơi
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        grad = cv2.magnitude(gx, gy)
        # Phần trăm pixel "phẳng" (gradient gần 0)
        flat_ratio = float((grad < 3.0).mean())
        v_std = float(gray.std())
        # UI: hầu hết diện tích phẳng + độ sáng ít biến thiên toàn cục
        return flat_ratio > 0.85 and v_std < self.SOLID_COLOR_VAR

    def _water_ratio(self, bgr: np.ndarray) -> float:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        total = max(1, hsv.shape[0] * hsv.shape[1])
        mask_blue = cv2.inRange(hsv, *self.WATER_HSV_BLUE)
        mask_brown = cv2.inRange(hsv, *self.WATER_HSV_BROWN)
        water = cv2.bitwise_or(mask_blue, mask_brown)
        return float(cv2.countNonZero(water)) / total * 100.0

    def _looks_outdoor(self, bgr: np.ndarray) -> bool:
        """Ước lượng sơ bộ có phải cảnh outdoor (nhiều cây xanh/đất/trời) hay không."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        total = max(1, hsv.shape[0] * hsv.shape[1])
        # Xanh lá (vegetation)
        green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        # Nâu/đất/cát/bê tông ấm
        brown = cv2.inRange(hsv, (0, 30, 40), (30, 200, 255))
        outdoor = (float(cv2.countNonZero(green)) + float(cv2.countNonZero(brown))) / total
        return outdoor > 0.15

    @staticmethod
    def _clip(v: float) -> float:
        return float(max(0.0, min(1.0, v)))

    def _load_pil_fallback(self, image):
        try:
            with Image.open(str(image)) as im:
                return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
