# -*- coding: utf-8 -*-
"""
depth_analysis/sam_segmentor.py
================================
SAM2 (Segment Anything 2) — mask pixel chính xác cho object đo ngập.

Vấn đề giải quyết:
    YOLO bbox là HÌNH CHỮ NHẬT thô. Người đứng nghiêng/chụp lệch góc thì
    bbox chứa nhiều background → đếm pixel trên/dưới water line bị sai
    → water_cm sai. SAM2 cho mask theo ĐÚNG hình dáng object.

Cách dùng:
    from depth_analysis.sam_segmentor import get_sam_segmentor
    seg = get_sam_segmentor(cfg)          # lazy load, fail → None
    if seg:
        sub_px = seg.count_below_water(img_rgb, bbox, water_line_y)

Model: chạy qua ultralytics (đã có sẵn dependency của YOLO):
    pip install ultralytics   # SAM2 đi kèm, tự download weights lần đầu
    Model nhỏ "sam2_t.pt" (~160MB) đủ dùng; lớn hơn: sam2_s/b/l.pt

Config (config.yaml):
    models.use_sam: true
    models.sam: "sam2_t.pt"
"""

from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional, Tuple

import numpy as np

log = logging.getLogger("depth.sam_segmentor")


class SAMSegmentor:
    """Wrapper ultralytics SAM2 — API tối giản cho đo ngập."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        mcfg = cfg.get("models", {}) or {}
        self.model_name: str = mcfg.get("sam", "sam2_t.pt")
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Loading ────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        if self.is_loaded:
            return True
        if self._failed:
            return False
        with self._lock:
            if self.is_loaded:
                return True
            try:
                from ultralytics import SAM
                log.info(f"  [SAM] Loading {self.model_name} …")
                self._model = SAM(self.model_name)
                log.info("  [SAM] Sẵn sàng")
                return True
            except Exception as exc:
                log.warning(f"  [SAM] Không tải được ({exc}) — đo bằng bbox như cũ")
                self._failed = True
                return False

    def unload(self) -> None:
        self._model = None
        self._failed = False

    # ── Core API ───────────────────────────────────────────────────────────────

    def segment(
        self,
        img_rgb: np.ndarray,
        bboxes: List[List[int]],
    ) -> List[Optional[np.ndarray]]:
        """
        Segment các object theo bboxes (prompted boxes).

        Returns:
            list mask (H,W) uint8 {0,255} hoặc None nếu fail.
        """
        if not bboxes or not self._ensure_loaded():
            return [None] * len(bboxes)
        try:
            results = self._model(img_rgb, bboxes=bboxes, verbose=False)
            masks: List[Optional[np.ndarray]] = []
            for i in range(len(bboxes)):
                m = None
                if results and len(results) > 0:
                    r = results[0]
                    if getattr(r, "masks", None) is not None:
                        data = r.masks.data          # tensor (N, h', w')
                        if i < len(data):
                            m = data[i].cpu().numpy()
                            # resize về size ảnh gốc nếu cần
                            if m.shape[:2] != img_rgb.shape[:2]:
                                import cv2
                                m = cv2.resize(
                                    m, (img_rgb.shape[1], img_rgb.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            m = (m > 0.5).astype(np.uint8) * 255
                masks.append(m)
            return masks
        except Exception as exc:
            log.debug(f"  [SAM] segment fail: {exc}")
            return [None] * len(bboxes)

    def count_below_water(
        self,
        img_rgb: np.ndarray,
        bbox: List[int],
        water_line_y: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Đếm pixel object THỰC (qua mask) dưới/trên water line.

        Returns:
            (pixels_below, pixels_above) hoặc (None, None) nếu không mask được.
        """
        masks = self.segment(img_rgb, [bbox])
        m = masks[0]
        if m is None or not np.any(m):
            return None, None
        below = int(((m > 0) & (np.arange(m.shape[0])[:, None] >= water_line_y)).sum())
        total = int((m > 0).sum())
        return below, max(0, total - below)


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[SAMSegmentor] = None
_inst_lock = threading.Lock()


def get_sam_segmentor(cfg: Optional[dict] = None) -> Optional[SAMSegmentor]:
    """
    Lấy singleton. Trả về None nếu bị tắt trong config:
        models.use_sam: false  (hoặc thiếu key)
    """
    global _instance
    mcfg = (cfg or {}).get("models", {}) or {}
    if not mcfg.get("use_sam", False):
        return None
    if _instance is None:
        with _inst_lock:
            if _instance is None:
                _instance = SAMSegmentor(cfg)
    return _instance


def refine_submersion_pixels(
    img_rgb: np.ndarray,
    det: dict,
    effective_y2: int,
    effective_wl: int,
    y1: int,
    bbox_sub_px: int,
    cfg: Optional[dict] = None,
) -> Tuple[int, float]:
    """
    Hook chính cho reference_estimator: tinh chỉnh sub_px bằng SAM2 mask.

    Args:
        img_rgb:        ảnh gốc
        det:            detection dict ("bbox" bắt buộc)
        effective_y2:   chân object sau hiệu chỉnh (foot correction)
        effective_wl:   local/global water line (pixel y)
        y1:             đầu trên bbox
        bbox_sub_px:    sub_px tính từ bbox (fallback khi SAM fail)
        cfg:            config toàn cục

    Returns:
        (sub_px_final, sam_conf) — sam_conf=0 nghĩa là không dùng SAM.
        Khi có mask: confidence 0.85 (mask chính xác hơn bbox rõ rệt).
    """
    seg = get_sam_segmentor(cfg)
    if seg is None:
        return bbox_sub_px, 0.0

    x1b, y1b, x2b, _ = det.get("bbox", [0, 0, 0, 0])
    below, above = seg.count_below_water(
        img_rgb, [x1b, y1b, x2b, effective_y2], effective_wl,
    )
    if below is None or above is None:
        return bbox_sub_px, 0.0

    total_mask = below + above
    bh_bbox = max(effective_y2 - y1, 1)
    if total_mask < bh_bbox * 0.15:
        # Mask quá nhỏ (SAM miss) → tin bbox hơn
        return bbox_sub_px, 0.0

    # Clamp an toàn
    sub_px = max(0, min(below, bh_bbox))
    return sub_px, 0.85
