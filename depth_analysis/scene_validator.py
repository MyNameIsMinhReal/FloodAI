# -*- coding: utf-8 -*-
"""
depth_analysis/scene_validator.py
===================================
SceneValidator — kiểm tra kết quả phân tích có logic/nhất quán không.

Không phải mọi ảnh đều có lũ thật. Cần phân biệt:
  - Đường ướt sau mưa (water_pct nhỏ, không chạm ground nhiều)
  - Bóng đổ từ cây/nhà (overlap cao với shadow)
  - Phản chiếu mặt kính/tường bóng
  - Lũ thật (water chạm ground, chạm người/cửa, area lớn)

Output:
    ValidationResult(
        scene_score:  float (0–1) — độ tin cậy scene có ngập thật
        reasons:      List[str]  — điểm cộng
        warnings:     List[str]  — vấn đề phát hiện
        scene_type:   str        — "flood" | "wet_road" | "reflection" | "dry" | "uncertain"
    )

Dùng:
    sv = SceneValidator()
    val = sv.validate(water_mask=..., road_mask=..., shadow_mask=..., ...)
    if val.scene_type == "flood":
        ...
    elif val.scene_type == "wet_road":
        # hạ confidence, đừng báo ngập
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import cv2
import numpy as np

log = logging.getLogger("depth_analysis.scene_validator")

# Scene types
SCENE_FLOOD     = "flood"
SCENE_WET_ROAD  = "wet_road"
SCENE_REFLECTION= "reflection"
SCENE_DRY       = "dry"
SCENE_UNCERTAIN = "uncertain"


@dataclass
class ValidationResult:
    scene_score: float = 0.5         # 0 = chắc không ngập, 1 = chắc ngập
    reasons:     List[str] = field(default_factory=list)    # điểm cộng
    warnings:    List[str] = field(default_factory=list)    # vấn đề
    scene_type:  str = SCENE_UNCERTAIN

    def is_flood(self) -> bool:
        return self.scene_type == SCENE_FLOOD

    def confidence_penalty(self) -> float:
        """Số trừ thêm vào confidence nếu scene không nhất quán."""
        if self.scene_type in (SCENE_WET_ROAD, SCENE_REFLECTION):
            return 0.20
        if self.scene_type == SCENE_DRY:
            return 0.40
        if self.scene_type == SCENE_UNCERTAIN:
            return 0.05
        return 0.0


class SceneValidator:
    """
    Kiểm tra tính nhất quán của scene.

    Dùng:
        sv = SceneValidator(cfg)
        val = sv.validate(
            water_mask=water_mask,
            road_mask=road_mask,
            building_mask=bseg.building_mask,
            shadow_mask=shadow_mask,
            person_count=2,
            vehicle_count=1,
        )
        confidence -= val.confidence_penalty()
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.min_water_pct        = cfg.get("min_water_pct",       0.01)
        self.min_ground_overlap   = cfg.get("min_ground_overlap",  0.08)
        self.max_shadow_overlap   = cfg.get("max_shadow_overlap",  0.40)
        self.max_wall_overlap     = cfg.get("max_wall_overlap",    0.35)
        self.min_texture_variance = cfg.get("min_texture_var",     15.0)

    # ── Public ────────────────────────────────────────────────────────────────

    def validate(
        self,
        water_mask: np.ndarray,
        road_mask:  Optional[np.ndarray] = None,
        building_mask: Optional[np.ndarray] = None,
        shadow_mask:   Optional[np.ndarray] = None,
        img_bgr:       Optional[np.ndarray] = None,
        person_count:  int = 0,
        vehicle_count: int = 0,
        door_count:    int = 0,
    ) -> ValidationResult:
        """
        Đánh giá tính hợp lý của scene.

        Args:
            water_mask:     mask nước (0/255)
            road_mask:      mask đường/mặt đất (0/255) — optional
            building_mask:  mask nhà/tường (0/255) — optional
            shadow_mask:    mask bóng đổ (0/255) — optional
            img_bgr:        ảnh gốc BGR — dùng để kiểm tra texture
            person_count:   số người phát hiện được
            vehicle_count:  số xe phát hiện được
            door_count:     số cửa hợp lệ

        Returns:
            ValidationResult
        """
        h, w = water_mask.shape[:2]
        total_px = h * w
        water_px = (water_mask > 0).sum()
        water_pct = float(water_px) / total_px

        score = 1.0
        reasons: List[str] = []
        warnings: List[str] = []

        # ── Rule 1: Diện tích nước ──────────────────────────────────────────
        if water_pct < self.min_water_pct:
            score -= 0.40
            warnings.append(f"Vùng nước quá nhỏ ({water_pct:.1%} < {self.min_water_pct:.1%})")
        elif water_pct > 0.05:
            reasons.append(f"✓ Vùng nước đáng kể ({water_pct:.1%})")

        # ── Rule 2: Nước phải chạm mặt đất ────────────────────────────────
        if road_mask is not None:
            road_water_overlap = (
                (water_mask > 0) & (road_mask > 0)
            ).sum() / max(water_px, 1)
            if road_water_overlap < self.min_ground_overlap:
                score -= 0.20
                warnings.append("Nước không nằm trên mặt đường/đất — có thể reflection")
            else:
                reasons.append(f"✓ Nước phủ trên mặt đường ({road_water_overlap:.0%})")

        # ── Rule 3: Nước trên tường → nhiều khả năng là reflection ────────
        if building_mask is not None and water_px > 0:
            wall_water = (
                (water_mask > 0) & (building_mask > 0)
            ).sum() / max(water_px, 1)
            if wall_water > self.max_wall_overlap:
                score -= 0.25
                warnings.append(f"⚠ {wall_water:.0%} nước nằm trên tường — có thể phản chiếu/biển quảng cáo")

        # ── Rule 4: Shadow overlap ──────────────────────────────────────────
        if shadow_mask is not None and water_px > 0:
            shadow_overlap = (
                (water_mask > 0) & (shadow_mask > 0)
            ).sum() / max(water_px, 1)
            if shadow_overlap > self.max_shadow_overlap:
                score -= 0.30
                warnings.append(f"❌ {shadow_overlap:.0%} vùng nước bị che bởi bóng đổ")
            elif shadow_overlap < 0.10:
                reasons.append("✓ Ít overlap với bóng đổ")

        # ── Rule 5: Reference objects ──────────────────────────────────────
        if person_count >= 1:
            reasons.append(f"✓ {person_count} người làm vật tham chiếu")
            score = min(score + 0.05, 1.0)
        if vehicle_count >= 1:
            reasons.append(f"✓ {vehicle_count} xe làm vật tham chiếu")
        if door_count >= 1:
            reasons.append(f"✓ {door_count} cửa làm vật tham chiếu (mạnh nhất)")
            score = min(score + 0.08, 1.0)

        # ── Rule 6: Texture variance (nước thật có texture) ────────────────
        if img_bgr is not None and water_px > 100:
            water_region = img_bgr.copy()
            water_region[water_mask == 0] = 0
            gray_water = cv2.cvtColor(water_region, cv2.COLOR_BGR2GRAY)
            masked_pixels = gray_water[water_mask > 0]
            if len(masked_pixels) > 50:
                texture_var = float(np.var(masked_pixels.astype(np.float32)))
                if texture_var < self.min_texture_variance:
                    score -= 0.10
                    warnings.append("⚠ Vùng nước thiếu texture — có thể là bề mặt bóng/kính")

        # ── Determine scene type ────────────────────────────────────────────
        score = max(0.0, min(1.0, score))

        if score >= 0.70:
            scene_type = SCENE_FLOOD
        elif score >= 0.45:
            scene_type = SCENE_UNCERTAIN
        elif water_pct > 0.01 and any("mặt đường" in w for w in warnings):
            scene_type = SCENE_WET_ROAD
        elif any("phản chiếu" in w or "tường" in w for w in warnings):
            scene_type = SCENE_REFLECTION
        else:
            scene_type = SCENE_DRY

        if not reasons:
            reasons.append("— Không đủ dữ liệu để xác nhận lũ")

        return ValidationResult(
            scene_score=round(score, 3),
            reasons=reasons,
            warnings=warnings,
            scene_type=scene_type,
        )

    def validate_water(
        self,
        water_mask: np.ndarray,
        road_mask: Optional[np.ndarray] = None,
        building_mask: Optional[np.ndarray] = None,
        shadow_mask: Optional[np.ndarray] = None,
    ):
        """Alias ngắn gọn cho validate() — chỉ nhận mask, không nhận img/counts."""
        return self.validate(
            water_mask=water_mask,
            road_mask=road_mask,
            building_mask=building_mask,
            shadow_mask=shadow_mask,
        )
