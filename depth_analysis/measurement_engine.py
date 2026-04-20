# -*- coding: utf-8 -*-
"""
depth_analysis/measurement_engine.py
--------------------------------------
Engine do muc nuoc chinh xac bang Confidence-Weighted Voting.

Thay vi chi dung 1 phuong phap, ket hop nhieu nguon:
  1. YOLO reference objects (co pose correction)
  2. Water detector (color + texture)
  3. Depth Anything V2 (relative depth)
  4. Perspective-corrected estimates
  5. Flood classifier (DINOv2) prior

Moi nguon co confidence rieng, vote co trong so.
Final result = weighted median (robust hon mean).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from utils.constants import FLOOD_LEVEL_KNEE, FLOOD_LEVEL_HIP, FLOOD_LEVEL_CHEST, FLOOD_LEVEL_COMPLETE
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MeasurementVote:
    source:      str     # "yolo_person", "yolo_car", "depth_map", "color_water", "dino"
    water_cm:    float
    confidence:  float   # 0-1
    notes:       str = ""


@dataclass
class FinalMeasurement:
    water_cm:        float
    water_cm_low:    float   # khoang duoi (25th percentile)
    water_cm_high:   float   # khoang tren (75th percentile)
    flood_level:     str
    flood_level_desc:str
    confidence:      float
    votes:           List[MeasurementVote] = field(default_factory=list)
    dominant_source: str = ""
    notes:           str = ""


# Flood level thresholds (cm)
FLOOD_LEVELS = [
    (0,   0,    "NO_FLOOD",  "Khong co nuoc lu"),
    (0,   15,   "PUDDLE",    "Vung nuoc nho (< 15cm)"),
    (15,  40,   "ANKLE",     "Ngap mat ca chan (15-40cm)"),
    (40,  70,   "KNEE",      FLOOD_LEVEL_KNEE),
    (70,  120,  "WAIST",     FLOOD_LEVEL_HIP),
    (120, 200,  "CHEST",     FLOOD_LEVEL_CHEST),
    (200, 9999, "SUBMERGED", FLOOD_LEVEL_COMPLETE),
]


def classify_level(water_cm: float) -> Tuple[str, str]:
    for lo, hi, level, desc in FLOOD_LEVELS:
        if lo <= water_cm < hi:
            return level, desc
    return "SUBMERGED", "Ngap hoan toan (> 200cm)"


class MeasurementEngine:
    """
    Ket hop nhieu nguon do de tinh muc nuoc chinh xac.
    """

    def __init__(self, min_confidence: float = 0.25):
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    def measure(
        self,
        yolo_objects:    List[dict],      # tu reference_estimator
        water_result,                     # WaterDetectionResult
        depth_norm:      np.ndarray,      # depth map normalized 0-1
        perspective,                      # PerspectiveResult
        flood_prob:      float = 0.5,     # tu DINOv2 classifier
        img_h:           int   = 0,
    ) -> FinalMeasurement:
        """
        Ket hop tat ca nguon, tra ve FinalMeasurement.
        """
        votes: List[MeasurementVote] = []

        # === SOURCE 1: YOLO objects ===
        for obj in yolo_objects:
            vote = self._vote_from_yolo(obj, perspective, img_h)
            if vote:
                votes.append(vote)

        # === SOURCE 2: Water color detector ===
        # Hạ ngưỡng xuống 1.5% để bắt được vũng nước nhỏ
        if water_result and water_result.water_area_pct > 1.5:
            vote = self._vote_from_color(water_result, img_h)
            if vote:
                votes.append(vote)

        # === SOURCE 2b: Puddle vote (vũng nước nhỏ riêng biệt) ===
        if water_result and getattr(water_result, "has_puddle", False):
            vote = self._vote_from_puddle(water_result)
            if vote:
                votes.append(vote)

        # === SOURCE 3: Depth map ===
        if depth_norm is not None and depth_norm.size > 0:
            vote = self._vote_from_depth(depth_norm, water_result, img_h)
            if vote:
                votes.append(vote)

        # === SOURCE 4: DINOv2 prior ===
        vote = self._vote_from_dino(flood_prob, water_result)
        if vote:
            votes.append(vote)

        # === Loai bo votes co confidence qua thap ===
        valid_votes = [v for v in votes if v.confidence >= self.min_confidence]

        if not valid_votes:
            return FinalMeasurement(
                water_cm=0, water_cm_low=0, water_cm_high=0,
                flood_level="UNKNOWN", flood_level_desc="Không đủ dữ liệu",
                confidence=0.0, votes=votes,
            )

        # === Uu tien theo thu tu reference ===
        # Nguyen tac: cua_nha > xe_may > nguoi > color/depth
        # Neu co reference cao cap (door), giam trong so cua cap thap hon
        has_door = any("cua_nha" in v.source for v in valid_votes)
        has_vehicle = any(
            any(x in v.source for x in ("motorcycle", "bicycle", "car", "truck", "bus"))
            for v in valid_votes
        )
        if has_door:
            # Co cua nha: giam trong so cua person va non-reference sources
            for v in valid_votes:
                if "person" in v.source or v.source in ("color_water", "depth_map", "dino"):
                    v.confidence *= 0.60
        elif has_vehicle:
            # Co xe (khong co cua): giam trong so depth/dino (kem chinh xac hon)
            for v in valid_votes:
                if v.source in ("depth_map", "dino"):
                    v.confidence *= 0.70

        # === Weighted median voting (robust hon mean, loai outliers) ===
        water_cm, ci_low, ci_high, dominant = self._weighted_median(valid_votes)

        level, desc = classify_level(water_cm)
        confidence  = self._calc_final_confidence(valid_votes, water_cm)

        # Bonus confidence neu nhieu sources dong thuan
        n_agree = sum(
            1 for v in valid_votes
            if abs(v.water_cm - water_cm) < max(10, water_cm * 0.3)
        )
        if n_agree >= 2:
            confidence = min(1.0, confidence + 0.1 * (n_agree - 1))

        log.info(
            f"  Measurement: {water_cm:.0f}cm [{ci_low:.0f}-{ci_high:.0f}] "
            f"| {level} | conf={confidence:.2f} "
            f"| {len(valid_votes)} votes | dominant={dominant}"
        )

        return FinalMeasurement(
            water_cm         = round(water_cm, 1),
            water_cm_low     = round(ci_low, 1),
            water_cm_high    = round(ci_high, 1),
            flood_level      = level,
            flood_level_desc = desc,
            confidence       = round(confidence, 3),
            votes            = valid_votes,
            dominant_source  = dominant,
            notes            = self._generate_notes(valid_votes, water_cm, level),
        )

    # ------------------------------------------------------------------
    def _vote_from_yolo(
        self, obj: dict, perspective, img_h: int
    ) -> Optional[MeasurementVote]:
        """Vote tu 1 YOLO object."""
        water_cm = obj.get("water_height_cm", 0)
        conf     = obj.get("confidence", 0) * obj.get("pose_factor", 1.0)
        cls      = obj.get("class_name", "unknown")

        if water_cm < 0 or conf < 0.2:
            return None

        # Perspective correction
        if perspective and perspective.view_angle != "eye_level":
            from depth_analysis.perspective_analyzer import apply_perspective_correction
            water_cm, _ = apply_perspective_correction(water_cm, obj, perspective, img_h)

        # Giam trust neu chan kho
        if obj.get("foot_is_dry", False):
            conf *= 0.5

        # Giam trust nhe neu la ao mua — nguoi van la reference hop le,
        # chi giam nhe vi ao mua co the che khuat mot so diem keypoint
        if obj.get("rain_coat_prob", 0) > 0.5:
            conf *= 0.85

        source = f"yolo_{cls}"
        return MeasurementVote(
            source    = source,
            water_cm  = round(max(0, water_cm), 1),
            confidence= round(min(1.0, conf), 3),
            notes     = f"YOLO {cls} | pose={obj.get('pose_type','?')}",
        )

    def _vote_from_color(self, water_result, img_h: int) -> Optional[MeasurementVote]:
        """
        Vote tu color water detection.
        Uoc tinh muc nuoc tu % dien tich va vi tri water line.
        """
        if water_result is None:
            return None

        wl_y  = water_result.water_line_y
        pct   = water_result.water_area_pct / 100.0

        if img_h <= 0 or pct < 0.015:
            return None

        # Uoc tinh muc nuoc tu vi tri water line:
        # (img_h - wl_y) / img_h = ti le phan duoi la nuoc
        # Gia su nguoi trung binh chiem 75% chieu cao anh => 170 * 0.75 = 127.5cm
        water_depth_ratio = max(0.0, (img_h - wl_y) / max(img_h, 1))
        est_cm = water_depth_ratio * 170.0 * 0.75
        est_cm = max(0, min(300, est_cm))

        # Confidence tu water detector
        conf = water_result.confidence * 0.7   # color alone = 70% max

        return MeasurementVote(
            source    = "color_water",
            water_cm  = round(est_cm, 1),
            confidence= round(conf, 3),
            notes     = f"Color: {water_result.water_type} | area={water_result.water_area_pct:.1f}%",
        )

    def _vote_from_puddle(self, water_result) -> Optional[MeasurementVote]:
        """
        Vote chuyên biệt cho vũng nước nhỏ (PUDDLE, 0-15 cm).
        Dùng khi has_puddle=True: ưu tiên ước tính mức nước thấp.

        Logic: vũng nước thường 3-12 cm, dùng puddle_area_pct để scale.
        """
        puddle_pct = getattr(water_result, "puddle_area_pct", 0.0) / 100.0
        if puddle_pct <= 0:
            puddle_pct = 0.005  # fallback nếu area không có

        # Vũng càng lớn → khả năng sâu hơn, nhưng cap ở 14 cm (vẫn là PUDDLE)
        est_cm = min(14.0, max(3.0, puddle_pct * 300))
        conf   = min(0.55, 0.30 + puddle_pct * 5.0)

        return MeasurementVote(
            source    = "puddle_detector",
            water_cm  = round(est_cm, 1),
            confidence= round(conf, 3),
            notes     = f"Puddle: area={puddle_pct*100:.1f}%, est={est_cm:.1f}cm",
        )

    def _vote_from_depth(
        self, depth_norm: np.ndarray, water_result, img_h: int
    ) -> Optional[MeasurementVote]:
        """
        Vote tu depth map.
        Depth map cho biet cau truc 3D, ho tro uoc tinh muc nuoc.
        """
        h, w = depth_norm.shape

        # Lay vung nuoi duoi (50% duoi anh - thuong la nuoc)
        lower = depth_norm[h // 2:, :]

        # Phân tích depth distribution cua vung duoi
        # Vung nuoc: depth cao (gan camera), texture thap
        water_thresh = np.percentile(lower, 70)
        water_pct_d  = float((lower > water_thresh).mean())

        if water_pct_d < 0.1:
            return None

        # Uoc tinh muc nuoc tu depth (rat rough)
        est_cm = 170.0 * water_pct_d * 0.5
        est_cm = max(0, min(250, est_cm))

        conf = min(0.5, water_pct_d * 0.8)   # depth alone = 50% max

        # Neu co water mask, tang confidence
        if water_result and water_result.water_area_pct > 5:
            conf = min(0.65, conf + 0.15)

        return MeasurementVote(
            source    = "depth_map",
            water_cm  = round(est_cm, 1),
            confidence= round(conf, 3),
            notes     = f"Depth: water_pct={water_pct_d:.1%}",
        )

    def _vote_from_dino(
        self, flood_prob: float, water_result
    ) -> Optional[MeasurementVote]:
        """
        Vote tu DINOv2 flood classifier.
        DINOv2 cho biet co lu khong nhung khong biet muc do.
        Dung nhu prior: neu DINOv2 noi FLOOD thi co it nhat PUDDLE.
        """
        if flood_prob < 0.4:
            return None

        water_pct = (water_result.water_area_pct / 100.0) if water_result else 0.05

        # DINOv2 prior: neu co lu thi it nhat 5cm
        # Muc do dua tren water_area_pct + flood_prob
        est_cm = max(5.0, water_pct * 200 * flood_prob)
        est_cm = min(est_cm, 100.0)   # DINOv2 khong biet muc nuoc cu the

        conf = flood_prob * 0.4   # DINOv2 prior = 40% max

        return MeasurementVote(
            source    = "dino_prior",
            water_cm  = round(est_cm, 1),
            confidence= round(conf, 3),
            notes     = f"DINOv2 flood_prob={flood_prob:.2f}",
        )

    # ------------------------------------------------------------------
    def _weighted_median(
        self, votes: List[MeasurementVote]
    ) -> Tuple[float, float, float, str]:
        """
        Tinh weighted median de robust voi outliers.
        Tra ve (median, low, high, dominant_source).
        """
        if not votes:
            return 0.0, 0.0, 0.0, "none"

        # Sort theo water_cm
        sorted_votes = sorted(votes, key=lambda v: v.water_cm)
        total_weight = sum(v.confidence for v in sorted_votes)

        if total_weight == 0:
            vals = [v.water_cm for v in sorted_votes]
            return float(np.median(vals)), float(np.percentile(vals, 25)), float(np.percentile(vals, 75)), "none"

        # Weighted median
        cumulative = 0.0
        median_cm  = sorted_votes[-1].water_cm
        for v in sorted_votes:
            cumulative += v.confidence / total_weight
            if cumulative >= 0.5:
                median_cm = v.water_cm
                break

        # Confidence interval (25th and 75th weighted percentile)
        all_cm = [v.water_cm for v in votes]
        ci_low  = float(np.percentile(all_cm, 25))
        ci_high = float(np.percentile(all_cm, 75))

        # Dominant source = source co confidence cao nhat
        dominant = max(votes, key=lambda v: v.confidence).source

        return median_cm, ci_low, ci_high, dominant

    def _calc_final_confidence(
        self, votes: List[MeasurementVote], water_cm: float
    ) -> float:
        """Tinh confidence tong the."""
        if not votes:
            return 0.0

        # Mean confidence cua cac votes
        mean_conf = np.mean([v.confidence for v in votes])

        # Penalty neu cac votes qua phan tan
        all_cm = [v.water_cm for v in votes]
        if len(all_cm) > 1:
            cv = np.std(all_cm) / max(np.mean(all_cm), 1)
            dispersion_penalty = min(0.3, cv * 0.2)
        else:
            dispersion_penalty = 0.1   # 1 nguon thoi = penalty

        return float(max(0.1, float(mean_conf) - float(dispersion_penalty)))

    def _generate_notes(
        self, votes: List[MeasurementVote], water_cm: float, level: str
    ) -> str:
        sources = ", ".join(set(v.source for v in votes))
        return (
            f"Ket hop {len(votes)} nguon: {sources}. "
            f"Mức nước: {water_cm:.0f}cm ({level})."
        )
