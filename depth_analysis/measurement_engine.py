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
from typing import Any, List, Tuple, Optional

from utils.constants import FLOOD_LEVEL_KNEE, FLOOD_LEVEL_HIP, FLOOD_LEVEL_CHEST, FLOOD_LEVEL_COMPLETE
from utils.constants import classify_level
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
    # ── [v4] Disagreement tracking — dùng để đẩy vào review queue ──
    needs_review:    bool = False
    review_reason:   str  = ""
    method_spread_cm: float = 0.0   # p90 - p10 của các votes


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

    Fusion modes (config: measurement.fusion):
        "bayesian" — inverse-variance weighted mean + MAD outlier rejection
                     (mặc định, chính xác hơn khi có ≥2 nguồn độc lập)
        "median"   — weighted median cũ (robust, giữ lại để so sánh)
    """

    def __init__(self, min_confidence: float = 0.25, fusion_mode: str = "bayesian"):
        self.min_confidence = min_confidence
        self.fusion_mode = fusion_mode if fusion_mode in ("bayesian", "median") else "bayesian"

    # ------------------------------------------------------------------
    def _collect_sensor_votes(
        self, water_result, depth_norm: np.ndarray, flood_prob: float, img_h: int
    ) -> List[MeasurementVote]:
        votes: List[MeasurementVote] = []
        if water_result and water_result.water_area_pct > 1.5:
            v = self._vote_from_color(water_result, img_h)
            if v:
                votes.append(v)
        if water_result and getattr(water_result, "has_puddle", False):
            v = self._vote_from_puddle(water_result)
            if v:
                votes.append(v)
        if depth_norm is not None and depth_norm.size > 0:
            v = self._vote_from_depth(depth_norm, water_result, img_h)
            if v:
                votes.append(v)
        v = self._vote_from_dino(flood_prob, water_result)
        if v:
            votes.append(v)
        return votes

    def _collect_votes(
        self,
        yolo_objects: List[dict],
        water_result,
        depth_norm: np.ndarray,
        perspective,
        flood_prob: float,
        img_h: int,
    ) -> List[MeasurementVote]:
        votes: List[MeasurementVote] = []
        for obj in yolo_objects:
            v = self._vote_from_yolo(obj, perspective, img_h)
            if v:
                votes.append(v)
        votes += self._collect_sensor_votes(water_result, depth_norm, flood_prob, img_h)
        return votes

    def measure(
        self,
        yolo_objects:    List[dict],
        water_result,
        depth_norm:      np.ndarray,
        perspective,
        flood_prob:      float = 0.5,
        img_h:           int   = 0,
    ) -> FinalMeasurement:
        votes      = self._collect_votes(yolo_objects, water_result, depth_norm,
                                         perspective, flood_prob, img_h)
        valid_votes = [v for v in votes if v.confidence >= self.min_confidence]

        if not valid_votes:
            return FinalMeasurement(
                water_cm=0, water_cm_low=0, water_cm_high=0,
                flood_level="UNKNOWN", flood_level_desc="Không đủ dữ liệu",
                confidence=0.0, votes=votes,
            )

        self._apply_priority_weights(valid_votes)

        # === Fusion: Bayesian (mặc định) hoặc weighted median ===
        if self.fusion_mode == "bayesian" and len(valid_votes) >= 2:
            water_cm, ci_low, ci_high, dominant = self._bayesian_fusion(valid_votes)
        else:
            water_cm, ci_low, ci_high, dominant = self._weighted_median(valid_votes)

        level, desc = classify_level(water_cm)
        confidence  = self._calc_final_confidence(valid_votes, water_cm)
        confidence  = self._bonus_agreement(valid_votes, water_cm, confidence)

        # ── [v4] Disagreement → flag cho review queue ──────────────────────
        spread = self._vote_spread(valid_votes)
        needs_review, review_reason = False, ""
        if len(valid_votes) >= 2:
            spread_thresh = max(20.0, water_cm * 0.35)
            if spread > spread_thresh:
                needs_review = True
                review_reason = "method_disagreement"
                confidence *= 0.85   # phạt nhẹ khi bất đồng

        log.info(
            f"  Measurement: {water_cm:.0f}cm [{ci_low:.0f}-{ci_high:.0f}] "
            f"| {level} | conf={confidence:.2f} "
            f"| {len(valid_votes)} votes | dominant={dominant} "
            f"| fusion={self.fusion_mode}"
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
            needs_review     = needs_review,
            review_reason    = review_reason,
            method_spread_cm = round(spread, 1),
        )

    # ------------------------------------------------------------------
    _PRIORITY_MAP = [
        # (keyword_in_source, scale, weak_sources_to_suppress)
        ("cua_nha", 0.60, frozenset({"person", "color_water", "depth_map", "dino_prior"})),
        ("person",  0.45, frozenset({"color_water", "depth_map", "dino_prior"})),
    ]
    _VEHICLE_CLASSES = frozenset({"motorcycle", "bicycle", "car", "truck", "bus"})

    def _detect_priority(self, sources: set) -> Tuple[float, frozenset]:
        for key, scale, weak in self._PRIORITY_MAP:
            if any(key in s for s in sources):
                return scale, weak
        if any(cls in s for s in sources for cls in self._VEHICLE_CLASSES):
            return 0.70, frozenset({"depth_map", "dino_prior"})
        return 1.0, frozenset()

    def _apply_priority_weights(self, valid_votes: List[MeasurementVote]) -> None:
        """Giảm trọng số các nguồn thô khi có reference tốt hơn."""
        sources = {v.source for v in valid_votes}
        scale, weak = self._detect_priority(sources)
        for v in valid_votes:
            if v.source in weak:
                v.confidence *= scale

    def _bonus_agreement(
        self, valid_votes: List[MeasurementVote], water_cm: float, confidence: float
    ) -> float:
        """Tăng confidence nếu nhiều nguồn đồng thuận."""
        n_agree = sum(
            1 for v in valid_votes
            if abs(v.water_cm - water_cm) < max(10, water_cm * 0.3)
        )
        if n_agree >= 2:
            confidence = min(1.0, confidence + 0.1 * (n_agree - 1))
        return confidence

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

        # water_line_y là ranh giới mặt nước TỪ TRÊN. Với ảnh đường phố ngập nước
        # nông (9-30cm), nước lan rộng trên mặt đất → water_line_y có thể ở giữa
        # ảnh dù nước rất cạn. Giới hạn est_cm ở 50cm để tránh overestimate.
        water_depth_ratio = max(0.0, (img_h - wl_y) / max(img_h, 1))
        est_cm = water_depth_ratio * 50.0
        est_cm = max(0, min(50, est_cm))

        # Confidence thấp — color không đủ thông tin để ước lượng độ sâu chính xác
        conf = water_result.confidence * 0.40

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

        # Depth map rất rough — không thể phân biệt nước nông lan rộng vs nước sâu
        est_cm = 50.0 * water_pct_d
        est_cm = max(0, min(60, est_cm))

        conf = min(0.30, water_pct_d * 0.5)

        if water_result and water_result.water_area_pct > 5:
            conf = min(0.40, conf + 0.10)

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

        # DINOv2 chỉ biết "có lũ" hay không, không biết độ sâu — dùng làm prior nhẹ
        est_cm = max(5.0, water_pct * 50 * flood_prob)
        est_cm = min(est_cm, 40.0)

        conf = flood_prob * 0.4   # DINOv2 prior = 40% max

        return MeasurementVote(
            source    = "dino_prior",
            water_cm  = round(est_cm, 1),
            confidence= round(conf, 3),
            notes     = f"DINOv2 flood_prob={flood_prob:.2f}",
        )

    # ------------------------------------------------------------------
    def _vote_spread(self, votes: List[MeasurementVote]) -> float:
        """Độ phân tán p90-p10 của các votes (cm)."""
        if len(votes) < 2:
            return 0.0
        vals = np.array([v.water_cm for v in votes], dtype=np.float64)
        return float(np.percentile(vals, 90) - np.percentile(vals, 10))

    def _bayesian_fusion(
        self, votes: List[MeasurementVote]
    ) -> Tuple[float, float, float, str]:
        """
        Inverse-variance weighted mean với MAD outlier rejection.

        Mỗi nguồn có σ riêng suy từ confidence: nguồn tin cậy cao → σ nhỏ
        → trọng số lớn. Outlier (lệch cụm > 3×MAD) bị loại trước khi gộp.

        Returns:
            (fused_cm, ci_low, ci_high, dominant_source)
        """
        vals = np.array([v.water_cm for v in votes], dtype=np.float64)

        # ── Bước 1: MAD outlier rejection ──────────────────────────────
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        # MAD=1.4826 ≈ std của phân phối chuẩn; floor để không chia 0
        sigma_mad = max(1.4826 * mad, 5.0)
        keep_idx = [
            i for i, v in enumerate(votes)
            if abs(v.water_cm - med) <= 3.0 * sigma_mad
        ]
        # Giữ tối thiểu 1 vote (nếu tất cả đều là "outlier" thì giữ hết)
        kept = [votes[i] for i in keep_idx] or votes

        # ── Bước 2: inverse-variance weights ───────────────────────────
        # σ_i = base × (1 − conf) + floor  → conf=1 → σ=8cm; conf=0.3 → σ≈26cm
        weights, weighted_sum = [], 0.0
        for v in kept:
            sigma = 8.0 * (1.0 - v.confidence) + 4.0
            w = (v.confidence ** 2) / (sigma ** 2)
            weights.append(w)
            weighted_sum += w * v.water_cm

        total_w = sum(weights)
        fused = weighted_sum / total_w if total_w > 0 else med

        # ── Bước 3: CI từ weighted std ─────────────────────────────────
        kvals = np.array([v.water_cm for v in kept], dtype=np.float64)
        if len(kept) > 1 and total_w > 0:
            var = float(np.sum(weights * (kvals - fused) ** 2) / total_w)
            ci_half = max(1.96 * np.sqrt(max(var, 0.0)), 4.0)
        else:
            ci_half = 10.0

        dominant = max(kept, key=lambda v: v.confidence).source
        return (
            round(float(fused), 1),
            round(float(fused - ci_half), 1),
            round(float(fused + ci_half), 1),
            dominant,
        )

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


# ── [v4] Disagreement assessment cho postprocess/review queue ─────────────────

def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def assess_disagreement(result) -> Tuple[bool, float, str]:
    """
    Kiểm tra bất đồng giữa kết quả cuối và các reference objects.

    Dùng khi result KHÔNG lưu FinalMeasurement (vd ReferenceFloodResult):
    so sánh water_height_cm cuối với từng object đo được. Nếu ≥1 reference
    mạnh (confidence cao) lệch quá 30% so với kết quả cuối → flag review.

    Args:
        result: ReferenceFloodResult / dict có detected_objects,
                water_height_cm

    Returns:
        (needs_review, max_dev_pct, reason)
    """
    final_cm = _get(result, "water_height_cm", None)
    objects = _get(result, "detected_objects", []) or []
    if final_cm is None or not objects or final_cm <= 0:
        return False, 0.0, ""

    max_dev, n_strong = 0.0, 0
    for o in objects:
        if _get(o, "skip_measure", False):
            continue
        obj_cm = _get(o, "water_height_cm", 0.0)
        conf   = float(_get(o, "confidence", 0.0))
        if not obj_cm or obj_cm <= 0 or conf < 0.5:
            continue
        n_strong += 1
        dev = abs(obj_cm - final_cm) / max(final_cm, 1.0)
        # Object mạnh mà nói ngập nhiều hơn hẳn kết quả cuối → đáng ngờ nhất
        if obj_cm > final_cm * 1.2 and dev > max_dev:
            max_dev = dev

    if n_strong >= 1 and max_dev > 0.30:
        return True, round(max_dev * 100, 1), "method_disagreement"
    return False, round(max_dev * 100, 1), ""
