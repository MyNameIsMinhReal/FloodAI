# -*- coding: utf-8 -*-
"""
pipeline/uncertainty.py
========================
Uncertainty handling — thay vì lúc nào cũng đưa kết quả chắc chắn, hệ thống
phân loại rõ:

    HIGH confidence  (≥ 0.70)  → accept:       dùng kết quả trực tiếp
    MED  confidence  (0.40–0.69)→ needs_review: đưa vào review queue
    LOW  confidence  (< 0.40)  → reject:       từ chối, gắn cờ cảnh báo

Output mẫu:
    {
      "flood_detected": true,
      "level": "knee",
      "depth_cm": 52,
      "confidence": 0.74,
      "decision": "accept",
      "reasons": ["Có người làm vật tham chiếu", "Water mask rõ"],
      "warnings": ["Ảnh hơi tối"],
    }
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("pipeline.uncertainty")

# ── Thresholds (có thể override qua cfg) ──────────────────────────────────────
ACCEPT_THRESHOLD = 0.70
REJECT_THRESHOLD = 0.40

DECISION_ACCEPT  = "accept"
DECISION_REVIEW  = "needs_review"
DECISION_REJECT  = "reject"


@dataclass
class UncertaintyReport:
    """Kết quả đánh giá uncertainty cho 1 ảnh."""
    decision: str           # accept / needs_review / reject
    confidence: float
    flood_detected: bool
    level: str
    depth_cm: Optional[float]
    reasons: List[str] = field(default_factory=list)    # lý do tự tin
    warnings: List[str] = field(default_factory=list)   # cờ cảnh báo
    breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "flood_detected": self.flood_detected,
            "level":          self.level,
            "depth_cm":       round(self.depth_cm, 1) if self.depth_cm is not None else None,
            "confidence":     round(self.confidence, 3),
            "decision":       self.decision,
            "reasons":        self.reasons,
            "warnings":       self.warnings,
            "breakdown":      {k: round(v, 3) for k, v in self.breakdown.items()},
        }


class UncertaintyEvaluator:
    """
    Đánh giá mức độ chắc chắn của kết quả phân tích và đưa ra quyết định.

    Dùng:
        ev = UncertaintyEvaluator(cfg)
        report = ev.evaluate(depth_result)
        if report.decision == "accept":
            ...
        elif report.decision == "needs_review":
            queue.add(image_path, report)
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.accept_threshold = cfg.get("accept_threshold", ACCEPT_THRESHOLD)
        self.reject_threshold = cfg.get("reject_threshold", REJECT_THRESHOLD)

    # ── Public ────────────────────────────────────────────────────────────────

    def evaluate(self, result: Any) -> UncertaintyReport:
        """
        Đánh giá uncertainty của một depth_result.

        Args:
            result: dict hoặc object từ DepthStage

        Returns:
            UncertaintyReport
        """
        conf       = self._get(result, "confidence", 0.5)
        level      = self._get(result, "flood_level", "unknown") or "unknown"
        depth_cm   = self._get(result, "depth_cm", None)
        flood      = level not in ("dry", "unknown") and (depth_cm or 0) > 0
        breakdown  = self._extract_breakdown(result)

        reasons, warnings = self._analyze(result, conf, level, depth_cm, breakdown)
        decision = self._decide(conf, reasons, warnings)

        report = UncertaintyReport(
            decision=decision,
            confidence=conf,
            flood_detected=flood,
            level=level,
            depth_cm=depth_cm,
            reasons=reasons,
            warnings=warnings,
            breakdown=breakdown,
        )

        # Gắn report vào result object để downstream dùng
        self._attach(result, report)
        return report

    def evaluate_batch(self, results: List[Any]) -> List[UncertaintyReport]:
        return [self.evaluate(r) for r in results]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _decide(self, conf: float, reasons: List[str], warnings: List[str]) -> str:
        # Hard reject nếu quá nhiều cảnh báo nghiêm trọng
        critical_warnings = [w for w in warnings if "❌" in w]
        if len(critical_warnings) >= 2 or conf < self.reject_threshold:
            return DECISION_REJECT
        if conf >= self.accept_threshold and not critical_warnings:
            return DECISION_ACCEPT
        return DECISION_REVIEW

    def _analyze(
        self,
        result: Any,
        conf: float,
        level: str,
        depth_cm: Optional[float],
        breakdown: Dict[str, float],
    ):
        """Sinh danh sách reasons (điểm mạnh) và warnings (vấn đề)."""
        reasons: List[str]  = []
        warnings: List[str] = []

        ref_objects = self._get(result, "reference_objects", []) or []
        has_person  = any(o.get("class") in ("person", "người") for o in ref_objects if isinstance(o, dict))
        has_vehicle = any(o.get("class") in ("car", "motorcycle", "xe") for o in ref_objects if isinstance(o, dict))
        has_raincoat= self._get(result, "has_raincoat", False)
        water_conf  = breakdown.get("water_detection", breakdown.get("water", 0.0))
        depth_cons  = breakdown.get("depth_consistency", breakdown.get("depth", 0.0))
        img_quality = breakdown.get("image_quality", breakdown.get("quality", 0.0))

        # ── Reasons (điểm tự tin) ──────────────────────────────────────────────
        if has_person:
            reasons.append("✓ Có người làm vật tham chiếu chiều cao")
        if has_vehicle:
            reasons.append("✓ Có phương tiện làm vật tham chiếu")
        if water_conf >= 0.65:
            reasons.append("✓ Water mask được phát hiện rõ ràng")
        if depth_cons >= 0.65:
            reasons.append("✓ Depth map ổn định trong vùng nước")
        if has_raincoat:
            reasons.append("✓ Phát hiện áo mưa — xác nhận môi trường lũ lụt")
        if len(ref_objects) >= 3:
            reasons.append(f"✓ {len(ref_objects)} vật tham chiếu — ước lượng chiều sâu đáng tin")

        # ── Warnings (vấn đề) ──────────────────────────────────────────────────
        if not ref_objects:
            warnings.append("⚠ Không có vật tham chiếu — chiều sâu chỉ là ước tính")
        if water_conf < 0.40:
            warnings.append("❌ Tỉ lệ phát hiện nước thấp — có thể không có lũ thật sự")
        if depth_cons < 0.40:
            warnings.append("❌ Depth map không ổn định ở vùng trung tâm")
        if img_quality < 0.40:
            warnings.append("⚠ Chất lượng ảnh thấp (tối, mờ, nhiễu)")
        if depth_cm is not None and depth_cm > 200:
            warnings.append("⚠ Chiều sâu ước tính > 200cm — có thể không chính xác")
        if level == "unknown":
            warnings.append("❌ Không thể xác định mức lũ")

        # Phản chiếu / glare
        glare = self._get(result, "has_glare", False) or self._get(result, "has_reflection", False)
        if glare:
            warnings.append("⚠ Vùng nước bị phản chiếu mạnh — ảnh hưởng water detection")

        if not reasons:
            reasons.append("— Không đủ dữ liệu để đưa ra lý do chắc chắn")

        return reasons, warnings

    def _extract_breakdown(self, result: Any) -> Dict[str, float]:
        """Lấy breakdown từ confidence_info nếu có."""
        ci = self._get(result, "confidence_info", None)
        if ci and isinstance(ci, dict):
            return {
                "water_detection":   ci.get("water_detection", 0.5),
                "depth_consistency": ci.get("depth_consistency", 0.5),
                "reference_match":   ci.get("reference_match", 0.5),
                "image_quality":     ci.get("image_quality", 0.5),
            }
        # Fallback từ raw fields
        return {
            "water_detection":   self._get(result, "water_confidence", 0.5),
            "depth_consistency": self._get(result, "depth_confidence", 0.5),
            "reference_match":   0.5 if self._get(result, "reference_objects", []) else 0.0,
            "image_quality":     self._get(result, "image_quality", 0.5),
        }

    @staticmethod
    def _get(obj: Any, key: str, default: Any) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _attach(obj: Any, report: "UncertaintyReport"):
        if isinstance(obj, dict):
            obj["uncertainty"] = report.to_dict()
        else:
            try:
                object.__setattr__(obj, "uncertainty", report.to_dict())
            except Exception:
                pass


# ── Convenience ───────────────────────────────────────────────────────────────

def evaluate_uncertainty(results: List[Any], cfg: Optional[dict] = None) -> List[UncertaintyReport]:
    """Shorthand: đánh giá uncertainty cho toàn bộ depth_results."""
    ev = UncertaintyEvaluator(cfg)
    return ev.evaluate_batch(results)
