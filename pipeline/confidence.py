# -*- coding: utf-8 -*-
"""
Confidence Scorer  —  v2 (4-component formula)
===============================================
Công thức nâng cấp:

    confidence = (
        water_detection_conf * 0.30 +
        depth_consistency    * 0.30 +
        reference_match_score* 0.20 +
        image_quality        * 0.20
    )

Thêm mới:
  - image_quality: đánh giá chất lượng ảnh (blur, exposure, noise)
  - depth_consistency: nhất quán giữa depth model + geometry + reference
  - Caching confidence vào result object với breakdown chi tiết
"""
import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

log = logging.getLogger("pipeline.confidence")

# ── Trọng số 5 components (v3 — thêm raincoat) ────────────────────────────────
W_WATER_DETECTION  = 0.28
W_DEPTH_CONSISTENCY = 0.28
W_REFERENCE_MATCH  = 0.18
W_IMAGE_QUALITY    = 0.18
W_RAINCOAT         = 0.08   # bonus nhỏ khi phát hiện áo mưa (xác nhận lũ)

# Thresholds
LOW_THRESHOLD  = 0.40
HIGH_THRESHOLD = 0.70
class ConfidenceScorer:
    """
    Tính confidence tổng hợp — v3 (5-component formula, thêm raincoat).

    confidence = (
        water_detection_conf * 0.28 +
        depth_consistency    * 0.28 +
        reference_match_score* 0.18 +
        image_quality        * 0.18 +
        raincoat_score       * 0.08   ← NEW: bonus khi có áo mưa
    )

    Raincoat signal:
      - Người mặc áo mưa → môi trường mưa lớn → tăng confidence lũ lụt
      - Không phải: yolo = 0.5 / CLIP = 0.5 * rule → không giảm score
    """

    def __init__(
        self,
        w_water_detection:   float = W_WATER_DETECTION,
        w_depth_consistency: float = W_DEPTH_CONSISTENCY,
        w_reference_match:   float = W_REFERENCE_MATCH,
        w_image_quality:     float = W_IMAGE_QUALITY,
        w_raincoat:          float = W_RAINCOAT,
        low_threshold:       float = LOW_THRESHOLD,
        high_threshold:      float = HIGH_THRESHOLD,
    ):
        # [FIX] Tong 5 weights = 0.28+0.28+0.18+0.18+0.08 = 1.00
        # Cu: normalize lai → self.w_* > 1.0, nhung compute() dung weights cuc (0.30, 0.20...)
        # → co 2 he weight khac nhau, instance weights KHONG BAO GIO duoc dung.
        # Moi: dung weights cuc trong compute() (da on dinh), chi luu instance
        # weights de backward-compat, khong normalize nua.
        self.w_water    = w_water_detection
        self.w_depth    = w_depth_consistency
        self.w_ref      = w_reference_match
        self.w_quality  = w_image_quality
        self.w_raincoat = w_raincoat
        self.low_threshold  = low_threshold
        self.high_threshold = high_threshold

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self, result: Any, image_path: Optional[str] = None) -> float:
        """
        Tính confidence score với multi-component formula (v4).

        Components:
            water_conf       (0.30) — water detection quality
            depth_cons       (0.20) — depth consistency
            ref_score        (0.20) — reference objects (người/xe/cửa)
            img_quality      (0.15) — image quality
            scene_cons       (0.15) — scene consistency (SceneValidator)
            raincoat_score   (0.08) — raincoat signal bonus

        Nếu không có vật tham chiếu: trả về kết quả nhưng thêm warning.

        Returns:
            float confidence [0.0, 1.0]
        """
        # Component 1: Water detection confidence
        water_conf = self._compute_water_detection_conf(result)

        # Component 2: Depth consistency (agreement giữa các methods)
        depth_cons = self._compute_depth_consistency(result)

        # Component 3: Reference match score (quality of reference objects)
        ref_score  = self._compute_reference_match_score(result)

        # Component 4: Image quality (blur, exposure, noise)
        img_quality = self._compute_image_quality(result, image_path)

        # Component 5: Raincoat signal (context confirmer)
        raincoat_score = self._compute_raincoat_score(result)

        # Component 6: Scene consistency (từ SceneValidator nếu có)
        scene_cons = self._compute_scene_consistency(result)

        # Component 7: Scene graph relation boost
        relation_boost = self._compute_relation_boost(result)

        # Weighted combination (tổng = 1.0 + raincoat bonus nhỏ)
        confidence = (
            0.30 * water_conf   +
            0.20 * depth_cons   +
            0.20 * ref_score    +
            0.15 * img_quality  +
            0.15 * scene_cons   +
            0.08 * raincoat_score
        )
        confidence = float(np.clip(confidence + relation_boost, 0.0, 1.0))

        # No-reference warning: có ngập nhưng không đủ reference để đo cm
        no_ref_warning = (ref_score < 0.15)

        self._attach_v4(result, confidence, {
            "water_conf":      water_conf,
            "depth_cons":      depth_cons,
            "ref_score":       ref_score,
            "img_quality":     img_quality,
            "scene_cons":      scene_cons,
            "raincoat_signal": raincoat_score,
            "relation_boost":  relation_boost,
        }, no_ref_warning=no_ref_warning)

        log.debug(
            "  Confidence v4: %.2f (water=%.2f depth=%.2f ref=%.2f "
            "quality=%.2f scene=%.2f boost=%.2f)",
            confidence, water_conf, depth_cons, ref_score,
            img_quality, scene_cons, relation_boost,
        )
        return confidence

    def label(self, confidence: float) -> str:
        if confidence >= self.high_threshold:
            return "HIGH"
        if confidence >= self.low_threshold:
            return "MEDIUM"
        return "LOW"

    def needs_review(self, confidence: float) -> bool:
        return confidence < self.high_threshold

    # ── Component 1: Water Detection Confidence ────────────────────────────────

    def _compute_water_detection_conf(self, result: Any) -> float:
        """
        Confidence từ water detection model.

        Lấy từ:
          - water_detector confidence (color + texture voting)
          - YOLO detection confidence nếu có
          - Segmentation confidence nếu có
        """
        # Thử lấy từ các attribute khác nhau
        for attr in ["water_detection_conf", "water_conf", "confidence", "conf", "model_confidence"]:
            val = self._get(result, attr)
            if val is not None:
                return float(np.clip(val, 0.0, 1.0))

        # Nếu có water_area_pct → estimate
        water_pct = self._get(result, "water_area_pct") or self._get(result, "water_region_area")
        if water_pct is not None:
            pct = float(water_pct) / 100.0 if float(water_pct) > 1 else float(water_pct)
            return float(np.clip(pct * 1.5, 0.1, 0.9))

        return 0.5  # fallback

    # ── Component 2: Depth Consistency ────────────────────────────────────────

    def _compute_depth_consistency(self, result: Any) -> float:
        """
        Nhất quán giữa các phương pháp depth estimation.

        Nếu depth_model + geometry + reference_obj đồng thuận → high consistency.
        """
        # Lấy các component floods nếu có (từ fusion)
        fusion_components = self._get(result, "fusion_components")
        if isinstance(fusion_components, dict):
            values = [v for v in fusion_components.values() if isinstance(v, (int, float))]
            if len(values) >= 2:
                std = np.std(values)
                # std thấp = đồng thuận cao
                return float(np.clip(1.0 - std * 5.0, 0.0, 1.0))

        # Fallback: consistency từ reference objects
        refs = (
            self._get(result, "reference_objects") or
            self._get(result, "references") or []
        )
        if not isinstance(refs, (list, tuple)) or len(refs) < 2:
            return 0.5

        depths = []
        for ref in refs:
            d = (self._get(ref, "estimated_depth") or
                 self._get(ref, "depth") or
                 self._get(ref, "flood_cm"))
            if d is not None:
                depths.append(float(d))

        if len(depths) < 2:
            return 0.5

        mean_d = np.mean(depths)
        cv = np.std(depths) / (mean_d + 1e-6)  # coefficient of variation
        if cv < 0.10: return 1.0
        if cv < 0.25: return 0.75
        if cv < 0.50: return 0.50
        return 0.25

    # ── Component 3: Reference Match Score ────────────────────────────────────

    def _compute_reference_match_score(self, result: Any) -> float:
        """
        Chất lượng reference objects.

        Priority: wheel > car > person (như đã thiết kế)
        Score = weighted average của confidence từng reference.
        """
        refs = (
            self._get(result, "reference_objects") or
            self._get(result, "references") or
            self._get(result, "detections") or []
        )

        if not isinstance(refs, (list, tuple)) or len(refs) == 0:
            return 0.2  # không có reference → thấp

        # Trọng số theo loại reference
        type_weights = {
            "wheel":   1.0,
            "tire":    1.0,
            "car":     0.85,
            "vehicle": 0.85,
            "truck":   0.90,
            "person":  0.50,
            "human":   0.50,
            "default": 0.60,
        }

        total_score = 0.0
        total_weight = 0.0

        for ref in refs[:5]:  # lấy tối đa 5 refs
            ref_type = str(
                self._get(ref, "type") or
                self._get(ref, "class") or
                self._get(ref, "label") or
                "default"
            ).lower()

            conf = float(
                self._get(ref, "confidence") or
                self._get(ref, "conf") or
                0.5
            )

            # Tìm weight phù hợp nhất
            w = type_weights.get("default", 0.60)
            for k, v in type_weights.items():
                if k in ref_type:
                    w = v
                    break

            total_score  += conf * w
            total_weight += w

        if total_weight == 0:
            return 0.3

        raw_score = total_score / total_weight

        # Bonus: nhiều reference → more confident
        count_bonus = min(0.20, len(refs) * 0.04)
        return float(np.clip(raw_score + count_bonus, 0.0, 1.0))

    # ── Component 5: Raincoat Signal ──────────────────────────────────────────

    def _compute_raincoat_score(self, result: Any) -> float:
        """
        Raincoat = context confirmer: người mặc áo mưa → xác nhận mưa lớn.

        Logic:
          - has_raincoat=True, raincoat_confidence > 0.6 → score = 0.8–1.0
          - has_raincoat=True, conf < 0.6                → score = 0.5–0.7
          - has_raincoat=False (hoặc không có)            → score = 0.5 (neutral)

        Không phạt khi không có áo mưa — nhiều tình huống lũ không có người.
        """
        has_rc = self._get(result, "has_raincoat")
        rc_conf = self._get(result, "raincoat_confidence") or 0.0
        rc_count = self._get(result, "raincoat_count") or 0

        if has_rc is None:
            return 0.5   # không có raincoat stage → neutral

        if not has_rc:
            return 0.50  # không phát hiện → neutral (không phạt)

        # Có raincoat → bonus theo confidence
        base = 0.60 + float(rc_conf) * 0.35    # 0.60 → 0.95
        # Bonus thêm nếu nhiều người mặc áo mưa
        count_bonus = min(0.05, rc_count * 0.02)
        return float(np.clip(base + count_bonus, 0.0, 1.0))

    # ── Component 4: Image Quality ─────────────────────────────────────────────

    def _compute_image_quality(
        self, result: Any, image_path: Optional[str] = None
    ) -> float:
        """
        Đánh giá chất lượng ảnh đầu vào.

        Metrics:
          - Blur score (Laplacian variance)
          - Exposure (mean brightness)
          - Noise level (local std)
          - Resolution (pixel count)
        """
        # Nếu có image_quality sẵn
        iq = self._get(result, "image_quality")
        if iq is not None:
            return float(np.clip(iq, 0.0, 1.0))

        if image_path is None:
            # Thử lấy từ result
            image_path = (
                self._get(result, "original_path") or
                self._get(result, "image_path")
            )

        if image_path is None:
            return 0.6  # không có ảnh → fallback trung bình

        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                return 0.6

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]

            # 1. Blur score: Laplacian variance (cao = nét, thấp = mờ)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = float(np.clip(lap_var / 500.0, 0.0, 1.0))

            # 2. Exposure: mean brightness (quá tối hoặc quá sáng → kém)
            mean_brightness = gray.mean()
            if mean_brightness < 30:
                exp_score = 0.2   # quá tối
            elif mean_brightness > 230:
                exp_score = 0.2   # quá sáng (overexposed)
            else:
                # Optimal: 80-170
                dist_from_optimal = abs(mean_brightness - 125) / 125.0
                exp_score = float(np.clip(1.0 - dist_from_optimal, 0.3, 1.0))

            # 3. Resolution: full HD = 1.0, nhỏ hơn giảm dần
            pixels = h * w
            res_score = float(np.clip(pixels / (1920 * 1080), 0.2, 1.0))

            # 4. Noise: local variance (thấp = ít noise, nhưng quá thấp = blur)
            local_std = cv2.blur(gray.astype(np.float32)**2, (5,5)) - \
                        cv2.blur(gray.astype(np.float32), (5,5))**2
            avg_noise = float(np.sqrt(np.maximum(local_std, 0)).mean())
            noise_score = float(np.clip(1.0 - avg_noise / 50.0, 0.3, 1.0))

            # Weighted: blur quan trọng nhất
            quality = (
                blur_score * 0.40 +
                exp_score  * 0.30 +
                res_score  * 0.15 +
                noise_score * 0.15
            )
            return float(np.clip(quality, 0.0, 1.0))

        except Exception as e:
            log.debug(f"  Image quality check failed: {e}")
            return 0.6

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _get(self, obj: Any, attr: str):
        if isinstance(obj, dict):
            return obj.get(attr)
        return getattr(obj, attr, None)

    def _compute_scene_consistency(self, result: Any) -> float:
        """Lấy scene_score từ SceneValidator nếu đã chạy."""
        # Từ ValidationResult được attach vào result
        val = self._get(result, "scene_validation")
        if val and isinstance(val, dict):
            return float(val.get("scene_score", 0.5))
        # Từ SceneGraph
        sg = self._get(result, "scene_graph")
        if sg and hasattr(sg, "compute_confidence"):
            try:
                return float(sg.compute_confidence())
            except Exception:
                pass
        return 0.5  # neutral nếu không có

    def _compute_relation_boost(self, result: Any) -> float:
        """Lấy relation confidence boost từ SceneGraph."""
        sg = self._get(result, "scene_graph")
        if sg and hasattr(sg, "relation_confidence_boost"):
            try:
                return float(sg.relation_confidence_boost())
            except Exception:
                pass
        return 0.0

    def _attach_v4(
        self,
        result: Any,
        confidence: float,
        components: dict,
        no_ref_warning: bool = False,
    ) -> None:
        """
        Gắn confidence_info v4 vào result với breakdown đầy đủ.

        Nếu no_ref_warning = True, thêm cảnh báo:
        "Có khả năng ngập, nhưng không có vật tham chiếu → không thể ước lượng độ sâu chính xác"
        """
        warnings = []
        if no_ref_warning:
            warnings.append(
                "Không có vật tham chiếu (người/xe/cửa) — "
                "chiều sâu chỉ là ước tính thô, không đủ tin cậy để đo cm"
            )
        if components.get("water_conf", 1.0) < 0.35:
            warnings.append("Water detection yếu — có thể không có lũ thật")
        if components.get("scene_cons", 1.0) < 0.40:
            warnings.append("Scene không nhất quán (đường ướt / phản chiếu?)")

        info = {
            "score":        confidence,
            "label":        self.label(confidence),
            "needs_review": self.needs_review(confidence),
            "warnings":     warnings,
            "components": {
                "water_conf":      round(components.get("water_conf",      0.5), 3),
                "depth_cons":      round(components.get("depth_cons",      0.5), 3),
                "reference_score": round(components.get("ref_score",       0.5), 3),
                "image_quality":   round(components.get("img_quality",     0.5), 3),
                "scene_cons":      round(components.get("scene_cons",      0.5), 3),
                "raincoat_signal": round(components.get("raincoat_signal", 0.5), 3),
                "relation_boost":  round(components.get("relation_boost",  0.0), 3),
            },
        }
        if isinstance(result, dict):
            result["confidence_info"] = info
        else:
            try:
                result.confidence_info = info
                result.confidence = confidence
            except AttributeError:
                pass

    def _attach(
        self,
        result: Any,
        confidence: float,
        water_conf: float,
        depth_cons: float,
        ref_score: float,
        img_quality: float,
        raincoat_score: float = 0.5,
    ) -> None:
        """Legacy _attach — kept for backward compat."""
        self._attach_v4(result, confidence, {
            "water_conf": water_conf, "depth_cons": depth_cons,
            "ref_score": ref_score, "img_quality": img_quality,
            "raincoat_signal": raincoat_score,
        })


# Backward-compat: expose old 3-component API as alias
class ConfidenceScorerV1(ConfidenceScorer):
    """Legacy wrapper cho code cũ dùng w_model/w_detect/w_consist."""
    def __init__(self, w_model=0.40, w_detect=0.35, w_consist=0.25,
                 low_threshold=0.40, high_threshold=0.70):
        super().__init__(
            w_water_detection=w_model,
            w_depth_consistency=w_consist,
            w_reference_match=w_detect,
            w_image_quality=0.15,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )


# ── Standalone usage ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ví dụ test nhanh
    scorer = ConfidenceScorer()

    fake_result = {
        "confidence": 0.82,
        "flood_level": "KNEE",
        "reference_objects": [
            {"estimated_depth": 45.0},
            {"estimated_depth": 48.0},
            {"estimated_depth": 43.0},
        ],
    }
    conf = scorer.compute(fake_result)
    print(f"Confidence: {conf:.2f} → {scorer.label(conf)}")
    print(f"Needs review: {scorer.needs_review(conf)}")
