# -*- coding: utf-8 -*-
"""
depth_analysis/raincoat_detector.py
-------------------------------------
Phát hiện áo mưa (raincoat detection) qua 3 lớp:

  Layer 1 — Rule-based (màu sắc + texture + shape từ pose)
  Layer 2 — CLIP zero-shot  (crop person → similarity với text prompt)
  Layer 3 — Ensemble fusion (weighted score)

Pipeline áp dụng sau PoseAnalyzer:

    detect person
    → expand bbox (pad 15%)
    → pose analysis (filtered keypoints)
    → torso extraction
    → RaincoatDetector.detect(img, bbox, keypoints)
    → RaincoatResult

Lợi ích:
  ✓ Không cần train dataset — CLIP zero-shot
  ✓ Rule-based hoạt động khi CLIP không có GPU
  ✓ Ensemble giảm false positive

Cách dùng:
    detector = RaincoatDetector()
    result = detector.detect(img_rgb, bbox=[x1,y1,x2,y2], keypoints=kpts)
    if result.is_raincoat:
        print(f"Áo mưa detected: conf={result.confidence:.2f}")
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, cast

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Màu áo mưa phổ biến (HSV ranges)
# Vàng, Xanh lá, Xanh dương, Đỏ/Cam, Trắng đục
RAINCOAT_HSV_RANGES = [
    # (lower_hsv, upper_hsv, label)
    (np.array([20,  80, 100]), np.array([35,  255, 255]), "yellow"),   # vàng
    (np.array([36,  60, 80]),  np.array([85,  255, 255]), "green"),    # xanh lá
    (np.array([86,  60, 80]),  np.array([130, 255, 255]), "blue"),     # xanh dương
    (np.array([0,   80, 100]), np.array([15,  255, 255]), "red_low"),  # đỏ/cam
    (np.array([160, 80, 100]), np.array([180, 255, 255]), "red_high"), # đỏ
    (np.array([0,   0,  160]), np.array([180, 40,  255]), "white"),    # trắng đục
]

# CLIP text prompts
CLIP_POSITIVE_PROMPTS = [
    "a person wearing a raincoat",
    "a person wearing waterproof jacket",
    "a person in rain gear",
    "a person wearing yellow raincoat",
    "a person wearing plastic rain poncho",
]
CLIP_NEGATIVE_PROMPTS = [
    "a person wearing normal clothes",
    "a person wearing a t-shirt",
    "a person wearing a jacket",
]

# Ensemble weights
W_RULE = 0.40
W_CLIP = 0.40
W_SHAPE = 0.20

# Keypoint indices (YOLO-Pose)
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RaincoatResult:
    """Kết quả phát hiện áo mưa cho 1 người."""
    is_raincoat:    bool
    confidence:     float           # [0, 1]
    rule_score:     float           # score từ rule-based
    clip_score:     float           # score từ CLIP (−1 nếu không có CLIP)
    shape_score:    float           # score từ shape analysis
    dominant_color: str             # màu chủ đạo vùng torso
    texture_var:    float           # variance → thấp = áo mưa trơn
    bright_ratio:   float           # tỉ lệ pixel sáng (phản quang)
    shoulder_bbox_ratio: float      # shoulder_width / bbox_width
    details:        List[str] = field(default_factory=list)


# ── Main detector ─────────────────────────────────────────────────────────────

class RaincoatDetector:
    """
    Phát hiện áo mưa qua 3 lớp: Rule-based + CLIP + Shape.

    Args:
        use_clip:    Bật/tắt CLIP (cần transformers + torch)
        clip_thresh: Ngưỡng CLIP score để coi là áo mưa
        rule_thresh: Ngưỡng rule score
        final_thresh: Ngưỡng ensemble cuối
        bbox_pad:    Hệ số padding bbox trước khi crop
    """

    def __init__(
        self,
        use_clip:     bool  = True,
        clip_thresh:  float = 0.28,
        rule_thresh:  float = 0.45,
        final_thresh: float = 0.40,
        bbox_pad:     float = 0.15,
    ):
        self.use_clip     = use_clip
        self.clip_thresh  = clip_thresh
        self.rule_thresh  = rule_thresh
        self.final_thresh = final_thresh
        self.bbox_pad     = bbox_pad
        self._clip_model  = None
        self._clip_proc   = None

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        img_rgb:   np.ndarray,          # ảnh đầy đủ RGB
        bbox:      List[int],           # [x1, y1, x2, y2] của person
        keypoints: Optional[np.ndarray] = None,  # (17, 3) từ PoseAnalyzer
    ) -> RaincoatResult:
        """
        Phát hiện áo mưa cho 1 người.

        Returns:
            RaincoatResult với đầy đủ breakdown
        """
        # Expand bbox với padding
        x1, y1, x2, y2 = self._expand_bbox(bbox, img_rgb.shape, self.bbox_pad)
        person_crop = img_rgb[y1:y2, x1:x2]

        if person_crop.size == 0:
            return self._empty_result()

        # Extract torso region
        torso_crop = self._extract_torso(
            img_rgb, bbox, keypoints, x1, y1, x2, y2
        )

        # Layer 1: Rule-based features
        rule_score, color_name, tex_var, bright_ratio = self._rule_based_score(
            torso_crop if torso_crop is not None else person_crop
        )

        # Layer 2: Shape score từ pose
        shape_score, shoulder_bbox_ratio = self._shape_score(bbox, keypoints)

        # Layer 3: CLIP score
        clip_score = -1.0
        if self.use_clip:
            clip_score = self._clip_score(person_crop)

        # Ensemble
        confidence, details = self._ensemble(rule_score, clip_score, shape_score)

        return RaincoatResult(
            is_raincoat=confidence >= self.final_thresh,
            confidence=round(confidence, 3),
            rule_score=round(rule_score, 3),
            clip_score=round(clip_score, 3) if clip_score >= 0 else -1.0,
            shape_score=round(shape_score, 3),
            dominant_color=color_name,
            texture_var=round(tex_var, 2),
            bright_ratio=round(bright_ratio, 3),
            shoulder_bbox_ratio=round(shoulder_bbox_ratio, 3),
            details=details,
        )

    def detect_batch(
        self,
        img_rgb:   np.ndarray,
        persons:   List[dict],          # list of {"bbox": [...], "keypoints": ...}
    ) -> List[RaincoatResult]:
        """Phát hiện cho nhiều người trong 1 ảnh."""
        results = []
        for p in persons:
            bbox = p.get("bbox", [0, 0, img_rgb.shape[1], img_rgb.shape[0]])
            kpts = p.get("keypoints")
            results.append(self.detect(img_rgb, bbox, kpts))
        return results

    # ── Layer 1: Rule-based ───────────────────────────────────────────────────

    def _rule_based_score(
        self, crop: np.ndarray
    ) -> Tuple[float, str, float, float]:
        """
        Tính rule score từ màu + texture + reflectivity.

        Returns:
            (score [0,1], dominant_color, texture_variance, bright_ratio)
        """
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return 0.0, "unknown", 0.0, 0.0

        crop_resized = cv2.resize(crop, (64, 96)) if min(crop.shape[:2]) > 10 else crop
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 1. Color score: tỉ lệ pixel trong range màu áo mưa
        color_score, dominant_color = self._color_score(hsv)

        # 2. Texture score: variance thấp = bề mặt trơn (áo mưa nhựa)
        texture_var = float(np.var(gray))
        # Áo mưa: texture_var thường < 800 (bề mặt trơn)
        # Quần áo thường: 800–2500
        tex_score = float(np.clip(1.0 - texture_var / 1200.0, 0.0, 1.0))

        # 3. Reflectivity: tỉ lệ pixel rất sáng (áo mưa phản quang)
        bright_mask = gray > 200
        bright_ratio = float(bright_mask.sum()) / (gray.size + 1e-6)
        reflect_score = float(np.clip(bright_ratio * 3.0, 0.0, 1.0))

        # 4. Saturation uniformity: áo mưa thường có màu đồng đều
        sat = hsv[:, :, 1].astype(np.float32)
        sat_std = float(np.std(sat))
        uniformity_score = float(np.clip(1.0 - sat_std / 80.0, 0.0, 1.0))

        # Weighted combination
        rule_score = (
            color_score      * 0.45 +
            tex_score        * 0.25 +
            reflect_score    * 0.15 +
            uniformity_score * 0.15
        )

        log.debug(
            f"    Rule: color={color_score:.2f} tex={tex_score:.2f} "
            f"reflect={reflect_score:.2f} unif={uniformity_score:.2f} "
            f"→ {rule_score:.2f}"
        )

        return float(np.clip(rule_score, 0.0, 1.0)), dominant_color, texture_var, bright_ratio

    def _color_score(self, hsv: np.ndarray) -> Tuple[float, str]:
        """Tính tỉ lệ pixel trong range màu áo mưa, trả về score và màu chủ đạo."""
        best_ratio = 0.0
        best_label = "none"
        total_pixels = hsv.shape[0] * hsv.shape[1]

        for lower, upper, label in RAINCOAT_HSV_RANGES:
            mask = cv2.inRange(hsv, lower, upper)
            ratio = float(mask.sum() / 255) / (total_pixels + 1e-6)
            if ratio > best_ratio:
                best_ratio = ratio
                best_label = label

        # score: tuyến tính, 20% pixel = score 1.0
        score = float(np.clip(best_ratio / 0.20, 0.0, 1.0))
        return score, best_label

    # ── Layer 2: Shape từ pose ────────────────────────────────────────────────

    def _shape_score(
        self,
        bbox: List[int],
        keypoints: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        """
        Phát hiện "loose clothing" từ pose keypoints.

        Áo mưa = hình dạng rộng:
          ratio = shoulder_width / bbox_width
          Nếu ratio < 0.4 → vai hẹp so với bbox → có thể áo mưa rộng

        Returns:
            (shape_score [0,1], shoulder_bbox_ratio)
        """
        x1, y1, x2, y2 = bbox
        bbox_w = max(x2 - x1, 1)
        bbox_h = max(y2 - y1, 1)

        # Aspect ratio filter: người thường h/w ≈ 1.5–4
        ar = bbox_h / (bbox_w + 1e-6)
        if ar < 1.2 or ar > 5.0:
            # bbox bất thường → không phải người đứng bình thường
            return 0.0, 0.0

        if keypoints is None:
            return 0.3, 0.5  # không có pose → neutral

        def kp(idx):
            if keypoints[idx, 2] > 0.3:
                return float(keypoints[idx, 0]), float(keypoints[idx, 1])
            return None

        l_sh = kp(KP_LEFT_SHOULDER)
        r_sh = kp(KP_RIGHT_SHOULDER)

        if l_sh is None or r_sh is None:
            return 0.2, 0.5

        shoulder_w = abs(l_sh[0] - r_sh[0])
        shoulder_bbox_ratio = shoulder_w / bbox_w

        # Nếu vai hẹp so với bbox → áo rộng → khả năng cao là áo mưa
        # ratio < 0.35 → score cao
        # ratio > 0.60 → score thấp
        if shoulder_bbox_ratio < 0.30:
            shape_score = 0.90
        elif shoulder_bbox_ratio < 0.40:
            shape_score = 0.70
        elif shoulder_bbox_ratio < 0.50:
            shape_score = 0.50
        elif shoulder_bbox_ratio < 0.60:
            shape_score = 0.30
        else:
            shape_score = 0.10  # vai rộng = áo vừa người = không phải áo mưa

        return float(shape_score), float(shoulder_bbox_ratio)

    # ── Layer 3: CLIP ─────────────────────────────────────────────────────────

    def _clip_score(self, person_crop: np.ndarray) -> float:
        """
        Dùng CLIP zero-shot để tính similarity giữa crop và text prompts.

        Returns:
            float [0,1] — score áo mưa, hoặc −1 nếu CLIP không khả dụng
        """
        try:
            model, processor = self._load_clip()
            if model is None or processor is None:
                return -1.0

            # _load_clip returns None when the optional dependency/model is
            # unavailable; narrow the dynamically loaded Hugging Face objects
            # for static type checkers before calling them.
            model_obj = cast(Any, model)
            processor_obj = cast(Any, processor)

            import torch
            from PIL import Image

            pil_img = Image.fromarray(person_crop)

            all_prompts = CLIP_POSITIVE_PROMPTS + CLIP_NEGATIVE_PROMPTS
            inputs = processor_obj(
                text=all_prompts,
                images=pil_img,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = model_obj(**inputs)
                logits = outputs.logits_per_image  # (1, n_prompts)
                probs = logits.softmax(dim=-1).squeeze().cpu().numpy()

            n_pos = len(CLIP_POSITIVE_PROMPTS)
            pos_score = float(probs[:n_pos].sum())
            # Normalize lại: nếu 50/50 → 0.5
            total = float(probs.sum()) + 1e-9
            clip_score = pos_score / total

            log.debug(f"    CLIP: pos={pos_score:.3f} → score={clip_score:.3f}")
            return float(np.clip(clip_score, 0.0, 1.0))

        except Exception as e:
            log.debug(f"    CLIP not available: {e}")
            return -1.0

    def _load_clip(self):
        """Lazy load CLIP model."""
        if self._clip_model is not None:
            return self._clip_model, self._clip_proc
        try:
            from transformers import CLIPModel, CLIPProcessor
            log.info("  Loading CLIP model (openai/clip-vit-base-patch32)...")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model.eval()
            log.info("  CLIP loaded")
            return self._clip_model, self._clip_proc
        except Exception as e:
            log.warning(f"  Cannot load CLIP: {e} — falling back to rule-based only")
            self._clip_model = None
            self._clip_proc  = None
            return None, None

    # ── Ensemble ──────────────────────────────────────────────────────────────

    def _ensemble(
        self,
        rule_score:  float,
        clip_score:  float,    # −1 nếu không có CLIP
        shape_score: float,
    ) -> Tuple[float, List[str]]:
        """
        Kết hợp 3 scores theo weights.

        Nếu CLIP không khả dụng → re-distribute weight sang rule + shape.
        """
        details = [
            f"rule={rule_score:.2f}",
            f"shape={shape_score:.2f}",
        ]

        if clip_score >= 0:
            # Có CLIP
            confidence = (
                W_RULE  * rule_score  +
                W_CLIP  * clip_score  +
                W_SHAPE * shape_score
            )
            details.append(f"clip={clip_score:.2f}")
        else:
            # Không có CLIP → re-weight
            w_rule  = W_RULE  / (W_RULE + W_SHAPE)
            w_shape = W_SHAPE / (W_RULE + W_SHAPE)
            confidence = w_rule * rule_score + w_shape * shape_score
            details.append("clip=N/A (no model)")

        confidence = float(np.clip(confidence, 0.0, 1.0))
        details.append(f"final={confidence:.2f}")

        log.debug(f"    Ensemble: {' | '.join(details)}")
        return confidence, details

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _expand_bbox(
        self, bbox: List[int], img_shape: tuple, pad: float
    ) -> Tuple[int, int, int, int]:
        """Expand bbox với padding để không cắt mất áo mưa rộng."""
        x1, y1, x2, y2 = bbox
        h_img, w_img = img_shape[:2]
        bw = x2 - x1
        bh = y2 - y1
        x1 = max(0,     int(x1 - bw * pad))
        y1 = max(0,     int(y1 - bh * pad))
        x2 = min(w_img, int(x2 + bw * pad))
        y2 = min(h_img, int(y2 + bh * pad))
        return x1, y1, x2, y2

    def _extract_torso(
        self,
        img_rgb:   np.ndarray,
        orig_bbox: List[int],
        keypoints: Optional[np.ndarray],
        pad_x1:    int,
        pad_y1:    int,
        pad_x2:    int,
        pad_y2:    int,
    ) -> Optional[np.ndarray]:
        """
        Cắt vùng torso từ keypoints (shoulder → hip).
        Nếu không có keypoints → dùng 1/4 đến 3/4 bbox theo chiều cao.
        """
        if keypoints is not None:
            def kp(idx):
                if keypoints[idx, 2] > 0.3:
                    return float(keypoints[idx, 0]), float(keypoints[idx, 1])
                return None

            l_sh = kp(KP_LEFT_SHOULDER)
            r_sh = kp(KP_RIGHT_SHOULDER)
            l_hp = kp(KP_LEFT_HIP)
            r_hp = kp(KP_RIGHT_HIP)

            # Cần ít nhất 2 điểm trong 4
            pts = [p for p in [l_sh, r_sh, l_hp, r_hp] if p is not None]
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                tx1 = max(0,             int(min(xs)) - 10)
                ty1 = max(0,             int(min(ys)) - 10)
                tx2 = min(img_rgb.shape[1], int(max(xs)) + 10)
                ty2 = min(img_rgb.shape[0], int(max(ys)) + 10)

                if tx2 > tx1 and ty2 > ty1:
                    return img_rgb[ty1:ty2, tx1:tx2]

        # Fallback: dùng phần giữa của bbox (20%–75% chiều cao)
        x1, y1, x2, y2 = orig_bbox
        bh = y2 - y1
        ty1 = max(0, y1 + int(bh * 0.20))
        ty2 = max(0, y1 + int(bh * 0.75))
        if ty2 > ty1:
            return img_rgb[ty1:ty2, x1:x2]
        return None

    def _empty_result(self) -> RaincoatResult:
        return RaincoatResult(
            is_raincoat=False, confidence=0.0,
            rule_score=0.0, clip_score=-1.0, shape_score=0.0,
            dominant_color="unknown", texture_var=0.0,
            bright_ratio=0.0, shoulder_bbox_ratio=0.0,
            details=["empty crop"],
        )

    # ── Debug overlay ─────────────────────────────────────────────────────────

    def draw_overlay(
        self,
        img_bgr: np.ndarray,
        bboxes:  List[List[int]],
        results: List[RaincoatResult],
    ) -> np.ndarray:
        """Vẽ bbox + label lên ảnh debug."""
        overlay = img_bgr.copy()
        for bbox, res in zip(bboxes, results):
            x1, y1, x2, y2 = bbox
            color = (0, 255, 255) if res.is_raincoat else (128, 128, 128)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"Raincoat {res.confidence:.0%}" if res.is_raincoat else f"No coat {res.confidence:.0%}"
            sub   = f"rule={res.rule_score:.2f} clip={res.clip_score:.2f} shp={res.shape_score:.2f}"
            cv2.putText(overlay, label, (x1, max(y1-20, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.putText(overlay, sub,   (x1, max(y1-6, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        return overlay
