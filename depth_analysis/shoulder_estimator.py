# -*- coding: utf-8 -*-
"""
depth_analysis/shoulder_estimator.py  —  v2
---------------------------------------------
Ước tính chiều cao người qua vai (shoulder width estimation).

Cải tiến v2:
  1. Head-to-ankle trực tiếp — khi thấy cả đầu lẫn mắt cá chân → scale chính xác nhất
  2. Head-to-hip — khi thấy hông mà không thấy chân
  3. Head size ratio — dùng khoảng cách 2 tai / 2 mắt để suy ra px/cm
  4. Ensemble + weighted average — kết hợp nhiều method thay vì chỉ lấy 1
  5. Perspective correction — dùng depth_norm để scale theo khoảng cách camera
  6. Smart fallback — thứ tự ưu tiên rõ ràng, mỗi method tự báo confidence

Thứ tự ưu tiên method:
  head_to_ankle  → ±3cm   (thấy đầu + mắt cá → tính trực tiếp)
  head_to_hip    → ±6cm   (thấy đầu + hông, áp dụng tỉ lệ Dempster)
  head_size      → ±8cm   (dùng kích thước đầu làm thước kẻ)
  keypoint_sh    → ±5cm   (keypoint vai chuẩn → dùng px/cm từ bbox)
  neck           → ±10cm  (phân tích hình ảnh neck region)
  upper_body     → ±15cm  (fallback: 70th percentile of upper body width)
  default        → 170cm  (không ước tính được gì)

Tài liệu tham khảo:
  - Dempster (1955) body segment proportions
  - Phân tích nhân trắc học người Việt Nam (Đỗ Xuân Hợp, 2001)
  - ANSUR II (US Army anthropometric survey)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Hằng số nhân trắc học ────────────────────────────────────────────────────
# Tỉ lệ vai / chiều cao (nghiên cứu nhân trắc)
SHOULDER_TO_HEIGHT_RATIO   = 0.240   # trung bình
SHOULDER_TO_HEIGHT_RATIO_M = 0.245   # nam
SHOULDER_TO_HEIGHT_RATIO_F = 0.232   # nữ

# [IMPROVE] Vietnamese-specific anthropometric constants
# Nguồn: "Phân tích nhân trắc học người Việt Nam" (Đỗ Xuân Hợp, 2001) + các khảo sát mới hơn
# Người VN thường có torso ngắn hơn, chân dài hơn so với người phương Tây
SHOULDER_TO_HEIGHT_RATIO_VN       = 0.235   # trung bình VN
SHOULDER_TO_HEIGHT_RATIO_VN_M     = 0.240   # nam VN
SHOULDER_TO_HEIGHT_RATIO_VN_F     = 0.228   # nữ VN

# Dempster ratios cho người VN (torso ngắn hơn, chân dài hơn)
SEG_HEAD_NECK_VN   = 0.135   # đầu + cổ (hơi lớn hơn)
SEG_TRUNK_VN       = 0.285   # thân (vai → hông) - ngắn hơn
SEG_UPPER_LEG_VN   = 0.250   # đùi
SEG_LOWER_LEG_VN   = 0.251   # cẳng chân
SEG_HEAD_TO_HIP_VN = SEG_HEAD_NECK_VN + SEG_TRUNK_VN  # = 0.420

# Tỷ lệ đầu: khoảng cách tai-tai / chiều cao
HEAD_WIDTH_TO_HEIGHT_RATIO_VN = 0.148  # VN hơi nhỏ hơn
EYE_WIDTH_TO_HEIGHT_RATIO_VN  = 0.064

# Chiều cao mặc định nếu không ước tính được
DEFAULT_HEIGHT_CM  = 165.0  # [IMPROVE] Trung bình VN ~165cm (nam 168, nữ 158)
DEFAULT_HEIGHT_MIN = 120.0  # [IMPROVE] Cho phép trẻ em thấp hơn
DEFAULT_HEIGHT_MAX = 200.0

# ── [v5] Upper Body Anatomical Ruler ─────────────────────────────────
# Đo px_per_cm TỪ PHẦN TRÊN CƠ THỂ (luôn nổi trên mặt nước).
# Bao giờ dùng bbox (bh/DEFAULT) cũng sai khi người ngập 1 phần: bbox chỉ
# bao phần nổi → px_per_cm bị co → height bị đội lên tới DEFAULT_HEIGHT_MAX.
HEAD_TO_CLAVICLE_CM = 25.0   # đỉnh đầu → mỏm vai (acromion) ≈ 25cm
BIACROMIAL_CM       = 39.0   # rộng 2 mỏm vai ≈ 38-40cm
PX_PER_CM_MIN       = 0.30   # ngưỡng hợp lệ px/cm
PX_PER_CM_MAX       = 15.0

# Ngưỡng confidence keypoint YOLO-Pose
KP_CONF_THRESH = 0.30
KP_CONF_HIGH   = 0.55   # confident cao — dùng cho ensemble

# YOLO-Pose keypoint indices
KP_NOSE        = 0
KP_LEFT_EYE    = 1;  KP_RIGHT_EYE   = 2
KP_LEFT_EAR    = 3;  KP_RIGHT_EAR   = 4
KP_LEFT_SHLD   = 5;  KP_RIGHT_SHLD  = 6
KP_LEFT_ELBOW  = 7;  KP_RIGHT_ELBOW = 8
KP_LEFT_HIP    = 11; KP_RIGHT_HIP   = 12
KP_LEFT_KNEE   = 13; KP_RIGHT_KNEE  = 14
KP_LEFT_ANKLE  = 15; KP_RIGHT_ANKLE = 16

# Tỉ lệ Dempster (body segment / total height)
# Nguồn: Dempster 1955, adapted cho người châu Á
SEG_HEAD_NECK   = 0.130   # đầu + cổ
SEG_TRUNK       = 0.300   # thân (vai → hông)
SEG_UPPER_LEG   = 0.245   # đùi
SEG_LOWER_LEG   = 0.246   # cẳng chân
SEG_HEAD_TO_HIP = SEG_HEAD_NECK + SEG_TRUNK  # = 0.430

# Tỉ lệ đầu: khoảng cách tai-tai / chiều cao ≈ 0.140-0.160
HEAD_WIDTH_TO_HEIGHT_RATIO = 0.150  # ear-to-ear / height

# Tỉ lệ mắt: inter-eye distance / chiều cao ≈ 0.060-0.070
EYE_WIDTH_TO_HEIGHT_RATIO  = 0.065

# bbox shoulder ratio (shoulder thực / bbox_width ở vùng vai)
BBOX_SHOULDER_RATIO = 0.82

# Giới hạn bbox để shoulder detection có nghĩa
MIN_BBOX_HEIGHT_FOR_SHOULDER = 60
MIN_BBOX_WIDTH_FOR_SHOULDER  = 30

POSE_FACTOR_ADJUST_WEIGHT = 0.06

# [IMPROVE] Raincoat / loose clothing detection constants
RAINCOAT_SATURATION_THRESH   = 130   # áo mưa thường saturation cao (màu đồng nhất)
RAINCOAT_BRIGHTNESS_MIN      = 80    # không quá tối
RAINCOAT_HUE_RANGE           = (80, 140)  # xanh/cam - màu áo mưa phổ biến
RAINCOAT_TEXTURE_MAX         = 12    # áo mưa có texture thấp (vai đồng nhất)
RAINCOAT_SHOULDER_INFLATE    = 1.25  # áo mưa làm vai to ra ~25%
LOOSE_CLOTHING_SHOULDER_INFLATE = 1.15  # áo rộng thường

# [IMPROVE] Child detection
CHILD_HEIGHT_THRESHOLD_CM    = 140.0  # dưới này coi là trẻ em
CHILD_SHOULDER_TO_HEIGHT     = 0.210  # trẻ em vai/chiều cao thấp hơn


@dataclass
class ShoulderEstimateResult:
    """Kết quả ước tính chiều cao qua vai."""
    estimated_height_cm: float
    shoulder_width_px:   float
    shoulder_width_cm:   float
    px_per_cm:           float
    method:              str
    confidence:          float
    kp_left:             Optional[Tuple[float, float]] = None
    kp_right:            Optional[Tuple[float, float]] = None
    notes:               str = ""


# ── Helper ───────────────────────────────────────────────────────────────────

def _kpt(kpts: np.ndarray, idx: int) -> Tuple[float, float, float]:
    """Lấy (x, y, conf) của keypoint idx, trả về (0,0,0) nếu không có."""
    if kpts is None or kpts.shape[0] <= idx:
        return 0.0, 0.0, 0.0
    return float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])


def _visible(kpts: np.ndarray, idx: int, thresh: float = KP_CONF_THRESH) -> bool:
    _, _, c = _kpt(kpts, idx)
    return c >= thresh


def _clamp_height(h: float) -> float:
    return max(DEFAULT_HEIGHT_MIN, min(DEFAULT_HEIGHT_MAX, h))


def _get_height_ratio(shoulder_cm: float, torso_ratio: float = 1.0) -> float:
    """
    Tỉ lệ vai/chiều cao thích ứng theo kích cỡ vai và torso ratio.
    Vai lớn → tỉ lệ hơi cao hơn; vai nhỏ → tỉ lệ thấp hơn.
    """
    base = SHOULDER_TO_HEIGHT_RATIO
    # Vai quá rộng (>50cm) → người to → tỉ lệ lớn hơn chút
    if shoulder_cm > 48:
        base = min(SHOULDER_TO_HEIGHT_RATIO_M + 0.005, base + 0.010)
    elif shoulder_cm < 34:
        base = max(SHOULDER_TO_HEIGHT_RATIO_F - 0.005, base - 0.008)
    # Torso ratio: vai/hông. Nếu vai rộng hơn hông nhiều → người lớn hơn
    if torso_ratio > 1.05:
        base += 0.003
    return max(0.210, min(0.270, base))


# [IMPROVE] Raincoat / loose clothing detection
def _detect_raincoat_on_person(img_rgb: np.ndarray, bbox: List[int], kpts: Optional[np.ndarray]) -> dict:
    """
    Phát hiện áo mưa / quần áo rộng trên người để điều chỉnh shoulder width.
    Returns: dict với keys: is_raincoat, is_loose, inflate_factor, confidence
    """
    x1, y1, x2, y2 = bbox
    # Crop vùng người (mở rộng nhẹ để lấy áo)
    pad = max(10, int((y2 - y1) * 0.05))
    crop_x1 = max(0, x1 - pad)
    crop_y1 = max(0, y1 - pad)
    crop_x2 = min(img_rgb.shape[1], x2 + pad)
    crop_y2 = min(img_rgb.shape[0], y2 + pad)
    
    person_crop = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    if person_crop.size == 0:
        return {"is_raincoat": False, "is_loose": False, "inflate_factor": 1.0, "confidence": 0.0}
    
    hsv = cv2.cvtColor(person_crop, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(person_crop, cv2.COLOR_RGB2GRAY)
    
    # 1. Saturation analysis - áo mưa thường có màu đồng nhất, saturation cao
    sat_mean = float(hsv[:, :, 1].mean())
    sat_std  = float(hsv[:, :, 1].std())
    
    # 2. Texture analysis - áo mưa texture thấp (vai đồng nhất)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture_score = float(np.abs(lap).mean())
    
    # 3. Color analysis - áo mưa thường xanh/cam/đỏ đồng nhất
    hue_mean = float(hsv[:, :, 0].mean())
    val_mean = float(hsv[:, :, 2].mean())
    
    # 4. Brightness consistency - áo mưa sáng đồng đều
    val_std = float(hsv[:, :, 2].std())
    
    is_raincoat = (
        sat_mean > RAINCOAT_SATURATION_THRESH and
        texture_score < RAINCOAT_TEXTURE_MAX and
        RAINCOAT_HUE_RANGE[0] <= hue_mean <= RAINCOAT_HUE_RANGE[1] and
        val_mean > RAINCOAT_BRIGHTNESS_MIN and
        val_std < 35  # màu đồng đều
    )
    
    # Loose clothing: texture thấp nhưng không đủ điều kiện áo mưa
    is_loose = (
        not is_raincoat and
        texture_score < 18 and
        sat_std < 25 and
        val_std < 40
    )
    
    inflate_factor = 1.0
    if is_raincoat:
        inflate_factor = RAINCOAT_SHOULDER_INFLATE
    elif is_loose:
        inflate_factor = LOOSE_CLOTHING_SHOULDER_INFLATE
    
    confidence = 0.0
    if is_raincoat:
        confidence = min(0.9, (sat_mean - 130) / 50 * 0.5 + (15 - texture_score) / 15 * 0.5)
    elif is_loose:
        confidence = min(0.7, (20 - texture_score) / 20 * 0.5 + (30 - sat_std) / 30 * 0.5)
    
    return {
        "is_raincoat": is_raincoat,
        "is_loose": is_loose,
        "inflate_factor": inflate_factor,
        "confidence": float(confidence),
        "sat_mean": sat_mean,
        "texture_score": texture_score,
    }


# [IMPROVE] Gender classification from pose/shoulder ratio
def _classify_gender_from_pose(kpts: Optional[np.ndarray], bbox: List[int], shoulder_cm: float) -> dict:
    """
    Phân loại giới tính từ pose và tỉ lệ vai/hông.
    Nam: vai rộng, hông hẹp → torso_ratio > 1.05
    Nữ: vai hẹp, hông rộng → torso_ratio < 1.0
    Returns: {"gender": "male"|"female"|"unknown", "confidence": float}
    """
    if kpts is None:
        return {"gender": "unknown", "confidence": 0.0, "torso_ratio": 1.0}
    
    # Tính tỉ lệ vai/hông từ keypoints
    ls_x, ls_y, ls_c = _kpt(kpts, KP_LEFT_SHLD)
    rs_x, rs_y, rs_c = _kpt(kpts, KP_RIGHT_SHLD)
    lh_x, lh_y, lh_c = _kpt(kpts, KP_LEFT_HIP)
    rh_x, rh_y, rh_c = _kpt(kpts, KP_RIGHT_HIP)
    
    shoulder_conf = (ls_c + rs_c) / 2 if (ls_c > KP_CONF_THRESH and rs_c > KP_CONF_THRESH) else max(ls_c, rs_c)
    hip_conf = (lh_c + rh_c) / 2 if (lh_c > KP_CONF_THRESH and rh_c > KP_CONF_THRESH) else max(lh_c, rh_c)
    
    if shoulder_conf < KP_CONF_THRESH or hip_conf < KP_CONF_THRESH:
        # Fallback: dùng shoulder_cm nếu có
        if shoulder_cm > 42:
            return {"gender": "male", "confidence": 0.55, "torso_ratio": 1.0}
        elif shoulder_cm < 36:
            return {"gender": "female", "confidence": 0.55, "torso_ratio": 1.0}
        return {"gender": "unknown", "confidence": 0.0, "torso_ratio": 1.0}
    
    shoulder_w = abs(rs_x - ls_x)
    hip_w = abs(rh_x - lh_x)
    
    if hip_w < 5:
        return {"gender": "unknown", "confidence": 0.0, "torso_ratio": 1.0}
    
    torso_ratio = shoulder_w / hip_w
    
    # Nam: vai rộng hơn hông (torso_ratio > 1.05)
    # Nữ: hông rộng hơn vai (torso_ratio < 1.0)
    if torso_ratio > 1.08:
        gender = "male"
        conf = min(0.85, (torso_ratio - 1.0) * 2.0)
    elif torso_ratio < 0.98:
        gender = "female"
        conf = min(0.85, (1.0 - torso_ratio) * 2.5)
    else:
        gender = "unknown"
        conf = 0.3
    
    return {"gender": gender, "confidence": float(conf), "torso_ratio": float(torso_ratio)}


# [IMPROVE] Child detection - trẻ em có tỷ lệ cơ thể khác người lớn
def _detect_child(estimated_height_cm: float, kpts: Optional[np.ndarray], bbox: List[int]) -> dict:
    """
    Phát hiện trẻ em dựa trên chiều cao ước tính và tỉ lệ đầu/cơ thể.
    Trẻ em: đầu to hơn theo tỷ lệ, chân ngắn hơn, vai hẹp hơn.
    Returns: {"is_child": bool, "confidence": float, "age_group": "infant|child|adolescent|adult"}
    """
    if estimated_height_cm >= CHILD_HEIGHT_THRESHOLD_CM:
        return {"is_child": False, "confidence": 0.9, "age_group": "adult"}
    
    # Dưới 140cm → có thể là trẻ em hoặc người lớn thấp
    is_child = True
    confidence = 0.7
    age_group = "child"
    
    if kpts is not None:
        # Kiểm tra tỉ lệ đầu/chân
        head_y = None
        for idx in [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE]:
            _, y, c = _kpt(kpts, idx)
            if c >= KP_CONF_THRESH:
                if head_y is None or y < head_y:
                    head_y = y
        
        ankle_y = None
        for idx in [KP_LEFT_ANKLE, KP_RIGHT_ANKLE]:
            _, y, c = _kpt(kpts, idx)
            if c >= KP_CONF_THRESH:
                if ankle_y is None or y > ankle_y:
                    ankle_y = y
        
        if head_y is not None and ankle_y is not None:
            body_px = ankle_y - head_y
            # Ước tính kích thước đầu từ tai-tai hoặc mắt-mắt
            le_x, _, le_c = _kpt(kpts, KP_LEFT_EAR)
            re_x, _, re_c = _kpt(kpts, KP_RIGHT_EAR)
            if le_c >= KP_CONF_THRESH and re_c >= KP_CONF_THRESH:
                head_w = abs(re_x - le_x)
                if body_px > 0:
                    head_body_ratio = head_w / body_px
                    # Trẻ em: head/body ratio > 0.18 (người lớn ~0.14)
                    if head_body_ratio > 0.18:
                        confidence = min(0.95, confidence + 0.2)
                        if estimated_height_cm < 100:
                            age_group = "infant"
                        elif estimated_height_cm < 130:
                            age_group = "child"
                        else:
                            age_group = "adolescent"
    
    if estimated_height_cm < 90:
        age_group = "infant"
        confidence = max(confidence, 0.85)
    elif estimated_height_cm < 120:
        age_group = "child"
        confidence = max(confidence, 0.8)
    elif estimated_height_cm < 140:
        age_group = "adolescent"
    
    return {"is_child": is_child, "confidence": float(confidence), "age_group": age_group}


class ShoulderWidthEstimator:
    """
    Ước tính chiều cao người qua vai — v2 với ensemble nhiều method.
    """

    # ── Public entry point ─────────────────────────────────────────────────
    def estimate(
        self,
        img_rgb:     np.ndarray,
        bbox:        List[int],
        keypoints:   Optional[np.ndarray] = None,   # (17, 3) [x, y, conf]
        depth_norm:  Optional[np.ndarray] = None,   # depth map 0-1
        pose_factor: float = 1.0,
    ) -> ShoulderEstimateResult:
        """
        Ước tính chiều cao người. Thử các method theo thứ tự ưu tiên,
        kết hợp ensemble khi có nhiều method đều cho kết quả.
        [IMPROVE] Phát hiện áo mưa/quần áo rộng để điều chỉnh shoulder width.
        """
        x1, y1, x2, y2 = bbox
        bh = max(y2 - y1, 1)
        bw = max(x2 - x1, 1)

        # [IMPROVE] Raincoat / loose clothing detection
        raincoat_info = _detect_raincoat_on_person(img_rgb, bbox, keypoints)
        inflate_factor = raincoat_info["inflate_factor"]

        # [IMPROVE] Gender classification for appropriate shoulder-to-height ratio
        gender_info = {"gender": "unknown", "confidence": 0.0, "torso_ratio": 1.0}
        # Sẽ gọi sau khi có shoulder_cm estimate (trong ensemble)

        # [IMPROVE] Vanishing point detection cho perspective-aware depth scale
        vp_result = None
        if depth_norm is not None:
            try:
                from depth_analysis.perspective_analyzer import PerspectiveAnalyzer
                pa = PerspectiveAnalyzer()
                vp_result = pa.detect_vanishing_point(img_rgb)
            except Exception:
                pass

        # [FIX v5] Upper Body Anatomical Ruler thay cho px_per_cm từ bbox.
        # px_per_cm_base = bh / DEFAULT_HEIGHT_CM — SAI khi người ngập một phần:
        #   bbox chỉ bao phần NỔI trên mặt nước (y2 = mực nước) → bh bị co
        #   xuống → px_per_cm quá nhỏ (sai 2-3 lần) → các method như
        #   _estimate_head_to_hip / _estimate_from_head_size lấy pixel chia
        #   ngược cho px_per_cm này → chiều cao luôn bị đội lên chạm trần
        #   DEFAULT_HEIGHT_MAX (200cm).
        # Mới: đo px_per_cm từ upper body (đầu→vai 25cm / biacromial 39cm)
        #   — phàn trên cơ thể LUÔN nổi trên nước, không bị ảnh hưởng bởi
        #   ngập → tỉ lệ pixel/cm chính xác tại đúng cự ly đó.
        px_per_cm_anatomy = self._anatomical_px_per_cm(keypoints)
        if px_per_cm_anatomy is not None:
            px_per_cm_base = px_per_cm_anatomy
        else:
            px_per_cm_base = bh / DEFAULT_HEIGHT_CM   # fallback khi không có kpts

        # Perspective scale từ depth_norm + vanishing point
        depth_scale = self._depth_scale(depth_norm, bbox, vp_result) if depth_norm is not None else 1.0
        px_per_cm   = px_per_cm_base * depth_scale

        results: List[Tuple[ShoulderEstimateResult, float]] = []  # (result, weight)

        # ── Method 1: Head-to-Ankle trực tiếp (chính xác nhất) ──────────
        r = self._estimate_head_to_ankle(keypoints, bbox, px_per_cm_base)
        if r is not None:
            results.append((r, 3.0))   # weight cao nhất

        # ── Method 2: Head-to-Hip (khi thấy hông, không thấy chân) ──────
        r = self._estimate_head_to_hip(keypoints, bbox, px_per_cm_base)
        if r is not None:
            results.append((r, 2.0))

        # ── Method 3: Head size (tai-tai hoặc mắt-mắt làm thước kẻ) ────
        r = self._estimate_from_head_size(keypoints, bbox, px_per_cm_base)
        if r is not None:
            results.append((r, 1.5))

        # ── Method 4: Keypoint vai (method gốc, cải tiến) ───────────────
        if bh >= MIN_BBOX_HEIGHT_FOR_SHOULDER and bw >= MIN_BBOX_WIDTH_FOR_SHOULDER:
            r = self._estimate_from_keypoints(keypoints, bbox, px_per_cm)
            if r is not None:
                results.append((r, 1.0))

        # ── Method 5: Neck detection ─────────────────────────────────────
        if not results and bh >= 60:
            r = self._estimate_from_neck(img_rgb, bbox, px_per_cm)
            if r is not None:
                results.append((r, 0.5))

        # ── Method 6: Upper body width ───────────────────────────────────
        if not results:
            r = self._estimate_from_upper_body(img_rgb, bbox, px_per_cm)
            if r is not None:
                results.append((r, 0.3))

        # ── Ensemble ─────────────────────────────────────────────────────
        if results:
            final = self._ensemble(results, inflate_factor, keypoints, bbox)
        else:
            final = ShoulderEstimateResult(
                estimated_height_cm=DEFAULT_HEIGHT_CM,
                shoulder_width_px=0, shoulder_width_cm=0,
                px_per_cm=px_per_cm, method="default",
                confidence=0.10, notes="no_method_succeeded",
            )

        # Pose factor adjustment
        final = self._adjust_for_pose(final, pose_factor)
        
        # [IMPROVE] Lưu raincoat info vào notes
        if raincoat_info["is_raincoat"] or raincoat_info["is_loose"]:
            final.notes += f" | raincoat={raincoat_info['is_raincoat']} loose={raincoat_info['is_loose']} inflate={inflate_factor:.2f}"
        
        return final

    # ── Method 1: Head-to-Ankle ──────────────────────────────────────────
    def _estimate_head_to_ankle(
        self,
        kpts: Optional[np.ndarray],
        bbox: List[int],
        px_per_cm_base: float,
    ) -> Optional[ShoulderEstimateResult]:
        """
        Khi thấy đầu (nose/mắt) VÀ mắt cá chân → tính chiều cao trực tiếp
        từ pixel distance, không cần giả định px/cm.

        Đây là method chính xác nhất vì không phụ thuộc vào px/cm giả định.
        """
        if kpts is None:
            return None

        # Tìm điểm đầu cao nhất (y nhỏ nhất)
        head_y = None
        head_conf = 0.0
        for idx in [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR]:
            x, y, c = _kpt(kpts, idx)
            if c >= KP_CONF_THRESH:
                if head_y is None or y < head_y:
                    head_y = y
                    head_conf = c

        if head_y is None:
            return None

        # Tìm mắt cá chân
        la_x, la_y, la_c = _kpt(kpts, KP_LEFT_ANKLE)
        ra_x, ra_y, ra_c = _kpt(kpts, KP_RIGHT_ANKLE)

        ankle_y = None
        ankle_conf = 0.0
        if la_c >= KP_CONF_THRESH and ra_c >= KP_CONF_THRESH:
            ankle_y    = (la_y + ra_y) / 2
            ankle_conf = (la_c + ra_c) / 2
        elif la_c >= KP_CONF_THRESH:
            ankle_y    = la_y
            ankle_conf = la_c
        elif ra_c >= KP_CONF_THRESH:
            ankle_y    = ra_y
            ankle_conf = ra_c

        if ankle_y is None:
            return None

        # Pixel distance đầu → mắt cá chân
        height_px = ankle_y - head_y
        if height_px < 30:   # quá nhỏ → không hợp lệ
            return None

        # Chiều cao thực: cộng thêm ~7% vì đỉnh đầu cao hơn nose/mắt
        # và đế chân thấp hơn mắt cá chân ~4%
        head_offset_ratio  = 0.07   # nose→top_of_head ≈ 7% height
        ankle_offset_ratio = 0.04   # ankle→sole ≈ 4% height
        total_offset = 1.0 + head_offset_ratio + ankle_offset_ratio

        height_cm_raw = height_px * total_offset / px_per_cm_base
        height_cm     = _clamp_height(height_cm_raw)

        # px_per_cm tính ngược từ kết quả thực tế (chuẩn hơn baseline)
        px_per_cm_real = height_px / (height_cm / total_offset)

        # Shoulder từ px_per_cm thực
        shoulder_cm_est = height_cm * SHOULDER_TO_HEIGHT_RATIO
        shoulder_px_est = shoulder_cm_est * px_per_cm_real

        conf = min(0.95, (head_conf + ankle_conf) / 2 * 0.90 + 0.10)

        log.debug(
            f"  ShoulderEst [head_to_ankle]: "
            f"height_px={height_px:.0f} → {height_cm:.0f}cm conf={conf:.2f}"
        )

        return ShoulderEstimateResult(
            estimated_height_cm = round(height_cm, 1),
            shoulder_width_px   = round(shoulder_px_est, 1),
            shoulder_width_cm   = round(shoulder_cm_est, 1),
            px_per_cm           = round(px_per_cm_real, 4),
            method              = "head_to_ankle",
            confidence          = round(conf, 3),
            notes               = f"h_px={height_px:.0f} offset={total_offset:.2f}",
        )

    # ── Method 2: Head-to-Hip ─────────────────────────────────────────────
    def _estimate_head_to_hip(
        self,
        kpts: Optional[np.ndarray],
        bbox: List[int],
        px_per_cm_base: float,
    ) -> Optional[ShoulderEstimateResult]:
        """
        Khi thấy đầu VÀ hông (nhưng không thấy mắt cá chân).
        Dùng tỉ lệ Dempster: đầu→hông = 43% chiều cao.
        """
        if kpts is None:
            return None

        # Chỉ dùng method này nếu KHÔNG thấy cả 2 mắt cá chân
        la_c = _kpt(kpts, KP_LEFT_ANKLE)[2]
        ra_c = _kpt(kpts, KP_RIGHT_ANKLE)[2]
        if la_c >= KP_CONF_THRESH and ra_c >= KP_CONF_THRESH:
            return None   # method head_to_ankle đã xử lý

        # Đầu (y nhỏ nhất)
        head_y = None
        head_conf = 0.0
        for idx in [KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE]:
            x, y, c = _kpt(kpts, idx)
            if c >= KP_CONF_THRESH:
                if head_y is None or y < head_y:
                    head_y = y
                    head_conf = c
        if head_y is None:
            return None

        # Hông
        lh_x, lh_y, lh_c = _kpt(kpts, KP_LEFT_HIP)
        rh_x, rh_y, rh_c = _kpt(kpts, KP_RIGHT_HIP)
        if lh_c < KP_CONF_THRESH and rh_c < KP_CONF_THRESH:
            return None

        if lh_c >= KP_CONF_THRESH and rh_c >= KP_CONF_THRESH:
            hip_y    = (lh_y + rh_y) / 2
            hip_conf = (lh_c + rh_c) / 2
        elif lh_c >= KP_CONF_THRESH:
            hip_y, hip_conf = lh_y, lh_c
        else:
            hip_y, hip_conf = rh_y, rh_c

        head_to_hip_px = hip_y - head_y
        if head_to_hip_px < 20:
            return None

        # Đầu nằm trên nose ~7% chiều cao
        head_offset    = 0.07
        # head_to_hip thực = SEG_HEAD_TO_HIP - head_offset = 0.430 - 0.07 = 0.360
        # Nhưng head_y là nose, không phải đỉnh đầu → bù thêm
        effective_ratio = SEG_HEAD_TO_HIP - head_offset   # ≈ 0.360

        height_px  = head_to_hip_px / effective_ratio
        height_cm  = _clamp_height(height_px / px_per_cm_base)

        shoulder_cm = height_cm * SHOULDER_TO_HEIGHT_RATIO
        shoulder_px = shoulder_cm * px_per_cm_base

        conf = min(0.80, (head_conf + hip_conf) / 2 * 0.70 + 0.10)

        log.debug(
            f"  ShoulderEst [head_to_hip]: "
            f"h→hip_px={head_to_hip_px:.0f} → height={height_cm:.0f}cm conf={conf:.2f}"
        )

        return ShoulderEstimateResult(
            estimated_height_cm = round(height_cm, 1),
            shoulder_width_px   = round(shoulder_px, 1),
            shoulder_width_cm   = round(shoulder_cm, 1),
            px_per_cm           = round(px_per_cm_base, 4),
            method              = "head_to_hip",
            confidence          = round(conf, 3),
            notes               = f"h_hip_px={head_to_hip_px:.0f} ratio={effective_ratio:.3f}",
        )

    # ── Method 3: Head size ───────────────────────────────────────────────
    def _estimate_from_head_size(
        self,
        kpts: Optional[np.ndarray],
        bbox: List[int],
        px_per_cm_base: float,
    ) -> Optional[ShoulderEstimateResult]:
        """
        Dùng kích thước đầu (tai-tai hoặc mắt-mắt) làm thước kẻ.
        Khoảng cách tai-tai ≈ 15% chiều cao.
        Khoảng cách mắt-mắt ≈ 6.5% chiều cao.
        """
        if kpts is None:
            return None

        estimates = []

        # Tai-tai (ear-to-ear) — đáng tin hơn
        le_x, le_y, le_c = _kpt(kpts, KP_LEFT_EAR)
        re_x, re_y, re_c = _kpt(kpts, KP_RIGHT_EAR)
        if le_c >= KP_CONF_THRESH and re_c >= KP_CONF_THRESH:
            ear_dist_px = abs(re_x - le_x)
            if ear_dist_px > 5:
                h_from_ear = ear_dist_px / HEAD_WIDTH_TO_HEIGHT_RATIO
                estimates.append((h_from_ear, (le_c + re_c) / 2, "ear"))

        # Mắt-mắt (eye-to-eye)
        lye_x, lye_y, lye_c = _kpt(kpts, KP_LEFT_EYE)
        rye_x, rye_y, rye_c = _kpt(kpts, KP_RIGHT_EYE)
        if lye_c >= KP_CONF_THRESH and rye_c >= KP_CONF_THRESH:
            eye_dist_px = abs(rye_x - lye_x)
            if eye_dist_px > 3:
                h_from_eye = eye_dist_px / EYE_WIDTH_TO_HEIGHT_RATIO
                estimates.append((h_from_eye, (lye_c + rye_c) / 2 * 0.85, "eye"))

        if not estimates:
            return None

        # Weighted average (confidence làm weight)
        total_w  = sum(c for _, c, _ in estimates)
        height_px = sum(h * c for h, c, _ in estimates) / total_w
        avg_conf  = total_w / len(estimates)
        methods   = "+".join(m for _, _, m in estimates)

        height_cm = _clamp_height(height_px / px_per_cm_base)
        shoulder_cm = height_cm * SHOULDER_TO_HEIGHT_RATIO
        shoulder_px = shoulder_cm * px_per_cm_base
        conf = min(0.75, avg_conf * 0.60 + 0.10)

        log.debug(
            f"  ShoulderEst [head_size/{methods}]: "
            f"h_px={height_px:.0f} → {height_cm:.0f}cm conf={conf:.2f}"
        )

        return ShoulderEstimateResult(
            estimated_height_cm = round(height_cm, 1),
            shoulder_width_px   = round(shoulder_px, 1),
            shoulder_width_cm   = round(shoulder_cm, 1),
            px_per_cm           = round(px_per_cm_base, 4),
            method              = f"head_size_{methods}",
            confidence          = round(conf, 3),
            notes               = f"methods={methods}",
        )

    # ── Method 4: Keypoint vai (cải tiến từ v1) ───────────────────────────
    def _estimate_from_keypoints(
        self,
        kpts:      Optional[np.ndarray],
        bbox:      List[int],
        px_per_cm: float,
    ) -> Optional[ShoulderEstimateResult]:
        """
        Dùng keypoint vai (kpt 5, 6). Cải tiến: dùng hip để validate,
        dùng shoulder_width thực thay vì bbox-based px_per_cm.
        """
        if kpts is None:
            return None

        ls_x, ls_y, ls_c = _kpt(kpts, KP_LEFT_SHLD)
        rs_x, rs_y, rs_c = _kpt(kpts, KP_RIGHT_SHLD)

        ls_valid = ls_c >= KP_CONF_THRESH
        rs_valid = rs_c >= KP_CONF_THRESH

        if not ls_valid and not rs_valid:
            return None

        x1, y1, x2, y2 = bbox
        torso_ratio = 1.0
        kp_left = kp_right = None

        if ls_valid and rs_valid:
            shoulder_width_px = abs(rs_x - ls_x)
            avg_conf          = (ls_c + rs_c) / 2
            kp_left           = (ls_x, ls_y)
            kp_right          = (rs_x, rs_y)

            # Kiểm tra tỉ lệ vai/chiều_cao hợp lý
            shoulder_ratio = shoulder_width_px / max(y2 - y1, 1)
            if shoulder_ratio < 0.15 or shoulder_ratio > 0.60:
                avg_conf *= 0.5

            # Hip validation
            lh_x, _, lh_c = _kpt(kpts, KP_LEFT_HIP)
            rh_x, _, rh_c = _kpt(kpts, KP_RIGHT_HIP)
            if lh_c >= KP_CONF_THRESH and rh_c >= KP_CONF_THRESH:
                hip_w   = abs(rh_x - lh_x)
                torso_ratio = shoulder_width_px / max(hip_w, 1)
                torso_ratio = np.clip(torso_ratio, 0.70, 1.40)
                if 0.80 < torso_ratio < 1.25:
                    avg_conf = min(0.95, avg_conf + 0.08)

        elif ls_valid:
            cx = (x1 + x2) / 2
            shoulder_width_px = abs(ls_x - cx) * 2
            avg_conf = ls_c * 0.65
            kp_left  = (ls_x, ls_y)
        else:
            cx = (x1 + x2) / 2
            shoulder_width_px = abs(rs_x - cx) * 2
            avg_conf = rs_c * 0.65
            kp_right = (rs_x, rs_y)

        if shoulder_width_px < 5:
            return None

        shoulder_cm  = shoulder_width_px / px_per_cm
        shoulder_cm  = max(28.0, min(55.0, shoulder_cm))
        height_ratio = _get_height_ratio(shoulder_cm, torso_ratio)
        height_cm    = _clamp_height(shoulder_cm / height_ratio)

        conf = min(0.90, avg_conf * 0.80 + 0.15 * (1 if 30 < shoulder_cm < 50 else 0.2))

        log.debug(
            f"  ShoulderEst [keypoint]: "
            f"sh={shoulder_width_px:.0f}px={shoulder_cm:.0f}cm → {height_cm:.0f}cm conf={conf:.2f}"
        )

        return ShoulderEstimateResult(
            estimated_height_cm = round(height_cm, 1),
            shoulder_width_px   = round(shoulder_width_px, 1),
            shoulder_width_cm   = round(shoulder_cm, 1),
            px_per_cm           = round(px_per_cm, 4),
            method              = "keypoint_shoulder",
            confidence          = round(conf, 3),
            kp_left             = kp_left,
            kp_right            = kp_right,
            notes               = f"torso_ratio={torso_ratio:.2f} sh={shoulder_cm:.0f}cm",
        )

    # ── Method 5: Neck detection ──────────────────────────────────────────
    def _estimate_from_neck(
        self,
        img_rgb:   np.ndarray,
        bbox:      List[int],
        px_per_cm: float,
    ) -> Optional[ShoulderEstimateResult]:
        """Phân tích hình ảnh vùng cổ để ước tính vai."""
        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        bw = x2 - x1
        if bh < 40 or bw < 15:
            return None
        try:
            roi  = img_rgb[y1:y2, max(0, x1):min(img_rgb.shape[1], x2)]
            if roi.shape[0] < 20 or roi.shape[1] < 10:
                return None

            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            h_roi = gray.shape[0]

            neck_start = max(0, int(h_roi * 0.12))
            neck_end   = min(h_roi, int(h_roi * 0.32))
            shld_start = max(0, int(h_roi * 0.28))
            shld_end   = min(h_roi, int(h_roi * 0.45))

            def row_widths_in(start, end):
                ws = []
                for row in range(start, end, max(1, (end - start) // 20)):
                    cols = np.nonzero(binary[row] > 128)[0]
                    if len(cols) >= 5:
                        ws.append(cols[-1] - cols[0])
                return ws

            nw = row_widths_in(neck_start, neck_end)
            sw = row_widths_in(shld_start, shld_end)

            if len(nw) < 2 or len(sw) < 2:
                return None

            neck_min_w = float(np.percentile(nw, 20))
            shld_max_w = float(np.percentile(sw, 80))

            if shld_max_w < neck_min_w * 1.20:
                return None

            shoulder_px  = shld_max_w * BBOX_SHOULDER_RATIO
            shoulder_cm  = max(28.0, min(55.0, shoulder_px / px_per_cm))
            height_cm    = _clamp_height(shoulder_cm / _get_height_ratio(shoulder_cm))
            ratio        = shld_max_w / max(neck_min_w, 1)
            conf         = min(0.65, 0.30 + (ratio - 1.2) * 0.20)

            return ShoulderEstimateResult(
                estimated_height_cm = round(height_cm, 1),
                shoulder_width_px   = round(shoulder_px, 1),
                shoulder_width_cm   = round(shoulder_cm, 1),
                px_per_cm           = round(px_per_cm, 4),
                method              = "neck",
                confidence          = round(conf, 3),
                notes               = f"neck/sh ratio={ratio:.2f}",
            )
        except Exception as e:
            log.debug(f"  Neck estimation failed: {e}")
            return None

    # ── Method 6: Upper body width ────────────────────────────────────────
    def _estimate_from_upper_body(
        self,
        img_rgb:   np.ndarray,
        bbox:      List[int],
        px_per_cm: float,
    ) -> Optional[ShoulderEstimateResult]:
        """Fallback: dùng độ rộng phần thân trên."""
        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        bw = x2 - x1
        if bh < 30 or bw < 10:
            return None
        try:
            roi  = img_rgb[y1:y2, max(0, x1):min(img_rgb.shape[1], x2)]
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            h_roi = gray.shape[0]

            s1 = max(0, int(h_roi * 0.28))
            s2 = min(h_roi, int(h_roi * 0.42))
            if s2 <= s1:
                return None

            band = binary[s1:s2]
            ws   = []
            for row in range(band.shape[0]):
                cols = np.nonzero(band[row] > 128)[0]
                if len(cols) >= 5:
                    ws.append(cols[-1] - cols[0])
            if len(ws) < 3:
                return None

            shoulder_px = float(np.percentile(ws, 70)) * BBOX_SHOULDER_RATIO
            shoulder_cm = max(28.0, min(55.0, shoulder_px / px_per_cm))
            height_cm   = _clamp_height(shoulder_cm / _get_height_ratio(shoulder_cm))

            return ShoulderEstimateResult(
                estimated_height_cm = round(height_cm, 1),
                shoulder_width_px   = round(shoulder_px, 1),
                shoulder_width_cm   = round(shoulder_cm, 1),
                px_per_cm           = round(px_per_cm, 4),
                method              = "upper_body",
                confidence          = 0.35,
                notes               = f"ub_px={shoulder_px:.0f}",
            )
        except Exception as e:
            log.debug(f"  Upper-body estimation failed: {e}")
            return None

    # ── Ensemble ──────────────────────────────────────────────────────────
    def _ensemble(
        self,
        results: List[Tuple[ShoulderEstimateResult, float]],
        inflate_factor: float = 1.0,
        keypoints: Optional[np.ndarray] = None,
        bbox: Optional[List[int]] = None,
    ) -> ShoulderEstimateResult:
        """
        Kết hợp các method bằng weighted average.
        Weight = method_weight * confidence.
        [IMPROVE] Apply inflate_factor correction for raincoat/loose clothing.
        """
        if len(results) == 1:
            r = results[0][0]
            # Apply inflate_factor correction
            if inflate_factor != 1.0:
                corrected_shoulder_cm = r.shoulder_width_cm / inflate_factor
                corrected_height_cm = corrected_shoulder_cm / SHOULDER_TO_HEIGHT_RATIO
                return ShoulderEstimateResult(
                    estimated_height_cm = round(_clamp_height(corrected_height_cm), 1),
                    shoulder_width_px   = round(r.shoulder_width_px / inflate_factor, 1),
                    shoulder_width_cm   = round(corrected_shoulder_cm, 1),
                    px_per_cm           = r.px_per_cm,
                    method              = r.method + "_raincoat_corrected",
                    confidence          = r.confidence,
                    notes               = r.notes + f" inflate_corrected={inflate_factor:.2f}",
                )
            return r

        total_w  = 0.0
        height_sum = 0.0
        methods  = []
        best_conf = 0.0
        best_r    = results[0][0]

        for r, w in results:
            eff_w    = w * r.confidence
            height_sum += r.estimated_height_cm * eff_w
            total_w  += eff_w
            methods.append(r.method)
            if r.confidence > best_conf:
                best_conf = r.confidence
                best_r    = r

        if total_w < 1e-9:
            return best_r

        final_height = _clamp_height(height_sum / total_w)

        # Confidence ensemble: dùng confidence method tốt nhất nhưng giảm chút
        # nếu các method cho kết quả khác nhau nhiều
        heights = [r.estimated_height_cm for r, _ in results]
        spread  = max(heights) - min(heights)
        conf    = best_conf * max(0.70, 1.0 - spread / 60.0)

        # [IMPROVE] Child detection - sử dụng child-specific ratios
        child_info = _detect_child(final_height, keypoints, bbox or [0, 0, 0, 0])
        if child_info["is_child"]:
            # Trẻ em: dùng shoulder-to-height ratio khác
            sh_ratio = CHILD_SHOULDER_TO_HEIGHT
            final_height = _clamp_height(final_height)  # keep height
            conf = max(0.1, conf - 0.15)  # giảm confidence chút do uncertainty
            log.debug(f"  Child detected: age_group={child_info['age_group']} height={final_height:.0f}cm")

        # [IMPROVE] Gender classification for appropriate shoulder-to-height ratio
        # Sử dụng shoulder_cm từ best_r để classify gender
        gender_info = _classify_gender_from_pose(
            keypoints, bbox or [0, 0, 0, 0], best_r.shoulder_width_cm
        )
        
        # Chọn shoulder-to-height ratio phù hợp
        if gender_info["gender"] == "male":
            sh_ratio = SHOULDER_TO_HEIGHT_RATIO_VN_M
        elif gender_info["gender"] == "female":
            sh_ratio = SHOULDER_TO_HEIGHT_RATIO_VN_F
        else:
            sh_ratio = SHOULDER_TO_HEIGHT_RATIO_VN
        
        # Apply adaptive ratio based on shoulder size and torso ratio
        adaptive_ratio = _get_height_ratio(best_r.shoulder_width_cm, gender_info["torso_ratio"])
        # Combine with gender-specific ratio
        final_ratio = (sh_ratio + adaptive_ratio) / 2.0

        # Recompute shoulder từ final height
        shoulder_cm = final_height * final_ratio

        # [IMPROVE] Apply inflate_factor correction to shoulder
        if inflate_factor != 1.0:
            shoulder_cm = shoulder_cm / inflate_factor

        return ShoulderEstimateResult(
            estimated_height_cm = round(final_height, 1),
            shoulder_width_px   = round(best_r.shoulder_width_px / inflate_factor, 1),
            shoulder_width_cm   = round(shoulder_cm, 1),
            px_per_cm           = best_r.px_per_cm,
            method              = "ensemble_" + "+".join(methods),
            confidence          = round(conf, 3),
            notes               = f"methods={'+'.join(methods)} inflate={inflate_factor:.2f}",
        )

    # ── [v5] Upper Body Anatomical Ruler ───────────────────────────────
    def _anatomical_px_per_cm(self, keypoints: Optional[np.ndarray]) -> Optional[float]:
        """
        Đo px_per_cm từ PHẦN TRÊN CƠ THỂ — vùng LUÔN nổi trên mặt nước.

        Phục vụ đề xuất "Upper Body Anatomical Ruler":
          - đầu → vai (đỉnh đầu→acromion) ≈ 25cm
          - chiều rộng 2 mỏm vai (biacromial) ≈ 39cm
        Không phụ thuộc bbox → không bị sai khi người ngập một phần
        (bbox chỉ bao phần nổi → bh/165 bị co 2-3 lần).

        Args:
            keypoints: (17, 3) [x, y, conf] or None

        Returns:
            px_per_cm (pixels per cm) tại cự ly đó, None nếu không đủ tin cậy.
        """
        if keypoints is None or keypoints.shape[0] < 17:
            return None

        # 1) Biacromial width → px_per_cm = width_px / 39cm
        shoulder_est = None
        ls_c, rs_c = _kpt(keypoints, KP_LEFT_SHLD)[2], _kpt(keypoints, KP_RIGHT_SHLD)[2]
        if ls_c >= KP_CONF_THRESH and rs_c >= KP_CONF_THRESH:
            width_px = abs(_kpt(keypoints, KP_RIGHT_SHLD)[0] - _kpt(keypoints, KP_LEFT_SHLD)[0])
            if width_px >= 8:
                shoulder_est = (width_px / BIACROMIAL_CM, min(ls_c, rs_c))

        # 2) Đầu → vai ≈ 25cm
        head_est = None
        head_y, head_conf = None, 0.0
        for idx in (KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR):
            x, y, c = _kpt(keypoints, idx)
            if c >= KP_CONF_THRESH:
                if head_y is None or y < head_y:
                    head_y, head_conf = y, c
        sh_y, sh_conf = None, 0.0
        for idx in (KP_LEFT_SHLD, KP_RIGHT_SHLD):
            _, y, c = _kpt(keypoints, idx)
            if c >= KP_CONF_THRESH:
                if sh_y is None or y < sh_y:
                    sh_y, sh_conf = y, c
        if head_y is not None and sh_y is not None and (sh_y - head_y) >= 5:
            head_est = ((sh_y - head_y) / HEAD_TO_CLAVICLE_CM,
                        min(head_conf, sh_conf))

        candidates = [c for c in (shoulder_est, head_est) if c]
        if not candidates:
            return None

        best = max(candidates, key=lambda c: c[1])   # chọn nguồn conf cao nhất
        px_per_cm = best[0]
        if not (PX_PER_CM_MIN <= px_per_cm <= PX_PER_CM_MAX):
            return None
        return px_per_cm

    # ── Depth scale ───────────────────────────────────────────────────────
    def _depth_scale(self, depth_norm: np.ndarray, bbox: List[int], vp_result: Optional[dict] = None) -> float:
        """
        Ước tính scale correction từ depth map.
        Người ở xa (depth cao) → px/cm thấp → cần scale lên.
        
        [IMPROVE] Perspective-aware scaling:
        - Dùng vanishing point nếu có để tính scale chính xác hơn
        - Scale = f(depth, distance_from_vp) thay vì chỉ depth đơn thuần
        """
        try:
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            h, w = depth_norm.shape[:2]
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))

            # Lấy depth trung bình vùng giữa người (5x5 pixels)
            d_patch = depth_norm[
                max(0, cy - 2):min(h, cy + 3),
                max(0, cx - 2):min(w, cx + 3),
            ]
            depth_val = float(np.median(d_patch)) if d_patch.size > 0 else 0.5

            # [IMPROVE] Perspective-aware scaling:
            # Nếu có vanishing point → tính scale dựa trên vị trí tương đối đến VP
            if vp_result is not None and vp_result.get("vp_confidence", 0) > 0.3:
                vp_y = vp_result.get("vp_y", h / 3)
                scale_rate = vp_result.get("scale_rate", 1.0)
                
                # Scale theo ground plane: scale tăng từ VP xuống dưới
                if vp_y < cy:
                    rel_pos = (cy - vp_y) / max(h - vp_y, 1)
                    perspective_scale = 1.0 + scale_rate * rel_pos
                    # Kết hợp depth scale + perspective scale
                    depth_scale = 1.3 - 0.6 * depth_val
                    scale = (depth_scale + perspective_scale) / 2.0
                else:
                    # VP ở trên người → người rất gần
                    scale = 1.3 - 0.6 * depth_val
            else:
                # Fallback: linear depth scale
                scale = 1.3 - 0.6 * depth_val
            
            return max(0.5, min(2.0, scale))
        except Exception:
            return 1.0

    # ── Pose adjustment ───────────────────────────────────────────────────
    def _adjust_for_pose(self, r: ShoulderEstimateResult, pose_factor: float) -> ShoulderEstimateResult:
        """Fine-tune theo pose_factor (người ngồi, cúi → pose_factor < 1)."""
        if pose_factor <= 0 or pose_factor > 1.2:
            return r
        if pose_factor < 0.92:
            adj = 1.0 + (1.0 - pose_factor) * POSE_FACTOR_ADJUST_WEIGHT * 0.75
            r.estimated_height_cm = _clamp_height(r.estimated_height_cm * adj)
            r.confidence *= (0.70 + pose_factor * 0.30)
            r.notes += f"; pose_adj={adj:.3f}"
        if r.px_per_cm < 0.35 or r.px_per_cm > 15.0:
            r.confidence *= 0.70
            r.notes += "; risk_scale"
        r.confidence = round(max(0.0, min(1.0, r.confidence)), 3)
        return r


# ── Singleton ─────────────────────────────────────────────────────────────────
_estimator: Optional[ShoulderWidthEstimator] = None

def get_shoulder_estimator() -> ShoulderWidthEstimator:
    global _estimator
    if _estimator is None:
        _estimator = ShoulderWidthEstimator()
    return _estimator

def estimate_height_from_shoulder(
    img_rgb:   np.ndarray,
    bbox:      List[int],
    keypoints: Optional[np.ndarray] = None,
) -> Tuple[float, float, str]:
    """Convenience function. Returns (height_cm, confidence, method)."""
    est = get_shoulder_estimator()
    r   = est.estimate(img_rgb, bbox, keypoints)
    return r.estimated_height_cm, r.confidence, r.method
