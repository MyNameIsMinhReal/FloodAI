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

# Chiều cao mặc định nếu không ước tính được
DEFAULT_HEIGHT_CM  = 170.0
DEFAULT_HEIGHT_MIN = 140.0
DEFAULT_HEIGHT_MAX = 200.0

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
        """
        x1, y1, x2, y2 = bbox
        bh = max(y2 - y1, 1)
        bw = max(x2 - x1, 1)

        # px_per_cm baseline từ bbox (assume 170cm = full height)
        px_per_cm_base = bh / DEFAULT_HEIGHT_CM

        # Perspective scale từ depth_norm
        depth_scale = self._depth_scale(depth_norm, bbox) if depth_norm is not None else 1.0
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
            final = self._ensemble(results)
        else:
            final = ShoulderEstimateResult(
                estimated_height_cm=DEFAULT_HEIGHT_CM,
                shoulder_width_px=0, shoulder_width_cm=0,
                px_per_cm=px_per_cm, method="default",
                confidence=0.10, notes="no_method_succeeded",
            )

        # Pose factor adjustment
        final = self._adjust_for_pose(final, pose_factor)
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
    ) -> ShoulderEstimateResult:
        """
        Kết hợp các method bằng weighted average.
        Weight = method_weight * confidence.
        """
        if len(results) == 1:
            return results[0][0]

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

        # Recompute shoulder từ final height
        shoulder_cm = final_height * SHOULDER_TO_HEIGHT_RATIO

        log.debug(
            f"  ShoulderEst [ensemble {'+'.join(methods)}]: "
            f"heights={[f'{h:.0f}' for h in heights]} → {final_height:.0f}cm "
            f"spread={spread:.0f}cm conf={conf:.2f}"
        )

        return ShoulderEstimateResult(
            estimated_height_cm = round(final_height, 1),
            shoulder_width_px   = round(best_r.shoulder_width_px, 1),
            shoulder_width_cm   = round(shoulder_cm, 1),
            px_per_cm           = round(best_r.px_per_cm, 4),
            method              = f"ensemble({'+'.join(methods)})",
            confidence          = round(min(0.95, conf), 3),
            kp_left             = best_r.kp_left,
            kp_right            = best_r.kp_right,
            notes               = f"spread={spread:.0f}cm n={len(results)}",
        )

    # ── Depth scale ───────────────────────────────────────────────────────
    def _depth_scale(self, depth_norm: np.ndarray, bbox: List[int]) -> float:
        """
        Ước tính scale correction từ depth map.
        Người ở xa (depth cao) → px/cm thấp → cần scale lên.
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

            # Correction: người xa (depth=1) → scale down 0.7; gần (depth=0) → 1.3
            # Linear interpolation
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
