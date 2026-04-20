# -*- coding: utf-8 -*-
"""
depth_analysis/water_detector.py  —  v3 (nâng cấp toàn diện)
================================================================
COLOR SPACES: HSV (15 profiles) + LAB + YCbCr + Normalized RGB
MỚI: WaterColorProfile, turbidity_score, foam/oil/night detection
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List
import cv2
import numpy as np

log = logging.getLogger(__name__)

# ─── 15 HSV Water Color Profiles ─────────────────────────────────────────────
HSV_WATER_PROFILES: Dict[str, dict] = {
    # Nước trong
    "clear_blue":       {"hsv": ((90,  35,  35), (135, 255, 255)), "name_vi": "Nước trong xanh",     "turbidity": 0.10},
    "teal_shallow":     {"hsv": ((80,  18,  40), (100, 130, 210)), "name_vi": "Nước cạn xanh ngọc",  "turbidity": 0.20},
    "greenish":         {"hsv": ((60,  25,  35), (85,  220, 225)), "name_vi": "Nước xanh lá",         "turbidity": 0.30},
    # Nước bùn / lũ
    "muddy_brown":      {"hsv": ((5,   45,  25), (22,  230, 195)), "name_vi": "Nước bùn nâu",         "turbidity": 0.85},
    "muddy_orange":     {"hsv": ((14,  55,  40), (30,  245, 215)), "name_vi": "Nước bùn cam",         "turbidity": 0.90},
    "reddish_clay":     {"hsv": ((0,   60,  35), (12,  240, 200)), "name_vi": "Nước đất sét đỏ",      "turbidity": 0.95},
    "yellowish_flood":  {"hsv": ((20,  50,  80), (38,  230, 240)), "name_vi": "Nước lũ vàng đất",    "turbidity": 0.80},
    "deep_brown":       {"hsv": ((8,   80,  15), (20,  255, 140)), "name_vi": "Nước bùn nâu sẫm",    "turbidity": 1.00},
    # Nước đục / xám
    "turbid_gray":      {"hsv": ((0,    0,  50), (180,  40, 180)), "name_vi": "Nước xám đục",         "turbidity": 0.75},
    "ash_gray":         {"hsv": ((0,    0,  80), (180,  25, 210)), "name_vi": "Nước tro xám nhạt",    "turbidity": 0.60},
    # Nước tối / ban đêm
    "dark_water":       {"hsv": ((90,  20,  10), (135, 200, 110)), "name_vi": "Nước tối",             "turbidity": 0.50},
    "night_reflection": {"hsv": ((90,   8,  15), (135, 120,  80)), "name_vi": "Nước đêm phản chiếu", "turbidity": 0.40},
    # Đặc biệt
    "foam_white":       {"hsv": ((0,    0, 180), (180,  30, 255)), "name_vi": "Bọt sóng trắng",       "turbidity": 0.20},
    "algae_green":      {"hsv": ((50,  40,  25), (75,  220, 180)), "name_vi": "Nước tảo xanh",        "turbidity": 0.65},
    "oil_dark":         {"hsv": ((20,  30,  10), (60,  180, 100)), "name_vi": "Nước nhiễm dầu",       "turbidity": 1.00},
}

LAB_WATER_PROFILES = {
    "lab_wet":   {"lab": ((20, 128, 125), (255, 142, 145)), "name_vi": "Bề mặt ướt"},
    "lab_muddy": {"lab": ((30, 130, 130), (200, 155, 158)), "name_vi": "Nước bùn LAB"},
    "lab_turbid":{"lab": ((40, 122, 120), (190, 138, 142)), "name_vi": "Nước đục LAB"},
    "lab_dark":  {"lab": ((10, 126, 120), (80,  136, 135)), "name_vi": "Nước tối LAB"},
}

YCBCR_WATER_PROFILES = {
    "ycbcr_muddy":  {"ycbcr": ((60, 100, 120), (180, 130, 148)), "name_vi": "Nước bùn YCbCr"},
    "ycbcr_turbid": {"ycbcr": ((80, 116, 124), (200, 132, 142)), "name_vi": "Nước đục YCbCr"},
    "ycbcr_clear":  {"ycbcr": ((50, 108, 100), (200, 126, 128)), "name_vi": "Nước trong YCbCr"},
}


@dataclass
class WaterColorProfile:
    dominant_type:    str
    dominant_name_vi: str
    turbidity:        float          # 0=trong .. 1=đục
    is_foam:          bool
    is_oil:           bool
    is_nighttime:     bool
    detected_profiles: List[str]
    color_confidence: float
    channel_scores:   Dict[str, float] = field(default_factory=dict)


@dataclass
class WaterDetectionResult:
    water_mask:      np.ndarray
    water_line_y:    int
    water_area_pct:  float
    water_type:      str
    confidence:      float
    has_reflection:  bool
    color_profile:   WaterColorProfile
    turbidity_score: float
    water_level_pct: float           # 0-1, tỉ lệ mực nước trong khung hình
    scene_context:   dict  = field(default_factory=dict)  # vegetation_pct, sky_pct, roof_pct
    has_puddle:      bool  = False   # phát hiện vũng nước nhỏ (< 15cm) trên mặt đất
    puddle_area_pct: float = 0.0     # % diện tích vũng nước trong ảnh


class WaterDetector:
    def __init__(
        self,
        min_water_area: float = 0.03,
        use_reflection: bool  = True,
        use_texture:    bool  = True,
        use_lab:        bool  = True,
        use_ycbcr:      bool  = True,
        use_norm_rgb:   bool  = True,
    ):
        self.min_water_area = min_water_area
        self.use_reflection = use_reflection
        self.use_texture    = use_texture
        self.use_lab        = use_lab
        self.use_ycbcr      = use_ycbcr
        self.use_norm_rgb   = use_norm_rgb

    def detect(self, img_rgb: np.ndarray) -> WaterDetectionResult:
        h, w    = img_rgb.shape[:2]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        ycbcr   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

        ch: Dict[str, float] = {}

        hsv_mask, hsv_scores, dom = self._detect_hsv(hsv, h, w)
        ch["hsv"] = float(hsv_mask.sum() / 255) / (h * w)

        lab_mask = self._detect_lab(lab) if self.use_lab else np.zeros((h, w), np.uint8)
        if self.use_lab: ch["lab"] = float(lab_mask.sum() / 255) / (h * w)

        ycbcr_mask = self._detect_ycbcr(ycbcr) if self.use_ycbcr else np.zeros((h, w), np.uint8)
        if self.use_ycbcr: ch["ycbcr"] = float(ycbcr_mask.sum() / 255) / (h * w)

        norm_mask = self._detect_norm_rgb(img_rgb) if self.use_norm_rgb else np.zeros((h, w), np.uint8)
        if self.use_norm_rgb: ch["norm_rgb"] = float(norm_mask.sum() / 255) / (h * w)

        tex_mask = self._detect_by_texture(img_bgr) if self.use_texture else np.zeros((h, w), np.uint8)
        if self.use_texture: ch["texture"] = float(tex_mask.sum() / 255) / (h * w)

        # [MỚI v4] NDWI channel
        ndwi_mask = self._ndwi_mask(img_rgb)
        ch["ndwi"] = float(ndwi_mask.sum() / 255) / (h * w)

        # [MỚI v4] Shadow exclusion mask — loại bỏ false positive từ bóng đổ
        shadow_excl = self._shadow_exclusion_mask(img_bgr)

        # ── [CẢI TIẾN v4] Voting thay vì pure OR ────────────────────────────
        # Mỗi pixel nhận 1 vote từ mỗi channel phát hiện nó là nước.
        # Ngưỡng: cần ít nhất 2 channels đồng ý → giảm false positive đáng kể.
        all_masks = [hsv_mask, lab_mask, ycbcr_mask, norm_mask, tex_mask, ndwi_mask]
        vote_map  = np.zeros((h, w), dtype=np.int32)
        for m in all_masks:
            vote_map += (m > 0).astype(np.int32)

        # HSV được tin tưởng hơn (15 profiles chuyên biệt) → count double
        vote_map += (hsv_mask > 0).astype(np.int32)
        # NDWI cũng có độ tin cậy cao → count double
        vote_map += (ndwi_mask > 0).astype(np.int32)

        # Threshold: ≥ 2 votes = water (trong 8 total votes kể cả double)
        voted = (vote_map >= 2).astype(np.uint8) * 255

        # Áp dụng shadow exclusion
        not_shadow = cv2.bitwise_not(shadow_excl)
        voted      = cv2.bitwise_and(voted, not_shadow)

        # Loại trừ vùng thực vật rõ ràng (cây, cỏ, lá) — không phải nước
        veg_excl = self._vegetation_mask(img_bgr)
        voted    = cv2.bitwise_and(voted, cv2.bitwise_not(veg_excl))

        k9 = np.ones((9, 9), np.uint8)
        cleaned = cv2.morphologyEx(voted, cv2.MORPH_CLOSE, k9)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k9)

        # Giảm ngưỡng min_area để bắt được vũng nước nhỏ (0.3% thay vì 0.5%)
        min_area = int(h * w * 0.003)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(cleaned)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)

        # Puddle pass: phát hiện vũng nước nhỏ bị lọc bởi main pipeline
        has_puddle, puddle_mask, puddle_area_pct_raw = self._detect_puddles(img_bgr, h, w)
        if has_puddle:
            # Chỉ thêm puddle pixels chưa có trong mask chính
            puddle_new = cv2.bitwise_and(puddle_mask, cv2.bitwise_not(filtered))
            filtered   = cv2.bitwise_or(filtered, puddle_new)

        water_area_pct = float(filtered.sum() / 255) / (h * w)

        has_reflection = False
        if self.use_reflection and water_area_pct > 0.05:
            has_reflection = self._detect_reflection(img_bgr, filtered)
            ch["reflection"] = 1.0 if has_reflection else 0.0

        water_line_y, wl_conf = self._find_water_line(filtered, h, w)
        water_level_pct = 1.0 - (water_line_y / max(h, 1))

        turbidity = self._calc_turbidity(dom, hsv_scores)
        scene_ctx = self._compute_scene_context(img_bgr, h, w)

        # [MỚI v4] Tính fragmentation để dùng trong confidence
        frag_score = self._compute_fragmentation(filtered, h, w)
        ch["fragmentation"] = frag_score

        cp = WaterColorProfile(
            dominant_type    = dom.get("type", "none"),
            dominant_name_vi = dom.get("name_vi", "Không rõ"),
            turbidity        = turbidity,
            is_foam          = "foam" in dom.get("type", ""),
            is_oil           = "oil"  in dom.get("type", ""),
            is_nighttime     = any(k in dom.get("type", "") for k in ("night", "dark")),
            detected_profiles= list(hsv_scores.keys()),
            color_confidence = min(1.0, ch.get("hsv", 0) * 3 + ch.get("lab", 0) * 2),
            channel_scores   = ch,
        )

        confidence = self._calc_confidence(water_area_pct, wl_conf, has_reflection,
                                           dom.get("type", "none"), ch, frag_score)
        water_type = dom.get("type", "none") if water_area_pct >= self.min_water_area else "none"

        return WaterDetectionResult(
            water_mask      = filtered,
            water_line_y    = water_line_y,
            water_area_pct  = round(water_area_pct * 100, 2),
            water_type      = water_type,
            confidence      = round(confidence, 3),
            has_reflection  = has_reflection,
            color_profile   = cp,
            turbidity_score = round(turbidity, 3),
            water_level_pct = round(water_level_pct, 3),
            scene_context   = scene_ctx,
            has_puddle      = has_puddle,
            puddle_area_pct = round(puddle_area_pct_raw * 100, 2),
        )

    # ── Detectors ─────────────────────────────────────────────────────────────

    def _detect_hsv(self, hsv, h, w):
        result, scores = np.zeros((h, w), np.uint8), {}
        for key, prof in HSV_WATER_PROFILES.items():
            lo, hi = prof["hsv"]
            mask   = cv2.inRange(hsv, np.array(lo), np.array(hi))
            sc     = float(mask.sum() / 255)
            if sc > 0:
                scores[key] = sc
                result = cv2.bitwise_or(result, mask)
        if not scores:
            return result, scores, {"type": "none", "name_vi": "Không có nước"}
        dom_key  = max(scores, key=scores.get)
        dom_prof = HSV_WATER_PROFILES[dom_key]
        return result, scores, {"type": dom_key, "name_vi": dom_prof["name_vi"]}

    def _detect_lab(self, lab):
        r = np.zeros(lab.shape[:2], np.uint8)
        for _, p in LAB_WATER_PROFILES.items():
            lo, hi = p["lab"]
            r = cv2.bitwise_or(r, cv2.inRange(lab, np.array(lo), np.array(hi)))
        return r

    def _detect_ycbcr(self, ycbcr):
        r = np.zeros(ycbcr.shape[:2], np.uint8)
        for _, p in YCBCR_WATER_PROFILES.items():
            lo, hi = p["ycbcr"]
            r = cv2.bitwise_or(r, cv2.inRange(ycbcr, np.array(lo), np.array(hi)))
        return r

    def _detect_norm_rgb(self, img_rgb):
        img_f = img_rgb.astype(np.float32) + 1e-6
        total = img_f.sum(axis=2, keepdims=True)
        n     = img_f / total
        r, g, b = n[:,:,0], n[:,:,1], n[:,:,2]
        clear  = ((r < 0.30) & (b > 0.33)).astype(np.uint8) * 255
        muddy  = ((r > 0.37) & (g > 0.28) & (g < 0.45)).astype(np.uint8) * 255
        gray_w = ((r>0.27)&(r<0.39)&(g>0.27)&(g<0.39)&(b>0.27)&(b<0.39)).astype(np.uint8)*255
        combined = cv2.bitwise_or(cv2.bitwise_or(clear, muddy), gray_w)
        h = combined.shape[0]
        mask = np.zeros_like(combined)
        mask[h//3:, :] = 255
        return cv2.bitwise_and(combined, mask)

    def _detect_by_texture(self, img_bgr):
        """
        Texture-based water detection.
        Cải tiến v4:
          - Ngưỡng sig < 12 (thay vì 20) — chặt hơn, ít false positive
          - Loại bỏ sky (top 15% ảnh) — tránh nhầm bầu trời
          - Loại bỏ shadow bằng gradient consistency
          - Yêu cầu vùng low-texture phải đủ liên tục (connected region)
        """
        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w   = gray.shape
        result = np.zeros((h, w), np.uint8)

        # [CẢI TIẾN] Ngưỡng chặt hơn: sig < 12 thay vì 20
        for ks in (11, 17, 25):
            mu   = cv2.blur(gray, (ks, ks))
            mu2  = cv2.blur(gray * gray, (ks, ks))
            sig  = np.sqrt(np.maximum(mu2 - mu * mu, 0))
            result = cv2.bitwise_or(result, (sig < 12).astype(np.uint8) * 255)

        # [CẢI TIẾN] Loại bỏ top 15% (sky) thay vì top 33%
        upper = np.zeros_like(result)
        upper[int(h * 0.15):, :] = 255
        result = cv2.bitwise_and(result, upper)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Loại vùng quá tối (shadow/bóng đổ sẽ loại riêng)
        not_dark = (hsv[:, :, 2] > 20).astype(np.uint8) * 255
        result   = cv2.bitwise_and(result, not_dark)

        # [CẢI TIẾN] Loại bỏ vùng có saturation cao (cây, tường màu)
        # Nước thường có saturation thấp-trung (< 180), màu sặc sỡ không phải nước
        low_sat = (hsv[:, :, 1] < 200).astype(np.uint8) * 255
        result  = cv2.bitwise_and(result, low_sat)

        lap      = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        low_edge = (lap / (lap.max() + 1e-6) < 0.08).astype(np.uint8) * 255
        result   = cv2.bitwise_and(result, low_edge)

        # [CẢI TIẾN] Loại bỏ blob quá nhỏ (< 0.3% ảnh) — tránh noise
        min_area = int(h * w * 0.003)
        cnts, _  = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(result)
        for cnt in cnts:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(filtered, [cnt], -1, 255, -1)
        return filtered

    def _detect_reflection(self, img_bgr, _mask):
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        top  = gray[:h//2, :]
        bot  = cv2.flip(gray[h//2:, :], 0)
        mh   = min(top.shape[0], bot.shape[0])
        t_r  = cv2.resize(top, (w, mh))
        b_r  = cv2.resize(bot, (w, mh))
        corr = np.corrcoef(t_r.flatten().astype(float), b_r.flatten().astype(float))[0,1]
        return bool(corr > 0.28)

    # ─── [MỚI v4] Shadow, NDWI, Fragmentation ────────────────────────────────

    def _detect_puddles(
        self, img_bgr: np.ndarray, h: int, w: int
    ) -> tuple:
        """
        Phát hiện vũng nước nhỏ (PUDDLE) trên mặt đất.

        Đặc điểm vũng nước:
          - Xuất hiện ở nửa dưới ảnh (mặt đường)
          - Bề mặt phẳng, texture rất thấp (phản chiếu gương)
          - Màu: xám / xanh nhạt (phản chiếu bầu trời) hoặc nâu bẩn (đường ngập)
          - Hình dạng: rộng ngang, không quá thẳng đứng (aspect ≥ 0.8)
          - Kích thước nhỏ hơn ngập toàn bộ nhưng > 0.2% ảnh

        Returns: (has_puddle, puddle_mask_full, puddle_area_pct)
        """
        lower_start = int(h * 0.45)
        lower_bgr   = img_bgr[lower_start:, :]
        lh, lw      = lower_bgr.shape[:2]

        gray = cv2.cvtColor(lower_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv  = cv2.cvtColor(lower_bgr, cv2.COLOR_BGR2HSV)

        # ── 1. Bề mặt phẳng: local variance thấp ──────────────────────
        mu   = cv2.blur(gray, (13, 13))
        mu2  = cv2.blur(gray * gray, (13, 13))
        sig  = np.sqrt(np.maximum(mu2 - mu * mu, 0))
        flat = (sig < 22).astype(np.uint8) * 255   # rộng hơn detector chính (12)

        # ── 2. Màu phù hợp với vũng nước ─────────────────────────────
        # Xám nhạt (trời phản chiếu), nâu nhạt (đường bẩn), tối (ban đêm)
        # Loại trừ: xanh lá cao saturation (cây, cỏ)
        water_color = cv2.inRange(
            hsv, np.array([0, 0, 12]), np.array([180, 110, 250])
        )
        not_veg     = cv2.bitwise_not(
            cv2.inRange(hsv, np.array([25, 60, 30]), np.array([90, 255, 255]))
        )
        water_color = cv2.bitwise_and(water_color, not_veg)

        # ── 3. Candidate: flat + water color ─────────────────────────
        candidate = cv2.morphologyEx(
            cv2.bitwise_and(flat, water_color),
            cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        # ── 4. Lọc contour theo kích thước + hình dạng ───────────────
        min_px = int(lh * lw * 0.002)  # ≥ 0.2% vùng dưới
        cnts, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        puddle_lower = np.zeros((lh, lw), np.uint8)
        found = False
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_px:
                continue
            _, _, bw, bh = cv2.boundingRect(cnt)
            if bh < 1:
                continue
            aspect = bw / bh
            # Vũng nước: không quá hẹp đứng, không quá phình ngang (cột, tường)
            if aspect < 0.8 or aspect > 14.0:
                continue
            # Solidity: vũng nước tương đối đặc, không rỗng
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if area / (hull_area + 1e-6) < 0.25:
                continue
            cv2.drawContours(puddle_lower, [cnt], -1, 255, -1)
            found = True

        # ── 5. Project về ảnh đầy đủ ─────────────────────────────────
        puddle_mask = np.zeros((h, w), np.uint8)
        puddle_mask[lower_start:, :] = puddle_lower

        puddle_area = float(puddle_mask.sum() / 255) / (h * w)
        has_puddle  = found and puddle_area > 0.003  # > 0.3% ảnh
        return has_puddle, puddle_mask, puddle_area

    def _vegetation_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Tạo mask thực vật (cây, cỏ, lá) để loại khỏi ứng viên nước.
        Thực vật có saturation cao + hue xanh lá — không bao giờ là nước.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        veg = cv2.inRange(hsv, np.array([25, 55, 30]), np.array([90, 255, 255]))
        k   = np.ones((5, 5), np.uint8)
        # Dilate nhẹ để loại cả viền lá
        return cv2.dilate(veg, k, iterations=1)

    def _compute_scene_context(self, img_bgr: np.ndarray, h: int, w: int) -> dict:
        """
        Phân tích ngữ cảnh cảnh quan: tỉ lệ thực vật, bầu trời, mái nhà.
        Dùng để nhận dạng ảnh không có nước (cây, nhà, cổng).
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Thực vật xanh — hue 25-90, saturation cao
        veg      = cv2.inRange(hsv, np.array([25, 55, 30]), np.array([90, 255, 255]))
        veg_pct  = float(veg.sum() / 255) / (h * w)

        # Bầu trời (top 30%): xanh blue hoặc trắng đục
        sky_h    = max(1, int(h * 0.30))
        sky_reg  = hsv[:sky_h]
        sky_blue = cv2.inRange(sky_reg, np.array([90, 25, 130]), np.array([130, 220, 255]))
        sky_wht  = cv2.inRange(sky_reg, np.array([0,  0, 185]),  np.array([180, 35, 255]))
        sky_pct  = float((sky_blue.sum() + sky_wht.sum()) / 255) / (sky_h * w)

        # Mái ngói / tường gạch đỏ — hue 0-15 và 165-180, saturation vừa
        roof_lo  = cv2.inRange(hsv, np.array([0,   55, 50]), np.array([15,  255, 220]))
        roof_hi  = cv2.inRange(hsv, np.array([165, 55, 50]), np.array([180, 255, 220]))
        roof_pct = float((roof_lo.sum() + roof_hi.sum()) / 255) / (h * w)

        return {
            "vegetation_pct": round(veg_pct, 3),
            "sky_pct":        round(sky_pct, 3),
            "roof_pct":       round(roof_pct, 3),
        }

    def _shadow_exclusion_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Tạo mask loại trừ bóng đổ — tránh nhầm shadow thành nước.

        Bóng đổ vs nước thật:
          Bóng : tối (value thấp) + saturation thấp + CÓ texture (gradient cao)
          Nước  : tối HOẶC trong + saturation thấp-vừa + SMOOTH (gradient thấp)

        Returns: mask (255 = vùng CẦN LOẠI — là shadow, không phải nước)
        """
        hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Vùng tối + desaturated → candidate shadow
        dark_desat = (
            (hsv[:, :, 2] < 85) &
            (hsv[:, :, 1] < 55)
        ).astype(np.uint8)

        # Gradient cao trong candidate shadow → shadow chứ không phải nước
        gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        # Normalize
        grad_norm = grad / (grad.max() + 1e-6)
        high_grad = (grad_norm > 0.12).astype(np.uint8)

        shadow = (dark_desat & high_grad).astype(np.uint8) * 255

        # Dilate nhẹ để loại vùng biên shadow
        k = np.ones((5, 5), np.uint8)
        return cv2.dilate(shadow, k, iterations=1)

    def _ndwi_mask(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        NDWI-like index — phát hiện nước bằng tỉ lệ kênh màu.

        Clear water  : NDWI_clear = (G−R)/(G+R) > 0.04  (hấp thụ đỏ, phản chiếu xanh)
        Muddy water  : NDWI_muddy = (B−R)/(B+R) > 0.02  (nước bùn có blue > red)
        Flood turbid : ratio_gb = G/(B+1) ∈ [0.6, 1.4] AND value không quá sáng

        Chỉ áp dụng từ 20% ảnh trở xuống để tránh nhầm bầu trời.
        """
        h    = img_rgb.shape[0]
        img  = img_rgb.astype(np.float32)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        ndwi_clear = (g - r) / (g + r + 1e-6)
        ndwi_muddy = (b - r) / (b + r + 1e-6)

        # Turbid flood: nước bùn nâu-vàng — range hẹp hơn để tránh match nhà/cây
        # ndwi_clear nhỏ âm (R hơi > G) + ndwi_muddy hơi âm (R hơi > B) = màu bùn
        bright = (r + g + b) / 3.0
        turbid_flood = (
            (ndwi_clear > -0.22) & (ndwi_clear < 0.12) &
            (ndwi_muddy > -0.28) & (ndwi_muddy < 0.06) &
            (bright > 35) & (bright < 195)
        ).astype(np.uint8) * 255

        water_clear = (ndwi_clear > 0.04).astype(np.uint8) * 255
        water_muddy = (ndwi_muddy > 0.02).astype(np.uint8) * 255

        combined = cv2.bitwise_or(cv2.bitwise_or(water_clear, water_muddy), turbid_flood)

        # Giới hạn vùng áp dụng: từ 20% ảnh trở xuống
        region = np.zeros(img_rgb.shape[:2], np.uint8)
        region[int(h * 0.20):, :] = 255
        combined = cv2.bitwise_and(combined, region)

        k = np.ones((9, 9), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)

        # Loại blob quá nhỏ — giảm ngưỡng để bắt vũng nước nhỏ
        min_area = int(img_rgb.shape[0] * img_rgb.shape[1] * 0.002)
        cnts, _  = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result   = np.zeros_like(combined)
        for cnt in cnts:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(result, [cnt], -1, 255, -1)
        return result

    def _compute_fragmentation(self, mask: np.ndarray, h: int, w: int) -> float:
        """
        Đo mức độ phân mảnh của water mask.
        Nước thật thường tạo thành 1-3 vùng liên tục lớn.
        Nhiều blob nhỏ phân tán → nhiều khả năng là false positive.

        [CẢI TIẾN]: Giảm penalty khi các blob tập trung ở nửa dưới ảnh
        (vũng nước nhiều cái trên mặt đường ≠ false positive).

        Returns: fragmentation score [0, 1]
          0 = hoàn toàn liên tục (1 blob lớn)
          1 = rất phân mảnh (nhiều blob nhỏ)
        """
        total_pixels = float(mask.sum() / 255)
        if total_pixels < 10:
            return 1.0

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 1.0

        areas    = sorted([cv2.contourArea(c) for c in cnts], reverse=True)
        n_blobs  = len(areas)
        if n_blobs == 0:
            return 1.0

        # Tỉ lệ blob lớn nhất so với tổng
        largest_ratio = areas[0] / (total_pixels + 1e-6)

        # Blob nhỏ hơn 1% tổng diện tích ảnh
        tiny_count = sum(1 for a in areas if a < (h * w * 0.01))
        tiny_ratio = tiny_count / max(n_blobs, 1)

        frag = (1.0 - largest_ratio) * 0.6 + tiny_ratio * 0.4

        # Kiểm tra xem các blob có tập trung ở nửa dưới không (vũng nước)
        lower_y  = h // 2
        lower_px = float(mask[lower_y:, :].sum() / 255)
        lower_ratio = lower_px / (total_pixels + 1e-6)
        if lower_ratio > 0.75:
            # Phần lớn water ở nửa dưới → có thể là nhiều vũng trên mặt đường
            frag *= 0.55

        return float(np.clip(frag, 0.0, 1.0))

    def _find_water_line(self, water_mask, h, w):
        if water_mask.sum() == 0:
            return h, 0.0
        col_top_y = []
        for col in range(0, w, max(1, w//128)):
            rows = np.nonzero(water_mask[:,col]>0)[0]
            if len(rows): col_top_y.append(rows[0])
        row_count = np.count_nonzero(water_mask>0, axis=1)
        row_top   = h
        if row_count.max() > 0:
            sm  = cv2.GaussianBlur(row_count.astype(np.float32),(1,9),0)
            cands = np.where(sm >= sm.max()*0.05)[0]
            if len(cands): row_top = int(cands[0])
        col_top = h
        if col_top_y:
            arr   = np.array(col_top_y)
            med   = np.median(arr)
            valid = arr[(arr>med*0.25)&(arr<min(h-1,med*1.8+20))]
            col_top = int(np.percentile(valid if len(valid) else arr, 30))
        wl  = max(int(h*0.05), min(min(col_top, row_top), h-5))
        std = np.std(col_top_y)/max(h,1) if col_top_y else 0.5
        conf = float(np.clip(1.0 - std*2.5, 0.25, 0.96))
        if wl < h*0.15: wl = int(h*0.18); conf *= 0.6
        if abs(row_top-col_top) > h*0.12: conf *= 0.75
        return wl, float(np.clip(conf, 0, 1))

    def _calc_turbidity(self, dom, hsv_scores):
        t = dom.get("type","none")
        if t == "none" or t not in HSV_WATER_PROFILES: return 0.0
        base   = HSV_WATER_PROFILES[t].get("turbidity", 0.5)
        muddy  = sum(v for k,v in hsv_scores.items() if k in ("muddy_brown","muddy_orange","reddish_clay","deep_brown","yellowish_flood"))
        clear  = sum(v for k,v in hsv_scores.items() if k in ("clear_blue","teal_shallow"))
        total  = muddy+clear+1e-6
        return round(float(np.clip(0.5*base + 0.5*(muddy/total)*0.9 + 0.5*(clear/total)*0.1, 0, 1)), 3)

    def _calc_confidence(self, water_area_pct, wl_conf, has_reflection,
                         water_type, ch, frag_score: float = 0.5):
        """
        Tính confidence.
        [CẢI TIẾN v4]:
          - Thêm fragmentation penalty: mask phân mảnh → confidence giảm
          - Thêm NDWI agreement bonus: NDWI đồng ý → rất chắc là nước
          - Điều chỉnh trọng số agreeing channels
        """
        c = min(0.35, water_area_pct * 3.5)
        if water_area_pct > 0.4: c += 0.10
        c += wl_conf * 0.25
        if has_reflection: c += 0.15

        # [CẢI TIẾN] Đếm channels đồng ý với ngưỡng cao hơn (0.03 thay vì 0.02)
        agreeing = sum(1 for k, v in ch.items()
                       if k not in ("fragmentation", "reflection") and v > 0.03)
        c += min(0.15, agreeing * 0.025)

        # [MỚI] NDWI agreement bonus
        if ch.get("ndwi", 0) > 0.03:
            c += 0.08

        if water_type not in ("none", ""): c += 0.05

        if ch.get("hsv", 0) < 0.01 and ch.get("texture", 0) > 0.05:
            c -= 0.10

        # [MỚI] Fragmentation penalty: mask phân mảnh → likely false positive
        # frag_score=0 (consolidated) → no penalty; frag_score=1 (fragmented) → -0.20
        c -= frag_score * 0.20

        return float(np.clip(c, 0.0, 1.0))
