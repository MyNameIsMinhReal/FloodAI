# -*- coding: utf-8 -*-
"""
depth_analysis/perspective_analyzer.py
----------------------------------------
Phân tích goc chup va chinh sua sai so phoi canh.

Van de:
  - Anh chup tren cao (flycam): vat the nhin tu tren xuong,
    chieu cao bbox KHONG phan anh chieu cao thuc, chi phan anh
    chieu rong. Can dung ty le chieu rong thay vi chieu cao.
  - Anh chup nghieng: can chinh horizon line

Giai phap:
  1. Phan loai anh: eye-level / elevated / aerial
  2. Chon phuong phap do phu hop voi goc chup
  3. Perspective factor: he so hieu chinh
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PerspectiveResult:
    view_angle:         str      # "eye_level", "elevated", "aerial"
    tilt_deg:           float    # do nghieng cua camera (0=ngang, 90=thang dung)
    horizon_y:          int      # vi tri duong chan troi (pixel y)
    horizon_confidence: float
    scale_factor:       float    # he so scale theo chieu sau
    measure_method:     str      # "height" hoac "width" hoac "shadow"
    # [IMPROVE] Thêm fields tu object-based + low-angle detection
    is_aerial:          bool     = False    # object-size based confirmation
    angle_factor:       float    = 1.0      # aerial correction factor
    is_low_angle:       bool     = False    # near water surface
    fov_deg:            float    = 65.0     # estimated FOV (degrees)
    camera_height_m:    float    = 1.5      # estimated camera height


class PerspectiveAnalyzer:
    """Phân tích goc chup de hieu chinh phuong phap do muc nuoc."""

    def __init__(self):
        """Empty init - no state needed."""
        pass

    def analyze(self, img_rgb: np.ndarray) -> PerspectiveResult:
        h, w = img_rgb.shape[:2]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # === 1. Tim duong chan troi (horizon line) ===
        horizon_y, horizon_conf = self._find_horizon(gray, h, w)

        # === 2. Phan loai goc chup ===
        view_angle, tilt_deg = self._classify_view(
            img_bgr, gray, horizon_y, horizon_conf, h, w
        )

        # === 3. Tinh scale factor theo khoang cach ===
        scale_factor = self._calc_scale_factor(view_angle, horizon_y, h)

        # === 4. Chon phuong phap do ===
        measure_method = self._choose_measure_method(view_angle, tilt_deg)

        return PerspectiveResult(
            view_angle         = view_angle,
            tilt_deg           = tilt_deg,
            horizon_y          = horizon_y,
            horizon_confidence = horizon_conf,
            scale_factor       = scale_factor,
            measure_method     = measure_method,
            is_aerial          = view_angle == "aerial",
            angle_factor       = 1.0,
            is_low_angle       = False,
            fov_deg            = self._estimate_fov(img_rgb, w, h),
            camera_height_m    = self._estimate_camera_height(view_angle, tilt_deg),
        )

    # ------------------------------------------------------------------
    # [IMPROVE] Unified aerial detection — horizon + object-size + low-angle
    # Cu: reference_estimator.py co `_detect_camera_angle()` (object-size)
    #     rieng, PerspectiveAnalyzer co horizon-based rieng → mâu thuẫn.
    # Moi: hop nhat vao PerspectiveAnalyzer, dung ca 2 signal.
    # ------------------------------------------------------------------
    def detect_camera_angle(self, img_rgb: np.ndarray, dets: list,
                             h: int, w: int) -> dict:
        """
        Phat hien goc camera UNIFIED — ket hop:
          1. Horizon-based (tu analyze())
          2. Object-size-based (tu _detect_camera_angle cu)
          3. Low-angle detection

        Returns: { is_aerial, angle_factor, is_low_angle, confidence }
        """
        # Signal 1: horizon-based (tu analyze())
        perspective = self.analyze(img_rgb)

        # Signal 2: object-size-based
        aerial_score_obj = 0.0
        if dets:
            obj_heights = [d["bbox"][3] - d["bbox"][1] for d in dets]
            avg_h_ratio = float(np.mean(obj_heights)) / h
            if avg_h_ratio < 0.18:
                aerial_score_obj = 0.8
            elif avg_h_ratio < 0.25:
                aerial_score_obj = 0.5
            else:
                aerial_score_obj = 0.1

            centers_y = [(d["bbox"][1] + d["bbox"][3]) / 2 / h for d in dets]
            avg_cy = float(np.mean(centers_y))
            if 0.35 < avg_cy < 0.65 and aerial_score_obj > 0.4:
                aerial_score_obj = min(1.0, aerial_score_obj + 0.2)

            # Signal 3: line angle analysis
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                                    minLineLength=w//6, maxLineGap=20)
            if lines is not None:
                angles = []
                for l in lines[:30]:
                    x1, y1, x2, y2 = l[0]
                    if abs(x2-x1) > 10:
                        angles.append(abs(np.degrees(np.arctan2(y2-y1, x2-x1))))
                if angles:
                    near_horizontal = sum(1 for a in angles if a < 20 or a > 160)
                    h_ratio = near_horizontal / len(angles)
                    if h_ratio > 0.6:
                        aerial_score_obj = min(1.0, aerial_score_obj + 0.15)

        # Signal 4: low-angle detection
        low_angle_score = self._detect_low_angle_score(img_rgb, h, w)

        # --- MERGE scores ---
        # Horizon-based: view_angle → aerial_score
        horizon_aerial = 1.0 if perspective.view_angle == "aerial" else (
            0.5 if perspective.view_angle == "elevated" else 0.0)

        # Final aerial = max(horizon, object-size) — ca 2 deu quan trong
        final_aerial_score = max(horizon_aerial, aerial_score_obj)
        is_aerial = final_aerial_score > 0.55

        if is_aerial:
            angle_factor = max(0.2, 1.0 - final_aerial_score * 0.8)
        else:
            angle_factor = 1.0

        is_low_angle = low_angle_score > 0.55

        log.debug(f"  CameraUnified: aerial={is_aerial} score_h={horizon_aerial:.2f} "
                  f"score_obj={aerial_score_obj:.2f} low_angle={is_low_angle}")

        return {
            "is_aerial":    is_aerial,
            "angle_factor": round(angle_factor, 2),
            "is_low_angle": is_low_angle,
            "confidence":   round(min(1.0, final_aerial_score), 2),
            "view_angle":   perspective.view_angle,
            "tilt_deg":     perspective.tilt_deg,
            "fov_deg":      perspective.fov_deg,
            "camera_height_m": perspective.camera_height_m,
        }

    def _detect_low_angle_score(self, img_rgb: np.ndarray,
                                 h: int, w: int) -> float:
        """
        Phat hien anh chup goc thap (lens gan mat nuoc).
        Tra ve score 0-1 (0.55 = nguong is_low_angle).
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dai_h = h // 10
        row_vars = []
        for i in range(10):
            dai = gray[i*dai_h:(i+1)*dai_h, :]
            row_vars.append(float(dai.var()))

        bottom3 = np.mean(row_vars[7:])
        top4    = np.mean(row_vars[:4])
        mid3    = np.mean(row_vars[3:7])

        score = 0.0
        if bottom3 < mid3 * 0.4:
            score += 0.4
        if bottom3 < top4 * 0.3:
            score += 0.3

        grad_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        horizon_zone = grad_y[int(h*0.30):int(h*0.60), :]
        horizon_score_val = float(horizon_zone.mean())
        full_score_val    = float(grad_y.mean())
        if horizon_score_val > full_score_val * 1.5:
            score += 0.3

        return score

    # ------------------------------------------------------------------
    # [IMPROVE] FOV estimation — estimat FOV tu image features
    # ------------------------------------------------------------------
    def _estimate_fov(self, img_rgb: np.ndarray, w: int, h: int) -> float:
        """
        Uoc tinh FOV (field of view) tu image:
          1. EXIF focal length (neu co)
          2. Heuristic tu perspective features

        Tra ve FOV trong degrees (typical: 55-75 cho phone, 90-120 cho wide-angle).
        """
        # Try EXIF first
        fov = self._fov_from_exif(img_rgb)
        if fov is not None:
            return fov

        # Heuristic: camera height + horizon position → FOV estimate
        # Wide-angle (FOV>90): horizon very low, objects distorted at edges
        # Normal (FOV 55-75): standard phone camera
        # Tele (FOV<40): horizon high, objects compressed
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                                minLineLength=w//6, maxLineGap=20)

        distortion_score = 0.0
        if lines is not None:
            # Wide-angle: nhieu duong cong/cheo o bien anh
            edge_lines = 0
            center_lines = 0
            for l in lines[:40]:
                x1, y1, x2, y2 = l[0]
                cx = (x1 + x2) / 2
                if abs(cx - w/2) > w * 0.3:
                    edge_lines += 1
                else:
                    center_lines += 1
            if edge_lines > center_lines * 0.5:
                distortion_score = 0.3

        # Default: 65 deg (typical phone)
        return 65.0 + distortion_score * 30.0

    def _fov_from_exif(self, img_rgb: np.ndarray) -> Optional[float]:
        """
        Extract FOV tu EXIF focal length + sensor width.
        Dung PIL.ExifTags de doc EXIF, tinh FOV = 2*atan(sensor_width/(2*focal_length)).
        """
        try:
            from PIL import Image
            import io
            buf = io.BytesIO()
            Image.fromarray(img_rgb).save(buf, format='JPEG')
            buf.seek(0)
            img = Image.open(buf)
            exif_data = img.getexif()
            if exif_data is None:
                return None

            # Focal length in 35mm equivalent
            focal_35mm = exif_data.get(37386)  # FocalLengthIn35mmFilm
            if focal_35mm and focal_35mm > 0:
                # FOV = 2 * atan(36 / (2 * focal_35mm)) * 180/pi
                # 36mm = sensor width for 35mm film
                fov_rad = 2 * np.arctan(36.0 / (2.0 * focal_35mm))
                return float(fov_rad * 180.0 / np.pi)

            # Actual focal length
            focal_rational = exif_data.get(37386)  # FocalLength
            if focal_rational:
                focal_mm = float(focal_rational)
                # Assume typical phone sensor width ~6.2mm
                sensor_width_mm = 6.2
                fov_rad = 2 * np.arctan(sensor_width_mm / (2.0 * focal_mm))
                return float(fov_rad * 180.0 / np.pi)

        except Exception:
            pass
        return None

    def _estimate_camera_height(self, view_angle: str, tilt_deg: float) -> float:
        """Uoc tinh chieu cao camera tu view angle + tilt."""
        if view_angle == "aerial":
            return 50.0  # drone typical
        elif view_angle == "elevated":
            return 3.0 + tilt_deg * 0.05  # 2nd floor ~3-7m
        else:
            return 1.5  # eye level

    # ------------------------------------------------------------------
    def _find_horizon(self, gray: np.ndarray, h: int, w: int) -> Tuple[int, float]:
        """
        Tim duong chan troi bang Hough line detection.
        Duong chan troi = duong ngang dai nhat o giua anh.

        Neu khong co duong ro, dung mau troi + phan huy (sky / land) de dua ra horizon uoc luong.
        """
        edges = cv2.Canny(gray, 40, 130, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=max(30, w // 8),
            minLineLength=max(30, w // 4),
            maxLineGap=25,
        )

        if lines is None:
            # fallback ket hop voi brightness gradient: tim dong nguong cao nhat
            row_sum = gray.mean(axis=1)
            grad    = np.abs(np.gradient(row_sum))
            if grad.size == 0:
                return h // 3, 0.2
            candidate_y = int(np.argmax(grad))
            candidate_y = max(int(h * 0.1), min(int(h * 0.7), candidate_y))
            return candidate_y, 0.25

        horizontal = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 12 or angle > 168:
                length  = np.hypot(x2-x1, y2-y1)
                mid_y   = (y1 + y2) / 2.0
                horizontal.append((mid_y, length))

        if not horizontal:
            # fallback: same as before
            row_sum = gray.mean(axis=1)
            grad    = np.abs(np.gradient(row_sum))
            candidate_y = int(np.argmax(grad))
            candidate_y = max(int(h * 0.1), min(int(h * 0.7), candidate_y))
            return candidate_y, 0.25

        horizontal.sort(key=lambda x: -x[1])
        top3 = horizontal[:3]
        best_y = int(np.mean([y for y, _ in top3]))
        conf   = min(0.95, sum([l for _, l in top3]) / (3.0 * w))

        best_y = max(int(h * 0.08), min(int(h * 0.75), best_y))
        return best_y, conf

    # ------------------------------------------------------------------
    def _classify_view(
        self,
        img_bgr: np.ndarray,
        gray:    np.ndarray,
        horizon_y: int,
        horizon_conf: float,
        h: int, w: int,
    ) -> Tuple[str, float]:
        """
        Phan loai goc chup:
          - eye_level: chup ngang tam mat (horizon ~ giua anh)
          - elevated:  chup tu tren xuong mot chut (horizon o cao)
          - aerial:    chup tren cao / flycam (khong co horizon ro)
        """
        horizon_ratio = horizon_y / h

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sky_mask = cv2.inRange(hsv, np.array([95, 10, 120]), np.array([135, 120, 255]))
        sky_pct  = sky_mask.sum() / 255 / (h * w)

        ground_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        ground_pct  = ground_mask.sum() / 255 / (h * w)

        lines = cv2.HoughLinesP(
            cv2.Canny(gray, 40, 150, apertureSize=3),
            1, np.pi/180, threshold=max(20, w // 10), minLineLength=max(30, w // 5), maxLineGap=20
        )
        horizon_edges = 0
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                if ang < 15 or ang > 165:
                    horizon_edges += 1

        aerial_score = 0.0
        if sky_pct < 0.05 and ground_pct > 0.60:
            aerial_score += 0.35
        if horizon_ratio < 0.25:
            aerial_score += 0.25
        if horizon_conf < 0.35:
            aerial_score += 0.20
        if horizon_edges < 5:
            aerial_score += 0.20

        # Elevated: nho horizon_ratio < 0.45 + co sky % khaon 15-40
        elevated_score = 0.0
        if 0.15 < sky_pct < 0.6 and 0.20 < horizon_ratio < 0.45:
            elevated_score += 0.45
        if 0.35 <= horizon_ratio < 0.55:
            elevated_score += 0.30

        if aerial_score >= 0.6:
            tilt = min(90.0, 70.0 + aerial_score * 20)
            return "aerial", tilt
        elif elevated_score >= 0.4 or horizon_ratio < 0.35:
            tilt = min(80.0, 20.0 + (0.35 - horizon_ratio) * 90)
            return "elevated", tilt
        else:
            tilt = max(0.0, min(45.0, (0.5 - horizon_ratio) * 80))
            return "eye_level", tilt

    # ------------------------------------------------------------------
    def _calc_scale_factor(
        self, view_angle: str, horizon_y: int, h: int
    ) -> float:
        """
        Tinh scale factor de hieu chinh khoang cach.

        Trong phoi canh: vat the o xa (gan duong chan troi) nho hon vat the
        o gan (cuoi anh). Scale factor giup chuan hoa.

        Tra ve he so: vat the o giua anh = 1.0 (baseline)
        """
        if view_angle == "aerial":
            return 1.0

        if horizon_y <= 0 or horizon_y >= h:
            return 1.0

        # distance from image center to horizon
        mid_y = h / 2.0
        d_hor = abs(mid_y - horizon_y) / h

        # scale co ban: horizon gan duoi -> do goc cao -> dung height chinh
        # horizon gan tren -> phai dung width ( => scale >1 ).
        if d_hor < 0.10:
            scale = 1.0
        else:
            scale = 1.0 + min(1.2, d_hor * 2.5)

        # neu elevated / aerial => tang them
        if view_angle == "elevated":
            scale = max(1.0, scale * 1.1)
        elif view_angle == "aerial":
            scale = max(1.0, scale * 1.2)

        return float(np.clip(scale, 0.75, 2.5))

    # ------------------------------------------------------------------
    def _choose_measure_method(
        self, view_angle: str, tilt_deg: float
    ) -> str:
        """
        Chon phuong phap do muc nuoc phu hop voi goc chup.

        - eye_level: dung chieu cao (height) cua vat the
        - elevated:  dung chieu cao nhung co hieu chinh goc
        - aerial:    dung chieu rong (width) hoac bong do (shadow length)
        """
        if view_angle == "aerial":
            return "width"
        elif view_angle == "elevated" and tilt_deg > 45:
            return "width"
        else:
            return "height"


    # ------------------------------------------------------------------
    # 3.1 VANISHING POINT DETECTION (MỚI)
    # ------------------------------------------------------------------

    def detect_vanishing_point(self, img_rgb: np.ndarray) -> dict:
        """
        Phát hiện vanishing point (điểm tụ) và normalize scale theo ground plane.

        Tầm quan trọng:
          - Khi camera nghiêng: depth sẽ sai nếu không có vanishing point
          - Vanishing point → xác định hướng chiều sâu thực sự
          - Scale correction: normalize object size theo khoảng cách

        Returns:
            dict với vp_x, vp_y, vp_confidence, scale_at_y, ground_plane_y, ...
        """
        h, w = img_rgb.shape[:2]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # --- Bước 1: Detect edges + lines ---
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=max(30, h // 8),
            minLineLength=max(30, w // 10),
            maxLineGap=20,
        )

        if lines is None or len(lines) < 5:
            return self._vanishing_point_fallback(h, w)

        # --- Bước 2: Tìm vanishing point bằng RANSAC-style voting ---
        # Chỉ lấy các đường có hướng gần với "perspective lines" (không nằm ngang hoàn toàn)
        persp_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1 + 1e-6)) % 180
            # Bỏ đường nằm ngang (85-95°) và đường dọc (0-5°, 175-180°) gần hoàn toàn
            if 10 < angle < 80 or 100 < angle < 170:
                persp_lines.append((x1, y1, x2, y2))

        if len(persp_lines) < 4:
            return self._vanishing_point_fallback(h, w)

        # Tìm điểm giao nhau của các cặp đường (tất cả combinations)
        intersections = []
        for i in range(len(persp_lines)):
            for j in range(i + 1, min(i + 8, len(persp_lines))):
                pt = self._line_intersection(persp_lines[i], persp_lines[j])
                if pt is not None:
                    ix, iy = pt
                    # Chỉ lấy intersections trong vùng ảnh mở rộng
                    if -w < ix < 2 * w and -h // 2 < iy < h:
                        intersections.append((ix, iy))

        if not intersections:
            return self._vanishing_point_fallback(h, w)

        # Clustering intersections (median để robust)
        int_arr = np.array(intersections)
        # Loại outliers bằng IQR
        q1_x, q3_x = np.percentile(int_arr[:, 0], [25, 75])
        q1_y, q3_y = np.percentile(int_arr[:, 1], [25, 75])
        iqr_x = max(q3_x - q1_x, 50)
        iqr_y = max(q3_y - q1_y, 50)
        mask = (
            (int_arr[:, 0] >= q1_x - 1.5 * iqr_x) &
            (int_arr[:, 0] <= q3_x + 1.5 * iqr_x) &
            (int_arr[:, 1] >= q1_y - 1.5 * iqr_y) &
            (int_arr[:, 1] <= q3_y + 1.5 * iqr_y)
        )
        filtered = int_arr[mask]
        if len(filtered) == 0:
            filtered = int_arr

        vp_x = float(np.median(filtered[:, 0]))
        vp_y = float(np.median(filtered[:, 1]))

        # --- Bước 3: Confidence dựa trên spread của intersections ---
        spread_x = float(np.std(filtered[:, 0])) / (w + 1e-6)
        spread_y = float(np.std(filtered[:, 1])) / (h + 1e-6)
        vp_confidence = float(np.clip(1.0 - (spread_x + spread_y) * 2, 0.1, 0.95))

        # --- Bước 4: Scale normalization theo ground plane ---
        # Giả định: vanishing point ở horizon → scale tăng dần từ VP xuống dưới
        # scale_at_y(y) = 1 + k * (y - vp_y) / (h - vp_y)
        # k = scaling constant (thường 0.5-2.0)
        if vp_y < h and (h - vp_y) > 10:
            scale_rate = 1.5 / max((h - vp_y) / h, 0.1)
        else:
            scale_rate = 1.0

        # Ground plane y (nơi camera tiếp xúc với mặt đường)
        ground_plane_y = int(np.clip(vp_y + (h - vp_y) * 0.9, vp_y, h - 1))

        log.info(
            f"  VanishingPoint: ({vp_x:.0f}, {vp_y:.0f}) "
            f"conf={vp_confidence:.2f} scale_rate={scale_rate:.2f}"
        )

        return {
            "vp_x":              vp_x,
            "vp_y":              vp_y,
            "vp_confidence":     vp_confidence,
            "scale_rate":        scale_rate,
            "ground_plane_y":    ground_plane_y,
            "n_intersections":   len(filtered),
            "n_lines_used":      len(persp_lines),
        }

    def scale_at_y(self, y: float, vp_result: dict, h: int) -> float:
        """
        Tính scale factor tại row y, dựa trên vanishing point.

        Dùng để normalize chiều cao object theo khoảng cách:
          object gần (y lớn) → scale lớn → chia cho scale để chuẩn hóa

        Args:
            y:          row pixel trong ảnh
            vp_result:  kết quả từ detect_vanishing_point()
            h:          chiều cao ảnh

        Returns:
            scale_factor [0.1, 5.0]
        """
        vp_y       = vp_result.get("vp_y", h // 3)
        scale_rate = vp_result.get("scale_rate", 1.0)

        if vp_y >= h or vp_y >= y:
            return 1.0

        # Scale tăng tuyến tính từ horizon (VP) xuống dưới
        rel_pos = (y - vp_y) / max(h - vp_y, 1)
        scale   = 1.0 + scale_rate * rel_pos
        return float(np.clip(scale, 0.1, 5.0))

    def _line_intersection(self, l1, l2) -> Optional[Tuple[float, float]]:
        """Tính giao điểm của 2 đoạn thẳng (hoặc đường thẳng mở rộng)."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None  # song song

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    def _vanishing_point_fallback(self, h: int, w: int) -> dict:
        """Fallback khi không detect được vanishing point."""
        return {
            "vp_x":           w / 2.0,
            "vp_y":           h / 3.0,
            "vp_confidence":  0.2,
            "scale_rate":     1.0,
            "ground_plane_y": int(h * 0.8),
            "n_intersections": 0,
            "n_lines_used":    0,
        }


def apply_perspective_correction(
    water_cm:    float,
    detected_obj: dict,
    perspective: PerspectiveResult,
    img_h:       int,  # unused
) -> Tuple[float, float]:
    """
    Hieu chinh ket qua do muc nuoc theo goc chup.

    Tra ve (corrected_water_cm, correction_factor).
    """
    if perspective.view_angle == "eye_level":
        return water_cm, 1.0

    x1, y1, x2, y2 = detected_obj["bbox"]
    _  = y2 - y1  # unused bbox_h
    bbox_w  = x2 - x1
    _  = (y1 + y2) / 2  # unused obj_cy

    if perspective.view_angle == "aerial":
        # Aerial: bbox_h khong phan anh chieu cao thuc
        # Dung bbox_w va ty le chieu rong / chieu cao thuc cua xe/nguoi
        ref_h  = detected_obj["ref_height_cm"]
        cls    = detected_obj.get("class_name", "")
        # Ty le aspect ratio thuc te
        aspect_ratios = {
            "person":     0.30,   # nguoi: rong/cao ~ 0.3
            "car":        1.24,   # xe con: rong/cao ~ 1.24
            "motorcycle": 0.59,
            "truck":      0.86,
        }
        ar = aspect_ratios.get(cls, 0.5)
        if ar > 0 and bbox_w > 0:
            est_h_px = bbox_w / ar
            # Tinh lai ty le ngap
            submersion_px = max(0, y2 - detected_obj.get("water_line_y_used", y2))
            if est_h_px > 0:
                corrected_cm = ref_h * (submersion_px / est_h_px)
                factor = corrected_cm / max(water_cm, 0.1)
                return corrected_cm, factor

    elif perspective.view_angle == "elevated":
        # Elevated: hieu chinh theo goc nghieng
        tilt_rad = np.radians(perspective.tilt_deg)
        factor   = 1.0 / max(np.cos(tilt_rad), 0.3)   # tranh chia 0
        corrected_cm = water_cm * min(factor, 2.5)     # gioi han max x2.5
        return corrected_cm, factor

    return water_cm, 1.0
