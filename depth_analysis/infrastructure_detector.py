# -*- coding: utf-8 -*-
"""
depth_analysis/infrastructure_detector.py
-------------------------------------------
Phat hien cac vat the co dinh tren duong pho:
  - Cot dien / vien thong (khong co trong YOLO COCO)
  - Cot den duong
  - Than cay (de do vet muc nuoc - "tideline")
  - Bien bao duong (ho tro YOLO stop_sign/traffic_light)
  - Nguoi mac ao phao (tren thuyen cuu ho - KHONG tinh la ngap)

Phuong phap:
  1. YOLO detect traffic_light, stop_sign (co san trong COCO)
  2. Custom color+shape detector cho cot dien, than cay
  3. Life jacket detector: phat hien mau cam/vang dac trung
  4. Tideline detector: tim vet nuoc tren tuong/cay

Dac biet: "Tideline" (vet muc nuoc) tren tuong/cot la indicator chinh xac nhat!
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class InfraDetection:
    class_name:    str
    bbox:          List[int]       # [x1, y1, x2, y2]
    confidence:    float
    ref_height_cm: float
    color_bgr:     Tuple
    is_on_boat:    bool = False    # True neu dang tren thuyen
    tideline_y:    Optional[int] = None   # y-coord cua vet muc nuoc
    notes:         str = ""


class InfrastructureDetector:
    """Phat hien co so ha tang duong pho de do muc nuoc chinh xac hon."""

    def __init__(self, conf_thresh: float = 0.30):
        self.conf_thresh = conf_thresh

    # ------------------------------------------------------------------
    def detect(
        self,
        img_rgb:    np.ndarray,
        yolo_model = None,    # YOLO model da load san
    ) -> List[InfraDetection]:
        """
        Phat hien tat ca vat the co so ha tang.
        Ket hop YOLO + custom color/shape detectors.
        """
        h, w     = img_rgb.shape[:2]
        img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        results  = []

        # --- 1. YOLO: traffic_light, stop_sign ---
        if yolo_model:
            yolo_results = self._detect_yolo_infra(img_rgb, yolo_model, h, w)
            results.extend(yolo_results)

        # --- 2. Custom: cot dien (gray vertical line) ---
        poles = self._detect_utility_poles(img_bgr, h, w)
        results.extend(poles)

        # --- 3. Life jacket detection (ao phao cuu ho) ---
        jackets = self._detect_life_jackets(img_bgr, h, w)
        results.extend(jackets)

        # --- 3b. Bien bao duong - phu hop VN (tam giac/tron/cam) ---
        signs = self._detect_street_signs(img_bgr, h, w)
        results.extend(signs)

        # --- 4. Tideline detection ---
        tidemarks = self._detect_tideline(img_bgr, h, w)
        results.extend(tidemarks)

        log.debug(
            f"  Infra: {len(results)} detected "
            f"({sum(1 for r in results if r.is_on_boat)} on boat)"
        )
        return results

    # ------------------------------------------------------------------
    def detect_doors(
        self, img_rgb: np.ndarray, h: int, w: int
    ) -> List[InfraDetection]:
        """
        Phat hien cua nha (door) de dung lam reference cao nhat.

        Cua nha Viet Nam: chieu cao chuan ~200cm, ty le rong/cao ~0.45.
        Phuong phap: edge detection + contour filter theo aspect ratio.

        Chi tim cua trong vung TREN cua anh (nuoc o duoi, cua phai nhin thay phan tren).
        """
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        results: List[InfraDetection] = []

        # Chi tim trong 80% tren cua anh (cua phai co it nhat phan tren trong)
        search_h = int(h * 0.80)
        gray_roi = gray[:search_h, :]

        # Edge detection + morphological closing de lam lien mach
        edges = cv2.Canny(gray_roi, 30, 100)
        kernel = np.ones((5, 3), np.uint8)
        edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        MIN_DOOR_H  = int(h * 0.15)   # cua toi thieu 15% chieu cao anh
        MAX_DOOR_H  = int(h * 0.80)   # cua toi da 80% chieu cao anh
        MIN_DOOR_W  = int(w * 0.04)   # chieu rong toi thieu 4% frame

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch < MIN_DOOR_H or ch > MAX_DOOR_H or cw < MIN_DOOR_W:
                continue

            # Aspect ratio: cua rong/cao ~ 0.35-0.65
            ar = cw / ch
            if not (0.32 < ar < 0.68):
                continue

            # Contour phai "day du" — tranh cac phat hien ngau nhien
            rect_area = cw * ch
            cnt_area  = cv2.contourArea(cnt)
            if rect_area == 0 or cnt_area / rect_area < 0.25:
                continue

            # Loai vung qua nho (co the la cua so, khong phai cua nha)
            if rect_area < (h * w * 0.005):
                continue

            # Gradient trong bbox phai cao (cua co canh / khung ro)
            roi_gray = gray_roi[y:y+ch, x:x+cw]
            if roi_gray.size == 0:
                continue
            grad_x  = cv2.Sobel(roi_gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y  = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)
            mean_grad = float(np.sqrt(grad_x**2 + grad_y**2).mean())
            if mean_grad < 8.0:   # qua mo -> khong phai canh cua
                continue

            # Confidence dua tren muc do phu hop aspect ratio va kich thuoc
            ar_score  = 1.0 - abs(ar - 0.45) / 0.20   # 0.45 la ty le ly tuong
            size_score = min(1.0, ch / (h * 0.40))     # cang cao cang tot
            confidence = round(min(0.80, 0.40 + ar_score * 0.25 + size_score * 0.15), 2)

            results.append(InfraDetection(
                class_name    = "cua_nha",
                bbox          = [x, y, x + cw, y + ch],
                confidence    = confidence,
                ref_height_cm = 200,
                color_bgr     = (30, 100, 255),
                notes         = f"Door ar={ar:.2f} grad={mean_grad:.1f}",
            ))

        # Giu lai toi da 2 cua co confidence cao nhat (tranh false positives)
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:2]

    # ------------------------------------------------------------------
    def _detect_yolo_infra(
        self, img_rgb: np.ndarray, yolo_model, h: int, w: int
    ) -> List[InfraDetection]:
        """Dung YOLO detect cac class co san: traffic light, stop sign."""
        target = {
            "traffic light", "stop sign", "fire hydrant",
            "motorbike", "bicycle", "bus", "truck", "car"
        }
        info   = {
            "traffic light": (350, (0, 255, 255)),
            "stop sign":     (220, (0, 0, 220)),
            "fire hydrant":  (70,  (0, 60, 200)),
            "motorbike":     (150, (0, 0, 255)),
            "bicycle":       (140, (0, 255, 0)),
            "bus":           (300, (255, 0, 0)),
            "truck":         (500, (255, 128, 0)),
            "car":           (140, (200, 200, 0)),
        }
        results = []
        try:
            yolo_results = yolo_model(img_rgb, conf=self.conf_thresh, verbose=False)
            for res in yolo_results:
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    name   = res.names[cls_id]
                    if name not in target:
                        continue
                    conf = float(box.conf[0])
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    ref_h, color = info.get(name, (100, (128,128,128)))
                    results.append(InfraDetection(
                        class_name    = name,
                        bbox          = [max(0,x1), max(0,y1), min(w,x2), min(h,y2)],
                        confidence    = conf,
                        ref_height_cm = ref_h,
                        color_bgr     = color,
                        notes         = f"YOLO {name}",
                    ))
        except Exception as e:
            log.debug(f"  YOLO infra failed: {e}")
        return results

    # ------------------------------------------------------------------
    def _detect_utility_poles(
        self, img_bgr: np.ndarray, h: int, w: int
    ) -> List[InfraDetection]:
        """
        Phat hien cot dien / cot den bang vertical line detection.
        Cot: mau xam/den, chieu cao lon, chieu rong hep.
        """
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120, apertureSize=3)

        # Hough line: tim duong doc dai
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=h // 5,
            minLineLength=h // 3,   # it nhat 33% chieu cao anh
            maxLineGap=20,
        )

        results  = []
        min_w    = max(3, w // 200)   # cot toi thieu 0.5% chieu rong
        max_w    = max(20, w // 25)   # cot toi da 4% chieu rong

        if lines is None:
            return results

        pole_candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Chi lay duong gan doc (angle < 15 do so voi doc)
            if x2 == x1:
                angle = 90.0
            else:
                angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if 75 <= angle <= 105:   # 75-105 do = gan doc
                cx     = (x1 + x2) // 2
                length = abs(y2 - y1)
                pole_candidates.append((cx, min(y1,y2), max(y1,y2), length))

        # Gom cac line gan nhau thanh 1 cot
        used = [False] * len(pole_candidates)
        for i, (cx_i, top_i, bot_i, len_i) in enumerate(pole_candidates):
            if used[i]:
                continue
            used[i] = True
            group_xs  = [cx_i]
            group_top = top_i
            group_bot = bot_i

            for j, (cx_j, top_j, bot_j, len_j) in enumerate(pole_candidates):
                if used[j] or i == j:
                    continue
                if abs(cx_i - cx_j) < max_w * 2:
                    group_xs.append(cx_j)
                    group_top = min(group_top, top_j)
                    group_bot = max(group_bot, bot_j)
                    used[j] = True

            pole_h = group_bot - group_top
            pole_w = max(3, int(np.std(group_xs) * 2 + min_w))

            if pole_h < h // 4:   # qua ngan
                continue

            # Kiem tra mau cot (xam/den/nau)
            cx_mean = int(np.mean(group_xs))
            roi     = img_bgr[group_top:group_bot,
                               max(0, cx_mean - pole_w):min(w, cx_mean + pole_w)]
            if roi.size == 0:
                continue

            # Cot xam/nau: sat thap, val trung binh
            hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            sat_mean= float(hsv[:,:,1].mean())
            val_mean= float(hsv[:,:,2].mean())

            if sat_mean > 80:   # qua mau = khong phai cot
                continue
            if val_mean < 20 or val_mean > 220:   # qua toi/sang
                continue

            # Classify: den duong hay cot dien
            if pole_h > h * 0.7:
                cls  = "utility_pole"
                rh   = 800
                col  = (120, 80, 40)
            else:
                cls  = "street_lamp"
                rh   = 600
                col  = (200, 200, 50)

            x1 = max(0, cx_mean - pole_w)
            x2 = min(w, cx_mean + pole_w)
            results.append(InfraDetection(
                class_name    = cls,
                bbox          = [x1, group_top, x2, group_bot],
                confidence    = 0.55,
                ref_height_cm = rh,
                color_bgr     = col,
                notes         = f"Custom pole: h={pole_h}px sat={sat_mean:.0f}",
            ))

        return results[:3]   # Giu toi da 3 cot de tranh nhieu

    # ------------------------------------------------------------------
    def _detect_street_signs(
        self, img_bgr: np.ndarray, h: int, w: int
    ) -> List[InfraDetection]:
        """
        Phat hien bien bao duong (tam giac, tron, chu nhat) nguoi Viet Nam hay gap.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Mau chung: de y bien bao cam/mau do/vang
        red1 = cv2.inRange(hsv, np.array([0, 120, 80]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 120, 80]), np.array([180, 255, 255]))
        yellow = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))

        candidate = cv2.bitwise_or(cv2.bitwise_or(red1, red2), yellow)
        kernel = np.ones((7, 7), np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (h * w * 0.001):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / max(1.0, bh)
            if not (0.3 < aspect_ratio < 1.5):
                continue

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            width = rect[1][0]
            height = rect[1][1]

            # Tam giac/tron/chu nhat: radar simple, duoc bo sung tu Yolo
            if width == 0 or height == 0:
                continue

            shape_score = 0.0
            if abs(aspect_ratio - 1.0) < 0.4:
                shape_score = 0.8  # nghi tam giac/tron
            elif abs(aspect_ratio - 1.0) < 0.2:
                shape_score = 0.95

            # Kiem tra to mau la do/vang
            roi = hsv[y:y+bh, x:x+bw]
            if roi.size == 0:
                continue
            red_ratio = (cv2.countNonZero(cv2.inRange(roi, np.array([0,120,80]), np.array([10,255,255]))) +
                         cv2.countNonZero(cv2.inRange(roi, np.array([170,120,80]), np.array([180,255,255])))) / max(1, area)
            yellow_ratio = cv2.countNonZero(cv2.inRange(roi, np.array([15,100,100]), np.array([35,255,255]))) / max(1, area)

            if red_ratio < 0.02 and yellow_ratio < 0.02:
                continue

            result = InfraDetection(
                class_name    = "road_sign",
                bbox          = [x, y, x+bw, y+bh],
                confidence    = min(0.95, 0.5 + shape_score * 0.4 + max(0.0, red_ratio*10, yellow_ratio*10)),
                ref_height_cm = 180,
                color_bgr     = (0, 128, 255),
                notes         = f"Street sign candidate red={red_ratio:.2f} yellow={yellow_ratio:.2f}",
            )
            results.append(result)

        return results[:5]

    # ------------------------------------------------------------------
    def _detect_life_jackets(
        self, img_bgr: np.ndarray, h: int, w: int
    ) -> List[InfraDetection]:
        """
        Phat hien ao phao cuu ho (cam/vang neon).
        Nguoi mac ao phao thuong dang tren thuyen = KHONG tinh la ngap.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Ao phao cuu ho VN: cam neon (HSV 5-20, S>150, V>150)
        mask_orange = cv2.inRange(hsv, np.array([3, 150, 140]), np.array([22, 255, 255]))
        # Ao phao vang (HSV 22-35, S>150)
        mask_yellow = cv2.inRange(hsv, np.array([22, 150, 140]), np.array([38, 255, 255]))

        combined = cv2.bitwise_or(mask_orange, mask_yellow)

        # Morphological: loai nhieu
        kernel   = np.ones((10, 10), np.uint8)
        cleaned  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned  = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  kernel)

        results  = []
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ao phao: dien tich hop ly (0.5% - 25% anh)
            if not (h * w * 0.005 < area < h * w * 0.25):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Ty le chieu cao/rong hop ly cho nguoi
            if bh < bw * 0.3:   # qua dep ngang = khong phai nguoi
                continue

            results.append(InfraDetection(
                class_name    = "life_jacket_person",
                bbox          = [x, y, x+bw, y+bh],
                confidence    = 0.65,
                ref_height_cm = 170,
                color_bgr     = (0, 100, 255),
                is_on_boat    = True,   # Gia dinh la tren thuyen
                notes         = "Life jacket detected - likely on rescue boat",
            ))

        log.debug(f"  Life jackets: {len(results)} found")
        return results[:5]

    # ------------------------------------------------------------------
    def _detect_tideline(
        self, img_bgr: np.ndarray, h: int, w: int
    ) -> List[InfraDetection]:
        """
        Phat hien 'tideline' - vet muc nuoc tren tuong, cay, cot.
        Tideline la indicator chinh xac nhat ve muc nuoc da qua.

        Vet nuoc: duong ngang co su thay doi dot ngot ve mau sac:
          - Phan tren vet: kho (mau goc)
          - Phan duoi vet: uot/am (mau dam/toi hon)
          - Tai vet: co the co vach nuoc (water stain line)
        """
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Tim duong ngang co su thay doi dot ngot (gradient cao theo doc)
        gy    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gy_abs= np.abs(gy)

        # Tinh mean gradient theo hang
        row_grad = gy_abs.mean(axis=1)

        # Tim cac hang co gradient cao bat thuong (tideline candidates)
        thresh    = row_grad.mean() + 2.5 * row_grad.std()
        candidate_rows = np.where(row_grad > thresh)[0]

        if len(candidate_rows) == 0:
            return []

        # Chi xet vung giua anh (tidal mark thuong o 20%-80% chieu cao)
        valid_rows = candidate_rows[
            (candidate_rows > h * 0.20) &
            (candidate_rows < h * 0.85)
        ]

        if len(valid_rows) == 0:
            return []

        # Lay y-coordinate co gradient cao nhat
        best_row = int(valid_rows[np.argmax(row_grad[valid_rows])])
        grad_val = row_grad[best_row]

        # Kiem tra xac nhan: phan tren va duoi co su khac biet mau sac
        above = img_bgr[max(0, best_row-15):best_row, :]
        below = img_bgr[best_row:min(h, best_row+15), :]

        if above.size == 0 or below.size == 0:
            return []

        above_val = float(cv2.cvtColor(above, cv2.COLOR_BGR2GRAY).mean())
        below_val = float(cv2.cvtColor(below, cv2.COLOR_BGR2GRAY).mean())

        # Phan uot (duoi) thuong toi hon phan kho (tren)
        color_diff = above_val - below_val
        if color_diff < 5:   # khong co su khac biet ro rang
            return []

        confidence = min(0.75, (color_diff / 50.0) * (grad_val / thresh))

        log.info(
            f"  Tideline detected at y={best_row} "
            f"(diff={color_diff:.0f}, conf={confidence:.2f})"
        )

        return [InfraDetection(
            class_name    = "tideline",
            bbox          = [0, best_row, w, best_row + 5],
            confidence    = round(confidence, 3),
            ref_height_cm = 0,   # tideline la indicator, khong phai vat the
            color_bgr     = (255, 0, 255),
            tideline_y    = best_row,
            notes         = f"Tideline/water mark at y={best_row} (diff={color_diff:.0f})",
        )]

    # ------------------------------------------------------------------
    def draw_detections(
        self,
        img_bgr:    np.ndarray,
        detections: List[InfraDetection],
    ) -> np.ndarray:
        """Ve ket qua len anh."""
        overlay = img_bgr.copy()
        h, w    = img_bgr.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = det.color_bgr

            if det.class_name == "tideline" and det.tideline_y is not None:
                # Ve duong tideline ngang
                ty = det.tideline_y
                cv2.line(overlay, (0, ty), (w, ty), color, 2, cv2.LINE_AA)
                cv2.putText(overlay, "Tideline ~water mark",
                            (10, ty - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                continue

            # Bbox
            style = 2 if not det.is_on_boat else 1
            cv2.rectangle(overlay, (x1,y1), (x2,y2), color, style)

            # Label
            lbl = f"{det.class_name}"
            if det.is_on_boat:
                lbl += " [BOAT]"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(overlay, (x1, y1-th-4), (x1+tw+4, y1), (0,0,0), -1)
            cv2.putText(overlay, lbl, (x1+2, y1-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return overlay
