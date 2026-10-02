# -*- coding: utf-8 -*-
"""
depth_analysis/person_segmentor.py
====================================
Person detection + segmentation + context classification cho pipeline ngập.

3 bước:
  1. Person detection (YOLO bbox)
  2. Person segmentation (YOLO-seg hoặc SAM2-style mask từ bbox)
  3. Context classification (standing/riding/sitting/on_boat)

Chỉ người "standing" hoặc "walking" mới dùng làm reference object chính.

Context classes:
  standing_person, walking_person, riding_motorbike, sitting_person,
  person_on_boat, raincoat_person, partially_occluded_person

Dùng:
    ps = PersonSegmentor(cfg)
    persons = ps.detect(img_bgr, vehicle_bboxes=vehicles, water_mask=water_mask)
    ref_persons = [p for p in persons if p.usable_as_reference]
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("depth_analysis.person_seg")

PERSON_CONTEXT = {
    "standing_person",
    "walking_person",
    "riding_motorbike",
    "sitting_person",
    "person_on_boat",
    "raincoat_person",
    "partially_occluded_person",
}

# Keypoint indices (COCO 17-keypoint format)
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# ── [v5] Upper Body Anatomical Ruler constants ─────────────────────
# Chiều cao người trung bình (Việt Nam ~165cm)
PERSON_HEIGHT_CM = 165.0
# Khoảng cách đỉnh đầu → mỏm vai (acromion) người VN trưởng thành ≈ 25cm
HEAD_TO_CLAVICLE_CM = 25.0
# Chiều rộng 2 mỏm vai (biacromial) người VN ≈ 38-40cm
BIACROMIAL_CM = 39.0
# Ngưỡng hợp lệ cho px_per_cm (0.3 ~ rất xa, 15 ~ rất gần)
PX_PER_CM_MIN, PX_PER_CM_MAX = 0.3, 15.0
KP_CONF_MIN = 0.30


@dataclass
class PersonResult:
    bbox:              List[int]      # [x1, y1, x2, y2]
    confidence:        float = 0.5
    context:           str = "standing_person"
    mask:              Optional[np.ndarray] = None   # segmentation mask (0/255)
    keypoints:         Optional[Dict[str, Any]] = None
    water_depth_cm:    Optional[float] = None
    usable_as_reference: bool = False
    invalid_reason:    str = ""

    # Đặc điểm thân thể ước tính
    shoulder_height_cm: Optional[float] = None   # chiều cao vai so mặt đất
    hip_height_cm:      Optional[float] = None
    knee_height_cm:     Optional[float] = None


class PersonSegmentor:
    """
    Detect và phân loại người trong ảnh ngập.

    Dùng:
        ps = PersonSegmentor(cfg)
        persons = ps.detect(img_bgr, vehicle_bboxes=[], water_mask=water_mask)
        for p in persons:
            if p.usable_as_reference:
                depth = p.water_depth_cm  # chiều sâu nước ở người này
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.yolo_model_path = cfg.get("yolo_model", "yolov8n.pt")
        self.pose_model_path = cfg.get("pose_model", "yolov8n-pose.pt")
        self.min_confidence  = cfg.get("person_min_conf", 0.45)
        self._det_model  = None
        self._pose_model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        img_bgr: np.ndarray,
        vehicle_bboxes: Optional[List[List[int]]] = None,
        water_mask: Optional[np.ndarray] = None,
    ) -> List[PersonResult]:
        """
        Phát hiện người, phân loại context, đánh giá khả năng làm reference.

        Args:
            img_bgr:        ảnh BGR
            vehicle_bboxes: danh sách bbox xe (để detect người ngồi trên xe)
            water_mask:     mask nước (0/255)

        Returns:
            List[PersonResult]
        """
        h, w = img_bgr.shape[:2]
        raw = self._detect_persons(img_bgr)

        vehicle_bboxes = vehicle_bboxes or []
        results = []

        for bbox, conf in raw:
            if conf < self.min_confidence:
                continue

            # Keypoints từ pose model
            kps = self._get_keypoints(img_bgr, bbox)

            # Context classification
            context = self._classify_context(bbox, kps, vehicle_bboxes, water_mask, h)

            # Generate mask
            mask = self._bbox_to_mask(bbox, h, w)

            # Validate cho reference
            usable, reason = validate_person_for_depth(bbox, kps, context, conf)

            # Ước lượng chiều sâu nước
            water_depth = None
            if usable and water_mask is not None:
                water_depth = self._estimate_water_depth(bbox, kps, water_mask, h)

            results.append(PersonResult(
                bbox=bbox,
                confidence=conf,
                context=context,
                mask=mask,
                keypoints=kps,
                water_depth_cm=water_depth,
                usable_as_reference=usable,
                invalid_reason=reason,
            ))

        usable_count = sum(1 for p in results if p.usable_as_reference)
        log.debug("[PersonSeg] %d persons, %d usable as reference", len(results), usable_count)
        return results

    def get_person_mask(self, persons: List[PersonResult], h: int, w: int) -> np.ndarray:
        """Gộp tất cả person masks thành 1 mask — dùng để loại vùng người khỏi water mask."""
        combined = np.zeros((h, w), dtype=np.uint8)
        for p in persons:
            if p.mask is not None:
                combined = cv2.bitwise_or(combined, p.mask)
            else:
                x1, y1, x2, y2 = p.bbox
                combined[y1:y2, x1:x2] = 255
        return combined

    # ── Detection backends ────────────────────────────────────────────────────

    def _detect_persons(self, img_bgr: np.ndarray) -> List[Tuple[List[int], float]]:
        try:
            from ultralytics import YOLO
            if self._det_model is None:
                self._det_model = YOLO(self.yolo_model_path)
            results: Any = self._det_model(img_bgr, classes=[0], verbose=False)  # 0 = person
            persons = []
            for r in results or []:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    conf = float(box.conf)
                    persons.append((bbox, conf))
            return persons
        except Exception as exc:
            log.debug("[PersonSeg] Detection failed: %s", exc)
            return []

    def _get_keypoints(
        self,
        img_bgr: np.ndarray,
        bbox: List[int],
    ) -> Optional[Dict[str, Any]]:
        """Chạy pose estimation trên crop của người."""
        try:
            from ultralytics import YOLO
            if self._pose_model is None:
                self._pose_model = YOLO(self.pose_model_path)
            x1, y1, x2, y2 = bbox
            h, w = img_bgr.shape[:2]
            crop = img_bgr[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size == 0:
                return None
            results = self._pose_model(crop, verbose=False)
            for r in results:
                keypoints = getattr(r, "keypoints", None)
                if keypoints is not None and len(keypoints) > 0:
                    kps_xy  = keypoints.xy[0].cpu().numpy()   # (17, 2)
                    kps_conf= keypoints.conf[0].cpu().numpy()  # (17,)
                    # Chuyển về tọa độ ảnh gốc
                    kps_xy[:, 0] += x1
                    kps_xy[:, 1] += y1
                    kp_dict = {}
                    for name, idx in KP.items():
                        kp_dict[name] = {
                            "x": float(kps_xy[idx, 0]),
                            "y": float(kps_xy[idx, 1]),
                            "conf": float(kps_conf[idx]),
                        }
                    return kp_dict
        except Exception as exc:
            log.debug("[PersonSeg] Pose failed: %s", exc)
        return None

    # ── Context classification ────────────────────────────────────────────────

    def _classify_context(
        self,
        bbox: List[int],
        kps: Optional[Dict],
        vehicle_bboxes: List[List[int]],
        water_mask: Optional[np.ndarray],
        img_h: int,
    ) -> str:
        x1, y1, x2, y2 = bbox
        bbox_h = y2 - y1

        # Người ngồi trên xe?
        if vehicle_bboxes:
            for vbox in vehicle_bboxes:
                if _bbox_overlap_ratio(bbox, vbox) > 0.35:
                    return "riding_motorbike"

        # Người bị khuất >50%?
        if kps:
            visible = sum(1 for kp in kps.values() if kp["conf"] > 0.30)
            if visible < 6:
                return "partially_occluded_person"

        # Người ngồi (tỉ lệ bbox thấp, khớp hông gần đáy)?
        if kps:
            hip_y = max(
                kps.get("left_hip", {}).get("y", 0),
                kps.get("right_hip", {}).get("y", 0),
            )
            knee_y = max(
                kps.get("left_knee", {}).get("y", 0),
                kps.get("right_knee", {}).get("y", 0),
            )
            if hip_y > 0 and knee_y > 0 and (knee_y - hip_y) < bbox_h * 0.15:
                return "sitting_person"

        # Aspect ratio rất ngang → người trên thuyền / ngồi
        aspect = (x2 - x1) / max(bbox_h, 1)
        if aspect > 1.2:
            return "person_on_boat"

        # Mặc định: đứng hoặc đi bộ
        return "standing_person"

    # ── [v5] Upper Body Anatomical Ruler ─────────────────────────────
    @staticmethod
    def _anatomical_px_per_cm(kps: Optional[Dict]) -> Optional[float]:
        """
        Tính px_per_cm TỪ PHẦN TRÊN CƠ THỂ (luôn nổi trên mặt nước).

        Dùng tỉ lệ nhân trắc học bất biến ở upper body:
          - Đầu → vai (đỉnh đầu→mỏm vai) ≈ 25cm
          - Chiều rộng 2 mỏm vai (biacromial) ≈ 39cm

        Trả về px_per_cm thực tế tại cự ly đó, None nếu không đủ keypoints
        tin cậy. KHÔNG dùng bbox vì bbox bị co ngắn khi người ngập.
        """
        if not kps:
            return None

        def _kp(name: str) -> Optional[Tuple[float, float, float]]:
            d = kps.get(name)
            if not d or not isinstance(d, dict):
                return None
            return (float(d.get("x", 0)), float(d.get("y", 0)), float(d.get("conf", 0)))

        # 1) Biacromial width (2 mỏm vai) → px_per_cm = width_px / 39cm
        ls = _kp("left_shoulder")
        rs = _kp("right_shoulder")
        shoulder_est = None
        if ls and rs and ls[2] >= KP_CONF_MIN and rs[2] >= KP_CONF_MIN:
            width_px = abs(rs[0] - ls[0])
            if width_px >= 8:
                shoulder_est = (width_px / BIACROMIAL_CM,
                                min(ls[2], rs[2]))

        # 2) Đỉnh đầu → vai ≈ 25cm
        head_est = None
        head_y, head_conf = None, 0.0
        for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            k = _kp(name)
            if k and k[2] >= KP_CONF_MIN:
                if head_y is None or k[1] < head_y:
                    head_y, head_conf = k[1], k[2]
        sh_y, sh_conf = None, 0.0
        for name in ("left_shoulder", "right_shoulder"):
            k = _kp(name)
            if k and k[2] >= KP_CONF_MIN:
                if sh_y is None or k[1] < sh_y:
                    sh_y, sh_conf = k[1], k[2]
        if head_y is not None and sh_y is not None and (sh_y - head_y) >= 5:
            head_est = ((sh_y - head_y) / HEAD_TO_CLAVICLE_CM,
                        min(head_conf, sh_conf))

        # Ưu tiên biacromial (đáng tin hơn), falls back đầu→vai
        candidates = [c for c in (shoulder_est, head_est) if c]
        if not candidates:
            return None

        best = max(candidates, key=lambda c: c[1])  # conf cao nhất
        px_per_cm = best[0]
        if not (PX_PER_CM_MIN <= px_per_cm <= PX_PER_CM_MAX):
            return None
        return px_per_cm

    def _estimate_water_depth(
        self,
        bbox: List[int],
        kps: Optional[Dict],
        water_mask: np.ndarray,
        img_h: int,
    ) -> Optional[float]:
        """
        Ước lượng chiều sâu nước ở người bằng Upper Body Anatomical Ruler.

        [FIX v5] Bug cũ (luôn ra 8-15cm):
          - Bbox khi người ngập CHỈ bao phần nổi trên mặt nước
            (y2 ≈ mực nước, không phải chân). person_h_px = phần nổi.
          - Nước chỉ chạm vài hàng pixel đáy bbox
            → highest_water_row ≈ person_h_px → water_pct ≈ 6%
            → 0.06 × 165 = ~10cm BẤT KỂ ngập đến hông (80cm) hay cổ (130cm).
          - Biến kps được truyền vào nhưng hoàn toàn bị bỏ qua.

        Công thức mới (độc lập bbox, dùng upper body luôn nổi):
          1. px_per_cm từ phần trên cơ thể (đầu→vai 25cm / biacromial 39cm)
          2. y_ground = y_head + 165 × px_per_cm  (vị trí chân ảo)
          3. water_depth_cm = (y_ground − y_waterline) / px_per_cm

        Returns:
            cm độ ngập, None nếu không đủ dữ liệu.
        """
        x1, y1, x2, y2 = bbox
        if y2 - y1 < 10:
            return None

        px_per_cm = self._anatomical_px_per_cm(kps)
        if px_per_cm is None:
            return None

        # Đỉnh đầu (dùng keypoint; fallback top bbox)
        head_y = y1
        for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            d = (kps or {}).get(name)
            if d and isinstance(d, dict) and d.get("conf", 0) >= KP_CONF_MIN:
                head_y = min(float(d.get("y", y1)), head_y)

        # Vị trí chân ảo: y_head + 165cm × px_per_cm
        y_ground = head_y + PERSON_HEIGHT_CM * px_per_cm

        # Tìm y mực nước (y nhỏ nhất có nước trong vùng người)
        person_water = water_mask[y1:y2, x1:x2]
        water_rows = np.where((person_water > 0).any(axis=1))[0]
        if len(water_rows) == 0:
            return None
        waterline_y = y1 + float(water_rows.min())  # row nước cao nhất

        # Nước ngập thực tế
        water_depth_cm = (y_ground - waterline_y) / px_per_cm
        return round(max(0.0, water_depth_cm), 1)

    @staticmethod
    def _bbox_to_mask(bbox: List[int], h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 255
        return mask


# ── Validation function ────────────────────────────────────────────────────────

def validate_person_for_depth(
    bbox: List[int],
    kps: Optional[Dict],
    context: str,
    confidence: float,
) -> Tuple[bool, str]:
    """
    Kiểm tra người có phù hợp để dùng làm reference object cho depth không.

    Rules:
      1. confidence ≥ 0.45
      2. Có pose keypoints (ít nhất vai + hông hoặc vai + đầu gối)
      3. Không phải riding_motorbike hoặc person_on_boat
    """
    if confidence < 0.45:
        return False, f"low confidence ({confidence:.2f})"

    if context in ("riding_motorbike", "person_on_boat", "sitting_person"):
        return False, f"context not suitable: {context}"

    if kps is not None:
        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        visible = sum(1 for k in required if kps.get(k, {}).get("conf", 0) > 0.35)
        if visible < 2:
            alt_required = ["left_shoulder", "right_shoulder", "left_knee", "right_knee"]
            visible_alt = sum(1 for k in alt_required if kps.get(k, {}).get("conf", 0) > 0.35)
            if visible_alt < 2:
                return False, "insufficient visible keypoints"
    else:
        # Không có pose → vẫn cho phép nhưng flag
        pass

    return True, ""


def _bbox_overlap_ratio(bbox_a: List[int], bbox_b: List[int]) -> float:
    """Tỉ lệ diện tích giao / diện tích nhỏ hơn."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(min(area_a, area_b), 1)
