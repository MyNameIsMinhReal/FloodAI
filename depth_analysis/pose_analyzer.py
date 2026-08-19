# -*- coding: utf-8 -*-
"""
depth_analysis/pose_analyzer.py
---------------------------------
Phân tích tu the (pose) cua nguoi dung YOLO-Pose (yolov8n-pose.pt).

Phat hien:
  - STANDING:   dung thang, hai chan xuong thap
  - SITTING:    ngoi (tren xe, ghe), chan gap, khong duoc tinh la ngap
  - WADING:     loi nuoc (hai tay giang ra, chan trong nuoc)
  - CROUCHING:  ngoi xom
  - RIDING:     ngoi tren xe (motorcycle/bicycle) -> khong tinh ngap
  - UNKNOWN:    khong xac dinh duoc

Keypoints YOLO-Pose (17 diem):
  0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
  5:left_shoulder     6:right_shoulder
  7:left_elbow        8:right_elbow
  9:left_wrist        10:right_wrist
  11:left_hip         12:right_hip
  13:left_knee        14:right_knee
  15:left_ankle       16:right_ankle

Cai dat: pip install ultralytics
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple, cast

from utils.constants import DEFAULT_POSE_MODEL

import cv2
import numpy as np

log = logging.getLogger(__name__)


class PoseType(str, Enum):
    STANDING  = "STANDING"    # dung - do muc nuoc binh thuong
    WADING    = "WADING"      # loi nuoc - dau hieu ro nhat
    SITTING   = "SITTING"     # ngoi - giam chieu cao tham chieu
    CROUCHING = "CROUCHING"   # ngoi xom - giam chieu cao
    BENDING   = "BENDING"     # cuoi nguoi
    HUNCHING  = "HUNCHING"    # gu lung
    RIDING    = "RIDING"      # ngoi xe - KHONG tinh ngap
    UNKNOWN   = "UNKNOWN"     # khong xac dinh


# He so dieu chinh chieu cao tham chieu theo tu the
# RIDING: 0 = bo qua hoan toan (nguoi ngoi xe khong the do muc nuoc)
POSE_HEIGHT_FACTOR = {
    PoseType.STANDING:  1.00,   # dung thang = 100% chieu cao
    PoseType.WADING:    1.00,   # loi nuoc, van dung thang
    PoseType.SITTING:   0.55,   # ngoi = ~55% chieu cao dung
    PoseType.CROUCHING: 0.50,   # ngoi xom = ~50%
    PoseType.BENDING:   0.85,   # cuoi nguoi nguoc
    PoseType.HUNCHING:  0.90,   # gu lung
    PoseType.RIDING:    0.00,   # ngoi xe = bo qua
    PoseType.UNKNOWN:   0.85,   # mac dinh bao toan
}


@dataclass
class PersonPose:
    bbox:           List[int]        # [x1, y1, x2, y2]
    pose_type:      PoseType
    height_factor:  float            # he so dieu chinh
    confidence:     float            # do tin cay cua phan tich pose
    keypoints:      Optional[np.ndarray]  # (17, 3) array [x, y, conf]
    notes:          str              # mo ta chi tiet


class PoseAnalyzer:
    """Phân tích tu the nguoi dung YOLO-Pose.

    Nâng cấp v3:
      - Chỉ dùng keypoints quan trọng (shoulders, hips, knees, ankles)
      - Smooth keypoints theo thời gian: kp_t = alpha*kp_t-1 + (1-alpha)*kp_t
      - Reject pose lỗi nếu thiếu > 2 keypoints quan trọng
      - Aspect ratio filter: reject bbox bất thường
      - Detect "loose clothing" → signal áo mưa rộng
    """

    # Keypoints quan trọng — chỉ dùng 8/17 điểm này
    ESSENTIAL_KP = [5, 6, 11, 12, 13, 14, 15, 16]  # shoulders, hips, knees, ankles

    def __init__(
        self,
        pose_model:   str   = DEFAULT_POSE_MODEL,
        conf_thresh:  float = 0.4,
        smooth_alpha: float = 0.70,   # hệ số smooth keypoints
        max_missing:  int   = 2,      # reject nếu thiếu > max_missing kp thiết yếu
    ):
        self.pose_model   = pose_model
        self.conf_thresh  = conf_thresh
        self.smooth_alpha = smooth_alpha
        self.max_missing  = max_missing
        self._model       = None
        # Keypoint history per track (track_id → kpts array)
        self._kp_history: dict = {}

    def _load(self):
        if self._model:
            return
        from ultralytics import YOLO
        log.info(f"  Loading YOLO-Pose: {self.pose_model}")
        self._model = YOLO(self.pose_model)
        log.info("  YOLO-Pose loaded")

    # ------------------------------------------------------------------
    def analyze_image(
        self,
        img_rgb:    np.ndarray,
        track_ids:  Optional[List[int]] = None,  # optional tracker IDs per detection
    ) -> List[PersonPose]:
        """
        Phân tích tất cả người trong ảnh — v3 nâng cấp.

        Cải tiến:
          1. Aspect ratio filter: reject bbox h/w < 1.2 hoặc > 5.0
          2. Reject pose lỗi: thiếu > max_missing keypoints thiết yếu
          3. Smooth keypoints theo track_id (nếu có)
          4. Gắn thêm loose_clothing flag (shoulder/bbox_w ratio < 0.4)
        """
        self._load()
        assert self._model is not None
        results = self._model(img_rgb, conf=self.conf_thresh, verbose=False)
        poses   = []

        for result in results:
            # The YOLO type stubs may infer each item as a Tensor, while
            # runtime inference returns a Results object.
            result = cast(Any, result)
            if result.keypoints is None:
                continue
            kpts_data = result.keypoints.data   # Tensor (N, 17, 3)
            boxes     = result.boxes

            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                kpts = kpts_data[i].cpu().numpy()   # (17, 3): x, y, conf
                conf = float(boxes.conf[i])

                bbox_h = max(y2 - y1, 1)
                bbox_w = max(x2 - x1, 1)

                # ── Filter 1: Aspect ratio ──────────────────────────────
                ar = bbox_h / bbox_w
                if ar < 1.2 or ar > 5.0:
                    log.debug(f"  Pose: reject bbox ar={ar:.2f} (out of [1.2, 5.0])")
                    continue

                # ── Filter 2: Reject pose lỗi ──────────────────────────
                missing = sum(
                    1 for idx in self.ESSENTIAL_KP
                    if kpts[idx, 2] <= 0.3
                )
                if missing > self.max_missing:
                    log.debug(f"  Pose: reject — missing {missing} essential kpts")
                    continue

                # ── Smooth keypoints theo track_id ─────────────────────
                track_id = track_ids[i] if track_ids and i < len(track_ids) else None
                kpts = self._smooth_keypoints(kpts, track_id)

                pose = self._classify_pose(kpts, [x1, y1, x2, y2], img_rgb.shape[:2])
                pose.confidence = conf

                # ── Loose clothing detection ────────────────────────────
                loose = self._detect_loose_clothing(kpts, bbox_w)
                if loose:
                    pose.notes += " | LOOSE_CLOTHING(possible_raincoat)"

                poses.append(pose)

        log.debug(f"  Pose: detected {len(poses)} persons (after filters)")
        return poses

    def _smooth_keypoints(
        self,
        kpts:     np.ndarray,
        track_id: Optional[int],
    ) -> np.ndarray:
        """
        Smooth keypoints theo thời gian: kp_t = alpha*kp_t-1 + (1-alpha)*kp_t

        Chỉ smooth x, y — giữ confidence hiện tại.
        """
        if track_id is None:
            return kpts

        prev = self._kp_history.get(track_id)
        if prev is not None and prev.shape == kpts.shape:
            smoothed = kpts.copy()
            for k in range(17):
                if prev[k, 2] > 0.3 and kpts[k, 2] > 0.3:
                    smoothed[k, 0] = self.smooth_alpha * prev[k, 0] + (1 - self.smooth_alpha) * kpts[k, 0]
                    smoothed[k, 1] = self.smooth_alpha * prev[k, 1] + (1 - self.smooth_alpha) * kpts[k, 1]
            kpts = smoothed

        self._kp_history[track_id] = kpts.copy()
        # Giữ history nhỏ gọn (tối đa 200 tracks)
        if len(self._kp_history) > 200:
            oldest = next(iter(self._kp_history))
            del self._kp_history[oldest]

        return kpts

    def _detect_loose_clothing(
        self,
        kpts:   np.ndarray,
        bbox_w: float,
    ) -> bool:
        """
        Phát hiện quần áo rộng từ pose — dấu hiệu áo mưa.

        Nếu shoulder_width / bbox_width < 0.4 → vai hẹp so với bbox
        → áo rộng bao quanh người → possible raincoat
        """
        def kp(idx):
            if kpts[idx, 2] > 0.3:
                return float(kpts[idx, 0])
            return None

        l_sh = kp(5)
        r_sh = kp(6)

        if l_sh is None or r_sh is None:
            return False

        shoulder_w = abs(l_sh - r_sh)
        ratio = shoulder_w / (bbox_w + 1e-6)
        return ratio < 0.40

    # ------------------------------------------------------------------
    def _compute_pose_features(
        self,
        kpts: np.ndarray,
        bbox: List[int],
        img_shape: Tuple,
    ) -> dict:
        """Compute all pose features from keypoints and bbox."""
        _, _ = img_shape  # unused
        x1, y1, x2, y2 = bbox
        bbox_h = max(y2 - y1, 1)
        bbox_w = max(x2 - x1, 1)

        # Index keypoints (chi lay nhung diem co conf > 0.3)
        def kp(idx) -> Optional[Tuple[float, float]]:
            if kpts[idx, 2] > 0.3:
                return float(kpts[idx, 0]), float(kpts[idx, 1])
            return None

        l_shoulder   = kp(5);  r_shoulder = kp(6)
        _            = kp(7);  _          = kp(8)  # unused elbows
        l_wrist      = kp(9);  r_wrist    = kp(10)
        l_hip        = kp(11); r_hip      = kp(12)
        l_knee       = kp(13); r_knee     = kp(14)
        l_ankle      = kp(15); r_ankle    = kp(16)

        # --- Tinh cac ti le quan trong ---
        # Vi tri hip (hong) trong bbox: 0 = dinh, 1 = day
        hip_y = None
        if l_hip and r_hip:
            hip_y = (l_hip[1] + r_hip[1]) / 2
        elif l_hip:
            hip_y = l_hip[1]
        elif r_hip:
            hip_y = r_hip[1]

        hip_ratio = (hip_y - y1) / bbox_h if hip_y is not None else 0.5

        # Vi tri knee (goi)
        knee_y = None
        if l_knee and r_knee:
            knee_y = (l_knee[1] + r_knee[1]) / 2
        elif l_knee:
            knee_y = l_knee[1]
        elif r_knee:
            knee_y = r_knee[1]

        knee_ratio = (knee_y - y1) / bbox_h if knee_y is not None else 0.7

        # Vi tri ankle
        ankle_y = None
        if l_ankle and r_ankle:
            ankle_y = (l_ankle[1] + r_ankle[1]) / 2
        elif l_ankle:
            ankle_y = l_ankle[1]
        elif r_ankle:
            ankle_y = r_ankle[1]

        ankle_ratio = (ankle_y - y1) / bbox_h if ankle_y is not None else 0.9

        # Chieu rong vai
        shoulder_width = 0
        if l_shoulder and r_shoulder:
            shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
            _ = (l_shoulder[0] + r_shoulder[0]) / 2.0  # unused mid_x
        shoulder_ratio = shoulder_width / bbox_w

        # Goc than (shoulder->hip)
        torso_angle = self._calc_torso_angle(l_shoulder, r_shoulder, l_hip, r_hip)

        # Goc gap goi (knee angle)
        knee_angle = self._calc_knee_angle(l_hip, l_knee, l_ankle,
                                            r_hip, r_knee, r_ankle)

        # Tay giang cao (dau hieu loi nuoc)
        wrist_high = False
        if l_wrist and l_shoulder:
            wrist_high = l_wrist[1] < l_shoulder[1] + 20
        if r_wrist and r_shoulder:
            wrist_high = wrist_high or (r_wrist[1] < r_shoulder[1] + 20)

        notes_parts = [
            f"hip_ratio={hip_ratio:.2f}",
            f"knee_ratio={knee_ratio:.2f}",
            f"torso_angle={torso_angle:.1f}deg",
            f"knee_angle={knee_angle:.0f}deg",
            f"shoulder_ratio={shoulder_ratio:.2f}",
        ]

        return {
            'hip_ratio': hip_ratio,
            'knee_ratio': knee_ratio,
            'ankle_ratio': ankle_ratio,
            'shoulder_ratio': shoulder_ratio,
            'torso_angle': torso_angle,
            'knee_angle': knee_angle,
            'wrist_high': wrist_high,
            'bbox_w': bbox_w,
            'bbox_h': bbox_h,
            'notes_parts': notes_parts,
        }

    # ------------------------------------------------------------------
    def _classify_pose(
        self,
        kpts:  np.ndarray,    # (17, 3)
        bbox:  List[int],
        img_shape: Tuple,
    ) -> PersonPose:
        """
        Phân tích tu the dua tren vi tri keypoints.
        
        Logic:
          RIDING:    hong cao (> 40% bbox), goi thap, chan khep
          SITTING:   hong cao, goi gap nhieu
          WADING:    vai giang rong, tay giang cao, chan ro
          CROUCHING: hong xuong thap, goi rat gap
          STANDING:  hong thap, chan thang
        """
        # Index keypoints (chi lay nhung diem co conf > 0.3)
        def kp(idx) -> Optional[Tuple[float, float]]:
            if kpts[idx, 2] > 0.3:
                return float(kpts[idx, 0]), float(kpts[idx, 1])
            return None

        # Dem so keypoints hop le — chỉ đếm ESSENTIAL_KP
        valid_keypoints = [kp(i) for i in self.ESSENTIAL_KP]
        valid_kpts = sum(1 for k in valid_keypoints if k is not None)

        if valid_kpts < 3:
            return PersonPose(bbox=bbox, pose_type=PoseType.UNKNOWN,
                              height_factor=0.85, confidence=0.3,
                              keypoints=kpts, notes="Too few keypoints")

        features = self._compute_pose_features(kpts, bbox, img_shape)

        pose_type, height_factor, confidence, note_prefix = self._determine_pose_type(
            hip_ratio=features['hip_ratio'],
            knee_ratio=features['knee_ratio'],
            knee_angle=features['knee_angle'],
            torso_angle=features['torso_angle'],
            shoulder_ratio=features['shoulder_ratio'],
            wrist_high=features['wrist_high'],
            ankle_ratio=features['ankle_ratio'],
            bbox_w=features['bbox_w'],
            bbox_h=features['bbox_h'],
        )

        return PersonPose(
            bbox=bbox,
            pose_type=pose_type,
            height_factor=height_factor,
            confidence=confidence,
            keypoints=kpts,
            notes=f"{note_prefix} {', '.join(features['notes_parts'])}",
        )

    # ------------------------------------------------------------------
    def _calc_knee_angle(self, l_hip, l_knee, l_ankle,
                          r_hip, r_knee, r_ankle) -> float:
        """
        Tinh goc gap goi trung binh (0 = gap hoan toan, 180 = thang).
        """
        angles = []
        for hip, knee, ankle in [(l_hip, l_knee, l_ankle),
                                   (r_hip, r_knee, r_ankle)]:
            if hip and knee and ankle:
                v1 = np.array([hip[0] - knee[0],    hip[1] - knee[1]])
                v2 = np.array([ankle[0] - knee[0],  ankle[1] - knee[1]])
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
                angles.append(angle)
        return float(np.mean(angles)) if angles else 160.0

    # ------------------------------------------------------------------
    def _is_hunching(self, torso_angle: float, shoulder_ratio: float) -> bool:
        return 20 < torso_angle <= 45 and shoulder_ratio > 0.35

    def _is_bending(self, torso_angle: float, knee_angle: float) -> bool:
        return 35 < torso_angle <= 80 and knee_angle > 120

    def _is_riding(self, hip_ratio: float, knee_ratio: float, knee_angle: float,
                   bbox_w: float, bbox_h: float) -> bool:
        return (hip_ratio < 0.55 and knee_ratio < 0.75
                and knee_angle < 110 and bbox_w < bbox_h * 0.9)

    def _is_sitting(self, hip_ratio: float, knee_angle: float) -> bool:
        return hip_ratio < 0.55 and knee_angle < 120

    def _is_crouching(self, hip_ratio: float, knee_angle: float) -> bool:
        return hip_ratio > 0.55 and knee_angle < 90

    def _is_wading(self, wrist_high: bool, ankle_ratio: float, knee_angle: float) -> bool:
        return wrist_high and ankle_ratio > 0.75 and knee_angle > 140

    def _is_standing(self, knee_angle: float, ankle_ratio: float) -> bool:
        return knee_angle > 150 and ankle_ratio > 0.75

    def _determine_pose_type(
        self,
        hip_ratio: float,
        knee_ratio: float,
        knee_angle: float,
        torso_angle: float,
        shoulder_ratio: float,
        wrist_high: bool,
        ankle_ratio: float,
        bbox_w: float,
        bbox_h: float,
    ) -> tuple[PoseType, float, float, str]:
        if self._is_hunching(torso_angle, shoulder_ratio):
            return PoseType.HUNCHING, 0.90, 0.70, "Hunching: gu lung nhon"
        if self._is_bending(torso_angle, knee_angle):
            return PoseType.BENDING, 0.85, 0.72, "Bending: cuoi nguoi (torso goc)"
        if self._is_riding(hip_ratio, knee_ratio, knee_angle, bbox_w, bbox_h):
            return PoseType.RIDING, 0.0, 0.75, "Riding: high hip, bent knee, narrow bbox."
        if self._is_sitting(hip_ratio, knee_angle):
            return PoseType.SITTING, 0.55, 0.70, "Sitting: high hip, bent knee."
        if self._is_crouching(hip_ratio, knee_angle):
            return PoseType.CROUCHING, 0.50, 0.70, "Crouching: low hip, very bent knee."
        if self._is_wading(wrist_high, ankle_ratio, knee_angle):
            return PoseType.WADING, 1.00, 0.80, "Wading: raised arms, straight legs."
        if self._is_standing(knee_angle, ankle_ratio):
            return PoseType.STANDING, 1.00, 0.85, "Standing: straight legs."
        return PoseType.UNKNOWN, 0.85, 0.50, "Unknown pose."

    # ------------------------------------------------------------------
    def _calc_torso_angle(
        self,
        l_shoulder: Optional[Tuple[float,float]],
        r_shoulder: Optional[Tuple[float,float]],
        l_hip: Optional[Tuple[float,float]],
        r_hip: Optional[Tuple[float,float]],
    ) -> float:
        """Tinh goc giua duong shoulder->hip so voi truc dung (0=thang)."""
        if not (l_shoulder and r_shoulder and l_hip and r_hip):
            return 0.0
        sx = (l_shoulder[0] + r_shoulder[0]) / 2.0
        sy = (l_shoulder[1] + r_shoulder[1]) / 2.0
        hx = (l_hip[0] + r_hip[0]) / 2.0
        hy = (l_hip[1] + r_hip[1]) / 2.0
        dx = hx - sx
        dy = hy - sy
        if abs(dy) < 1e-3:
            return 90.0
        return float(abs(np.degrees(np.arctan2(abs(dx), abs(dy)))))

    # ------------------------------------------------------------------
    # 3.4 ROBUST FLOOD DEPTH FROM MULTIPLE POSES (MEDIAN + OUTLIER FILTER)
    # ------------------------------------------------------------------

    def estimate_flood_depth_robust(
        self,
        poses: List["PersonPose"],
        img_h: int,
        img_w: int,
        assumed_height_cm: float = 165.0,  # chiều cao trung bình người Việt Nam
        iqr_multiplier: float = 1.5,       # hệ số IQR cho outlier detection
    ) -> Optional[dict]:
        """
        Ước tính mực nước từ nhiều poses.

        Cải tiến v2:
          1. Loại outlier keypoints: dùng IQR thay vì mean
          2. Dùng MEDIAN thay vì mean để giảm ảnh hưởng của outliers
          3. Ưu tiên STANDING và WADING poses (chính xác nhất)
          4. Penalty cho poses không đáng tin cậy (SITTING, RIDING, UNKNOWN)

        Args:
            poses:            list PersonPose từ analyze_image()
            img_h, img_w:     kích thước ảnh
            assumed_height_cm: chiều cao giả định
            iqr_multiplier:   hệ số IQR (1.5 = standard, 3.0 = very lenient)

        Returns:
            dict với flood_depth_cm, confidence, method, n_poses_used, ...
        """
        # Lọc poses hợp lệ (bỏ RIDING và UNKNOWN kém chất lượng)
        valid_poses = [
            p for p in poses
            if (p.pose_type not in (PoseType.RIDING,) and
                p.confidence > 0.25 and
                p.height_factor > 0)
        ]

        if not valid_poses:
            log.debug("  No valid poses for flood depth estimation")
            return None

        # Ưu tiên poses: WADING=1.0, STANDING=0.9, rest lower
        pose_reliability = {
            PoseType.STANDING:  0.90,
            PoseType.WADING:    1.00,
            PoseType.BENDING:   0.70,
            PoseType.HUNCHING:  0.75,
            PoseType.CROUCHING: 0.50,
            PoseType.SITTING:   0.40,
            PoseType.UNKNOWN:   0.25,
        }

        # Ước tính flood depth từ từng pose
        depth_estimates = []
        weights = []

        for pose in valid_poses:
            # Chiều cao bbox thực của người (pixels)
            x1, y1, x2, y2 = pose.bbox
            bbox_h = y2 - y1
            if bbox_h < 20:
                continue

            # px_per_cm: pixels per cm dựa trên chiều cao pose
            effective_height_cm = assumed_height_cm * pose.height_factor
            if effective_height_cm < 1:
                continue
            px_per_cm = bbox_h / effective_height_cm

            # Foot position (y coordinate của chân trong ảnh)
            foot_y = y2

            # Water line estimate: vùng nước ở dưới → foot_y là chân ngập
            # Flood depth = (img_h - foot_y) * px_per_cm ... không, cần position nước
            # Dùng heuristic: nếu bottom của bbox KHÔNG ở bottom ảnh → có nước ở dưới
            margin_below = img_h - foot_y  # pixels bên dưới chân người
            flood_depth_px = max(0, margin_below)

            # Nếu người đang lội nước (WADING) → chân đang trong nước → flood = margin + 1 bước chân
            if pose.pose_type == PoseType.WADING:
                # Giả sử nước đến đầu gối hoặc hông
                flood_depth_px += bbox_h * 0.3

            flood_depth_cm = flood_depth_px / (px_per_cm + 1e-6)

            # Confidence-weighted
            reliability = pose_reliability.get(pose.pose_type, 0.5)
            w = pose.confidence * reliability
            depth_estimates.append(flood_depth_cm)
            weights.append(w)

        if not depth_estimates:
            return None

        depth_arr = np.array(depth_estimates)
        weight_arr = np.array(weights)

        # Outlier filtering: IQR method
        if len(depth_arr) >= 4:
            q1 = np.percentile(depth_arr, 25)
            q3 = np.percentile(depth_arr, 75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            valid_idx = (depth_arr >= lower_bound) & (depth_arr <= upper_bound)

            if valid_idx.sum() >= 2:
                depth_arr  = depth_arr[valid_idx]
                weight_arr = weight_arr[valid_idx]
                n_removed  = (~valid_idx).sum()
                if n_removed > 0:
                    log.debug(f"  PoseAnalyzer: removed {n_removed} outlier depth estimates")

        # MEDIAN thay vì mean (robust hơn)
        median_depth = float(np.median(depth_arr))

        # Weighted mean để so sánh
        if weight_arr.sum() > 0:
            weighted_mean = float(np.average(depth_arr, weights=weight_arr))
        else:
            weighted_mean = float(np.mean(depth_arr))

        # Confidence: cao khi nhiều poses đồng thuận (std thấp)
        std_depth = float(np.std(depth_arr))
        cv = std_depth / (median_depth + 1e-6)
        base_conf = float(np.mean(weight_arr))
        consistency_bonus = max(0.0, 0.2 - cv * 0.1)
        confidence = min(0.95, base_conf + consistency_bonus)

        # n_poses ưu tiên
        n_reliable = sum(
            1 for p in valid_poses
            if p.pose_type in (PoseType.STANDING, PoseType.WADING)
        )

        return {
            "flood_depth_cm":   round(median_depth, 1),
            "flood_depth_mean": round(weighted_mean, 1),
            "std_cm":           round(std_depth, 1),
            "confidence":       round(confidence, 3),
            "n_poses_used":     len(depth_arr),
            "n_reliable_poses": n_reliable,
            "method":           "median_iqr_filtered",
            "pose_types_used":  [p.pose_type.value for p in valid_poses[:5]],
            "notes": (
                f"Median: {median_depth:.1f}cm ± {std_depth:.1f}cm "
                f"from {len(depth_arr)} poses (IQR filtered). "
                f"Reliable (STANDING/WADING): {n_reliable}"
            ),
        }

    # ------------------------------------------------------------------

        """Ve keypoints va nhan tu the len anh."""
        overlay = img_bgr.copy()
        colors = {
            PoseType.STANDING:  (0,   255,   0),
            PoseType.WADING:    (0,   200, 255),
            PoseType.SITTING:   (255, 165,   0),
            PoseType.CROUCHING: (255, 200,   0),
            PoseType.RIDING:    (200, 200, 200),
            PoseType.UNKNOWN:   (128, 128, 128),
        }
        skeleton = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),    # arms
            (5, 11), (6, 12), (11, 12),                    # torso
            (11, 13), (13, 15), (12, 14), (14, 16),       # legs
            (0, 5), (0, 6),                                # neck
        ]

        for pose in poses:
            if pose.keypoints is None:
                continue
            color = colors.get(pose.pose_type, (128, 128, 128))
            kpts  = pose.keypoints

            # Ve skeleton
            for a, b in skeleton:
                if kpts[a, 2] > 0.3 and kpts[b, 2] > 0.3:
                    pt1 = (int(kpts[a, 0]), int(kpts[a, 1]))
                    pt2 = (int(kpts[b, 0]), int(kpts[b, 1]))
                    cv2.line(overlay, pt1, pt2, color, 2)

            # Ve keypoints
            for k in range(17):
                if kpts[k, 2] > 0.3:
                    cx, cy = int(kpts[k, 0]), int(kpts[k, 1])
                    cv2.circle(overlay, (cx, cy), 4, color, -1)

            # Nhan tu the
            x1, y1 = pose.bbox[0], pose.bbox[1]
            label  = f"{pose.pose_type.value} (x{pose.height_factor:.2f})"
            cv2.putText(overlay, label, (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return overlay
