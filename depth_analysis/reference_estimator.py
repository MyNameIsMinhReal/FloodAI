# -*- coding: utf-8 -*-
"""
depth_analysis/reference_estimator.py  -  v4 (fixed)
======================================================
Do muc nuoc lu bang reference objects.

Cac bug da fix:
  v3: ref_height_cm bi nhan voi height_factor -> label sai (85cm, 144cm)
      Fix: giu ref_height_cm = chieu cao thuc, dung pose_factor chi khi tinh water_cm
  v3: water line qua cao -> object "0% visible" du nhin thay ro
      Fix: sanity check manh hon, giam confidence khi mau thuan
  v3: arrow ve ca khi object KHONG bi ngap
      Fix: chi ve arrow khi pixels_below_water > 5% tong chieu cao
  v3: label che nhau khi nhieu object
      Fix: smart placement - thu phia tren, neu bi che thi dat phia duoi
"""

import logging
from collections import Counter
from dataclasses import dataclass, field, asdict

from utils.constants import FLOOD_LEVEL_KNEE, FLOOD_LEVEL_HIP, FLOOD_LEVEL_CHEST, FLOOD_LEVEL_COMPLETE
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np
from PIL import Image
from depth_analysis.pose_analyzer import PoseType
from utils.constants import DEFAULT_YOLO_MODEL, DEFAULT_DEPTH_MODEL, DEFAULT_DINO_MODEL, DEFAULT_POSE_MODEL

# [IMPROVE] Unified camera analyzer — thay the _detect_camera_angle rieng le
_cam_analyzer = None
def _get_cam_analyzer():
    global _cam_analyzer
    if _cam_analyzer is None:
        from depth_analysis.perspective_analyzer import PerspectiveAnalyzer
        _cam_analyzer = PerspectiveAnalyzer()
    return _cam_analyzer

log = logging.getLogger(__name__)

# Shoulder width estimator (lazy import to avoid circular)
_shoulder_est = None
def _get_shoulder_est():
    global _shoulder_est
    if _shoulder_est is None:
        from depth_analysis.shoulder_estimator import ShoulderWidthEstimator
        _shoulder_est = ShoulderWidthEstimator()
    return _shoulder_est

# ─── Flood level thresholds (cm) ─────────────────────────────────────────────
FLOOD_LEVELS = [
    (0,    0,   "NO_FLOOD",  "Không có lũ"),
    (0,   15,   "PUDDLE",    "Vũng nước nhỏ (<15cm)"),
    (15,  40,   "ANKLE",     "Ngập mắt cá (15-40cm)"),
    (40,  70,   "KNEE",      FLOOD_LEVEL_KNEE),
    (70, 120,   "WAIST",     FLOOD_LEVEL_HIP),
    (120, 200,  "CHEST",     FLOOD_LEVEL_CHEST),
    (200, 9999, "SUBMERGED", FLOOD_LEVEL_COMPLETE),
]

# ─── Chieu cao THUC cua tung loai vat the (cm) - KHONG thay doi ──────────────
OBJ_STOP_SIGN = "stop sign"
OBJ_FIRE_HYDRANT = "fire hydrant"
OBJ_TRAFFIC_LIGHT = "traffic light"
OBJ_DOOR = "cua_nha"   # cua nha: chieu cao chuan ~200cm, khong thay doi

REF_HEIGHTS: Dict[str, int] = {
    OBJ_DOOR:         200,  # cua nha - cao chuan, khong thay doi theo tu the
    "person":         165,  # chieu cao trung binh nguoi Viet Nam
    "motorcycle":     110,
    "bicycle":         95,
    "car":            145,
    "truck":          280,
    "bus":            310,
    OBJ_STOP_SIGN:    220,
    OBJ_FIRE_HYDRANT:  70,
    OBJ_TRAFFIC_LIGHT: 350,
}

# Vat the dat tren cao - thuong KHONG bi ngap -> weight thap trong synthesize
HIGH_MOUNT_OBJECTS = {OBJ_TRAFFIC_LIGHT, OBJ_STOP_SIGN}

# ─── Uu tien tham chieu (giam dan) ───────────────────────────────────────────
# 1. Cua nha (OBJ_DOOR):  co dinh, chieu cao chuan ~2m, khong phu thuoc tu the
# 2. Xe may (motorcycle):  kich thuoc co dinh, it bi anh huong tu the hon nguoi
# 3. Nguoi (person):       chieu cao on đinh nhung tu the thay doi → shoulder la fallback
#
# Cong thuc: scale = ref_height_cm / bbox_height_px
#            water_cm = water_pixel_height * scale
#                     = (water_px / ref_height_px) * ref_height_cm
OBJ_CONFIDENCE_WEIGHT: Dict[str, float] = {
    OBJ_DOOR:       0.95,   # CAO NHAT: cau truc co dinh, chieu cao chuan
    "person":       0.80,   # Tu the thay doi, shoulder est la fallback
    "motorcycle":   0.65,   # On dinh hon nguoi trong dieu kien binh thuong
    "bicycle":      0.55,   # Tuong tu motorcycle
    "car":          0.70,   # On đinh, it bi bias hon moto
    "truck":        0.60,
    "bus":          0.60,
    OBJ_FIRE_HYDRANT: 0.55,   # co định → kha tin cay nhung hiem
    OBJ_STOP_SIGN:    0.20,   # HIGH MOUNT → loai o synthesize
    OBJ_TRAFFIC_LIGHT:0.10,   # HIGH MOUNT → loai o synthesize
}

VEHICLE_WATER_BIAS: Dict[str, float] = {
    "motorcycle": 0.75,   # reduce 25% — hay bi overestimate
    "bicycle":    0.80,
    "car":        0.90,
    "truck":      0.85,
    "bus":        0.85,
    OBJ_DOOR:     1.00,   # khong bias — cua nha khong bi chim / nghieng
}

OBJ_COLORS_BGR: Dict[str, tuple] = {
    OBJ_DOOR:        (30,  100, 255),   # cam — de nhan biet
    "person":        (0, 230, 0),
    "motorcycle":    (0, 128, 255),
    "bicycle":       (0, 200, 255),
    "car":           (0, 165, 255),
    "truck":         (0, 0, 255),
    "bus":           (128, 0, 255),
    OBJ_STOP_SIGN:     (0, 0, 200),
    OBJ_FIRE_HYDRANT:  (0, 60, 200),
    OBJ_TRAFFIC_LIGHT: (0, 255, 255),
}

LEVEL_COLOR_BGR: Dict[str, tuple] = {
    "NO_FLOOD":  (0, 200, 0),
    "PUDDLE":    (0, 200, 0),
    "ANKLE":     (0, 210, 130),
    "KNEE":      (0, 165, 255),
    "WAIST":     (0, 80, 255),
    "CHEST":     (0, 0, 220),
    "SUBMERGED": (0, 0, 160),
    "UNKNOWN":   (120, 120, 120),
}

_WATER_HSV = [
    # Nuoc xanh duong (ao, song, nuoc lu trong)
    # S thap (28-120): nuoc that co saturation thap hon vai/ao mua (S>150)
    # → Loai ao mua xanh (H~100-115, S~150-255)
    (np.array([88,  28, 22]),  np.array([137, 120, 200])),  # xanh nuoc (S thap)

    # Nuoc bun/nau (lu mien Trung, đồng bang)
    (np.array([4,   42, 22]),  np.array([30,  200, 215])),  # nau/bun

    # Nuoc xam đục
    (np.array([0,    0, 42]),  np.array([180,  40, 188])),  # xam đục

    # Nuoc teal (nong can, pha xanh la)
    (np.array([76,  16, 28]),  np.array([102, 110, 218])),  # teal (S thap)

    # Ban đêm: nuoc phan chieu toi
    (np.array([0,    0,  8]),  np.array([180,  35, 80])),   # xam toi
    (np.array([85,  10, 10]),  np.array([140,  70, 110])),  # xanh xam toi

    # Nuoc bun vang nau (lu mua)
    (np.array([15,  40, 60]),  np.array([35,  185, 230])),  # vang nau
]

# Mau ao mua pho bien để LOAI khoi water mask
# (ao mua: saturation cao, mau đồng nhat, texture thap)
_RAINCOAT_HSV = [
    (np.array([88,  130, 60]),  np.array([137, 255, 255])),  # xanh duong đậm
    (np.array([0,   150, 80]),  np.array([10,  255, 255])),  # đo cam
    (np.array([170, 150, 80]),  np.array([180, 255, 255])),  # đo (wrap-around)
    (np.array([35,  130, 80]),  np.array([85,  255, 255])),  # xanh la, vang
]


def classify_level(cm: float) -> Tuple[str, str]:
    for lo, hi, lvl, desc in FLOOD_LEVELS:
        if lo <= cm < hi:
            return lvl, desc
    return "SUBMERGED", "Ngập hoàn toàn (>200cm)"


def _level_range(level: str) -> str:
    return {"NO_FLOOD": "0 cm", "PUDDLE": "0-15 cm", "ANKLE": "15-40 cm",
            "KNEE": "40-70 cm", "WAIST": "70-120 cm",
            "CHEST": "120-200 cm", "SUBMERGED": ">200 cm"}.get(level, "N/A")


def _iou(b1: list, b2: list) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    return inter / max((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter, 1)


def _water_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Tao binary mask cho vung nuoc.

    Cai tien:
    - Loai ao mua (saturation cao + texture thap = vai đồng nhat)
    - Nuoc that: saturation thap, co texture ripple
    - Ao mua: saturation cao (>130), mau đồng nhat, nam phia tren anh
    """
    hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # Buoc 1: Ap dung water HSV ranges
    for lo, hi in _WATER_HSV:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    # Buoc 2: Tao raincoat mask (ao mua = loai ra)
    raincoat_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _RAINCOAT_HSV:
        raincoat_mask = cv2.bitwise_or(raincoat_mask, cv2.inRange(hsv, lo, hi))

    # Buoc 3: Loai vung raincoat co texture thap (vai đồng nhat)
    # Nuoc that co texture (ripple), ao mua khong co texture
    gray     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap      = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs  = np.abs(lap).astype(np.float32)
    # Lam mo để lay texture local
    lap_blur = cv2.GaussianBlur(lap_abs, (15, 15), 0)
    # Vung texture thap (<15): kha nang cao la vai/be mat phang
    low_texture = (lap_blur < 15).astype(np.uint8)

    # Ao mua = raincoat_mask AND low_texture
    is_raincoat = cv2.bitwise_and(raincoat_mask, low_texture * 255)

    # Chi loai ao mua o NUA TREN anh (55% tren cung).
    # Phan duoi la vung nuoc lu — neu loai o day se mat water pixels
    # quan trong (nuoc gan nguoi mac ao mua co the co mau xanh phan chieu).
    h_img = hsv.shape[0]
    upper_zone = np.zeros(hsv.shape[:2], dtype=np.uint8)
    upper_zone[:int(h_img * 0.55), :] = 255
    is_raincoat = cv2.bitwise_and(is_raincoat, upper_zone)

    # Loai raincoat khoi water mask
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(is_raincoat))

    # Buoc 4: Morphology cleanup
    k    = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


@dataclass
class DetectedObject:
    class_name:         str
    bbox:               List[int]
    confidence:         float
    ref_height_cm:      float   # chiều cao THỰC của vật thể (KHÔNG đổi)
    pose_factor:        float   # hệ số tư thế (1.0=đứng, 0.55=ngồi, 0.5=xổm)
    water_height_cm:    float   # độ ngập tính từ local water line (cm)
    visibility_ratio:   float   # tỉ lệ phần nhìn thấy (0-1)
    pixels_total:       int
    pixels_above_water: int
    pixels_below_water: int
    local_wl_y:         int   = 0    # water line CỤC BỘ tại vị trí object (pixel y)
    local_wl_conf:      float = 0.0
    # Shoulder-based height estimation (chi co cho person)
    estimated_height_cm: float = 0.0   # chiều cao ước tính qua vai (0 = dùng default)
    shoulder_width_cm:   float = 0.0   # độ rộng vaii uoc tinh (cm)
    shoulder_method:     str   = ""    # "keypoint"|"neck"|"upper_body"|"default"
    shoulder_conf:       float = 0.0   # độ tin cậy shoulder estimate.0  # đo tin cay cua local water line
    body_part_contact:  str   = ""   # "ankle","knee","waist","chest","submerged","none"


@dataclass
class ReferenceFloodResult:
    filename:           str
    original_path:      str
    overlay_path:       str
    depth_map_path:     str
    flood_level:        str
    flood_level_desc:   str
    water_height_cm:    float
    water_height_range: str
    confidence:         float
    detected_objects:   List[dict] = field(default_factory=list)
    depth_flood_pct:    float = 0.0
    notes:              str = ""
    vehicles_detected:  List[dict] = field(default_factory=list)  # VehicleDetection.to_dict()
    # [v4] Scene quality + context — dùng bởi DepthCalibrator (context-aware)
    scene_score:        float = 0.5        # 0=bad, 1=good (từ SceneValidator)
    is_night:           bool  = False      # brightness < 60 → night calibration
    brightness:         float = 128.0      # avg brightness [0-255]


class ReferenceEstimator:
    """
    Do muc nuoc lu bang reference objects.

    Pipeline (analyze):
      1. Color water analysis  -> has_flood, water_mask
      2. DINOv2 classifier     -> flood_prob (optional)
      3. Depth Anything V2     -> depth_norm
      4. YOLO detect           -> raw_dets
      5. Pose filter           -> loai RIDING, luu pose_factor (KHONG sua ref_height)
      6. SegFormer             -> road/water hint
      7. Find water line       -> wl_y, wl_conf
      8. Measure objects       -> List[DetectedObject] dung pose_factor khi tinh water_cm
      9. Synthesize            -> water_cm cuoi cung
      10. Draw overlay         -> anh ket qua
    """

    def __init__(
        self,
        yolo_model:    str   = DEFAULT_YOLO_MODEL,
        depth_model:   str   = DEFAULT_DEPTH_MODEL,
        output_dir:    Optional[Path] = None,
        device:        str   = "auto",
        conf_thresh:   float = 0.35,
        use_dino:      bool  = True,
        dino_model:    str   = DEFAULT_DINO_MODEL,
        use_pose:      bool  = True,
        pose_model:    str   = DEFAULT_POSE_MODEL,
        use_segformer: bool  = True,
    ):
        self.yolo_model    = yolo_model
        self.depth_model   = depth_model
        self.output_dir    = Path(output_dir or "output/depth")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device        = device
        self.conf_thresh   = conf_thresh
        self.use_dino      = use_dino
        self.dino_model    = dino_model
        self.use_pose      = use_pose
        self.pose_model    = pose_model
        self.use_segformer = use_segformer
        self._yolo = self._depth_pipe = self._classifier = None
        self._pose_mdl = self._seg_model = self._seg_proc = None
        self._seg_dev  = "cpu"
        # Full pipeline config (gán bởi DepthStage._run_estimator: estimator._cfg = self.cfg)
        # Dùng cho:
        #   - models.use_sam → SAM2 mask hook trong _measure_objects_local
        #   - water_detection.refine_line → WaterDetector snap water line
        #   - models.use_sam / sam_model_type / sam_model_path → SAM segmentor
        # ⚠ Không nên dùng _cfg cho bất kỳ thứ gì khác — ưu tiên truyền
        #   qua constructor param nếu cần mở rộng.
        self._cfg: dict = {}

    def _load_yolo(self):
        if self._yolo: return
        from ultralytics import YOLO
        self._yolo = YOLO(self.yolo_model)

    def _load_depth(self):
        if self._depth_pipe: return
        import torch
        from transformers import pipeline as hf_pipeline
        dev = "cuda" if (self.device == "auto" and torch.cuda.is_available()) else "cpu"
        self._depth_pipe = hf_pipeline("depth-estimation", model=self.depth_model, device=dev)

    def _load_classifier(self):
        if self._classifier: return
        try:
            from depth_analysis.flood_classifier import FloodClassifier
            self._classifier = FloodClassifier(
                dino_model=self.dino_model, device=self.device,
                cfg=self._cfg,  # [v4] pass config → WaterDetector refine_line
            )
        except Exception as e:
            log.debug(f"  FloodClassifier skip: {e}")

    def _load_pose(self):
        if self._pose_mdl: return
        try:
            from depth_analysis.pose_analyzer import PoseAnalyzer
            self._pose_mdl = PoseAnalyzer(pose_model=self.pose_model, conf_thresh=self.conf_thresh)
        except Exception as e:
            log.debug(f"  PoseAnalyzer skip: {e}")

    def _load_segformer(self):
        if self._seg_model: return
        try:
            import torch
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            dev = "cuda" if (self.device == "auto" and torch.cuda.is_available()) else "cpu"
            mid = "nvidia/segformer-b0-finetuned-ade-512-512"
            self._seg_proc  = SegformerImageProcessor.from_pretrained(mid)
            self._seg_model = SegformerForSemanticSegmentation.from_pretrained(mid)
            # Keep the dynamically loaded Transformers model opaque to static
            # type checkers; some versions expose an incomplete ``to`` stub.
            seg_model = self._seg_model
            if seg_model is None:
                raise RuntimeError("SegFormer model failed to load")
            # Resolve dynamically so type checkers do not bind the incomplete
            # Transformers stub to the wrong callable signature.
            getattr(seg_model, "to")(torch.device(dev))
            self._seg_model.eval(); self._seg_dev = dev
        except Exception as e:
            log.debug(f"  SegFormer skip: {e}")

    # ─── Core ────────────────────────────────────────────────────────────────
    def analyze(self, image_path: Path) -> Optional[ReferenceFloodResult]:
        try:
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None: raise ValueError("Cannot read")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w    = img_bgr.shape[:2]
            # NOTE: img_bgr được giữ lại — cần dùng cho _draw_overlay và VehicleDetector bên dưới
            pil_img = Image.fromarray(img_rgb)
        except Exception as e:
            log.warning(f"  {image_path.name}: {e}"); return None

        # [IMPROVE] Wide-angle undistortion truoc khi analyze:
        # Doc EXIF de lay FOV, neu > 85° thi undistort
        from utils.constants import undistort_wide_angle
        fov_deg = _get_cam_analyzer()._estimate_fov(img_rgb, w, h)
        if fov_deg > 85.0:
            img_bgr = undistort_wide_angle(img_bgr, fov_deg)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            log.debug(f"  Undistort: FOV={fov_deg:.0f}°")

        # 1. Water color + surface water check
        wmask     = _water_mask(img_rgb)
        water_pct = float(wmask.sum()/255) / (h*w) * 100
        lower_pct = float(wmask[int(h*0.4):].sum()/255) / max(wmask[int(h*0.4):].size, 1) * 100

        # Kiem tra nuoc be mat thuc su (phan biet đường nhua uot vs nuoc lu)
        has_surface_water, surf_conf, color_water_lower = self._has_surface_water(
            img_rgb, wmask, h, w)

        # has_flood: can CA mau nuoc VA dau hieu be mat nuoc thuc
        has_flood = (lower_pct >= 5.0 or water_pct >= 8.0) and has_surface_water
        # Fallback: neu rat nhieu mau nuoc → co the thuc su co lu
        if not has_flood and (lower_pct >= 12.0 or water_pct >= 15.0):
            has_flood = True

        # ── [v4 Gap A] SceneValidator: kiểm tra chất lượng ảnh ────────────
        # Chạy SAU water mask (cần mask để validate), TRƯỚC YOLO/depth (đắt tiền).
        # Nếu ảnh quá mờ/tối → gắn scene_score thấp + flag vào result.
        scene_score_val, is_night_val, brightness_val = 0.5, False, 128.0
        try:
            from depth_analysis.scene_validator import SceneValidator
            _sv = SceneValidator()
            sv_res = _sv.validate(water_mask=wmask, img_bgr=img_bgr)
            scene_score_val = sv_res.scene_score
            brightness_val  = float(np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)))
            is_night_val    = brightness_val < 60.0
            if scene_score_val < 0.4:
                log.warning(
                    f"  [Scene] Quality LOW: score={scene_score_val:.2f} "
                    f"brightness={brightness_val:.0f} — measurements may be inaccurate"
                )
        except Exception as _sv_exc:
            log.debug(f"  [Scene] Validator skip: {_sv_exc}")

        # 2. YOLO (can truoc để check camera angle)
        self._load_yolo()
        raw_dets = self._detect_objects(img_rgb, h, w)

        # 3. Pose filter
        valid_dets = self._filter_riding(img_rgb, raw_dets)

        # 4. [IMPROVE] Phat hien goc nhin camera — UNIFIED
        cam_info  = _get_cam_analyzer().detect_camera_angle(img_rgb, valid_dets, h, w)
        is_aerial     = cam_info["is_aerial"]
        angle_factor  = cam_info["angle_factor"]

        # Low-angle unified
        is_low_angle   = cam_info["is_low_angle"]
        fov_deg        = cam_info.get("fov_deg", 65.0)
        camera_height_m = cam_info.get("camera_height_m", 1.5)

        # Phat hien nguoi tren thuyen → khong dung person measurements
        on_boat = self._detect_boat_context(img_rgb, valid_dets, h, w)
        if on_boat:
            # Loai tat ca person detections neu tren thuyen
            valid_dets = [d for d in valid_dets if d.get("class_name") != "person"]
            log.info("  Boat context: removed person detections")

        if is_aerial:
            has_flood = has_flood and (lower_pct >= 8.0 or water_pct >= 10.0)
            log.info(f"  Aerial view: angle_factor={angle_factor:.2f}")

        if is_low_angle:
            log.info(f"  Low-angle shot detected: unified")

        # 5. Kiem tra kho hoan toan
        # Neu TAT CA objects đều 100% visible + khong co nuoc be mat → NO_FLOOD
        if valid_dets and not has_surface_water:
            all_visible = all(
                (d["bbox"][3] - d["bbox"][1]) / h > 0.05  # object khong bi cat
                for d in valid_dets
            )
            # Sanity: neu water_pct rat thap va khong co surface water → NO_FLOOD
            if all_visible and color_water_lower < 0.03:
                has_flood = False
                log.info("  No surface water + all objects visible → NO_FLOOD")

        # 6. DINOv2
        flood_prob = 0.5
        if self.use_dino and has_flood:
            self._load_classifier()
            if self._classifier:
                try:
                    r = self._classifier.classify(image_path)
                    flood_prob = r.flood_prob
                    if not has_flood and r.has_flood: has_flood = True
                except Exception:
                    flood_prob = min(0.8, water_pct / 12.0)

        # 7. Depth map
        try:
            self._load_depth()
            depth_pipe = self._depth_pipe
            if depth_pipe is None:
                raise RuntimeError("Depth pipeline failed to load")
            # [FIX] Resize ảnh lớn trước khi đưa vào depth model để tránh RAM spike.
            # Model depth chạy ổn ở 1024px — kết quả sẽ được resize lại về (w,h) sau.
            _MAX_DEPTH_SIDE = 1024
            if max(h, w) > _MAX_DEPTH_SIDE:
                _scale = _MAX_DEPTH_SIDE / max(h, w)
                pil_for_depth = pil_img.resize(
                    (int(w * _scale), int(h * _scale)), Image.Resampling.LANCZOS
                )
            else:
                pil_for_depth = pil_img
            d_out      = depth_pipe(pil_for_depth)
            del pil_for_depth  # [FIX] giải phóng ngay sau khi model chạy xong
            depth_np   = np.array(d_out["depth"], dtype=np.float32)
            del d_out          # [FIX] giải phóng output của model
            dmin, dmax = depth_np.min(), depth_np.max()
            depth_norm = (depth_np - dmin) / max(dmax - dmin, 1e-6)
            del depth_np       # [FIX] giải phóng mảng float32 gốc
            depth_r    = cv2.resize(depth_norm, (w, h))
            del depth_norm     # [FIX] giải phóng mảng trung gian
        except Exception as e:
            log.warning(f"  Depth fail: {e}"); return None

        # 8. SegFormer
        seg_wl_y, seg_road_pct, _ = self._run_segformer(img_rgb, h, w)

        # 8b. Phat hien cua nha (reference uu tien cao nhat) — khong can YOLO
        try:
            from depth_analysis.infrastructure_detector import InfrastructureDetector
            door_dets = InfrastructureDetector(conf_thresh=self.conf_thresh).detect_doors(
                img_rgb, h, w
            )
            for dd in door_dets:
                valid_dets.insert(0, {   # chen vao dau — xu ly truoc
                    "class_name":    OBJ_DOOR,
                    "bbox":          dd.bbox,
                    "confidence":    dd.confidence * OBJ_CONFIDENCE_WEIGHT[OBJ_DOOR],
                    "ref_height_cm": REF_HEIGHTS[OBJ_DOOR],
                    "pose_factor":   1.0,   # cua khong co tu the
                    "color_bgr":     OBJ_COLORS_BGR[OBJ_DOOR],
                    "is_high_mount": False,
                    "skip_measure":  False,
                })
            if door_dets:
                log.info(f"  Door reference: {len(door_dets)} cua nha detected")
        except Exception as _de:
            log.debug(f"  Door detect skip: {_de}")

        # 9. Global water line (dung lam fallback)
        wl_y, wl_conf = self._find_water_line(
            img_rgb, wmask, valid_dets, h, w, has_flood, lower_pct, seg_wl_y, seg_road_pct)

        # 10. đo tung object voi LOCAL water line (quan trong nhat)
        measured = self._measure_objects_local(
            valid_dets, wmask, wl_y, wl_conf, h, w,
            img_rgb=img_rgb,
            is_aerial=is_aerial, angle_factor=angle_factor,
            is_low_angle=is_low_angle)

        # 11. Synthesize — [IMPROVE] them fov_deg
        water_cm, conf, level, desc = self._synthesize(
            measured, wl_y, wl_conf, water_pct, lower_pct,
            has_flood, depth_r, h, flood_prob,
            is_aerial=is_aerial, angle_factor=angle_factor,
            fov_deg=fov_deg)

        # ── [v4 Gap A] Scene quality penalty ──────────────────────────────
        # Ảnh mờ/tối → confidence bị phạt thêm (không thay đổi water_cm,
        # vì con số đo vẫn có giá trị tham khảo).
        if scene_score_val < 0.4:
            conf *= 0.70   # phạt nặng khi scene rất xấu
        elif scene_score_val < 0.6:
            conf *= 0.88   # phạt nhẹ khi scene hơi xấu

        # 12. Vehicle detection (ô tô + xe máy, bỏ qua xe đạp)
        vehicles = []
        try:
            from depth_analysis.vehicle_detector import VehicleDetector
            _vd = VehicleDetector(conf_thresh=self.conf_thresh)
            vehicles = _vd.detect(
                img_rgb      = img_bgr[:, :, ::-1],
                yolo_model   = self._yolo,
                water_line_y = wl_y if wl_y < img_bgr.shape[0] else None,
                img_h        = img_bgr.shape[0],
                img_w        = img_bgr.shape[1],
            )
            _vs = VehicleDetector.summarize(vehicles)
            if _vs["total"]:
                log.info(f"  [Vehicle] {_vs['total']} xe: {_vs['by_type']} | ngập: {_vs['by_status']}")
        except Exception as _ve:
            log.debug(f"  [Vehicle] Bỏ qua: {_ve}")
            vehicles = []

        # 13. Draw
        ov_path    = self._draw_overlay(img_bgr, valid_dets, measured, wl_y, water_cm, level, image_path, vehicles=vehicles)
        depth_path = self._save_depth_colormap(depth_r, image_path)

        notes = (f"wl={wl_y}(conf={wl_conf:.2f}) water={water_pct:.1f}% "
                 f"lower={lower_pct:.1f}% aerial={is_aerial} surf={has_surface_water}")
        result = ReferenceFloodResult(
            filename=image_path.name, original_path=str(image_path),
            overlay_path=str(ov_path), depth_map_path=str(depth_path),
            flood_level=level, flood_level_desc=desc,
            water_height_cm=round(water_cm, 1), water_height_range=_level_range(level),
            confidence=round(conf, 3), detected_objects=[asdict(o) for o in measured],
            depth_flood_pct=round(lower_pct, 1),
            notes=notes,
            vehicles_detected=[v.to_dict() for v in vehicles],
            scene_score=round(scene_score_val, 3),
            is_night=is_night_val,
            brightness=round(brightness_val, 1),
        )
        log.info(f"  [{image_path.name}] {level} {water_cm:.0f}cm conf={conf:.2f} "
                 f"aerial={is_aerial}")
        return result

    def analyze_batch(self, paths: list, chunk_size: int = 8,
                      stop_check=None) -> list:
        """
        Phân tích theo lô nhỏ (chunk) để tiết kiệm RAM.

        Mỗi chunk_size ảnh:
          1. Chạy analyze() từng ảnh
          2. Giải phóng cache numpy/PIL
          3. Empty CUDA cache nếu dùng GPU
          4. gc.collect() để trả RAM về OS

        chunk_size=8 phù hợp với RAM 8GB. Giảm xuống 4 nếu ít RAM hơn.

        stop_check: callable() → bool, trả về True khi cần dừng sớm.
                    Được kiểm tra đầu mỗi chunk và sau mỗi ảnh.
        """
        import gc
        out        = []
        total      = len(paths)
        n_chunks   = (total + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            # [FIX] Kiểm tra stop_check đầu mỗi chunk
            if stop_check and stop_check():
                log.warning("  analyze_batch: nhận tín hiệu dừng, thoát sớm.")
                break

            start = chunk_idx * chunk_size
            end   = min(start + chunk_size, total)
            chunk = paths[start:end]

            log.info(f"  Chunk {chunk_idx+1}/{n_chunks} "
                     f"[{start+1}-{end}/{total}]")

            for i, p in enumerate(chunk, start+1):
                # [FIX] Kiểm tra stop_check trước mỗi ảnh
                if stop_check and stop_check():
                    log.warning(f"  analyze_batch: dừng tại ảnh {i}/{total}.")
                    return out

                log.info(f"  [{i}/{total}] {Path(p).name}")
                try:
                    r = self.analyze(Path(p))
                    if r:
                        out.append(r)
                except Exception as e:
                    # ── [v4 #7] OOM handling: free memory + log chi tiết ──
                    _is_oom = isinstance(e, (MemoryError,) ) or "out of memory" in str(e).lower()
                    if _is_oom:
                        log.error(
                            f"  OOM {Path(p).name}: GPU RAM hết. "
                            f"Giảm chunk_size hoặc giảm kích thước ảnh."
                        )
                        # Free GPU memory ngay lập tức
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass
                        gc.collect()
                    else:
                        log.error(f"  FAILED {Path(p).name}: {e}")

            # ── Giải phóng RAM sau mỗi chunk ──────────────────────────
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            # Giải phóng PIL cache
            try:
                Image.core.clear_cache()
            except Exception:
                pass

            log.info(f"  Chunk {chunk_idx+1} done: {len(out)} results total, RAM freed")

        return out

    def _analyze_chunk_oom_safe(self, chunk, start_idx, total, stop_check=None, max_retries=2):
        """
        [v4 #7] Phân tích 1 chunk với OOM retry: nếu OOM xảy ra → giảm batch
        size (tách chunk thành nửa) và thử lại. Dùng recursion depth limit.
        """
        import gc
        results = []
        current_chunk = list(chunk)

        for attempt in range(max_retries + 1):
            try:
                for i, p in enumerate(current_chunk, start_idx + 1):
                    if stop_check and stop_check():
                        return results
                    log.info(f"  [{i}/{total}] {Path(p).name}")
                    r = self.analyze(Path(p))
                    if r:
                        results.append(r)
                return results  # success
            except Exception as e:
                _is_oom = isinstance(e, MemoryError) or "out of memory" in str(e).lower()
                if not _is_oom or attempt >= max_retries:
                    raise
                # OOM → free memory + split chunk in half
                log.warning(f"  OOM on chunk (attempt {attempt+1}) — retrying with smaller chunks")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                gc.collect()
                mid = len(current_chunk) // 2
                if mid == 0:
                    raise  # single image OOM → give up
                first_half = current_chunk[:mid]
                second_half = current_chunk[mid:]
                r1 = self._analyze_chunk_oom_safe(first_half, start_idx, total, stop_check, max_retries - attempt - 1)
                r2 = self._analyze_chunk_oom_safe(second_half, start_idx + mid, total, stop_check, max_retries - attempt - 1)
                return r1 + r2
        return results

    def unload_heavy_models(self):
        """
        Giải phóng các model nặng (Depth, SegFormer, DINOv2) khỏi RAM/VRAM.
        Gọi sau khi analyze_batch() xong nếu cần làm việc khác.
        YOLO và Pose vẫn giữ vì nhỏ và load nhanh.
        """
        import gc
        for attr in ("_depth_pipe", "_classifier", "_seg_model", "_seg_proc"):
            if getattr(self, attr, None) is not None:
                try:
                    m = getattr(self, attr)
                    if hasattr(m, "to"):
                        m.to("cpu")
                    del m
                except Exception:
                    pass
                setattr(self, attr, None)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("  Heavy models unloaded (Depth/SegFormer/DINOv2)")


    # ─── Phat hien goc nhin (aerial/bird's eye view) ────────────────────
    def _detect_camera_angle(self, img_rgb: np.ndarray, dets: list,
                              h: int, w: int) -> dict:
        """
        Phat hien goc nhin cua camera để điều chinh tinh toan.

        Aerial (tu tren cao):
          - Objects nho so voi anh (chieu cao < 20% frame)
          - Tam objects nam giua anh doc
          - đường vach ke đường nhin thay goc nghieng

        Eye-level (ngang mat):
          - Objects lon, horizon ro rang o ~50% chieu cao anh

        Tra ve: {
            "is_aerial":      bool,
            "angle_factor":   float,  # 1.0=ngang mat, 0.0=thang đúng
            "confidence":     float,
        }
        """
        result = {"is_aerial": False, "angle_factor": 1.0, "confidence": 0.5}

        if not dets:
            return result

        # Signal 1: Ti le chieu cao trung binh cua objects / frame
        obj_heights  = [d["bbox"][3] - d["bbox"][1] for d in dets]
        avg_h_ratio  = float(np.mean(obj_heights)) / h

        # Aerial: objects nho (< 20% chieu cao frame)
        # Eye-level: objects lon (> 35%)
        if avg_h_ratio < 0.18:
            aerial_score = 0.8
        elif avg_h_ratio < 0.25:
            aerial_score = 0.5
        else:
            aerial_score = 0.1

        # Signal 2: Vi tri Y trung tam cua objects
        centers_y = [(d["bbox"][1] + d["bbox"][3]) / 2 / h for d in dets]
        avg_cy    = float(np.mean(centers_y))

        # Eye-level: objects tap trung o 40-80% chieu cao (horizon o giua)
        # Aerial: objects phan tan hon, thuong o 30-70%
        if 0.35 < avg_cy < 0.65 and aerial_score > 0.4:
            aerial_score = min(1.0, aerial_score + 0.2)

        # Signal 3: Kiem tra vanishing point
        # Eye-level: vanishing point o đường horizon ro rang
        gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                                minLineLength=w//6, maxLineGap=20)
        if lines is not None:
            # Tinh goc nghieng trung binh cua cac đường
            angles = []
            for l in lines[:30]:
                x1,y1,x2,y2 = l[0]
                if abs(x2-x1) > 10:
                    angles.append(abs(np.degrees(np.arctan2(y2-y1, x2-x1))))
            if angles:
                # Aerial: nhieu đường ngang (goc ~0°, ~180°)
                near_horizontal = sum(1 for a in angles if a < 20 or a > 160)
                h_ratio = near_horizontal / len(angles)
                if h_ratio > 0.6:
                    aerial_score = min(1.0, aerial_score + 0.15)

        is_aerial    = aerial_score > 0.55
        angle_factor = max(0.2, 1.0 - aerial_score * 0.8) if is_aerial else 1.0

        log.debug(f"  Camera: aerial={is_aerial} score={aerial_score:.2f} "
                  f"avg_h={avg_h_ratio:.2f} cy={avg_cy:.2f}")

        return {
            "is_aerial":    is_aerial,
            "angle_factor": round(angle_factor, 2),
            "confidence":   round(min(1.0, aerial_score), 2),
        }

    # ─── đo water line CUC BO tai vi tri tung object ───────────────────
    def _local_water_line(self, wmask: np.ndarray, det: dict,
                          h: int, w: int) -> tuple:
        """
        Tim water line CUC BO tai vi tri cua object.

        FIX BUG 16% visibility:
        Bug cu: tim nuoc tu y1 (dinh bbox) -> lay nuoc background phia tren
               -> local_wl_y qua cao -> visibility 16% -> water_cm sai
        Fix: chi tim nuoc trong vung DUOI cua bbox (50%-120% tu dinh)
             + confirm bang water density tai vung chan

        Logic:
          1. Kiem tra nuoc tai vung chan (y2-30 -> y2+20): co water contact khongđ
          2. Neu co: tim ranh gioi nuoc-kho tu duoi len trong vung 40-100% bbox
          3. Neu khong: tra ve y2 (khong ngap)
        """
        x1, y1, x2, y2 = det["bbox"]
        bh = max(y2 - y1, 1)
        bw = max(x2 - x1, 1)

        # Mo rong sang 2 ben de bat nuoc canh doi tuong
        margin = max(8, bw // 6)
        cx1    = max(0, x1 - margin)
        cx2    = min(w, x2 + margin)

        # ── Buoc 1: Kiem tra co nuoc tai vung chan khong ──────────────
        # Vung chan = 20px tren + 20px duoi y2
        foot_y1 = max(0, y2 - int(bh * 0.15))
        foot_y2 = min(h, y2 + 20)
        foot_mask   = wmask[foot_y1:foot_y2, cx1:cx2]
        if foot_mask.size == 0:
            return y2, 0.0

        foot_density = float(foot_mask.sum() / 255) / max(foot_mask.size, 1)

        if foot_density < 0.03:
            # Khong co nuoc tai chan -> object KHONG bi ngap
            # Kiem tra them: co nuoc trong vung rong hon khong (cho xe may etc)
            wider_mask = wmask[max(0, y2 - int(bh*0.3)):foot_y2,
                               max(0, cx1 - margin):min(w, cx2 + margin)]
            wider_density = float(wider_mask.sum()/255)/max(wider_mask.size,1) if wider_mask.size > 0 else 0
            if wider_density < 0.02:
                # Fallback dac biet cho nguoi mac ao mua: ao mua che mau nuoc xung quanh
                # → mo rong pham vi kiem tra sang 2 ben va xuong duoi nhieu hon
                if det.get("class_name") == "person" and det.get("rain_coat_prob", 0) > 0.3:
                    wide2_mask = wmask[
                        max(0, y2 - int(bh * 0.5)):min(h, y2 + 40),
                        max(0, cx1 - margin * 2):min(w, cx2 + margin * 2)
                    ]
                    wide2_density = (
                        float(wide2_mask.sum() / 255) / max(wide2_mask.size, 1)
                        if wide2_mask.size > 0 else 0
                    )
                    if wide2_density >= 0.015:
                        foot_density = wide2_density  # tiep tuc do
                    else:
                        return y2, 0.05
                else:
                    return y2, 0.05   # khong ngap

        # ── Buoc 2: Tim ranh gioi nuoc - kho trong vung duoi bbox ────
        # Chi tim trong vung 40% duoi den 120% y2 (tranh lay nuoc background tren)
        search_top = max(0, y1 + int(bh * 0.40))   # bat dau tu 40% chieu cao bbox
        search_bot = min(h, y2 + 15)
        search_mask = wmask[search_top:search_bot, cx1:cx2]

        if search_mask.size == 0:
            return y2, 0.0

        rows_with_water = np.where(search_mask.any(axis=1))[0]
        if len(rows_with_water) == 0:
            # Khong co nuoc trong vung duoi -> tinh theo global
            return y2, 0.10

        # Lay percentile 20 cua cac hang co nuoc (loai outlier)
        # Cong vao search_top de ra toa do anh tuyet doi
        water_top_rel  = int(np.percentile(rows_with_water, 20))
        local_wl_y     = search_top + water_top_rel

        # ── Buoc 3: Tinh confidence ───────────────────────────────────
        # Confidence cao khi: mat do nuoc cao + water line gan chan (hop le)
        local_conf = min(0.88, foot_density * 6.0 + 0.25)

        # Penalty: neu local_wl_y < 50% chieu cao bbox -> co the lay nuoc sai
        wl_relative = (local_wl_y - y1) / bh
        if wl_relative < 0.45:
            # Water line o qua cao (tren giua bbox) -> giam confidence
            local_conf *= 0.50
            # Day water_line xuong vi tri hop ly hon (60% chieu cao bbox)
            local_wl_y = y1 + int(bh * 0.60)
            log.debug(f"  LocalWL [{det['class_name']}]: WL too high ({wl_relative:.0%}), "
                      f"clamped to 60% bbox")

        # Clamp: water line phai nam trong [y1+20%, y2]
        local_wl_y = max(y1 + int(bh*0.20), min(y2, local_wl_y))

        log.debug(f"  LocalWL [{det['class_name']}]: y={local_wl_y} "
                  f"rel={wl_relative:.0%} foot_dens={foot_density:.2f} conf={local_conf:.2f}")
        return local_wl_y, round(local_conf, 2)

    # ─── Nhan dien bo phan co the tiep xuc nuoc ────────────────────────
    BODY_PART_RATIO = {
        "ankle": 0.20,
        "knee":  0.35,
        "waist": 0.55,
        "chest": 0.75,
    }

    def _classify_body_contact(self, water_cm: float, ref_h: float) -> str:
        """
        Phan loai nuoc đang tiep xuc bo phan nao cua co the.

        Ti le chieu cao co the (% tu duoi len):
          Mat ca  : 10-20%  → ~17-34cm  voi nguoi 170cm
          đầu goi : 25-35%  → ~42-60cm
          Hong    : 45-55%  → ~77-94cm
          Nguc    : 60-75%  → ~102-128cm
          Co/đầu : >80%    → >136cm
        """
        if ref_h <= 0 or water_cm <= 0:
            return "none"
        ratio = water_cm / ref_h
        if ratio < 0.05:
            return "none"
        if ratio < self.BODY_PART_RATIO["ankle"]:
            return "ankle"
        if ratio < self.BODY_PART_RATIO["knee"]:
            return "knee"
        if ratio < self.BODY_PART_RATIO["waist"]:
            return "waist"
        if ratio < self.BODY_PART_RATIO["chest"]:
            return "chest"
        return "submerged"

    def _classify_body_contact_from_keypoints(self, det: dict, wl_y: int) -> Optional[str]:
        """Ước tính vị trí tiếp xúc nước từ keypoints (ankle/knee/hip/shoulder)."""
        kpts = det.get("keypoints")
        if kpts is None or not isinstance(kpts, np.ndarray):
            return None

        def get_y(idx):
            if idx < 0 or idx >= len(kpts):
                return None
            ky = float(kpts[idx][1]); kc = float(kpts[idx][2])
            return ky if kc >= 0.25 else None

        l_ankle = get_y(15)
        r_ankle = get_y(16)
        l_knee  = get_y(13)
        r_knee  = get_y(14)
        l_hip   = get_y(11)
        r_hip   = get_y(12)
        l_shoul = get_y(5)
        r_shoul = get_y(6)

        ankle_sub = any(y is not None and y > wl_y for y in (l_ankle, r_ankle) if y is not None)
        knee_sub  = any(y is not None and y > wl_y for y in (l_knee, r_knee) if y is not None)
        hip_sub   = any(y is not None and y > wl_y for y in (l_hip, r_hip) if y is not None)
        sh_sub    = any(y is not None and y > wl_y for y in (l_shoul, r_shoul) if y is not None)

        if ankle_sub and not knee_sub:
            return "ankle"
        if knee_sub and not hip_sub:
            return "knee"
        if hip_sub and not sh_sub:
            return "waist"
        if sh_sub:
            return "chest"
        return None

    # ─── Kiem tra nuoc be mat (surface water texture) ───────────────────
    def _has_surface_water(self, img_rgb: np.ndarray, wmask: np.ndarray,
                            h: int, w: int) -> tuple:
        """
        Kiem tra co nuoc be mat thuc su khong (ripples, reflections).
        Phan biet: đường nhua xam uot vs nuoc lu thuc su.

        Nuoc thuc su co:
          1. Vung smooth thap variance (mat nuoc phang)
          2. Reflections (mirror-like regions)
          3. Ripple texture (wavelet pattern)

        đường nhua co:
          1. Texture đều (asphalt grain)
          2. Khong co vung phan chieu
          3. Mau xam đồng nhat

        Tra ve (has_water: bool, confidence: float, water_ratio: float)
        """
        gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lower = gray[int(h * 0.55):, :]    # phan duoi anh

        if lower.size == 0:
            return False, 0.0, 0.0

        # --- Test 1: Variance phan vung (smooth patches = mat nuoc) ---
        ph, pw    = 16, 16
        lh, lw    = lower.shape
        variances = []
        for i in range(0, lh - ph, ph):
            for j in range(0, lw - pw, pw):
                patch = lower[i:i+ph, j:j+pw]
                variances.append(float(patch.var()))

        if not variances:
            return False, 0.0, 0.0

        var_arr      = np.array(variances)
        smooth_ratio = float((var_arr < 80).mean())
        var_cv       = var_arr.std() / max(var_arr.mean(), 1)

        # --- Test 2: Horizontal edge (ranh gioi nuoc-khong khi) ---
        grad_y           = cv2.Sobel(lower, cv2.CV_64F, 0, 1, ksize=3)
        strong_h_edges   = float((np.abs(grad_y) > 30).mean())

        # --- Test 3: Color water mask ---
        lower_wmask      = wmask[int(h * 0.55):, :]
        color_water_pct  = float(lower_wmask.sum() / 255) / max(lower_wmask.size, 1)

        # --- Test 4: Reflection pattern ---
        # Nuoc co vung sang bat thuong (specular reflection)
        bright_lower     = (lower > 200).astype(np.float32)
        bright_ratio     = float(bright_lower.mean())
        # đường nhua: it vung cuc sang, nuoc: co ripple reflection
        has_reflection   = bright_ratio > 0.03 and bright_ratio < 0.40

        # --- Test 5: Texture uniformity trong vung co mau nuoc ---
        # đường nhua xam: texture asphalt đều, bien thien nho
        # Nuoc: texture ripple, bien thien nhieu hon trong vung xam
        asphalt_like     = (var_cv < 0.5) and (smooth_ratio < 0.25)
        # Neu texture đường nhua đien hinh → penalty
        asphalt_penalty  = 0.3 if asphalt_like else 0.0

        # --- Test 6: Color saturation of lower region ---
        lower_bgr  = img_rgb[int(h * 0.55):, :]
        lower_hsv  = cv2.cvtColor(lower_bgr, cv2.COLOR_RGB2HSV)
        sat_lower  = float(lower_hsv[:,:,1].mean())
        # đường nhua: saturation thap (~15-30)
        # Nuoc lu (bun nau): saturation cao hon (~40-80)
        # Nuoc sach: saturation thap nhung co reflection
        is_asphalt_color = sat_lower < 25 and color_water_pct < 0.08

        # --- Tong hop ---
        water_score = 0.0
        if smooth_ratio > 0.40:       water_score += 0.25
        if smooth_ratio > 0.60:       water_score += 0.15
        if color_water_pct > 0.08:    water_score += 0.25
        if color_water_pct > 0.20:    water_score += 0.15
        if has_reflection:             water_score += 0.15
        if var_cv > 1.2:              water_score += 0.10
        if strong_h_edges > 0.06:     water_score += 0.10

        # Penalties
        water_score -= asphalt_penalty
        if is_asphalt_color:
            water_score -= 0.20   # mau đường nhua → giam manh

        has_water  = water_score > 0.45
        confidence = min(1.0, max(0.0, water_score))

        log.debug(f"  SurfaceWater: score={water_score:.2f} smooth={smooth_ratio:.2f} "
                  f"color_pct={color_water_pct:.2f} asphalt={is_asphalt_color}")
        return has_water, round(confidence, 2), round(color_water_pct, 3)


    def _detect_low_angle_shot(self, img_rgb: np.ndarray,
                                h: int, w: int) -> dict:
        """
        Phat hien anh chup goc thap (lens gan mat nuoc).
        đac điểm:
          - Horizon rat thap (20-40% tu duoi len) hoac khong thay horizon
          - Nuoc chiem phan lon 50% duoi anh
          - Gradient ngang manh o 40-60% chieu cao = mat nuoc gan
        """
        gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # Chia anh thanh 10 dai ngang
        dai_h  = h // 10
        row_vars = []
        for i in range(10):
            dai  = gray[i*dai_h:(i+1)*dai_h, :]
            row_vars.append(float(dai.var()))

        # Low-angle: variance thap o dai duoi (mat nuoc phang)
        # va variance cao o dai giua/tren (xe/nguoi)
        bottom3 = np.mean(row_vars[7:])   # 3 dai duoi
        top4    = np.mean(row_vars[:4])    # 4 dai tren
        mid3    = np.mean(row_vars[3:7])   # giua

        # Low-angle: bottom variance THAP HON mid nhieu (nuoc gan lens = blur/flat)
        low_angle_score = 0.0
        if bottom3 < mid3 * 0.4:
            low_angle_score += 0.4
        if bottom3 < top4 * 0.3:
            low_angle_score += 0.3

        # Kiem tra them: gradient ngang manh o khoang 50-70% chieu cao
        grad_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        horizon_zone = grad_y[int(h*0.30):int(h*0.60), :]
        horizon_score = float(horizon_zone.mean())
        full_score    = float(grad_y.mean())
        if horizon_score > full_score * 1.5:
            low_angle_score += 0.3

        is_low_angle = low_angle_score > 0.55
        log.debug(f"  LowAngle: score={low_angle_score:.2f} bottom_var={bottom3:.1f} mid={mid3:.1f}")
        return {
            "is_low_angle": is_low_angle,
            "score":        round(low_angle_score, 2),
        }

    def _detect_boat_context(self, img_rgb: np.ndarray,
                              dets: list, h: int, w: int) -> bool:
        """
        Phat hien nguoi đang o TREN THUYEN/BE cuu ho.
        Neu đúng → KHONG dung person measurements vi nguoi khong tiep xuc nuoc.

        Dau hieu:
          1. Mau đo/cam đặc trưng cua thuyen cuu ho
          2. Nguoi co visibility thap bat thuong (10-20%) nhung xep canh nhau
          3. Ranh gioi ngang thang o đây nhom nguoi (be thuyen)
        """
        # Kiem tra mau thuyen cuu ho (đo tuoi / cam)
        hsv        = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        # đo: hue <10 hoac >170, sat >150, val >100
        red_mask   = (((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170)) &
                      (hsv[:,:,1] > 150) & (hsv[:,:,2] > 80))
        # Cam: hue 8-20, sat >150
        orange_mask= ((hsv[:,:,0] > 8) & (hsv[:,:,0] < 20) &
                      (hsv[:,:,1] > 150) & (hsv[:,:,2] > 100))

        rescue_color = (red_mask | orange_mask).astype(np.float32)

        # Ti le mau cuu ho trong nua duoi anh
        lower_rescue = rescue_color[h//2:, :].mean()

        # Kiem tra nhom nguoi xep canh nhau o cung muc y (tren thuyen)
        persons = [d for d in dets if d.get("class_name") == "person"]
        on_boat = False

        if lower_rescue > 0.05 and len(persons) >= 2:
            # Nhieu nguoi co đây bbox gan nhau (±10% chieu cao)
            bot_ys   = [d["bbox"][3] for d in persons]
            if len(bot_ys) >= 2:
                y_range  = max(bot_ys) - min(bot_ys)
                avg_bh   = np.mean([d["bbox"][3]-d["bbox"][1] for d in persons])
                if y_range < avg_bh * 0.3:   # đây bbox ngang hang nhau
                    on_boat = True
                    log.info(f"  Boat context detected: rescue_color={lower_rescue:.2f}")

        # Fallback: neu khong co mau cuu ho nhung nguoi xep hang ngang
        if not on_boat and len(persons) >= 3:
            bot_ys = [d["bbox"][3] for d in persons]
            avg_bh = np.mean([d["bbox"][3]-d["bbox"][1] for d in persons])
            y_range = max(bot_ys) - min(bot_ys)
            if y_range < avg_bh * 0.25:
                # Kiem tra them: co nuoc ben duoi nhom nguoi
                max_bot = max(bot_ys)
                zone_below = img_rgb[max_bot:min(h, max_bot+30), :]
                if zone_below.size > 0:
                    # Nuoc thuong co mau xanh/nau uniform
                    zone_hsv = cv2.cvtColor(zone_below, cv2.COLOR_RGB2HSV)
                    zone_sat_std = zone_hsv[:,:,1].std()
                    if zone_sat_std < 40:   # mau kha đồng nhat = co the la nuoc
                        on_boat = True
                        log.debug(f"  Boat context (row alignment): y_range={y_range:.0f} avg_bh={avg_bh:.0f}")

        return on_boat

    # ─── Steps ───────────────────────────────────────────────────────────────
    def _detect_objects(self, img_rgb, h, w):
        dets = []
        try:
            yolo = self._yolo
            if yolo is None:
                log.debug("  YOLO unavailable; skipping object detection")
                return dets
            results = yolo(img_rgb, conf=self.conf_thresh, verbose=False)
            # Some YOLO wrappers/type stubs may return None or raw tensors
            # instead of Ultralytics Results objects.  Ignore those safely.
            if results is None:
                return dets
            for res in results:
                boxes = getattr(res, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    # Results may be tensor-like objects without a ``names``
                    # attribute; class names are also available on the model.
                    names = getattr(res, "names", None)
                    if names is None:
                        names = getattr(yolo, "names", None)
                    if names is None:
                        names = getattr(getattr(yolo, "model", None), "names", None)
                    if names is None:
                        continue
                    class_id = int(box.cls[0])
                    name = names.get(class_id) if isinstance(names, dict) else names[class_id]
                    if name not in REF_HEIGHTS: continue
                    conf = float(box.conf[0])
                    x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
                    bh = y2 - y1
                    bw_det = x2 - x1
                    if bh < h*0.03 or bh > h*0.93: continue
                    # FIX: Bo qua objects qua nho (co the la thumbnail trong banner)
                    # Object nho hon 4% dien tich anh = co the la anh trong anh
                    obj_area_ratio = (bh * bw_det) / max(h * w, 1)
                    if obj_area_ratio < 0.004 and len(dets) >= 1:
                        # Neu da co object lon hon, bo qua object nho
                        log.debug(f"  Skip tiny object: {name} area={obj_area_ratio:.3f}")
                        continue
                    # Giam confidence cho vat the dat tren cao (traffic light...)
                    # Loai HOAN TOAN traffic light va stop sign
                    # (lap tren cot cao, khong bao gio bi ngap thuc su)
                    # Chi giu lai để ve bbox tham khao, khong dung đo muc nuoc
                    if name in HIGH_MOUNT_OBJECTS:
                        dets.append({
                            "class_name":    name,
                            "bbox":          [x1, y1, x2, y2],
                            "confidence":    conf * 0.05,  # weight cuc thap
                            "ref_height_cm": REF_HEIGHTS[name],
                            "pose_factor":   1.0,
                            "color_bgr":     OBJ_COLORS_BGR.get(name, (128,128,128)),
                            "is_high_mount": True,
                            "skip_measure":  True,  # flag: khong dung đo nuoc
                        })
                        continue

                    obj_weight = OBJ_CONFIDENCE_WEIGHT.get(name, 0.80)

                    # Kiem tra ao mua cho person: phan tich mau o 60% tren cua bbox
                    # (ao mua phu khat tu dau den hong/eo → chi ket tra phan tren)
                    rain_coat_prob = 0.0
                    if name == "person":
                        try:
                            uc_y2 = min(h, y1 + int(bh * 0.60))
                            crop  = img_rgb[y1:uc_y2, x1:x2]
                            if crop.size > 200:
                                hsv_c  = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
                                rc_m   = np.zeros(hsv_c.shape[:2], dtype=np.uint8)
                                for _lo, _hi in _RAINCOAT_HSV:
                                    rc_m = cv2.bitwise_or(rc_m, cv2.inRange(hsv_c, _lo, _hi))
                                # Ao mua: phu >25% dien tich phan tren = co ao mua
                                rain_coat_prob = min(
                                    1.0,
                                    float(rc_m.sum() / 255) / max(rc_m.size * 0.25, 1)
                                )
                        except Exception:
                            rain_coat_prob = 0.0

                    dets.append({"class_name": name, "bbox": [x1,y1,x2,y2],
                                 "confidence": conf * obj_weight,
                                 "ref_height_cm": REF_HEIGHTS[name],
                                 "pose_factor": 1.0,
                                 "color_bgr": OBJ_COLORS_BGR.get(name, (128,128,128)),
                                 "is_high_mount": False,
                                 "skip_measure":  False,
                                 "rain_coat_prob": round(rain_coat_prob, 3)})
        except Exception as e:
            log.warning(f"  YOLO: {e}")
        return dets

    def _filter_riding(self, img_rgb, dets):
        """
        Loai RIDING person; giu lai non-riding nhung cap nhat pose_factor.
        QUAN TRONG: KHONG sua ref_height_cm - chi luu pose_factor.
        pose_factor duoc dung trong _measure_objects de tinh water_cm chinh xac.
        """
        vehicle_bboxes = [d["bbox"] for d in dets if d["class_name"] in ("motorcycle","bicycle")]
        out = []
        poses = []
        if self.use_pose:
            try:
                self._load_pose()
                if self._pose_mdl:
                    poses = self._pose_mdl.analyze_image(img_rgb)
            except Exception:
                poses = []

        for det in dets:
            if det["class_name"] == "person":
                if any(_iou(det["bbox"], vb) > 0.25 for vb in vehicle_bboxes):
                    log.debug(f"  Skip RIDING (IoU) {det['bbox']}")
                    continue

                if poses:
                    best_pose = None
                    best_iou = 0.0
                    for pose in poses:
                        iou_val = _iou(list(pose.bbox), det["bbox"])
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_pose = pose

                    if best_pose and best_iou > 0.20:
                        if best_pose.pose_type == PoseType.RIDING:
                            log.debug(f"  Skip RIDING (pose) {det['bbox']} iou={best_iou:.2f}")
                            continue
                        det = dict(det)
                        det["pose_factor"] = best_pose.height_factor
                        if best_pose.keypoints is not None:
                            det["keypoints"] = best_pose.keypoints
                        # If very close match, no need to search further

            out.append(det)
        return out

    def _run_segformer(self, img_rgb, h, w):
        if not self.use_segformer: return h, 0.0, 0.0
        self._load_segformer()
        seg_model = self._seg_model
        seg_proc = self._seg_proc
        if seg_model is None or seg_proc is None: return h, 0.0, 0.0
        try:
            import torch
            inp = seg_proc(images=Image.fromarray(img_rgb), return_tensors="pt")
            inp = {k: v.to(self._seg_dev) for k,v in inp.items()}
            with torch.no_grad():
                logits = seg_model(**inp).logits
            seg = torch.nn.functional.interpolate(
                logits, size=(h,w), mode="bilinear", align_corners=False
            ).argmax(1).squeeze(0).cpu().numpy()
            road_pct  = float(np.isin(seg, [6,11,3,13]).mean() * 100)
            water_pct = float(np.isin(seg, [21,26,60]).mean() * 100)
            wl_hint   = h
            if water_pct > 1.5:
                rows = np.where(np.isin(seg, [21,26,60]).any(axis=1))[0]
                if len(rows): wl_hint = int(np.percentile(rows, 10))
            return wl_hint, road_pct, water_pct
        except Exception as e:
            log.debug(f"  SegFormer: {e}"); return h, 0.0, 0.0

    def _find_water_line(self, img_rgb, wmask, dets, h, w, has_flood, lower_pct,
                          seg_wl_y=None, seg_road_pct=0.0):
        """
        Tim duong nuoc (water line).

        Tin hieu 1: Color boundary (mau nuoc HSV)
        Tin hieu 2: Texture boundary (nuoc co reflection texture khac mat duong)
        Tin hieu 3: YOLO foot consensus
        Tin hieu 4: SegFormer hint

        Ban dem: color mask yeu -> dua nhieu vao texture boundary va YOLO foot.
        Sanity check:
          - visibility trung binh > 75% nhung WL qua cao -> day WL xuong
          - nhieu object high-visibility -> anh it ngap
        """
        lower_pct2 = float(wmask[h//2:].sum()/255) / max(wmask[h//2:].size, 1) * 100

        # Phat hien anh ban dem (brightness thap)
        gray_img   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        is_night   = float(gray_img.mean()) < 80   # anh toi trung binh

        # Bo sung: Texture boundary (nuoc co reflection shimmer)
        # Laplacian variance thap = mat nuoc phang; cao = mat duong/nen kho
        if is_night:
            lap = cv2.Laplacian(gray_img, cv2.CV_64F)
            lap_abs = np.abs(lap)
            # Chia anh thanh dai ngang, tim dai chuyen tu texture cao sang thap
            n_bands = 20
            band_h  = max(1, h // n_bands)
            band_vars = []
            for i in range(n_bands):
                band = lap_abs[i*band_h:(i+1)*band_h, :]
                band_vars.append(float(band.mean()))
            # Tim ranh gioi: tu vung texture cao (nen kho) -> thap (mat nuoc)
            # Tim band dau tien co lap_mean giam xuong < 50% cua phan tren
            tex_wl_y = h
            if len(band_vars) >= 4:
                upper_mean = np.mean(band_vars[:len(band_vars)//2])
                for bi in range(len(band_vars)//2, len(band_vars)):
                    if band_vars[bi] < upper_mean * 0.45:
                        tex_wl_y = bi * band_h
                        break
        else:
            tex_wl_y = h

        if lower_pct2 < 2.0 and not has_flood and not is_night:
            return h, 0.08

        # Ban dem: co the has_flood = False vi color mask yeu, nhung thuc ra co nuoc
        if is_night and lower_pct2 < 1.0 and tex_wl_y < h * 0.85:
            has_flood = True   # override: texture cho thay co mat nuoc

        # Tin hieu 1: Color boundary
        # FIX: Neu co detections, khong cho water_line len cao hon 55% bbox cao nhat
        # (tranh lay nuoc background/tuong/nen)
        search_top = h // 3
        if dets:
            # Tim bh trung binh va y1 cua vat the thap nhat
            avg_bh     = int(np.mean([d["bbox"][3]-d["bbox"][1] for d in dets[:4]]))
            min_y1     = min(d["bbox"][1] for d in dets[:4])
            max_y2     = max(d["bbox"][3] for d in dets[:4])
            # Water line khong len cao hon 55% tu chan vat the
            safe_top   = max(min_y1 + int(avg_bh * 0.40),  # 40% tu dinh vat the
                             max_y2  - int(avg_bh * 0.75))  # 75% tu chan vat the
            search_top = max(search_top, safe_top)

        rows = np.where(wmask[search_top:].any(axis=1))[0]
        if len(rows):
            color_wl   = int(np.percentile(rows, 10)) + search_top
            color_conf = min(0.88, lower_pct2/15.0 + 0.30)
        else:
            color_wl, color_conf = h, 0.10

        # Tin hieu 2: YOLO foot
        foot_cands = []
        for det in dets[:6]:
            x1,y1,x2,y2 = det["bbox"]
            roi = wmask[max(0,y2-10):min(h,y2+15), max(0,x1-5):min(w,x2+5)]
            if roi.size > 0 and (roi.sum()/255)/max(roi.size,1) > 0.12:
                foot_cands.append((y2, det["confidence"]))

        if foot_cands:
            foot_ys  = sorted([y for y, _ in foot_cands])
            foot_p25 = int(np.percentile(foot_ys, 25))
            # FIX: YOLO foot consensus la anchor chinh xac hon color boundary
            # Khi background co nuoc sau (cuoi pho, song...), color_wl bi keo len
            # YOLO foot cho biet vi tri THUC cua chan vat the = nuoc chinh xac nhat
            # => uu tien foot consensus, chi dung color_wl neu no THAP HON (an toan)
            if len(foot_cands) >= 2:
                # Nhieu vat the dong thuan: tin vao foot consensus
                wl_y    = foot_p25
                wl_conf = min(0.92, 0.55 + 0.08 * min(len(foot_cands), 4))
            else:
                # 1 vat the: ket hop voi color (nhung khong de color keo len)
                # water_line = max cua 2 gia tri (chon gia tri THAP HON = an toan hon)
                wl_y    = max(foot_p25, color_wl)   # lay gia tri THAP HON (larger y = lower)
                wl_conf = min(0.85, color_conf + 0.06)
        else:
            wl_y, wl_conf = color_wl, color_conf

        # Tin hieu 3: Texture boundary (ban dem)
        if is_night and tex_wl_y < h * 0.90 and wl_conf < 0.55:
            wl_y    = int(0.55 * wl_y + 0.45 * tex_wl_y)
            wl_conf = min(0.70, wl_conf + 0.15)

        # Tin hieu 4: SegFormer
        if wl_conf < 0.50 and seg_wl_y and seg_wl_y < h*0.92:
            wl_y    = int(0.6*wl_y + 0.4*seg_wl_y)
            wl_conf = min(0.65, wl_conf + 0.12)

        # SegFormer road penalty
        if seg_road_pct > 30.0:
            wl_y = min(h, wl_y + min(int(h*0.08), int((seg_road_pct-30)*1.5)))

        # Sanity: neu nhieu doi tuong co visibility cao -> WL qua cao
        if dets:
            vis_list = []
            for det in dets[:5]:
                x1,y1,x2,y2 = det["bbox"]
                bh  = max(y2-y1, 1)
                sub = max(0, y2-wl_y) if wl_y < y2 else 0
                vis_list.append((bh-sub)/bh)
            avg_vis = float(np.mean(vis_list))
            n_high_vis = sum(1 for v in vis_list if v > 0.7)

            # Neu trung binh visibility cao va WL qua cao -> day xuong
            if avg_vis > 0.75 and wl_y < int(h*0.60):
                foot_mean = int(np.mean([d["bbox"][3] for d in dets[:5]]))
                new_wl    = int(foot_mean * 0.85 + h * 0.15)
                if new_wl > wl_y:
                    wl_y    = new_wl
                    wl_conf *= 0.65
                    log.debug(f"  WL sanity down: avg_vis={avg_vis:.2f} -> {wl_y}")

            # Them: neu NHIEU object co high visibility (>= 3) -> anh it ngap
            if n_high_vis >= 3 and wl_y < int(h * 0.70):
                foot_max = int(np.max([d["bbox"][3] for d in dets[:5]]))
                wl_y     = max(wl_y, int(foot_max * 0.80 + h * 0.20))
                wl_conf *= 0.60

        wl_y = max(int(h*0.08), min(h, wl_y))

        # ── Force waterline from person ankle keypoints (local strong prior)
        ankle_ys = []
        for det_item in dets:
            if det_item.get("class_name") != "person":
                continue
            kpts = det_item.get("keypoints")
            if not isinstance(kpts, np.ndarray) or kpts.shape[0] < 17:
                continue
            l_ankle = float(kpts[15][1]) if kpts[15][2] > 0.25 else None
            r_ankle = float(kpts[16][1]) if kpts[16][2] > 0.25 else None
            x1,y1,x2,y2 = det_item["bbox"]
            for pt in (l_ankle, r_ankle):
                if pt is not None and y1 + 5 < pt < y2 - 5:
                    # Chỉ dùng khi ankle nằm trong bbox và có khả năng chìm
                    ankle_ys.append(pt)

        if ankle_ys:
            # Dùng median, tránh 1 người phản lại toàn bộ
            candidate_ankle = int(np.median(ankle_ys))
            if candidate_ankle > wl_y:
                wl_y = min(candidate_ankle + 2, h - 3)
                wl_conf = min(0.95, wl_conf + 0.12)
                log.debug(f"  ForceWL from ankle keypoints: ankle={candidate_ankle} -> wl_y={wl_y}")

        # ── Sanity: neu water_line qua cao + tat ca objects visibility ~16%
        # → đây la dau hieu water_mask bi nhiem (ao mua, fence...)
        # → Reset water_line xuong thap hon
        if dets and wl_y < int(h * 0.40):
            # Tinh visibility trung binh NEU dung wl_y nay
            vis_list = []
            for det in dets[:8]:
                x1,y1,x2,y2 = det["bbox"]
                bh   = max(y2-y1, 1)
                sub  = max(0, y2-wl_y) if wl_y < y2 else 0
                vis_list.append((bh-sub)/bh)

            if vis_list:
                avg_vis  = float(np.mean(vis_list))
                vis_std  = float(np.std(vis_list))
                # "16% visible pattern": tat ca objects cung visibility thap
                # va it bien thien → water_line sai he thong
                # Extended: bat ca truong hop 10-30% visibility dong deu
                all_low_similar = (avg_vis < 0.28 and vis_std < 0.08
                                   and len(vis_list) >= 2)
                if all_low_similar:
                    # Tim bottom cua objects lam reference
                    bottoms  = [d["bbox"][3] for d in dets[:6]]
                    foot_avg = int(np.mean(bottoms))
                    foot_med = int(np.median(bottoms))

                    # Water line = 10px tren foot median (nuoc tiep xuc chan)
                    # Neu co nuoc roi, max ngap la 70% bbox height tu day
                    bh_avg = int(np.mean([d["bbox"][3]-d["bbox"][1] for d in dets[:6]]))
                    # Conservative: nuoc cao max 70% chieu cao vat the
                    max_water_y = foot_med - int(bh_avg * 0.70)
                    new_wl = max(max_water_y, int(h * 0.55), foot_avg - int(bh_avg * 0.65))
                    log.info(
                        f"  WL sanity RESET: avg_vis={avg_vis:.0%} std={vis_std:.2f} "
                        f"→ wl_y {wl_y}→{new_wl} (16%-pattern fix)"
                    )
                    wl_y    = new_wl
                    wl_conf = min(wl_conf, 0.35)

        return wl_y, round(wl_conf, 3)



    def _estimate_person_foot_y(self, det: dict, all_dets: list,
                                 img_rgb: np.ndarray, h: int, w: int) -> tuple:
        """
        Uoc tinh vi tri chan thuc (foot_y) cua nguoi.

        vấn đề: Khi nguoi đúng canh/tren xe may bi ngap, bbox YOLO bao gom ca
        phan xe ben duoi → bh bi phong đai → water_cm tinh sai.

        Phuong phap uu tien:
          1. Pose keypoints (ankle/hip): chinh xac nhat
          2. Tach xe may ben duoi: neu co moto bbox overlap
          3. Edge detection: tim ranh gioi nguoi/xe
          4. Fallback: dung y2 goc

        Tra ve (foot_y, confidence)
        """
        x1, y1, x2, y2 = det["bbox"]
        bh = y2 - y1
        # ── Phuong phap 1: Pose keypoints ────────────────────────────
        if hasattr(self, '_pose_mdl') and self._pose_mdl:
            try:
                poses = self._pose_mdl.analyze_image(img_rgb)
                for pose in poses:
                    pb = pose.bbox
                    # Tim pose tuong ung voi detection nay
                    # _iou is a module-level function, no import needed
                    if _iou(list(pb), det["bbox"]) > 0.40 and pose.keypoints is not None:
                        kpts = pose.keypoints  # shape (17, 3) [x, y, conf]
                        # Keypoint 15,16 = left/right ankle
                        ankles = [(kpts[15][1], kpts[15][2]),
                                  (kpts[16][1], kpts[16][2])]
                        valid_ankle = [(y, c) for y, c in ankles if c > 0.2 and y > y1]
                        if valid_ankle:
                            ankle_y = float(max(y for y, _ in valid_ankle))
                            # Hieu chinh boi vi point ngoai da co 1-2cm tu mongo
                            foot_y = min(y2, int(ankle_y + max(6, int(bh * 0.04))))
                            conf   = float(np.mean([c for _, c in valid_ankle]))
                            log.debug(f"  Foot from ankle keypoints: ankle_y={ankle_y} -> foot_y={foot_y} conf={conf:.2f}")
                            return foot_y, conf

                        # Fallback: dung knee neu ankle khong ro
                        knees = [(kpts[13][1], kpts[13][2]),
                                 (kpts[14][1], kpts[14][2])]
                        valid_knee = [(y, c) for y, c in knees if c > 0.3 and y > y1]
                        if valid_knee:
                            knee_y = float(max(y for y, _ in valid_knee))
                            foot_y = min(y2, int(knee_y + max(10, int(bh * 0.18))))
                            conf   = float(np.mean([c for _, c in valid_knee])) * 0.8
                            log.debug(
                                f"  Foot fallback from knees: knee_y={knee_y} -> foot_y={foot_y} conf={conf:.2f}"
                            )
                            return foot_y, conf
            except Exception as e:
                log.debug(f"  Pose keypoint foot failed: {e}")

        # ── Phuong phap 2: Loai phan xe may ben duoi ─────────────────
        # Tim motorcycle/bicycle bbox co overlap duoi cung cua person bbox
        for other in all_dets:
            if other is det: continue
            if other["class_name"] not in ("motorcycle", "bicycle"): continue
            ox1, oy1, ox2, oy2 = other["bbox"]
            # Xe ben duoi person: oy1 phai nam trong nua duoi bbox nguoi
            mid_person = y1 + bh // 2
            iou_val    = _iou([x1,y1,x2,y2], [ox1,oy1,ox2,oy2])
            if iou_val > 0.15 and oy1 > mid_person:
                # Chan nguoi đ top cua xe may (hoac 10px tren top xe)
                foot_y = max(y1 + int(bh * 0.3), oy1 - 5)
                log.debug(f"  Foot from motorcycle bbox: y={foot_y} (moto_top={oy1})")
                return foot_y, 0.65

        # ── Phuong phap 3: Edge detection để tach nguoi / xe ────────
        # Tim ranh gioi ngang manh trong nua duoi bbox
        # (ranh gioi nguoi-xe thuong co edge manh)
        try:
            roi = img_rgb[y1:y2, max(0,x1):min(w,x2)]
            if roi.shape[0] > 20:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                # Tinh gradient theo chieu doc
                grad_y   = cv2.Sobel(gray_roi.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
                row_grad = np.abs(grad_y).mean(axis=1)
                # Tim hang co gradient manh nhat trong 40-80% duoi bbox
                search_start = int(bh * 0.40)
                search_end   = int(bh * 0.80)
                if search_end > search_start:
                    sub_grad = row_grad[search_start:search_end]
                    peak_rel = int(np.argmax(sub_grad))
                    peak_abs = search_start + peak_rel
                    if row_grad[peak_abs] > row_grad.mean() * 2.5:
                        foot_y   = y1 + peak_abs
                        log.debug(f"  Foot from edge peak: y={foot_y}")
                        return foot_y, 0.45
        except Exception:
            pass

        # ── Fallback: dung y2 goc ────────────────────────────────────
        return y2, 0.20

    def _measure_objects_local(self, dets, wmask, global_wl_y, global_wl_conf,
                                h, w, img_rgb=None, is_aerial=False,
                                angle_factor=1.0, is_low_angle=False):
        """
        đo đo ngap voi LOCAL water line cho tung object.

        Tai sao can local:
          - Global water line bi keo cao boi nuoc sau o xa (background)
          - Object o gan co the chi ngap đến goi du background ngap đến nguc
          - Local water line = đường nuoc tai VI TRI cua object đo

        Uu tien:
          1. Local water contact (water mask trong vung object)
          2. Global water line (fallback neu khong co local)

        Aerial view correction:
          - Tu tren cao, phan "visible" cua object khong tuong đường voi đo ngap
          - Can giam manh water_cm voi angle_factor
        """
        measured = []
        for det in dets:
            # Bo qua objects bi đanh dau skip (traffic light, stop sign)
            # Van ve bbox nhung khong dung để đo muc nuoc
            if det.get("skip_measure", False):
                continue

            x1, y1, x2, y2 = det["bbox"]
            ref_h       = float(det.get("ref_height_cm", 170))
            pose_factor = float(det.get("pose_factor", 1.0))
            conf        = float(det.get("confidence", 0.5))
            obj_class   = det.get("class_name", "unknown")

            # ── Height estimation cho PERSON ─────────────────────────
            # Uu tien: keypoint (ankle/hip visible) > head_size > shoulder (fallback)
            # Shoulder width chi dung khi KHONG co keypoint dang tin cay
            if obj_class == "person" and img_rgb is not None:
                kpts = det.get("keypoints", None)   # (17,3) neu co pose

                # Kiem tra xem keypoints co du dung tin cay khong
                # (index 15/16=ankle, 13/14=knee, 11/12=hip, 5/6=shoulder)
                _has_good_kpts = False
                if kpts is not None and isinstance(kpts, np.ndarray) and len(kpts) >= 17:
                    ankles = [kpts[i][2] for i in (15, 16) if kpts[i][2] > 0.30]
                    hips   = [kpts[i][2] for i in (11, 12) if kpts[i][2] > 0.30]
                    _has_good_kpts = len(ankles) >= 1 or len(hips) >= 1

                # Shoulder estimation: chi dung khi keypoints khong du
                if not _has_good_kpts:
                    try:
                        shoulder_est = _get_shoulder_est()
                        sh_result = shoulder_est.estimate(
                            img_rgb     = img_rgb,
                            bbox        = [x1, y1, x2, y2],
                            keypoints   = kpts,
                            pose_factor = pose_factor,
                        )
                        if sh_result.confidence > 0.30:
                            old_ref_h = ref_h
                            blend_w   = min(0.65, sh_result.confidence)   # blend nhe hon (0.65 vs 0.75)
                            ref_h     = blend_w * sh_result.estimated_height_cm + (1 - blend_w) * old_ref_h
                            ref_h     = max(140.0, min(200.0, ref_h))
                            conf     *= (0.80 + sh_result.confidence * 0.15)   # boost nhe hon
                            log.debug(
                                f"  ShoulderEst (fallback) [{sh_result.method}]: "
                                f"default={old_ref_h:.0f}cm → blended={ref_h:.0f}cm "
                                f"(conf={sh_result.confidence:.2f})"
                            )
                            det = dict(det)
                            det["estimated_height_cm"] = round(ref_h, 1)
                            det["shoulder_method"]     = sh_result.method
                            det["shoulder_conf"]       = sh_result.confidence
                    except Exception as e:
                        log.debug(f"  ShoulderEst failed: {e}")
                else:
                    log.debug(f"  Person: keypoints OK → skip shoulder fallback")

            # ── Uoc tinh vi tri chan thuc cua PERSON ─────────────────
            # Tranh truong hop bbox bao gom ca xe may ben duoi
            # → bh bi phong đai → water_cm tinh sai cao
            effective_y2 = y2
            if obj_class == "person" and img_rgb is not None:
                est_foot_y, foot_conf = self._estimate_person_foot_y(
                    det, dets, img_rgb, h, w)
                if foot_conf > 0.30 and est_foot_y < y2:
                    # Chi điều chinh neu uoc tinh đang tin cay VA
                    # foot_y thuc su nho hon y2 (co su chenh lech)
                    reduction_ratio = (y2 - est_foot_y) / max(y2 - y1, 1)
                    if reduction_ratio > 0.10:  # chenh > 10% moi điều chinh
                        effective_y2 = est_foot_y
                        log.debug(f"  Person foot correction: y2={y2}→{effective_y2} "
                                  f"(cut {reduction_ratio:.0%})")

            bh = max(effective_y2 - y1, 1)
            # ── Tim local water line ──────────────────────────────────
            # Dung effective_y2 để tim water contact tai vi tri chan thuc
            det_for_local = dict(det)
            det_for_local["bbox"] = [x1, y1, x2, effective_y2]
            local_wl_y, local_conf = self._local_water_line(wmask, det_for_local, h, w)

            # Chon water line tot nhat
            if local_conf > 0.30 and local_wl_y < y2:
                # Co local water line tin cay → dung
                effective_wl = local_wl_y
                effective_conf = local_conf
                wl_source = "local"
            elif global_wl_conf > 0.20 and global_wl_y < y2:
                # Fallback ve global
                effective_wl = global_wl_y
                effective_conf = global_wl_conf * 0.7  # giam confidence
                wl_source = "global"
            else:
                # Khong co nuoc tai object nay
                effective_wl = y2  # water line = đây bbox = khong ngap
                effective_conf = 0.1
                wl_source = "none"

            # --- Person ankle override (keypoint evidence) ---
            if obj_class == "person" and det.get("keypoints") is not None:
                kpts = det.get("keypoints")
                if isinstance(kpts, np.ndarray) and kpts.shape[0] >= 17:
                    l_ankle = float(kpts[15][1]) if kpts[15][2] > 0.25 else None
                    r_ankle = float(kpts[16][1]) if kpts[16][2] > 0.25 else None
                    ankle_y = None
                    for val in (l_ankle, r_ankle):
                        if val is not None and y1 + 5 < val < y2 - 5:
                            ankle_y = val if ankle_y is None else max(ankle_y, val)

                    if ankle_y is not None and ankle_y > effective_wl + 4:
                        # Ưu tiên ankle keypoint (càng gần chân càng đúng)
                        new_wl = min(int(ankle_y + 3), y2 - 2)
                        if new_wl > effective_wl:
                            log.debug(
                                f"  Ankle override: wl {effective_wl}->{new_wl} ankle={ankle_y}"
                            )
                            effective_wl = new_wl
                            effective_conf = max(effective_conf, 0.5)

            # ── Tinh pixels bi ngap ──────────────────────────────────
            # QUAN TRONG: dung effective_y2 (sau foot correction), KHONG dung y2_orig
            # Neu dung y2_orig thi sub_px > bh → vis_ratio am → sai
            if effective_wl >= effective_y2:
                sub_px = 0               # water line duoi chan → khong ngap
            elif effective_wl <= y1:
                sub_px = bh             # water line tren đầu → ngap hoan toan
            else:
                sub_px = effective_y2 - effective_wl   # phan duoi water line

            # Clamp để tranh sub_px > bh (defensive programming)
            sub_px = max(0, min(sub_px, bh))
            vis_px = bh - sub_px
            vis_r  = vis_px / max(bh, 1)   # tranh chia 0

            # ── [SAM2] Tinh chỉnh pixel bằng mask chính xác ────────────
            # Bbox hình chữ nhật chứa background → đếm px sai khi người nghiêng.
            # SAM mask theo đúng hình dáng object → chính xác hơn rõ rệt.
            if img_rgb is not None and effective_wl < effective_y2:
                try:
                    from depth_analysis.sam_segmentor import refine_submersion_pixels
                    sub_px_sam, sam_conf = refine_submersion_pixels(
                        img_rgb, det, effective_y2, effective_wl, y1, sub_px,
                        cfg=getattr(self, "_cfg", None),
                    )
                    if sam_conf > 0 and abs(sub_px_sam - sub_px) > 2:
                        log.debug(
                            f"  [SAM] {obj_class}: sub_px {sub_px}→{sub_px_sam} "
                            f"(mask-based)"
                        )
                        sub_px = sub_px_sam
                        vis_px = bh - sub_px
                        vis_r  = vis_px / max(bh, 1)
                except Exception as e_sam:
                    log.debug(f"  [SAM] hook fail (dùng bbox): {e_sam}")

            # ── Tinh water_cm ─────────────────────────────────────────
            # Cong thuc: ref_height × pose_factor × ti_le_ngap
            water_cm = ref_h * pose_factor * (sub_px / bh)

            # Aerial view correction: giam water_cm
            if is_aerial:
                water_cm *= angle_factor
                conf     *= max(0.3, angle_factor)

            # Low-angle shot correction:
            # Camera gan mat nuoc → car/vehicle bi cat phan duoi khong phai do ngap
            # Vehicles bi anh huong nhieu hon nguoi (nguoi thuong đúng cao hon)
            if is_low_angle and obj_class in ("car", "truck", "bus", "motorcycle", "bicycle"):
                # Giam water_cm cho vehicle trong low-angle shot
                water_cm *= 0.50
                conf     *= 0.60
                log.debug(f"  LowAngle vehicle reduce [{obj_class}]: {water_cm:.0f}cm")

            # ── Sanity checks ─────────────────────────────────────────
            # Neu object 80%+ visible nhung water_cm > 50% ref → mau thuan
            if vis_r > 0.80 and water_cm > ref_h * 0.45:
                water_cm *= 0.20
                conf     *= 0.40
                log.debug(f"  Sanity reduce: {obj_class} vis={vis_r:.0%} → {water_cm:.0f}cm")

            # Neu khong co local water contact va global thap → khong ngap
            if wl_source == "none":
                water_cm = 0.0
                conf    *= 0.3

            # ── [IMPROVE] Adaptive vehicle bias correction ──────────────
            # Cu: VEHICLE_WATER_BIAS固定 (motorcycle=0.75, car=0.90...)
            #     khong phu vao goc chup.
            # Moi: adaptive — goc chup (aerial, low-angle) thay doi
            #   muc overestimate cua vehicle.
            #   - Aerial (bird's eye): xe bi cat ngan → overestimate nhieu hon
            #   - Low-angle: camera gan mat nuoc → underestimate hon
            vehicle_bias = VEHICLE_WATER_BIAS.get(obj_class, 1.0)
            if vehicle_bias < 1.0 and water_cm > 0:
                # Neu aerial: tang bias len (giam nhieu hon)
                if is_aerial:
                    vehicle_bias = max(0.45, vehicle_bias - 0.12)
                elif is_low_angle:
                    vehicle_bias = min(0.95, vehicle_bias + 0.10)
                water_cm_orig = water_cm
                water_cm     *= vehicle_bias
                log.debug(f"  Vehicle bias [{obj_class}]: {water_cm_orig:.0f} → {water_cm:.0f}cm "
                          f"(adaptive={vehicle_bias:.2f} aerial={is_aerial})")

            # ── Nhan dien bo phan tiep xuc ───────────────────────────
            # [FIX] Dung ref_h (chieu cao THUC) thay vi ref_h*pose_factor
            # Vi _classify_body_contact so sanh ratio = water_cm / ref_h voi
            # nguong body_part (ankle=0.20, knee=0.35...). Neu nguoi ngoi
            # (pose_factor=0.55), ref_h*pose_factor = 93.5cm → ratio bi phong dai
            # → gan sai body part (VD: nuoc 30cm → ratio=0.32="knee" thay vi "ankle").
            body_part = self._classify_body_contact(water_cm, ref_h)

            if obj_class == "person":
                kp_contact = self._classify_body_contact_from_keypoints(det, effective_wl)
                if kp_contact is not None:
                    body_part = kp_contact

                    part_ratio = {
                        "ankle": 0.20,
                        "knee":  0.32,
                        "waist": 0.48,
                        "chest": 0.66,
                    }.get(body_part)
                    if part_ratio is not None:
                        predicted = ref_h * pose_factor * part_ratio
                        if water_cm > predicted * 1.2:
                            log.debug(
                                f"  LOSSY: adjust CM from {water_cm:.1f} to {predicted:.1f} "
                                f"based on keypoint body_part={body_part}"
                            )
                            water_cm = min(water_cm, predicted * 1.15)

            measured.append(DetectedObject(
                class_name          = obj_class,
                bbox                = [x1, y1, x2, y2],
                confidence          = round(min(conf, 0.95), 3),
                ref_height_cm       = ref_h,
                pose_factor         = round(pose_factor, 2),
                water_height_cm     = round(max(0.0, water_cm), 1),
                visibility_ratio    = round(vis_r, 3),
                pixels_total        = bh,
                pixels_above_water  = vis_px,
                pixels_below_water  = sub_px,
                local_wl_y          = effective_wl,
                local_wl_conf       = round(effective_conf, 2),
                body_part_contact   = body_part,
                estimated_height_cm = round(det.get("estimated_height_cm", 0.0), 1),
                shoulder_width_cm   = round(det.get("shoulder_width_cm", 0.0), 1),
                shoulder_method     = det.get("shoulder_method", ""),
                shoulder_conf       = round(det.get("shoulder_conf", 0.0), 3),
            ))

            log.debug(f"  Measured {obj_class}: {water_cm:.0f}cm ({body_part}) "
                      f"vis={vis_r:.0%} wl_src={wl_source}")
        return measured

    def _measure_objects(self, dets, wl_y, img_h):
        """
        Do muc nuoc cho tung vat the.

        FIX QUAN TRONG:
        - ref_height_cm = chieu cao THUC (170 cho nguoi, KHONG doi)
        - pose_factor dieu chinh do ngap: mot nguoi ngoi (pose_factor=0.55)
          co chieu cao "huu dung" = 170*0.55 = 93.5cm
          => do ngap cung duoc nhan theo he so nay
        - Label: "person | 170cm (23% visible)" -> LUON hien thi ref_height_cm thuc
        """
        measured = []
        for det in dets:
            x1,y1,x2,y2 = det["bbox"]
            bh          = max(y2-y1, 1)
            ref_h       = float(det.get("ref_height_cm", 170))  # KHONG thay doi
            pose_factor = float(det.get("pose_factor", 1.0))    # he so tu the
            conf        = float(det.get("confidence", 0.5))

            # Tinh pixels bi ngap
            if wl_y >= y2:     sub_px = 0       # khong ngap
            elif wl_y <= y1:   sub_px = bh       # ngap hoan toan
            else:              sub_px = y2 - wl_y

            vis_px   = bh - sub_px
            vis_r    = vis_px / bh

            # water_cm: dung pose_factor vi khi nguoi ngoi, chieu cao thuc te thap hon
            # Cong thuc: water_cm = ref_height * pose_factor * (sub_px / bh)
            water_cm = ref_h * pose_factor * (sub_px / bh)

            # Sanity: vis > 80% nhung water > 50% ref -> mau thuan -> reduce
            if vis_r > 0.80 and water_cm > ref_h * 0.45:
                water_cm *= 0.20
                conf     *= 0.40

            measured.append(DetectedObject(
                class_name=det["class_name"], bbox=[x1,y1,x2,y2],
                confidence=round(conf, 3),
                ref_height_cm=ref_h,            # LUON la chieu cao thuc
                pose_factor=round(pose_factor, 2),
                water_height_cm=round(max(0.0, water_cm), 1),
                visibility_ratio=round(vis_r, 3),
                pixels_total=bh, pixels_above_water=vis_px, pixels_below_water=sub_px,
            ))
        return measured

    def _synthesize(self, measured, wl_y, wl_conf, water_pct, lower_pct,
                    has_flood, depth_r, img_h, flood_prob,
                    is_aerial=False, angle_factor=1.0, fov_deg=65.0):
        """
        Tong hop ket qua cuoi cung tu nhieu nguon.

        Uu tien nguon (tu cao đến thap):
          1. Local-measured objects (co local water contact)  → tin cay nhat
          2. Global water line estimate
          3. Color water area estimate                        → it tin cay nhat

        Aerial correction:
          - Anh tu tren cao: visibility cao khong co nghia la khong ngap
          - Cap water_cm theo angle_factor
        """
        # Khong co lu
        if not has_flood and water_pct < 4.0:
            return 0.0, 0.85, "NO_FLOOD", "Không có lũ"

        # ── Nguon 1: Measured objects ────────────────────────────────
        if measured:
            # Loai HIGH_MOUNT objects va objects khong co water contact
            reliable = [o for o in measured
                        if o.class_name not in HIGH_MOUNT_OBJECTS
                        and o.body_part_contact != "none"
                        and o.water_height_cm > 0]
            if not reliable:
                reliable = [o for o in measured
                            if o.class_name not in HIGH_MOUNT_OBJECTS]
            if not reliable:
                reliable = measured

            # ── Person-first strategy ───────────────────────────────
            # Person la reference tot nhat cho flood depth
            # Neu co person measurement tin cay → uu tien cao hon nhieu so voi vehicle
            persons   = [o for o in reliable if o.class_name == "person" and o.water_height_cm > 0]
            vehicles  = [o for o in reliable if o.class_name in ("motorcycle","bicycle","car","truck","bus")]

            if persons:
                # Co person: dung person lam primary, vehicle lam sanity check
                p_confs  = sum(o.confidence * max(o.local_wl_conf, 0.15) for o in persons)
                p_cm     = sum(o.water_height_cm * o.confidence * max(o.local_wl_conf, 0.15)
                               for o in persons) / max(p_confs, 1e-6)
                # FIX: Loai outlier truoc khi tinh median
                # Person voi visibility cao (> 60%) -> water_cm phai thap
                # Person voi visibility < 30% -> co the ngap nhieu
                filtered_persons = []
                for p in persons:
                    vis = p.visibility_ratio
                    wcm = p.water_height_cm
                    rh  = p.ref_height_cm
                    expected_cm = rh * (1.0 - vis)
                    if expected_cm > 0 and wcm > expected_cm * 2.5:
                        corrected = expected_cm
                        log.debug(f"  Person outlier fix: {wcm:.0f}cm → {corrected:.0f}cm "
                                  f"(vis={vis:.0%} expected={expected_cm:.0f}cm)")
                        # [FIX] Dung dataclasses.replace thay vi copy.copy()
                        # copy.copy() shallow → shared list/array (bbox, keypoints)
                        # → neu modify bbox sau bi anh huong nguoi khac.
                        from dataclasses import replace
                        p2 = replace(p, water_height_cm=round(corrected, 1))
                        filtered_persons.append(p2)
                    else:
                        filtered_persons.append(p)
                persons = filtered_persons if filtered_persons else persons

                # Use median person estimate for multi-person robustness
                person_median = float(np.median([o.water_height_cm for o in persons]))
                # Default keeps the estimate defined when no vehicle adjustment
                # is applied (for example, with three or more persons).
                yolo_cm = person_median

                if len(persons) >= 3:
                    # ── [IMPROVE] MAD-based outlier detection ────────────
                    # Cu: cluster_bins (5cm bin) + threshold 7.5cm — he so vo canh.
                    # Moi: Median Absolute Deviation (MAD) — 3.5x MAD = outlier.
                    # MAD = median(|xi - median(x)|), robust hon std khi co outliers.
                    person_cms = np.array([o.water_height_cm for o in persons])
                    mad = float(np.median(np.abs(person_cms - person_median)))
                    mad = max(mad, 1.0)  # tranh mad=0 voi tat ca giong nhau
                    # MAD * 1.4826 ≈ std neu normal distribution
                    # Dung 3.0 * 1.4826 * mad ≈ 4.45 * mad lam nguong outlier
                    outlier_bound = 4.45 * mad
                    inliers = [o for o in persons
                               if abs(o.water_height_cm - person_median) <= outlier_bound]
                    if len(inliers) >= 2:
                        inlier_medians = [o.water_height_cm for o in inliers]
                    else:
                        inlier_medians = [o.water_height_cm for o in persons]

                    person_major_median = float(np.median(inlier_medians))
                    yolo_cm = 0.90 * person_major_median + 0.10 * person_median
                    if len(inliers) < len(persons):
                        log.debug(f"  MAD outlier: {len(persons)-len(inliers)} removed "
                                  f"(bound={outlier_bound:.1f}cm, mad={mad:.1f}cm)")

                # [IMPROVE] Multi-person height consensus:
                # Cross-validate estimated heights của multiple people.
                # Nếu nhiều người trong cùng ảnh → height estimates nên consistent.
                if len(persons) >= 3:
                    est_heights = [o.estimated_height_cm for o in persons if o.estimated_height_cm > 0]
                    if len(est_heights) >= 3:
                        height_median = float(np.median(est_heights))
                        height_mad = float(np.median(np.abs(np.array(est_heights) - height_median)))
                        height_mad = max(height_mad, 2.0)
                        # Nếu spread quá lớn (> 15cm) → có thể có lỗi đo
                        if height_mad > 7.5:
                            log.warning(f"  Height consensus: large spread MAD={height_mad:.1f}cm "
                                        f"among {len(persons)} persons → check measurements")
                            # Flag outliers
                            for p in persons:
                                if p.estimated_height_cm > 0 and abs(p.estimated_height_cm - height_median) > height_mad * 3:
                                    log.warning(f"    Person height outlier: {p.estimated_height_cm:.0f}cm "
                                                f"(median={height_median:.0f}cm)")
                else:
                    if vehicles:
                        v_confs = sum(o.confidence * max(o.local_wl_conf, 0.15) for o in vehicles)
                        v_cm    = sum(o.water_height_cm * o.confidence * max(o.local_wl_conf, 0.15)
                                      for o in vehicles) / max(v_confs, 1e-6)
                        if v_cm > person_median * 1.8:
                            v_cm = person_median * 1.3
                        # majority: persons quyết định, vehicles chỉ thả once
                        yolo_cm = 0.85 * person_median + 0.15 * v_cm
                    else:
                        yolo_cm = person_median

                total_w = p_confs + 0.1
                log.debug(f"  Person-first: person_avg={p_cm:.0f}cm median={person_median:.0f}cm yolo={yolo_cm:.0f}cm")
            else:
                # Khong co person: dung vehicle
                total_w = sum(o.confidence * max(o.local_wl_conf, 0.1) for o in reliable)
                yolo_cm = sum(
                    o.water_height_cm * o.confidence * max(o.local_wl_conf, 0.1)
                    for o in reliable
                ) / max(total_w, 1e-6)
                water_cms = [o.water_height_cm for o in reliable if o.water_height_cm > 0]
                if len(water_cms) >= 2:
                    yolo_cm = float(np.median(water_cms)) * 0.6 + yolo_cm * 0.4
        else:
            yolo_cm, total_w = 0.0, 0.0

        # ── Nguon 2: Global water line ───────────────────────────────
        # [FIX] Cong thuc cu: water_ratio * 170 * 0.72 = SAI
        #   - water_ratio = ty le pixel tu water_line den day anh (0-1)
        #   - 170 = chieu cao nguoi, 0.72 = "magic number"
        #   → Coi water_line position = ty le co the bi ngap. SAI vi wl_y
        #     la pixel coordinate, phu hoan toan vao goc camera.
        #   Vi du: water line giua anh → 0.5*170*0.72 = 61cm du nuoc chi 10cm.
        #
        # [FIX v2] Dung perspective-aware non-linear mapping:
        #   - Camera mat duong cham (eye-level, ~150cm): water line o giua
        #     anh ≈ 25-35cm (khong phai 61cm).
        #   - Camera nhin xuong (elevated): water line o giua ≈ 15-20cm.
        #   - Khong biet goc camera → dung non-linear S-curve + cap conservative.
        #
        # [IMPROVE] FOV compensation:
        #   FOV anh huong den muc do "compressed" cua water line position.
        #   - Wide-angle (FOV>90): cung 1 water_line position → nuoc ngan hon
        #     vi objects o bien bi "keo dai" → water line bi "day xuong".
        #   - Tele (FOV<40): cung 1 water_line position → nuoc sau hon
        #     vi objects bi "nén" → water line bi "day len".
        #   → Dieu chinh max_depth theo FOV:
        #     FOV=65 (phone) → max=80cm (baseline)
        #     FOV=100 (wide) → max=60cm (ngan hon)
        #     FOV=35 (tele)  → max=110cm (sau hon)
        water_ratio = max(0.0, (img_h - wl_y) / img_h)  # 0..1

        # FOV-adjusted max depth
        fov_factor = np.clip(1.3 - (fov_deg - 35.0) / 100.0, 0.5, 1.5)
        # FOV=35 → 1.30, FOV=65 → 1.0, FOV=100 → 0.65
        wl_max = 80.0 * fov_factor  # adjusted cap

        # S-curve: 0.5*(1 - cos(pi*x)) — compressed ở 2 đầu (an toan hon)
        wl_cm = wl_max * 0.5 * (1.0 - np.cos(np.pi * water_ratio))
        wl_cm = float(np.clip(wl_cm, 0.0, wl_max))

        # ── Nguon 3: Color area ──────────────────────────────────────
        area_cm = lower_pct * 1.6

        # ── [IMPROVE] Dynamic weight fusion ─────────────────────────
        # Cu: hardcoded weights theo so luong objects (0.70/0.20/0.10...)
        # Moi: tinh weights dong theo chat luong THUC TE cua tung nguon.
        #   - YOLO weight: proportional do confidence + so luong objects
        #   - WL weight: proportional wl_conf
        #   - Area weight: proportional lower_pct (zone confidence)
        # → Nguon tot hon duoc uu tien, khong phai chi dem so object.

        if yolo_cm > 0 and total_w > 0:
            yolo_quality = min(1.0, total_w * 1.5)   # 0.13 object → 0.20, 0.67 → 1.0
        else:
            yolo_quality = 0.0

        wl_quality    = wl_conf                          # 0-1
        area_quality  = min(1.0, lower_pct / 15.0)     # 15% area → 1.0

        raw_w_yolo = yolo_quality * 0.65                 # max 65% YOLO
        raw_w_wl   = wl_quality   * 0.25                 # max 25% water line
        raw_w_area = area_quality * 0.10                 # max 10% color area

        total_raw = raw_w_yolo + raw_w_wl + raw_w_area
        if total_raw > 0.01:
            w_yolo = raw_w_yolo / total_raw
            w_wl   = raw_w_wl   / total_raw
            w_area = raw_w_area / total_raw
        else:
            w_yolo, w_wl, w_area = 0.0, 0.4, 0.6

        water_cm = w_yolo * yolo_cm + w_wl * wl_cm + w_area * area_cm
        conf = min(0.92, wl_conf * 0.35 + yolo_quality * 0.50 + area_quality * 0.15)

        # Aerial correction: tu tren cao → actual water sau hon ve be ngoai
        # angle_factor đã ap dung trong _measure_objects_local roi,
        # nhung neu khong co objects thi ap dung o đây
        if is_aerial and total_w < 0.1:
            water_cm *= angle_factor
            conf     *= 0.7

        # [FIX] Cap khi khong co object tham chieu — SOFT cap theo wl_conf
        # Cu: hard cap 70cm → sai khi nuoc that su sau (anh chup tu tang cao,
        # khong thay nguoi/xe nhung nuoc van sau).
        # Moi: wl_conf cao (>0.6) → cho phep water_cm cao hon.
        #       wl_conf thap (<0.4) → cap chat hon vi khong chac.
        if not measured and water_cm > 0:
            max_depth = 40.0 + wl_conf * 80.0   # wl_conf=0.3 → 64cm, wl_conf=0.7 → 96cm
            if water_cm > max_depth:
                water_cm = max_depth
                conf    *= 0.55

        # Minimum: neu has_flood thi it nhat la PUDDLE
        if has_flood and water_cm < 2.0:
            water_cm = max(water_cm, lower_pct * 0.5)

        water_cm = max(0.0, water_cm)
        level, desc = classify_level(water_cm)
        return round(water_cm, 1), round(conf, 3), level, desc

    # ─── Draw overlay ────────────────────────────────────────────────────────
    def _draw_overlay(self, img_bgr, dets, measured, wl_y, water_cm, level, original, vehicles=None):
        """
        Ve overlay thong tin:
        - Header 2 dong (mau theo muc do lu)
        - Duong water line (cyan, co shadow)
        - Bbox vat the (mau rieng, co shadow)
        - Label: "{class} | {ref_height}cm ({visibility}% visible)"
          * ref_height la chieu cao THUC (170 cho person, KHONG phu thuoc pose)
          * visibility = ti le phan nhin thay
        - Arrow tu water line xuong chan (chi khi bi ngap >= 5% chieu cao)
        - Smart label placement: tranh che nhau
        """
        ov   = img_bgr.copy()
        h, w = ov.shape[:2]
        sc   = max(0.5, min(1.5, w/1280.0))
        thick     = max(1, round(sc*1.8))
        lbl_fs    = max(0.36, sc*0.44)
        hdr_fs    = max(0.55, sc*0.80)
        arr_thick = max(1, round(sc*1.5))

        # Water tint
        if wl_y < h:
            tint = ov.copy()
            cv2.rectangle(tint, (0,wl_y), (w,h), (255,180,30), -1)
            cv2.addWeighted(tint, 0.15, ov, 0.85, 0, ov)
            cv2.line(ov, (0,wl_y), (w,wl_y), (0,0,0),     thick+2)
            cv2.line(ov, (0,wl_y), (w,wl_y), (0,215,255), thick)
            lbl = f"Water line ~{water_cm:.0f} cm"
            (tw,th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, sc*0.60, thick)
            ly = max(wl_y-8, th+6)
            cv2.putText(ov, lbl, (12,ly+1), cv2.FONT_HERSHEY_SIMPLEX, sc*0.60, (0,0,0),     thick+1, cv2.LINE_AA)
            cv2.putText(ov, lbl, (12,ly),   cv2.FONT_HERSHEY_SIMPLEX, sc*0.60, (0,215,255), thick,   cv2.LINE_AA)

        meas_map  = {tuple(o.bbox): o for o in measured}
        used_rects = []

        for det in dets:
            x1,y1,x2,y2 = det["bbox"]
            color = det.get("color_bgr", (128,128,128))
            obj   = meas_map.get((x1,y1,x2,y2))

            cv2.rectangle(ov, (x1,y1), (x2,y2), (0,0,0), thick+2)
            cv2.rectangle(ov, (x1,y1), (x2,y2), color,   thick)

            # Arrow tu water line cuc bo xuong chan (chi khi thuc su ngap)
            arrow_wl = obj.local_wl_y if obj and obj.local_wl_conf > 0.20 else wl_y
            if obj and obj.pixels_below_water > max(5, obj.pixels_total*0.05) and arrow_wl < y2:
                cx = (x1+x2)//2
                cv2.arrowedLine(ov, (cx, arrow_wl), (cx, y2), (0,0,0),     arr_thick+2, tipLength=0.20)
                cv2.arrowedLine(ov, (cx, arrow_wl), (cx, y2), (0,255,255), arr_thick,   tipLength=0.20)

            # Label: hien thi ref_height + visibility + body part neu co
            if obj:
                vis_pct = int(obj.visibility_ratio * 100)
                # Them body part contact vao label
                # Body part labels - ASCII để cv2 render được tren moi he thong
                part_str = {
                    "ankle":     " [ankle]",
                    "knee":      " [knee]",
                    "waist":     " [waist]",
                    "chest":     " [chest]",
                    "submerged": " [submerged]",
                    "none":      "",
                }.get(obj.body_part_contact, "")
                lbl = f"{det['class_name']} | {int(obj.ref_height_cm)}cm ({vis_pct}% visible){part_str}"
            else:
                lbl = det["class_name"]

            (tw,th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, lbl_fs, 1)
            pad = 4

            # Smart placement: thu phia tren truoc, neu bi che hoac bi cat thi dat phia duoi
            def overlaps(r):
                return any(not(r[2]<ur[0] or r[0]>ur[2] or r[3]<ur[1] or r[1]>ur[3]) for ur in used_rects)

            lx = max(0, min(w-tw-pad*2-2, x1))
            ly = y1 - pad - 2
            rect_top = (lx, ly-th-pad, lx+tw+pad*2, ly+pad//2)

            if overlaps(rect_top) or ly-th-pad < 0:
                ly    = y2 + th + pad + 2
                rect_top = (lx, y2+2, lx+tw+pad*2, ly+pad)

            ly = max(th+pad+2, min(h-2, ly))
            cv2.rectangle(ov, (lx,ly-th-pad), (lx+tw+pad*2,ly+pad//2), (0,0,0), -1)
            cv2.putText(ov, lbl, (lx+pad,ly), cv2.FONT_HERSHEY_SIMPLEX, lbl_fs, color, 1, cv2.LINE_AA)
            used_rects.append((lx,ly-th-pad,lx+tw+pad*2,ly+pad//2))

        # Header
        lc = LEVEL_COLOR_BGR.get(level, (200,200,200))
        h1 = f"FLOOD LEVEL: {level}"
        h2 = f"Water depth: ~{water_cm:.0f} cm"
        (h1w,h1h),_ = cv2.getTextSize(h1, cv2.FONT_HERSHEY_SIMPLEX, hdr_fs,     thick)
        (h2w,h2h),_ = cv2.getTextSize(h2, cv2.FONT_HERSHEY_SIMPLEX, hdr_fs*0.88, thick)
        bar_w = max(h1w,h2w)+24; bar_h = h1h+h2h+28
        ov2 = ov.copy()
        cv2.rectangle(ov2,(0,0),(bar_w,bar_h),(0,0,0),-1)
        cv2.addWeighted(ov2,0.75,ov,0.25,0,ov)
        cv2.putText(ov, h1, (10,h1h+8),        cv2.FONT_HERSHEY_SIMPLEX, hdr_fs,      lc,         thick, cv2.LINE_AA)
        cv2.line(ov,(8,h1h+14),(bar_w-8,h1h+14),(80,80,80),1)
        cv2.putText(ov, h2, (10,h1h+h2h+20),   cv2.FONT_HERSHEY_SIMPLEX, hdr_fs*0.88, (0,215,255), thick, cv2.LINE_AA)

        # ── Vẽ xe (ô tô + xe máy) lên overlay ──────────────────────────────
        if vehicles:
            try:
                from depth_analysis.vehicle_detector import VehicleDetector
                VehicleDetector.draw_vehicles(ov, vehicles, scale=sc)
            except Exception as _vde:
                log.debug(f"  [Vehicle draw] {_vde}")

        dest = self.output_dir / f"{original.stem}_overlay.jpg"
        cv2.imwrite(str(dest), ov, [cv2.IMWRITE_JPEG_QUALITY, 94])
        return dest

    def _save_depth_colormap(self, depth_norm, original):
        dest = self.output_dir / f"{original.stem}_depth.png"
        cv2.imwrite(str(dest), cv2.applyColorMap((depth_norm*255).astype(np.uint8), cv2.COLORMAP_TURBO))
        return dest
