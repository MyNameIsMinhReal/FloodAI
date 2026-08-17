# -*- coding: utf-8 -*-
"""Shared constants across flood_pipeline modules."""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DOCUMENT_EXTENSIONS = {".csv", ".html", ".json", ".xlsx"}
DRIVE_UPLOAD_EXTENSIONS = IMAGE_EXTENSIONS.union(DOCUMENT_EXTENSIONS)

DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_POSE_MODEL = "yolov8n-pose.pt"
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_DINO_MODEL = "facebook/dinov2-small"
DEFAULT_FLOOD_RESNET_MODEL = "models/flood_resnet/model_flood_v20.pth"

# Label map cho flood ResNet-18 (index → nghĩa)
# Class 0 = dry/no_flood, Class 1 = flood, Class 2 = heavy_flood
FLOOD_RESNET_LABELS = ["dry", "flood", "heavy_flood"]

DEFAULT_YOLO_MODELS = [DEFAULT_YOLO_MODEL, "yolov8s.pt", "yolov8m.pt"]
DEFAULT_DEPTH_MODELS = [
    DEFAULT_DEPTH_MODEL,
    "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything/Depth-Anything-V2-Large-hf",
]

# Flood level labels
FLOOD_LEVEL_KNEE = "Ngập đầu gối (40-70cm)"
FLOOD_LEVEL_HIP = "Ngập ngang hông (70-120cm)"
FLOOD_LEVEL_CHEST = "Ngập ngang ngực (120-200cm)"
FLOOD_LEVEL_COMPLETE = "Ngập hoàn toàn (>200cm)"

# ── Flood level classification (dung chung) ───────────────────────────────────
# Truoc day: classify_level() duoc dinh nghia TRUNG LAP o ca
# reference_estimator.py va measurement_engine.py → maintenance hazard.
# Moi: dinh nghia 1 lan o constants.py, import tu ca 2 noi.

FLOOD_LEVEL_THRESHOLDS = [
    (0,    0,   "NO_FLOOD",  "Không có lũ"),
    (0,   15,   "PUDDLE",    "Vũng nước nhỏ (<15cm)"),
    (15,  40,   "ANKLE",     "Ngập mắt cá chân (15-40cm)"),
    (40,  70,   "KNEE",      FLOOD_LEVEL_KNEE),
    (70, 120,   "WAIST",     FLOOD_LEVEL_HIP),
    (120, 200,  "CHEST",     FLOOD_LEVEL_CHEST),
    (200, 9999, "SUBMERGED", FLOOD_LEVEL_COMPLETE),
]

FLOOD_LEVEL_RANGES = {
    "NO_FLOOD": "0 cm",
    "PUDDLE":   "0-15 cm",
    "ANKLE":    "15-40 cm",
    "KNEE":     "40-70 cm",
    "WAIST":    "70-120 cm",
    "CHEST":    "120-200 cm",
    "SUBMERGED": ">200 cm",
}

def classify_level(water_cm: float) -> tuple:
    """Phan loai muc nuoc theo cm. Tra ve (level, description)."""
    for lo, hi, lvl, desc in FLOOD_LEVEL_THRESHOLDS:
        if lo <= water_cm < hi:
            return lvl, desc
    return "SUBMERGED", "Ngập hoàn toàn (>200cm)"

def level_range(level: str) -> str:
    """Tra ve chuoi khoang cm cho muc lu."""
    return FLOOD_LEVEL_RANGES.get(level, "N/A")

DEFAULT_QUERY = "flood disaster 2024"

# Logging messages
LOG_OK = "  [OK] "
LOG_SKIP = "  [SKIP] "
LOG_DOWNLOADED = "  Downloaded "
LOG_IMAGES = " images -> "
LOG_XONG = "  Xong: "
LOG_BLUR_FILTER = "  Blur filter: "
LOG_BANNER_FILTER = "  Banner filter: "
LOG_QUALITY_IMAGES = " ảnh sau khi lọc"
LOG_CM_CONF = "cm conf="
LOG_SCORE = " score="
LOG_STREETVIEW_DB = "streetview_db.npz"
LOG_VIETNAM = " Vietnam"
LOG_FLOOD_PIPELINE = "FloodPipeline/1.0"
LOG_HANOI = "Ha Noi"
LOG_HO_CHI_MINH = "Ho Chi Minh"
LOG_HAI_PHONG = "Hai Phong"
LOG_VINH_LONG = "Vinh Long"
LOG_GOOGLE_MAPS_DQ = "https://maps.google.com/đq="
LOG_COLOR = "#4ade80"

# Output folder names
OUTPUT_FOLDER_RAW = "00_raw"
OUTPUT_FOLDER_ORIGINAL = "original_images"
OUTPUT_FOLDER_OVERLAY = "depth_overlays"
OUTPUT_FOLDER_DEPTHMAP = "depth_maps"
OUTPUT_FOLDER_WATERMARK_CLEANED = "watermark_cleaned"
OUTPUT_FOLDER_TMP_DEPTH = "_tmp_depth"

# Output file names
REPORT_CSV = "flood_analysis_report.csv"
REPORT_HTML = "flood_analysis_report.html"
REPORT_XLSX = "flood_analysis_report.xlsx"
PIPELINE_SUMMARY_JSON = "pipeline_summary.json"

# ── [IMPROVE] Wide-angle lens undistortion ──────────────────────────────────────
# Smartphone wide-angle (FOV > 85°) co barrel distortion → object size bi meo
# o bien anh → water height estimate sai.
# Utility nay undistort anh truoc khi measure.
# Chi can thuc hien khi FOV > 85° (smartphone ultra-wide).

import cv2
import numpy as np

# Default distortion coefficients cho smartphone wide-angle
# (estimation — can calibrate cho tung dong dien thoai)
DEFAULT_CAMERA_MATRIX = np.array([
    [600.0,   0.0, 320.0],
    [  0.0, 600.0, 240.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float32)

DEFAULT_DIST_COEFFS = np.array([-0.25, 0.05, 0.0, 0.0], dtype=np.float32)

def undistort_wide_angle(img_bgr: np.ndarray, fov_deg: float = 65.0,
                          focal_length_px: float = 600.0) -> np.ndarray:
    """
    Undistort anh wide-angle neu FOV > 85°.

    Args:
        img_bgr:          anh BGR
        fov_deg:          FOV estimate (degrees)
        focal_length_px:  focal length in pixels (uoc luong tu FOV)

    Returns:
        anh da undistort (neu FOV > 85), nguoc lai tra ve goc.
    """
    if fov_deg <= 85.0:
        return img_bgr  # khong can undistort

    h, w = img_bgr.shape[:2]

    # Tinh camera matrix tu FOV + kich thuoc anh
    # focal_length_px = w / (2 * tan(fov/2))
    if focal_length_px <= 0:
        focal_length_px = w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))

    camera_matrix = np.array([
        [focal_length_px, 0.0, w / 2.0],
        [0.0, focal_length_px, h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    # Distortion coefficient — estimation tu FOV
    # Wider FOV → larger negative k1 (barrel distortion)
    k1 = -0.3 * (fov_deg - 85.0) / 40.0  # FOV=125 → k1=-0.3
    dist_coeffs = np.array([k1, 0.02, 0.0, 0.0], dtype=np.float32)

    # Undistort
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(
        img_bgr, camera_matrix, dist_coeffs, None, new_camera_matrix
    )

    return undistorted
