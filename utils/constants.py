# -*- coding: utf-8 -*-
"""Shared constants across flood_pipeline modules."""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DOCUMENT_EXTENSIONS = {".csv", ".html", ".json", ".xlsx"}
DRIVE_UPLOAD_EXTENSIONS = IMAGE_EXTENSIONS.union(DOCUMENT_EXTENSIONS)

DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_POSE_MODEL = "yolov8n-pose.pt"
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_DINO_MODEL = "facebook/dinov2-small"

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
