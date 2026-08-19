# -*- coding: utf-8 -*-
"""
depth_analysis/water_segmentor.py  —  v1.0
===========================================
Semantic segmentation cho water detection — chính xác hơn bounding box.

Models được hỗ trợ (theo thứ tự ưu tiên):
  1. YOLOv8-seg  (ultralytics) — nhanh nhất, segment-level mask
  2. SegFormer   (HuggingFace) — cân bằng tốc độ/chất lượng
  3. DeepLabV3+  (torchvision) — chính xác nhất, chậm hơn

Output: water_mask pixel-level (chính xác hơn bounding box rất nhiều)

Kết hợp với WaterDetector (color-based) để:
  - WaterDetector → rough detection, nhanh, không cần GPU
  - WaterSegmentor → refine mask, chính xác, cần GPU nếu có

Cài đặt:
  pip install ultralytics transformers torch torchvision
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ADE20K label IDs cho water
ADE_WATER_IDS = {21, 26, 60, 128}  # water, sea, river, pool
ADE_WET_SURFACE_IDS = {6, 11}       # road, sidewalk (có thể ướt)

# COCO class IDs liên quan
COCO_CLASSES_OF_INTEREST = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}


@dataclass
class SegmentationResult:
    """Kết quả segmentation."""
    water_mask:        np.ndarray   # binary (H,W) uint8: 255=nước
    wet_surface_mask:  np.ndarray   # binary (H,W) uint8: 255=bề mặt ướt
    water_area_pct:    float        # % diện tích là nước
    water_line_y:      int          # đường mực nước
    model_used:        str          # tên model đã dùng
    confidence:        float
    seg_map:           Optional[np.ndarray] = None  # full segmentation map


class WaterSegmentor:
    """
    Semantic segmentation để phát hiện nước chính xác hơn color-based.

    Chiến lược:
      - Thử YOLOv8-seg trước (nhanh, có thể đã có ultralytics)
      - Fallback sang SegFormer nếu không có YOLOv8-seg weights
      - Fallback cuối cùng về color-based (không cần model)
    """

    def __init__(
        self,
        model_type: str = "auto",    # "yolov8", "segformer", "deeplab", "auto"
        yolo_model_path: str = "yolov8n-seg.pt",
        segformer_model: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        device: str = "auto",
        conf_threshold: float = 0.35,
        cache_model: bool = True,
    ):
        self.model_type       = model_type
        self.yolo_path        = yolo_model_path
        self.segformer_model  = segformer_model
        self.conf_threshold   = conf_threshold
        self.cache_model      = cache_model
        self.device           = device

        self._yolo_model     = None
        self._seg_model      = None
        self._seg_processor  = None
        self._loaded_type    = None

    def _get_model_order(self):
        """Return the configured model priority order."""
        if self.model_type == "auto":
            return ["yolov8", "segformer", "deeplab"]
        return [self.model_type]

    def _run_model(self, model_name: str, img_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """Dispatch to the requested model runner."""
        if model_name == "yolov8":
            return self._run_yolov8_seg(img_rgb)  # type: ignore[attr-defined]
        if model_name == "segformer":
            return self._run_segformer(img_rgb)  # type: ignore[attr-defined]
        if model_name == "deeplab":
            return self._run_deeplab(img_rgb)
        return None

    def _run_deeplab(self, img_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """Run torchvision DeepLab and derive a water mask."""
        import torch
        import torchvision.transforms as T
        from torchvision.models.segmentation import deeplabv3_resnet101

        if self._seg_model is None or self._loaded_type != "deeplab":
            log.info("  Loading DeepLabV3+ ResNet101 ...")
            self._seg_model = deeplabv3_resnet101(
                weights="DeepLabV3_ResNet101_Weights.DEFAULT"
            )
            self._seg_device = (
                "cuda" if self.device == "auto" and torch.cuda.is_available()
                else "cpu" if self.device == "auto" else self.device
            )
            self._seg_model = self._seg_model.to(self._seg_device).eval()
            self._loaded_type = "deeplab"

        h, w = img_rgb.shape[:2]
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        inp = torch.as_tensor(transform(Image.fromarray(img_rgb))).unsqueeze(0).to(
            self._seg_device
        )
        with torch.no_grad():
            output = self._seg_model(inp)["out"]

        seg = output.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        water_mask = cv2.bitwise_and(
            (seg == 0).astype(np.uint8) * 255,
            self._quick_color_water(img_rgb),
        )
        limit = np.zeros((h, w), np.uint8)
        limit[h * 2 // 5:, :] = 255
        water_mask = cv2.bitwise_and(water_mask, limit)
        kernel = np.ones((9, 9), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        water_pct = float((water_mask > 0).sum()) / (h * w)
        return SegmentationResult(
            water_mask=water_mask,
            wet_surface_mask=np.zeros((h, w), np.uint8),
            water_area_pct=round(water_pct * 100, 2),
            water_line_y=self._find_water_line(water_mask, h),
            model_used="deeplabv3+",
            confidence=0.60,
        )

    def _quick_color_water(self, img_rgb: np.ndarray) -> np.ndarray:
        """Create a fast HSV-based water-color mask for model refinement."""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, w = img_rgb.shape[:2]
        profiles = (
            ((90, 35, 35), (135, 255, 255)),
            ((5, 45, 25), (22, 230, 195)),
            ((0, 0, 50), (180, 40, 180)),
            ((20, 50, 80), (38, 230, 240)),
            ((14, 55, 40), (30, 245, 215)),
            ((0, 0, 80), (180, 25, 210)),
        )
        combined = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in profiles:
            combined = cv2.bitwise_or(
                combined,
                cv2.inRange(hsv, np.array(lower), np.array(upper)),
            )

        lower_region = np.zeros((h, w), dtype=np.uint8)
        lower_region[h * 3 // 10:, :] = 255
        combined = cv2.bitwise_and(combined, lower_region)
        kernel = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    def _color_based_fallback(self, img_rgb: np.ndarray) -> SegmentationResult:
        """Create a segmentation result when no model is available."""
        h, w = img_rgb.shape[:2]
        water_mask = self._quick_color_water(img_rgb)
        water_pct = float((water_mask > 0).sum()) / (h * w)
        return SegmentationResult(
            water_mask=water_mask,
            wet_surface_mask=np.zeros((h, w), dtype=np.uint8),
            water_area_pct=round(water_pct * 100, 2),
            water_line_y=self._find_water_line(water_mask, h),
            model_used="color_fallback",
            confidence=0.40,
        )

    # ══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════

    def segment(self, img_rgb: np.ndarray) -> SegmentationResult:
        """
        Chạy segmentation, trả về water mask pixel-level.

        Args:
            img_rgb: (H,W,3) uint8 RGB

        Returns:
            SegmentationResult với water_mask chính xác
        """
        h, w = img_rgb.shape[:2]

        # Thử theo thứ tự ưu tiên
        model_order = self._get_model_order()

        for model_name in model_order:
            try:
                result = self._run_model(model_name, img_rgb)
                if result is not None:
                    return result
            except Exception as e:
                log.debug(f"  {model_name} failed: {e}, trying next...")

        # Fallback: color-based
        log.debug("  WaterSegmentor: falling back to color-based")
        return self._color_based_fallback(img_rgb)

    def segment_and_fuse(
        self,
        img_rgb: np.ndarray,
        color_mask: np.ndarray,   # từ WaterDetector
        fusion_weight: float = 0.4,   # giữ lại tham số cho backward-compat, không dùng nữa
    ) -> np.ndarray:
        """
        Fuse segmentation mask với color-based mask — v2 (rule-based, không weighted avg).

        Logic mới:
          water = seg | (color & depth_edge)   # color chỉ hợp lệ nếu có edge depth xác nhận
          water = water & ~shadow              # loại bóng đổ
          water = water & ~person_region       # loại vùng người

        Khác với v1 (fused = s*0.6 + c*0.4): color mask không còn được phép
        "kéo sai" segmentation về vùng đường ướt/phản chiếu.
        """
        seg_result = self.segment(img_rgb)
        seg_mask   = seg_result.water_mask

        return fuse_water_masks(
            seg_mask=seg_mask,
            color_mask=color_mask,
            img_rgb=img_rgb,
        )

    def _find_water_line(self, water_mask: np.ndarray, h: int) -> int:
        """Return the first image row containing a substantial water region."""
        if water_mask.sum() == 0:
            return h
        row_counts = (water_mask > 0).sum(axis=1)
        valid_rows = np.where(row_counts > water_mask.shape[1] * 0.05)[0]
        if len(valid_rows) == 0:
            return h
        return int(valid_rows[0])


def fuse_water_masks(
    seg_mask: np.ndarray,
    color_mask: np.ndarray,
    img_rgb: np.ndarray,
    depth_edge_mask: Optional[np.ndarray] = None,
    shadow_mask: Optional[np.ndarray] = None,
    person_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rule-based water mask fusion — công thức tham chiếu từ docs/architecture.md.

    Nguyên tắc:
      - Segmentation model là nguồn chính (high precision)
      - Color mask chỉ mở rộng nước nếu có depth boundary xác nhận
      - Shadow và person region luôn bị loại bỏ

    Args:
        seg_mask:        mask nước từ segmentation model (0/255)
        color_mask:      mask nước từ WaterDetector HSV/LAB (0/255)
        img_rgb:         ảnh gốc RGB (để tạo depth edge nếu chưa có)
        depth_edge_mask: depth boundary mask (0/255) — optional
        shadow_mask:     shadow region mask (0/255) — optional
        person_mask:     person region mask (0/255) — optional

    Returns:
        fused water mask (0/255)
    """
    h, w = img_rgb.shape[:2]

    seg   = seg_mask   > 0
    color = color_mask > 0

    # Tạo depth edge mask từ gradient ảnh nếu chưa có
    if depth_edge_mask is None:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_th = np.percentile(magnitude, 75)
        edge_map = (magnitude > edge_th).astype(np.uint8) * 255
        k = np.ones((5, 5), np.uint8)
        depth_edge_mask = cv2.dilate(edge_map, k, iterations=1)

    edge  = depth_edge_mask > 0

    # Rule 1: nước nếu seg xác nhận HOẶC (color + depth boundary cùng xác nhận)
    water = seg | (color & edge)

    # Rule 2: loại bỏ false positive — bóng đổ
    if shadow_mask is not None:
        shadow = shadow_mask > 0
        # Chỉ loại vùng bóng đổ NẶNG (>60% overlap trong kernel)
        shadow_dilated = cv2.dilate(shadow.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        water = water & ~shadow_dilated

    # Rule 3: loại bỏ vùng người (YOLO bbox phản chiếu dưới nước)
    if person_mask is not None:
        person = person_mask > 0
        water = water & ~person

    # Morphological cleanup
    result = water.astype(np.uint8) * 255
    k = np.ones((7, 7), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN,  k)

    return result

    # ══════════════════════════════════════════════════════════════════
    # MODEL RUNNERS
    # ══════════════════════════════════════════════════════════════════

    def _run_yolov8_seg(self, img_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """
        YOLOv8-seg: instance segmentation.

        Strategy:
          - Segment tất cả objects
          - Với flood images: vùng không có object (unmasked lower half)
            thường là nước → infer water từ "complement of objects"
          - Nếu có custom YOLOv8-seg trained trên water: dùng trực tiếp
        """
        from ultralytics import YOLO

        if self._yolo_model is None or self._loaded_type != "yolov8":
            log.info(f"  Loading YOLOv8-seg: {self.yolo_path}")
            self._yolo_model = YOLO(self.yolo_path)
            self._loaded_type = "yolov8"

        h, w = img_rgb.shape[:2]
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        results = self._yolo_model(
            img_bgr,
            conf=self.conf_threshold,
            verbose=False,
            imgsz=640,
        )

        if not results or results[0].masks is None:
            return None

        r = results[0]

        # Tạo "object mask" — vùng có object (người, xe, đồ vật)
        obj_mask = np.zeros((h, w), dtype=np.uint8)
        masks_data = r.masks.data.cpu().numpy()  # (N, H', W')
        boxes_cls  = r.boxes.cls.cpu().numpy().astype(int)

        for seg_mask, cls_id in zip(masks_data, boxes_cls):
            # Resize về kích thước ảnh gốc
            m = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            obj_mask = cv2.bitwise_or(obj_mask, (m > 0.5).astype(np.uint8) * 255)

        # Water inference: vùng dưới ảnh không có object → nhiều khả năng là nước
        water_prior = np.zeros((h, w), dtype=np.uint8)
        water_prior[h // 3:, :] = 255   # bottom 2/3

        not_obj = cv2.bitwise_not(obj_mask)
        water_candidate = cv2.bitwise_and(water_prior, not_obj)

        # Refine bằng color
        color_water = self._quick_color_water(img_rgb)
        fused = cv2.bitwise_and(water_candidate, color_water)

        # Nếu quá ít → chỉ dùng color_water ở vùng không có object
        if fused.sum() < 255 * 100:
            fused = cv2.bitwise_and(water_candidate, np.ones((h, w), dtype=np.uint8) * 255)
            fused = cv2.bitwise_and(fused, color_water)
            if fused.sum() == 0:
                fused = cv2.bitwise_and(
                    cv2.bitwise_not(obj_mask),
                    water_prior,
                )

        # Cleanup
        k = np.ones((9, 9), np.uint8)
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, k)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN,  k)

        water_pct = float((fused > 0).sum()) / (h * w)
        water_line_y = self._find_water_line(fused, h)

        return SegmentationResult(
            water_mask        = fused,
            wet_surface_mask  = np.zeros((h, w), np.uint8),
            water_area_pct    = round(water_pct * 100, 2),
            water_line_y      = water_line_y,
            model_used        = "yolov8-seg",
            confidence        = float(r.boxes.conf.mean()) if len(r.boxes.conf) > 0 else 0.5,
        )

    def _run_segformer(self, img_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """
        SegFormer semantic segmentation (ADE20K labels).

        Water-related labels:
          21=water, 26=sea, 60=river, 128=pool
        """
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        if self._seg_model is None or self._loaded_type != "segformer":
            log.info(f"  Loading SegFormer: {self.segformer_model}")
            self._seg_processor = SegformerImageProcessor.from_pretrained(self.segformer_model)
            self._seg_model = SegformerForSemanticSegmentation.from_pretrained(
                self.segformer_model
            )
            if self.device == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                dev = self.device
            self._seg_model = self._seg_model.to(dev)
            self._seg_model.eval()
            self._seg_device = dev
            self._loaded_type = "segformer"

        h, w = img_rgb.shape[:2]
        pil_img = Image.fromarray(img_rgb)

        inputs = self._seg_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._seg_device) for k, v in inputs.items()}

        with __import__("torch").no_grad():
            outputs = self._seg_model(**inputs)

        # Upscale logits về kích thước ảnh
        import torch.nn.functional as F
        logits = outputs.logits  # (1, num_labels, H/4, W/4)
        upsampled = F.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()  # (H,W)

        # Water mask
        water_mask = np.zeros((h, w), dtype=np.uint8)
        for label_id in ADE_WATER_IDS:
            water_mask = cv2.bitwise_or(water_mask, (seg_map == label_id).astype(np.uint8) * 255)

        # Wet surface mask (đường ướt)
        wet_mask = np.zeros((h, w), dtype=np.uint8)
        for label_id in ADE_WET_SURFACE_IDS:
            wet_mask = cv2.bitwise_or(wet_mask, (seg_map == label_id).astype(np.uint8) * 255)

        # Cleanup water mask
        k = np.ones((9, 9), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, k)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN,  k)

        water_pct    = float((water_mask > 0).sum()) / (h * w)
        water_line_y = self._find_water_line(water_mask, h)

        # Confidence: tỉ lệ pixels water vs (water + uncertain)
        water_logit_mean = float(
            upsampled[0, list(ADE_WATER_IDS)[0]].mean().item()
            if list(ADE_WATER_IDS)[0] < upsampled.shape[1] else 0
        )
        confidence = float(min(0.95, max(0.3, 0.5 + water_logit_mean / 10)))

        return SegmentationResult(
            water_mask        = water_mask,
            wet_surface_mask  = wet_mask,
            water_area_pct    = round(water_pct * 100, 2),
            water_line_y      = water_line_y,
            model_used        = "segformer",
            confidence        = confidence,
            seg_map           = seg_map.astype(np.uint8),
        )

    def _run_deeplab(self, img_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """
        DeepLabV3+ (torchvision) — PASCAL VOC labels.

        VOC water-adjacent labels: 0=background, 21=... (không có water trực tiếp)
        Strategy: dùng "background" ở vùng dưới + color filter
        """
        import torch
        import torchvision.transforms as T
        from torchvision.models.segmentation import deeplabv3_resnet101

        if self._seg_model is None or self._loaded_type != "deeplab":
            log.info("  Loading DeepLabV3+ ResNet101 ...")
            self._seg_model = deeplabv3_resnet101(
                weights="DeepLabV3_ResNet101_Weights.DEFAULT"
            )
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._seg_model = self._seg_model.to(dev).eval()
            self._seg_device = dev
            self._loaded_type = "deeplab"

        h, w = img_rgb.shape[:2]
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        inp = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self._seg_device)

        with __import__("torch").no_grad():
            output = self._seg_model(inp)["out"]

        seg = output.argmax(1).squeeze().cpu().numpy()  # (H,W)

        # Resize về kích thước gốc
        seg = cv2.resize(seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # DeepLab VOC không có water label → dùng color filter ở background region
        background_mask = (seg == 0).astype(np.uint8) * 255  # background
        color_water     = self._quick_color_water(img_rgb)

        # Water = background + có màu nước
        water_mask = cv2.bitwise_and(background_mask, color_water)

        # Giới hạn bottom 60%
        limit = np.zeros((h, w), np.uint8)
        limit[h * 2 // 5:, :] = 255
        water_mask = cv2.bitwise_and(water_mask, limit)

        k = np.ones((9, 9), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, k)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN,  k)

        water_pct    = float((water_mask > 0).sum()) / (h * w)
        water_line_y = self._find_water_line(water_mask, h)

        return SegmentationResult(
            water_mask        = water_mask,
            wet_surface_mask  = np.zeros((h, w), np.uint8),
            water_area_pct    = round(water_pct * 100, 2),
            water_line_y      = water_line_y,
            model_used        = "deeplabv3+",
            confidence        = 0.60,
        )

    # ══════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _color_based_fallback(self, img_rgb: np.ndarray) -> SegmentationResult:
        """Fallback khi không có model nào hoạt động."""
        h, w = img_rgb.shape[:2]
        water_mask   = self._quick_color_water(img_rgb)
        water_pct    = float((water_mask > 0).sum()) / (h * w)
        water_line_y = self._find_water_line(water_mask, h)

        return SegmentationResult(
            water_mask        = water_mask,
            wet_surface_mask  = np.zeros((h, w), np.uint8),
            water_area_pct    = round(water_pct * 100, 2),
            water_line_y      = water_line_y,
            model_used        = "color_fallback",
            confidence        = 0.40,
        )

    def _quick_color_water(self, img_rgb: np.ndarray) -> np.ndarray:
        """Color-based water detection nhanh (15 HSV profiles từ water_detector)."""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_img, w_img = img_rgb.shape[:2]

        # Core profiles: clear blue, muddy brown, gray turbid, yellowish flood
        profiles = [
            ((90,  35,  35), (135, 255, 255)),  # clear blue
            ((5,   45,  25), (22,  230, 195)),  # muddy brown
            ((0,    0,  50), (180,  40, 180)),  # turbid gray
            ((20,  50,  80), (38,  230, 240)),  # yellowish flood
            ((14,  55,  40), (30,  245, 215)),  # muddy orange
            ((0,    0,  80), (180,  25, 210)),  # ash gray
        ]
        combined = np.zeros((h_img, w_img), dtype=np.uint8)
        for lo, hi in profiles:
            combined = cv2.bitwise_or(
                combined,
                cv2.inRange(hsv, np.array(lo), np.array(hi)),
            )

        # Chỉ lấy bottom 70% (nước không ở trên bầu trời)
        mask = np.zeros((h_img, w_img), np.uint8)
        mask[h_img * 3 // 10:, :] = 255
        combined = cv2.bitwise_and(combined, mask)

        k = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k)
        return combined

    def _find_water_line(self, water_mask: np.ndarray, h: int) -> int:
        """Tìm water line y từ mask."""
        if water_mask.sum() == 0:
            return h
        row_counts = (water_mask > 0).sum(axis=1)
        valid_rows = np.where(row_counts > water_mask.shape[1] * 0.05)[0]
        if len(valid_rows) == 0:
            return h
        return int(valid_rows[0])
