# -*- coding: utf-8 -*-
"""
depth_analysis/water_segmentor.py  —  v2.0
===========================================
Semantic segmentation cho water detection — chính xác hơn bounding box.

Models được hỗ trợ (theo thứ tự ưu tiên):
  1. SegFormer (fine-tuned 2-class flood hoặc ADE20K pretrained)
  2. YOLOv8-seg (ultralytics) — nhanh, segment-level mask
  3. DeepLabV3+ (torchvision) — fallback

Output: water_mask pixel-level (chính xác hơn bounding box rất nhiều)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ADE20K label IDs cho water
ADE_WATER_IDS = {21, 26, 60, 128}   # water, sea, river, pool
ADE_WET_SURFACE_IDS = {6, 11}       # road, sidewalk (có thể ướt)
ADE_DRY_ROAD_IDS = {6, 11, 3, 13}   # road, sidewalk, floor, ground

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
    wet_surface_mask:  np.ndarray   # binary (H,W) uint8: 255=bề mặt đường/ướt
    water_area_pct:    float        # % diện tích là nước (0-100)
    water_line_y:      int          # đường mực nước
    model_used:        str          # tên model đã dùng
    confidence:        float
    seg_map:           Optional[np.ndarray] = None  # full segmentation map
    is_finetuned:      bool = False                 # True nếu dùng SegFormer fine-tuned
    confident_bg_mask: Optional[np.ndarray] = None  # binary (H,W) uint8: 255=chắc chắn không phải nước (fine-tuned)

    @property
    def water_pct(self) -> float:
        return float(self.water_area_pct)

    @property
    def dry_road_mask(self) -> Optional[np.ndarray]:
        return self.wet_surface_mask


def resolve_segformer_model_path(
    configured_path: Optional[str] = None,
    fallback: str = "nvidia/segformer-b0-finetuned-ade-512-512",
) -> str:
    """
    Chuẩn hóa đường dẫn model SegFormer:
      - Hỗ trợ prefix 'local:models/flood_segnet_b2'
      - Nếu là path local nhưng chưa tồn tại trên đĩa -> tự fallback về model HuggingFace
    """
    if not configured_path:
        return fallback
    clean = configured_path.strip()
    if clean.startswith("local:"):
        clean = clean[len("local:"):].strip()

    # Kiểm tra xem có phải đường dẫn local không
    p = Path(clean)
    if p.exists():
        return str(p)
    # Nếu là HuggingFace repo ID (như nvidia/segformer-...)
    if clean.startswith(("nvidia/", "openmmlab/", "facebook/")):
        return clean
    # Nếu là path local được cấu hình sẵn nhưng chưa copy model vào -> fallback an toàn
    log.debug(f"  SegFormer local path '{clean}' chưa tồn tại, fallback về '{fallback}'")
    return fallback


class WaterSegmentor:
    """
    Semantic segmentation để phát hiện nước chính xác hơn color-based.

    Chiến lược:
      - Ưu tiên SegFormer (đặc biệt khi có bản fine-tuned 2-class lũ lụt)
      - Fallback sang YOLOv8-seg hoặc DeepLabV3+
      - Fallback cuối cùng về color-based
    """

    def __init__(
        self,
        model_type: str = "auto",    # "segformer", "yolov8", "deeplab", "auto"
        yolo_model_path: str = "yolov8n-seg.pt",
        segformer_model: str = "models/flood_segnet_b2",
        device: str = "auto",
        conf_threshold: float = 0.35,
        cache_model: bool = True,
    ):
        self.model_type       = model_type
        self.yolo_path        = yolo_model_path
        self.segformer_model  = resolve_segformer_model_path(segformer_model)
        self.conf_threshold   = conf_threshold
        self.cache_model      = cache_model
        self.device           = device

        self._yolo_model      = None
        self._seg_model       = None
        self._seg_processor   = None
        self._seg_device      = "cpu"
        self._seg_img_size    = 512
        self._seg_is_binary   = False
        self._seg_use_letterbox = False
        self._seg_water_id    = 1
        self._loaded_type     = None

    @property
    def _model(self):
        return self._seg_model

    @property
    def _processor(self):
        return self._seg_processor

    def _load(self):
        if self.model_type in ("auto", "segformer"):
            self._load_segformer_if_needed()

    def _get_model_order(self):
        """Return the configured model priority order."""
        if self.model_type == "auto":
            return ["segformer", "yolov8", "deeplab"]
        return [self.model_type]

    def _run_model(self, model_name: str, img_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """Dispatch to the requested model runner."""
        if model_name == "segformer":
            return self._run_segformer(img_rgb)
        if model_name == "yolov8":
            return self._run_yolov8_seg(img_rgb)
        if model_name == "deeplab":
            return self._run_deeplab(img_rgb)
        return None

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
        for model_name in self._get_model_order():
            try:
                result = self._run_model(model_name, img_rgb)
                if result is not None:
                    return result
            except Exception as e:
                log.debug(f"  {model_name} failed: {e}, trying next...")

        log.debug("  WaterSegmentor: falling back to color-based")
        return self._color_based_fallback(img_rgb)

    def segment_and_fuse(
        self,
        img_rgb: np.ndarray,
        color_mask: np.ndarray,   # từ WaterDetector
        fusion_weight: float = 0.4,
    ) -> np.ndarray:
        """
        Fuse segmentation mask với color-based mask — v2 (rule-based).
        """
        seg_result = self.segment(img_rgb)
        seg_mask   = seg_result.water_mask

        return fuse_water_masks(
            seg_mask=seg_mask,
            color_mask=color_mask,
            img_rgb=img_rgb,
        )

    # ══════════════════════════════════════════════════════════════════
    # MODEL RUNNERS
    # ══════════════════════════════════════════════════════════════════

    def _load_segformer_if_needed(self):
        if self._seg_model is not None and self._loaded_type == "segformer":
            return
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        model_id = resolve_segformer_model_path(self.segformer_model)
        log.info(f"  Loading SegFormer: {model_id}")
        self._seg_model = SegformerForSemanticSegmentation.from_pretrained(model_id)

        num_labels = getattr(self._seg_model.config, "num_labels", 150)
        self._seg_is_binary = (num_labels <= 5)

        # Xác định ID của class flood_water / water từ config.json
        label2id = getattr(self._seg_model.config, "label2id", {}) or {}
        id2label = getattr(self._seg_model.config, "id2label", {}) or {}
        water_id = 1
        for k, v in label2id.items():
            if "water" in str(k).lower() or "flood" in str(k).lower():
                water_id = int(v)
                break
        else:
            for k, v in id2label.items():
                if "water" in str(v).lower() or "flood" in str(v).lower():
                    water_id = int(k)
                    break
        self._seg_water_id = water_id

        # Kiểm tra finetune_meta.json (nếu train bằng learning/finetune_segmentation.py có letterbox)
        meta_file = Path(model_id) / "finetune_meta.json"
        if meta_file.exists():
            self._seg_use_letterbox = True
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                self._seg_img_size = int(meta.get("img_size", 768))
            except Exception:
                self._seg_img_size = 768
        else:
            self._seg_use_letterbox = False
            try:
                self._seg_processor = SegformerImageProcessor.from_pretrained(model_id)
                if hasattr(self._seg_processor, "size") and hasattr(self._seg_processor.size, "get"):
                    self._seg_img_size = int(self._seg_processor.size.get("height", 512))
            except Exception:
                self._seg_processor = SegformerImageProcessor.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                )

        if self.device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = self.device
        getattr(self._seg_model, "to")(torch.device(dev))
        self._seg_model.eval()
        self._seg_device = dev
        self._loaded_type = "segformer"

    def _run_segformer(self, img_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """
        SegFormer semantic segmentation:
          - Chế độ 1 (Fine-tuned binary/custom, num_labels <= 5):
            + Nếu có preprocessor_config.json (HuggingFace chuẩn 512x512): dùng SegformerImageProcessor
              và nội suy logits trực tiếp về (h, w).
            + Nếu có finetune_meta.json: dùng letterbox giữ nguyên aspect ratio.
            + Class flood_water (id=1) là vùng nước ngập, class background (id=0) là vùng không ngập.
          - Chế độ 2 (Pretrained ADE20K, num_labels == 150):
            Water labels: {21=water, 26=sea, 60=river, 128=pool}
            Road labels : {6=road, 11=sidewalk, 3=floor, 13=ground}
        """
        import torch
        import torch.nn.functional as F

        self._load_segformer_if_needed()
        h, w = img_rgb.shape[:2]

        if self._seg_is_binary:
            wid = self._seg_water_id
            if self._seg_use_letterbox or self._seg_processor is None:
                # ── Nhánh 1a: Letterbox preprocessing (finetune_segmentation.py) ──
                target = self._seg_img_size
                scale = target / max(h, w)
                nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
                resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

                top = (target - nh) // 2
                left = (target - nw) // 2
                canvas = np.zeros((target, target, 3), dtype=np.uint8)
                canvas[top:top + nh, left:left + nw] = resized

                img_f = canvas.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_norm = (img_f - mean) / std
                inp = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self._seg_device)

                with torch.no_grad():
                    outputs = self._seg_model(pixel_values=inp)
                    logits = F.interpolate(
                        outputs.logits, size=(target, target), mode="bilinear", align_corners=False
                    )
                    probs = F.softmax(logits, dim=1)[0, wid]
                    pred = (logits.argmax(dim=1)[0] == wid)

                prob_crop = probs[top:top + nh, left:left + nw].cpu().numpy()
                pred_crop = pred[top:top + nh, left:left + nw].cpu().numpy().astype(np.uint8)

                water_mask = cv2.resize(pred_crop * 255, (w, h), interpolation=cv2.INTER_NEAREST)
                prob_full = cv2.resize(prob_crop, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                # ── Nhánh 1b: HuggingFace SegformerImageProcessor chuẩn (preprocessor_config.json) ──
                pil_img = Image.fromarray(img_rgb)
                inputs = self._seg_processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self._seg_device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._seg_model(**inputs)
                    upsampled = F.interpolate(
                        outputs.logits, size=(h, w), mode="bilinear", align_corners=False
                    )
                    probs = F.softmax(upsampled, dim=1)[0, wid]
                    pred = (upsampled.argmax(dim=1)[0] == wid)

                prob_full = probs.cpu().numpy()
                water_mask = (pred.cpu().numpy().astype(np.uint8)) * 255

            wet_mask = np.zeros((h, w), dtype=np.uint8)
            confident_bg_mask = ((prob_full < 0.12).astype(np.uint8)) * 255
            seg_map = (water_mask > 0).astype(np.uint8)

            k = np.ones((7, 7), np.uint8)
            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, k)
            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, k)

            water_pct = float((water_mask > 0).sum()) / (h * w)
            water_line_y = self._find_water_line(water_mask, h)
            mean_prob = float(prob_full[water_mask > 0].mean()) if (water_mask > 0).any() else float(1.0 - prob_full.mean())
            confidence = float(np.clip(mean_prob, 0.35, 0.98))

            return SegmentationResult(
                water_mask        = water_mask,
                wet_surface_mask  = wet_mask,
                water_area_pct    = round(water_pct * 100, 2),
                water_line_y      = water_line_y,
                model_used        = "segformer_finetuned",
                confidence        = confidence,
                seg_map           = seg_map,
                is_finetuned      = True,
                confident_bg_mask = confident_bg_mask,
            )

        # ── Chế độ Pretrained ADE20K (150 classes) ──
        pil_img = Image.fromarray(img_rgb)
        inputs = self._seg_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._seg_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._seg_model(**inputs)

        logits = outputs.logits
        upsampled = F.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        seg_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

        water_mask = (np.isin(seg_map, list(ADE_WATER_IDS)).astype(np.uint8)) * 255
        wet_mask = (np.isin(seg_map, list(ADE_DRY_ROAD_IDS)).astype(np.uint8)) * 255

        k = np.ones((9, 9), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, k)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, k)

        water_pct = float((water_mask > 0).sum()) / (h * w)
        water_line_y = self._find_water_line(water_mask, h)

        water_logit_mean = float(
            upsampled[0, 21].mean().item() if upsampled.shape[1] > 21 else 0.0
        )
        confidence = float(min(0.95, max(0.3, 0.5 + water_logit_mean / 10)))

        return SegmentationResult(
            water_mask       = water_mask,
            wet_surface_mask = wet_mask,
            water_area_pct   = round(water_pct * 100, 2),
            water_line_y     = water_line_y,
            model_used       = "segformer-ade20k",
            confidence       = confidence,
            seg_map          = seg_map.astype(np.uint8),
            is_finetuned     = False,
        )

    def _run_yolov8_seg(self, img_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """YOLOv8-seg instance segmentation runner."""
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
        obj_mask = np.zeros((h, w), dtype=np.uint8)
        masks_data = r.masks.data.cpu().numpy()

        for seg_mask in masks_data:
            m = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            obj_mask = cv2.bitwise_or(obj_mask, (m > 0.5).astype(np.uint8) * 255)

        water_prior = np.zeros((h, w), dtype=np.uint8)
        water_prior[h // 3:, :] = 255

        not_obj = cv2.bitwise_not(obj_mask)
        water_candidate = cv2.bitwise_and(water_prior, not_obj)

        color_water = self._quick_color_water(img_rgb)
        fused = cv2.bitwise_and(water_candidate, color_water)

        k = np.ones((9, 9), np.uint8)
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, k)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, k)

        water_pct = float((fused > 0).sum()) / (h * w)
        water_line_y = self._find_water_line(fused, h)

        return SegmentationResult(
            water_mask       = fused,
            wet_surface_mask = np.zeros((h, w), np.uint8),
            water_area_pct   = round(water_pct * 100, 2),
            water_line_y     = water_line_y,
            model_used       = "yolov8-seg",
            confidence       = float(r.boxes.conf.mean()) if len(r.boxes.conf) > 0 else 0.5,
        )

    def _run_deeplab(self, img_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """Run torchvision DeepLabV3+ and derive a water mask."""
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
            water_mask       = water_mask,
            wet_surface_mask = np.zeros((h, w), np.uint8),
            water_area_pct   = round(water_pct * 100, 2),
            water_line_y     = self._find_water_line(water_mask, h),
            model_used       = "deeplabv3+",
            confidence       = 0.60,
        )

    # ══════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════

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
            water_mask       = water_mask,
            wet_surface_mask = np.zeros((h, w), np.uint8),
            water_area_pct   = round(water_pct * 100, 2),
            water_line_y     = self._find_water_line(water_mask, h),
            model_used       = "color_fallback",
            confidence       = 0.40,
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
    img_rgb: Optional[np.ndarray] = None,
    depth_edge_mask: Optional[np.ndarray] = None,
    shadow_mask: Optional[np.ndarray] = None,
    person_mask: Optional[np.ndarray] = None,
    dry_road_mask: Optional[np.ndarray] = None,
    depth_map: Optional[np.ndarray] = None,
    weights: Optional[tuple] = None,
    confident_bg_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rule-based water mask fusion — kết hợp Segmentation + Multi-color + Depth Plane.

    Nguyên tắc:
      1. Segmentation model (SegFormer) là nguồn chính (high precision)
      2. Color mask từ WaterDetector (voted ≥2 kênh) được giữ ở vùng đồng phẳng
         theo phương ngang (depth flatness) hoặc có biên xác nhận
      3. Vùng mặt đường khô chắc chắn (dry_road_mask), vùng background tự tin cao từ
         SegFormer fine-tuned (confident_bg_mask), và vùng tường đứng (steep depth gradient)
         được loại trừ khỏi color mask để tránh nhận nhầm đường khô / tường nhà thành nước
      4. Shadow và person region luôn bị loại bỏ
    """
    seg   = seg_mask > 0
    color = color_mask > 0

    # Nếu có depth_map từ Depth Anything V2: lọc mặt phẳng ngang
    if depth_map is not None and depth_map.shape == seg.shape:
        # Mặt nước là mặt phẳng ngang: độ biến thiên depth theo phương ngang (dx) rất nhỏ
        dx = np.abs(cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
        dx_thresh = max(0.08, float(np.percentile(dx, 82)))
        horizontal_plane = dx <= dx_thresh
        color_valid = color & horizontal_plane
    elif depth_edge_mask is not None:
        color_valid = color & (depth_edge_mask > 0)
    elif img_rgb is not None:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        flat_th = np.percentile(magnitude, 80)
        flat_surface = magnitude <= flat_th
        color_valid = color & flat_surface
    else:
        color_valid = color

    # Nếu SegFormer tìm thấy vùng đường khô, loại trừ khỏi color_valid (trừ nơi seg xác nhận là nước)
    if dry_road_mask is not None and dry_road_mask.shape == seg.shape:
        dry_road = dry_road_mask > 0
        color_valid = color_valid & ~dry_road

    # Nếu SegFormer fine-tuned tự tin cao (>88%) một vùng là background (không ngập):
    # - Khi SegFormer đã phát hiện có nước trong ảnh (seg.mean() >= 0.02): loại bỏ toàn bộ
    #   color false positive ở vùng confident_bg_mask.
    # - Khi SegFormer chưa bắt được nước (ảnh tối/ngoài tập train): chỉ chặn color false positive
    #   ở 45% nửa trên ảnh, giữ nửa dưới cho multi-space color + depth plane.
    if confident_bg_mask is not None and confident_bg_mask.shape == seg.shape:
        conf_bg = confident_bg_mask > 0
        if float(seg.mean()) >= 0.02:
            color_valid = color_valid & ~conf_bg
        else:
            h_img = seg.shape[0]
            upper_bg = np.zeros_like(conf_bg)
            upper_bg[:int(h_img * 0.45), :] = conf_bg[:int(h_img * 0.45), :]
            color_valid = color_valid & ~upper_bg

    water = seg | color_valid

    # Loại bỏ bóng đổ nặng
    if shadow_mask is not None and shadow_mask.shape == seg.shape:
        shadow = shadow_mask > 0
        shadow_dilated = cv2.dilate(shadow.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        water = water & (~shadow_dilated | seg)

    # Loại bỏ vùng người
    if person_mask is not None and person_mask.shape == seg.shape:
        person = person_mask > 0
        water = water & ~person

    result = water.astype(np.uint8) * 255
    k = np.ones((7, 7), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, k)

    return result
