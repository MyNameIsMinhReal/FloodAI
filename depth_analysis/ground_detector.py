# -*- coding: utf-8 -*-
"""
depth_analysis/ground_detector.py  [DEPRECATED — dead code, không dùng trong pipeline]
=========================================================================================
Phân biet "duong nhua kho" vs "mat nuoc" bang SegFormer semantic segmentation.

⚠ ĐÃ KHÔNG CÒN DÙNG TRONG PIPELINE. SegFormer đã được tích hợp trực tiếp
  trong ReferenceEstimator._run_segformer() — cung cấp seg_wl_y, seg_road_pct
  và dùng làm water-line hint. Class GroundDetector dưới đây giữ lại để tham
  khảo nếu cần tách riêng logic trong tương lai.

Pipeline hiện tại dùng:
  - ReferenceEstimator._run_segformer()  → SegFormer water/road mask
  - WaterDetector (15 HSV profiles)      → color-based water detection
  - FloodClassifier (DINOv2)             → flood probability
"""
import logging
from dataclasses import dataclass
from typing import Any, Tuple, Optional, Dict
import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ADE20K label indices can quan tam
ADE_LABELS = {
    "road":        6,
    "sidewalk":    11,
    "water":       21,
    "sea":         26,
    "river":       60,
    "ground":      13,
    "field":       29,
    "floor":       3,
}

# Nhom label
DRY_GROUND_IDS  = {6, 11, 3, 13}   # road, sidewalk, floor, ground
WATER_IDS       = {21, 26, 60}      # water, sea, river
VEGETATION_IDS  = {9, 17, 29, 68}   # grass, plant, field, earth


@dataclass
class GroundAnalysisResult:
    seg_map:          Optional[np.ndarray]  # full segmentation map (H,W)
    road_mask:        np.ndarray            # binary mask: 1=duong kho
    water_mask_seg:   np.ndarray            # binary mask: 1=nuoc (tu segmentation)
    road_area_pct:    float                 # % anh la duong kho
    water_area_pct:   float                 # % anh la nuoc (tu seg)
    is_dry_road:      bool                  # True neu > 20% anh la duong kho
    adjusted_water_line_y: int              # water line sau hieu chinh
    dry_confidence:   float                 # do tin cay duong kho
    notes:            str


class GroundDetector:
    """
    Phat hien duong kho bang SegFormer semantic segmentation.
    Tranh nham duong nhua thanh mat nuoc.
    """

    def __init__(
        self,
        model_id:      str   = "nvidia/segformer-b0-finetuned-ade-512-512",
        device:        str   = "auto",
        road_threshold:float = 0.15,   # > 15% duong kho = is_dry_road
        cache_results: bool  = True,
    ):
        self.model_id       = model_id
        self.device         = device
        self.road_threshold = road_threshold
        self.cache_results  = cache_results
        self._processor: Any  = None
        self._model: Any      = None
        self._device_str: str = "cpu"
        self._cache: Dict[str, GroundAnalysisResult] = {}

    # ------------------------------------------------------------------
    def _load(self):
        if self._model:
            return
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        dev = self.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._device_str = dev

        log.info(f"  Loading SegFormer ({self.model_id}) on {dev}...")
        self._processor = SegformerImageProcessor.from_pretrained(self.model_id)
        _seg: Any = SegformerForSemanticSegmentation.from_pretrained(self.model_id)
        self._model = _seg.to(dev)
        self._model.eval()
        log.info("  SegFormer loaded")

    # ------------------------------------------------------------------
    def analyze(
        self,
        img_rgb:      np.ndarray,
        water_line_y: int,
    ) -> GroundAnalysisResult:
        """
        Phân tích anh, tra ve GroundAnalysisResult.

        Args:
            img_rgb:      Anh RGB numpy array
            water_line_y: Water line hien tai (se duoc dieu chinh neu sai)
        """
        h, w = img_rgb.shape[:2]

        # Cache key
        cache_key = f"{id(img_rgb)}_{water_line_y}"
        if self.cache_results and cache_key in self._cache:
            return self._cache[cache_key]

        # Run segmentation
        try:
            self._load()
            seg_map = self._segment(img_rgb)
        except Exception as e:
            log.warning(f"  SegFormer failed: {e}, using color fallback")
            return self._color_fallback(img_rgb, water_line_y, h, w)

        # Tao masks
        road_mask  = np.isin(seg_map, list(DRY_GROUND_IDS)).astype(np.uint8)
        water_mask = np.isin(seg_map, list(WATER_IDS)).astype(np.uint8)

        road_area_pct  = float(road_mask.mean() * 100)
        water_area_pct = float(water_mask.mean() * 100)
        is_dry_road    = road_area_pct > (self.road_threshold * 100)

        # Dieu chinh water_line_y
        adjusted_y, dry_conf = self._adjust_water_line(
            seg_map, road_mask, water_mask, water_line_y, h, w
        )

        notes = self._build_notes(road_area_pct, water_area_pct, is_dry_road, adjusted_y, water_line_y)
        log.info(
            f"  Ground: road={road_area_pct:.1f}% water={water_area_pct:.1f}% "
            f"dry={is_dry_road} wl={water_line_y}->{adjusted_y} conf={dry_conf:.2f}"
        )

        result = GroundAnalysisResult(
            seg_map               = seg_map,
            road_mask             = road_mask,
            water_mask_seg        = water_mask,
            road_area_pct         = round(road_area_pct, 2),
            water_area_pct        = round(water_area_pct, 2),
            is_dry_road           = is_dry_road,
            adjusted_water_line_y = adjusted_y,
            dry_confidence        = round(dry_conf, 3),
            notes                 = notes,
        )

        if self.cache_results:
            self._cache[cache_key] = result

        return result

    # ------------------------------------------------------------------
    def _segment(self, img_rgb: np.ndarray) -> np.ndarray:
        """Chay SegFormer, tra ve segmentation map (H, W) dtype int."""
        import torch

        pil_img = Image.fromarray(img_rgb)
        inputs  = self._processor(images=pil_img, return_tensors="pt")
        inputs  = {k: v.to(self._device_str) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits  = outputs.logits   # (1, num_classes, H/4, W/4)

        # Upsample ve kich thuoc goc
        h, w = img_rgb.shape[:2]
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        seg_map = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()
        return seg_map.astype(np.int32)

    # ------------------------------------------------------------------
    def _adjust_water_line(
        self,
        seg_map:    np.ndarray,
        road_mask:  np.ndarray,
        water_mask: np.ndarray,
        current_y:  int,
        h: int, w: int,
    ) -> Tuple[int, float]:
        """
        Dieu chinh water_line_y dua tren segmentation.

        Logic:
          - Neu phan duoi current_y la DUONG KHO (road) -> water line sai
            -> Day water_line xuong day anh (khong ngap)
          - Neu co NUOC trong seg map -> dung ranh gioi nuoc/duong lam water line
          - Neu ca duong va nuoc deu co -> ranh gioi la water line chinh xac
        """
        # Kiem tra vung duoi water_line co phai duong kho khong
        below_mask = road_mask[current_y:, :]
        below_road_pct = float(below_mask.mean()) if below_mask.size > 0 else 0.0

        # Kiem tra vung tren water_line co nuoc khong
        above_water_mask = water_mask[:current_y, :]
        _ = float(above_water_mask.mean()) if above_water_mask.size > 0 else 0.0  # unused above_water_pct

        # Truong hop 1: Phan duoi water_line la DUONG KHO (>30%)
        # => Water line dang dat SAI (duong bi nham la nuoc)
        if below_road_pct > 0.30:
            true_boundary = self._find_road_water_boundary(
                seg_map, road_mask, water_mask, h, w
            )
            if true_boundary is not None:
                conf = min(0.95, below_road_pct * 1.5)
                log.debug(
                    f"  WL correction: {current_y} -> {true_boundary} "
                    f"(road_below={below_road_pct:.1%})"
                )
                return true_boundary, conf
            else:
                # Khong co nuoc -> khong ngap -> water_line = day anh
                conf = min(0.95, below_road_pct)
                return h, conf

        # Truong hop 2: Duong kho dam net va nuoc it -> chon duong kho
        if road_mask.mean() > 0.18 and water_mask.mean() < 0.08:
            log.debug(
                f"  WL correction: dry road strong (road={road_mask.mean():.1%}, water={water_mask.mean():.1%})"
            )
            return h, min(0.90, road_mask.mean() * 2.0)

        # Truong hop 3: Nuoc cao va duong it -> chon nuoc
        if water_mask.mean() > 0.30 and road_mask.mean() < 0.12:
            boundary = self._find_road_water_boundary(seg_map, road_mask, water_mask, h, w)
            if boundary is not None:
                return boundary, min(0.95, water_mask.mean() * 1.5)

        # Truong hop 4: Co nuoc trong seg map -> dung ranh gioi nuoc
        if water_mask.sum() > (h * w * 0.03):
            boundary = self._find_road_water_boundary(
                seg_map, road_mask, water_mask, h, w
            )
            if boundary is not None and abs(boundary - current_y) < h * 0.2:
                conf = 0.75
                return boundary, conf

        # Truong hop 5: Duong kho it, nuoc it hoac khong xac dinh -> giu nguyen
        return current_y, 0.5

    def _find_road_water_boundary(
        self,
        seg_map:    np.ndarray,
        road_mask:  np.ndarray,
        water_mask: np.ndarray,
        h: int, w: int,
    ) -> Optional[int]:
        """
        Tim y-coordinate cua ranh gioi giua NUOC (tren) va DUONG KHO (duoi).

        Neu co nuoc: tim hang cao nhat cua duong (road bao gio cung o duoi nuoc).
        Neu khong co nuoc: tra ve None (khong ngap).
        """
        # Neu khong co nuoc hoac it nuoc
        if water_mask.sum() < (h * w * 0.02):
            return None

        # Tim hang tren cung cua DUONG KHO
        road_rows = np.where(road_mask.any(axis=1))[0]
        if len(road_rows) == 0:
            return None

        # Boundary = hang cao nhat (nho y nhat) cua road
        road_top_y = int(road_rows[0])

        # Kiem tra: phan tren road_top_y co nuoc khong
        water_above = water_mask[:road_top_y, :].sum()
        if water_above < (road_top_y * w * 0.05):
            # Nuoc it qua o phan tren -> co the khong co lu that
            return None

        return road_top_y

    # ------------------------------------------------------------------
    def _color_fallback(
        self, img_rgb: np.ndarray, water_line_y: int, h: int, w: int
    ) -> GroundAnalysisResult:
        """
        Fallback khi SegFormer khong chay duoc.
        Dung color analysis de phat hien duong nhua.
        """
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        # Duong nhua: sat thap (<45), val trung binh (60-200), texture trung binh
        sat   = hsv[:, :, 1]
        val   = hsv[:, :, 2]
        road_color_mask = (
            (sat < 45) & (val > 55) & (val < 210)
        ).astype(np.uint8)

        # Lam sach mask
        kernel     = np.ones((15, 15), np.uint8)
        road_mask  = cv2.morphologyEx(road_color_mask, cv2.MORPH_CLOSE, kernel)
        road_mask  = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

        # Chi xet phan duoi anh (duong thuong o duoi)
        lower_road = road_mask[h // 2:, :]
        road_pct   = float(lower_road.mean() * 100)
        is_dry     = road_pct > 20.0

        # Neu phan duoi la duong kho, day water_line xuong
        adjusted_y = water_line_y
        dry_conf   = 0.0
        if is_dry and water_line_y < int(h * 0.7):
            road_rows = np.where(road_mask.any(axis=1))[0]
            if len(road_rows) > 0:
                road_top = int(road_rows[0])
                if road_top < water_line_y:
                    adjusted_y = h   # khong co nuoc
                    dry_conf   = min(0.80, road_pct / 100 * 1.5)

        return GroundAnalysisResult(
            seg_map               = None,
            road_mask             = road_mask,
            water_mask_seg        = np.zeros((h, w), dtype=np.uint8),
            road_area_pct         = round(road_pct, 2),
            water_area_pct        = 0.0,
            is_dry_road           = is_dry,
            adjusted_water_line_y = adjusted_y,
            dry_confidence        = round(dry_conf, 3),
            notes                 = f"Color fallback: road={road_pct:.1f}%",
        )

    # ------------------------------------------------------------------
    # 3.2 RANSAC GROUND PLANE ESTIMATION (MỚI)
    # ------------------------------------------------------------------

    def fit_ground_plane_ransac(
        self,
        depth_norm: np.ndarray,
        img_rgb: np.ndarray,
        n_iterations: int = 100,
        inlier_thresh: float = 0.05,
    ) -> dict:
        """
        Fit "mặt đường chuẩn" từ depth map bằng RANSAC.

        Ý tưởng:
          - Mặt đường là một plane phẳng trong không gian depth
          - RANSAC: chọn ngẫu nhiên 3 điểm, fit plane, đếm inliers
          - Plane tốt nhất → ground plane chuẩn
          - Từ đó đo chiều cao nước so với ground plane

        Args:
            depth_norm:    depth map đã normalize (H,W) [0,1]
            img_rgb:       ảnh RGB để lấy color mask làm weight
            n_iterations:  số vòng lặp RANSAC
            inlier_thresh: ngưỡng distance để tính inlier

        Returns:
            dict với:
              plane_normal:    (a,b,c) normal vector của ground plane
              plane_d:         hệ số d trong ax+by+cz=d
              inlier_mask:     (H,W) bool mask - pixels thuộc ground plane
              ground_level_z:  z-value trung bình của mặt đường
              water_above_ground: depth map đã subtract ground plane
              confidence:      độ tin cậy của plane fit
        """
        h, w = depth_norm.shape

        # Chỉ xét vùng dưới 60% ảnh (nơi có đường)
        start_row = h * 2 // 5
        roi_depth = depth_norm[start_row:, :]
        roi_h, roi_w = roi_depth.shape

        # Tạo 3D points (x, y, z) từ depth map
        # x, y = tọa độ ảnh (normalized); z = depth value
        xs = np.linspace(0, 1, roi_w)
        ys = np.linspace(0, 1, roi_h)
        X, Y = np.meshgrid(xs, ys)
        Z = roi_depth.copy()

        # Flatten thành N×3
        pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

        # Color-based weight: ưu tiên pixels có màu đường (xám) → likely ground
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        road_roi = hsv[start_row:, :]
        road_weight = (
            (road_roi[:, :, 1] < 50) &   # low saturation = gray road
            (road_roi[:, :, 2] > 40) &   # not too dark
            (road_roi[:, :, 2] < 220)    # not too bright (sky)
        ).astype(np.float32).flatten()
        road_idx = np.where(road_weight > 0)[0]

        if len(road_idx) < 30:
            # Không đủ road pixels → fallback
            return self._ground_plane_fallback(depth_norm, start_row)

        # RANSAC
        best_inliers = 0
        best_normal  = np.array([0, 0, 1], dtype=np.float64)
        best_d       = float(np.median(Z))
        best_inlier_mask_flat = np.zeros(len(pts), dtype=bool)

        rng = np.random.default_rng(42)
        for _ in range(n_iterations):
            # Sample 3 points từ road candidates
            if len(road_idx) < 3:
                break
            sample_idx = rng.choice(road_idx, 3, replace=False)
            p1, p2, p3 = pts[sample_idx[0]], pts[sample_idx[1]], pts[sample_idx[2]]

            # Fit plane
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue
            normal /= norm_len
            d = float(np.dot(normal, p1))

            # Count inliers (trong toàn bộ roi, không chỉ road_idx)
            distances = np.abs(pts @ normal - d)
            inlier_mask = distances < inlier_thresh
            n_inliers = inlier_mask.sum()

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_normal  = normal.copy()
                best_d       = d
                best_inlier_mask_flat = inlier_mask.copy()

        # Refine: refit plane dùng tất cả inliers
        if best_inliers > 10:
            inlier_pts = pts[best_inlier_mask_flat]
            # SVD-based plane fit
            centroid = inlier_pts.mean(axis=0)
            centered = inlier_pts - centroid
            _, _, Vt = np.linalg.svd(centered)
            best_normal = Vt[-1]  # smallest singular value = normal
            best_d = float(np.dot(best_normal, centroid))

        # Tạo full-image masks
        inlier_mask_roi = best_inlier_mask_flat.reshape(roi_h, roi_w)
        full_inlier_mask = np.zeros((h, w), dtype=bool)
        full_inlier_mask[start_row:, :] = inlier_mask_roi

        # Tính ground level z (mặt đường = z của plane tại mỗi điểm)
        xs_full = np.linspace(0, 1, w)
        ys_full = np.linspace(0, 1, h)
        Xf, Yf = np.meshgrid(xs_full, ys_full)
        # Ground z = (d - a*x - b*y) / c  (nếu c != 0)
        a, b, c = best_normal
        if abs(c) > 1e-6:
            ground_z = (best_d - a * Xf - b * Yf) / c
        else:
            ground_z = np.full((h, w), best_d)

        # Water above ground: depth > ground_z → có vật thể cao hơn mặt đường
        water_above = np.clip(depth_norm - ground_z, 0, 1)

        ground_level_z = float(np.median(ground_z[full_inlier_mask])) if full_inlier_mask.any() else float(best_d)
        inlier_ratio = best_inliers / max(len(pts), 1)
        confidence = min(0.95, inlier_ratio * 1.5)

        log.info(
            f"  RANSAC ground: normal=({a:.3f},{b:.3f},{c:.3f}) "
            f"d={best_d:.3f} inliers={best_inliers} ({inlier_ratio:.0%}) "
            f"conf={confidence:.2f}"
        )

        return {
            "plane_normal":       tuple(best_normal.tolist()),
            "plane_d":            float(best_d),
            "inlier_mask":        full_inlier_mask,
            "ground_level_z":     ground_level_z,
            "water_above_ground": water_above,
            "confidence":         round(confidence, 3),
            "n_inliers":          int(best_inliers),
        }

    def _ground_plane_fallback(self, depth_norm: np.ndarray, start_row: int) -> dict:
        """Fallback khi RANSAC không đủ data."""
        h, w = depth_norm.shape
        bottom_half = depth_norm[start_row:, :]
        ground_z = float(np.percentile(bottom_half, 30))  # 30th percentile = likely ground
        water_above = np.clip(depth_norm - ground_z, 0, 1)
        dummy_mask = np.zeros((h, w), dtype=bool)
        dummy_mask[start_row:, :] = True
        return {
            "plane_normal":       (0.0, 0.0, 1.0),
            "plane_d":            ground_z,
            "inlier_mask":        dummy_mask,
            "ground_level_z":     ground_z,
            "water_above_ground": water_above,
            "confidence":         0.30,
            "n_inliers":          0,
        }

    def _build_notes(
        self, road_pct: float, water_pct: float, is_dry: bool, adj_y: int, orig_y: int
    ) -> str:
        parts = [f"Road={road_pct:.1f}% Water={water_pct:.1f}%"]
        if is_dry:
            parts.append("DRY ROAD DETECTED")
        if adj_y != orig_y:
            parts.append(f"WL adjusted {orig_y}->{adj_y}")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    def create_debug_overlay(
        self,
        img_rgb:  np.ndarray,
        result:   GroundAnalysisResult,
    ) -> np.ndarray:
        """Ve segmentation overlay de debug."""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        overlay = img_bgr.copy()
        h, w    = img_bgr.shape[:2]

        # To mau: duong kho = xanh la, nuoc = xanh duong
        color_map = np.zeros((h, w, 3), dtype=np.uint8)
        color_map[result.road_mask > 0]      = (0,  200,  0)    # xanh la = duong
        color_map[result.water_mask_seg > 0] = (200, 0,   0)    # do = nuoc (BGR)

        blended = cv2.addWeighted(overlay, 0.6, color_map, 0.4, 0)

        # Ve water line
        if result.adjusted_water_line_y < h:
            cv2.line(blended, (0, result.adjusted_water_line_y),
                     (w, result.adjusted_water_line_y), (0, 255, 255), 2)

        # Label
        lbl = f"Road={result.road_area_pct:.0f}% | Water={result.water_area_pct:.0f}% | {'DRY' if result.is_dry_road else 'WET'}"
        cv2.putText(blended, lbl, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return blended
