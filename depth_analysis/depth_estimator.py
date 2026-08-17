import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Optional
from utils.constants import DEFAULT_DEPTH_MODEL

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class FloodDepthResult:
    filename:          str
    original_path:     str
    depth_map_path:    str
    overlay_path:      str
    flood_level:       str        # LOW / MEDIUM / HIGH / SEVERE
    flood_percentage:  float      # % dien tich anh bi ngap
    mean_depth_score:  float      # trung binh normalized depth (0-1)
    max_depth_score:   float
    water_region_area: float      # % dien tich duoc phan loai la nuoc
    confidence:        float      # 0-1, do tin cay phan loai
    notes:             str        # ghi chu them
    mc_uncertainty:    float = 0.0  # [IMPROVE] MC Dropout uncertainty (0=noise, 1=khong chac chan)


class DepthEstimator:
    """
    Uoc tinh muc nuoc lu — v2 (Multi-Model Fusion).

    Chiến lược fusion:
        final_depth = w1 * depth_model_score
                    + w2 * geometry_estimation
                    + w3 * reference_object_estimation

    Models depth được hỗ trợ:
      - "depth_anything_v2"  : Depth Anything V2 (mặc định, relative)
      - "midas"              : MiDaS DPT (tốt cho outdoor)
      - "zoedepth"           : ZoeDepth (metric depth, tốt nhất cho outdoor)
      - "auto"               : thử ZoeDepth → MiDaS → Depth Anything V2
    """

    # Nguong phan loai muc ngap
    FLOOD_THRESHOLDS = {
        "LOW":    (0.0,  0.20),
        "MEDIUM": (0.20, 0.40),
        "HIGH":   (0.40, 0.65),
        "SEVERE": (0.65, 1.01),
    }

    # Trọng số fusion
    FUSION_WEIGHTS = {
        "depth_model":   0.40,
        "geometry":      0.30,
        "reference_obj": 0.30,
    }

    def __init__(self,
                 model_id: str = DEFAULT_DEPTH_MODEL,
                 depth_backend: str = "auto",   # "depth_anything_v2"|"midas"|"zoedepth"|"auto"
                 output_dir: Optional[Path] = None,
                 device: str = "auto",
                 use_fusion: bool = True):
        self.model_id      = model_id
        self.depth_backend = depth_backend
        self.output_dir    = Path(output_dir) if output_dir else Path("output/depth")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device        = device
        self.use_fusion    = use_fusion
        self._pipe: Any          = None    # lazy load
        self._midas_model: Any   = None
        self._midas_transform: Any = None
        self._loaded_backend: str = ""

    # -------------------------------------------------------------
    def _load_model(self):
        """Lazy-load depth model theo backend được chọn."""
        if self._pipe is not None or self._midas_model is not None:
            return

        import torch
        dev = self.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch_device = dev

        backend = self.depth_backend
        if backend == "auto":
            # Thử theo thứ tự ưu tiên
            for b in ["zoedepth", "midas", "depth_anything_v2"]:
                try:
                    self._load_specific_backend(b, dev)
                    log.info(f"  DepthEstimator: loaded backend={b}")
                    return
                except Exception as e:
                    log.debug(f"  Backend {b} failed: {e}")
            # fallback
            backend = "depth_anything_v2"

        self._load_specific_backend(backend, dev)

    def _load_specific_backend(self, backend: str, dev: str):
        """Load một backend cụ thể."""
        if backend == "zoedepth":
            import torch
            log.info("  Loading ZoeDepth (metric depth)...")
            _zoe: Any = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
            self._midas_model = _zoe.to(dev)
            self._midas_model.eval()
            self._loaded_backend = "zoedepth"

        elif backend == "midas":
            import torch
            log.info("  Loading MiDaS DPT-Large...")
            _midas: Any = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            self._midas_model = _midas.to(dev)
            _transforms: Any = torch.hub.load("intel-isl/MiDaS", "transforms")
            self._midas_transform = _transforms.dpt_transform
            self._midas_model.eval()
            self._loaded_backend = "midas"

        elif backend == "depth_anything_v2":
            import torch
            from transformers import pipeline
            log.info(f"  Loading Depth Anything V2: {self.model_id}")
            self._pipe = pipeline(
                task="depth-estimation",
                model=self.model_id,
                device=dev,
            )
            self._loaded_backend = "depth_anything_v2"

        else:
            raise ValueError(f"Unknown depth backend: {backend}")

    # -------------------------------------------------------------
    def analyze(self, image_path: Path) -> Optional[FloodDepthResult]:
        """
        Phân tích 1 ảnh với multi-model fusion.
        final_depth = w1*depth_model + w2*geometry + w3*reference_obj
        """
        self._load_model()

        try:
            pil_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            log.warning(f"  Cannot open {image_path.name}: {e}")
            return None

        img_np = np.array(pil_img)

        # -- Depth inference (Model 1: depth model) ----------------
        depth_norm = self._run_depth_inference(pil_img)
        if depth_norm is None:
            return None

        # -- Geometry estimation (Model 2: perspective + ground plane) --
        geo_flood_pct = self._geometry_estimation(img_np, depth_norm)

        # -- Reference object estimation (Model 3) ------------------
        ref_flood_pct = self._reference_obj_estimation(img_np, depth_norm)

        # -- Multi-model FUSION ------------------------------------
        w = self.FUSION_WEIGHTS
        mc_uncertainty = 0.0  # [IMPROVE] MC Dropout uncertainty
        if self.use_fusion:
            # Depth model flood pct
            dm_flood, water_area, dm_conf = self._estimate_flood_level(depth_norm, pil_img)

            # [IMPROVE] MC Dropout uncertainty estimation:
            # Chay N forward passes voi dropout → tinh std.
            # Std cao → model khong chac chan → giam confidence.
            mc_result = self._mc_dropout_depth(pil_img)
            if mc_result is not None:
                mc_depth, mc_uncertainty = mc_result
                # Giam confidence theo uncertainty: mc_unc=0 → giu nguyen, mc_unc=0.3 → giam 0.15
                mc_penalty = min(0.20, mc_uncertainty * 0.5)
                dm_conf = max(0.1, dm_conf - mc_penalty)
                log.debug(f"  MC Dropout: uncertainty={mc_uncertainty:.4f} penalty={mc_penalty:.3f}")

            # Weighted fusion
            flood_pct = (
                w["depth_model"]   * dm_flood +
                w["geometry"]      * geo_flood_pct +
                w["reference_obj"] * ref_flood_pct
            )
            # Confidence tăng khi các model đồng thuận
            variance = np.var([dm_flood, geo_flood_pct, ref_flood_pct])
            agreement_bonus = max(0.0, 0.15 - variance * 2.0)
            confidence = min(0.95, dm_conf + agreement_bonus)

            log.debug(
                f"  Fusion: depth={dm_flood:.2f} geo={geo_flood_pct:.2f} "
                f"ref={ref_flood_pct:.2f} → fused={flood_pct:.2f}"
            )
        else:
            flood_pct, water_area, confidence = self._estimate_flood_level(depth_norm, pil_img)

        flood_level = self._classify(flood_pct)
        mean_depth  = float(depth_norm.mean())
        max_depth   = float(depth_norm.max())

        # -- Save outputs ------------------------------------------
        depth_map_path = self._save_depth_colormap(depth_norm, image_path)
        overlay_path   = self._save_overlay(
            img_np, depth_norm, image_path, flood_level, flood_pct
        )

        result = FloodDepthResult(
            filename         = image_path.name,
            original_path    = str(image_path),
            depth_map_path   = str(depth_map_path),
            overlay_path     = str(overlay_path),
            flood_level      = flood_level,
            flood_percentage = round(flood_pct * 100, 1),
            mean_depth_score = round(mean_depth, 4),
            max_depth_score  = round(max_depth, 4),
            water_region_area= round(water_area * 100 if self.use_fusion else 0, 1),
            confidence       = round(float(confidence), 3),
            notes            = self._generate_notes(flood_level, flood_pct),
            mc_uncertainty   = round(float(mc_uncertainty), 4),  # [IMPROVE]
        )

        meta_path = self.output_dir / f"{image_path.stem}_meta.json"
        meta_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False))

        log.info(
            f"  {image_path.name}: {flood_level} ({flood_pct*100:.1f}% "
            f"flooded, conf={confidence:.2f}, mc_unc={mc_uncertainty:.3f}, "
            f"backend={self._loaded_backend})"
        )
        return result

    def _run_depth_inference(self, pil_img: Image.Image) -> Optional[np.ndarray]:
        """Chạy depth inference với backend đã load."""
        try:
            if self._loaded_backend == "zoedepth":
                import torch
                img_t = __import__("torchvision").transforms.ToTensor()(pil_img).unsqueeze(0)
                img_t = img_t.to(self._torch_device)
                with torch.no_grad():
                    depth = self._midas_model.infer(img_t).squeeze().cpu().numpy()
                # ZoeDepth trả về metric depth (meters) → normalize
                d_min, d_max = depth.min(), depth.max()
                depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
                return depth_norm

            elif self._loaded_backend == "midas":
                import torch
                img_np = np.array(pil_img)
                inp = self._midas_transform(img_np).to(self._torch_device)
                with torch.no_grad():
                    pred = self._midas_model(inp)
                    pred = torch.nn.functional.interpolate(
                        pred.unsqueeze(1),
                        size=img_np.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                depth = pred.cpu().numpy()
                d_min, d_max = depth.min(), depth.max()
                return (depth - d_min) / (d_max - d_min + 1e-6)

            else:  # depth_anything_v2
                output    = self._pipe(pil_img)
                depth_pil = output["depth"]
                depth_np  = np.array(depth_pil, dtype=np.float32)
                d_min, d_max = depth_np.min(), depth_np.max()
                if d_max - d_min < 1e-6:
                    return np.zeros_like(depth_np)
                return (depth_np - d_min) / (d_max - d_min)

        except Exception as e:
            log.warning(f"  Depth inference failed: {e}")
            return None

    def _geometry_estimation(
        self, img_np: np.ndarray, depth_norm: np.ndarray
    ) -> float:
        """
        Ước tính mực nước từ phối cảnh hình học.

        Strategy:
          - Detect ground plane từ depth map (RANSAC-like)
          - Tính vùng nằm trên ground plane (tức là nước)
          - Normalize thành flood_pct [0,1]
        """
        h, w = img_np.shape[:2]

        # Gradient map — vùng nước thường phẳng (gradient thấp)
        gx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Flat regions (gradient < threshold) ở nửa dưới ảnh → nước
        flat_mask = (grad_mag < 0.05).astype(np.uint8) * 255
        lower_mask = np.zeros((h, w), np.uint8)
        lower_mask[h // 2:, :] = 255
        water_geo = cv2.bitwise_and(flat_mask, lower_mask)

        # Refine: loại sky (depth cao + ở trên)
        deep_region = (depth_norm > np.percentile(depth_norm, 70)).astype(np.uint8) * 255
        water_geo = cv2.bitwise_and(water_geo, deep_region)

        k = np.ones((7, 7), np.uint8)
        water_geo = cv2.morphologyEx(water_geo, cv2.MORPH_CLOSE, k)

        return float((water_geo > 0).sum()) / (h * w)

    def _reference_obj_estimation(
        self, img_np: np.ndarray, depth_norm: np.ndarray
    ) -> float:
        """
        Ước tính mực nước từ reference objects trong ảnh.

        Dùng color analysis để detect:
          - Bánh xe (wheel) → ưu tiên cao nhất
          - Cửa xe (car door) → tốt
          - Người (person) → sai số cao nhất

        Kết hợp với depth map để estimate flood height.
        """
        h, w = img_np.shape[:2]
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Detect dark circular regions (bánh xe) ở vùng dưới ảnh
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lower_gray = gray[h * 2 // 5:, :]

        # Hough circle detection cho bánh xe
        circles = cv2.HoughCircles(
            lower_gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=40, param1=80, param2=30,
            minRadius=15, maxRadius=min(h, w) // 4,
        )

        flood_hints = []

        if circles is not None:
            circles = np.around(circles[0]).astype(np.uint16)
            for cx, cy, r in circles:
                cy_global = cy + h * 2 // 5
                # Kiểm tra màu tối (bánh xe cao su)
                roi_y1 = max(0, cy_global - r)
                roi_y2 = min(h, cy_global + r)
                roi_x1 = max(0, cx - r)
                roi_x2 = min(w, cx + r)
                roi = hsv[roi_y1:roi_y2, roi_x1:roi_x2]
                if roi.size == 0:
                    continue
                mean_val = float(roi[:, :, 2].mean())
                mean_sat = float(roi[:, :, 1].mean())

                # Bánh xe: tối + ít màu
                if mean_val < 80 and mean_sat < 60:
                    # Bottom of wheel → flood estimate
                    wheel_bottom_y = cy_global + r
                    flood_pct_hint = 1.0 - (wheel_bottom_y / h)
                    flood_hints.append(("wheel", flood_pct_hint, 1.0))  # highest weight

        # Detect người đứng (tall thin rectangles) → body landmark
        # Dùng sobel để tìm vertical edges
        sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        _, vert_edges = cv2.threshold(sobel_x.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(vert_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch_box = cv2.boundingRect(cnt)
            if ch_box > h * 0.25 and cw < ch_box * 0.6:  # tall narrow = person
                # Estimate: nếu chân ở vị trí y+ch_box thì flood = 1 - (y+ch_box)/h
                foot_y = y + ch_box
                flood_pct_hint = max(0, 1.0 - foot_y / h) * 0.5  # người có sai số cao
                flood_hints.append(("person", flood_pct_hint, 0.4))  # low weight

        if not flood_hints:
            # Fallback: estimate từ depth
            return float((depth_norm[h // 2:, :] > 0.6).sum()) / (h * w / 2)

        # Weighted average (wheel > car > person)
        total_w = sum(w_hint for _, _, w_hint in flood_hints)
        weighted_pct = sum(pct * w_hint for _, pct, w_hint in flood_hints) / (total_w + 1e-6)
        return float(np.clip(weighted_pct, 0, 1))

    # -------------------------------------------------------------
    # [IMPROVE] MC Dropout Uncertainty Quantification
    # Chay N forward passes voi dropout enabled → tinh variance.
    # Variance cao → model khong chac chan → confidence giam.
    # Chi hoat dong voi PyTorch models (midas, zoedepth).
    # Depth Anything V2 dung diffusers pipeline → khong ho tro MC Dropout.
    # -------------------------------------------------------------
    _MC_DROPOUT_RUNS = 5  # so forward passes

    def _mc_dropout_depth(self, pil_img: Image.Image) -> Optional[tuple]:
        """
        MC Dropout: chay N forward passes voi dropout enabled,
        tra ve (mean_depth, uncertainty).
        uncertainty = std Across N runs → cao = model khong chac chan.
        """
        try:
            import torch
        except ImportError:
            return None

        if self._loaded_backend not in ("midas", "zoedepth") or self._torch_device is None:
            return None

        # Bat dropout trong inference
        self._enable_dropout()

        runs = []
        for _ in range(self._MC_DROPOUT_RUNS):
            d = self._run_depth_inference(pil_img)
            if d is not None:
                runs.append(d)

        # Tat dropout
        self._disable_dropout()

        if len(runs) < 3:
            return None

        stack = np.stack(runs, axis=0)   # (N, H, W)
        mean_depth = stack.mean(axis=0)
        uncertainty = float(stack.std(axis=0).mean())  # avg pixel-level std
        return mean_depth, uncertainty

    def _enable_dropout(self):
        """Bat dropout trong tất cả modules (de MC Dropout hoat dong)."""
        try:
            import torch
            if self._loaded_backend == "midas" and self._midas_model is not None:
                for m in self._midas_model.modules():
                    if isinstance(m, torch.nn.Dropout):
                        m.train()   # keep dropout active
            elif self._loaded_backend == "zoedepth" and self._midas_model is not None:
                for m in self._midas_model.modules():
                    if isinstance(m, torch.nn.Dropout):
                        m.train()
        except Exception:
            pass

    def _disable_dropout(self):
        """Tat dropout (back to eval mode)."""
        try:
            import torch
            if self._loaded_backend == "midas" and self._midas_model is not None:
                self._midas_model.eval()
            elif self._loaded_backend == "zoedepth" and self._midas_model is not None:
                self._midas_model.eval()
        except Exception:
            pass

    # -------------------------------------------------------------
    def analyze_batch(self, image_paths: List[Path]) -> List[FloodDepthResult]:
        """Phân tích batch anh, bo qua anh loi."""
        results = []
        total = len(image_paths)
        for i, p in enumerate(image_paths, 1):
            log.info(f"  [{i}/{total}] {p.name}")
            r = self.analyze(p)
            if r:
                results.append(r)
        return results

    # -------------------------------------------------------------
    def _estimate_flood_level(
        self, depth_norm: np.ndarray, pil_img: Image.Image
    ):
        """
        Uoc tinh muc ngap tu depth map ket hop color analysis.

        Chien luoc:
        1. Vung depth cao (gan camera) o nua duoi anh = nuoc
        2. Ket hop voi mau xanh/nau bun de tang do tin cay
        3. Flood % = ty le dien tich vung nuoc so voi toan anh
        """
        h, w = depth_norm.shape
        lower_half = depth_norm[h // 2:, :]

        # [FIX] Depth threshold tinh tu lower_half nhung ap dung len TOAN ANH
        # → nua tren (xa camera, depth THAP) luon < threshold → khong detect nuoc.
        # Fix: chi ap dung water_mask_depth o vung nua duoi (lower_mask).
        depth_thresh = np.percentile(lower_half, 60)
        water_mask_depth = depth_norm > depth_thresh

        # Color analysis: phat hien mau nuoc/bun
        img_np = np.array(pil_img)
        water_mask_color = self._detect_water_color(img_np)

        # Ket hop 2 mask
        # Uu tien vung nua duoi anh (nuoc thuong o thap)
        lower_mask = np.zeros_like(depth_norm, dtype=bool)
        lower_mask[h // 2:, :] = True

        combined = (water_mask_depth & lower_mask) | (water_mask_color & lower_mask)

        flood_pct  = float(combined.sum()) / combined.size
        water_area = float(water_mask_color.sum()) / water_mask_color.size

        # Confidence: cao neu ca depth lan mau deu nhat quan
        # [FIX] Chi tinh agreement o vung nua duoi (lower_mask) vi nua tren
        # depth luon thap → false disagreement voi color mask.
        agreement = float((water_mask_depth[lower_mask] == water_mask_color[lower_mask]).mean())
        confidence = float(0.5 + 0.5 * agreement)

        return flood_pct, water_area, confidence

    def _detect_water_color(self, img_rgb: np.ndarray) -> np.ndarray:
        """Phat hien vung nuoc/bun dua tren mau sac (HSV analysis)."""
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Nuoc xanh duong/xanh la nhat
        mask_blue = cv2.inRange(
            hsv, np.array([90, 30, 30]), np.array([130, 255, 255])
        )
        # Nuoc nau bun
        mask_brown = cv2.inRange(
            hsv, np.array([10, 40, 30]), np.array([30, 200, 200])
        )
        # Nuoc xam duc
        mask_gray = cv2.inRange(
            hsv, np.array([0, 0, 80]), np.array([180, 40, 180])
        )

        combined = cv2.bitwise_or(mask_blue, mask_brown)
        combined = cv2.bitwise_or(combined, mask_gray)

        # Morphological cleanup
        kernel  = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

        return cleaned > 0

    # -------------------------------------------------------------
    def _classify(self, flood_pct: float) -> str:
        for level, (lo, hi) in self.FLOOD_THRESHOLDS.items():
            if lo <= flood_pct < hi:
                return level
        return "SEVERE"

    def _generate_notes(self, level: str, pct: float) -> str:
        notes = {
            "LOW":    f"Mức ngập thap ({pct*100:.1f}%). Co the la vung nuoc nho hoac ngap cuc bo.",
            "MEDIUM": f"Mức ngập trung binh ({pct*100:.1f}%). Ngap duong va via he.",
            "HIGH":   f"Mức ngập cao ({pct*100:.1f}%). Ngap sau, anh huong nha cua.",
            "SEVERE": f"Ngap nghiem trong ({pct*100:.1f}%). Can di tan khan cap.",
        }
        return notes.get(level, "")

    # -------------------------------------------------------------
    def _save_depth_colormap(self, depth_norm: np.ndarray, original: Path) -> Path:
        """Luu depth map dang colormap (Turbo colormap)."""
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        colormap    = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
        dest = self.output_dir / f"{original.stem}_depth.png"
        cv2.imwrite(str(dest), colormap)
        return dest

    def _save_overlay(
        self,
        img_rgb: np.ndarray,
        depth_norm: np.ndarray,
        original: Path,
        flood_level: str,
        flood_pct: float,
    ) -> Path:
        """Luu anh goc chong voi depth colormap + text annotation."""
        # Resize depth map ve kich thuoc anh goc
        h, w = img_rgb.shape[:2]
        depth_resized = cv2.resize(depth_norm, (w, h))
        depth_uint8   = (depth_resized * 255).astype(np.uint8)
        colormap_bgr  = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

        img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Blend 40% depth + 60% original
        overlay  = cv2.addWeighted(img_bgr, 0.6, colormap_bgr, 0.4, 0)

        # Text annotation
        level_colors = {
            "LOW":    (0, 200, 0),
            "MEDIUM": (0, 165, 255),
            "HIGH":   (0, 0, 255),
            "SEVERE": (0, 0, 180),
        }
        color = level_colors.get(flood_level, (255, 255, 255))
        label = f"FLOOD: {flood_level}  ({flood_pct*100:.1f}%)"

        # Background box cho text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(overlay, (8, 8), (20 + tw, 20 + th + 6), (0, 0, 0), -1)
        cv2.putText(overlay, label, (14, 14 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        dest = self.output_dir / f"{original.stem}_overlay.jpg"
        cv2.imwrite(str(dest), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return dest
