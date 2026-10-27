# -*- coding: utf-8 -*-
"""
depth_analysis/flood_classifier.py  —  v2
==========================================
DINOv2 + WaterDetector v3 (15 color profiles, LAB, YCbCr, Norm-RGB).
FloodClassification giờ trả thêm: water_color_name, turbidity, water_type.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
from PIL import Image
from utils.constants import DEFAULT_DINO_MODEL

log = logging.getLogger(__name__)


@dataclass
class FloodClassification:
    has_flood:        bool
    flood_prob:       float
    water_area_pct:   float
    road_dry:         bool
    confidence:       str         # "HIGH"/"MEDIUM"/"LOW"
    reason:           str
    # ── MỚI ─────────────────────────────
    water_color_name: str = ""    # tên Tiếng Việt màu nước (VD: "Nước bùn nâu")
    water_type_key:   str = ""    # key profile (VD: "muddy_brown")
    turbidity:        float = 0.0 # 0=trong .. 1=đục
    is_foam:          bool  = False
    is_oil:           bool  = False
    is_nighttime:     bool  = False
    water_level_pct:  float = 0.0 # tỉ lệ mực nước chiếm khung hình
    channel_scores:   Optional[dict] = None
    # ── Nhận dạng ảnh không có nước ─────────────────────────────────
    no_flood_scene:   bool  = False  # True = ảnh cây/nhà/cổng, chắc chắn không có nước
    scene_type:       str   = ""     # "vegetation"/"building_gate"/"dry_outdoor"
    vegetation_pct:   float = 0.0    # % diện tích thực vật phát hiện được
    has_puddle:       bool  = False  # phát hiện vũng nước nhỏ (PUDDLE level)
    water_mask:       Optional[np.ndarray] = None  # mask nước từ WaterDetector v3/v4
    water_line_y:     int   = 0      # water line từ WaterDetector


class FloodClassifier:
    def __init__(
        self,
        dino_model:      str   = DEFAULT_DINO_MODEL,
        flood_threshold: float = 0.45,
        min_water_area:  float = 0.03,
        device:          str   = "auto",
        resnet_path:     str   = "",            # path đến .pth, "" = tắt
        resnet_labels:   Optional[list] = None, # ["dry","flood","heavy_flood"]
        ensemble_weights: Optional[dict] = None, # {"color":0.55,"dino":0.30,"resnet":0.15}
        cfg:             Optional[dict] = None,  # [v4] pipeline config → WaterDetector & model options
    ):
        self._cfg             = cfg or {}
        models_cfg            = self._cfg.get("models", {}) or {}
        wc_cfg                = self._cfg.get("water_color", {}) or {}

        self.dino_model       = dino_model or models_cfg.get("dino", DEFAULT_DINO_MODEL)
        self.flood_threshold  = flood_threshold
        self.min_water_area   = wc_cfg.get("min_water_area", min_water_area)
        self.device           = device
        self.resnet_path      = resnet_path or models_cfg.get("flood_resnet", "")
        self.resnet_labels    = (
            resnet_labels
            or models_cfg.get("flood_resnet_labels")
            or ["dry", "flood", "heavy_flood"]
        )
        self.ensemble_weights = (
            ensemble_weights
            or models_cfg.get("ensemble_weights")
            or {"color": 0.55, "dino": 0.30, "resnet": 0.15}
        )
        self._extractor: Any = None
        self._model: Any     = None
        self._water_detector = None  # lazy load
        self._resnet: Any    = None  # lazy load

    def _get_water_detector(self):
        if self._water_detector is None:
            from depth_analysis.water_detector import WaterDetector
            wd_cfg = self._cfg.get("water_detection", {}) or {}
            wc_cfg = self._cfg.get("water_color", {}) or {}
            self._water_detector = WaterDetector(
                min_water_area=self.min_water_area,
                use_reflection=wc_cfg.get("use_reflection", True),
                use_texture=wc_cfg.get("use_texture", True),
                use_lab=wc_cfg.get("use_lab", True),
                use_ycbcr=wc_cfg.get("use_ycbcr", True),
                use_norm_rgb=wc_cfg.get("use_norm_rgb", True),
                refine_line=wd_cfg.get("refine_line", True),
            )
        return self._water_detector

    def _load_dino(self):
        if self._model: return
        from transformers import AutoImageProcessor, AutoModel
        dev = "cuda" if __import__("torch").cuda.is_available() else "cpu" if self.device == "auto" else self.device
        self._device = dev
        log.info(f"  Loading DINOv2 ({self.dino_model}) on {dev}...")
        self._extractor = AutoImageProcessor.from_pretrained(self.dino_model)
        self._model     = AutoModel.from_pretrained(self.dino_model).to(dev)
        self._model.eval()

    def _extract_features(self, pil_img) -> np.ndarray:
        import torch
        inputs = self._extractor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
            # Trả về toàn bộ sequence (CLS + spatial patch tokens): shape (1 + N_patches, D)
            tokens = out.last_hidden_state[0].cpu().numpy()
        return tokens

    def _get_resnet(self):
        """Lazy-load ResNet-18. Trả về None nếu không config hoặc load thất bại."""
        if self._resnet is not None:
            return self._resnet if isinstance(self._resnet, dict) and self._resnet else None
        if not self.resnet_path:
            self._resnet = {}  # sentinel: không config
            return None
        try:
            from core.model_loader import _loader_resnet18
            self._resnet = _loader_resnet18({
                "model_path":  self.resnet_path,
                "num_classes": len(self.resnet_labels),
                "labels":      self.resnet_labels,
            })
        except FileNotFoundError as e:
            log.warning(f"  [flood_resnet] Không tìm thấy model, bỏ qua: {e}")
            self._resnet = {}
        except Exception as e:
            log.warning(f"  [flood_resnet] Load thất bại ({e}), bỏ qua")
            self._resnet = {}
        return self._resnet if self._resnet else None

    def _resnet_flood_score(self, img_rgb: np.ndarray) -> float:
        """
        Chạy ResNet-18 trên ảnh RGB, trả về flood probability [0.0 – 1.0].

        Score = softmax[flood]*0.6 + softmax[heavy_flood]*1.0  (clipped 0–1)

        - Nếu model không load được → 0.5 (neutral, không kéo ensemble)
        - Label map dựa theo self.resnet_labels (có thể đổi qua config.yaml)
        """
        bundle = self._get_resnet()
        if not bundle:
            return 0.5

        import torch
        model     = bundle["model"]
        transform = bundle["transform"]
        device    = bundle["device"]
        labels    = bundle["labels"]

        try:
            tensor = transform(img_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            prob_map = {lbl: float(probs[i]) for i, lbl in enumerate(labels)}
            p_flood  = prob_map.get("flood",       0.0)
            p_heavy  = prob_map.get("heavy_flood", 0.0)

            # heavy_flood = tín hiệu mạnh hơn → hệ số 1.0
            # flood vừa   → conservative 0.6
            raw = p_flood * 0.6 + p_heavy * 1.0
            return float(np.clip(raw, 0.0, 1.0))

        except Exception as e:
            log.warning(f"  [flood_resnet] Inference lỗi ({e}), dùng neutral 0.5")
            return 0.5

    def _dino_flood_score(self, features: np.ndarray) -> float:
        """
        Ước lượng flood probability từ DINOv2 spatial patch tokens + CLS token.
        [CẢI TIẾN v5]:
          - Sử dụng tính chất tương đồng ngữ nghĩa không gian (spatial self-similarity)
            của DINOv2 patch tokens ở nửa dưới ảnh: mặt nước tạo ra bề mặt đồng nhất
            ngữ nghĩa theo phương ngang (horizontal coherence) và tách biệt rõ với
            nửa trên (công trình/cây cối/bầu trời).
          - Vẫn tương thích ngược nếu truyền vào vector 1D (CLS token đơn lẻ).
        """
        if features.ndim == 2 and features.shape[0] >= 16:
            # Hỗ trợ mọi biến thể DINOv2: (K^2, D), (1 + K^2, D) có CLS, hoặc (1 + 4 + K^2, D) có registers
            side = int(np.sqrt(features.shape[0]))
            n_patches = side * side
            patches = features[-n_patches:]  # lấy đúng K×K spatial patch tokens ở cuối
            # Chuẩn hóa L2 từng patch vector
            norms = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8
            p_norm = patches / norms

            if side >= 4:
                grid = p_norm.reshape(side, side, -1)
                lower = grid[side // 2:, :, :]   # nửa dưới ảnh
                upper = grid[:side // 2, :, :]   # nửa trên ảnh

                # 1. Horizontal semantic coherence ở nửa dưới (mặt nước liền mạch ngang)
                horiz_sim = float(np.sum(lower[:, :-1, :] * lower[:, 1:, :], axis=-1).mean())

                # 2. Độ tập trung ngữ nghĩa (homogeneity) của nửa dưới quanh tâm vùng dưới
                lower_flat = lower.reshape(-1, lower.shape[-1])
                lower_proto = lower_flat.mean(axis=0)
                lower_proto /= (np.linalg.norm(lower_proto) + 1e-8)
                lower_homog = float((lower_flat @ lower_proto).mean())

                # 3. Độ tương phản ngữ nghĩa giữa nửa trên (nhà/cây/trời) và nửa dưới (mặt nước)
                upper_flat = upper.reshape(-1, upper.shape[-1])
                upper_proto = upper_flat.mean(axis=0)
                upper_proto /= (np.linalg.norm(upper_proto) + 1e-8)
                vert_contrast = float(1.0 - np.clip(np.dot(upper_proto, lower_proto), -1.0, 1.0))

                # Quy đổi về [0, 1]
                coh_score   = float(np.clip((horiz_sim - 0.35) / 0.45, 0.0, 1.0))
                homog_score = float(np.clip((lower_homog - 0.40) / 0.45, 0.0, 1.0))
                cont_score  = float(np.clip(vert_contrast / 0.70, 0.0, 1.0))

                raw_score = 0.45 * coh_score + 0.35 * homog_score + 0.20 * cont_score
                return float(np.clip(raw_score * 0.85, 0.05, 0.78))

            features = features[0]

        # Fallback cho vector 1D (CLS token)
        fn = features / (np.linalg.norm(features) + 1e-8)
        kurt = float(np.mean(fn ** 4) / (np.mean(fn ** 2) ** 2 + 1e-12))
        peak_ratio = float((np.abs(fn) > 0.09).mean())
        score = 0.50 * min(1.0, kurt / 6.0) + 0.50 * min(1.0, peak_ratio * 4.0)
        return float(np.clip(score * 0.75, 0.10, 0.70))

    def classify(
        self,
        image_path: Path | str | np.ndarray,
        precomputed_wdr: Any = None,
    ) -> FloodClassification:
        img_name = "ndarray_image"
        try:
            if isinstance(image_path, np.ndarray):
                img_rgb = image_path
                pil_img = Image.fromarray(img_rgb)
            else:
                img_name = Path(image_path).name
                img_bgr = cv2.imread(str(image_path))
                if img_bgr is None:
                    raise ValueError("Cannot read")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
        except Exception as e:
            log.warning(f"  Cannot open {img_name}: {e}")
            return FloodClassification(
                has_flood=False, flood_prob=0.0, water_area_pct=0.0,
                road_dry=True, confidence="LOW", reason="Cannot read image"
            )

        # === 1. WaterDetector v3/v4 — phân tích màu đa chiều (dùng lại nếu đã tính) ===
        if precomputed_wdr is not None:
            wdr = precomputed_wdr
        else:
            wd  = self._get_water_detector()
            wdr = wd.detect(img_rgb)
        cp  = wdr.color_profile

        water_pct = wdr.water_area_pct / 100.0  # convert % → ratio

        # === 1b. Nhận dạng ảnh không có nước (cây, nhà, cổng) ===
        no_water_scene, nw_penalty, nw_scene_type, nw_reason = self._detect_no_water_scene(
            wdr.scene_context, water_pct
        )

        # Kiểm tra đường khô (độc lập WaterDetector cũ)
        road_dry  = self._check_road_dry(img_rgb, water_pct)

        # === 2. DINOv2 ===
        try:
            self._load_dino()
            dino_score = self._dino_flood_score(self._extract_features(pil_img))
        except Exception as e:
            log.warning(f"  DINOv2 failed ({e}), using color only")
            dino_score = 0.5

        # === 2b. ResNet-18 flood classifier (nếu được config) ===
        resnet_score = self._resnet_flood_score(img_rgb)
        has_resnet   = bool(self._get_resnet())  # True = model đã load thành công

        # === 3. Kết hợp score — v4 với dynamic weighting + ResNet ensemble ===

        color_score = min(1.0, water_pct / max(self.min_water_area * 3, 1e-6))

        # [CẢI TIẾN] road_dry: penalty theo mức turbidity
        # Nước bùn đục cao (>0.70) + road_dry = ngập đô thị (một phần đường vẫn thấy)
        # → chỉ áp dụng soft penalty, KHÔNG veto hoàn toàn
        if road_dry:
            if cp.turbidity > 0.70:
                # Nước bùn + nhìn thấy đường = ngập đô thị — penalty nhẹ
                color_score  *= 0.55
                dino_score   *= 0.60
                resnet_score *= 0.70  # ResNet đã train trên ảnh thực → penalty ít hơn DINOv2
            elif cp.turbidity > 0.40:
                # Nước đục vừa + đường — penalty trung bình
                color_score  *= 0.35
                dino_score   *= 0.50
                resnet_score *= 0.55
            else:
                # Nước trong + đường khô — rất có thể không phải lũ — penalty nặng
                color_score  *= 0.15
                dino_score   *= 0.40
                resnet_score *= 0.40

        # [CẢI TIẾN] Turbidity multiplier — nước bùn = dấu hiệu mạnh nhất của lũ thật
        # Thay vì chỉ cộng 8%, dùng multiplier lên đến 35%
        if cp.turbidity > 0.80:
            turbidity_mult = 1.35   # nước bùn sẫm → gần chắc chắn lũ
        elif cp.turbidity > 0.60:
            turbidity_mult = 1.20
        elif cp.turbidity > 0.40:
            turbidity_mult = 1.08
        else:
            turbidity_mult = 1.00

        # [CẢI TIẾN] Channel agreement bonus — nhiều channels đồng ý → boost confidence
        ch_scores = cp.channel_scores or {}
        agreeing_channels = sum(
            1 for k, v in ch_scores.items()
            if k not in ("fragmentation", "reflection") and v > 0.03
        )
        channel_bonus = min(0.12, agreeing_channels * 0.02)

        # [CẢI TIẾN] Spatial consistency: nước ở 15% trên của ảnh mà không có reflection
        # thì đây là false positive (bầu trời, biển biểu trưng, v.v.)
        spatial_penalty = 0.0
        if wdr.water_level_pct > 0.85 and not wdr.has_reflection and water_pct < 0.3:
            spatial_penalty = 0.15  # water line rất cao + không có reflection → suspicious

        # === Ensemble weights — đọc từ config + tự động chuẩn hóa tổng = 1.0 ===
        ew = self.ensemble_weights or {}
        w_color  = float(ew.get("color",  0.55))
        w_dino   = float(ew.get("dino",   0.30))
        w_resnet = float(ew.get("resnet", 0.15)) if has_resnet else 0.0
        total_w  = w_color + w_dino + w_resnet
        if total_w > 0:
            w_color  /= total_w
            w_dino   /= total_w
            w_resnet /= total_w

        if has_resnet:
            combined = (
                w_color  * color_score * turbidity_mult
                + w_dino   * dino_score
                + w_resnet * resnet_score
                + channel_bonus
            ) - spatial_penalty
            log.debug(
                f"  [ensemble-v5] color={color_score:.2f}×{turbidity_mult:.2f} "
                f"dino={dino_score:.2f} resnet={resnet_score:.2f} → {combined:.2f}"
            )
        else:
            combined = (
                w_color * color_score * turbidity_mult
                + w_dino * dino_score
                + channel_bonus
            ) - spatial_penalty
            log.debug(
                f"  [ensemble-v5-no-resnet] color={color_score:.2f}×{turbidity_mult:.2f} "
                f"dino={dino_score:.2f} → {combined:.2f}"
            )

        # [MỚI] No-water scene penalty: cây/nhà/cổng → giảm điểm mạnh
        if nw_penalty > 0.0:
            combined *= (1.0 - nw_penalty * 0.75)

        # [MỚI] Puddle boost: vũng nước nhỏ trên đường = vẫn là nước thật
        # Không áp dụng nếu cảnh là cây/nhà/cổng (no_water_scene đã xử lý rồi)
        if wdr.has_puddle and not no_water_scene:
            combined = min(1.0, combined + 0.08)

        combined = float(np.clip(combined, 0.0, 1.0))

        # [CẢI TIẾN] has_flood condition: turbidity cao → ngưỡng thấp hơn
        effective_threshold = self.flood_threshold
        if cp.turbidity > 0.7:
            effective_threshold *= 0.80
        elif cp.turbidity > 0.5:
            effective_threshold *= 0.90

        # [MỚI] Puddle: hạ cả ngưỡng và min_water_area
        effective_min_water = self.min_water_area
        if wdr.has_puddle and not no_water_scene:
            effective_threshold  *= 0.85
            effective_min_water   = self.min_water_area * 0.45  # vũng nhỏ hơn

        has_flood = (
            combined > effective_threshold and
            water_pct >= effective_min_water and
            # Không veto puddle rõ ràng dựa trên road_dry + nước trong
            not (road_dry and cp.turbidity < 0.35 and not wdr.has_puddle)
        )

        conf = ("HIGH" if combined > 0.75 or combined < 0.2
                else "MEDIUM" if combined > 0.55 or combined < 0.35
                else "LOW")

        puddle_tag = " [VŨNG NƯỚC]" if wdr.has_puddle else ""
        if no_water_scene and nw_penalty > 0.4:
            reason = f"Không có nước — {nw_reason}"
        elif road_dry and water_pct < effective_min_water:
            reason = f"Đường khô ({water_pct:.1%} nước, turbid={cp.turbidity:.2f}){puddle_tag}"
        elif water_pct < effective_min_water:
            reason = f"Diện tích nước quá nhỏ ({water_pct:.1%}){puddle_tag}"
        elif spatial_penalty > 0:
            reason = f"Nghi vấn spatial (water line quá cao={wdr.water_level_pct:.0%}, không reflection)"
        elif has_flood:
            reason = (f"Xác nhận lũ{puddle_tag}: nước={water_pct:.1%}, màu={cp.dominant_name_vi}, "
                      f"độ đục={cp.turbidity:.2f}, score={combined:.2f}")
        else:
            reason = f"Borderline{puddle_tag}: nước={water_pct:.1%}, score={combined:.2f}"

        log.info(
            f"  {img_name}: {'LŨ' if has_flood else 'KHÔNG LŨ'} "
            f"(prob={combined:.2f}, màu={cp.dominant_name_vi}, đục={cp.turbidity:.2f}"
            + (f", resnet={resnet_score:.2f}" if has_resnet else "")
            + ")"
        )

        return FloodClassification(
            has_flood        = has_flood,
            flood_prob       = round(combined, 3),
            water_area_pct   = round(water_pct * 100, 2),
            road_dry         = road_dry,
            confidence       = conf,
            reason           = reason,
            water_color_name = cp.dominant_name_vi,
            water_type_key   = cp.dominant_type,
            turbidity        = cp.turbidity,
            is_foam          = cp.is_foam,
            is_oil           = cp.is_oil,
            is_nighttime     = cp.is_nighttime,
            water_level_pct  = wdr.water_level_pct,
            channel_scores   = cp.channel_scores,
            no_flood_scene   = no_water_scene,
            scene_type       = nw_scene_type,
            vegetation_pct   = round(wdr.scene_context.get("vegetation_pct", 0.0) * 100, 2),
            has_puddle       = wdr.has_puddle,
            water_mask       = wdr.water_mask,
            water_line_y     = wdr.water_line_y,
        )

    def _check_road_dry(self, img_rgb: np.ndarray, water_pct: float) -> bool:
        """
        Kiểm tra đường khô.
        [CẢI TIẾN v3]:
          - Thêm phát hiện nhựa đường ướt (dark + low sat + smooth texture)
          - Dùng LAB để phân biệt tốt hơn giữa đường khô và đường ướt
          - Tăng ngưỡng: cần > 35% đường khô (thay vì 30%) để tránh veto sai
          - Đường ướt ≠ đường khô: không veto nếu có nhiều đường ướt
        """
        h, w  = img_rgb.shape[:2]
        lower = img_rgb[int(h * 0.4):, :]

        lower_bgr = cv2.cvtColor(lower, cv2.COLOR_RGB2BGR)
        lower_hsv = cv2.cvtColor(lower_bgr, cv2.COLOR_BGR2HSV)
        lower_lab = cv2.cvtColor(lower_bgr, cv2.COLOR_BGR2LAB)

        # [CẢI TIẾN] Đường khô: xám nhạt, saturation rất thấp, value trung bình-cao
        dry_mask = cv2.inRange(
            lower_hsv,
            np.array([0,  0, 50]),
            np.array([180, 40, 200])
        )
        dry_pct = float(dry_mask.sum() / 255) / (lower.shape[0] * lower.shape[1])

        # [MỚI] Đường ướt: tối hơn đường khô, vẫn low saturation nhưng value thấp
        # Đường ướt có thể bị nhầm là nước — loại trường hợp này
        wet_road_mask = cv2.inRange(
            lower_hsv,
            np.array([0,  0, 15]),
            np.array([180, 45, 90])
        )
        wet_road_pct = float(wet_road_mask.sum() / 255) / (lower.shape[0] * lower.shape[1])

        # [MỚI] LAB check: L cao (sáng), a và b gần trung tính → đường khô bê tông
        lab_l = lower_lab[:, :, 0]
        lab_a = lower_lab[:, :, 1].astype(np.int16) - 128
        lab_b = lower_lab[:, :, 2].astype(np.int16) - 128
        concrete_mask = (
            (lab_l > 80) &
            (np.abs(lab_a) < 15) &
            (np.abs(lab_b) < 15)
        ).astype(np.uint8) * 255
        concrete_pct = float(concrete_mask.sum() / 255) / (lower.shape[0] * lower.shape[1])

        combined_dry_pct = max(dry_pct, concrete_pct)

        # [CẢI TIẾN] Ngưỡng tăng lên 35% (thay vì 30%)
        # Nếu có nhiều đường ướt → có thể đang ngập, không veto
        is_dry = (
            combined_dry_pct > 0.35 and
            water_pct < self.min_water_area * 2 and
            wet_road_pct < 0.25   # không quá nhiều đường ướt
        )
        return is_dry

    def _detect_no_water_scene(
        self, scene_ctx: dict, water_pct: float
    ) -> tuple:
        """
        Nhận dạng ảnh không có nước lũ: chỉ có cây cối, nhà cửa, cổng ngõ.

        Returns: (is_no_flood, penalty_factor, scene_type, reason)
          penalty_factor: 0.0 = không ảnh hưởng, 1.0 = veto hoàn toàn
        """
        veg_pct  = scene_ctx.get("vegetation_pct", 0.0)
        roof_pct = scene_ctx.get("roof_pct", 0.0)

        # Cảnh cây cối chiếm đa số + nước ít
        if veg_pct > 0.35 and water_pct < 0.08:
            penalty = min(0.90, veg_pct * 2.0)
            reason  = f"Cảnh cây cối ({veg_pct:.0%} thực vật, nước={water_pct:.1%})"
            return True, penalty, "vegetation", reason

        # Cảnh nhà/cổng: kết hợp cây + mái ngói + nước thấp
        if (veg_pct + roof_pct) > 0.30 and water_pct < 0.06:
            penalty = min(0.80, (veg_pct + roof_pct) * 1.5)
            reason  = (f"Cảnh nhà/cổng ({veg_pct:.0%} cây, "
                       f"{roof_pct:.0%} công trình, nước={water_pct:.1%})")
            return True, penalty, "building_gate", reason

        # Cây vừa phải nhưng nước rất ít
        if veg_pct > 0.20 and water_pct < 0.04:
            penalty = min(0.60, veg_pct * 1.5)
            reason  = f"Cảnh khô ngoài trời ({veg_pct:.0%} cây, nước={water_pct:.1%})"
            return True, penalty, "dry_outdoor", reason

        return False, 0.0, "", ""

    def classify_batch(self, image_paths: list) -> dict:
        results = {}
        total   = len(image_paths)
        for i, p in enumerate(image_paths, 1):
            log.info(f"  [{i}/{total}] Classifying {Path(p).name}")
            results[str(p)] = self.classify(Path(p))
        return results
