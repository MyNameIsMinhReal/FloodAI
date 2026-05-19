# -*- coding: utf-8 -*-
"""
utils/explainability.py
========================
Tạo ảnh giải thích (explainability overlay) cho mỗi kết quả phân tích lũ.

Output: folder explain/ chứa ảnh với:
  - Vùng nước được highlight
  - Vật tham chiếu được đánh dấu + nhãn
  - Mực nước ước lượng (đường ngang)
  - Confidence score + decision
  - Warning badges

Dùng:
    from utils.explainability import ExplainabilityWriter
    writer = ExplainabilityWriter(output_dir / "explain")
    writer.write(image_path, depth_result, report)
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pipeline.uncertainty import UncertaintyReport

log = logging.getLogger("utils.explain")

# ── Màu sắc theo mức lũ ───────────────────────────────────────────────────────
LEVEL_BGR = {
    "dry":       (80,  200, 80),    # xanh lá
    "ankle":     (0,   220, 220),   # vàng cyan
    "knee":      (0,   165, 255),   # cam
    "waist":     (0,   0,   220),   # đỏ
    "chest":     (180, 0,   180),   # tím
    "submerged": (80,  0,   0),     # tím đậm
    "unknown":   (128, 128, 128),
}
DECISION_BGE = {
    "accept":       (50, 180, 50),
    "needs_review": (0,  165, 255),
    "reject":       (0,  0,   220),
}


class ExplainabilityWriter:
    """
    Vẽ explain overlay cho mỗi ảnh và lưu vào explain_dir.

    Dùng:
        writer = ExplainabilityWriter(Path("output/run_001/explain"))
        writer.write(image_path, depth_result, report)
    """

    def __init__(self, explain_dir: Path):
        self.explain_dir = Path(explain_dir)
        self.explain_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        image_path: "str | Path",
        depth_result: Any,
        report: Optional["UncertaintyReport"] = None,
    ) -> Optional[Path]:
        """
        Vẽ explain overlay và lưu file PNG.

        Returns:
            Path đến file explain đã lưu, hoặc None nếu lỗi
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                log.warning("[Explain] Không đọc được ảnh: %s", image_path)
                return None
            out = self._draw(img, depth_result, report)
            out_path = self.explain_dir / (Path(image_path).stem + "_explain.jpg")
            cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
            return out_path
        except Exception as exc:
            log.warning("[Explain] Lỗi khi vẽ explain %s: %s", image_path, exc)
            return None

    def write_batch(
        self,
        image_paths: List[Path],
        depth_results: List[Any],
        reports: Optional[List[Any]] = None,
    ) -> List[Path]:
        reports = reports or [None] * len(depth_results)
        out = []
        for img_p, dr, rp in zip(image_paths, depth_results, reports):
            p = self.write(img_p, dr, rp)
            if p:
                out.append(p)
        return out

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, img: np.ndarray, result: Any, report) -> np.ndarray:
        h, w = img.shape[:2]
        canvas = img.copy()

        level      = _get(result, "flood_level", "unknown") or "unknown"
        depth_cm   = _get(result, "depth_cm", 0) or 0
        confidence = _get(result, "confidence", 0.0) or 0.0
        ref_objects= _get(result, "reference_objects", []) or []
        color      = LEVEL_BGR.get(level, LEVEL_BGR["unknown"])

        # ── 1. Water mask semi-transparent overlay ────────────────────────────
        water_mask = _get(result, "water_mask", None)
        if water_mask is not None:
            try:
                mask = np.array(water_mask, dtype=np.uint8)
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                overlay = canvas.copy()
                overlay[mask > 0] = (
                    overlay[mask > 0] * 0.5 + np.array(color) * 0.5
                ).astype(np.uint8)
                canvas = overlay
            except Exception:
                pass

        # ── 2. Estimated waterline ────────────────────────────────────────────
        waterline_y = _get(result, "waterline_y", None)
        if waterline_y is None and depth_cm > 0:
            # Ước tính vị trí waterline dựa trên chiều sâu (heuristic)
            waterline_y = int(h * (1 - min(depth_cm / 200.0, 0.95)))
        if waterline_y is not None:
            cv2.line(canvas, (0, int(waterline_y)), (w, int(waterline_y)),
                     color, 3, cv2.LINE_AA)
            cv2.putText(canvas, f"~ {depth_cm:.0f} cm",
                        (8, int(waterline_y) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2, cv2.LINE_AA)

        # ── 3. Reference objects ──────────────────────────────────────────────
        for obj in ref_objects[:5]:
            if not isinstance(obj, dict):
                continue
            bbox = obj.get("bbox") or obj.get("box")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 200, 0), 2)
                label = obj.get("class", "obj")
                est_d = obj.get("estimated_depth")
                if est_d:
                    label += f" ~{est_d:.0f}cm"
                cv2.putText(canvas, label, (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

        # ── 4. Top info bar ───────────────────────────────────────────────────
        bar_h = 52
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar[:] = (30, 30, 30)
        # Level badge
        badge_color = LEVEL_BGR.get(level, LEVEL_BGR["unknown"])
        cv2.rectangle(bar, (4, 4), (120, bar_h - 4), badge_color, -1)
        cv2.putText(bar, level.upper(), (8, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)
        # Depth
        cv2.putText(bar, f"{depth_cm:.0f} cm", (130, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 230, 255), 2, cv2.LINE_AA)
        # Confidence
        cv2.putText(bar, f"conf {confidence:.2f}", (240, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA)
        # Decision
        if report:
            dec_color = DECISION_BGE.get(report.decision, (128, 128, 128))
            cv2.putText(bar, report.decision.upper(), (w - 180, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, dec_color, 2, cv2.LINE_AA)

        canvas = np.vstack([bar, canvas])

        # ── 5. Warning strip at bottom ────────────────────────────────────────
        if report and report.warnings:
            warn_h = 22 * len(report.warnings[:3]) + 8
            strip = np.zeros((warn_h, w, 3), dtype=np.uint8)
            strip[:] = (20, 20, 50)
            for i, w_txt in enumerate(report.warnings[:3]):
                cv2.putText(strip, w_txt[:80], (6, 18 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 180, 255), 1)
            canvas = np.vstack([canvas, strip])

        return canvas


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
