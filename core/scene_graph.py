# -*- coding: utf-8 -*-
"""
core/scene_graph.py  —  Unified Scene Graph
=============================================
Thay vì mỗi module (person, water, vehicle) estimate riêng rồi ghép lại,
Scene Graph biểu diễn toàn bộ scene dưới dạng một đồ thị thống nhất,
sau đó reasoning để cho ra depth chính xác hơn.

Architecture:
    [water_segmentor] ─┐
    [pose_analyzer]   ─┤→  SceneGraph  →  unified_depth_reasoning()
    [vehicle_detector]─┘

SceneGraph giữ:
  - Các node: Person, Water, Vehicle, Road (reference plane)
  - Edges: spatial relationships ("person standing in water")
  - Global context: lighting, image quality, camera angle

Sau đó depth_reasoning() dùng graph để:
  1. Chọn reference object tốt nhất
  2. Cross-validate depth từ nhiều sources
  3. Cho ra kết quả với explanation

Sử dụng:
    graph = SceneGraph.from_result(result)
    depth = graph.unified_depth_cm()
    print(graph.explain())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("core.scene_graph")


# ── Node types ─────────────────────────────────────────────────────────────────

@dataclass
class PersonNode:
    """Người trong ảnh."""
    height_cm: float = 170.0          # ước tính chiều cao
    water_level_cm: Optional[float] = None   # mực nước so với người
    pose_visibility: float = 0.5      # 0..1 — pose keypoints visibility
    body_ratio: float = 0.5           # tỉ lệ % cơ thể bị ngập
    confidence: float = 0.5
    has_raincoat: bool = False

    def estimated_flood_cm(self) -> Optional[float]:
        if self.water_level_cm is not None:
            return self.water_level_cm
        if self.body_ratio > 0 and self.height_cm > 0:
            return self.height_cm * self.body_ratio
        return None


@dataclass
class WaterNode:
    """Vùng nước trong ảnh."""
    area_pct: float = 0.0             # % diện tích ảnh là nước
    color_confidence: float = 0.5
    texture_confidence: float = 0.5
    segmentation_mask: Optional[Any] = None  # numpy mask nếu có

    def overall_confidence(self) -> float:
        return (self.color_confidence * 0.5 + self.texture_confidence * 0.5)


@dataclass
class VehicleNode:
    """Xe (ô tô, xe máy, xe tải)."""
    vehicle_type: str = "car"         # car | truck | motorcycle
    wheel_height_cm: float = 35.0    # chiều cao trục bánh xe so mặt đường
    water_level_cm: Optional[float] = None
    confidence: float = 0.5

    WHEEL_HEIGHTS = {
        "car":        35.0,
        "truck":      55.0,
        "motorcycle": 28.0,
        "bus":        60.0,
    }

    def __post_init__(self):
        self.wheel_height_cm = self.WHEEL_HEIGHTS.get(
            self.vehicle_type, self.wheel_height_cm
        )

    def estimated_flood_cm(self) -> Optional[float]:
        return self.water_level_cm


@dataclass
class RoadNode:
    """Mặt đường — reference plane chính."""
    detected: bool = False
    road_visible_pct: float = 0.0    # % đường nhìn thấy được
    camera_angle_deg: float = 0.0    # góc camera so mặt đường


@dataclass
class SceneContext:
    """Global context của toàn bộ scene."""
    image_quality: float = 0.6        # blur, exposure, noise
    lighting: str = "normal"          # dark | normal | bright
    camera_height_m: float = 1.5     # ước tính chiều cao camera
    perspective: str = "ground"       # ground | elevated | aerial
    frame_source: str = "photo"       # photo | video_frame | satellite


# ── Scene Graph ────────────────────────────────────────────────────────────────

class SceneGraph:
    """
    Đồ thị biểu diễn scene lũ lụt.

    Thay vì mỗi module estimate riêng và ghép sau,
    SceneGraph reasoning toàn bộ cùng một lúc để có kết quả nhất quán.

    Sử dụng:
        graph = SceneGraph.from_result(depth_result)
        depth_cm, explanation = graph.unified_depth_reasoning()
        confidence = graph.compute_confidence()
    """

    def __init__(self):
        self.persons: List[PersonNode] = []
        self.waters:  List[WaterNode] = []
        self.vehicles: List[VehicleNode] = []
        self.road:    RoadNode = RoadNode()
        self.context: SceneContext = SceneContext()

        # Kết quả sau reasoning
        self._depth_cm: Optional[float] = None
        self._confidence: Optional[float] = None
        self._explanation: List[str] = []

    # ── Builder ────────────────────────────────────────────────────────────────

    @classmethod
    def from_result(cls, result: Any) -> "SceneGraph":
        """
        Tạo SceneGraph từ kết quả depth estimation hiện có.
        Tương thích với ReferenceFloodResult (object hoặc dict).
        """
        g = cls()
        get = _get_attr

        # --- Water ---
        water = WaterNode(
            area_pct=float(get(result, "water_area_pct") or
                           get(result, "water_region_area") or 0.0),
            color_confidence=float(get(result, "water_conf") or
                                   get(result, "water_detection_conf") or 0.5),
        )
        g.waters.append(water)

        # --- Persons ---
        refs = (get(result, "reference_objects") or
                get(result, "references") or [])
        for ref in (refs if isinstance(refs, list) else []):
            rtype = str(get(ref, "type") or get(ref, "class") or "")
            if "person" in rtype.lower() or "human" in rtype.lower():
                depth_val = (get(ref, "estimated_depth") or
                             get(ref, "flood_cm") or
                             get(ref, "depth"))
                person = PersonNode(
                    water_level_cm=float(depth_val) if depth_val else None,
                    confidence=float(get(ref, "confidence") or 0.5),
                    has_raincoat=bool(get(result, "has_raincoat")),
                )
                g.persons.append(person)

            elif any(v in rtype.lower() for v in ["car", "truck", "vehicle", "wheel"]):
                depth_val = (get(ref, "estimated_depth") or
                             get(ref, "flood_cm") or
                             get(ref, "depth"))
                vehicle = VehicleNode(
                    vehicle_type="car" if "car" in rtype.lower() else "truck",
                    water_level_cm=float(depth_val) if depth_val else None,
                    confidence=float(get(ref, "confidence") or 0.5),
                )
                g.vehicles.append(vehicle)

        # --- Context ---
        g.context = SceneContext(
            image_quality=float(get(result, "image_quality") or 0.6),
        )

        return g

    @classmethod
    def build(
        cls,
        persons: Optional[List[PersonNode]] = None,
        waters: Optional[List[WaterNode]] = None,
        vehicles: Optional[List[VehicleNode]] = None,
        road: Optional[RoadNode] = None,
        context: Optional[SceneContext] = None,
    ) -> "SceneGraph":
        """Builder tiện lợi khi có data riêng từng module."""
        g = cls()
        g.persons  = persons  or []
        g.waters   = waters   or []
        g.vehicles = vehicles or []
        g.road     = road     or RoadNode()
        g.context  = context  or SceneContext()
        return g

    # ── Reasoning ──────────────────────────────────────────────────────────────

    def unified_depth_reasoning(self) -> Tuple[Optional[float], str]:
        """
        Reasoning toàn bộ scene graph để ước tính độ sâu ngập (cm).

        Priority logic:
          1. Vehicle wheel → chính xác nhất (size cố định)
          2. Person pose → tốt nếu visibility cao
          3. Water area proxy → fallback

        Cross-validation: nếu nhiều nguồn đồng thuận (CV < 25%) → tin cậy cao

        Returns:
            (depth_cm, explanation_string)
        """
        self._explanation = []
        estimates: List[Tuple[float, float, str]] = []  # (depth, weight, source)

        # 1. Vehicle estimates (highest trust)
        for v in self.vehicles:
            d = v.estimated_flood_cm()
            if d is not None and d >= 0:
                weight = v.confidence * 1.2   # bonus weight
                estimates.append((d, weight, f"vehicle({v.vehicle_type})"))
                self._explanation.append(
                    f"🚗 Vehicle {v.vehicle_type}: {d:.1f}cm (conf={v.confidence:.2f})"
                )

        # 2. Person estimates
        for p in self.persons:
            d = p.estimated_flood_cm()
            if d is not None and d >= 0:
                weight = p.confidence * p.pose_visibility
                estimates.append((d, weight, "person"))
                self._explanation.append(
                    f"🧍 Person: {d:.1f}cm (conf={p.confidence:.2f}, "
                    f"visibility={p.pose_visibility:.2f})"
                )

        # 3. Water area proxy (fallback)
        for w in self.waters:
            if w.area_pct > 10:
                # crude proxy: nhiều nước → ngập nhiều hơn
                proxy_depth = min(w.area_pct * 0.8, 120.0)
                weight = w.overall_confidence() * 0.4  # low weight
                estimates.append((proxy_depth, weight, "water_area"))
                self._explanation.append(
                    f"💧 Water area {w.area_pct:.1f}%: proxy={proxy_depth:.1f}cm"
                )

        if not estimates:
            self._explanation.append("❌ Không đủ data để estimate depth")
            self._depth_cm = None
            return None, self.explain()

        # Weighted average
        depths  = np.array([e[0] for e in estimates])
        weights = np.array([e[1] for e in estimates])
        weights = weights / (weights.sum() + 1e-9)

        depth_cm = float(np.dot(depths, weights))
        cv = float(np.std(depths) / (np.mean(depths) + 1e-6))

        self._explanation.append(
            f"📊 Tổng hợp {len(estimates)} nguồn: {depth_cm:.1f}cm "
            f"(CV={cv:.2f}, {'✅ đồng thuận' if cv < 0.25 else '⚠️ phân kỳ'})"
        )

        self._depth_cm = depth_cm
        self._confidence = self._compute_confidence_from_graph(cv, len(estimates))
        return depth_cm, self.explain()

    def compute_confidence(self) -> float:
        """Tính confidence từ graph structure."""
        if self._depth_cm is None:
            self.unified_depth_reasoning()
        if self._confidence is not None:
            return self._confidence

        # Fallback
        has_vehicle = len(self.vehicles) > 0
        has_person  = len(self.persons) > 0
        has_water   = any(w.area_pct > 5 for w in self.waters)

        score = 0.3
        if has_water:   score += 0.2
        if has_person:  score += 0.2
        if has_vehicle: score += 0.3
        return min(score, 1.0)

    def explain(self) -> str:
        return "\n".join(self._explanation) if self._explanation else "No explanation"

    def to_dict(self) -> Dict:
        """Serialize graph cho logging / debug."""
        return {
            "n_persons":  len(self.persons),
            "n_vehicles": len(self.vehicles),
            "n_waters":   len(self.waters),
            "depth_cm":   self._depth_cm,
            "confidence": self._confidence,
            "explanation": self._explanation,
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _compute_confidence_from_graph(self, cv: float, n_sources: int) -> float:
        """Confidence từ CV (coefficient of variation) và số sources."""
        base = 1.0 - min(cv * 2.0, 0.8)    # CV cao → confidence thấp
        count_bonus = min(0.2, n_sources * 0.05)

        # Bonus nếu có vehicle (most reliable)
        vehicle_bonus = 0.15 if self.vehicles else 0.0

        # Bonus raincoat
        raincoat_bonus = 0.05 if any(p.has_raincoat for p in self.persons) else 0.0

        # Context penalty nếu image quality thấp
        quality_penalty = max(0.0, (0.4 - self.context.image_quality) * 0.5)

        confidence = base + count_bonus + vehicle_bonus + raincoat_bonus - quality_penalty
        return float(np.clip(confidence, 0.0, 1.0))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_attr(obj: Any, attr: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)
