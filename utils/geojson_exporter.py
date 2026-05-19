# -*- coding: utf-8 -*-
"""
utils/geojson_exporter.py
==========================
Xuất kết quả phân tích lũ sang GeoJSON để hiển thị trên bản đồ Leaflet / MapBox.

Dùng:
    from utils.geojson_exporter import export_geojson
    path = export_geojson(state, output_dir)

Output: FeatureCollection với mỗi ảnh có GPS là 1 Feature (Point).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.orchestrator import PipelineState

log = logging.getLogger("utils.geojson")

# Màu hiển thị theo mức lũ (dùng cho Leaflet marker)
LEVEL_COLOR = {
    "dry":       "#4CAF50",   # xanh lá
    "ankle":     "#FFEB3B",   # vàng
    "knee":      "#FF9800",   # cam
    "waist":     "#F44336",   # đỏ
    "chest":     "#9C27B0",   # tím
    "submerged": "#1A237E",   # tím đậm
    "unknown":   "#9E9E9E",   # xám
}

LEVEL_VI = {
    "dry":       "Khô ráo",
    "ankle":     "Ngập mắt cá (~20cm)",
    "knee":      "Ngập đầu gối (~50cm)",
    "waist":     "Ngập ngang eo (~90cm)",
    "chest":     "Ngập ngang ngực (~130cm)",
    "submerged": "Ngập hoàn toàn (>150cm)",
    "unknown":   "Không xác định",
}


def _get_coords(result: Any, location_map: Dict[str, Any]) -> Optional[tuple]:
    """Trích xuất (longitude, latitude) từ result + location_map."""
    # Thử lấy từ location_map theo original_path
    img_key = getattr(result, "original_path", None) or getattr(result, "image_path", None)
    if img_key:
        loc = location_map.get(str(img_key))
        if loc:
            lat = loc.get("latitude") or loc.get("lat")
            lon = loc.get("longitude") or loc.get("lon") or loc.get("lng")
            if lat is not None and lon is not None:
                return (float(lon), float(lat))
    # Thử đọc trực tiếp từ result
    lat = getattr(result, "latitude", None)
    lon = getattr(result, "longitude", None)
    if lat is not None and lon is not None:
        return (float(lon), float(lat))
    return None


def results_to_features(
    depth_results: List[Any],
    location_map: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """Chuyển depth_results sang list GeoJSON Feature."""
    location_map = location_map or {}
    features = []

    for r in depth_results:
        coords = _get_coords(r, location_map)
        if coords is None:
            continue  # bỏ qua ảnh không có GPS

        level      = getattr(r, "flood_level", "unknown") or "unknown"
        depth_cm   = getattr(r, "depth_cm", 0) or 0
        confidence = getattr(r, "confidence", 0.0) or 0.0
        img_path   = str(getattr(r, "original_path", "") or getattr(r, "image_path", ""))

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": list(coords),  # [lon, lat] theo chuẩn GeoJSON
            },
            "properties": {
                "flood_level":    level,
                "flood_level_vi": LEVEL_VI.get(level, level),
                "depth_cm":       round(float(depth_cm), 1),
                "confidence":     round(float(confidence), 3),
                "marker_color":   LEVEL_COLOR.get(level, LEVEL_COLOR["unknown"]),
                "image_path":     img_path,
            },
        }

        # Thêm địa chỉ nếu có
        loc = location_map.get(img_path)
        if loc:
            feature["properties"]["address"] = loc.get("address", "")
            feature["properties"]["district"] = loc.get("district", "")
            feature["properties"]["city"] = loc.get("city", "")

        features.append(feature)

    return features


def export_geojson(
    state: "PipelineState",
    output_dir: Optional[Path] = None,
    filename: str = "flood_results.geojson",
) -> Optional[Path]:
    """
    Xuất PipelineState sang file .geojson.

    Args:
        state: PipelineState sau khi pipeline chạy xong
        output_dir: thư mục output (mặc định dùng state.output_dir)
        filename: tên file output

    Returns:
        Path đến file .geojson đã tạo, hoặc None nếu không có dữ liệu GPS
    """
    out_dir = output_dir or state.output_dir
    if not out_dir:
        log.warning("[GeoJSON] output_dir chưa được set")
        return None

    features = results_to_features(state.depth_results, state.location_map)

    if not features:
        log.info("[GeoJSON] Không có ảnh nào có GPS — bỏ qua export")
        return None

    collection = {
        "type": "FeatureCollection",
        "metadata": {
            "run_id":         state.run_id,
            "total_features": len(features),
            "generated_by":   "FloodPipeline / geojson_exporter",
        },
        "features": features,
    }

    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(collection, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("[GeoJSON] Đã xuất %d điểm → %s", len(features), out_path)
    return out_path


def export_danger_zones(
    state: "PipelineState",
    output_dir: Optional[Path] = None,
    min_level: str = "knee",
    filename: str = "danger_zones.geojson",
) -> Optional[Path]:
    """
    Giống export_geojson nhưng chỉ lấy các điểm nguy hiểm (>= min_level).

    Args:
        min_level: mức lũ tối thiểu để đưa vào danger zone
                   ("ankle" | "knee" | "waist" | "chest" | "submerged")
    """
    LEVEL_ORDER = ["dry", "ankle", "knee", "waist", "chest", "submerged"]
    min_idx = LEVEL_ORDER.index(min_level) if min_level in LEVEL_ORDER else 2

    filtered = [
        r for r in state.depth_results
        if LEVEL_ORDER.index(
            getattr(r, "flood_level", "dry") or "dry"
        ) >= min_idx
        if (getattr(r, "flood_level", "dry") or "dry") in LEVEL_ORDER
    ]

    if not filtered:
        return None

    # Tạm thời swap depth_results để gọi lại export
    class _FakeState:
        def __init__(self, orig, results):
            self.run_id = orig.run_id
            self.output_dir = orig.output_dir
            self.location_map = orig.location_map
            self.depth_results = results

    return export_geojson(_FakeState(state, filtered), output_dir, filename)
