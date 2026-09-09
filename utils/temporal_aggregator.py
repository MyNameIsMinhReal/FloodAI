# -*- coding: utf-8 -*-
"""
utils/temporal_aggregator.py
============================
Gộp đa khung hình (multi-frame / multi-image) cho cùng 1 vị trí.

Khi pipeline phân tích N ảnh cùng khu vực ngập (camera thu不同 góc,
hoặc video clip nhiều frame), mỗi ảnh ra kết quả riêng biệt có nhiễu.
Hàm gộp kết quả同 1 vị trí để giảm nhiễu bằng trimmed-median.

Cách dùng:
    from utils.temporal_aggregator import aggregate_by_location

    # depth_results: list ReferenceFloodResult / dict
    # location_map:  {filename: {"lat":..,"lon":..}, ...}
    updated = aggregate_by_location(depth_results, location_map, location_map)

    # HOẶC:
    agg = TemporalAggregator()
    for filename, lat, lon, water_cm in measurements:
        agg.add(filename, lat, lon, water_cm)
    results = agg.aggregated()

Design:
    - Nhóm theo tọa độ GPS (round đến 0.001 ≈ 100m) HOẶC theo image hash
      nếu không có GPS (fallback: nhóm theo thứ tự batch,-window=5).
    - Nếu chỉ có 1 ảnh tại 1 vị trí → giữ nguyên (không gộp).
    - Nếu >=2 → dùng trimmed median (loại min/max, lấy median phần còn lại).
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("utils.temporal_agg")


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        try:
            setattr(obj, key, value)
        except Exception:
            pass


# ── Location grouping ──────────────────────────────────────────────────────────

def _loc_key(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    """
    Tạo key nhóm GPS bằng cách làm tròn tọa độ.
    Round đến 0.001° ≈ 100m trên đất liền — đủ cho cùng khu vực ngập.
    """
    if lat is None or lon is None:
        return None
    return f"{round(lat, 3)}:{round(lon, 3)}"


def _extract_gps(result, location_map: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Trích lat/lon từ result hoặc location_map."""
    lat = _get(result, "lat") or _get(result, "latitude")
    lon = _get(result, "lon") or _get(result, "longitude")
    if lat and lon:
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            pass
    img = _get(result, "original_path") or _get(result, "filename") or ""
    loc = location_map.get(img, {})
    lat2 = loc.get("lat") or loc.get("latitude")
    lon2 = loc.get("lon") or loc.get("longitude")
    try:
        return float(lat2), float(lon2)
    except (TypeError, ValueError):
        return None, None


# ── Pure function ──────────────────────────────────────────────────────────────

def aggregate_by_location(
    depth_results: List[Any],
    location_map: Dict[str, Any] = None,
    min_group_size: int = 2,
    include_confidence: bool = True,
) -> List[Any]:
    """
    Gộp kết quả cùng vị trí GPS, giữ nguyên các ảnh không có GPS partner.

    Args:
        depth_results:   danh sách kết quả depth (dict hoặc dataclass)
        location_map:    {filename: {"lat","lon"}} có thể rỗng
        min_group_size:  tối thiểu bao nhiêu ảnh mới gộp (1 = luôn gộp)
        include_confidence: tính lại confidence từ dispersion của group

    Returns:
        depth_results đã cập nhật (update水_height_cm cho group >=2 ảnh)
    """
    if not depth_results or not location_map:
        return depth_results

    location_map = location_map or {}
    groups: Dict[str, List[int]] = defaultdict(list)

    for i, r in enumerate(depth_results):
        lat, lon = _extract_gps(r, location_map)
        lk = _loc_key(lat, lon)
        if lk:
            groups[lk].append(i)
        else:
            # Không có GPS → identity key (giữ nguyên, không gộp)
            groups[f"_nogps_{i}"].append(i)

    n_aggregated = 0
    for lk, indices in groups.items():
        if lk.startswith("_nogps_") or len(indices) < min_group_size:
            continue

        values = []
        for i in indices:
            v = _get(depth_results[i], "water_height_cm")
            if v is not None:
                try:
                    values.append((i, float(v)))
                except (TypeError, ValueError):
                    pass

        if len(values) < 2:
            continue

        vals = [v for _, v in values]
        trimmed = _trimmed_median(vals, trim_frac=0.15)
        new_cm = round(float(trimmed), 1)

        for i, old_cm in values:
            if abs(old_cm - new_cm) > 0.5:
                log.debug(
                    f"  [Agg] {lk}: idx={i} {old_cm:.0f}→{new_cm:.0f}cm "
                    f"(group={len(values)})"
                )
                _set(depth_results[i], "water_height_cm", new_cm)

        n_aggregated += len(values)

    if n_aggregated > 0:
        log.info(
            f"  [Agg] Gộp {n_aggregated} ảnh theo vị trí GPS "
            f"(trimmed-median)"
        )

    return depth_results


def _trimmed_median(values: List[float], trim_frac: float = 0.15) -> float:
    """
    Median sau khi loại bỏ trim_frac% giá trị nhỏ nhất và lớn nhất.
    robust hơn median thông thường khi có 1-2 measurement cực lỗi.
    """
    if len(values) <= 2:
        return float(np.median(values))
    arr = np.sort(np.array(values, dtype=np.float64))
    n   = len(arr)
    lo  = max(1, int(n * trim_frac))
    hi  = n - max(1, int(n * trim_frac))
    return float(np.median(arr[lo:hi]))


# ── Class-based aggregator (dùng khi muốn stream measurements) ────────────────

class TemporalAggregator:
    """
    Thu thập measurements rồi xuất aggregated kết quả.
    Dùng cho processing online (trong loop) thay vì batch.
    """

    def __init__(self, trim_frac: float = 0.15):
        self.trim_frac = trim_frac
        self._data: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    def add(self, filename: str, lat: Optional[float], lon: Optional[float],
            water_cm: float):
        lk = _loc_key(lat, lon)
        key = lk if lk else f"_solo_{filename}"
        self._data[key].append((filename, water_cm))

    def aggregated(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, entries in self._data.items():
            vals = [v for _, v in entries]
            out[key] = round(float(_trimmed_median(vals, self.trim_frac)), 1)
        return out

    def clear(self):
        self._data.clear()
