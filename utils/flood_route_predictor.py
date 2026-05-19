# -*- coding: utf-8 -*-
"""
utils/flood_route_predictor.py
================================
Dự đoán tuyến đường an toàn dựa trên dữ liệu ngập lụt đã phân tích.

Workflow:
  1. Nhận vào list FloodPoint (vị trí GPS + mức ngập từ depth analysis)
  2. Tải đồ thị đường bộ từ OpenStreetMap (osmnx) trong bounding box
  3. Gán "flood penalty" cho từng đoạn đường dựa trên các điểm đo gần nhất
  4. Tìm đường ngắn nhất có thể đi được với Dijkstra (networkx)
  5. Xuất ra GeoJSON + HTML map (Folium) + báo cáo Excel

Mức ngập → khả năng đi được:
  dry / safe  (< 10 cm)  : xe máy + ô tô đều qua
  ankle       (10–30 cm) : ô tô qua, xe máy cẩn thận
  knee        (30–60 cm) : chỉ xe tải cao, xe máy không nên
  chest       (60–120cm) : không nên đi bất kỳ phương tiện nào
  submerged   (>120 cm)  : đường bị chặn hoàn toàn

Cài đặt:
  pip install osmnx networkx folium scipy
  (osmnx yêu cầu Python ≥ 3.9 và geopandas)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import math

log = logging.getLogger(__name__)

# ─── Flood level → vehicle passability ───────────────────────────────────────
FLOOD_LEVEL_CM: Dict[str, float] = {
    "dry":       0,
    "safe":      5,
    "ankle":     20,
    "knee":      45,
    "chest":     90,
    "submerged": 150,
}

# Penalty nhân vào edge weight (1.0 = bình thường, inf = chặn)
FLOOD_PENALTY: Dict[str, Dict[str, float]] = {
    #               motorbike  car     truck
    "dry":       {"motorbike": 1.0,  "car": 1.0,  "truck": 1.0},
    "safe":      {"motorbike": 1.1,  "car": 1.0,  "truck": 1.0},
    "ankle":     {"motorbike": 2.0,  "car": 1.2,  "truck": 1.1},
    "knee":      {"motorbike": float('inf'), "car": 5.0,  "truck": 2.0},
    "chest":     {"motorbike": float('inf'), "car": float('inf'), "truck": 10.0},
    "submerged": {"motorbike": float('inf'), "car": float('inf'), "truck": float('inf')},
}

INFLUENCE_RADIUS_M = 200   # Điểm đo ảnh hưởng các đường trong bán kính 200m


@dataclass
class FloodPoint:
    """Một điểm đo mức ngập từ depth analysis."""
    latitude:    float
    longitude:   float
    depth_cm:    float
    flood_level: str         # "dry","ankle","knee","chest","submerged"
    confidence:  float = 1.0
    image_path:  str = ""
    timestamp:   str = ""
    address:     str = ""


@dataclass
class RouteSegment:
    """Một đoạn đường trên tuyến dự đoán."""
    from_node: int
    to_node:   int
    length_m:  float
    flood_level: str = "dry"
    depth_cm:  float = 0.0
    passable_motorbike: bool = True
    passable_car:       bool = True
    passable_truck:     bool = True
    street_name: str = ""


@dataclass
class RouteResult:
    """Kết quả dự đoán tuyến đường."""
    origin:      Tuple[float, float]    # (lat, lon)
    destination: Tuple[float, float]
    vehicle:     str                    # "motorbike","car","truck"

    found:       bool = False
    total_distance_m: float = 0.0
    segments:    List[RouteSegment] = field(default_factory=list)

    max_flood_level: str = "dry"
    max_depth_cm:    float = 0.0
    risk_score:      float = 0.0       # 0.0 (an toàn) → 1.0 (nguy hiểm)

    warning:     str = ""
    geojson_path: str = ""
    map_html_path: str = ""

    timestamp:   str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["origin"]      = list(self.origin)
        d["destination"] = list(self.destination)
        return d


class FloodRoutePredictor:
    """
    Dự đoán tuyến đường an toàn trong vùng ngập lụt.
    """

    def __init__(
        self,
        output_dir: str = "output/routes",
        cache_graph: bool = True,
    ):
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_graph  = cache_graph
        self._graph_cache: Dict[str, Any] = {}   # bbox_key → graph

    # ── Main API ──────────────────────────────────────────────────────────────

    def predict_route(
        self,
        origin:      Tuple[float, float],
        destination: Tuple[float, float],
        flood_points: List[FloodPoint],
        vehicle: str = "motorbike",
    ) -> RouteResult:
        """
        Tìm tuyến đường an toàn nhất từ origin → destination
        tránh các vùng ngập được ghi nhận.

        Args:
            origin:       (lat, lon) điểm xuất phát
            destination:  (lat, lon) điểm đến
            flood_points: danh sách điểm đo ngập
            vehicle:      "motorbike" | "car" | "truck"

        Returns:
            RouteResult với segments và bản đồ HTML
        """
        result = RouteResult(
            origin=origin,
            destination=destination,
            vehicle=vehicle,
            timestamp=datetime.now().isoformat(),
        )

        try:
            import osmnx as ox
            import networkx as nx
        except ImportError:
            log.error("osmnx / networkx chưa được cài. Chạy: pip install osmnx networkx")
            result.warning = "Missing dependency: osmnx networkx"
            return result

        # 1. Tải đồ thị đường bộ
        log.info("[Route] Tải đồ thị đường bộ từ OSM...")
        G = self._get_graph(origin, destination, ox)

        if G is None or len(G.nodes) == 0:
            result.warning = "Không thể tải đồ thị đường bộ (kiểm tra kết nối internet)"
            return result

        # 2. Gán flood penalty
        log.info(f"[Route] Gán flood penalty từ {len(flood_points)} điểm đo...")
        G = self._apply_flood_penalties(G, flood_points, vehicle)

        # 3. Tìm node gần nhất với origin / destination
        try:
            orig_node = ox.nearest_nodes(G, X=origin[1], Y=origin[0])
            dest_node = ox.nearest_nodes(G, X=destination[1], Y=destination[0])
        except Exception as e:
            result.warning = f"Không tìm được node gần nhất: {e}"
            return result

        # 4. Dijkstra với weighted edge
        log.info("[Route] Tính đường ngắn nhất (Dijkstra)...")
        try:
            path_nodes = nx.shortest_path(
                G, orig_node, dest_node, weight="flood_weight"
            )
        except nx.NetworkXNoPath:
            result.warning = (
                f"Không tìm được đường đi cho {vehicle} "
                f"(tất cả đường đều bị ngập hoặc chặn)"
            )
            result.found = False
            return result

        # 5. Build segments
        result.segments, result.total_distance_m = self._build_segments(
            G, path_nodes
        )
        result.found = True

        # 6. Tính risk score và max flood
        result = self._compute_risk(result)

        # 7. Export GeoJSON + HTML map
        geojson_path = self._export_geojson(result, flood_points)
        result.geojson_path = str(geojson_path)

        map_path = self._export_map(result, flood_points, G, ox)
        result.map_html_path = str(map_path)

        log.info(
            f"[Route] Tuyến tìm được: {result.total_distance_m:.0f}m, "
            f"risk={result.risk_score:.2f}, max_flood={result.max_flood_level}"
        )
        return result

    def predict_routes_batch(
        self,
        flood_points: List[FloodPoint],
        origin:       Tuple[float, float],
        destinations: List[Tuple[float, float]],
        vehicle: str = "motorbike",
    ) -> List[RouteResult]:
        """Dự đoán nhiều tuyến đường từ 1 điểm xuất phát."""
        results = []
        for i, dest in enumerate(destinations, 1):
            log.info(f"[Route] Batch {i}/{len(destinations)}")
            r = self.predict_route(origin, dest, flood_points, vehicle)
            results.append(r)
        return results

    def build_flood_map(
        self,
        flood_points: List[FloodPoint],
        output_name: str = "flood_overview",
    ) -> str:
        """
        Tạo bản đồ HTML hiển thị tất cả điểm ngập (không cần origin/destination).
        Hữu ích để visualize dữ liệu trước khi route.
        """
        try:
            import folium
        except ImportError:
            log.error("folium chưa cài: pip install folium")
            return ""

        if not flood_points:
            return ""

        center_lat = sum(p.latitude for p in flood_points) / len(flood_points)
        center_lon = sum(p.longitude for p in flood_points) / len(flood_points)

        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        color_map = {
            "dry":       "#28a745",
            "safe":      "#6fcf97",
            "ankle":     "#f2c94c",
            "knee":      "#f2994a",
            "chest":     "#eb5757",
            "submerged": "#4f0000",
        }

        for pt in flood_points:
            color = color_map.get(pt.flood_level, "#888")
            popup_text = (
                f"<b>{pt.flood_level.upper()}</b><br>"
                f"Độ sâu: {pt.depth_cm:.0f} cm<br>"
                f"Confidence: {pt.confidence:.0%}<br>"
                f"{pt.address or ''}"
            )
            folium.CircleMarker(
                location=[pt.latitude, pt.longitude],
                radius=max(6, int(pt.depth_cm / 10)),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"{pt.flood_level} ({pt.depth_cm:.0f}cm)",
            ).add_to(m)

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:10px;border-radius:8px;
                    border:1px solid #ccc;font-size:13px;">
          <b>Mức ngập</b><br>
          <span style="color:#28a745">●</span> Khô / An toàn<br>
          <span style="color:#f2c94c">●</span> Mắt cá chân (10–30cm)<br>
          <span style="color:#f2994a">●</span> Đầu gối (30–60cm)<br>
          <span style="color:#eb5757">●</span> Ngực (60–120cm)<br>
          <span style="color:#4f0000">●</span> Chìm hoàn toàn (>120cm)
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        out_path = self.output_dir / f"{output_name}.html"
        m.save(str(out_path))
        log.info(f"[Route] Flood map saved: {out_path}")
        return str(out_path)

    # ── Internal Methods ──────────────────────────────────────────────────────

    def _get_graph(self, origin, destination, ox):
        """Tải đồ thị OSM với cache theo bounding box."""
        # Bounding box bao phủ cả 2 điểm + buffer 500m
        lats = [origin[0], destination[0]]
        lons = [origin[1], destination[1]]
        north = max(lats) + 0.005
        south = min(lats) - 0.005
        east  = max(lons) + 0.005
        west  = min(lons) - 0.005

        bbox_key = f"{north:.4f}_{south:.4f}_{east:.4f}_{west:.4f}"
        if self.cache_graph and bbox_key in self._graph_cache:
            return self._graph_cache[bbox_key]

        try:
            G = ox.graph_from_bbox(
                bbox=(north, south, east, west),
                network_type="drive",
                simplify=True,
            )
            if self.cache_graph:
                self._graph_cache[bbox_key] = G
            return G
        except Exception as e:
            log.error(f"[Route] Lỗi tải OSM graph: {e}")
            return None

    def _apply_flood_penalties(self, G, flood_points: List[FloodPoint], vehicle: str):
        """Gán flood_weight cho từng edge dựa trên flood points gần nhất."""
        import networkx as nx

        for u, v, key, data in G.edges(keys=True, data=True):
            # Tọa độ trung điểm của edge
            mid_lat, mid_lon = self._edge_midpoint(G, u, v, data)
            if mid_lat is None:
                continue

            # Tìm flood point gần nhất trong INFLUENCE_RADIUS_M
            nearest_level, nearest_depth = self._nearest_flood(
                mid_lat, mid_lon, flood_points
            )

            base_length = data.get("length", 1.0)
            penalty = FLOOD_PENALTY.get(nearest_level, {}).get(vehicle, 1.0)

            # flood_weight = length × penalty (inf nếu bị chặn)
            if penalty == float('inf'):
                G[u][v][key]["flood_weight"] = float('inf')
            else:
                G[u][v][key]["flood_weight"] = base_length * penalty

            # Lưu metadata để build segments
            G[u][v][key]["flood_level"] = nearest_level
            G[u][v][key]["flood_depth"] = nearest_depth

        return G

    def _edge_midpoint(self, G, u, v, data) -> Tuple[Optional[float], Optional[float]]:
        """Tính tọa độ trung điểm của edge."""
        try:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            lat = (u_data['y'] + v_data['y']) / 2
            lon = (u_data['x'] + v_data['x']) / 2
            return lat, lon
        except Exception:
            return None, None

    def _nearest_flood(
        self,
        lat: float,
        lon: float,
        flood_points: List[FloodPoint],
    ) -> Tuple[str, float]:
        """
        Tìm flood point gần nhất trong INFLUENCE_RADIUS_M.
        Returns: (flood_level, depth_cm). Mặc định "dry" nếu không có.
        """
        best_dist = float('inf')
        best_level = "dry"
        best_depth = 0.0

        for pt in flood_points:
            dist_m = _haversine_m(lat, lon, pt.latitude, pt.longitude)
            if dist_m <= INFLUENCE_RADIUS_M and dist_m < best_dist:
                best_dist  = dist_m
                best_level = pt.flood_level
                best_depth = pt.depth_cm

        return best_level, best_depth

    def _build_segments(
        self,
        G,
        path_nodes: List[int],
    ) -> Tuple[List[RouteSegment], float]:
        """Chuyển path_nodes thành danh sách RouteSegment."""
        segments = []
        total_dist = 0.0

        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            # Lấy edge tốt nhất (shortest)
            edge_data = min(
                G[u][v].values(),
                key=lambda d: d.get("length", float('inf'))
            )
            length     = edge_data.get("length", 0.0)
            flood_lv   = edge_data.get("flood_level", "dry")
            flood_dep  = edge_data.get("flood_depth", 0.0)
            street     = edge_data.get("name", "")
            if isinstance(street, list):
                street = street[0] if street else ""

            seg = RouteSegment(
                from_node=u, to_node=v,
                length_m=length,
                flood_level=flood_lv,
                depth_cm=flood_dep,
                passable_motorbike=FLOOD_PENALTY.get(flood_lv, {}).get("motorbike", 1.0) != float('inf'),
                passable_car=      FLOOD_PENALTY.get(flood_lv, {}).get("car",       1.0) != float('inf'),
                passable_truck=    FLOOD_PENALTY.get(flood_lv, {}).get("truck",     1.0) != float('inf'),
                street_name=street,
            )
            segments.append(seg)
            total_dist += length

        return segments, total_dist

    def _compute_risk(self, result: RouteResult) -> RouteResult:
        """Tính risk score và max flood level cho tuyến đường."""
        level_order = ["dry", "safe", "ankle", "knee", "chest", "submerged"]
        level_risk  = {k: i / (len(level_order) - 1) for i, k in enumerate(level_order)}

        if not result.segments:
            return result

        max_level_idx = 0
        weighted_risk = 0.0
        total_len     = max(result.total_distance_m, 1)

        for seg in result.segments:
            idx = level_order.index(seg.flood_level) if seg.flood_level in level_order else 0
            if idx > max_level_idx:
                max_level_idx = idx
                result.max_flood_level = seg.flood_level
                result.max_depth_cm    = seg.depth_cm

            seg_risk = level_risk.get(seg.flood_level, 0.0)
            weighted_risk += seg_risk * (seg.length_m / total_len)

        result.risk_score = round(weighted_risk, 3)

        # Cảnh báo
        if result.max_flood_level == "submerged":
            result.warning = "⛔ Tuyến đi qua vùng ngập hoàn toàn — KHÔNG AN TOÀN"
        elif result.max_flood_level == "chest":
            result.warning = "🚨 Tuyến đi qua vùng ngập ngực — CỰC KỲ NGUY HIỂM"
        elif result.max_flood_level == "knee":
            result.warning = "⚠️ Tuyến đi qua vùng ngập đầu gối — Chỉ xe tải mới qua được"
        elif result.max_flood_level == "ankle":
            result.warning = "⚠️ Tuyến đi qua vùng ngập mắt cá — Lái cẩn thận"
        else:
            result.warning = "✅ Tuyến an toàn"

        return result

    def _export_geojson(
        self,
        result: RouteResult,
        flood_points: List[FloodPoint],
    ) -> Path:
        """Xuất GeoJSON cho tuyến đường và các điểm ngập."""
        features = []

        # Route line
        if result.segments:
            # Simplified: dùng origin/destination làm LineString
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [result.origin[1],      result.origin[0]],
                        [result.destination[1], result.destination[0]],
                    ]
                },
                "properties": {
                    "type":             "route",
                    "vehicle":          result.vehicle,
                    "total_distance_m": result.total_distance_m,
                    "risk_score":       result.risk_score,
                    "max_flood_level":  result.max_flood_level,
                }
            })

        # Flood points
        for pt in flood_points:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type":        "Point",
                    "coordinates": [pt.longitude, pt.latitude],
                },
                "properties": {
                    "type":        "flood_point",
                    "flood_level": pt.flood_level,
                    "depth_cm":    pt.depth_cm,
                    "confidence":  pt.confidence,
                    "address":     pt.address,
                }
            })

        geojson = {"type": "FeatureCollection", "features": features}
        out = self.output_dir / f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson"
        out.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"[Route] GeoJSON saved: {out}")
        return out

    def _export_map(
        self,
        result: RouteResult,
        flood_points: List[FloodPoint],
        G,
        ox,
    ) -> Path:
        """Xuất HTML interactive map (Folium)."""
        try:
            import folium
        except ImportError:
            log.warning("folium chưa cài, bỏ qua HTML map")
            return self.output_dir / "route_no_map.html"

        center_lat = (result.origin[0] + result.destination[0]) / 2
        center_lon = (result.origin[1] + result.destination[1]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        # Flood points
        color_map = {
            "dry":       "#28a745", "safe":      "#6fcf97",
            "ankle":     "#f2c94c", "knee":      "#f2994a",
            "chest":     "#eb5757", "submerged": "#4f0000",
        }
        for pt in flood_points:
            folium.CircleMarker(
                location=[pt.latitude, pt.longitude],
                radius=max(5, int(pt.depth_cm / 10)),
                color=color_map.get(pt.flood_level, "#888"),
                fill=True,
                fill_color=color_map.get(pt.flood_level, "#888"),
                fill_opacity=0.7,
                tooltip=f"{pt.flood_level} {pt.depth_cm:.0f}cm",
            ).add_to(m)

        # Route (nếu tìm được)
        if result.found and result.segments:
            try:
                route_nodes = [s.from_node for s in result.segments]
                route_nodes.append(result.segments[-1].to_node)
                ox.plot_route_folium(G, route_nodes, route_map=m,
                                     color="#2d6a4f", weight=5, opacity=0.8)
            except Exception:
                # Fallback: vẽ đường thẳng
                folium.PolyLine(
                    [result.origin, result.destination],
                    color="#2d6a4f", weight=4, opacity=0.8,
                ).add_to(m)

        # Markers origin / destination
        folium.Marker(result.origin,  tooltip="Điểm xuất phát",
                      icon=folium.Icon(color="green",  icon="play")).add_to(m)
        folium.Marker(result.destination, tooltip="Điểm đến",
                      icon=folium.Icon(color="red",    icon="flag")).add_to(m)

        # Info box
        risk_pct = int(result.risk_score * 100)
        info_html = f"""
        <div style="position:fixed;top:20px;right:20px;z-index:1000;
                    background:white;padding:12px;border-radius:8px;
                    border:1px solid #ccc;min-width:200px;font-size:13px;">
          <b>Tuyến đường — {result.vehicle}</b><br>
          Khoảng cách: {result.total_distance_m:.0f} m<br>
          Mức ngập cao nhất: <b>{result.max_flood_level}</b>
            ({result.max_depth_cm:.0f} cm)<br>
          Risk score: <b>{risk_pct}%</b><br>
          <span style="color:{'red' if risk_pct>50 else 'green'}">{result.warning}</span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(info_html))

        out = self.output_dir / f"route_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        m.save(str(out))
        log.info(f"[Route] HTML map saved: {out}")
        return out

    def generate_report(self, results: List[RouteResult]) -> dict:
        """Tóm tắt báo cáo nhiều tuyến đường."""
        report = {
            "generated_at":  datetime.now().isoformat(),
            "total_routes":  len(results),
            "safe_routes":   sum(1 for r in results if r.risk_score < 0.2),
            "risky_routes":  sum(1 for r in results if 0.2 <= r.risk_score < 0.6),
            "blocked_routes":sum(1 for r in results if not r.found or r.risk_score >= 0.6),
            "routes": [
                {
                    "origin":      r.origin,
                    "destination": r.destination,
                    "vehicle":     r.vehicle,
                    "found":       r.found,
                    "distance_m":  r.total_distance_m,
                    "risk_score":  r.risk_score,
                    "max_flood":   r.max_flood_level,
                    "warning":     r.warning,
                    "map":         r.map_html_path,
                }
                for r in results
            ]
        }
        out = self.output_dir / "route_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"[Route] Report saved: {out}")
        return report


# ── Utility ───────────────────────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Khoảng cách Haversine tính bằng mét."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def depth_results_to_flood_points(depth_results, location_results) -> List[FloodPoint]:
    """
    Helper: chuyển đổi từ depth_results + location_results của pipeline hiện tại
    sang List[FloodPoint] để dùng với FloodRoutePredictor.

    Dùng trong main.py:
        from utils.flood_route_predictor import depth_results_to_flood_points
        flood_points = depth_results_to_flood_points(depth_results, location_results)
    """
    points = []
    for dr, lr in zip(depth_results, location_results):
        if lr is None or lr.latitude is None:
            continue  # Bỏ qua ảnh không có vị trí GPS

        depth_cm    = getattr(dr, 'depth_cm', 0) or 0
        flood_level = getattr(dr, 'flood_level', 'dry') or 'dry'
        confidence  = getattr(dr, 'confidence', 0.5) or 0.5

        points.append(FloodPoint(
            latitude=lr.latitude,
            longitude=lr.longitude,
            depth_cm=depth_cm,
            flood_level=flood_level,
            confidence=confidence,
            image_path=getattr(dr, 'image_path', ''),
            address=getattr(lr, 'address', ''),
            timestamp=datetime.now().isoformat(),
        ))

    log.info(f"[Route] Converted {len(points)}/{len(depth_results)} depth results to FloodPoints")
    return points


# ════════════════════════════════════════════════════════════════════════════
# 5.3 FLOOD SIMULATION INTEGRATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FloodSpreadCell:
    """Một ô trong grid mô phỏng lan rộng lũ."""
    lat:       float
    lon:       float
    depth_cm:  float
    flooded:   bool
    timestamp: str = ""


@dataclass
class FloodSimulationResult:
    """Kết quả mô phỏng lan rộng lũ."""
    origin_points:     List[FloodPoint]        # điểm đo ban đầu từ ảnh
    spread_cells:      List[FloodSpreadCell]   # các ô lan rộng dự đoán
    affected_area_km2: float
    max_depth_cm:      float
    risk_zones: Dict[str, List[Tuple[float, float]]]  # "high"/"medium"/"low" → list(lat,lon)
    map_html_path:     str
    geojson_path:      str
    simulation_time_s: float
    notes:             str


class FloodSimulator:
    """
    5.3 Flood simulation integration.

    Workflow:
      1. Ảnh → estimate water level (từ depth pipeline)
      2. Map → predict lan rộng theo địa hình
         - Cellular automaton: nước lan từ điểm cao → điểm thấp
         - Gravity-based: dựa trên DEM (Digital Elevation Model)
         - Influence radius: mỗi điểm đo ảnh hưởng bán kính R mét

    Không cần DEM: dùng heuristic đường phố (OSM altitude data).
    Có DEM: kết quả chính xác hơn nhiều.
    """

    SPREAD_RADIUS_M     = 300   # bán kính lan rộng mặc định (m)
    GRID_RESOLUTION_M   = 50    # độ phân giải grid (m)
    MAX_SPREAD_STEPS    = 8     # số bước cellular automaton

    # Tỉ lệ lan rộng theo mức độ
    SPREAD_FACTOR = {
        "submerged": 1.0,
        "chest":     0.85,
        "knee":      0.65,
        "ankle":     0.45,
        "safe":      0.20,
        "dry":       0.0,
    }

    def __init__(self, output_dir: str = "output/simulation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def simulate(
        self,
        flood_points: List[FloodPoint],
        spread_radius_m: float = SPREAD_RADIUS_M,
        use_elevation: bool = False,
        output_name: str = "flood_sim",
    ) -> FloodSimulationResult:
        """
        Mô phỏng lan rộng lũ từ các điểm đo.

        Args:
            flood_points:    điểm đo từ ảnh (lat, lon, depth_cm)
            spread_radius_m: bán kính lan rộng tối đa (m)
            use_elevation:   True để dùng SRTM elevation data (cần internet)
            output_name:     tên file output

        Returns:
            FloodSimulationResult với bản đồ HTML + GeoJSON
        """
        import time
        t0 = time.time()

        if not flood_points:
            return FloodSimulationResult(
                origin_points=[], spread_cells=[], affected_area_km2=0,
                max_depth_cm=0, risk_zones={}, map_html_path="",
                geojson_path="", simulation_time_s=0, notes="No flood points"
            )

        log.info(f"[Sim] Starting flood simulation: {len(flood_points)} seed points")

        # Lấy elevation data nếu cần
        elevation_cache: Dict[Tuple[float, float], float] = {}
        if use_elevation:
            elevation_cache = self._fetch_elevation_batch(flood_points)

        # Tạo grid spread cells
        spread_cells = self._cellular_automaton_spread(
            flood_points, spread_radius_m, elevation_cache
        )

        # Tính metrics
        affected_km2 = self._compute_area_km2(spread_cells)
        max_depth    = max((c.depth_cm for c in spread_cells if c.flooded), default=0)

        # Phân vùng rủi ro
        risk_zones = self._classify_risk_zones(spread_cells)

        # Tạo bản đồ HTML
        map_path  = self._build_simulation_map(
            flood_points, spread_cells, risk_zones, output_name
        )
        geo_path  = self._build_geojson(spread_cells, risk_zones, output_name)

        sim_time = time.time() - t0
        log.info(
            f"[Sim] Done: {len(spread_cells)} cells, "
            f"area={affected_km2:.2f} km², max_depth={max_depth:.0f}cm, "
            f"time={sim_time:.1f}s"
        )

        return FloodSimulationResult(
            origin_points     = flood_points,
            spread_cells      = spread_cells,
            affected_area_km2 = round(affected_km2, 3),
            max_depth_cm      = max_depth,
            risk_zones        = risk_zones,
            map_html_path     = map_path,
            geojson_path      = geo_path,
            simulation_time_s = round(sim_time, 2),
            notes = (
                f"Simulated from {len(flood_points)} image-derived flood points. "
                f"Spread radius: {spread_radius_m}m. "
                f"{'Elevation-aware.' if use_elevation else 'Flat-terrain approximation.'}"
            ),
        )

    # ── Cellular Automaton ───────────────────────────────────────────────────

    def _cellular_automaton_spread(
        self,
        seed_points: List[FloodPoint],
        radius_m: float,
        elevation: Dict,
    ) -> List[FloodSpreadCell]:
        """
        Mô phỏng lan rộng lũ bằng cellular automaton đơn giản.

        Thuật toán:
          1. Tạo grid cells trong bounding box
          2. Seed: assign depth cho các cell gần điểm đo
          3. Lặp: mỗi bước, nước lan sang cells lân cận thấp hơn
          4. Dừng khi không còn lan rộng hoặc hết bước
        """
        if not seed_points:
            return []

        # Bounding box + buffer
        lats = [p.latitude  for p in seed_points]
        lons = [p.longitude for p in seed_points]
        lat_min = min(lats) - radius_m / 111000
        lat_max = max(lats) + radius_m / 111000
        lon_min = min(lons) - radius_m / (111000 * math.cos(math.radians(sum(lats)/len(lats))))
        lon_max = max(lons) + radius_m / (111000 * math.cos(math.radians(sum(lats)/len(lats))))

        # Grid resolution
        lat_step = self.GRID_RESOLUTION_M / 111000
        lon_step = self.GRID_RESOLUTION_M / (111000 * math.cos(math.radians((lat_min + lat_max) / 2)))

        grid_lats = np.arange(lat_min, lat_max, lat_step)
        grid_lons = np.arange(lon_min, lon_max, lon_step)
        n_lat, n_lon = len(grid_lats), len(grid_lons)

        if n_lat * n_lon > 200000:
            # Grid quá lớn → coarsen
            factor = int(math.sqrt(n_lat * n_lon / 100000)) + 1
            grid_lats = grid_lats[::factor]
            grid_lons = grid_lons[::factor]
            n_lat, n_lon = len(grid_lats), len(grid_lons)

        # Depth grid: -1 = chưa xác định, 0 = khô, >0 = có nước (cm)
        depth_grid = np.full((n_lat, n_lon), -1.0)
        elev_grid  = np.zeros((n_lat, n_lon))

        # Điền elevation nếu có
        for i, lat in enumerate(grid_lats):
            for j, lon in enumerate(grid_lons):
                nearest_elev = self._nearest_elevation(lat, lon, elevation)
                elev_grid[i, j] = nearest_elev if nearest_elev is not None else 0.0

        # Seed từ flood_points
        for pt in seed_points:
            if pt.flood_level in ("dry", "safe") or pt.depth_cm <= 0:
                continue
            spread_factor = self.SPREAD_FACTOR.get(pt.flood_level, 0.3)
            # Tìm cell gần nhất
            i_seed = int((pt.latitude  - lat_min) / lat_step)
            j_seed = int((pt.longitude - lon_min) / lon_step)
            i_seed = max(0, min(n_lat - 1, i_seed))
            j_seed = max(0, min(n_lon - 1, j_seed))
            depth_grid[i_seed, j_seed] = pt.depth_cm

            # Spread trong radius (confidence-weighted)
            r_cells = int(radius_m * spread_factor / self.GRID_RESOLUTION_M)
            for di in range(-r_cells, r_cells + 1):
                for dj in range(-r_cells, r_cells + 1):
                    ni, nj = i_seed + di, j_seed + dj
                    if 0 <= ni < n_lat and 0 <= nj < n_lon:
                        dist_m = math.sqrt(di**2 + dj**2) * self.GRID_RESOLUTION_M
                        if dist_m <= radius_m * spread_factor:
                            # Depth giảm theo khoảng cách
                            decay = max(0, 1.0 - (dist_m / (radius_m * spread_factor)) ** 1.5)
                            new_depth = pt.depth_cm * decay * pt.confidence
                            if new_depth > depth_grid[ni, nj]:
                                depth_grid[ni, nj] = new_depth

        # Cellular automaton: lan sang cells lân cận
        for step in range(self.MAX_SPREAD_STEPS):
            changed = False
            new_grid = depth_grid.copy()
            for i in range(1, n_lat - 1):
                for j in range(1, n_lon - 1):
                    if depth_grid[i, j] > 5:  # có nước
                        # Lan sang 4 hướng
                        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ni, nj = i + di, j + dj
                            if depth_grid[ni, nj] < 0:  # chưa xác định
                                # Kiểm tra elevation (nước chảy xuống chỗ thấp hơn)
                                elev_diff = elev_grid[i, j] - elev_grid[ni, nj]
                                if elev_diff >= 0:  # nước chảy xuống
                                    spread_depth = depth_grid[i, j] * 0.7 + elev_diff * 10
                                    if spread_depth > 2:
                                        new_grid[ni, nj] = max(0, spread_depth * 0.8)
                                        changed = True
            depth_grid = new_grid
            if not changed:
                break

        # Chuyển grid → cells
        cells = []
        ts = datetime.now().isoformat()
        for i, lat in enumerate(grid_lats):
            for j, lon in enumerate(grid_lons):
                d = depth_grid[i, j]
                if d > 0:
                    cells.append(FloodSpreadCell(
                        lat=float(lat), lon=float(lon),
                        depth_cm=round(float(d), 1),
                        flooded=True, timestamp=ts,
                    ))

        log.info(f"[Sim] Automaton: {len(cells)} flooded cells from {n_lat}×{n_lon} grid")
        return cells

    # ── Analysis ────────────────────────────────────────────────────────────

    def _compute_area_km2(self, cells: List[FloodSpreadCell]) -> float:
        """Tính diện tích bị ngập (km²) từ grid cells."""
        n_flooded = sum(1 for c in cells if c.flooded)
        cell_area_km2 = (self.GRID_RESOLUTION_M / 1000) ** 2
        return n_flooded * cell_area_km2

    def _classify_risk_zones(
        self, cells: List[FloodSpreadCell]
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Phân chia vùng rủi ro theo mức độ ngập."""
        zones: Dict[str, List] = {"high": [], "medium": [], "low": []}
        for c in cells:
            if not c.flooded:
                continue
            if c.depth_cm >= 60:
                zones["high"].append((c.lat, c.lon))
            elif c.depth_cm >= 20:
                zones["medium"].append((c.lat, c.lon))
            else:
                zones["low"].append((c.lat, c.lon))
        return zones

    # ── Map Building ─────────────────────────────────────────────────────────

    def _build_simulation_map(
        self,
        seed_points: List[FloodPoint],
        cells: List[FloodSpreadCell],
        risk_zones: Dict,
        output_name: str,
    ) -> str:
        """Tạo bản đồ HTML với folium."""
        try:
            import folium
            from folium.plugins import HeatMap
        except ImportError:
            log.warning("[Sim] folium not installed, skipping map")
            return ""

        all_lats = [c.lat for c in cells] + [p.latitude for p in seed_points]
        all_lons = [c.lon for c in cells] + [p.longitude for p in seed_points]
        if not all_lats:
            return ""

        center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]
        m = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron")

        # HeatMap từ flooded cells
        heat_data = [
            [c.lat, c.lon, min(c.depth_cm / 150.0, 1.0)]
            for c in cells if c.flooded
        ]
        if heat_data:
            HeatMap(
                heat_data,
                radius=20, blur=15,
                gradient={"0.0": "blue", "0.4": "yellow", "0.7": "orange", "1.0": "red"},
                min_opacity=0.4,
            ).add_to(m)

        # Seed points (điểm đo từ ảnh)
        level_colors = {
            "submerged": "#1a0000", "chest": "#cc0000",
            "knee": "#ff6600",      "ankle": "#ffcc00",
            "safe": "#66cc00",      "dry":   "#00cc44",
        }
        for pt in seed_points:
            color = level_colors.get(pt.flood_level, "#888")
            folium.CircleMarker(
                location=[pt.latitude, pt.longitude],
                radius=10,
                color="white", weight=2,
                fill=True, fill_color=color, fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>📸 Ảnh thực tế</b><br>"
                    f"Mức: {pt.flood_level}<br>"
                    f"Sâu: {pt.depth_cm:.0f} cm<br>"
                    f"Conf: {pt.confidence:.0%}",
                    max_width=200,
                ),
                tooltip=f"Ảnh: {pt.flood_level} ({pt.depth_cm:.0f}cm)",
            ).add_to(m)

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                    background:rgba(255,255,255,0.92);padding:12px 16px;
                    border-radius:10px;border:1px solid #ccc;font-size:13px;
                    box-shadow:2px 2px 8px rgba(0,0,0,0.15);">
          <b>🌊 Flood Simulation</b><br>
          <div style="height:12px;background:linear-gradient(to right,blue,yellow,red);
                      border-radius:4px;margin:6px 0;"></div>
          Thấp → Trung bình → Nguy hiểm<br>
          <span style="color:#888">● Điểm đo từ ảnh</span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        out_path = self.output_dir / f"{output_name}.html"
        m.save(str(out_path))
        log.info(f"[Sim] Saved simulation map: {out_path}")
        return str(out_path)

    def _build_geojson(
        self,
        cells: List[FloodSpreadCell],
        risk_zones: Dict,
        output_name: str,
    ) -> str:
        """Xuất GeoJSON để dùng với GIS tools khác."""
        features = []
        for c in cells:
            if not c.flooded:
                continue
            # Xác định risk level
            if c.depth_cm >= 60:
                risk = "high"
            elif c.depth_cm >= 20:
                risk = "medium"
            else:
                risk = "low"

            half = self.GRID_RESOLUTION_M / 2 / 111000
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [c.lon - half, c.lat - half],
                        [c.lon + half, c.lat - half],
                        [c.lon + half, c.lat + half],
                        [c.lon - half, c.lat + half],
                        [c.lon - half, c.lat - half],
                    ]],
                },
                "properties": {
                    "depth_cm":  c.depth_cm,
                    "risk_zone": risk,
                    "flooded":   True,
                    "timestamp": c.timestamp,
                },
            })

        geojson = {"type": "FeatureCollection", "features": features}
        out_path = self.output_dir / f"{output_name}.geojson"
        with open(str(out_path), "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        log.info(f"[Sim] Saved GeoJSON: {out_path} ({len(features)} features)")
        return str(out_path)

    # ── Elevation ────────────────────────────────────────────────────────────

    def _fetch_elevation_batch(
        self, points: List[FloodPoint]
    ) -> Dict[Tuple[float, float], float]:
        """
        Lấy elevation từ Open-Elevation API (miễn phí).
        Fallback về 0 nếu không có internet.
        """
        import urllib.request
        cache: Dict[Tuple[float, float], float] = {}
        try:
            locations = [{"latitude": p.latitude, "longitude": p.longitude} for p in points]
            payload   = json.dumps({"locations": locations}).encode()
            url  = "https://api.open-elevation.com/api/v1/lookup"
            req  = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            for r in data.get("results", []):
                cache[(r["latitude"], r["longitude"])] = r.get("elevation", 0.0)
            log.info(f"[Sim] Fetched elevation for {len(cache)} points")
        except Exception as e:
            log.debug(f"[Sim] Elevation fetch failed: {e}, using flat terrain")
        return cache

    def _nearest_elevation(
        self, lat: float, lon: float, cache: Dict
    ) -> Optional[float]:
        """Tìm elevation gần nhất trong cache."""
        if not cache:
            return None
        best_dist = float("inf")
        best_elev = 0.0
        for (clat, clon), elev in cache.items():
            d = (lat - clat) ** 2 + (lon - clon) ** 2
            if d < best_dist:
                best_dist = d
                best_elev = elev
        return best_elev if best_dist < 0.01 else None


# ── CLI demo / quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Tạo vài điểm ngập giả (Hà Nội example)
    flood_pts = [
        FloodPoint(21.0285, 105.8542, depth_cm=45, flood_level="knee",    confidence=0.85, address="Hàng Bài"),
        FloodPoint(21.0260, 105.8520, depth_cm=80, flood_level="chest",   confidence=0.90, address="Tràng Tiền"),
        FloodPoint(21.0300, 105.8560, depth_cm=15, flood_level="ankle",   confidence=0.75, address="Đinh Tiên Hoàng"),
        FloodPoint(21.0240, 105.8490, depth_cm= 5, flood_level="safe",    confidence=0.95, address="Lý Thái Tổ"),
    ]

    predictor = FloodRoutePredictor(output_dir="output/routes")

    # Tạo flood overview map
    map_path = predictor.build_flood_map(flood_pts, output_name="flood_demo")
    print(f"Flood map: {map_path}")

    # Dự đoán tuyến đường
    origin      = (21.0285, 105.8542)
    destination = (21.0240, 105.8490)

    print(f"\nTìm tuyến {origin} → {destination} cho xe máy...")
    result = predictor.predict_route(origin, destination, flood_pts, vehicle="motorbike")

    print(f"\n=== Kết quả ===")
    print(f"Tìm được tuyến: {result.found}")
    print(f"Khoảng cách:   {result.total_distance_m:.0f} m")
    print(f"Risk score:    {result.risk_score:.2%}")
    print(f"Mức ngập cao:  {result.max_flood_level} ({result.max_depth_cm:.0f} cm)")
    print(f"Cảnh báo:      {result.warning}")
    print(f"Bản đồ:        {result.map_html_path}")
