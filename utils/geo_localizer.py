# -*- coding: utf-8 -*-
"""
utils/geo_localizer.py
=======================
Nhận diện vị trí từ ảnh bằng nhiều phương pháp, kết hợp lại thành
1 kết quả duy nhất kèm radius (bán kính không chắc chắn).

Thứ tự ưu tiên:
  1. GPS EXIF          → radius 50m   (chính xác nhất)
  2. Browser coords    → radius 200m  (WiFi/cell, không cần GPS bật)
  3. OCR biển hiệu     → radius 400m  (tìm thấy tên đường)
  4. GeoCLIP (GPU)     → radius 2000m (AI đoán từ nội dung ảnh)
  5. IP geolocation    → radius 5000m (fallback cuối, chỉ ra quận/huyện)

Mỗi nguồn trả về GeoEstimate với lat, lon, radius_m, method, confidence.
UI vẽ vòng tròn radius tương ứng để người xem biết mức độ chính xác.
"""

import json
import logging
import re
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

log = logging.getLogger("utils.geo_localizer")

# Bán kính theo từng phương pháp (mét)
RADIUS = {
    "gps_exif":    50,
    "browser":     200,
    "ocr_street":  400,
    "ocr_district":1500,
    "geoclip":     2000,
    "ip":          5000,
    "none":        10000,
}


@dataclass
class GeoEstimate:
    lat:        Optional[float] = None
    lon:        Optional[float] = None
    radius_m:   int   = 10000      # bán kính vòng tròn hiển thị trên map (mét)
    method:     str   = "none"     # nguồn dữ liệu
    confidence: float = 0.0
    address:    str   = ""         # địa chỉ text (nếu có)
    province:   str   = ""
    district:   str   = ""
    street:     str   = ""

    def has_coords(self) -> bool:
        return self.lat is not None and self.lon is not None

    def to_dict(self) -> dict:
        return {
            "lat":        self.lat,
            "lon":        self.lon,
            "radius_m":   self.radius_m,
            "method":     self.method,
            "confidence": round(self.confidence, 3),
            "address":    self.address,
            "province":   self.province,
            "district":   self.district,
            "street":     self.street,
        }


class GeoLocalizer:
    """
    Nhận diện vị trí từ ảnh — kết hợp EXIF + OCR + GeoCLIP + browser + IP.

    Dùng:
        loc = GeoLocalizer()
        est = loc.localize(
            image_path=Path("flood.jpg"),
            browser_lat=21.59, browser_lon=105.84, browser_accuracy=150,
            client_ip="1.2.3.4",
        )
        print(est.lat, est.lon, est.radius_m, est.method)
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.gmaps_key   = cfg.get("google_maps_key", "")
        self._geoclip    = None   # lazy init
        self._ocr_reader = None   # lazy init

    # ── Public ────────────────────────────────────────────────────────────────

    def localize(
        self,
        image_path: Optional[Path] = None,
        browser_lat: Optional[float] = None,
        browser_lon: Optional[float] = None,
        browser_accuracy: Optional[float] = None,  # mét
        client_ip: Optional[str] = None,
    ) -> GeoEstimate:
        """
        Chạy tất cả phương pháp theo thứ tự ưu tiên, trả về kết quả tốt nhất.
        """
        # 1. GPS EXIF — chính xác nhất
        if image_path:
            est = self._from_exif(image_path)
            if est.has_coords():
                log.info("[Geo] GPS EXIF: %.5f, %.5f", est.lat, est.lon)
                return est

        # 2. Browser geolocation — WiFi/cell, không cần GPS bật
        if browser_lat and browser_lon:
            acc = browser_accuracy or 500
            est = GeoEstimate(
                lat=browser_lat, lon=browser_lon,
                radius_m=max(int(acc * 1.5), RADIUS["browser"]),
                method="browser", confidence=0.85,
            )
            self._enrich_address(est)
            log.info("[Geo] Browser: %.5f, %.5f ±%dm", est.lat, est.lon, est.radius_m)
            return est

        # 3. OCR biển hiệu — tìm tên đường/địa chỉ trong ảnh
        if image_path:
            est = self._from_ocr(image_path)
            if est.has_coords():
                log.info("[Geo] OCR: %s → %.5f, %.5f", est.address, est.lat, est.lon)
                return est

        # 4. GeoCLIP — AI đoán vị trí từ nội dung ảnh (GPU)
        if image_path:
            est = self._from_geoclip(image_path)
            if est.has_coords():
                log.info("[Geo] GeoCLIP: %.5f, %.5f (conf=%.2f)", est.lat, est.lon, est.confidence)
                return est

        # 5. IP geolocation — fallback cuối, ra quận/huyện
        if client_ip:
            est = self._from_ip(client_ip)
            if est.has_coords():
                log.info("[Geo] IP: %.5f, %.5f (%s)", est.lat, est.lon, client_ip)
                return est

        return GeoEstimate(method="none")

    # ── Method 1: EXIF GPS ────────────────────────────────────────────────────

    def _from_exif(self, image_path: Path) -> GeoEstimate:
        try:
            import exifread
            with open(image_path, "rb") as f:
                tags = exifread.process_file(f, details=False, stop_tag="GPS GPSLongitude")
            lat = _parse_gps_tag(tags.get("GPS GPSLatitude"), tags.get("GPS GPSLatitudeRef"))
            lon = _parse_gps_tag(tags.get("GPS GPSLongitude"), tags.get("GPS GPSLongitudeRef"))
            if lat and lon:
                est = GeoEstimate(lat=lat, lon=lon, radius_m=RADIUS["gps_exif"],
                                  method="gps_exif", confidence=0.99)
                self._enrich_address(est)
                return est
        except Exception as e:
            log.debug("[Geo] EXIF: %s", e)
        return GeoEstimate()

    # ── Method 2: OCR → địa chỉ → geocode ───────────────────────────────────

    def _from_ocr(self, image_path: Path) -> GeoEstimate:
        texts = self._run_ocr(image_path)
        if not texts:
            return GeoEstimate()

        full = " ".join(texts)
        log.debug("[Geo] OCR raw: %s", full[:200])

        # Extract địa chỉ từ text
        addr_info = extract_vn_address(full)
        if not addr_info:
            return GeoEstimate()

        query = addr_info.get("query", "")
        if not query:
            return GeoEstimate()

        coords = _nominatim_geocode(query)
        if not coords:
            return GeoEstimate()

        method = "ocr_street" if addr_info.get("street") else "ocr_district"
        return GeoEstimate(
            lat=coords[0], lon=coords[1],
            radius_m=RADIUS[method],
            method=method,
            confidence=addr_info.get("confidence", 0.55),
            address=query,
            street=addr_info.get("street", ""),
            district=addr_info.get("district", ""),
            province=addr_info.get("province", ""),
        )

    def _run_ocr(self, image_path: Path) -> List[str]:
        """Chạy OCR, thử PaddleOCR trước rồi EasyOCR."""
        try:
            from paddleocr import PaddleOCR
            if self._ocr_reader is None:
                self._ocr_reader = PaddleOCR(use_angle_cls=True, lang="vi",
                                             use_gpu=True, show_log=False)
            result = self._ocr_reader.ocr(str(image_path), cls=True)
            if result and result[0]:
                return [line[1][0] for line in result[0] if line[1][1] > 0.3]
        except Exception:
            pass

        try:
            import easyocr
            if self._ocr_reader is None or not hasattr(self._ocr_reader, "readtext"):
                self._ocr_reader = easyocr.Reader(["vi", "en"], gpu=True, verbose=False)
            results = self._ocr_reader.readtext(str(image_path))
            return [r[1] for r in results if r[2] > 0.3]
        except Exception as e:
            log.debug("[Geo] OCR error: %s", e)
        return []

    # ── Method 3: GeoCLIP ─────────────────────────────────────────────────────

    def _from_geoclip(self, image_path: Path) -> GeoEstimate:
        """
        Dùng GeoCLIP để đoán vị trí từ nội dung ảnh.
        pip install geoclip   (model ~2GB, tự download lần đầu)
        """
        try:
            if self._geoclip is None:
                from geoclip import GeoCLIP
                self._geoclip = GeoCLIP()
                log.info("[Geo] GeoCLIP model loaded")

            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            # top_k=1: lấy dự đoán tốt nhất
            preds = self._geoclip.predict(img, top_k=3)
            # preds: list of (lat, lon, prob)
            if not preds:
                return GeoEstimate()

            best = max(preds, key=lambda x: x[2])
            lat, lon, prob = best

            # Lọc: chỉ dùng nếu tọa độ nằm trong khoảng Việt Nam
            # (tránh đoán sai sang nước khác)
            if not _in_vietnam_bbox(lat, lon):
                # Thử lấy pred nào trong VN
                vn_preds = [(la, lo, p) for la, lo, p in preds if _in_vietnam_bbox(la, lo)]
                if vn_preds:
                    lat, lon, prob = max(vn_preds, key=lambda x: x[2])
                else:
                    log.debug("[Geo] GeoCLIP predicted outside VN bbox: %.4f, %.4f", lat, lon)
                    return GeoEstimate()

            est = GeoEstimate(
                lat=float(lat), lon=float(lon),
                radius_m=RADIUS["geoclip"],
                method="geoclip",
                confidence=float(prob),
            )
            self._enrich_address(est)
            return est

        except ImportError:
            log.warning("[Geo] geoclip chưa cài: pip install geoclip")
        except Exception as e:
            log.debug("[Geo] GeoCLIP error: %s", e)
        return GeoEstimate()

    # ── Method 4: IP geolocation ──────────────────────────────────────────────

    def _from_ip(self, ip: str) -> GeoEstimate:
        """ip-api.com — miễn phí, không cần key, giới hạn 45 req/phút."""
        if not ip or ip in ("127.0.0.1", "::1", "localhost"):
            return GeoEstimate()
        try:
            url  = f"http://ip-api.com/json/{ip}?fields=status,lat,lon,city,regionName&lang=vi"
            req  = urllib.request.Request(url, headers={"User-Agent": "FloodPipeline/1.0"})
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read())
            if data.get("status") == "success":
                return GeoEstimate(
                    lat=float(data["lat"]), lon=float(data["lon"]),
                    radius_m=RADIUS["ip"],
                    method="ip",
                    confidence=0.40,
                    district=data.get("city", ""),
                    province=data.get("regionName", ""),
                    address=f"{data.get('city', '')}, {data.get('regionName', '')}",
                )
        except Exception as e:
            log.debug("[Geo] IP geo error: %s", e)
        return GeoEstimate()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _enrich_address(self, est: GeoEstimate):
        """Thêm địa chỉ text từ tọa độ bằng Nominatim (free)."""
        if not est.has_coords() or est.address:
            return
        try:
            url = (f"https://nominatim.openstreetmap.org/reverse"
                   f"?lat={est.lat}&lon={est.lon}&format=json&accept-language=vi")
            req  = urllib.request.Request(url, headers={"User-Agent": "FloodPipeline/1.0"})
            resp = urllib.request.urlopen(req, timeout=8)
            data = json.loads(resp.read())
            addr = data.get("address", {})
            est.address  = data.get("display_name", "")[:120]
            est.street   = addr.get("road") or addr.get("pedestrian", "")
            est.district = addr.get("city_district") or addr.get("suburb", "")
            est.province = addr.get("city") or addr.get("state", "")
        except Exception:
            pass


# ── Vietnamese address extraction ─────────────────────────────────────────────

# Danh sách tỉnh/thành phố + tọa độ trung tâm (fallback khi không geocode được)
VN_PROVINCES = {
    "hà nội": (21.0245, 105.8412), "hà nội": (21.0245, 105.8412),
    "hồ chí minh": (10.8231, 106.6297), "đà nẵng": (16.0544, 108.2022),
    "cần thơ": (10.0452, 105.7469), "hải phòng": (20.8449, 106.6881),
    "thái nguyên": (21.5942, 105.8481), "bắc ninh": (21.1860, 106.0763),
    "nam định": (20.4388, 106.1621), "ninh bình": (20.2506, 105.9745),
    "thanh hóa": (19.8071, 105.7851), "nghệ an": (18.6696, 105.6813),
    "hà tĩnh": (18.3560, 105.8877), "quảng bình": (17.4689, 106.6220),
    "quảng trị": (16.7403, 107.1854), "thừa thiên huế": (16.4637, 107.5909),
    "quảng nam": (15.5394, 108.0191), "quảng ngãi": (15.1214, 108.8044),
    "bình định": (13.7765, 109.2236), "phú yên": (13.0882, 109.0929),
    "khánh hòa": (12.2388, 109.1967), "bình thuận": (11.0904, 108.0721),
    "lâm đồng": (11.5753, 108.1429), "đắk lắk": (12.7100, 108.2378),
    "gia lai": (13.9833, 108.0000), "kon tum": (14.3545, 108.0076),
    "bình dương": (11.3254, 106.4770), "đồng nai": (10.9574, 107.1706),
    "bà rịa vũng tàu": (10.5417, 107.2429), "long an": (10.5354, 106.4101),
    "tiền giang": (10.4493, 106.3421), "bến tre": (10.2434, 106.3757),
    "trà vinh": (9.9477, 106.3415), "vĩnh long": (10.2397, 105.9571),
    "đồng tháp": (10.4938, 105.6882), "an giang": (10.5216, 105.1259),
    "kiên giang": (9.8250, 105.1259), "hậu giang": (9.7579, 105.6413),
    "sóc trăng": (9.6030, 105.9739), "bạc liêu": (9.2940, 105.7217),
    "cà mau": (9.1769, 105.1500),
}


def extract_vn_address(text: str) -> Optional[dict]:
    """
    Trích xuất thông tin địa chỉ Việt Nam từ text OCR.

    Tìm các pattern:
      - "123 Đường Lương Ngọc Quyến" / "45 Phố Hoàng Văn Thụ"
      - "Phường Trưng Vương, Thái Nguyên"
      - "Quận Đống Đa" / "Huyện Phổ Yên"
    """
    text_lower = text.lower()

    # --- Tìm số nhà + tên đường ---
    street_patterns = [
        r"(?:số\s*)?(\d+\s*[a-z]?)\s*(?:đường|phố|ph\.|đ\.)\s*([^\n,;]+)",
        r"(?:đường|phố)\s+([^\n,;0-9]{4,40})",
    ]
    street = ""
    for pat in street_patterns:
        m = re.search(pat, text_lower)
        if m:
            street = m.group(len(m.groups())).strip().title()
            street = re.sub(r"\s+", " ", street)[:50]
            break

    # --- Tìm phường/xã ---
    ward_m = re.search(r"(?:phường|p\.|xã|thị trấn)\s+([^\n,;0-9]{2,30})", text_lower)
    ward   = ward_m.group(1).strip().title() if ward_m else ""

    # --- Tìm quận/huyện ---
    dist_m = re.search(r"(?:quận|q\.|huyện|h\.|thị xã|tx\.)\s*([^\n,;]{2,30})", text_lower)
    district = dist_m.group(1).strip().title() if dist_m else ""

    # --- Tìm tỉnh/thành phố ---
    province = ""
    for prov_name in sorted(VN_PROVINCES.keys(), key=len, reverse=True):
        if prov_name in text_lower:
            province = prov_name.title()
            break

    if not any([street, ward, district, province]):
        return None

    # Xây query geocode theo thứ tự chi tiết nhất có thể
    parts = []
    if street:    parts.append(street)
    if ward:      parts.append(ward)
    if district:  parts.append(district)
    if province:  parts.append(province)
    parts.append("Việt Nam")

    confidence = 0.35
    if street:   confidence += 0.25
    if district: confidence += 0.15
    if province: confidence += 0.10

    return {
        "street":   street,
        "ward":     ward,
        "district": district,
        "province": province,
        "query":    ", ".join(parts),
        "confidence": min(confidence, 0.85),
    }


# ── Nominatim geocode ─────────────────────────────────────────────────────────

def _nominatim_geocode(query: str) -> Optional[Tuple[float, float]]:
    """Text → (lat, lon) qua Nominatim OpenStreetMap (miễn phí)."""
    try:
        q   = urllib.parse.quote(query)
        url = (f"https://nominatim.openstreetmap.org/search"
               f"?q={q}&format=json&limit=1&countrycodes=vn")
        req  = urllib.request.Request(url, headers={"User-Agent": "FloodPipeline/1.0"})
        resp = urllib.request.urlopen(req, timeout=8)
        data = json.loads(resp.read())
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        log.debug("[Geo] Nominatim error: %s", e)
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _in_vietnam_bbox(lat: float, lon: float) -> bool:
    """Kiểm tra tọa độ có nằm trong bounding box Việt Nam không."""
    return 8.0 <= lat <= 23.5 and 102.0 <= lon <= 110.0


def _parse_gps_tag(tag, ref_tag) -> Optional[float]:
    """Parse EXIF GPS tag sang decimal degrees."""
    if tag is None:
        return None
    try:
        vals = tag.values
        def to_float(v):
            return float(v.num) / float(v.den) if hasattr(v, "num") else float(v)
        deg = to_float(vals[0]) + to_float(vals[1]) / 60 + to_float(vals[2]) / 3600
        if ref_tag and str(ref_tag.values) in ("S", "W"):
            deg = -deg
        return round(deg, 7)
    except Exception:
        return None
