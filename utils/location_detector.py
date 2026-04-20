# -*- coding: utf-8 -*-
"""
utils/location_detector.py
---------------------------
Nhan dien vi tri dia ly cua anh lu lut bang nhieu phuong phap:

  1. EXIF GPS     - Lay toa do GPS tu metadata anh (chinh xac nhat)
  2. License Plate- Nhan dien bien so xe -> xac dinh tinh/thanh pho
  3. Text OCR     - Doc ten duong, bien hieu bang chu
  4. GeoGuessr-style Visual - So sanh dac trung anh voi database (experimental)
  5. Reverse Geocoding - Tu toa do GPS -> ten duong, phuong, quan, tinh

Kết quả duoc them vao report va overlay anh.

Cai dat:
    pip install pillow exifread googlemaps easyocr
    pip install paddleocr  (optional, tot hon cho tieng Viet)
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json

log = logging.getLogger(__name__)


@dataclass
class LocationResult:
    image_path:   str
    method:       str          # "gps", "plate", "ocr_text", "none"
    latitude:     Optional[float] = None
    longitude:    Optional[float] = None
    address:      str = ""     # dia chi day du
    province:     str = ""     # tinh/thanh pho
    district:     str = ""     # quan/huyen
    ward:         str = ""     # phuong/xa
    street:       str = ""     # ten duong
    plate_number: str = ""     # bien so xe (neu nhan dien duoc)
    plate_region: str = ""     # tinh dua tren bien so
    confidence:   float = 0.0
    notes:        str = ""
    map_url:      str = ""     # link Google Maps


class LocationDetector:
    """Nhan dien vi tri dia ly tu anh."""

    def __init__(
        self,
        google_maps_key:    str  = "",
        use_ocr:            bool = True,
        use_plate:          bool = True,
        use_exif:           bool = True,
        use_streetview:     bool = True,   # So sanh voi Street View
    ):
        self.gmaps_key       = google_maps_key
        self.use_ocr         = use_ocr
        self.use_plate       = use_plate
        self.use_exif        = use_exif
        self.use_streetview  = use_streetview
        self._ocr_reader     = None
        self._sv_matcher     = None   # lazy StreetViewMatcher

    # ==================================================================
    def detect(self, image_path: Path) -> LocationResult:
        """
        Nhan dien vi tri bang tat ca phuong phap co san.
        Thu theo thu tu uu tien: GPS -> OCR -> Plate -> None
        """
        image_path = Path(image_path)
        result = LocationResult(image_path=str(image_path), method="none")

        # --- Phuong phap 1: GPS EXIF (chinh xac nhat) ---
        if self.use_exif:
            gps = self._extract_gps_exif(image_path)
            if gps:
                lat, lon = gps
                result.latitude   = lat
                result.longitude  = lon
                result.method     = "gps"
                result.confidence = 0.99
                result.map_url    = f"https://maps.google.com/đq={lat},{lon}"
                # Reverse geocode
                addr = self._reverse_geocode(lat, lon)
                if addr:
                    result.address  = addr.get("address", "")
                    result.province = addr.get("province", "")
                    result.district = addr.get("district", "")
                    result.ward     = addr.get("ward", "")
                    result.street   = addr.get("street", "")
                log.info(f"  GPS found: {lat:.6f},{lon:.6f} -> {result.address}")
                return result

        # --- Phuong phap 2: OCR text (bien hieu, ten duong) ---
        if self.use_ocr:
            ocr_result = self._detect_by_ocr(image_path)
            if ocr_result:
                result.method     = "ocr_text"
                result.street     = ocr_result.get("street", "")
                result.province   = ocr_result.get("province", "")
                result.address    = ocr_result.get("address", "")
                result.confidence = ocr_result.get("confidence", 0.5)
                # Neu co ten duong -> geocode
                if result.street or result.address:
                    coords = self._geocode(result.address or result.street)
                    if coords:
                        result.latitude  = coords[0]
                        result.longitude = coords[1]
                        result.map_url   = f"https://maps.google.com/đq={coords[0]},{coords[1]}"
                log.info(f"  OCR location: {result.address} (conf={result.confidence:.2f})")

        # --- Phuong phap 3: License plate ---
        if self.use_plate:
            plate_result = self._detect_license_plate(image_path)
            if plate_result and result.method == "none":
                result.method       = "plate"
                result.plate_number = plate_result.get("plate", "")
                result.plate_region = plate_result.get("region", "")
                result.province     = plate_result.get("province", "")
                result.confidence   = plate_result.get("confidence", 0.6)
                result.notes        = f"Bien so: {result.plate_number}"
                log.info(f"  Plate: {result.plate_number} -> {result.province}")
            elif plate_result and result.method != "none":
                # Ket hop bien so de xac nhan tinh
                result.plate_number = plate_result.get("plate", "")
                if not result.province:
                    result.province = plate_result.get("province", "")
                result.notes += f" | Bien so: {result.plate_number}"

        # --- Phuong phap 4: Street View Visual Match ---
        if self.use_streetview and result.method == "none":
            sv_result = self._match_streetview(image_path)
            if sv_result:
                result.method     = sv_result.method
                result.latitude   = sv_result.latitude
                result.longitude  = sv_result.longitude
                result.address    = sv_result.landmark_name
                result.confidence = sv_result.confidence
                result.notes      = sv_result.notes
                if sv_result.latitude and result.method not in ("gps",):
                    # Enrich voi reverse geocode
                    addr = self._reverse_geocode(sv_result.latitude, sv_result.longitude)
                    if addr:
                        result.province = addr.get("province", "")
                        result.district = addr.get("district", "")
                        result.street   = addr.get("street", result.street)
                        if not result.address:
                            result.address = addr.get("address", "")
                log.info(f"  StreetView match: {result.address} (conf={result.confidence:.2f})")

        # Tao map_url neu chua co nhung co toa do
        if not result.map_url and result.latitude and result.longitude:
            result.map_url = f"https://maps.google.com/đq={result.latitude},{result.longitude}"

        return result

    def _match_streetview(self, image_path: Path):
        """Lazy-load va chay StreetViewMatcher."""
        try:
            if self._sv_matcher is None:
                from utils.streetview_matcher import StreetViewMatcher
                self._sv_matcher = StreetViewMatcher(
                    google_api_key   = self.gmaps_key,
                    use_vision_api   = bool(self.gmaps_key),
                    use_visual_match = True,
                    use_sign_detection = True,
                )
            return self._sv_matcher.match(image_path)
        except Exception as e:
            log.debug(f"  StreetView match failed: {e}")
            return None

    def detect_batch(self, image_paths: List[Path]) -> List[LocationResult]:
        """Nhan dien vi tri cho nhieu anh."""
        results = []
        located = 0
        for i, p in enumerate(image_paths, 1):
            r = self.detect(p)
            results.append(r)
            if r.method != "none":
                located += 1
            if i % 10 == 0:
                log.info(f"  Location [{i}/{len(image_paths)}]: {located} located")
        log.info(f"  Location detection: {located}/{len(image_paths)} images located")
        return results

    # ==================================================================
    # GPS EXIF
    # ==================================================================
    def _extract_gps_exif(self, image_path: Path) -> Optional[Tuple[float, float]]:
        """Lay toa do GPS tu EXIF metadata cua anh."""
        # Method 1: Dung PIL/Pillow
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS, GPSTAGS

            img  = Image.open(str(image_path))
            exif = img._getexif()
            if not exif:
                return None

            gps_info = {}
            for tag, val in exif.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "GPSInfo":
                    for gps_tag, gps_val in val.items():
                        sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                        gps_info[sub_tag] = gps_val

            if not gps_info:
                return None

            lat = self._dms_to_decimal(
                gps_info.get("GPSLatitude"),
                gps_info.get("GPSLatitudeRef", "N")
            )
            lon = self._dms_to_decimal(
                gps_info.get("GPSLongitude"),
                gps_info.get("GPSLongitudeRef", "E")
            )
            if lat is not None and lon is not None:
                return lat, lon
        except Exception as e:
            log.debug(f"  EXIF PIL failed: {e}")

        # Method 2: Dung exifread
        try:
            import exifread
            with open(str(image_path), "rb") as f:
                tags = exifread.process_file(f, details=False)

            lat_tag = tags.get("GPS GPSLatitude")
            lon_tag = tags.get("GPS GPSLongitude")
            lat_ref = str(tags.get("GPS GPSLatitudeRef", "N"))
            lon_ref = str(tags.get("GPS GPSLongitudeRef", "E"))

            if lat_tag and lon_tag:
                lat = self._parse_exifread_coord(str(lat_tag), lat_ref)
                lon = self._parse_exifread_coord(str(lon_tag), lon_ref)
                if lat and lon:
                    return lat, lon
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"  EXIF exifread failed: {e}")

        return None

    def _dms_to_decimal(self, dms, ref) -> Optional[float]:
        """Chuyen doi DMS (degrees, minutes, seconds) sang decimal degrees."""
        if not dms or len(dms) < 3:
            return None
        try:
            deg = float(dms[0])
            mn  = float(dms[1])
            sec = float(dms[2])
            val = deg + mn/60 + sec/3600
            if ref in ("S", "W"):
                val = -val
            return val
        except Exception:
            return None

    def _parse_exifread_coord(self, coord_str: str, ref: str) -> Optional[float]:
        """Parse exifread coordinate string '[d, m, s]' -> decimal."""
        try:
            parts = coord_str.strip("[]").split(", ")
            vals  = []
            for p in parts:
                if "/" in p:
                    n, d = p.split("/")
                    vals.append(float(n) / float(d))
                else:
                    vals.append(float(p))
            val = vals[0] + vals[1]/60 + vals[2]/3600
            if ref in ("S", "W"):
                val = -val
            return val
        except Exception:
            return None

    # ==================================================================
    # OCR TEXT DETECTION
    # ==================================================================
    def _detect_by_ocr(self, image_path: Path) -> Optional[dict]:
        """
        Nhan dien chu tren bien hieu, ten duong trong anh.
        Tim cac pattern: "Duong ...", "Pho ...", "Quan ...", "P.", "Q."
        """
        texts = self._run_ocr(image_path)
        if not texts:
            return None

        full_text = " ".join(texts)
        log.debug(f"  OCR texts: {full_text[:200]}")

        result = {
            "street": "", "province": "", "district": "",
            "address": "", "confidence": 0.0
        }
        found = False

        # --- Tim ten duong ---
        street_patterns = [
            r"(đ:duong|ph[o\u1ed1]|d\.)\s+([A-Za-z\u00C0-\u024F\s]+)",
            r"(đ:DU\u1EARENG|PH\u1ED0)\s+([A-Za-z\u00C0-\u024F\s]+)",
            r"(\d+[A-Z]đ\s*[A-Za-z\u00C0-\u024F\s]+(đ:STREET|ST|ROAD|RD|AVE))",
        ]
        for pattern in street_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                result["street"]     = match.group(1).strip()
                result["confidence"] = 0.65
                found = True
                break

        # --- Tim quan/huyen ---
        district_patterns = [
            r"(đ:quan|q\.)\s*(\d+|[A-Za-z\u00C0-\u024F\s]+)",
            r"(đ:huyen|h\.)\s+([A-Za-z\u00C0-\u024F\s]+)",
            r"DISTRICT\s+(\d+|[A-Z\s]+)",
        ]
        for pattern in district_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                result["district"]   = match.group(1).strip()
                result["confidence"] = max(result["confidence"], 0.55)
                found = True
                break

        # --- Tim tinh/thanh pho ---
        vn_provinces = [
            "Ha Noi", "Ho Chi Minh", "Da Nang", "Can Tho", "Hai Phong",
            "Hue", "Nha Trang", "Da Lat", "Vung Tau", "Bien Hoa",
            "Thai Nguyen", "Bac Ninh", "Nam Dinh", "Ninh Binh",
            "Thanh Hoa", "Vinh", "Ha Tinh", "Quang Binh", "Quang Tri",
            "Quang Nam", "Quang Ngai", "Binh Dinh", "Phu Yen",
            "Khanh Hoa", "Ninh Thuan", "Binh Thuan", "Kon Tum",
            "Gia Lai", "Dak Lak", "Dak Nong", "Lam Dong",
            "Binh Phuoc", "Tay Ninh", "Binh Duong", "Dong Nai",
            "Ba Ria", "Long An", "Tien Giang", "Ben Tre", "Tra Vinh",
            "Vinh Long", "Dong Thap", "An Giang", "Kien Giang",
            "Hau Giang", "Soc Trang", "Bac Lieu", "Ca Mau",
        ]
        for province in vn_provinces:
            # Kiem tra ca accent va khong accent
            if province.lower() in full_text.lower():
                result["province"]   = province
                result["confidence"] = max(result["confidence"], 0.70)
                found = True
                break

        if found:
            # Build address string
            parts = []
            if result["street"]:
                parts.append(result["street"])
            if result["district"]:
                parts.append(f"Quan {result['district']}")
            if result["province"]:
                parts.append(result["province"])
            result["address"] = ", ".join(parts)

        return result if found else None

    def _run_ocr(self, image_path: Path) -> List[str]:
        """Chay OCR, tra ve list cac text nhan dien duoc."""
        texts = []

        # Method 1: EasyOCR (ho tro tieng Viet tot)
        try:
            import easyocr
            if self._ocr_reader is None:
                log.info("  Loading EasyOCR (vi + en) ...")
                self._ocr_reader = easyocr.Reader(["vi", "en"], verbose=False)
            results = self._ocr_reader.readtext(str(image_path))
            texts   = [r[1] for r in results if r[2] > 0.3]
            if texts:
                return texts
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"  EasyOCR failed: {e}")

        # Method 2: PaddleOCR (rat tot cho tieng Viet)
        try:
            from paddleocr import PaddleOCR
            if self._ocr_reader is None:
                self._ocr_reader = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
            result = self._ocr_reader.ocr(str(image_path), cls=True)
            if result and result[0]:
                texts = [line[1][0] for line in result[0] if line[1][1] > 0.3]
            if texts:
                return texts
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"  PaddleOCR failed: {e}")

        # Method 3: pytesseract (fallback)
        try:
            import pytesseract
            from PIL import Image
            img  = Image.open(str(image_path))
            text = pytesseract.image_to_string(img, lang="vie+eng")
            texts = [t.strip() for t in text.split("\n") if t.strip()]
        except Exception:
            pass

        return texts

    # ==================================================================
    # LICENSE PLATE DETECTION
    # ==================================================================
    def _detect_license_plate(self, image_path: Path) -> Optional[dict]:
        """
        Nhan dien bien so xe Viet Nam va xac dinh tinh.
        Bien so VN: XX-YYYY.ZZ (tinh code - so - so)
        """
        # Bien so tinh/thanh pho Viet Nam
        PLATE_REGIONS = {
            "11": "Cao Bang", "12": "Lang Son", "14": "Quang Ninh",
            "15": "Hai Phong", "16": "Hai Phong", "17": "Thai Binh",
            "18": "Nam Dinh",  "19": "Ninh Binh", "20": "Thai Nguyen",
            "21": "Yen Bai",   "22": "Tuyen Quang","23": "Ha Giang",
            "24": "Lao Cai",   "25": "Lai Chau",  "26": "Son La",
            "27": "Dien Bien", "28": "Hoa Binh",  "29": "Ha Noi",
            "30": "Ha Noi",    "31": "Ha Noi",    "32": "Ha Noi",
            "33": "Ha Noi",    "34": "Hung Yen",  "35": "Ha Nam",
            "36": "Thanh Hoa", "37": "Nghe An",   "38": "Ha Tinh",
            "40": "Vinh Phuc", "41": "Ho Chi Minh","43": "Da Nang",
            "47": "Dak Lak",   "48": "Dak Nong",  "49": "Lam Dong",
            "50": "Ho Chi Minh","51": "Ho Chi Minh","52": "Ho Chi Minh",
            "53": "Ho Chi Minh","54": "Ho Chi Minh","55": "Ho Chi Minh",
            "56": "Ho Chi Minh","57": "Ho Chi Minh","58": "Ho Chi Minh",
            "59": "Ho Chi Minh","60": "Dong Nai",  "61": "Binh Duong",
            "62": "Long An",    "63": "Tien Giang","64": "Vinh Long",
            "65": "Can Tho",    "66": "Dong Thap", "67": "An Giang",
            "68": "Kien Giang", "69": "Ca Mau",    "70": "Tay Ninh",
            "71": "Ben Tre",    "72": "Ba Ria",    "73": "Quang Binh",
            "74": "Quang Tri",  "75": "Thua Thien Hue","76": "Quang Ngai",
            "77": "Binh Dinh",  "78": "Phu Yen",   "79": "Khanh Hoa",
            "81": "Gia Lai",    "82": "Kon Tum",   "83": "Soc Trang",
            "84": "Tra Vinh",   "85": "Ninh Thuan","86": "Binh Thuan",
            "88": "Vinh Long",  "89": "Hau Giang", "90": "Bac Lieu",
            "92": "Quang Nam",  "93": "Bac Giang", "97": "Bac Kan",
            "98": "Bac Giang",  "99": "Phu Tho",
        }

        texts = self._run_ocr(image_path)
        if not texts:
            return self._detect_plate_by_vision(image_path, PLATE_REGIONS)

        full_text = " ".join(texts)

        # Tim bien so VN: pattern XX[A-Z]-NNNN.NN hoac XX-NNNN.NN
        patterns = [
            r"\b(\d{2}[A-Z]{1,2}[-\s]đ\d{3,4}[.\s]đ\d{2,3})\b",
            r"\b(\d{2}[-\s][A-Z]{1,2}\d{3,4}[.\s]đ\d{2,3})\b",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for plate in matches:
                plate_clean = re.sub(r"[\s]", "", plate)
                prefix = plate_clean[:2]
                if prefix in PLATE_REGIONS:
                    return {
                        "plate":      plate_clean,
                        "region":     prefix,
                        "province":   PLATE_REGIONS[prefix],
                        "confidence": 0.75,
                    }

        # Fallback: chi lay 2 so dau
        num_match = re.search(r"\b(\d{2})[A-Z]", full_text)
        if num_match:
            prefix = num_match.group(1)
            if prefix in PLATE_REGIONS:
                return {
                    "plate":      num_match.group(0),
                    "region":     prefix,
                    "province":   PLATE_REGIONS[prefix],
                    "confidence": 0.5,
                }

        return self._detect_plate_by_vision(image_path, PLATE_REGIONS)

    def _detect_plate_by_vision(self, image_path: Path, regions: dict) -> Optional[dict]:
        """
        Dung YOLO hoac OpenCV de phat hien vi tri bien so, sau do chay OCR.
        """
        try:
            import cv2
            import numpy as np

            img  = cv2.imread(str(image_path))
            if img is None:
                return None
            h, w = img.shape[:2]

            # Tim vung bien so bang mau sac (bien VN: nen vang hoac trang)
            hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Bien trang (xe may, xe con moi)
            white_mask = cv2.inRange(hsv, np.array([0,0,200]), np.array([180,40,255]))
            # Bien vang (xe may cu, bien ngoai tinh)
            yellow_mask= cv2.inRange(hsv, np.array([20,80,150]), np.array([35,255,255]))

            combined = cv2.bitwise_or(white_mask, yellow_mask)
            kernel   = np.ones((3,3), np.uint8)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            plate_candidates = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                ar = cw / max(ch, 1)
                # Bien so VN: ty le chieu rong/cao ~ 2.5-5.0
                if 2.0 < ar < 6.0 and 20 < ch < h*0.15 and cw > 50:
                    plate_candidates.append((x, y, cw, ch, ar))

            # Sort theo dien tich, lay lon nhat
            plate_candidates.sort(key=lambda x: -(x[2]*x[3]))

            for x, y, cw, ch, ar in plate_candidates[:3]:
                roi = img[max(0,y-5):min(h,y+ch+5), max(0,x-5):min(w,x+cw+5)]
                if roi.size == 0:
                    continue
                # Chay OCR tren ROI
                roi_path = image_path.parent / f"_tmp_plate_{x}_{y}.jpg"
                cv2.imwrite(str(roi_path), roi)
                texts = self._run_ocr(roi_path)
                try:
                    roi_path.unlink()
                except Exception:
                    pass
                if texts:
                    text = " ".join(texts)
                    num_match = re.search(r"\b(\d{2})", text)
                    if num_match:
                        prefix = num_match.group(1)
                        if prefix in regions:
                            return {
                                "plate":      text[:15],
                                "region":     prefix,
                                "province":   regions[prefix],
                                "confidence": 0.55,
                            }
        except Exception as e:
            log.debug(f"  Vision plate detect failed: {e}")
        return None

    # ==================================================================
    # GEOCODING
    # ==================================================================
    def _reverse_geocode(self, lat: float, lon: float) -> Optional[dict]:
        """
        Chuyen toa do GPS -> dia chi cu the.
        Thu Google Maps API truoc, fallback sang Nominatim (free).
        """
        # Method 1: Google Maps Geocoding API
        if self.gmaps_key:
            try:
                import googlemaps
                gmaps  = googlemaps.Client(key=self.gmaps_key)
                result = gmaps.reverse_geocode((lat, lon), language="vi")
                if result:
                    comp   = result[0].get("address_components", [])
                    parsed = self._parse_google_components(comp)
                    parsed["address"] = result[0].get("formatted_address", "")
                    return parsed
            except Exception as e:
                log.debug(f"  Google Maps reverse geocode failed: {e}")

        # Method 2: Nominatim (free, no API key)
        try:
            import urllib.request
            url  = (
                f"https://nominatim.openstreetmap.org/reverse"
                f"đlat={lat}&lon={lon}&format=json&accept-language=vi"
            )
            req  = urllib.request.Request(url, headers={"User-Agent": "FloodPipeline/1.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            addr = data.get("address", {})
            return {
                "address":  data.get("display_name", ""),
                "street":   addr.get("road", addr.get("pedestrian", "")),
                "ward":     addr.get("suburb", addr.get("neighbourhood", "")),
                "district": addr.get("city_district", addr.get("county", "")),
                "province": addr.get("city", addr.get("state", "")),
            }
        except Exception as e:
            log.debug(f"  Nominatim reverse geocode failed: {e}")

        return None

    def _geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """Chuyen dia chi text -> toa do GPS."""
        if self.gmaps_key:
            try:
                import googlemaps
                gmaps  = googlemaps.Client(key=self.gmaps_key)
                result = gmaps.geocode(address + " Vietnam")
                if result:
                    loc = result[0]["geometry"]["location"]
                    return loc["lat"], loc["lng"]
            except Exception:
                pass

        # Nominatim fallback
        try:
            import urllib.request, urllib.parse
            query = urllib.parse.quote(address + " Vietnam")
            url   = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
            req   = urllib.request.Request(url, headers={"User-Agent": "FloodPipeline/1.0"})
            resp  = urllib.request.urlopen(req, timeout=10)
            data  = json.loads(resp.read())
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
        except Exception:
            pass
        return None

    def _parse_google_components(self, components: list) -> dict:
        result = {"street": "", "ward": "", "district": "", "province": ""}
        type_map = {
            "route":                      "street",
            "sublocality_level_1":        "ward",
            "administrative_area_level_2":"district",
            "administrative_area_level_1":"province",
            "locality":                   "district",
        }
        for comp in components:
            for t in comp.get("types", []):
                if t in type_map:
                    key = type_map[t]
                    if not result[key]:
                        result[key] = comp.get("long_name", "")
        return result
