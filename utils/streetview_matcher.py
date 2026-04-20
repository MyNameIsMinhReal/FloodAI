# -*- coding: utf-8 -*-
"""
utils/streetview_matcher.py
-----------------------------
So sanh anh lu voi Google Street View de nhan dien vi tri.

Cach hoat dong:
  1. Extract visual features tu anh lu (building facade, sign, landmark)
  2. Su dung Google Vision API hoac DINOv2 de trich xuat embedding
  3. Tim vi tri tren Street View co embedding gan nhat
  4. Xac nhan bang landmark detection (bien hieu, kien truc dac trung)

Phuong phap thay the (khong can API):
  - Nhan dien kieu kien truc (nha pho VN, nha tap the, nha ong)
  - Nhan dien bien hieu chu Viet
  - So mau sac dac trung (mau son tuong, mai nha)

API can co:
  - Google Cloud Vision API (landmark detection)
  - Google Maps Street View Static API
  - Google Maps Geocoding API

Cai dat: pip install google-cloud-vision googlemaps requests Pillow
"""

import logging
import json
import base64
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import urllib.request
import urllib.parse

from .constants import DEFAULT_DINO_MODEL

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class StreetViewMatch:
    method:          str      # "vision_landmark", "visual_embed", "street_sign"
    latitude:        Optional[float] = None
    longitude:       Optional[float] = None
    landmark_name:   str = ""
    landmark_type:   str = ""   # "building", "street", "area", "sign"
    confidence:      float = 0.0
    street_view_url: str = ""
    map_url:         str = ""
    matched_features:List[str] = field(default_factory=list)
    notes:           str = ""


class StreetViewMatcher:
    """
    So sanh anh voi Street View de xac dinh vi tri.
    Ho tro nhieu fallback method.
    """

    def __init__(
        self,
        google_api_key:     str  = "",
        use_vision_api:     bool = True,
        use_visual_match:   bool = True,
        use_sign_detection: bool = True,
        cache_dir:          Path = None,
    ):
        self.api_key          = google_api_key
        self.use_vision_api   = use_vision_api
        self.use_visual_match = use_visual_match
        self.use_sign_detect  = use_sign_detection
        self.cache_dir        = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dino_model = None
        self._dino_proc  = None

    # ==================================================================
    def match(self, image_path: Path) -> Optional[StreetViewMatch]:
        """
        Thu tat ca cac phuong phap, tra ve ket qua tot nhat.
        """
        image_path = Path(image_path)
        results: List[StreetViewMatch] = []

        # --- Method 1: Google Cloud Vision Landmark Detection ---
        if self.use_vision_api and self.api_key:
            r = self._vision_landmark_detect(image_path)
            if r:
                results.append(r)
                log.info(
                    f"  StreetView Vision: {r.landmark_name} "
                    f"({r.latitude:.5f},{r.longitude:.5f}) conf={r.confidence:.2f}"
                )

        # --- Method 2: Street Sign + Building Visual Match ---
        if self.use_sign_detect:
            r = self._detect_visual_context(image_path)
            if r:
                results.append(r)

        # --- Method 3: DINOv2 Scene Embedding Match ---
        if self.use_visual_match:
            r = self._dino_scene_match(image_path)
            if r:
                results.append(r)

        if not results:
            return None

        # Lay ket qua co confidence cao nhat
        best = max(results, key=lambda x: x.confidence)
        if best.latitude and best.longitude:
            best.map_url = (
                f"https://maps.google.com/đq={best.latitude},{best.longitude}"
            )
            best.street_view_url = self._build_streetview_url(
                best.latitude, best.longitude
            )
        return best

    # ==================================================================
    # METHOD 1: Google Cloud Vision API - Landmark Detection
    # ==================================================================
    def _vision_landmark_detect(
        self, image_path: Path
    ) -> Optional[StreetViewMatch]:
        """
        Dung Google Cloud Vision API de phat hien landmark trong anh.
        API tra ve: ten landmark, toa do GPS, do tin cay.

        Goi API: 1000 request/thang mien phi.
        """
        try:
            # Doc anh, encode base64
            with open(str(image_path), "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "requests": [{
                    "image": {"content": img_data},
                    "features": [
                        {"type": "LANDMARK_DETECTION", "maxResults": 5},
                        {"type": "TEXT_DETECTION",     "maxResults": 10},
                        {"type": "LOGO_DETECTION",     "maxResults": 5},
                    ]
                }]
            }

            url  = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
            data = json.dumps(payload).encode("utf-8")
            req  = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())

            responses = result.get("responses", [{}])[0]

            # --- Xu ly Landmark ---
            landmarks = responses.get("landmarkAnnotations", [])
            if landmarks:
                top = landmarks[0]
                locs = top.get("locations", [])
                lat = lon = None
                if locs:
                    ll  = locs[0].get("latLng", {})
                    lat = ll.get("latitude")
                    lon = ll.get("longitude")

                return StreetViewMatch(
                    method         = "vision_landmark",
                    latitude       = lat,
                    longitude      = lon,
                    landmark_name  = top.get("description", ""),
                    landmark_type  = "landmark",
                    confidence     = top.get("score", 0) * 0.9,
                    matched_features = [l.get("description","") for l in landmarks[:3]],
                    notes = f"Google Vision landmark: {top.get('description','')}",
                )

            # --- Xu ly Text (bien hieu duong) ---
            texts = responses.get("textAnnotations", [])
            if texts and len(texts) > 1:
                all_text = " ".join(t.get("description","") for t in texts[1:6])
                geocode  = self._geocode_from_text(all_text)
                if geocode:
                    return StreetViewMatch(
                        method        = "vision_text",
                        latitude      = geocode[0],
                        longitude     = geocode[1],
                        landmark_name = all_text[:80],
                        landmark_type = "sign",
                        confidence    = 0.55,
                        notes         = f"OCR from Vision API: {all_text[:80]}",
                    )

        except Exception as e:
            log.debug(f"  Vision API error: {e}")
        return None

    # ==================================================================
    # METHOD 2: Visual Context Detection (khong can API key)
    # ==================================================================
    def _detect_visual_context(
        self, image_path: Path
    ) -> Optional[StreetViewMatch]:
        """
        Phat hien dac trung visual de xac dinh khu vuc:
          - Kieu kien truc (nha pho, nha tap the, nha ong VN)
          - Bien hieu chu Viet Nam
          - Cot dien, duong day dien dac trung VN
          - Bien duong, bien pho
          - Xe co dac trung

        Tra ve region-level match (khong chinh xac den toa do).
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return None

        h, w   = img_bgr.shape[:2]
        hsv    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        features = []
        confidence = 0.0

        # --- Phat hien kieu kien truc Viet Nam ---
        arch_score, arch_type = self._detect_vn_architecture(img_bgr, gray, h, w)
        if arch_score > 0.4:
            features.append(f"VN architecture: {arch_type}")
            confidence = max(confidence, arch_score * 0.5)

        # --- Phat hien bien hieu co chu Viet ---
        sign_score, sign_texts = self._detect_vn_signs(img_bgr, gray)
        if sign_score > 0.3:
            features.extend([f"Sign: {t}" for t in sign_texts[:3]])
            confidence = max(confidence, sign_score * 0.6)

        # --- Phat hien cot dien, day dien ---
        pole_score = self._detect_utility_poles(gray, h, w)
        if pole_score > 0.5:
            features.append("Utility poles (VN style)")
            confidence = max(confidence, 0.25)

        # --- Phat hien xe dac trung ---
        vehicle_score = self._detect_vn_vehicles(img_bgr)
        if vehicle_score > 0.4:
            features.append("Vietnamese motorcycles")
            confidence = max(confidence, 0.30)

        if not features:
            return None

        # Neu co sign texts -> thu geocode
        lat = lon = None
        if sign_texts:
            for text in sign_texts:
                coords = self._geocode_from_text(text + " Vietnam")
                if coords:
                    lat, lon = coords
                    confidence = min(0.75, confidence + 0.2)
                    break

        return StreetViewMatch(
            method           = "visual_context",
            latitude         = lat,
            longitude        = lon,
            landmark_name    = sign_texts[0] if sign_texts else "Vietnam (unspecified)",
            landmark_type    = "area",
            confidence       = round(confidence, 3),
            matched_features = features,
            notes            = f"Visual: {'; '.join(features[:3])}",
        )

    def _detect_vn_architecture(
        self, img_bgr: np.ndarray, gray: np.ndarray, h: int, w: int
    ) -> Tuple[float, str]:
        """
        Phat hien kien truc Viet Nam:
          - Nha ong: nha cao nhung hep (ty le chieu cao/rong lon)
          - Mau sac dac trung: vang, xanh, hong pastel
          - Mai ngoi (do)
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        score = 0.0
        arch_type = "unknown"

        # Kiem tra mau tuong dac trung VN (vang, trang nghi vang, hong nhat)
        yellow_wall = cv2.inRange(hsv, np.array([15,40,150]), np.array([35,180,255]))
        pink_wall   = cv2.inRange(hsv, np.array([160,20,180]),np.array([180,80,255]))
        yellow_pct  = yellow_wall.sum()/255 / (h*w)
        pink_pct    = pink_wall.sum()/255 / (h*w)

        if yellow_pct > 0.08:
            score     = max(score, 0.55)
            arch_type = "VN yellow house"
        if pink_pct > 0.05:
            score     = max(score, 0.50)
            arch_type = "VN pink house"

        # Kiem tra mai ngoi do
        red_roof = cv2.inRange(hsv, np.array([0,80,100]), np.array([15,255,200]))
        red_pct  = red_roof[0:h//3, :].sum()/255 / (h//3*w)   # chi phan tren
        if red_pct > 0.05:
            score     = max(score, 0.45)
            arch_type = arch_type or "VN tiled roof"

        # Kiem tra duong day dien tren cao (dac trung VN)
        edges      = cv2.Canny(gray[0:h//3, :], 50, 150)
        lines      = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=w//5, maxLineGap=10)
        if lines is not None and len(lines) > 5:
            horiz = sum(
                1 for l in lines
                if abs(l[0][3]-l[0][1]) < 8   # duong ngang
            )
            if horiz > 3:
                score = max(score, 0.40)

        return score, arch_type

    def _detect_vn_signs(
        self, img_bgr: np.ndarray, gray: np.ndarray
    ) -> Tuple[float, List[str]]:
        """
        Phat hien bien hieu co chu Viet.
        Tim cac vung co text -> chay OCR nhanh.
        """
        h, w    = gray.shape
        texts   = []
        score   = 0.0

        # MSER text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        text_mask = np.zeros((h, w), dtype=np.uint8)
        for region in regions:
            x, y, rw, rh = cv2.boundingRect(region.reshape(-1,1,2))
            if 8 < rh < h*0.2 and 0.3 < rw/max(rh,1) < 15:
                text_mask[y:y+rh, x:x+rw] = 255

        # Tim cac vung text lon
        kernel   = np.ones((5,20), np.uint8)
        dilated  = cv2.dilate(text_mask, kernel)
        contours,_ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sign_regions = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > w*0.1 and ch > 10:
                sign_regions.append((x, y, cw, ch))

        if sign_regions:
            score = 0.4
            # Thu doc text tu cac vung bien hieu
            for x, y, cw, ch in sign_regions[:5]:
                roi = img_bgr[max(0,y-3):min(h,y+ch+3), max(0,x-3):min(w,x+cw+3)]
                if roi.size == 0:
                    continue
                # Enhance contrast cho OCR
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                # Dung tesseract neu co
                try:
                    import pytesseract
                    pil_roi = Image.fromarray(roi_bin)
                    text = pytesseract.image_to_string(pil_roi, lang="vie+eng",
                                                        config="--psm 7").strip()
                    if text and len(text) > 3:
                        texts.append(text)
                        score = min(0.75, score + 0.1)
                except Exception:
                    pass

        return score, texts

    def _detect_utility_poles(self, gray: np.ndarray, h: int, w: int) -> float:
        """Phat hien cot dien dung doc - dac trung VN."""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=h//5,
            minLineLength=h//3, maxLineGap=20
        )
        if lines is None:
            return 0.0
        vert = sum(
            1 for l in lines
            if abs(l[0][2]-l[0][0]) < 15   # duong doc
            and abs(l[0][3]-l[0][1]) > h//4
        )
        return min(1.0, vert / 3)

    def _detect_vn_vehicles(self, img_bgr: np.ndarray) -> float:
        """
        Phat hien xe may (phuong tien pho bien o VN).
        Dua tren ti le + hinh dang bbox tu YOLO hoac color pattern.
        """
        # Placeholder - trong thuc te se ket hop voi YOLO result
        # Tu anh co the phat hien qua mau xe may + hinh dang
        return 0.0

    # ==================================================================
    # METHOD 3: DINOv2 Scene Embedding Match
    # ==================================================================
    def _dino_scene_match(
        self, image_path: Path
    ) -> Optional[StreetViewMatch]:
        """
        Su dung DINOv2 de trich xuat scene embedding va so sanh
        voi database Street View embeddings da build san.

        QUAN TRONG: Method nay can database embeddings cua Street View
        trong khu vuc quan tam. De build database:
          1. Lay list toa do trong khu vuc (grid sampling)
          2. Tai anh Street View tu moi toa do
          3. Trich xuat DINOv2 embedding
          4. Luu vao file .npz

        Neu khong co database -> tra ve None.
        """
        # Kiem tra co database khong
        db_path = Path("streetview_db.npz")
        if not db_path.exists():
            log.debug("  No Street View database found. Run build_streetview_db.py first.")
            return None

        try:
            self._load_dino()

            img     = Image.open(str(image_path)).convert("RGB")
            query_emb = self._extract_embedding(img)

            # Load database
            db      = np.load(str(db_path), allow_pickle=True)
            db_embs = db["embeddings"]    # shape [N, D]
            db_lats = db["latitudes"]
            db_lons = db["longitudes"]
            db_desc = db.get("descriptions", np.array([""] * len(db_lats)))

            # Tinh cosine similarity
            q_norm   = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            db_norms = db_embs / (np.linalg.norm(db_embs, axis=1, keepdims=True) + 1e-8)
            sims     = db_norms @ q_norm

            # Top-3 matches
            top_idx  = np.argsort(sims)[::-1][:3]
            best_sim = float(sims[top_idx[0]])

            if best_sim < 0.65:   # nguong tuong dong toi thieu
                return None

            lat = float(db_lats[top_idx[0]])
            lon = float(db_lons[top_idx[0]])

            # Confidence calibration
            conf = (best_sim - 0.65) / 0.35 * 0.80   # map 0.65-1.0 -> 0-0.80

            return StreetViewMatch(
                method           = "visual_embed",
                latitude         = lat,
                longitude        = lon,
                landmark_name    = str(db_desc[top_idx[0]]),
                landmark_type    = "visual_match",
                confidence       = round(conf, 3),
                matched_features = [f"sim={best_sim:.3f}"],
                notes            = (
                    f"DINOv2 scene match: sim={best_sim:.3f} "
                    f"at ({lat:.5f},{lon:.5f})"
                ),
            )

        except Exception as e:
            log.debug(f"  DINOv2 match error: {e}")
        return None

    def _load_dino(self):
        if self._dino_model:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("  Loading DINOv2 for scene matching ...")
        self._dino_proc  = AutoImageProcessor.from_pretrained(DEFAULT_DINO_MODEL)
        self._dino_model = AutoModel.from_pretrained(DEFAULT_DINO_MODEL).to(dev)
        self._dino_model.eval()
        self._dino_dev   = dev

    def _extract_embedding(self, pil_img: Image.Image) -> np.ndarray:
        import torch
        inputs = self._dino_proc(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._dino_dev) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._dino_model(**inputs)
        return out.last_hidden_state[:, 0, :].cpu().numpy()[0]

    # ==================================================================
    # STREET VIEW DATABASE BUILDER
    # ==================================================================
    def build_database(
        self,
        locations: List[Tuple[float, float]],
        output_path: Path = Path("streetview_db.npz"),
        radius_m: int = 50,
    ):
        """
        Xay dung database Street View embeddings tu list toa do.

        Args:
            locations: [(lat, lon), ...] - cac toa do can xay dung DB
            output_path: duong dan luu file .npz
            radius_m: ban kinh lay anh xung quanh moi toa do (m)

        Can: Google Maps Street View Static API key.

        Chay rieng: python -c "from utils.streetview_matcher import *; build_db()"
        """
        if not self.api_key:
            log.error("  Google API key required to build Street View database")
            return

        self._load_dino()

        embeddings   = []
        valid_lats   = []
        valid_lons   = []
        descriptions = []

        log.info(f"  Building Street View DB: {len(locations)} locations ...")

        for i, (lat, lon) in enumerate(locations):
            # Lay anh Street View
            img = self._fetch_streetview_image(lat, lon)
            if img is None:
                continue

            # Trich xuat embedding
            emb = self._extract_embedding(img)
            embeddings.append(emb)
            valid_lats.append(lat)
            valid_lons.append(lon)

            # Reverse geocode lay mo ta
            addr = self._reverse_geocode_simple(lat, lon)
            descriptions.append(addr)

            if (i+1) % 10 == 0:
                log.info(f"  DB progress: {i+1}/{len(locations)}")
            time.sleep(0.1)   # Rate limit

        if embeddings:
            np.savez(
                str(output_path),
                embeddings   = np.array(embeddings),
                latitudes    = np.array(valid_lats),
                longitudes   = np.array(valid_lons),
                descriptions = np.array(descriptions),
            )
            log.info(f"  DB saved: {len(embeddings)} entries -> {output_path}")
        else:
            log.warning("  No embeddings collected")

    def _fetch_streetview_image(
        self, lat: float, lon: float, size: str = "640x480"
    ) -> Optional[Image.Image]:
        """Lay anh Google Street View tai toa do cho truoc."""
        if not self.api_key:
            return None
        try:
            url = (
                f"https://maps.googleapis.com/maps/api/streetview"
                f"đsize={size}&location={lat},{lon}"
                f"&key={self.api_key}&source=outdoor"
            )
            req  = urllib.request.Request(url, headers={"User-Agent": "FloodPipeline/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            img = Image.open(__import__("io").BytesIO(data)).convert("RGB")
            return img
        except Exception as e:
            log.debug(f"  Street View fetch failed ({lat},{lon}): {e}")
        return None

    def _reverse_geocode_simple(self, lat: float, lon: float) -> str:
        """Nominatim reverse geocode."""
        try:
            url  = (
                f"https://nominatim.openstreetmap.org/reverse"
                f"đlat={lat}&lon={lon}&format=json"
            )
            req  = urllib.request.Request(url, headers={"User-Agent": "FloodPipeline/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            return data.get("display_name", "")[:100]
        except Exception:
            return f"{lat:.5f},{lon:.5f}"

    # ==================================================================
    # GEOCODING HELPERS
    # ==================================================================
    def _geocode_from_text(self, text: str) -> Optional[Tuple[float, float]]:
        """Tim toa do tu text (ten duong, bien hieu)."""
        if not text or len(text) < 4:
            return None
        try:
            query = urllib.parse.quote(text + " Vietnam")
            url   = (
                f"https://nominatim.openstreetmap.org/search"
                f"đq={query}&format=json&limit=1&countrycodes=vn"
            )
            req  = urllib.request.Request(url, headers={"User-Agent": "FloodPipeline/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
        except Exception:
            pass
        return None

    def _build_streetview_url(self, lat: float, lon: float) -> str:
        return (
            f"https://www.google.com/maps/@{lat},{lon},3a,90y,0h,90t"
            f"/data=!3m6!1e1!3m4!1s0x0:0x0!2e0!7i13312!8i6656"
        )

    # ==================================================================
    # 5.2 FLOOD vs BASELINE MEASUREMENT (MỚI)
    # ==================================================================

    def measure_flood_vs_baseline(
        self,
        flood_image_path: Path,
        lat: float,
        lon: float,
        baseline_sv_image: Optional[Image.Image] = None,
    ) -> Optional[dict]:
        """
        Đo mực nước bằng cách so sánh ảnh lũ với Street View baseline.

        Workflow:
          1. Lấy ảnh Street View tại (lat, lon) [baseline = trước lũ]
          2. Align ảnh lũ với Street View bằng feature matching (ORB/SIFT)
          3. Phát hiện "water level" trong ảnh lũ
          4. Đo chiều cao nước so với các điểm landmark trong baseline

        Args:
            flood_image_path: đường dẫn ảnh lũ
            lat, lon:         tọa độ GPS ước tính
            baseline_sv_image: (optional) ảnh Street View đã fetch sẵn

        Returns:
            dict với flood_height_cm, confidence, alignment_score, ...
        """
        flood_bgr = cv2.imread(str(flood_image_path))
        if flood_bgr is None:
            log.warning(f"  Cannot read flood image: {flood_image_path}")
            return None

        # Lấy Street View baseline
        if baseline_sv_image is None:
            if not self.api_key:
                log.debug("  No API key for Street View baseline")
                return self._estimate_without_baseline(flood_bgr)
            sv_pil = self._fetch_streetview_image(lat, lon)
            if sv_pil is None:
                return self._estimate_without_baseline(flood_bgr)
        else:
            sv_pil = baseline_sv_image

        sv_bgr = cv2.cvtColor(np.array(sv_pil), cv2.COLOR_RGB2BGR)

        # Align ảnh
        aligned_flood, alignment_score, H = self._align_images(flood_bgr, sv_bgr)

        if alignment_score < 0.15:
            log.debug(f"  Poor alignment ({alignment_score:.2f}), using fallback")
            return self._estimate_without_baseline(flood_bgr)

        # Phát hiện water line trong ảnh lũ
        water_line_y, water_conf = self._detect_water_line_precise(aligned_flood)
        if water_line_y is None:
            return None

        h, w = sv_bgr.shape[:2]

        # Tìm reference landmarks trong baseline (edge của các đồ vật có chiều cao biết trước)
        ref_heights = self._extract_reference_heights(sv_bgr, aligned_flood, H)

        # Tính chiều cao nước
        flood_height_cm, method = self._compute_flood_height(
            water_line_y, h, ref_heights
        )

        # Tạo visualization
        viz_path = self._create_comparison_viz(
            sv_bgr, aligned_flood, water_line_y, flood_image_path
        )

        return {
            "flood_height_cm":  round(flood_height_cm, 1),
            "water_line_y_pct": round(1.0 - water_line_y / h, 3),
            "alignment_score":  round(alignment_score, 3),
            "measurement_method": method,
            "confidence":       round(min(0.95, alignment_score * water_conf), 3),
            "baseline_lat":     lat,
            "baseline_lon":     lon,
            "comparison_viz":   viz_path,
            "notes": (
                f"Aligned flood image vs Street View baseline. "
                f"Alignment: {alignment_score:.0%}. "
                f"Water at {1.0 - water_line_y/h:.0%} from bottom."
            ),
        }

    def _align_images(
        self,
        flood_bgr: np.ndarray,
        baseline_bgr: np.ndarray,
    ) -> tuple:
        """
        Align ảnh lũ với baseline bằng ORB feature matching + homography.

        Returns:
            (aligned_flood, alignment_score, H_matrix)
        """
        h_base, w_base = baseline_bgr.shape[:2]

        # Resize flood về kích thước baseline
        flood_resized = cv2.resize(flood_bgr, (w_base, h_base))

        gray_flood = cv2.cvtColor(flood_resized, cv2.COLOR_BGR2GRAY)
        gray_base  = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)

        # ORB detector (nhanh hơn SIFT, không cần license)
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(gray_flood, None)
        kp2, des2 = orb.detectAndCompute(gray_base,  None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return flood_resized, 0.0, np.eye(3, dtype=np.float32)

        # BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Lấy good matches (top 60%)
        n_good = max(10, int(len(matches) * 0.6))
        good_matches = matches[:n_good]

        if len(good_matches) < 8:
            # Không đủ matches → return unaligned với score thấp
            avg_dist = np.mean([m.distance for m in matches]) if matches else 255
            score = max(0.0, 1.0 - avg_dist / 64)
            return flood_resized, score * 0.3, np.eye(3, dtype=np.float32)

        # Homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            avg_dist = np.mean([m.distance for m in good_matches])
            return flood_resized, max(0, 0.3 - avg_dist / 200), np.eye(3, dtype=np.float32)

        # Warp flood image vào góc nhìn baseline
        aligned = cv2.warpPerspective(flood_resized, H, (w_base, h_base))

        # Score: tỉ lệ inliers × chất lượng matches
        inlier_ratio = float(mask.sum()) / len(mask)
        avg_dist = np.mean([m.distance for m in good_matches])
        quality = max(0, 1.0 - avg_dist / 64)
        score = inlier_ratio * 0.7 + quality * 0.3

        return aligned, float(score), H

    def _detect_water_line_precise(
        self, img_bgr: np.ndarray
    ) -> tuple:
        """
        Phát hiện đường mực nước chính xác trong ảnh đã align.

        Kỹ thuật:
          - Phát hiện vùng nước (color-based)
          - Tìm đường ranh giới trên cùng của vùng nước
          - Dùng Canny edge để refine
        """
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Water color profiles
        water_masks = []
        profiles = [
            ((90, 35, 35), (135, 255, 255)),  # clear blue
            ((5,  45, 25), (22, 230, 195)),   # muddy brown
            ((0,   0, 50), (180, 40, 180)),   # turbid gray
            ((20, 50, 80), (38, 230, 240)),   # yellowish flood
        ]
        combined = np.zeros((h, w), np.uint8)
        for lo, hi in profiles:
            combined = cv2.bitwise_or(
                combined,
                cv2.inRange(hsv, np.array(lo), np.array(hi)),
            )

        # Giới hạn bottom 80%
        combined[:h // 5, :] = 0

        # Cleanup
        k = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k)

        if combined.sum() == 0:
            return None, 0.0

        # Tìm water line: top của vùng nước lớn nhất
        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, 0.0

        largest = max(cnts, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        area_ratio = cv2.contourArea(largest) / (h * w)

        water_line_y = y
        confidence = min(0.95, area_ratio * 3 + 0.3)

        return water_line_y, float(confidence)

    def _extract_reference_heights(
        self,
        baseline_bgr: np.ndarray,
        aligned_flood_bgr: np.ndarray,
        H: np.ndarray,
    ) -> List[dict]:
        """
        Trích xuất reference landmarks từ baseline để đo độ cao nước.

        Tìm:
          - Cột điện (utility poles): chiều cao ≈ 8-12m
          - Bậc thềm, vỉa hè
          - Cửa sổ, cửa nhà (chiều cao tiêu chuẩn)
        """
        h, w = baseline_bgr.shape[:2]
        gray = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)
        refs = []

        # Detect vertical lines (cột điện, khung cửa)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=h // 6,
            minLineLength=h // 4, maxLineGap=15,
        )

        if lines is not None:
            for line in lines[:10]:
                x1, y1, x2, y2 = line[0]
                line_h = abs(y2 - y1)
                if line_h > h // 4:  # vertical line đủ dài
                    mid_x = (x1 + x2) // 2
                    bottom_y = max(y1, y2)
                    # Ước tính chiều cao pixel vs cm (giả định cột điện 10m)
                    pix_per_cm = line_h / 1000  # 10m = 1000cm
                    refs.append({
                        "type":        "pole",
                        "bottom_y":    bottom_y,
                        "top_y":       min(y1, y2),
                        "mid_x":       mid_x,
                        "height_cm":   1000,
                        "pix_per_cm":  pix_per_cm,
                        "confidence":  0.4,
                    })

        return refs

    def _compute_flood_height(
        self,
        water_line_y: int,
        image_height: int,
        ref_heights: List[dict],
    ) -> tuple:
        """Tính chiều cao nước từ water_line_y và reference heights."""
        if ref_heights:
            # Dùng reference gần nhất với water line
            valid_refs = [r for r in ref_heights if r["bottom_y"] > water_line_y]
            if valid_refs:
                ref = max(valid_refs, key=lambda r: r["confidence"])
                pix_water_height = ref["bottom_y"] - water_line_y
                flood_cm = pix_water_height / (ref["pix_per_cm"] + 1e-6)
                return float(np.clip(flood_cm, 0, 300)), "reference_object"

        # Fallback: heuristic từ vị trí water line trong ảnh
        # Giả định: bottom of image = ground level, ảnh cao ~200cm (2m scene height)
        scene_height_cm = 200
        water_height_cm = (1.0 - water_line_y / image_height) * scene_height_cm
        return float(np.clip(water_height_cm, 0, 200)), "position_heuristic"

    def _estimate_without_baseline(self, flood_bgr: np.ndarray) -> Optional[dict]:
        """Ước tính mực nước không cần baseline (fallback)."""
        h, w = flood_bgr.shape[:2]
        water_line_y, conf = self._detect_water_line_precise(flood_bgr)
        if water_line_y is None:
            return None
        flood_cm, method = self._compute_flood_height(water_line_y, h, [])
        return {
            "flood_height_cm":    round(flood_cm, 1),
            "water_line_y_pct":   round(1.0 - water_line_y / h, 3),
            "alignment_score":    0.0,
            "measurement_method": f"{method}_no_baseline",
            "confidence":         round(conf * 0.5, 3),
            "notes":              "No Street View baseline available",
        }

    def _create_comparison_viz(
        self,
        baseline_bgr: np.ndarray,
        flood_bgr: np.ndarray,
        water_line_y: int,
        flood_path: Path,
    ) -> str:
        """Tạo ảnh so sánh baseline vs flood side-by-side."""
        try:
            h, w = baseline_bgr.shape[:2]
            # Vẽ water line lên ảnh lũ
            flood_viz = flood_bgr.copy()
            cv2.line(flood_viz, (0, water_line_y), (w, water_line_y), (0, 255, 255), 3)
            cv2.putText(flood_viz, "Water Line", (10, water_line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Side by side
            comparison = np.hstack([baseline_bgr, flood_viz])
            cv2.putText(comparison, "Baseline (Street View)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(comparison, "Flood Image", (w + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            out_path = Path(str(flood_path).replace(
                flood_path.suffix, "_sv_comparison.jpg"
            ))
            cv2.imwrite(str(out_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 88])
            return str(out_path)
        except Exception as e:
            log.debug(f"  Comparison viz failed: {e}")
            return ""
