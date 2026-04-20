# -*- coding: utf-8 -*-
"""
Stage 3 — Analyze
==================
Phân tích metadata của ảnh: location, GPS, biển số xe, OCR.
Có thể chạy song song vì mỗi ảnh độc lập.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("pipeline.analyze")


class AnalyzeStage:
    """Phân tích ảnh: location detection, metadata extraction."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, images: List[Path]) -> Dict[str, Any]:
        """
        Chạy location detection trên batch ảnh.
        Parallel nếu số ảnh > threshold.
        """
        if not images:
            return {}

        use_parallel = self.cfg.get("parallel_analyze", True) and len(images) > 4
        max_workers  = self.cfg.get("parallel_workers", 4)

        if use_parallel:
            return self._run_parallel(images, max_workers)
        return self._run_sequential(images)

    def _run_sequential(self, images: List[Path]) -> Dict[str, Any]:
        from utils.location_detector import LocationDetector
        results_list = LocationDetector(
            google_maps_key=self.cfg.get("google_maps_key", ""),
            use_ocr=self.cfg.get("use_ocr_location", True),
            use_plate=self.cfg.get("use_plate_location", True),
            use_exif=True,
        ).detect_batch([Path(p) for p in images])

        located = sum(1 for r in results_list if r.method != "none")
        log.info(f"  [Analyze] {located}/{len(images)} ảnh định vị được")
        return {str(r.image_path): vars(r) for r in results_list}

    def _run_parallel(self, images: List[Path], max_workers: int) -> Dict[str, Any]:
        """
        Chạy location detection song song — an toàn vì mỗi ảnh độc lập.
        Dùng 1 LocationDetector per thread để tránh race condition.
        """
        log.info(f"  [Analyze] Parallel mode: {len(images)} ảnh, {max_workers} workers")

        location_map: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._detect_one, img): img for img in images}
            for future in as_completed(futures):
                img = futures[future]
                try:
                    key, result = future.result()
                    location_map[key] = result
                except Exception as exc:
                    log.warning(f"  [Analyze] {img.name}: {exc}")

        located = sum(1 for v in location_map.values() if v.get("method") != "none")
        log.info(f"  [Analyze] {located}/{len(images)} ảnh định vị được (parallel)")
        return location_map

    def _detect_one(self, image: Path):
        """Detect location cho 1 ảnh — thread-safe."""
        from utils.location_detector import LocationDetector
        detector = LocationDetector(
            google_maps_key=self.cfg.get("google_maps_key", ""),
            use_ocr=self.cfg.get("use_ocr_location", True),
            use_plate=self.cfg.get("use_plate_location", True),
            use_exif=True,
        )
        results = detector.detect_batch([image])
        if results:
            r = results[0]
            return str(r.image_path), vars(r)
        return str(image), {"method": "none"}
