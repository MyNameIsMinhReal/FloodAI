# -*- coding: utf-8 -*-
"""
pipeline/stages/raincoat_stage.py
------------------------------------
Stage phát hiện áo mưa — tích hợp vào pipeline sau depth_stage.

Flow:
    crawl → filter → analyze → depth
    → [raincoat_stage] ← NEW
    → postprocess → store → learn

Chức năng stage này:
  1. Nhận depth_results (list DepthResult objects)
  2. Load ảnh gốc
  3. Chạy PoseAnalyzer (nếu chưa có keypoints)
  4. Chạy RaincoatDetector cho từng person
  5. Chạy PersonTracker (temporal smoothing + majority voting)
  6. Gắn raincoat info vào mỗi depth_result
  7. Trả về enriched results

Config (config.yaml):
    raincoat:
      enabled: true
      use_clip: false          # tắt nếu không có GPU/transformers
      bbox_pad: 0.15
      final_thresh: 0.40
      tracker_max_age: 10
      tracker_history: 5
      debug_overlay: false     # vẽ bbox overlay debug
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

log = logging.getLogger("pipeline.raincoat")


class RaincoatStage:
    """
    Stage phát hiện áo mưa cho toàn bộ pipeline.

    Args:
        cfg: dict config (section 'raincoat' từ config.yaml)
    """

    def __init__(self, cfg: dict):
        rc_cfg        = cfg.get("raincoat", {})
        self.enabled  = rc_cfg.get("enabled", True)
        self.use_clip = rc_cfg.get("use_clip", False)
        self.bbox_pad = rc_cfg.get("bbox_pad", 0.15)
        self.final_thresh = rc_cfg.get("final_thresh", 0.40)
        self.max_age      = rc_cfg.get("tracker_max_age", 10)
        self.history_len  = rc_cfg.get("tracker_history", 5)
        self.debug_overlay = rc_cfg.get("debug_overlay", False)
        self.pose_conf    = cfg.get("pose_conf", 0.35)
        self.pose_model   = cfg.get("models", {}).get("pose", "yolov8n-pose.pt")

        self._detector = None
        self._tracker  = None
        self._pose_analyzer = None
        # [BUG FIX v2] Thread-safe initialization lock for lazy-loaded components
        from threading import Lock
        self._init_lock = Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        depth_results: List[Any],
        overlay_dir:   Optional[Path] = None,
    ) -> List[Any]:
        """
        Chạy raincoat detection cho toàn bộ kết quả depth.

        Args:
            depth_results: list object từ DepthStage (có attr original_path, detections...)
            overlay_dir:   (Optional) thư mục lưu debug overlays

        Returns:
            depth_results đã được gắn thêm:
              - result.raincoat_detections: list RaincoatResult
              - result.has_raincoat: bool (có ít nhất 1 người mặc áo mưa)
              - result.raincoat_count: int
              - result.raincoat_confidence: float (max confidence)
        """
        if not self.enabled:
            log.info("  [Raincoat] Stage disabled — skipping")
            return depth_results

        if not depth_results:
            return depth_results

        self._init_components()

        log.info(f"  [Raincoat] Processing {len(depth_results)} images")
        enriched = 0

        for result in depth_results:
            try:
                self._process_one(result, overlay_dir)
                enriched += 1
            except Exception as e:
                log.warning(f"  [Raincoat] Error on {getattr(result, 'original_path', '?')}: {e}")
                self._attach_empty(result)

        log.info(f"  [Raincoat] Done: {enriched}/{len(depth_results)} processed")
        return depth_results

    # ── Per-image processing ──────────────────────────────────────────────────

    def _process_one(self, result: Any, overlay_dir: Optional[Path]) -> None:
        """Xử lý 1 ảnh: detect person → raincoat → track → attach."""
        # `_init_components()` khởi tạo lazy, nhưng type checker không thể
        # suy luận trạng thái của các thuộc tính sau lời gọi đó.
        detector = self._detector
        tracker = self._tracker
        if detector is None or tracker is None:
            raise RuntimeError("Raincoat components chưa được khởi tạo")

        img_path = self._get_path(result)
        if img_path is None or not Path(img_path).exists():
            self._attach_empty(result)
            return

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            self._attach_empty(result)
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Lấy persons từ depth result hoặc chạy lại pose
        persons = self._extract_persons(result, img_rgb)

        if not persons:
            self._attach_empty(result)
            return

        # Detect raincoat cho từng person
        from depth_analysis.raincoat_detector import RaincoatResult
        rc_results = []
        for p in persons:
            rc = detector.detect(
                img_rgb=img_rgb,
                bbox=p["bbox"],
                keypoints=p.get("keypoints"),
            )
            rc_results.append(rc)

        # Update tracker
        track_outputs = tracker.update(
            bboxes=[p["bbox"] for p in persons],
            scores=[p.get("score", 1.0) for p in persons],
            keypoints=[p.get("keypoints") for p in persons],
            raincoats=[rc.is_raincoat for rc in rc_results],
            poses=[p.get("pose_type") for p in persons],
            depths=[p.get("depth_cm") for p in persons],
        )

        # Merge tracker votes vào rc_results
        for i, tr in enumerate(track_outputs):
            if i < len(rc_results) and tr.raincoat_vote is not None:
                # Ghi đè vote từ temporal history
                rc_results[i].is_raincoat = tr.raincoat_vote
                rc_results[i].confidence  = max(rc_results[i].confidence, tr.raincoat_conf)
                rc_results[i].details.append(f"track_id={tr.track_id} vote={tr.raincoat_vote}")

        # Tổng hợp và attach vào result
        has_raincoat = any(r.is_raincoat for r in rc_results)
        count        = sum(1 for r in rc_results if r.is_raincoat)
        max_conf     = max((r.confidence for r in rc_results), default=0.0)

        self._attach(result, rc_results, has_raincoat, count, max_conf)

        # Debug overlay
        if self.debug_overlay and overlay_dir is not None:
            bboxes = [p["bbox"] for p in persons]
            debug_img = detector.draw_overlay(img_bgr, bboxes, rc_results)
            stem     = Path(img_path).stem
            out_path = Path(overlay_dir) / f"{stem}_raincoat.jpg"
            cv2.imwrite(str(out_path), debug_img)
            log.debug(f"    Debug overlay → {out_path}")

        log.debug(
            f"  [Raincoat] {Path(img_path).name}: "
            f"{count}/{len(rc_results)} raincoat, max_conf={max_conf:.2f}"
        )

    # ── Person extraction ─────────────────────────────────────────────────────

    def _extract_persons(
        self, result: Any, img_rgb: np.ndarray
    ) -> List[Dict]:
        """
        Lấy danh sách persons từ depth result hoặc chạy PoseAnalyzer.

        Returns:
            list of dict: {bbox, keypoints, score, pose_type, depth_cm}
        """
        persons = []

        # Thử lấy từ depth result (đã có poses)
        raw_poses = (
            getattr(result, "poses", None) or
            getattr(result, "person_poses", None)
        )
        if raw_poses:
            for p in raw_poses:
                bbox = getattr(p, "bbox", None)
                if bbox is None:
                    continue
                persons.append({
                    "bbox":      bbox,
                    "keypoints": getattr(p, "keypoints", None),
                    "score":     getattr(p, "confidence", 0.8),
                    "pose_type": getattr(p, "pose_type", None),
                    "depth_cm":  None,
                })
            if persons:
                return persons

        # Thử lấy từ detections (YOLO boxes)
        detections = getattr(result, "detections", None) or []
        for det in detections:
            bbox = getattr(det, "bbox", None) or getattr(det, "box", None)
            cls  = getattr(det, "class_name", "") or getattr(det, "label", "")
            if bbox and "person" in str(cls).lower():
                persons.append({
                    "bbox":      list(map(int, bbox)),
                    "keypoints": None,
                    "score":     getattr(det, "confidence", 0.8),
                    "pose_type": None,
                    "depth_cm":  None,
                })
        if persons:
            return persons

        # Fallback: chạy PoseAnalyzer nếu chưa có
        if self._pose_analyzer is not None:
            try:
                poses = self._pose_analyzer.analyze_image(img_rgb)
                for p in poses:
                    persons.append({
                        "bbox":      p.bbox,
                        "keypoints": p.keypoints,
                        "score":     p.confidence,
                        "pose_type": p.pose_type,
                        "depth_cm":  None,
                    })
            except Exception as e:
                log.debug(f"    PoseAnalyzer fallback failed: {e}")

        return persons

    # ── Attach results ────────────────────────────────────────────────────────

    @staticmethod
    def _attach(
        result:      Any,
        rc_results:  list,
        has_raincoat: bool,
        count:       int,
        max_conf:    float,
    ) -> None:
        """Gắn raincoat info vào result object (hoạt động với cả dict và object)."""
        info = {
            "has_raincoat":       has_raincoat,
            "raincoat_count":     count,
            "raincoat_confidence": round(max_conf, 3),
            "raincoat_detections": [
                {
                    "is_raincoat":   r.is_raincoat,
                    "confidence":    r.confidence,
                    "rule_score":    r.rule_score,
                    "clip_score":    r.clip_score,
                    "shape_score":   r.shape_score,
                    "dominant_color": r.dominant_color,
                    "texture_var":   r.texture_var,
                    "bright_ratio":  r.bright_ratio,
                    "details":       r.details,
                }
                for r in rc_results
            ],
        }
        if isinstance(result, dict):
            result.update(info)
        else:
            for k, v in info.items():
                try:
                    setattr(result, k, v)
                except AttributeError:
                    pass

    @staticmethod
    def _attach_empty(result: Any) -> None:
        """Gắn giá trị mặc định khi không có gì để detect."""
        RaincoatStage._attach(result, [], False, 0, 0.0)

    @staticmethod
    def _get_path(result: Any) -> Optional[str]:
        for attr in ["original_path", "image_path", "path", "img_path"]:
            v = (result.get(attr) if isinstance(result, dict) else getattr(result, attr, None))
            if v:
                return str(v)
        return None

    # ── Lazy init ─────────────────────────────────────────────────────────────

    def _init_components(self) -> None:
        # [BUG FIX v2] Use lock to ensure thread-safe lazy initialization
        with self._init_lock:
            if self._detector is None:
                from depth_analysis.raincoat_detector import RaincoatDetector
                self._detector = RaincoatDetector(
                    use_clip=self.use_clip,
                    bbox_pad=self.bbox_pad,
                    final_thresh=self.final_thresh,
                )
                log.info(f"  [Raincoat] Detector ready (CLIP={'on' if self.use_clip else 'off'})")

            if self._tracker is None:
                from depth_analysis.person_tracker import PersonTracker
                self._tracker = PersonTracker(
                    max_age=self.max_age,
                    history_len=self.history_len,
                )
                log.info("  [Raincoat] Tracker ready")

            if self._pose_analyzer is None:
                try:
                    from depth_analysis.pose_analyzer import PoseAnalyzer
                    self._pose_analyzer = PoseAnalyzer(
                        pose_model=self.pose_model,
                        conf_thresh=self.pose_conf,
                    )
                    log.info("  [Raincoat] PoseAnalyzer fallback ready")
                except Exception as e:
                    log.debug(f"  [Raincoat] PoseAnalyzer not loaded: {e}")
