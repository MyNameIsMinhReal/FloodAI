# -*- coding: utf-8 -*-
"""
Stage 4 — Depth Estimation
===========================
Ước tính độ sâu ngập lụt từ ảnh bằng:
  - Depth Anything V2 (monocular depth)
  - YOLOv8 (object detection)
  - DINOv2 (feature extraction)
  - Pose estimation

Config model switching (config.yaml):
    models:
      depth: depth-anything/Depth-Anything-V2-Small-hf   # hoặc Base / Large / midas
      detector: yolov8n.pt                               # hoặc yolov8s / yolov8m
      pose: yolov8n-pose.pt
      dino: facebook/dinov2-small
"""
import logging
import shutil
from pathlib import Path
from typing import Any, List

log = logging.getLogger("pipeline.depth")


class DepthStage:
    """Ước tính độ sâu ngập lụt từ ảnh."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def compute(
        self,
        images: List[Path],
        overlay_dir: Path,
        depthmap_dir: Path,
    ) -> List[Any]:
        """
        Chạy toàn bộ depth estimation pipeline.
        Model được resolve từ config → cho phép switch dễ dàng.
        """
        if not images:
            return []

        # Resolve models từ config (hỗ trợ switching)
        models = self._resolve_models()
        log.info(f"  [Depth] Models: {models}")

        tmp_dir = overlay_dir.parent / "_tmp_depth"
        tmp_dir.mkdir(exist_ok=True)

        # [BUG FIX v2] Initialize results before try block to prevent NameError in finally
        results = []
        try:
            results = self._run_estimator(images, models, tmp_dir)
        finally:
            self._move_outputs(results, tmp_dir, overlay_dir, depthmap_dir)
            shutil.rmtree(tmp_dir, ignore_errors=True)

        log.info(f"  [Depth] {len(results)}/{len(images)} ảnh thành công")
        return results

    def _resolve_models(self) -> dict:
        """
        Resolve model từ config.yaml.
        Hỗ trợ shorthand: 'small' → đường dẫn đầy đủ.
        """
        from utils.constants import (
            DEFAULT_DEPTH_MODEL, DEFAULT_YOLO_MODEL,
            DEFAULT_POSE_MODEL, DEFAULT_DINO_MODEL,
        )

        # Shorthand mapping
        # Nhóm RELATIVE (cần reference object để suy ra tỉ lệ mét):
        depth_map = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base":  "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf",
            "midas": "Intel/dpt-hybrid-midas",
            # Nhóm METRIC — trả số mét TRỰC TIẾP, chính xác hơn hẳn cho đo mực nước
            # (vẫn tương thích pipeline vì output được normalize như cũ)
            "zoedepth":   "Intel/zoedepth-nyu-kitti",   # metric, indoor+outdoor (~1.5GB)
            "zoedepth-n": "Intel/zoedepth-nyu",          # metric, indoor
            "zoedepth-k": "Intel/zoedepth-kitti",        # metric, outdoor (khuyên dùng)
            "depthpro":   "apple/DepthPro-hf",           # metric, chi tiết cao (~2GB)
            "metric3d":   "Zigeng/Metric3D-v2-giant",    # metric mạnh nhất (rất nặng)
        }
        yolo_map = {
            "nano":   "yolov8n.pt",
            "small":  "yolov8s.pt",
            "medium": "yolov8m.pt",
        }

        # Đọc từ config, ưu tiên section `models` mới
        cfg_models = self.cfg.get("models", {})
        depth_key  = cfg_models.get("depth", self.cfg.get("depth_model", "small"))
        yolo_key   = cfg_models.get("detector", self.cfg.get("yolo_model", "nano"))
        pose_key   = cfg_models.get("pose", self.cfg.get("pose_model", DEFAULT_POSE_MODEL))
        dino_key   = cfg_models.get("dino", self.cfg.get("dino_model", DEFAULT_DINO_MODEL))

        return {
            "depth":    depth_map.get(depth_key, depth_key or DEFAULT_DEPTH_MODEL),
            "detector": yolo_map.get(yolo_key,  yolo_key  or DEFAULT_YOLO_MODEL),
            "pose":     pose_key or DEFAULT_POSE_MODEL,
            "dino":     dino_key or DEFAULT_DINO_MODEL,
        }

    def _run_estimator(self, images: List[Path], models: dict, tmp_dir: Path) -> List[Any]:
        from depth_analysis.reference_estimator import ReferenceEstimator

        estimator = ReferenceEstimator(
            yolo_model=models["detector"],
            depth_model=models["depth"],
            output_dir=tmp_dir,
            conf_thresh=self.cfg.get("yolo_conf", 0.35),
            use_dino=self.cfg.get("use_dino", True),
            dino_model=models["dino"],
            use_pose=self.cfg.get("use_pose", True),
            pose_model=models["pose"],
            use_segformer=self.cfg.get("use_segformer", True),
        )
        estimator._cfg = self.cfg   # cho SAM hook trong _measure_objects_local

        # [v4] depth_chunk_size: models.depth_chunk_size > cfg.depth_chunk_size > default 8
        chunk_size = (
            self.cfg.get("models", {}).get("depth_chunk_size")
            or self.cfg.get("depth_chunk_size", 8)
        )
        results = estimator.analyze_batch(images, chunk_size=chunk_size)
        estimator.unload_heavy_models()
        return results

    def _move_outputs(self, results: List[Any], tmp_dir: Path,
                      overlay_dir: Path, depthmap_dir: Path):
        for r in results:
            for attr, dest in [("overlay_path", overlay_dir), ("depth_map_path", depthmap_dir)]:
                src = Path(getattr(r, attr, "") or "")
                if src.exists():
                    dst = dest / src.name
                    shutil.move(str(src), str(dst))
                    setattr(r, attr, str(dst))
            stem = Path(getattr(r, "original_path", "")).stem
            for mf in tmp_dir.glob(f"{stem}_meta.json"):
                shutil.move(str(mf), str(overlay_dir.parent / mf.name))
