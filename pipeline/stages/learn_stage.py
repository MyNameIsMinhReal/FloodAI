# -*- coding: utf-8 -*-
"""
Stage 7 — Learn
================
Self-learning pipeline: cập nhật model từ kết quả mới.

Features:
  - Ghi kết quả vào review queue để human-in-the-loop
  - Theo dõi error rate → auto-retrain khi vượt ngưỡng
  - Adaptive thresholds: tự chỉnh ngưỡng blur/content dựa trên hiệu suất
  - [v4] Auto-trigger fine-tune segmentation khi đủ data review
"""
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

log = logging.getLogger("pipeline.learn")

if TYPE_CHECKING:
    pass


class LearnStage:
    """Self-learning update sau mỗi pipeline run."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._error_threshold = cfg.get("auto_retrain_threshold", 0.20)

    def update(
        self,
        depth_results: List[Any],
        cfg: dict,
        image_paths: List[Path],
    ) -> None:
        """
        Cập nhật learning system:
        1. Lưu kết quả vào review queue
        2. Chạy AI learner update
        3. Kiểm tra error rate → retrain nếu cần
        """
        try:
            from learning_update import SelfLearningPipeline
            sl  = SelfLearningPipeline()
            cfg = sl.get_adaptive_config(cfg)
            sl.process_results(depth_results, cfg, image_paths)

            # Kiểm tra error rate và auto-retrain
            error_rate = self._get_error_rate(sl)
            if error_rate > self._error_threshold:
                log.warning(
                    f"  [Learn] Error rate {error_rate:.0%} > threshold "
                    f"{self._error_threshold:.0%} → trigger retrain"
                )
                self._trigger_retrain(sl)
            else:
                log.info(f"  [Learn] Error rate: {error_rate:.0%} (OK)")

            sl.close()
        except Exception as exc:
            log.warning(f"  [Learn] Bỏ qua: {exc}")

    def _get_error_rate(self, sl) -> float:
        """Lấy error rate từ error tracker."""
        try:
            if hasattr(sl, "error_tracker") and sl.error_tracker:
                stats = sl.error_tracker.get_stats()
                total = stats.get("total", 0)
                errors = stats.get("errors", 0)
                return errors / total if total > 0 else 0.0
        except Exception:
            pass
        return 0.0

    def _trigger_retrain(self, sl) -> None:
        """
        Kích hoạt retrain model.
        1. Invalidate cache → model sẽ được retrain lần chạy sau.
        2. [v4] Nếu dataset flood segmentation tồn tại → chạy fine-tune
           segmentation song song (subprocess, background).
        """
        try:
            from learning.ai_learner import AiLearner
            learner = AiLearner()
            learner.invalidate()
            log.info("  [Learn] Cache đã invalidate → model sẽ retrain lần sau")
        except Exception as exc:
            log.warning(f"  [Learn] Retrain trigger thất bại: {exc}")

        # ── [v4 Gap C] Auto-trigger fine-tune segmentation ──────────────────
        # Kiểm tra xem dataset FloodNet/RescueNet đã có chưa. Nếu có → chạy
        # finetune_segmentation.py trong background (subprocess) để fine-tune
        # SegFormer-B0 thành flood binary classifier.
        finetune_cfg = self.cfg.get("finetune_seg", {})
        dataset_dir  = finetune_cfg.get("dataset_dir", "datasets/FloodNet")
        output_dir   = finetune_cfg.get("output_dir", "models/flood_segnet")
        epochs       = finetune_cfg.get("epochs", 20)

        ds_path = Path(dataset_dir)
        train_dir = ds_path / "train" / "images"
        if not train_dir.exists():
            log.debug(f"  [Finetune] Dataset không tồn tại: {train_dir} → bỏ qua")
            return

        try:
            script = Path(__file__).resolve().parents[2] / "learning" / "finetune_segmentation.py"
            if not script.exists():
                log.debug(f"  [Finetune] Script không tồn tại: {script} → bỏ qua")
                return

            log.info(f"  [Finetune] Launching segmentation fine-tune → {output_dir}")
            subprocess.Popen(
                [
                    sys.executable, str(script),
                    "--dataset_dir", dataset_dir,
                    "--output_dir", output_dir,
                    "--epochs", str(epochs),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            log.warning(f"  [Finetune] Launch thất bại: {exc}")
