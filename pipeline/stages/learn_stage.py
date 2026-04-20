# -*- coding: utf-8 -*-
"""
Stage 7 — Learn
================
Self-learning pipeline: cập nhật model từ kết quả mới.

Features:
  - Ghi kết quả vào review queue để human-in-the-loop
  - Theo dõi error rate → auto-retrain khi vượt ngưỡng
  - Adaptive thresholds: tự chỉnh ngưỡng blur/content dựa trên hiệu suất
"""
import logging
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
            sl.process_results(
                depth_results=depth_results,
                cfg=cfg,
                image_paths=image_paths,
            )

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
        Hiện tại: invalidate cache → model sẽ được retrain lần chạy sau.
        Future: gửi task tới Celery worker.
        """
        try:
            from learning.ai_learner import AiLearner
            learner = AiLearner()
            learner.invalidate()
            log.info("  [Learn] Cache đã invalidate → model sẽ retrain lần sau")
        except Exception as exc:
            log.warning(f"  [Learn] Retrain trigger thất bại: {exc}")
