# -*- coding: utf-8 -*-
"""
pipeline/service.py
====================
FloodAnalysisService — lớp service mỏng tách biệt pipeline core ra khỏi web / CLI.

Web app chỉ gọi service, không tự xử lý logic pipeline:

    service = FloodAnalysisService(cfg)
    state   = service.analyze_folder("/data/floods")
    state   = service.analyze_images([Path("a.jpg"), Path("b.jpg")])
    job_id  = service.submit_async(image_paths)  # chạy nền, trả về job_id ngay

Lợi ích:
  - Dùng qua CLI, web route, REST API mà không bị dính logic vào nhau
  - Dễ test mock
  - Chỉ 1 nơi khởi tạo FloodPipeline
"""

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.orchestrator import PipelineState

log = logging.getLogger("pipeline.service")


class FloodAnalysisService:
    """
    Service wrapper cho FloodPipeline.

    Dùng sync (CLI / scripts):
        service = FloodAnalysisService(cfg)
        state = service.analyze_folder("/data/floods")

    Dùng async (web uploads):
        job_id = service.submit_async(image_paths)
        status = service.job_status(job_id)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._pipeline = None  # lazy init

    # ── Lazy pipeline init ─────────────────────────────────────────────────────

    def _get_pipeline(self):
        if self._pipeline is None:
            from pipeline.orchestrator import FloodPipeline
            self._pipeline = FloodPipeline(self.cfg)
            log.info("[Service] FloodPipeline initialized")
        return self._pipeline

    # ── Synchronous API ────────────────────────────────────────────────────────

    def analyze_folder(self, folder_path: "str | Path") -> "PipelineState":
        """
        Phân tích toàn bộ ảnh trong folder_path.

        Args:
            folder_path: đường dẫn thư mục (str hoặc Path)

        Returns:
            PipelineState chứa kết quả đầy đủ
        """
        return self._get_pipeline().run_from_dir(Path(folder_path))

    def analyze_images(self, image_paths: List[Path]) -> "PipelineState":
        """
        Phân tích danh sách ảnh cụ thể.

        Args:
            image_paths: list[Path]

        Returns:
            PipelineState
        """
        return self._get_pipeline().run(image_paths)

    # ── Asynchronous API (job queue) ───────────────────────────────────────────

    def submit_async(
        self,
        image_paths: List[Path],
        input_label: str = "",
    ) -> str:
        """
        Gửi job vào hàng đợi chạy nền.

        Args:
            image_paths: list[Path] ảnh cần phân tích
            input_label: nhãn hiển thị (ví dụ tên folder)

        Returns:
            job_id (str) — dùng để poll trạng thái qua /api/jobs/<job_id>
        """
        from pipeline.job_queue import JobQueue
        return JobQueue.instance().submit(image_paths, self.cfg, input_label)

    def job_status(self, job_id: str):
        """Trả về JobStatus của job."""
        from pipeline.job_queue import JobQueue
        return JobQueue.instance().get(job_id)

    def recent_jobs(self, limit: int = 20):
        """Trả về danh sách job gần đây."""
        from pipeline.job_queue import JobQueue
        return JobQueue.instance().list_recent(limit)

    def stream_job(self, job_id: str):
        """Generator SSE events cho job — dùng trong Flask SSE route."""
        from pipeline.job_queue import JobQueue
        return JobQueue.instance().stream(job_id)

    # ── Health check ──────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Trả về thông tin sức khỏe cơ bản của service."""
        import sys
        from pipeline.job_queue import JobQueue

        jq = JobQueue.instance()
        recent = jq.list_recent(50)
        pending  = sum(1 for j in recent if j.status == "pending")
        running  = sum(1 for j in recent if j.status == "running")
        done     = sum(1 for j in recent if j.status == "done")
        failed   = sum(1 for j in recent if j.status == "failed")

        return {
            "status": "ok",
            "python": sys.version,
            "jobs": {
                "pending": pending,
                "running": running,
                "done":    done,
                "failed":  failed,
            },
        }


# ── Module-level singleton helpers ────────────────────────────────────────────

_service: Optional[FloodAnalysisService] = None


def get_service(cfg: Optional[dict] = None) -> FloodAnalysisService:
    """
    Trả về singleton FloodAnalysisService.
    Lần đầu gọi phải truyền cfg vào.
    """
    global _service
    if _service is None:
        if cfg is None:
            raise RuntimeError("FloodAnalysisService chưa được init. Truyền cfg vào lần đầu.")
        _service = FloodAnalysisService(cfg)
    return _service
