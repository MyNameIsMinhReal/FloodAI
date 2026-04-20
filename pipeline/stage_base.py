# -*- coding: utf-8 -*-
"""
pipeline/stage_base.py  —  Base Stage Interface
=================================================
Interface chuẩn cho tất cả pipeline stages.

Tại sao cần:
  - Hiện tại các stage không có contract chung → khó plug thêm stage mới
  - Không có timing, logging chuẩn
  - Không retry, không health check

Sau khi có StageBase:
  - Tất cả stage kế thừa → dễ thêm stage mới
  - Auto-timing, structured logging
  - Retry logic built-in
  - Health check dễ test

Sử dụng:
    class MyNewStage(StageBase):
        name = "my_stage"

        def process(self, state: PipelineState) -> PipelineState:
            # logic của bạn ở đây
            return state

    # Trong orchestrator:
    stage = MyNewStage(cfg)
    state = stage.run(state)    # auto timing + logging + retry
"""

from __future__ import annotations

import abc
import logging
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from pipeline.orchestrator import PipelineState

log = logging.getLogger("pipeline.stage_base")


class StageBase(abc.ABC):
    """
    Abstract base class cho tất cả pipeline stages.

    Subclass chỉ cần implement:
        - name: str  (class attribute)
        - process(state) → state

    Nhận free:
        - Auto timing
        - Structured logging
        - Exception handling
        - Retry logic (nếu enable)
    """

    # Subclass PHẢI override cái này
    name: str = "unnamed_stage"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._log = logging.getLogger(f"pipeline.stage.{self.name}")
        self._max_retries: int = cfg.get("stage_max_retries", 0)
        self._enabled: bool = not cfg.get(f"skip_{self.name}", False)

    # ── Abstract method ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def process(self, state: "PipelineState") -> "PipelineState":
        """
        Logic chính của stage.

        Args:
            state: PipelineState hiện tại

        Returns:
            PipelineState đã cập nhật
        """
        ...

    # ── Public API (gọi từ orchestrator) ───────────────────────────────────────

    def run(self, state: "PipelineState") -> "PipelineState":
        """
        Chạy stage với auto-timing, logging, retry.

        Không nên override — override `process()` thay vào đó.
        """
        if not self._enabled:
            self._log.info(f"  [SKIP] Stage '{self.name}' bị tắt trong config")
            return state

        self._log.info(f"\n── Stage: {self.name.upper()} ──────────────────────")

        attempts = 0
        last_exc = None

        while attempts <= self._max_retries:
            t0 = time.time()
            try:
                state = self.process(state)
                elapsed = time.time() - t0
                state.timings[self.name] = elapsed
                self._log_success(state, elapsed)
                return state

            except Exception as exc:
                elapsed = time.time() - t0
                last_exc = exc
                attempts += 1

                if attempts <= self._max_retries:
                    self._log.warning(
                        f"  [RETRY {attempts}/{self._max_retries}] "
                        f"Stage '{self.name}' lỗi: {exc} — thử lại…"
                    )
                    time.sleep(1.0 * attempts)  # exponential backoff đơn giản
                else:
                    self._log.error(
                        f"  [ERROR] Stage '{self.name}' thất bại sau "
                        f"{elapsed:.1f}s: {exc}",
                        exc_info=True,
                    )
                    state.errors.append(f"{self.name}: {exc}")

        return state

    # ── Health check ───────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """
        Kiểm tra stage có thể chạy không (dependencies, config, v.v.).

        Returns:
            {"ok": bool, "message": str}
        """
        return {"ok": True, "message": f"Stage '{self.name}' sẵn sàng"}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _log_success(self, state: "PipelineState", elapsed: float) -> None:
        """Log structured sau khi stage thành công."""
        self._log.info(
            f"  ✓ [{self.name}] "
            f"items={self._count_items(state)} — {elapsed:.1f}s"
        )

    def _count_items(self, state: "PipelineState") -> int:
        """Đếm số items đã xử lý (override nếu cần)."""
        return len(getattr(state, "depth_results", []) or
                   getattr(state, "input_images", []) or [])

    def get_cfg(self, key: str, default: Any = None) -> Any:
        """Helper để lấy config với default."""
        return self.cfg.get(key, default)
