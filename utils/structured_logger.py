# -*- coding: utf-8 -*-
"""
utils/structured_logger.py  —  Structured Logging cho Pipeline
===============================================================
Thay thế print() và logging.info(chuỗi thường) bằng structured logs.

Tại sao cần:
  - print() / log string thường → không searchable, không parseable
  - Với structured logs → dễ query, dễ đưa vào monitoring (ELK, Datadog, v.v.)

Output format: JSON per line (JSONL), máy đọc được.

Sử dụng:
    plog = PipelineLogger("depth_stage")

    plog.stage_start("depth", n_images=5)
    plog.stage_end("depth", duration=2.3, n_processed=5)
    plog.confidence_event("img001.jpg", confidence=0.32, route="human_review")
    plog.error("depth", "Model OOM", exc=e)
    plog.metric("flood_level", "KNEE", image="img001.jpg", depth_cm=45.2)

    # Lấy stats tổng hợp sau run
    summary = plog.get_run_summary()
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ── Structured log record ──────────────────────────────────────────────────────

def _make_record(
    event: str,
    stage: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Tạo một structured log record."""
    record = {
        "ts":    datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    if stage:
        record["stage"] = stage
    record.update({k: v for k, v in kwargs.items() if v is not None})
    return record


# ── Pipeline Logger ────────────────────────────────────────────────────────────

class PipelineLogger:
    """
    Structured logger cho flood pipeline.

    Thread-safe. Ghi ra:
      - Python logger (INFO level) → terminal / log file thường
      - JSONL file → cho monitoring / analytics (nếu log_dir được set)

    Sử dụng:
        plog = PipelineLogger("my_run_id", log_dir="output/logs")
        plog.stage_start("depth", n_images=10)
        plog.metric("confidence", 0.82, image="img1.jpg")
        plog.stage_end("depth", duration=3.2, n_processed=10)
    """

    def __init__(
        self,
        run_id: str = "unknown",
        log_dir: Optional[str] = None,
        component: str = "pipeline",
    ):
        self.run_id    = run_id
        self.component = component
        self._lock     = threading.Lock()
        self._events: List[dict] = []

        # JSONL file output
        self._file = None
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            jsonl_path = log_path / f"pipeline_{run_id}.jsonl"
            try:
                self._file = open(jsonl_path, "a", encoding="utf-8")
                log.debug(f"  [PipelineLogger] JSONL → {jsonl_path}")
            except OSError as e:
                log.warning(f"  [PipelineLogger] Cannot open log file: {e}")

        self._py_log = logging.getLogger(f"pipeline.{component}")

    # ── Core emit ──────────────────────────────────────────────────────────────

    def emit(self, event: str, stage: Optional[str] = None, **kwargs) -> None:
        """Emit một event có cấu trúc."""
        record = _make_record(event, stage=stage, run_id=self.run_id, **kwargs)

        with self._lock:
            self._events.append(record)
            if self._file:
                try:
                    self._file.write(json.dumps(record, default=str) + "\n")
                    self._file.flush()
                except OSError:
                    pass

        # Human-readable log
        self._py_log.info(self._format_human(record))

    # ── Stage events ───────────────────────────────────────────────────────────

    def stage_start(self, stage: str, **kwargs) -> float:
        """Log bắt đầu stage, trả về start timestamp."""
        self.emit("stage_start", stage=stage, **kwargs)
        return time.time()

    def stage_end(
        self,
        stage: str,
        duration: float,
        n_processed: int = 0,
        **kwargs,
    ) -> None:
        """Log kết thúc stage với timing."""
        self.emit(
            "stage_end",
            stage=stage,
            duration_s=round(duration, 3),
            n_processed=n_processed,
            **kwargs,
        )

    def stage_skip(self, stage: str, reason: str = "") -> None:
        """Log stage bị skip."""
        self.emit("stage_skip", stage=stage, reason=reason)

    # ── Confidence events ──────────────────────────────────────────────────────

    def confidence_event(
        self,
        image: str,
        confidence: float,
        route: str,
        components: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Log confidence score và routing decision."""
        self.emit(
            "confidence",
            image=image,
            confidence=round(confidence, 4),
            route=route,
            components=components,
            **kwargs,
        )

    # ── Metric events ──────────────────────────────────────────────────────────

    def metric(self, name: str, value: Any, **kwargs) -> None:
        """Log một metric (flood level, depth_cm, v.v.)."""
        self.emit("metric", metric=name, value=value, **kwargs)

    def flood_result(
        self,
        image: str,
        flood_level: str,
        depth_cm: Optional[float],
        confidence: float,
        **kwargs,
    ) -> None:
        """Log kết quả phát hiện lũ cho một ảnh."""
        self.emit(
            "flood_result",
            image=image,
            flood_level=flood_level,
            depth_cm=depth_cm,
            confidence=round(confidence, 4),
            **kwargs,
        )

    # ── Error events ───────────────────────────────────────────────────────────

    def error(self, stage: str, message: str, exc: Optional[Exception] = None, **kwargs) -> None:
        """Log lỗi có cấu trúc."""
        self.emit(
            "error",
            stage=stage,
            message=message,
            exc_type=type(exc).__name__ if exc else None,
            exc_msg=str(exc) if exc else None,
            **kwargs,
        )
        # Cũng log qua Python logger bình thường để thấy trong console
        self._py_log.error(f"  [ERROR] {stage}: {message}" + (f" — {exc}" if exc else ""))

    # ── Memory events ──────────────────────────────────────────────────────────

    def memory_snapshot(self, label: str = "") -> None:
        """Log memory usage hiện tại."""
        from utils.memory_manager import get_memory_usage
        info = get_memory_usage()
        self.emit("memory", label=label, **info)

    # ── Summary ────────────────────────────────────────────────────────────────

    def get_run_summary(self) -> dict:
        """Tổng hợp thống kê sau khi pipeline chạy xong."""
        with self._lock:
            events = list(self._events)

        stage_times: Dict[str, float] = {}
        errors: List[dict] = []
        flood_results: List[dict] = []
        routes: Dict[str, int] = {}

        for e in events:
            evt = e.get("event")
            if evt == "stage_end":
                stage_times[e.get("stage", "?")] = e.get("duration_s", 0)
            elif evt == "error":
                errors.append(e)
            elif evt == "flood_result":
                flood_results.append(e)
            elif evt == "confidence":
                route = e.get("route", "unknown")
                routes[route] = routes.get(route, 0) + 1

        return {
            "run_id":       self.run_id,
            "total_events": len(events),
            "stage_times":  stage_times,
            "total_time_s": sum(stage_times.values()),
            "n_errors":     len(errors),
            "n_processed":  len(flood_results),
            "routing":      routes,
            "errors":       [{"stage": e.get("stage"), "msg": e.get("message")} for e in errors],
        }

    def log_summary(self) -> None:
        """In summary ra log."""
        s = self.get_run_summary()
        self._py_log.info(f"\n  📊 Run Summary [{s['run_id']}]")
        self._py_log.info(f"     Processed: {s['n_processed']} images")
        self._py_log.info(f"     Total time: {s['total_time_s']:.1f}s")
        self._py_log.info(f"     Errors: {s['n_errors']}")
        self._py_log.info(f"     Routing: {s['routing']}")
        for stage, t in s["stage_times"].items():
            self._py_log.info(f"       {stage:20s}: {t:.2f}s")

    def close(self) -> None:
        """Đóng log file (gọi khi pipeline xong)."""
        if self._file:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def __del__(self):
        self.close()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_human(record: dict) -> str:
        """Chuyển structured record thành human-readable string."""
        parts = [f"[{record.get('event', '?')}]"]
        if "stage" in record:
            parts.append(f"stage={record['stage']}")
        for key in ["image", "flood_level", "depth_cm", "confidence", "route",
                    "duration_s", "n_processed", "message"]:
            if key in record and record[key] is not None:
                parts.append(f"{key}={record[key]}")
        return "  " + " ".join(parts)


# ── Setup helper ───────────────────────────────────────────────────────────────

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    json_mode: bool = False,
) -> None:
    """
    Cấu hình logging cho toàn bộ pipeline.

    Args:
        level:     Log level (default INFO)
        log_file:  File để ghi log (optional)
        json_mode: Nếu True, mỗi record là JSON line
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    fmt = (
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
        if not json_mode
        else '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
    )

    logging.basicConfig(
        level=level,
        handlers=handlers,
        format=fmt,
        datefmt="%H:%M:%S",
        force=True,
    )

    # Tắt bớt noise từ thư viện bên ngoài
    for noisy in ["urllib3", "PIL", "transformers", "ultralytics"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
