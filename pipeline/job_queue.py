# -*- coding: utf-8 -*-
"""
pipeline/job_queue.py
======================
Hàng đợi chạy nền (SQLite + thread worker) cho pipeline nặng.

Thay vì chạy model trực tiếp trong Flask request (gây treo web), luồng như sau:
    User upload ảnh → tạo job_id → worker xử lý → UI poll trạng thái

Dùng:
    from pipeline.job_queue import JobQueue
    queue = JobQueue.instance()
    job_id = queue.submit(image_paths, cfg)
    status = queue.get(job_id)
"""

import json
import logging
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("pipeline.job_queue")

_DB_PATH = Path("output/jobs.db")


# ─── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class JobStatus:
    job_id: str
    status: str          # pending / running / done / failed
    stage: str = ""
    progress: int = 0    # 0–100
    message: str = ""
    input_dir: str = ""
    output_dir: str = ""
    error: str = ""
    created_at: str = ""
    finished_at: str = ""
    result_summary: Dict[str, Any] = field(default_factory=dict)


# ─── DB helpers ────────────────────────────────────────────────────────────────

def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id       TEXT PRIMARY KEY,
            status       TEXT NOT NULL DEFAULT 'pending',
            stage        TEXT NOT NULL DEFAULT '',
            progress     INTEGER NOT NULL DEFAULT 0,
            message      TEXT NOT NULL DEFAULT '',
            input_dir    TEXT NOT NULL DEFAULT '',
            output_dir   TEXT NOT NULL DEFAULT '',
            error        TEXT NOT NULL DEFAULT '',
            created_at   TEXT NOT NULL,
            finished_at  TEXT NOT NULL DEFAULT '',
            result_json  TEXT NOT NULL DEFAULT '{}'
        )
    """)
    conn.commit()


# ─── JobQueue ──────────────────────────────────────────────────────────────────

class JobQueue:
    """
    Singleton job queue.

    Dùng:
        jq = JobQueue.instance()
        job_id = jq.submit(image_paths=[...], cfg={...})
        status = jq.get(job_id)     # JobStatus
        jq.stream(job_id)           # generator -> SSE lines
    """

    _instance: Optional["JobQueue"] = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls, db_path: Path = _DB_PATH) -> "JobQueue":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
        return cls._instance

    def __init__(self, db_path: Path = _DB_PATH):
        self._db_path = db_path
        self._conn = _connect(db_path)
        _init_db(self._conn)
        self._db_lock = threading.Lock()

        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        # SSE listeners: job_id -> list of Queue
        self._sse_listeners: Dict[str, List[queue.Queue]] = {}
        self._sse_lock = threading.Lock()

        log.info("[JobQueue] Started — db: %s", db_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def submit(
        self,
        image_paths: List[Path],
        cfg: dict,
        input_label: str = "",
    ) -> str:
        """Thêm job mới vào hàng đợi. Trả về job_id."""
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        now = datetime.now().isoformat(timespec="seconds")
        with self._db_lock:
            self._conn.execute(
                """INSERT INTO jobs
                   (job_id, status, input_dir, created_at)
                   VALUES (?, 'pending', ?, ?)""",
                (job_id, input_label or str(image_paths[0].parent if image_paths else ""), now),
            )
            self._conn.commit()
        self._queue.put((job_id, image_paths, cfg))
        log.info("[JobQueue] Queued job %s (%d images)", job_id, len(image_paths))
        return job_id

    def get(self, job_id: str) -> Optional[JobStatus]:
        """Trả về trạng thái hiện tại của job."""
        with self._db_lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if not row:
            return None
        result = {}
        try:
            result = json.loads(row["result_json"] or "{}")
        except Exception:
            pass
        return JobStatus(
            job_id=row["job_id"],
            status=row["status"],
            stage=row["stage"],
            progress=row["progress"],
            message=row["message"],
            input_dir=row["input_dir"],
            output_dir=row["output_dir"],
            error=row["error"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
            result_summary=result,
        )

    def list_recent(self, limit: int = 20) -> List[JobStatus]:
        """Trả về danh sách job gần đây nhất."""
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for row in rows:
            try:
                result = json.loads(row["result_json"] or "{}")
            except Exception:
                result = {}
            out.append(JobStatus(
                job_id=row["job_id"], status=row["status"], stage=row["stage"],
                progress=row["progress"], message=row["message"],
                input_dir=row["input_dir"], output_dir=row["output_dir"],
                error=row["error"], created_at=row["created_at"],
                finished_at=row["finished_at"], result_summary=result,
            ))
        return out

    def stream(self, job_id: str):
        """
        Generator trả về SSE events cho một job.
        Dùng trong Flask route với Response(stream_with_context(...)).
        """
        # Đăng ký listener
        q: queue.Queue = queue.Queue()
        with self._sse_lock:
            self._sse_listeners.setdefault(job_id, []).append(q)

        # Gửi trạng thái hiện tại ngay lập tức
        status = self.get(job_id)
        if status:
            yield _sse_event(status)

        try:
            while True:
                try:
                    event_data = q.get(timeout=30)
                    yield event_data
                    if event_data and ('"done"' in event_data or '"failed"' in event_data):
                        break
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with self._sse_lock:
                listeners = self._sse_listeners.get(job_id, [])
                if q in listeners:
                    listeners.remove(q)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _update(self, job_id: str, **kwargs):
        """Cập nhật trạng thái job trong DB và notify SSE listeners."""
        set_clause = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [job_id]
        with self._db_lock:
            self._conn.execute(
                f"UPDATE jobs SET {set_clause} WHERE job_id=?", vals
            )
            self._conn.commit()
        # Notify SSE
        status = self.get(job_id)
        if status:
            event = _sse_event(status)
            with self._sse_lock:
                for q in self._sse_listeners.get(job_id, []):
                    q.put(event)

    def _make_progress_callback(self, job_id: str) -> Callable:
        """Tạo callback để pipeline gọi khi có tiến độ mới."""
        def callback(stage: str, current: int, total: int, message: str = ""):
            pct = int(current / total * 100) if total > 0 else 0
            msg = message or f"{stage}: {current}/{total}"
            self._update(job_id, stage=stage, progress=pct, message=msg)
        return callback

    def _worker_loop(self):
        """Worker thread liên tục lấy job từ queue và xử lý."""
        log.info("[JobQueue] Worker thread started")
        while True:
            try:
                job_id, image_paths, cfg = self._queue.get(timeout=5)
                self._process_job(job_id, image_paths, cfg)
            except queue.Empty:
                continue
            except Exception as exc:
                log.error("[JobQueue] Worker error: %s", exc, exc_info=True)

    def _process_job(self, job_id: str, image_paths: List[Path], cfg: dict):
        log.info("[JobQueue] Processing job %s", job_id)
        self._update(job_id, status="running", stage="init", progress=0,
                     message="Đang khởi động pipeline...")
        try:
            from pipeline.orchestrator import FloodPipeline, PipelineState
            progress_cb = self._make_progress_callback(job_id)
            pipeline = FloodPipeline(cfg, progress_callback=progress_cb)
            state: PipelineState = pipeline.run(image_paths)

            result = {
                "run_id":        state.run_id,
                "total_input":   len(state.input_images),
                "total_analyzed": len(state.depth_results),
                "errors":        state.errors,
                "timings":       state.timings,
                "output_dir":    str(state.output_dir or ""),
            }
            self._update(
                job_id,
                status="done",
                stage="complete",
                progress=100,
                message=f"Xong — {len(state.depth_results)} ảnh phân tích",
                output_dir=str(state.output_dir or ""),
                finished_at=datetime.now().isoformat(timespec="seconds"),
                result_json=json.dumps(result, ensure_ascii=False),
            )
            log.info("[JobQueue] Job %s done", job_id)

        except Exception as exc:
            log.error("[JobQueue] Job %s failed: %s", job_id, exc, exc_info=True)
            self._update(
                job_id,
                status="failed",
                stage="error",
                progress=0,
                message="Pipeline thất bại",
                error=str(exc),
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )


# ── SSE helper ─────────────────────────────────────────────────────────────────

def _sse_event(status: JobStatus) -> str:
    data = {
        "job_id":   status.job_id,
        "status":   status.status,
        "stage":    status.stage,
        "progress": status.progress,
        "message":  status.message,
        "error":    status.error,
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
