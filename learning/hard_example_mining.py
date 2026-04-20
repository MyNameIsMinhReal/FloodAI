# -*- coding: utf-8 -*-
"""
learning/hard_example_mining.py  —  Hard Example Mining
=========================================================
Học từ các trường hợp KHÓ thay vì random sampling.

Tại sao quan trọng:
  - Random sampling: mất thời gian review case dễ (đã biết rồi)
  - Hard mining: chỉ học từ case SAI hoặc confidence THẤP
  → Model cải thiện nhanh hơn với ít data hơn

Hard examples là:
  1. Case SAI (predicted ≠ actual theo human review)
  2. Case confidence THẤP (< threshold)
  3. Case ở LEVEL HIẾM (CHEST, SUBMERGED → ít data training)
  4. Case outlier trong feature space (clustering)

Pipeline:
  prediction → [auto-score hardness] → hard queue → human review
  → confirmed error → training batch

Sử dụng:
    miner = HardExampleMiner()

    # Score một prediction
    score = miner.score_hardness(result, confidence=0.35)
    if score.is_hard:
        miner.add_to_hard_queue(result, score)

    # Lấy batch hard examples để review
    batch = miner.get_review_batch(n=20)

    # Sau khi human confirm lỗi:
    miner.confirm_error(case_id, actual_level="KNEE")

    # Lấy batch training
    training_batch = miner.get_training_batch(n=50)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("learning.hard_example_mining")

DB_PATH = "learning/hard_examples.db"

# Thresholds
CONF_HARD_THRESHOLD    = 0.55   # confidence < này → hard
CONF_VERY_HARD_THRESHOLD = 0.35 # confidence < này → very hard
RARE_LEVELS = {"CHEST", "SUBMERGED", "WAIST"}


# ── Hardness score ─────────────────────────────────────────────────────────────

@dataclass
class HardnessScore:
    """Điểm "độ khó" của một prediction."""
    total: float = 0.0           # 0..100, cao = khó hơn = ưu tiên review

    low_confidence: float = 0.0  # component: confidence thấp
    rare_level:     float = 0.0  # component: level hiếm gặp
    error_pattern:  float = 0.0  # component: khớp pattern lỗi đã biết
    uncertainty:    float = 0.0  # component: model uncertainty

    @property
    def is_hard(self) -> bool:
        return self.total >= 30.0

    @property
    def priority(self) -> str:
        if self.total >= 70: return "critical"
        if self.total >= 50: return "high"
        if self.total >= 30: return "medium"
        return "low"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Hard Example ───────────────────────────────────────────────────────────────

@dataclass
class HardExample:
    """Một hard example trong queue."""
    id:              Optional[int]
    image_path:      str
    predicted_level: str
    predicted_depth: Optional[float]
    confidence:      float
    hardness_score:  float
    priority:        str
    features:        str    # JSON
    status:          str    # pending | reviewed | confirmed_error | false_alarm
    created_at:      str
    actual_level:    Optional[str] = None
    actual_depth:    Optional[float] = None
    review_notes:    str = ""

    def is_error(self) -> bool:
        """True nếu confirmed là prediction sai."""
        return (self.status == "confirmed_error" and
                self.actual_level is not None and
                self.actual_level != self.predicted_level)


# ── Hard Example Miner ─────────────────────────────────────────────────────────

class HardExampleMiner:
    """
    Mining hard examples từ pipeline predictions.

    Features:
      - Score hardness từ nhiều signals
      - SQLite queue với priority ordering
      - Cluster error patterns
      - Export training batch
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

        # Theo dõi error patterns
        self._error_patterns: List[dict] = []
        self._load_error_patterns()

    # ── Scoring ────────────────────────────────────────────────────────────────

    def score_hardness(
        self,
        result: Any,
        confidence: float,
        predicted_level: Optional[str] = None,
    ) -> HardnessScore:
        """
        Tính điểm "độ khó" cho một prediction.

        Cao = nên review ngay.
        """
        score = HardnessScore()
        get = _get_attr

        # Component 1: Low confidence
        if confidence < CONF_VERY_HARD_THRESHOLD:
            score.low_confidence = 60.0
        elif confidence < CONF_HARD_THRESHOLD:
            score.low_confidence = 30.0
        else:
            score.low_confidence = max(0, (CONF_HARD_THRESHOLD - confidence) * 100)

        # Component 2: Rare level
        level = (predicted_level or
                 get(result, "flood_level") or
                 get(result, "level") or "")
        if level in RARE_LEVELS:
            score.rare_level = 25.0

        # Component 3: Error pattern match
        score.error_pattern = self._match_error_patterns(result)

        # Component 4: Model uncertainty (nếu có)
        uncertainty = get(result, "model_uncertainty") or get(result, "uncertainty")
        if uncertainty is not None:
            score.uncertainty = float(uncertainty) * 20.0

        # Tổng hợp (weighted)
        score.total = min(100.0, (
            score.low_confidence * 0.45 +
            score.rare_level     * 0.25 +
            score.error_pattern  * 0.20 +
            score.uncertainty    * 0.10
        ))

        return score

    # ── Queue management ───────────────────────────────────────────────────────

    def add_to_hard_queue(
        self,
        result: Any,
        score: HardnessScore,
        confidence: float,
    ) -> Optional[int]:
        """
        Thêm một hard example vào review queue.

        Returns:
            ID của record được tạo, hoặc None nếu đã tồn tại
        """
        get = _get_attr
        image_path = str(get(result, "original_path") or get(result, "image_path") or "")
        if not image_path:
            return None

        level = str(get(result, "flood_level") or get(result, "level") or "UNKNOWN")
        depth = get(result, "flood_depth_cm") or get(result, "depth_cm")

        features = {
            "water_area_pct":  get(result, "water_area_pct"),
            "reference_count": len(get(result, "reference_objects") or []),
            "has_raincoat":    get(result, "has_raincoat"),
        }

        with self._lock:
            conn = self._conn()
            try:
                # Kiểm tra duplicate
                existing = conn.execute(
                    "SELECT id FROM hard_examples WHERE image_path=? AND status='pending'",
                    (image_path,)
                ).fetchone()
                if existing:
                    log.debug(f"  [HardMining] Duplicate skip: {image_path}")
                    return None

                cursor = conn.execute(
                    """INSERT INTO hard_examples
                       (image_path, predicted_level, predicted_depth, confidence,
                        hardness_score, priority, features, status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                    (
                        image_path, level,
                        float(depth) if depth else None,
                        round(confidence, 4),
                        round(score.total, 2),
                        score.priority,
                        json.dumps(features),
                        datetime.now().isoformat(),
                    )
                )
                conn.commit()
                log.info(
                    f"  [HardMining] Added hard example: {Path(image_path).name} "
                    f"score={score.total:.0f} priority={score.priority}"
                )
                return cursor.lastrowid
            finally:
                conn.close()

    def get_review_batch(
        self,
        n: int = 20,
        priority_filter: Optional[str] = None,
    ) -> List[HardExample]:
        """Lấy batch cases cần review, sắp xếp theo priority."""
        with self._lock:
            conn = self._conn()
            try:
                query = "SELECT * FROM hard_examples WHERE status='pending'"
                params = []
                if priority_filter:
                    query += " AND priority=?"
                    params.append(priority_filter)
                query += " ORDER BY hardness_score DESC LIMIT ?"
                params.append(n)

                rows = conn.execute(query, params).fetchall()
                return [self._row_to_example(r) for r in rows]
            finally:
                conn.close()

    def confirm_error(
        self,
        case_id: int,
        actual_level: str,
        actual_depth: Optional[float] = None,
        notes: str = "",
    ) -> None:
        """Human confirm đây là prediction sai."""
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    """UPDATE hard_examples SET
                       status='confirmed_error',
                       actual_level=?,
                       actual_depth=?,
                       review_notes=?
                       WHERE id=?""",
                    (actual_level, actual_depth, notes, case_id)
                )
                conn.commit()

                # Học error pattern từ trường hợp này
                row = conn.execute(
                    "SELECT * FROM hard_examples WHERE id=?", (case_id,)
                ).fetchone()
                if row:
                    self._update_error_pattern(self._row_to_example(row))

                log.info(f"  [HardMining] Confirmed error case {case_id}: actual={actual_level}")
            finally:
                conn.close()

    def mark_false_alarm(self, case_id: int) -> None:
        """Đánh dấu trường hợp này là false alarm (prediction đúng)."""
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "UPDATE hard_examples SET status='false_alarm' WHERE id=?",
                    (case_id,)
                )
                conn.commit()
            finally:
                conn.close()

    def get_training_batch(self, n: int = 50) -> List[HardExample]:
        """
        Lấy batch confirmed errors để training.
        Chỉ bao gồm các case được confirm là SAI.
        """
        with self._lock:
            conn = self._conn()
            try:
                rows = conn.execute(
                    """SELECT * FROM hard_examples
                       WHERE status='confirmed_error' AND actual_level IS NOT NULL
                       ORDER BY hardness_score DESC LIMIT ?""",
                    (n,)
                ).fetchall()
                return [self._row_to_example(r) for r in rows]
            finally:
                conn.close()

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Thống kê hard example queue."""
        with self._lock:
            conn = self._conn()
            try:
                stats = {}
                for status in ["pending", "confirmed_error", "false_alarm", "reviewed"]:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM hard_examples WHERE status=?", (status,)
                    ).fetchone()[0]
                    stats[status] = count

                # Error rate
                total_reviewed = stats["confirmed_error"] + stats["false_alarm"]
                stats["error_rate"] = (
                    stats["confirmed_error"] / total_reviewed
                    if total_reviewed > 0 else 0.0
                )

                # By priority
                for prio in ["critical", "high", "medium"]:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM hard_examples WHERE priority=? AND status='pending'",
                        (prio,)
                    ).fetchone()[0]
                    stats[f"pending_{prio}"] = count

                return stats
            finally:
                conn.close()

    def log_stats(self) -> None:
        s = self.get_stats()
        log.info(f"  [HardMining] Queue stats:")
        log.info(f"    Pending:   {s['pending']} (critical={s.get('pending_critical',0)}, high={s.get('pending_high',0)})")
        log.info(f"    Confirmed: {s['confirmed_error']}")
        log.info(f"    False alarms: {s['false_alarm']}")
        log.info(f"    Error rate: {s.get('error_rate', 0):.1%}")

    # ── Error patterns ─────────────────────────────────────────────────────────

    def _match_error_patterns(self, result: Any) -> float:
        """
        Kiểm tra xem result có khớp với error patterns đã biết không.
        Trả về score 0..30.
        """
        if not self._error_patterns:
            return 0.0

        get = _get_attr
        level = str(get(result, "flood_level") or "")
        area  = float(get(result, "water_area_pct") or 0)

        score = 0.0
        for pattern in self._error_patterns:
            # Level pattern match
            if pattern.get("level") == level:
                score += 10.0
            # Area range match
            area_range = pattern.get("water_area_range", [0, 100])
            if area_range[0] <= area <= area_range[1]:
                score += 5.0

        return min(30.0, score)

    def _update_error_pattern(self, example: HardExample) -> None:
        """Cập nhật error patterns sau khi confirm một lỗi."""
        features = json.loads(example.features or "{}")
        pattern = {
            "level":            example.predicted_level,
            "actual_level":     example.actual_level,
            "water_area_range": [
                max(0, (features.get("water_area_pct") or 0) - 10),
                min(100, (features.get("water_area_pct") or 0) + 10),
            ],
        }
        self._error_patterns.append(pattern)
        self._save_error_patterns()

    def _load_error_patterns(self) -> None:
        path = self.db_path.parent / "error_patterns.json"
        if path.exists():
            try:
                self._error_patterns = json.loads(path.read_text())
            except Exception:
                self._error_patterns = []

    def _save_error_patterns(self) -> None:
        path = self.db_path.parent / "error_patterns.json"
        try:
            path.write_text(json.dumps(self._error_patterns, indent=2))
        except Exception as exc:
            log.warning(f"  [HardMining] Cannot save patterns: {exc}")

    # ── DB ─────────────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS hard_examples (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_path      TEXT NOT NULL,
                        predicted_level TEXT,
                        predicted_depth REAL,
                        confidence      REAL,
                        hardness_score  REAL,
                        priority        TEXT,
                        features        TEXT,
                        status          TEXT DEFAULT 'pending',
                        created_at      TEXT,
                        actual_level    TEXT,
                        actual_depth    REAL,
                        review_notes    TEXT DEFAULT ''
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_status ON hard_examples(status)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_score ON hard_examples(hardness_score)"
                )
                conn.commit()
            finally:
                conn.close()

    def _row_to_example(self, row) -> HardExample:
        d = dict(row)
        return HardExample(
            id=d["id"],
            image_path=d["image_path"],
            predicted_level=d.get("predicted_level", ""),
            predicted_depth=d.get("predicted_depth"),
            confidence=d.get("confidence", 0.0),
            hardness_score=d.get("hardness_score", 0.0),
            priority=d.get("priority", "medium"),
            features=d.get("features", "{}"),
            status=d.get("status", "pending"),
            created_at=d.get("created_at", ""),
            actual_level=d.get("actual_level"),
            actual_depth=d.get("actual_depth"),
            review_notes=d.get("review_notes", ""),
        )


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get_attr(obj: Any, attr: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)
