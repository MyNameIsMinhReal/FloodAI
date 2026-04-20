# -*- coding: utf-8 -*-
"""
Active Learner — v3
====================
Nâng cấp so với v2:
  1. [BUG FIX] unusual_level: dùng uppercase key 'SUBMERGED','CHEST' (khớp ReferenceFloodResult)
  2. [BUG FIX] _has_conflicting_signals: dùng water_height_cm thay vì depth_cm (không tồn tại)
  3. [BUG FIX] _has_conflicting_signals: detected_person/pose_detected không có trong result
     → đọc từ detected_objects list
  4. Scoring số thực (0–100) thay vì priority 0/1/2 → sắp xếp queue chính xác hơn
  5. Level distribution tracking: ưu tiên sample các level ít có data hơn
  6. Auto-trigger thresholds.adjust() sau AUTO_ADJUST_EVERY review được submit
  7. get_underrepresented_levels(): biết level nào thiếu training data
"""

import logging
import hashlib
import random
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3
from datetime import datetime, timedelta
import json

log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
LOW_CONF_THRESHOLD  = 0.60
EDGE_CASE_THRESHOLD = 0.15
RANDOM_SAMPLE_RATE  = 0.05
QUEUE_EXPIRY_DAYS   = 30
DIVERSITY_WINDOW    = 10

# Tự động điều chỉnh thresholds sau N case được review
AUTO_ADJUST_EVERY   = 20

# Flood levels theo thứ tự (uppercase, khớp ReferenceFloodResult.flood_level)
ALL_LEVELS = ["NO_FLOOD", "PUDDLE", "ANKLE", "KNEE", "WAIST", "CHEST", "SUBMERGED"]
# Level hiếm gặp → ưu tiên sample hơn
RARE_LEVELS = {"CHEST", "SUBMERGED", "WAIST"}


@dataclass
class ReviewCase:
    id: Optional[int] = None
    timestamp: str = ""
    image_path: str = ""
    image_hash: str = ""

    predicted_depth: Optional[float] = None
    predicted_level: Optional[str] = None
    confidence: float = 0.0

    review_reason: str = ""
    priority: int = 0          # 0=low / 1=medium / 2=high (backward compat)
    score: float = 0.0         # [MỚI v3] 0–100, dùng để sort queue

    features: str = "{}"

    status: str = "pending"
    reviewed_by: str = ""
    actual_depth: Optional[float] = None
    actual_level: Optional[str] = None
    review_notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _image_hash(image_path: str) -> str:
    return hashlib.sha256(image_path.encode()).hexdigest()[:16]


class ActiveLearnerV2:
    """
    Active Learning v3 (class name giữ nguyên để không vỡ import).
    Deduplication, diversity sampling, level distribution tracking, auto-trigger.
    """

    def __init__(
        self,
        db_path: str = "learning/review_queue.db",
        low_conf_threshold: float = LOW_CONF_THRESHOLD,
        edge_case_threshold: float = EDGE_CASE_THRESHOLD,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

        self.low_conf_threshold  = low_conf_threshold
        self.edge_case_threshold = edge_case_threshold
        self._recent_reasons: List[str] = []

        # [MỚI v3] Đếm số review đã submit trong session này để auto-trigger
        self._reviews_since_adjust = 0

    # ── Schema ───────────────────────────────────────────────────────────────

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                image_path      TEXT NOT NULL,
                predicted_depth REAL,
                predicted_level TEXT,
                confidence      REAL,
                review_reason   TEXT,
                priority        INTEGER,
                features        TEXT,
                status          TEXT DEFAULT 'pending',
                reviewed_by     TEXT,
                actual_depth    REAL,
                actual_level    TEXT,
                review_notes    TEXT
            )
        """)
        self.conn.commit()

        # Schema migration: thêm image_hash và score nếu chưa có
        for col, col_type in [("image_hash", "TEXT"), ("score", "REAL DEFAULT 0")]:
            try:
                self.conn.execute(f"ALTER TABLE review_queue ADD COLUMN {col} {col_type}")
                self.conn.commit()
                log.info(f"[AL v3] Migrated DB: added column {col}")
            except sqlite3.OperationalError:
                pass

        self.conn.executescript("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_image_hash ON review_queue(image_hash);
            CREATE INDEX IF NOT EXISTS idx_status   ON review_queue(status);
            CREATE INDEX IF NOT EXISTS idx_score    ON review_queue(score DESC, status);
            CREATE INDEX IF NOT EXISTS idx_priority ON review_queue(priority, status);
        """)
        self.conn.commit()

    # ── Decision Logic ────────────────────────────────────────────────────────

    def needs_review(self, depth_result, features: dict) -> Tuple[bool, str, int]:
        """
        Quyết định case có cần review không.
        Returns: (needs_review, reason, priority)
        """
        confidence  = getattr(depth_result, "confidence", 1.0)
        flood_level = getattr(depth_result, "flood_level", "") or ""

        # 1. Low confidence
        if confidence < self.low_conf_threshold:
            priority = 2 if confidence < 0.40 else 1
            return True, "low_confidence", priority

        # 2. Edge case
        edge, edge_reason = self._is_edge_case(features)
        if edge:
            return True, f"edge_case:{edge_reason}", 2

        # 3. Conflicting signals
        if self._has_conflicting_signals(depth_result, features):
            return True, "conflicting_signals", 1

        # 4. [BUG FIX v3] Dùng uppercase key khớp với ReferenceFloodResult.flood_level
        if flood_level in RARE_LEVELS:
            return True, "rare_level", 1

        # 5. Underrepresented level → luôn sample để cân bằng distribution
        if self._is_underrepresented_level(flood_level):
            return True, f"underrepresented:{flood_level}", 1

        # 6. Diversity-aware random sampling
        recent_random = self._recent_reasons.count("random_sample")
        if recent_random < DIVERSITY_WINDOW * RANDOM_SAMPLE_RATE * 2:
            if random.random() < RANDOM_SAMPLE_RATE:
                return True, "random_sample", 0

        return False, "", 0

    def score_case(self, depth_result, features: dict, reason: str) -> float:
        """
        [MỚI v3] Tính điểm ưu tiên 0–100 cho case.
        Score cao → review trước.

        [Cập nhật v3.1] Tích hợp FloodLossFunction để ước lượng loss
        từ confidence và features khi chưa có ground truth.
        Dùng proxy: confidence thấp → depth loss dự kiến cao.
        """
        score = 0.0
        confidence = getattr(depth_result, "confidence", 1.0) or 1.0
        flood_level = getattr(depth_result, "flood_level", "") or ""
        depth_cm = getattr(depth_result, "water_height_cm", 0) or 0

        # [MỚI] Ước lượng expected loss từ confidence (proxy trước khi có GT)
        # Dùng confidence calibration loss: nếu conf thấp → model không chắc → score cao
        try:
            from learning.loss_function import get_loss_function
            fn = get_loss_function()
            # Proxy: assume "wrong" nếu conf < 0.5, "right" nếu conf >= 0.5
            proxy_correct = confidence >= 0.5
            proxy_conf_loss = fn.confidence_calibration_loss(confidence, proxy_correct)
            # Proxy depth loss: dùng water_level_pct mâu thuẫn làm signal
            water_level_pct = features.get("water_level_pct", 0.5)
            proxy_depth_err = abs(depth_cm - water_level_pct * 200)  # rough estimate
            proxy_depth_loss = fn.depth_huber_loss(depth_cm, max(0, depth_cm - proxy_depth_err * 0.5))
            # Proxy level loss: nếu rare level → khó dự đoán → loss cao hơn
            proxy_level_loss = 2/6 if flood_level in {"CHEST", "SUBMERGED", "WAIST"} else 0.5/6
            proxy_composite = fn.alpha * proxy_depth_loss + fn.beta * proxy_level_loss + fn.gamma * proxy_conf_loss
            score += fn.loss_to_priority_score(proxy_composite) * 0.5  # 50% weight từ loss proxy
        except Exception:
            pass  # Fallback về logic cũ nếu loss function lỗi

        # Confidence thấp → score cao (logic gốc giữ nguyên để ổn định)
        if confidence < 0.40:
            score += 40
        elif confidence < 0.60:
            score += 25
        elif confidence < 0.75:
            score += 10

        # Level hiếm → ưu tiên
        if flood_level in RARE_LEVELS:
            score += 20
        elif flood_level in {"KNEE", "WAIST"}:
            score += 10

        # Conflicting signals
        if "conflicting" in reason:
            score += 15

        # Edge case
        if "edge_case" in reason:
            score += 12

        # Độ ngập sâu cao → quan trọng hơn
        if depth_cm > 120:
            score += 10
        elif depth_cm > 70:
            score += 5

        # Không có reference object → khó tin
        if features.get("num_reference_objects", 1) == 0:
            score += 8

        return min(score, 100.0)

    def _is_edge_case(self, features: dict) -> Tuple[bool, str]:
        brightness = features.get("brightness", 128)

        if brightness < 30:          return True, "very_dark"
        if brightness > 240:         return True, "overexposed"
        if brightness < 70 and features.get("is_night", False):
            return True, "night_scene"

        if features.get("blur_score", 1000) < 50:
            return True, "very_blurry"

        if features.get("num_reference_objects", 0) == 0:
            return True, "no_reference"

        if features.get("num_people", 1) == 0:
            return True, "no_people"

        if features.get("watermark_count", 0) > 2:
            return True, "many_watermarks"

        ar = features.get("aspect_ratio", 1.0)
        if ar > 3.0 or ar < 0.3:
            return True, "unusual_aspect_ratio"

        return False, ""

    def _has_conflicting_signals(self, depth_result, features: dict) -> bool:
        """
        [BUG FIX v3] Dùng đúng field names từ ReferenceFloodResult:
          - water_height_cm (không phải depth_cm)
          - detected_objects list (không phải detected_person/pose_detected attr)
        """
        detected_objects = getattr(depth_result, "detected_objects", []) or []

        # [FIX] Đọc từ detected_objects thay vì attribute không tồn tại
        has_person = any(
            d.get("class_name") == "person" for d in detected_objects
        )
        has_pose = any(
            d.get("keypoints") is not None for d in detected_objects
            if d.get("class_name") == "person"
        )
        if has_person and not has_pose:
            return True

        # [FIX] water_height_cm thay vì depth_cm
        depth_cm    = getattr(depth_result, "water_height_cm", 0) or 0
        water_level = features.get("water_level_pct", 0)
        if depth_cm > 100 and water_level < 0.25:
            return True
        if depth_cm < 20 and water_level > 0.60:
            return True

        # Nhiều depth estimate khác nhau
        depth_estimates = features.get("depth_estimates", [])
        if len(depth_estimates) >= 2:
            mn, mx = min(depth_estimates), max(depth_estimates)
            if mx > 0 and (mx - mn) / mx > 0.40:
                return True

        return False

    def _is_underrepresented_level(self, flood_level: str) -> bool:
        """
        [MỚI v3] True nếu flood_level này có < 10% tổng reviewed cases.
        Giúp cân bằng training data.
        """
        if not flood_level:
            return False
        try:
            total_reviewed = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM review_queue WHERE status='reviewed'"
            ).fetchone()["cnt"]

            if total_reviewed < 20:   # Chưa đủ data để đánh giá
                return False

            level_count = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM review_queue "
                "WHERE status='reviewed' AND actual_level=?",
                (flood_level,)
            ).fetchone()["cnt"]

            return (level_count / total_reviewed) < 0.10
        except Exception:
            return False

    # ── Queue Management ──────────────────────────────────────────────────────

    def add_to_queue(self, case: ReviewCase) -> bool:
        """Add case vào queue. Returns False nếu đã có (dedup)."""
        case.timestamp  = datetime.now().isoformat()
        case.image_hash = _image_hash(case.image_path)

        self._recent_reasons.append(case.review_reason)
        if len(self._recent_reasons) > DIVERSITY_WINDOW * 3:
            self._recent_reasons = self._recent_reasons[-DIVERSITY_WINDOW:]

        try:
            self.conn.execute("""
                INSERT INTO review_queue (
                    timestamp, image_path, image_hash,
                    predicted_depth, predicted_level, confidence,
                    review_reason, priority, score,
                    features, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                case.timestamp, case.image_path, case.image_hash,
                case.predicted_depth, case.predicted_level, case.confidence,
                case.review_reason, case.priority, case.score,
                case.features, "pending",
            ))
            self.conn.commit()
            log.info(f"[AL v3] Queued: {Path(case.image_path).name} "
                     f"(reason={case.review_reason}, score={case.score:.0f})")
            return True
        except sqlite3.IntegrityError:
            log.debug(f"[AL v3] Duplicate skipped: {Path(case.image_path).name}")
            return False

    def expire_old_cases(self) -> int:
        cutoff = (datetime.now() - timedelta(days=QUEUE_EXPIRY_DAYS)).isoformat()
        cursor = self.conn.execute("""
            UPDATE review_queue SET status='expired'
            WHERE status='pending' AND timestamp < ?
        """, (cutoff,))
        self.conn.commit()
        n = cursor.rowcount
        if n > 0:
            log.info(f"[AL v3] Expired {n} old cases")
        return n

    def get_pending_reviews(self, max_count: int = 50,
                            priority_filter: Optional[int] = None) -> List[ReviewCase]:
        """Lấy pending cases, sort theo score (v3) → priority → timestamp."""
        query  = "SELECT * FROM review_queue WHERE status='pending'"
        params: list = []
        if priority_filter is not None:
            query += " AND priority >= ?"
            params.append(priority_filter)
        # [MỚI v3] Sort theo score DESC để case quan trọng lên đầu
        query += " ORDER BY score DESC, priority DESC, timestamp ASC LIMIT ?"
        params.append(max_count)

        cursor = self.conn.execute(query, params)
        return [self._row_to_case(r) for r in cursor.fetchall()]

    def submit_review(self, case_id: int, actual_depth: float, actual_level: str,
                      reviewed_by: str = "human", notes: str = ""):
        self.conn.execute("""
            UPDATE review_queue
            SET status='reviewed', reviewed_by=?, actual_depth=?, actual_level=?, review_notes=?
            WHERE id=?
        """, (reviewed_by, actual_depth, actual_level, notes, case_id))
        self.conn.commit()
        log.info(f"[AL v3] Review submitted for case {case_id}")

        case = self._get_case_by_id(case_id)
        if case and case.predicted_depth is not None:
            is_wrong_depth = abs(case.predicted_depth - actual_depth) > 10
            is_wrong_level = case.predicted_level != actual_level
            if is_wrong_depth or is_wrong_level:
                error_type = "wrong_level" if is_wrong_level else "wrong_depth"
                self._log_to_error_tracker(case, actual_depth, actual_level, error_type)

        # [MỚI v3] Auto-trigger threshold adjustment sau N reviews
        self._reviews_since_adjust += 1
        if self._reviews_since_adjust >= AUTO_ADJUST_EVERY:
            self._reviews_since_adjust = 0
            self._auto_adjust_thresholds()

    def get_review_stats(self) -> dict:
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status='pending'  THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status='reviewed' THEN 1 ELSE 0 END) as reviewed,
                SUM(CASE WHEN status='expired'  THEN 1 ELSE 0 END) as expired,
                SUM(CASE WHEN priority=2 THEN 1 ELSE 0 END) as high_priority,
                SUM(CASE WHEN priority=1 THEN 1 ELSE 0 END) as medium_priority,
                SUM(CASE WHEN priority=0 THEN 1 ELSE 0 END) as low_priority
            FROM review_queue
        """)
        row = cursor.fetchone()
        return {
            "total":           row["total"] or 0,
            "pending":         row["pending"] or 0,
            "reviewed":        row["reviewed"] or 0,
            "expired":         row["expired"] or 0,
            "high_priority":   row["high_priority"] or 0,
            "medium_priority": row["medium_priority"] or 0,
            "low_priority":    row["low_priority"] or 0,
        }

    def get_reason_distribution(self) -> dict:
        cursor = self.conn.execute("""
            SELECT review_reason, COUNT(*) as cnt
            FROM review_queue
            GROUP BY review_reason ORDER BY cnt DESC
        """)
        return {row["review_reason"]: row["cnt"] for row in cursor.fetchall()}

    def get_underrepresented_levels(self) -> List[str]:
        """[MỚI v3] Trả về các flood level thiếu training data."""
        try:
            cursor = self.conn.execute("""
                SELECT actual_level, COUNT(*) as cnt
                FROM review_queue WHERE status='reviewed' AND actual_level IS NOT NULL
                GROUP BY actual_level
            """)
            counts = {r["actual_level"]: r["cnt"] for r in cursor.fetchall()}
            total  = sum(counts.values())
            if total < 20:
                return []
            return [
                lvl for lvl in ALL_LEVELS
                if counts.get(lvl, 0) / total < 0.08
            ]
        except Exception:
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _row_to_case(self, row) -> ReviewCase:
        keys = row.keys()
        return ReviewCase(
            id=row["id"],
            timestamp=row["timestamp"],
            image_path=row["image_path"],
            image_hash=row["image_hash"] if "image_hash" in keys else "",
            predicted_depth=row["predicted_depth"],
            predicted_level=row["predicted_level"],
            confidence=row["confidence"],
            review_reason=row["review_reason"],
            priority=row["priority"],
            score=row["score"] if "score" in keys else 0.0,
            features=row["features"],
            status=row["status"],
            reviewed_by=row["reviewed_by"],
            actual_depth=row["actual_depth"],
            actual_level=row["actual_level"],
            review_notes=row["review_notes"],
        )

    def _get_case_by_id(self, case_id: int) -> Optional[ReviewCase]:
        cursor = self.conn.execute(
            "SELECT * FROM review_queue WHERE id=?", (case_id,)
        )
        row = cursor.fetchone()
        return self._row_to_case(row) if row else None

    def _log_to_error_tracker(self, case: ReviewCase, actual_depth: float,
                               actual_level: str, error_type: str):
        try:
            from learning.error_tracker import ErrorTracker, ErrorRecord
            error = ErrorRecord(
                image_path=case.image_path,
                error_type=error_type,
                predicted_depth=case.predicted_depth,
                predicted_level=case.predicted_level,
                predicted_confidence=case.confidence,
                actual_depth=actual_depth,
                actual_level=actual_level,
                features=case.features,
                reviewed_by=case.reviewed_by,
                review_notes=case.review_notes,
            )
            tracker = ErrorTracker()
            tracker.log_error(error)
            tracker.close()
        except Exception as e:
            log.warning(f"[AL v3] Failed to log to error tracker: {e}")

    def _auto_adjust_thresholds(self):
        """[MỚI v3] Tự động điều chỉnh thresholds sau đủ reviews."""
        try:
            from learning.adaptive_thresholds import AdaptiveThresholdsV2
            adj = AdaptiveThresholdsV2()
            adjusted = adj.adjust()
            if adjusted:
                log.info(f"[AL v3] Auto-adjust triggered: {adjusted}")
            else:
                log.info("[AL v3] Auto-adjust: no changes needed")
        except Exception as e:
            log.warning(f"[AL v3] Auto-adjust failed: {e}")

    def close(self):
        self.conn.close()
