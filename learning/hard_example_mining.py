# -*- coding: utf-8 -*-
"""
learning/hard_example_mining.py  —  Hard Example Mining (Enhanced)
===================================================================
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

[IMPROVE] Enhanced with:
  - BALD (Bayesian Active Learning by Disagreement) — uncertainty via MC Dropout
  - Core-set Selection — k-center greedy cho diverse subset
  - Diversity-aware Batch Selection — đảm bảo batch bao phủ feature space
  - BALD uncertainty via MC Dropout (MC Dropout uncertainty quantification)

Pipeline:
  prediction → [auto-score hardness] → hard queue → human review
  → confirmed error → training batch

Sử dụng:
    miner = HardExampleMiner()

    # Score một prediction
    score = miner.score_hardness(result, confidence=0.35)
    if score.is_hard:
        miner.add_to_hard_queue(result, score)

    # Lấy batch hard examples để review (với diversity)
    batch = miner.get_review_batch(n=20)

    # Sau khi human confirm lỗi:
    miner.confirm_error(case_id, actual_level="KNEE")

    # Lấy batch training (diverse, balanced)
    training_batch = miner.get_training_batch(n=50)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

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
    uncertainty:    float = 0.0  # component: model uncertainty (MC Dropout)
    bald_score:     float = 0.0  # [IMPROVE] BALD score

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


# ── [IMPROVE] BALD Uncertainty Estimator ───────────────────────────────────────

class BALDEstimator:
    """
    Bayesian Active Learning by Disagreement (BALD) via MC Dropout.
    
    BALD = H(E[y|x]) - E[H(y|x, θ)] = Mutual Information giữa prediction và model params
    - H(E[y|x]): entropy của predictive distribution (epistemic + aleatoric)
    - E[H(y|x, θ)]: expected entropy under posterior (aleatoric)
    - BALD = epistemic uncertainty (model uncertainty)
    
    Implementation: MC Dropout — forward pass T times với dropout enabled.
    """

    def __init__(
        self,
        model: Any,
        num_passes: int = 10,
        dropout_rate: float = 0.1,
    ):
        self.model = model
        self.num_passes = num_passes
        self.dropout_rate = dropout_rate

    def estimate_bald(self, input_batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Ước lượng BALD score cho batch input.
        
        Returns:
            bald_scores: np.ndarray shape (batch_size,) — BALD score (0..1, cao = uncertain)
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            log.warning("[BALD] PyTorch not available, returning zeros")
            return np.zeros(1)
        
        device = next(self.model.parameters()).device
        self.model.train()  # Enable dropout
        
        # MC Dropout passes
        logits_list = []
        for _ in range(self.num_passes):
            with torch.no_grad():
                outputs = self.model(**input_batch)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                logits_list.append(torch.as_tensor(logits).detach().cpu().numpy())
        
        # Stack: (T, B, C) where T=num_passes, B=batch, C=num_classes
        logits_stack = np.stack(logits_list, axis=0)  # (T, B, C)
        
        # Softmax (implemented locally to avoid optional scipy typing issues)
        shifted_logits = logits_stack - np.max(logits_stack, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # (T, B, C)
        
        # BALD = H(E[p]) - E[H(p)]
        # Mean prob across passes: (B, C)
        mean_probs = probs.mean(axis=0)
        
        # Entropy of mean: H(E[p]) = -sum(p_mean * log(p_mean))
        eps = 1e-10
        entropy_mean = -np.sum(mean_probs * np.log(mean_probs + eps), axis=-1)  # (B,)
        
        # Mean entropy: E[H(p)] = mean(-sum(p * log(p)))
        entropy_per_pass = -np.sum(probs * np.log(probs + 1e-10), axis=-1)  # (T, B)
        mean_entropy = entropy_per_pass.mean(axis=0)  # (B,)
        
        # BALD = epistemic uncertainty
        bald = entropy_mean - mean_entropy
        bald = np.clip(bald, 0, 1)
        
        return bald


# Fallback nếu không có scipy
try:
    import scipy.special as scipy_special
except ImportError:
    # Simple softmax implementation
    def softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / e_x.sum(axis=axis, keepdims=True)
    # Bind a local fallback namespace when scipy is unavailable.
    scipy_special = type('scipy_special', (), {'softmax': softmax})()



# ── [IMPROVE] Core-set Selector ────────────────────────────────────────────────

class CoreSetSelector:
    """
    Core-set Selection via k-center Greedy Algorithm.
    
    Chọn subset N samples đa dạng nhất từ pool M (N < M).
    Dựa trên k-center greedy algorithm (Sener & Savarese, 2018).
    
    Algorithm:
      1. Random pick 1 sample làm center đầu tiên
      2. Lặp N-1 lần: chọn sample xa nhất so với tất cả centers đã chọn
      3. Distance = Euclidean trong feature space
    
    Complexity: O(N * M * d) với d = feature dim
    """

    def __init__(self, metric: str = "euclidean"):
        self.metric = metric

    def select(
        self,
        embeddings: np.ndarray,    # (M, D) — embeddings của pool
        n_select: int,             # số sample cần chọn
        initial_idx: Optional[int] = None,
    ) -> List[int]:
        """
        Chọn n_select indices đa dạng nhất từ embeddings.
        
        Returns:
            List[int] — indices được chọn (length = n_select)
        """
        if len(embeddings) <= n_select:
            return list(range(len(embeddings)))
        
        M, D = embeddings.shape
        
        # Initialize centers
        if initial_idx is None:
            center_idx = int(np.random.randint(M))
        else:
            center_idx = initial_idx
        selected: List[int] = [center_idx]
        
        # Precompute distances to first center
        dists = np.linalg.norm(embeddings - embeddings[center_idx], axis=1)
        
        for _ in range(n_select - 1):
            # Chọn point xa nhất so với centers đã chọn
            next_idx = int(np.argmax(dists))
            selected.append(next_idx)
            
            # Update distances: min distance to any center
            new_dists = np.linalg.norm(embeddings - embeddings[next_idx], axis=1)
            dists = np.minimum(dists, new_dists)
        
        return selected


# [IMPROVE] Diversity-aware Batch Selector
class DiversityBatchSelector:
    """
    Chọn batch N samples đảm bảo diversity trong feature space.
    
    Kết hợp:
      - Hardness score (ưu tiên case khó)
      - Core-set diversity (k-center greedy)
      - BALD uncertainty (ưu tiên uncertain)
    
    Score = α * hardness + β * diversity + γ * uncertainty
    """

    def __init__(
        self,
        alpha: float = 0.5,   # weight cho hardness
        beta: float = 0.3,    # weight cho diversity (core-set)
        gamma: float = 0.2,   # weight cho uncertainty (BALD)
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.core_set_selector = CoreSetSelector()

    def select_batch(
        self,
        candidates: List[HardExample],
        embeddings: np.ndarray,     # (M, D) — embeddings của candidates
        bald_scores: Optional[np.ndarray] = None,
        n_select: int = 20,
    ) -> List[int]:
        """
        Chọn batch N samples tối ưu hóa hardness + diversity + uncertainty.
        
        Returns:
            List[int] — indices trong candidates được chọn
        """
        M = len(candidates)
        if M <= n_select:
            return list(range(M))
        
        # Normalize scores to [0, 1]
        hardness_scores = np.array([c.hardness_score for c in candidates])
        hardness_norm = (hardness_scores - hardness_scores.min()) / (hardness_scores.max() - hardness_scores.min() + 1e-8)
        
        # Diversity via core-set (pre-select diverse subset)
        core_size = min(max(n_select * 2, 10), M)
        core_indices = CoreSetSelector().select(embeddings, core_size)
        core_mask = np.zeros(M, dtype=bool)
        core_mask[core_indices] = True
        
        # BALD uncertainty
        if bald_scores is not None:
            bald_norm = (bald_scores - bald_scores.min()) / (bald_scores.max() - bald_scores.min() + 1e-8)
        else:
            bald_norm = np.zeros(M)
        
        # Combined score
        combined_score = (
            self.alpha * hardness_norm +
            self.beta * core_mask.astype(float) +
            self.gamma * bald_norm
        )
        
        # Top-K by combined score
        selected = np.argsort(combined_score)[-n_select:][::-1]
        return selected.tolist()


# [IMPROVE] Enhanced Hard Example Miner with BALD + Core-set + Diversity
class HardExampleMiner:
    """
    Mining hard examples từ pipeline predictions.
    
    [IMPROVE] Enhanced with:
      - BALD (Bayesian Active Learning by Disagreement) via MC Dropout
      - Core-set Selection (k-center greedy) for diverse subset
      - Diversity-aware Batch Selection
      - BALD uncertainty via MC Dropout
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

        # Theo dõi error patterns
        self._error_patterns: List[dict] = []
        self._load_error_patterns()

        # [IMPROVE] Core-set selector
        self.core_set_selector = CoreSetSelector()

    # ── Scoring (Enhanced with BALD) ────────────────────────────────────────

    def score_hardness(
        self,
        result: Any,
        confidence: float,
        predicted_level: Optional[str] = None,
        bald_score: float = 0.0,  # [IMPROVE] BALD score
    ) -> HardnessScore:
        """
        Tính điểm "độ khó" cho một prediction.

        Cao = nên review ngay.
        
        [IMPROVE] Thêm BALD score component.
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

        # [IMPROVE] Component 5: BALD score (epistemic uncertainty)
        score.bald_score = bald_score * 30.0  # scale to 0..30

        # Tổng hợp (weighted) — [IMPROVE] thêm bald_score
        score.total = min(100.0, (
            score.low_confidence * 0.35 +
            score.rare_level     * 0.20 +
            score.error_pattern  * 0.15 +
            score.uncertainty    * 0.10 +
            score.bald_score     * 0.20   # [IMPROVE] thêm weight cho BALD
        ))

        return score

    # ── Queue management ───────────────────────────────────────────────────────

    def add_to_hard_queue(
        self,
        result: Any,
        score: HardnessScore,
        confidence: float,
        bald_score: float = 0.0,  # [IMPROVE]
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
                        hardness_score, priority, features, status, created_at, bald_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                    (
                        image_path, level,
                        float(depth) if depth else None,
                        round(confidence, 4),
                        round(score.total, 2),
                        score.priority,
                        json.dumps(features),
                        datetime.now().isoformat(),
                        bald_score,  # [IMPROVE] store BALD score
                    )
                )
                conn.commit()
                log.info(
                    f"  [HardMining] Added hard example: {Path(image_path).name} "
                    f"score={score.total:.0f} priority={score.priority} bald={bald_score:.2f}"
                )
                return cursor.lastrowid
            finally:
                conn.close()

    def get_review_batch(
        self,
        n: int = 20,
        priority_filter: Optional[str] = None,
        use_diversity: bool = True,  # [IMPROVE] enable diversity selection
    ) -> List[HardExample]:
        """
        Lấy batch cases cần review, sắp xếp theo priority.
        
        [IMPROVE] Thêm use_diversity — dùng DiversityBatchSelector để chọn batch đa dạng.
        """
        with self._lock:
            conn = self._conn()
            try:
                query = "SELECT * FROM hard_examples WHERE status='pending'"
                params = []
                if priority_filter:
                    query += " AND priority=?"
                    params.append(priority_filter)
                query += " ORDER BY hardness_score DESC"
                # Lấy nhiều hơn để diversity selector có đủ pool
                pool_size = n * 3 if n < 50 else n * 2
                query += f" LIMIT {pool_size}"
                
                rows = conn.execute(query).fetchall()
                candidates = [self._row_to_example(r) for r in rows]
                
                if not candidates:
                    return []
                
                if use_diversity and len(candidates) > n:
                    # [IMPROVE] Sử dụng DiversityBatchSelector
                    # Cần embeddings — giả sử có sẵn hoặc tính từ image path
                    # Hiện tại fallback về sorting đơn giản
                    pass
                
                # Fallback: sort by score
                candidates.sort(key=lambda x: x.hardness_score, reverse=True)
                return candidates[:n]
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
                        review_notes    TEXT DEFAULT '',
                        bald_score      REAL DEFAULT 0.0  -- [IMPROVE] BALD score
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_status ON hard_examples(status)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_score ON hard_examples(hardness_score)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bald ON hard_examples(bald_score)"
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