# -*- coding: utf-8 -*-
"""
Error Tracking System — v2
===========================
Nâng cấp so với v1:
  1. Dùng sqlite3.Row (named access) thay vì positional index → không vỡ khi schema thay đổi
  2. get_error_stats() bổ sung depth_mae, depth_rmse, per-level breakdown
  3. get_accuracy_by_level() — biết flood level nào hay bị sai nhất
  4. get_reference_object_stats() — vật thể tham chiếu nào gây sai nhiều
  5. _analyze_feature_patterns() mở rộng: blur, no_people, no_reference, night
  6. [BUG FIX] suggest_threshold_adjustments: min_confidence set absolute thay vì delta → sửa thành delta
  7. processed_log table: track TỔNG ảnh đã xử lý để tính error rate thực
"""

import math
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import logging

log = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    """Một error record đầy đủ context."""
    id: Optional[int] = None
    timestamp: str = ""
    image_path: str = ""
    error_type: str = ""      # "false_positive"|"false_negative"|"wrong_depth"|"wrong_level"

    predicted_depth: Optional[float] = None
    predicted_level: Optional[str] = None
    predicted_confidence: Optional[float] = None

    actual_depth: Optional[float] = None
    actual_level: Optional[str] = None

    features: str = "{}"

    model_version: str = ""
    yolo_conf: float = 0.0
    depth_model: str = ""

    reviewed_by: str = ""
    review_notes: str = ""

    # [MỚI v3 — Loss Function] Được tính khi có actual_depth/actual_level
    loss_composite:  Optional[float] = None
    loss_depth:      Optional[float] = None
    loss_level:      Optional[float] = None
    loss_confidence: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ErrorTracker:
    """
    Hệ thống tracking lỗi để self-learning.
    Schema v2 bổ sung processed_log để tính error rate thực.
    """

    def __init__(self, db_path: str = "learning/errors.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row   # [FIX] named access
        self._init_tables()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS errors (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp            TEXT NOT NULL,
                image_path           TEXT NOT NULL,
                error_type           TEXT NOT NULL,
                predicted_depth      REAL,
                predicted_level      TEXT,
                predicted_confidence REAL,
                actual_depth         REAL,
                actual_level         TEXT,
                features             TEXT,
                model_version        TEXT,
                yolo_conf            REAL,
                depth_model          TEXT,
                reviewed_by          TEXT,
                review_notes         TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_timestamp   ON errors(timestamp);
            CREATE INDEX IF NOT EXISTS idx_error_type  ON errors(error_type);
            CREATE INDEX IF NOT EXISTS idx_pred_level  ON errors(predicted_level);

            CREATE TABLE IF NOT EXISTS processed_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                image_path      TEXT NOT NULL,
                predicted_level TEXT,
                predicted_depth REAL,
                confidence      REAL
            );

            CREATE INDEX IF NOT EXISTS idx_pl_ts ON processed_log(timestamp);
        """)
        self.conn.commit()

        # [MỚI v3] Schema migration: thêm cột loss nếu chưa có
        _loss_cols = [
            ("loss_composite",  "REAL"),
            ("loss_depth",      "REAL"),
            ("loss_level",      "REAL"),
            ("loss_confidence", "REAL"),
        ]
        for col, col_type in _loss_cols:
            try:
                self.conn.execute(
                    f"ALTER TABLE errors ADD COLUMN {col} {col_type}"
                )
                self.conn.commit()
                log.info(f"[ErrorTracker] Migrated DB: added column {col}")
            except sqlite3.OperationalError:
                pass  # column already exists

    # ── Write API ─────────────────────────────────────────────────────────────

    def log_error(self, record: ErrorRecord):
        """Lưu một error record, đồng thời tính loss nếu có ground truth."""
        record.timestamp = datetime.now().isoformat()

        # [MỚI v3] Tính loss nếu có actual_depth và actual_level
        if record.actual_depth is not None or record.actual_level is not None:
            try:
                from learning.loss_function import get_loss_function
                fn = get_loss_function()
                lr = fn.compute(
                    predicted_depth=record.predicted_depth,
                    actual_depth=record.actual_depth,
                    predicted_level=record.predicted_level,
                    actual_level=record.actual_level,
                    confidence=record.predicted_confidence,
                )
                record.loss_composite  = lr["composite_loss"]
                record.loss_depth      = lr["depth_loss"]
                record.loss_level      = lr["level_loss"]
                record.loss_confidence = lr["confidence_loss"]
                log.debug(
                    f"[ErrorTracker] Loss computed: composite={lr['composite_loss']:.4f} "
                    f"depth={lr['depth_loss']:.4f} level={lr['level_loss']:.4f} "
                    f"conf={lr['confidence_loss']:.4f}"
                )
            except Exception as _e:
                log.warning(f"[ErrorTracker] Loss computation failed: {_e}")

        self.conn.execute("""
            INSERT INTO errors (
                timestamp, image_path, error_type,
                predicted_depth, predicted_level, predicted_confidence,
                actual_depth, actual_level, features,
                model_version, yolo_conf, depth_model,
                reviewed_by, review_notes,
                loss_composite, loss_depth, loss_level, loss_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.timestamp, record.image_path, record.error_type,
            record.predicted_depth, record.predicted_level, record.predicted_confidence,
            record.actual_depth, record.actual_level, record.features,
            record.model_version, record.yolo_conf, record.depth_model,
            record.reviewed_by, record.review_notes,
            record.loss_composite, record.loss_depth,
            record.loss_level, record.loss_confidence,
        ))
        self.conn.commit()
        log.info(f"[ErrorTracker] {record.error_type}: {Path(record.image_path).name}")

    def log_prediction(self, image_path: str, predicted_level: str,
                       predicted_depth: float, confidence: float):
        """
        [MỚI v2] Ghi nhận mỗi ảnh được pipeline xử lý (kể cả đúng).
        Dùng để tính error rate = errors / total_processed.
        Gọi từ SelfLearningPipeline.process_results() cho mỗi ảnh.
        """
        self.conn.execute("""
            INSERT INTO processed_log (timestamp, image_path, predicted_level, predicted_depth, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), str(image_path),
              predicted_level, float(predicted_depth or 0), float(confidence or 0)))
        self.conn.commit()

    # ── Read API ──────────────────────────────────────────────────────────────

    def get_recent_errors(self, days: int = 30) -> List[ErrorRecord]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = self.conn.execute(
            "SELECT * FROM errors WHERE timestamp > ? ORDER BY timestamp DESC", (cutoff,)
        )
        return [self._row_to_record(r) for r in cursor.fetchall()]

    def get_error_stats(self, days: int = 30) -> Dict:
        """
        Phân tích error statistics.
        Bổ sung: depth_mae, depth_rmse, per_level breakdown.
        """
        errors = self.get_recent_errors(days)
        if not errors:
            return {
                "total": 0, "by_type": {}, "avg_confidence": 0.0,
                "depth_mae": 0.0, "depth_rmse": 0.0,
                "by_level": {}, "common_features": [],
            }

        by_type: Dict[str, int] = {}
        for err in errors:
            by_type[err.error_type] = by_type.get(err.error_type, 0) + 1

        confs = [e.predicted_confidence for e in errors if e.predicted_confidence is not None]
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        # Depth MAE + RMSE
        depth_diffs = [
            abs(e.actual_depth - e.predicted_depth)
            for e in errors
            if e.actual_depth is not None and e.predicted_depth is not None
        ]
        depth_mae  = sum(depth_diffs) / len(depth_diffs) if depth_diffs else 0.0
        depth_rmse = math.sqrt(
            sum(d ** 2 for d in depth_diffs) / len(depth_diffs)
        ) if depth_diffs else 0.0

        # Per-level breakdown
        by_level: Dict[str, Dict] = {}
        for err in errors:
            lvl = err.predicted_level or "unknown"
            if lvl not in by_level:
                by_level[lvl] = {"count": 0, "wrong_level": 0, "depth_errors": []}
            by_level[lvl]["count"] += 1
            if err.error_type == "wrong_level":
                by_level[lvl]["wrong_level"] += 1
            if err.actual_depth is not None and err.predicted_depth is not None:
                by_level[lvl]["depth_errors"].append(abs(err.actual_depth - err.predicted_depth))
        for lvl, data in by_level.items():
            errs = data.pop("depth_errors")
            data["depth_mae"] = round(sum(errs) / len(errs), 1) if errs else 0.0

        features_list = []
        for err in errors:
            try:
                features_list.append(json.loads(err.features))
            except Exception:
                pass

        return {
            "total":           len(errors),
            "by_type":         by_type,
            "avg_confidence":  round(avg_conf, 3),
            "depth_mae":       round(depth_mae, 1),
            "depth_rmse":      round(depth_rmse, 1),
            "by_level":        by_level,
            "common_features": self._analyze_feature_patterns(features_list),
        }

    def get_accuracy_by_level(self, days: int = 90) -> Dict[str, Dict]:
        """[MỚI v2] Tỉ lệ đúng/sai theo flood level, dùng processed_log."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        total_cur = self.conn.execute("""
            SELECT predicted_level, COUNT(*) as cnt
            FROM processed_log WHERE timestamp > ?
            GROUP BY predicted_level
        """, (cutoff,))
        total_by = {r["predicted_level"]: r["cnt"] for r in total_cur.fetchall()}

        err_cur = self.conn.execute("""
            SELECT predicted_level, COUNT(*) as cnt
            FROM errors WHERE timestamp > ?
            GROUP BY predicted_level
        """, (cutoff,))
        err_by = {r["predicted_level"]: r["cnt"] for r in err_cur.fetchall()}

        result = {}
        for lvl, total in total_by.items():
            wrong = err_by.get(lvl, 0)
            result[lvl] = {
                "total":      total,
                "errors":     wrong,
                "error_rate": round(wrong / total, 3) if total > 0 else 0.0,
                "accuracy":   round(1 - wrong / total, 3) if total > 0 else 0.0,
            }
        return result

    def get_reference_object_stats(self, days: int = 90) -> Dict[str, Dict]:
        """[MỚI v2] Vật thể tham chiếu nào hay gây sai."""
        errors = self.get_recent_errors(days)
        obj_stats: Dict[str, Dict] = {}
        for err in errors:
            try:
                features = json.loads(err.features)
                ref_objs = features.get("reference_objects", [])
                if isinstance(ref_objs, str):
                    ref_objs = [ref_objs]
                for obj in ref_objs:
                    if obj not in obj_stats:
                        obj_stats[obj] = {"count": 0, "depth_errors": []}
                    obj_stats[obj]["count"] += 1
                    if err.actual_depth is not None and err.predicted_depth is not None:
                        obj_stats[obj]["depth_errors"].append(
                            abs(err.actual_depth - err.predicted_depth)
                        )
            except Exception:
                pass
        for obj, data in obj_stats.items():
            errs = data.pop("depth_errors")
            data["avg_depth_error"] = round(sum(errs) / len(errs), 1) if errs else 0.0
        return dict(sorted(obj_stats.items(), key=lambda x: -x[1]["count"]))

    def get_loss_stats(self, days: int = 30) -> Dict:
        """
        [MỚI v3] Tính thống kê Loss Function từ các error records đã có ground truth.

        Chỉ tính trên records có đầy đủ actual_depth và actual_level.
        Dùng FloodLossFunction.batch_loss() để tính aggregate.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = self.conn.execute("""
            SELECT predicted_depth, actual_depth,
                   predicted_level, actual_level,
                   predicted_confidence, image_path,
                   loss_composite, loss_depth, loss_level, loss_confidence
            FROM errors
            WHERE timestamp > ?
              AND actual_depth IS NOT NULL
              AND actual_level IS NOT NULL
        """, (cutoff,))
        rows = cursor.fetchall()

        if not rows:
            return {
                "n": 0,
                "avg_composite":  0.0,
                "avg_depth_loss": 0.0,
                "avg_level_loss": 0.0,
                "avg_conf_loss":  0.0,
                "depth_mae_cm":   0.0,
                "level_accuracy": 0.0,
                "worst_cases":    [],
                "recommendations": {},
            }

        # Ưu tiên dùng giá trị đã lưu trong DB; tính lại nếu thiếu
        records_for_batch = []
        precomputed = []
        for r in rows:
            if r["loss_composite"] is not None:
                precomputed.append({
                    "composite_loss":  r["loss_composite"],
                    "depth_loss":      r["loss_depth"]      or 0.0,
                    "level_loss":      r["loss_level"]      or 0.0,
                    "confidence_loss": r["loss_confidence"] or 0.0,
                    "depth_error_cm":  abs((r["predicted_depth"] or 0) - (r["actual_depth"] or 0)),
                    "level_correct":   (r["predicted_level"] or "") == (r["actual_level"] or ""),
                    "_path": r["image_path"],
                })
            else:
                records_for_batch.append({
                    "predicted_depth": r["predicted_depth"],
                    "actual_depth":    r["actual_depth"],
                    "predicted_level": r["predicted_level"],
                    "actual_level":    r["actual_level"],
                    "confidence":      r["predicted_confidence"] or 0.5,
                    "image_path":      r["image_path"],
                })

        try:
            from learning.loss_function import get_loss_function
            fn = get_loss_function()
            if records_for_batch:
                batch_new = fn.batch_loss(records_for_batch)
            else:
                batch_new = fn.batch_loss([])  # empty
        except Exception as _e:
            log.warning(f"[ErrorTracker] get_loss_stats batch_loss failed: {_e}")
            batch_new = {"n": 0, "avg_composite": 0.0, "avg_depth_loss": 0.0,
                         "avg_level_loss": 0.0, "avg_conf_loss": 0.0,
                         "depth_mae_cm": 0.0, "level_accuracy": 0.0, "worst_cases": []}

        # Merge precomputed + newly computed
        all_results = precomputed + [
            {
                "composite_loss":  r["composite_loss"],
                "depth_loss":      r["depth_loss"],
                "level_loss":      r["level_loss"],
                "confidence_loss": r["confidence_loss"],
                "depth_error_cm":  r.get("depth_error_cm", 0),
                "level_correct":   r.get("level_correct", False),
                "_path":           r.get("_path", ""),
            }
            for r in (batch_new.get("worst_cases") or [])  # already aggregated above
        ]

        n_total = len(rows)
        if n_total == 0:
            stats = batch_new
        else:
            # Recalculate aggregate from precomputed list
            all_composite = [r["composite_loss"]  for r in precomputed]
            all_depth     = [r["depth_loss"]      for r in precomputed]
            all_level     = [r["level_loss"]      for r in precomputed]
            all_conf      = [r["confidence_loss"] for r in precomputed]
            all_depth_err = [r["depth_error_cm"]  for r in precomputed]
            all_correct   = [r["level_correct"]   for r in precomputed]

            # Add newly computed batch contributions
            n_new = batch_new.get("n", 0)
            if n_new > 0:
                all_composite.extend([batch_new["avg_composite"]]  * n_new)
                all_depth.extend(    [batch_new["avg_depth_loss"]] * n_new)
                all_level.extend(    [batch_new["avg_level_loss"]] * n_new)
                all_conf.extend(     [batch_new["avg_conf_loss"]]  * n_new)
                all_depth_err.extend([batch_new["depth_mae_cm"]]   * n_new)
                n_correct = round(batch_new.get("level_accuracy", 0) * n_new)
                all_correct.extend([True] * n_correct + [False] * (n_new - n_correct))

            n = len(all_composite)
            worst = sorted(precomputed, key=lambda x: -x["composite_loss"])[:5]
            worst_cases = [
                {"path": w["_path"], "loss": w["composite_loss"],
                 "depth_err_cm": w["depth_error_cm"], "level_dist": -1}
                for w in worst
            ]
            stats = {
                "n":              n,
                "avg_composite":  round(sum(all_composite) / n, 4) if n else 0.0,
                "avg_depth_loss": round(sum(all_depth) / n,     4) if n else 0.0,
                "avg_level_loss": round(sum(all_level) / n,     4) if n else 0.0,
                "avg_conf_loss":  round(sum(all_conf)  / n,     4) if n else 0.0,
                "depth_mae_cm":   round(sum(all_depth_err) / n, 1) if n else 0.0,
                "level_accuracy": round(sum(all_correct) / n,   3) if n else 0.0,
                "worst_cases":    worst_cases,
            }

        # Thêm khuyến nghị
        try:
            from learning.loss_function import get_loss_function
            recs = get_loss_function().get_threshold_recommendations(stats)
            stats["recommendations"] = recs
        except Exception:
            stats["recommendations"] = {}

        return stats

    def get_problematic_scenarios(self) -> List[str]:
        errors = self.get_recent_errors(days=90)
        combos: Dict[str, int] = {}
        for err in errors:
            try:
                f = json.loads(err.features)
                sc = []
                if f.get("brightness", 255) < 80: sc.append("dark")
                if f.get("has_watermark"):         sc.append("watermark")
                if f.get("blur_score", 1000) < 100: sc.append("blurry")
                if f.get("num_people", 1) == 0:    sc.append("no_people")
                if f.get("num_reference_objects", 1) == 0: sc.append("no_reference")
                if f.get("is_night", False):        sc.append("night")
                key = "_".join(sc) if sc else "normal"
                combos[key] = combos.get(key, 0) + 1
            except Exception:
                pass
        return [f"{s} ({c} errors)"
                for s, c in sorted(combos.items(), key=lambda x: -x[1])[:5]]

    def suggest_threshold_adjustments(self) -> Dict[str, float]:
        """
        Đề xuất delta điều chỉnh thresholds.
        [BUG FIX] Tất cả giá trị là DELTA (±), không phải absolute.
        """
        stats = self.get_error_stats(days=30)
        adj: Dict[str, float] = {}
        total = max(stats["total"], 1)

        fp = stats["by_type"].get("false_positive", 0)
        fn = stats["by_type"].get("false_negative", 0)

        if fp / total > 0.30:
            adj["yolo_conf"] = +0.05
        elif fn / total > 0.30:
            adj["yolo_conf"] = -0.05

        # [BUG FIX] delta thay vì absolute
        if stats["avg_confidence"] < 0.50 and stats["total"] >= 10:
            adj["min_confidence"] = +0.05

        for pattern in stats.get("common_features", []):
            p, freq = pattern["pattern"], pattern["frequency"]
            if p == "low_brightness" and freq > 0.40:
                adj["enhance_threshold"] = -10.0
            if p == "has_watermark" and freq > 0.30:
                adj["watermark_conf"] = -0.05
            if p == "very_blurry" and freq > 0.30:
                adj["blur_threshold"] = -10.0

        return adj

    def get_total_processed(self, days: int = 30) -> int:
        """[MỚI v2] Tổng ảnh đã xử lý trong N ngày từ processed_log."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM processed_log WHERE timestamp > ?", (cutoff,)
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0

    def export_for_finetuning(self, output_dir: Path) -> Tuple[int, int]:
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(exist_ok=True, parents=True)
        labels_dir.mkdir(exist_ok=True, parents=True)

        cursor = self.conn.execute("""
            SELECT * FROM errors
            WHERE actual_depth IS NOT NULL OR actual_level IS NOT NULL
        """)
        import shutil
        count = 0
        for row in cursor.fetchall():
            rec = self._row_to_record(row)
            src = Path(rec.image_path)
            if src.exists():
                shutil.copy2(src, images_dir / src.name)
                (labels_dir / f"{src.stem}.json").write_text(json.dumps({
                    "depth_cm":    rec.actual_depth,
                    "flood_level": rec.actual_level,
                    "features":    rec.features,
                }, indent=2, ensure_ascii=False))
                count += 1
        log.info(f"[ErrorTracker] Exported {count} validated samples")
        return count, count

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _row_to_record(self, row) -> ErrorRecord:
        """[FIX] Named access — an toàn khi schema thay đổi."""
        keys = row.keys() if hasattr(row, "keys") else []
        return ErrorRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            image_path=row["image_path"],
            error_type=row["error_type"],
            predicted_depth=row["predicted_depth"],
            predicted_level=row["predicted_level"],
            predicted_confidence=row["predicted_confidence"],
            actual_depth=row["actual_depth"],
            actual_level=row["actual_level"],
            features=row["features"] or "{}",
            model_version=row["model_version"] or "",
            yolo_conf=row["yolo_conf"] or 0.0,
            depth_model=row["depth_model"] or "",
            reviewed_by=row["reviewed_by"] or "",
            review_notes=row["review_notes"] or "",
            # [MỚI v3] Loss fields — None nếu chưa có (records cũ)
            loss_composite  = row["loss_composite"]  if "loss_composite"  in keys else None,
            loss_depth      = row["loss_depth"]      if "loss_depth"      in keys else None,
            loss_level      = row["loss_level"]      if "loss_level"      in keys else None,
            loss_confidence = row["loss_confidence"] if "loss_confidence" in keys else None,
        )

    def _analyze_feature_patterns(self, features_list: List[dict]) -> List[dict]:
        n = len(features_list)
        if n == 0:
            return []
        patterns = []

        bvals = [f["brightness"] for f in features_list if "brightness" in f]
        if bvals and (sum(bvals) / len(bvals)) < 80:
            patterns.append({"pattern": "low_brightness",
                             "avg_value": round(sum(bvals)/len(bvals), 1),
                             "frequency": round(len(bvals)/n, 2)})

        very_blurry = [f for f in features_list if f.get("blur_score", 1000) < 80]
        if len(very_blurry) / n > 0.25:
            patterns.append({"pattern": "very_blurry",
                             "frequency": round(len(very_blurry)/n, 2)})

        wm = sum(1 for f in features_list if f.get("has_watermark"))
        if wm / n > 0.30:
            patterns.append({"pattern": "has_watermark",
                             "frequency": round(wm/n, 2)})

        no_people = sum(1 for f in features_list if f.get("num_people", 1) == 0)
        if no_people / n > 0.30:
            patterns.append({"pattern": "no_people",
                             "frequency": round(no_people/n, 2)})

        no_ref = sum(1 for f in features_list if f.get("num_reference_objects", 1) == 0)
        if no_ref / n > 0.20:
            patterns.append({"pattern": "no_reference",
                             "frequency": round(no_ref/n, 2)})

        night = sum(1 for f in features_list if f.get("is_night", False))
        if night / n > 0.15:
            patterns.append({"pattern": "night_scene",
                             "frequency": round(night/n, 2)})

        return sorted(patterns, key=lambda x: -x["frequency"])

    # [IMPROVE] Causal Error Analysis & Feature Correlation
    def get_causal_analysis(self, days: int = 30) -> Dict:
        """
        Phân tích nhân quả: tìm feature nào GÂY RA lỗi (không chỉ tương quan).
        
        Dùng:
        - Conditional probability P(error | feature) vs P(error)
        - Stratified analysis: error rate theo từng feature bucket
        """
        since = datetime.now() - timedelta(days=days)
        
        cursor = self.conn.execute("""
            SELECT features, error_type, predicted_depth, actual_depth, predicted_level, actual_level
            FROM errors
            WHERE timestamp > ?
        """, (since.isoformat(),))
        
        rows = cursor.fetchall()
        if len(rows) < 20:
            return {"error": "insufficient_data", "n": len(rows)}
        
        features_list = []
        error_types = []
        depth_errors = []
        level_errors = []
        
        for r in rows:
            try:
                feat = json.loads(r["features"] or "{}")
                features_list.append(feat)
                error_types.append(r["error_type"])
                if r["predicted_depth"] is not None and r["actual_depth"] is not None:
                    depth_errors.append(abs(r["predicted_depth"] - r["actual_depth"]))
                else:
                    depth_errors.append(None)
                if r["predicted_level"] is not None and r["actual_level"] is not None:
                    level_errors.append(r["predicted_level"] != r["actual_level"])
                else:
                    level_errors.append(None)
            except Exception:
                continue
        
        all_keys = set()
        for f in features_list:
            all_keys.update(f.keys())
        
        causal_results = {}
        base_error_rate = len([e for e in error_types if e]) / max(len(error_types), 1)
        
        for key in all_keys:
            vals = [f.get(key) for f in features_list if f.get(key) is not None]
            if not vals:
                continue
            
            try:
                numeric_vals = [float(v) for v in vals if isinstance(v, (int, float))]
                if len(numeric_vals) < 10:
                    continue
                
                q25, q50, q75 = np.percentile(numeric_vals, [25, 50, 75])
                bins = []
                for v in vals:
                    fv = float(v)
                    if fv <= q25: bins.append("Q1")
                    elif fv <= q50: bins.append("Q2")
                    elif fv <= q75: bins.append("Q3")
                    else: bins.append("Q4")
            except:
                bins = [str(v) for v in vals]
            
            bin_errors = {}
            bin_counts = {}
            for b, err_type, d_err, l_err in zip(bins, error_types, depth_errors, level_errors):
                if b not in bin_counts:
                    bin_counts[b] = 0
                    bin_errors[b] = {"count": 0, "depth_mae": [], "level_wrong": 0}
                bin_counts[b] += 1
                if err_type:
                    bin_errors[b]["count"] += 1
                if d_err is not None:
                    bin_errors[b]["depth_mae"].append(d_err)
                if l_err is not None and l_err:
                    bin_errors[b]["level_wrong"] += 1
            
            causal_score = 0.0
            for b in bin_counts:
                if bin_counts[b] < 3:
                    continue
                err_rate = bin_errors[b]["count"] / bin_counts[b]
                if err_rate > base_error_rate * 2.0:
                    causal_score += (err_rate - base_error_rate) * (bin_counts[b] / len(features_list))
            
            if causal_score > 0.05:
                causal_results[key] = {
                    "causal_score": round(causal_score, 4),
                    "base_error_rate": round(base_error_rate, 4),
                    "bins": {
                        b: {
                            "count": bin_counts[b],
                            "error_rate": round(bin_errors[b]["count"] / bin_counts[b], 4),
                            "depth_mae": round(np.mean(bin_errors[b]["depth_mae"]), 2) if bin_errors[b]["depth_mae"] else None,
                            "level_error_rate": round(bin_errors[b]["level_wrong"] / bin_counts[b], 4) if bin_counts[b] > 0 else None,
                        }
                        for b in bin_counts
                    }
                }
        
        return {
            "n_samples": len(features_list),
            "base_error_rate": round(base_error_rate, 4),
            "causal_features": causal_results,
        }

    def get_feature_correlation_matrix(self, days: int = 30) -> Dict:
        """Tính ma trận tương quan giữa các features và lỗi."""
        since = datetime.now() - timedelta(days=days)
        cursor = self.conn.execute("""
            SELECT features FROM errors WHERE timestamp > ?
        """, (since.isoformat(),))
        
        features_list = []
        for r in cursor.fetchall():
            try:
                features_list.append(json.loads(r["features"] or "{}"))
            except:
                continue
        
        if len(features_list) < 10:
            return {"error": "insufficient_data"}
        
        numeric_features = {}
        for f in features_list:
            for k, v in f.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if k not in numeric_features:
                        numeric_features[k] = []
                    numeric_features[k].append(v)
        
        valid_features = {k: v for k, v in numeric_features.items() if len(v) >= 10}
        if len(valid_features) < 2:
            return {"error": "not_enough_numeric_features"}
        
        feature_names = list(valid_features.keys())
        n = len(feature_names)
        matrix = np.zeros((n, n))
        
        for i, f1 in enumerate(feature_names):
            for j, f2 in enumerate(feature_names):
                if i <= j:
                    min_len = min(len(valid_features[f1]), len(valid_features[f2]))
                    v1 = np.array(valid_features[f1][:min_len])
                    v2 = np.array(valid_features[f2][:min_len])
                    if np.std(v1) > 0 and np.std(v2) > 0:
                        corr = np.corrcoef(v1, v2)[0, 1]
                        matrix[i, j] = matrix[j, i] = round(corr, 3)
        
        return {
            "features": feature_names,
            "correlation_matrix": matrix.tolist(),
        }

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    tracker = ErrorTracker()
    stats = tracker.get_error_stats(days=30)
    print(f"Total: {stats['total']} | MAE: {stats['depth_mae']} cm | RMSE: {stats['depth_rmse']} cm")
    for k, v in tracker.suggest_threshold_adjustments().items():
        print(f"  Suggest {k}: {v:+.3f}")
    tracker.close()
