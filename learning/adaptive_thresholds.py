# -*- coding: utf-8 -*-
"""
Adaptive Thresholds — v3
=========================
Nâng cấp so với v2:
  1. [BUG FIX] _estimate_total_processed: trước đây dùng errors*10 (sai)
     → nay dùng ErrorTracker.get_total_processed() — số thực từ processed_log
  2. Per-level accuracy tracking: biết threshold nào ảnh hưởng level nào
  3. Trend analysis cải tiến: dùng processed_log để tính error rate chính xác
  4. get_level_recommendations(): đề xuất riêng cho từng flood level hay sai
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# ─── Bounds ──────────────────────────────────────────────────────────────────
THRESHOLD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "yolo_conf":         (0.20, 0.70),
    "min_confidence":    (0.30, 0.85),
    "blur_threshold":    (30.0, 300.0),
    "watermark_conf":    (0.25, 0.75),
    "enhance_threshold": (40.0, 150.0),
}

EMA_ALPHA           = 0.30
MIN_SAMPLES         = 15       # samples tối thiểu để tin điều chỉnh
COOLDOWN_DAYS       = 7


class AdaptiveThresholdsV2:
    """
    Tự động học và điều chỉnh thresholds.
    Backward-compatible với file JSON v1/v2.
    """

    def __init__(self, config_path: str = "learning/adaptive_thresholds.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True, parents=True)

        self.thresholds: Dict[str, float] = {
            "yolo_conf":         0.35,
            "min_confidence":    0.50,
            "blur_threshold":    100.0,
            "watermark_conf":    0.45,
            "enhance_threshold": 80.0,
        }
        self._meta: Dict = {
            "last_updated":   datetime.now().isoformat(),
            "version":        "3.0",
            "adjustment_log": [],
            "last_adjusted":  {},
            "accuracy_trend": [],
        }
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if not self.config_path.exists():
            return
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            meta_keys = {"last_updated", "version", "adjustment_log",
                         "last_adjusted", "accuracy_trend"}
            for k, v in data.items():
                if k not in meta_keys and k in self.thresholds:
                    self.thresholds[k] = v
            for k in ("adjustment_log", "last_adjusted", "accuracy_trend"):
                if k in data:
                    self._meta[k] = data[k]
            log.info(f"[AdaptiveThresholds v3] Loaded from {self.config_path}")
        except Exception as e:
            log.warning(f"Failed to load thresholds: {e}")

    def _save(self):
        self._meta["last_updated"] = datetime.now().isoformat()
        self._meta["version"] = "3.0"
        payload = {**self.thresholds, **self._meta}
        self.config_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        log.info(f"[AdaptiveThresholds v3] Saved to {self.config_path}")

    # ── Core API ──────────────────────────────────────────────────────────────

    def get(self, key: str, default=None):
        return self.thresholds.get(key, default)

    def set(self, key: str, value: float, reason: str = "manual"):
        old = self.thresholds.get(key)
        clamped = self._clamp(key, value)
        self.thresholds[key] = clamped
        self._log_adjustment(key, old, clamped, reason)
        self._save()

    def get_all(self) -> Dict[str, float]:
        return self.thresholds.copy()

    def reset(self):
        self.thresholds = {
            "yolo_conf": 0.35, "min_confidence": 0.50,
            "blur_threshold": 100.0, "watermark_conf": 0.45,
            "enhance_threshold": 80.0,
        }
        self._meta["adjustment_log"] = []
        self._meta["last_adjusted"]  = {}
        self._save()

    # ── Main Learning Logic ───────────────────────────────────────────────────

    def adjust(self, error_tracker_db: str = "learning/errors.db") -> List[str]:
        """
        Phân tích errors và adjust thresholds với EMA + cooldown.
        [BUG FIX v3] Dùng get_total_processed() thay vì errors*10 để tính error rate.
        [MỚI v3.1] Tích hợp FloodLossFunction để quyết định mức độ điều chỉnh.
        """
        from learning.error_tracker import ErrorTracker

        tracker  = ErrorTracker(db_path=error_tracker_db)
        stats    = tracker.get_error_stats(days=30)
        total_errors = stats.get("total", 0)

        # [BUG FIX] Tính error rate thực dựa trên số ảnh đã xử lý thực sự
        total_processed = tracker.get_total_processed(days=30)
        if total_processed == 0:
            # Fallback nếu processed_log chưa có data (pipeline cũ)
            total_processed = max(total_errors * 5, 1)

        error_rate = total_errors / total_processed

        # [MỚI v3.1] Lấy loss stats để điều chỉnh mức độ EMA
        loss_stats = tracker.get_loss_stats(days=30)
        avg_composite_loss = loss_stats.get("avg_composite", 0.0)
        n_with_gt = loss_stats.get("n", 0)

        # Ghi vào trend — bổ sung avg_loss
        self._meta["accuracy_trend"].append({
            "timestamp":        datetime.now().isoformat(),
            "error_rate":       round(error_rate, 4),
            "total_errors":     total_errors,
            "total_processed":  total_processed,
            "avg_loss":         round(avg_composite_loss, 4),
        })
        self._meta["accuracy_trend"] = self._meta["accuracy_trend"][-52:]

        if total_errors < MIN_SAMPLES:
            log.info(f"[AdaptiveThresholds v3] Only {total_errors} errors, "
                     f"need {MIN_SAMPLES}+ to adjust")
            tracker.close()
            return []

        suggestions = tracker.suggest_threshold_adjustments()
        tracker.close()

        # [MỚI v3.1] Điều chỉnh EMA_ALPHA dựa trên composite loss
        # Loss cao → cần học nhanh hơn (alpha lớn hơn)
        # Loss thấp → học chậm, ổn định (alpha nhỏ hơn)
        if n_with_gt >= 5:
            if avg_composite_loss > 0.45:
                ema = min(EMA_ALPHA * 1.5, 0.60)   # học nhanh hơn
            elif avg_composite_loss < 0.10:
                ema = max(EMA_ALPHA * 0.5, 0.10)   # học chậm, ổn định
            else:
                ema = EMA_ALPHA  # bình thường
        else:
            ema = EMA_ALPHA  # không đủ data loss → dùng mặc định

        adjusted = []
        for key, delta in suggestions.items():
            if key not in self.thresholds:
                continue
            if self._in_cooldown(key):
                log.info(f"[AdaptiveThresholds v3] {key} in cooldown, skipping")
                continue

            old_val = self.thresholds[key]
            target  = old_val + delta
            new_val = self._clamp(key, ema * target + (1 - ema) * old_val)

            rel_change = abs(new_val - old_val) / (abs(old_val) + 1e-9)
            if rel_change < 0.005:
                continue

            self.thresholds[key] = new_val
            self._meta["last_adjusted"][key] = datetime.now().isoformat()
            reason = (
                f"auto (delta={delta:+.3f}, EMA={ema:.2f}, "
                f"{total_errors} errors/{total_processed} processed, "
                f"avg_loss={avg_composite_loss:.3f})"
            )
            self._log_adjustment(key, old_val, new_val, reason)
            adjusted.append(
                f"{key}: {old_val:.3f} → {new_val:.3f} "
                f"(delta={delta:+.3f}, loss={avg_composite_loss:.3f})"
            )
            log.info(f"[AdaptiveThresholds v3] Adjusted {key}: {old_val:.3f} → {new_val:.3f}")

        if adjusted:
            self._save()

        return adjusted

    # ── Trend & Analysis ─────────────────────────────────────────────────────

    def get_trend(self) -> Dict:
        trend = self._meta.get("accuracy_trend", [])
        if len(trend) < 2:
            return {"direction": "unknown", "weeks_of_data": len(trend),
                    "current_error_rate": 0.0}

        current       = trend[-1]["error_rate"]
        four_wks_ago  = trend[-4]["error_rate"] if len(trend) >= 4 else trend[0]["error_rate"]
        delta         = current - four_wks_ago
        direction     = ("improving" if delta < -0.005
                         else "worsening" if delta > 0.005 else "stable")

        return {
            "direction":           direction,
            "weeks_of_data":       len(trend),
            "current_error_rate":  round(current, 4),
            "four_week_delta":     round(delta, 4),
            "history":             trend[-8:],
        }

    def get_level_recommendations(self, error_tracker_db: str = "learning/errors.db") -> Dict:
        """
        [MỚI v3] Phân tích accuracy theo từng flood level,
        trả về đề xuất thu thập thêm data hay điều chỉnh.
        """
        from learning.error_tracker import ErrorTracker
        tracker = ErrorTracker(db_path=error_tracker_db)
        level_acc = tracker.get_accuracy_by_level(days=90)
        tracker.close()

        recs = {}
        for lvl, data in level_acc.items():
            if data["error_rate"] > 0.30:
                recs[lvl] = {
                    "status":     "poor",
                    "accuracy":   data["accuracy"],
                    "suggestion": "Thu thập thêm training data cho level này",
                }
            elif data["total"] < 10:
                recs[lvl] = {
                    "status":     "insufficient_data",
                    "accuracy":   data["accuracy"],
                    "suggestion": "Cần thêm review cases cho level này",
                }
            else:
                recs[lvl] = {
                    "status":   "ok",
                    "accuracy": data["accuracy"],
                }
        return recs

    def get_adjustment_history(self, last_n: int = 20) -> List[Dict]:
        return self._meta.get("adjustment_log", [])[-last_n:]

    def export_for_config(self, config_yaml_path: str = "config.yaml") -> Optional[Dict]:
        try:
            import yaml, shutil
            with open(config_yaml_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            mapping = {
                "filter":    {"blur_threshold":   "blur_threshold"},
                "watermark": {"confidence_thresh": "watermark_conf"},
                "depth":     {"yolo_conf":         "yolo_conf"},
                "enhance":   {"dark_threshold":    "enhance_threshold"},
            }
            for section, keys in mapping.items():
                if section in config:
                    for cfg_key, thr_key in keys.items():
                        if thr_key in self.thresholds:
                            config[section][cfg_key] = self.thresholds[thr_key]

            shutil.copy2(config_yaml_path, Path(config_yaml_path).with_suffix(".yaml.backup"))
            with open(config_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            log.info(f"Exported thresholds to {config_yaml_path}")
            return config
        except Exception as e:
            log.error(f"Failed to export: {e}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clamp(self, key: str, value: float) -> float:
        lo, hi = THRESHOLD_BOUNDS.get(key, (0.0, 1.0))
        return max(lo, min(hi, value))

    def _in_cooldown(self, key: str) -> bool:
        last_str = self._meta["last_adjusted"].get(key)
        if not last_str:
            return False
        return (datetime.now() - datetime.fromisoformat(last_str)) < timedelta(days=COOLDOWN_DAYS)

    def _log_adjustment(self, key: str, old: float | None, new: float, reason: str):
        self._meta["adjustment_log"].append({
            "timestamp": datetime.now().isoformat(),
            "key":       key,
            "old":       round(old, 4) if old is not None else None,
            "new":       round(new, 4),
            "reason":    reason,
        })
        self._meta["adjustment_log"] = self._meta["adjustment_log"][-500:]


if __name__ == "__main__":
    t = AdaptiveThresholdsV2()
    print("=== Thresholds ===")
    for k, v in t.get_all().items():
        lo, hi = THRESHOLD_BOUNDS.get(k, (0, 1))
        print(f"  {k}: {v:.3f}  [{lo}–{hi}]")
    print("\n=== Trend ===")
    trend = t.get_trend()
    print(f"  Direction: {trend['direction']} | Error rate: {trend['current_error_rate']:.1%}")
    print("\n=== History (5 gần nhất) ===")
    for adj in t.get_adjustment_history(last_n=5):
        print(f"  [{adj['timestamp'][:10]}] {adj['key']}: {adj['old']} → {adj['new']}")
