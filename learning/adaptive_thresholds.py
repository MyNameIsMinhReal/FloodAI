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
    
    [IMPROVE] Integrated Bayesian Thompson Sampling optimizer for smarter threshold tuning.
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
            "version":        "3.1",
            "adjustment_log": [],
            "last_adjusted":  {},
            "accuracy_trend": [],
        }
        
        # [IMPROVE] Bayesian Thompson Sampling optimizer
        self._bayes_opt = BayesianThresholdOptimizer(n_bins=15)
        self._bayes_opt.initialize(list(self.thresholds.keys()))
        
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

            # [IMPROVE] Bayesian Thompson Sampling: so sánh EMA vs Bayesian suggestion
            bayes_val = self._bayes_opt.get_best(key)
            # Combine: 70% EMA+delta, 30% Bayesian
            final_val = self._clamp(key, 0.7 * new_val + 0.3 * bayes_val)
            
            # Update Bayesian posterior with observed reward
            # Reward = negative error rate improvement
            error_rate_improvement = (1.0 - error_rate)  # reward higher when error rate low
            self._bayes_opt.update(key, final_val, error_rate_improvement)

            self.thresholds[key] = final_val
            self._meta["last_adjusted"][key] = datetime.now().isoformat()
            reason = (
                f"auto (delta={delta:+.3f}, EMA={ema:.2f}, "
                f"bayes={bayes_val:.3f}, {total_errors} errors/{total_processed} processed, "
                f"avg_loss={avg_composite_loss:.3f})"
            )
            self._log_adjustment(key, old_val, final_val, reason)
            adjusted.append(
                f"{key}: {old_val:.3f} → {final_val:.3f} "
                f"(delta={delta:+.3f}, bayes={bayes_val:.3f}, loss={avg_composite_loss:.3f})"
            )
            log.info(f"[AdaptiveThresholds v3.1] Adjusted {key}: {old_val:.3f} → {final_val:.3f}")

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


import json
import logging
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


# [IMPROVE] Bayesian Optimization for Threshold Tuning
class BayesianThresholdOptimizer:
    """
    Bayesian Optimization (Thompson Sampling) cho auto-tuning thresholds.
    
    Mỗi threshold là một "arm" trong multi-armed bandit.
    State: Beta(alpha, beta) distribution per threshold value bin.
    Reward: negative error rate after applying threshold.
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        # Per threshold: n_bins * (alpha, beta) pairs
        self.posterior = {}  # {threshold_name: [(alpha, beta), ...] for each bin}
        self.bounds = THRESHOLD_BOUNDS
        
    def initialize(self, threshold_names: List[str]):
        """Khởi tạo prior Beta(1, 1) uniform cho mỗi bin."""
        for name in threshold_names:
            lo, hi = self.bounds.get(name, (0.0, 1.0))
            self.posterior[name] = [(1.0, 1.0) for _ in range(self.n_bins)]
    
    def sample(self, name: str) -> float:
        """Thompson sampling: sample từ posterior, chọn bin tốt nhất."""
        if name not in self.posterior:
            lo, hi = self.bounds.get(name, (0.0, 1.0))
            return (lo + hi) / 2
        
        lo, hi = self.bounds.get(name, (0.0, 1.0))
        best_bin = 0
        best_sample = -1
        for i, (alpha, beta) in enumerate(self.posterior[name]):
            sample = random.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best_bin = i
        # Map bin → value
        return lo + (best_bin + 0.5) * (hi - lo) / self.n_bins
    
    def update(self, name: str, value: float, reward: float):
        """Cập nhật posterior với reward (negative error rate)."""
        if name not in self.posterior:
            self.initialize([name])
        
        lo, hi = self.bounds.get(name, (0.0, 1.0))
        bin_idx = min(int((value - lo) / (hi - lo) * self.n_bins), self.n_bins - 1)
        bin_idx = max(0, bin_idx)
        
        alpha, beta = self.posterior[name][bin_idx]
        # Reward ∈ [-1, 1] → shift to [0, 1] for Bernoulli likelihood
        # reward = 1 - error_rate (đã là positive)
        p = max(0.0, min(1.0, (reward + 1.0) / 2.0))
        
        # Bayesian update: Beta(alpha + p, beta + 1 - p)
        self.posterior[name][bin_idx] = (alpha + p, beta + (1.0 - p))
    
    def get_best(self, name: str) -> float:
        """Trả về giá trị threshold với posterior mean cao nhất."""
        if name not in self.posterior:
            lo, hi = self.bounds.get(name, (0.0, 1.0))
            return (lo + hi) / 2
        
        lo, hi = self.bounds.get(name, (0.0, 1.0))
        best_bin = 0
        best_mean = -1
        for i, (alpha, beta) in enumerate(self.posterior[name]):
            mean = alpha / (alpha + beta)
            if mean > best_mean:
                best_mean = mean
                best_bin = i
        return lo + (best_bin + 0.5) * (hi - lo) / self.n_bins
