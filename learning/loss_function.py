# -*- coding: utf-8 -*-
"""
Flood Pipeline — Loss Function (v1)
=====================================
Hàm mất mát tổng hợp để đo lường chất lượng dự đoán của pipeline.

    Loss = α · L_depth  +  β · L_level  +  γ · L_confidence

Components
----------
L_depth      : Huber loss cho depth regression (cm)
               → L2 khi error nhỏ, L1 khi error lớn (robust với outlier)

L_level      : Ordinal cross-entropy loss cho flood level classification
               → Phạt theo *khoảng cách thứ tự* giữa predicted và actual
               → ANKLE→WAIST (dist=2) nặng hơn ANKLE→KNEE (dist=1)

L_confidence : Calibration loss — |confidence − accuracy|
               → Tự tin cao mà sai → loss cao; tự tin thấp mà đúng → cũng loss

Tất cả 3 component đều normalize về [0, 1] trước khi gộp.

Tích hợp vào
------------
  learning/error_tracker.py      — lưu loss khi có human review
  learning/active_learner.py     — dùng loss để score review case
  learning/adaptive_thresholds.py— dùng avg loss để quyết định điều chỉnh
  learning_update.py             — hiển thị loss metrics trong update_learning()
"""

import math
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ─── Flood level ordinal scale ───────────────────────────────────────────────
LEVEL_ORDER: Dict[str, int] = {
    "NO_FLOOD":  0,
    "PUDDLE":    1,
    "ANKLE":     2,
    "KNEE":      3,
    "WAIST":     4,
    "CHEST":     5,
    "SUBMERGED": 6,
}
MAX_LEVEL_DIST = len(LEVEL_ORDER) - 1  # = 6

# ─── Default weights ─────────────────────────────────────────────────────────
DEFAULT_ALPHA       = 0.50  # depth regression
DEFAULT_BETA        = 0.30  # level classification
DEFAULT_GAMMA       = 0.20  # confidence calibration
DEFAULT_HUBER_DELTA = 30.0  # cm — chuyển từ L2→L1 khi error vượt ngưỡng này


class FloodLossFunction:
    """
    Composite Loss Function cho Flood Depth Estimation Pipeline.

    Đây *không phải* loss function của neural network — đây là hàm đánh giá
    heuristic được tính *sau* khi human review, dùng để:
      1. Đo lường độ lệch của từng prediction (depth + level + confidence)
      2. Ưu tiên review case có loss cao hơn
      3. Hướng dẫn điều chỉnh adaptive thresholds
      4. Theo dõi xu hướng chất lượng pipeline theo thời gian
    """

    def __init__(
        self,
        alpha:       float = DEFAULT_ALPHA,
        beta:        float = DEFAULT_BETA,
        gamma:       float = DEFAULT_GAMMA,
        huber_delta: float = DEFAULT_HUBER_DELTA,
    ):
        total = alpha + beta + gamma
        if not math.isclose(total, 1.0, rel_tol=1e-4):
            raise ValueError(
                f"alpha + beta + gamma phải = 1.0 (got {total:.4f}). "
                f"Điều chỉnh lại các weight."
            )
        self.alpha       = alpha
        self.beta        = beta
        self.gamma       = gamma
        self.huber_delta = huber_delta
        log.debug(
            f"[LossFunction] Khởi tạo α={alpha} β={beta} γ={gamma} "
            f"δ={huber_delta}cm"
        )

    # ─── Component losses (đều trả về float trong [0, 1]) ───────────────────

    def depth_huber_loss(self, predicted_cm: float, actual_cm: float) -> float:
        """
        Huber loss cho depth estimation — normalize về [0, 1].

        Công thức:
          error ≤ δ  →  0.5 * error² / δ      (L2 — nhạy với sai số nhỏ)
          error > δ  →  error − 0.5 * δ        (L1 — ít bị outlier ảnh hưởng)

        Normalize: chia cho (max_depth_error − 0.5*δ) với max_depth = 300cm.
        """
        error = abs(float(predicted_cm) - float(actual_cm))
        delta = self.huber_delta
        if error <= delta:
            raw = 0.5 * error ** 2 / delta
        else:
            raw = error - 0.5 * delta
        max_raw = max(300.0 - 0.5 * delta, 1.0)
        return min(raw / max_raw, 1.0)

    def level_ordinal_loss(self, predicted_level: str, actual_level: str) -> float:
        """
        Ordinal loss cho flood level classification — normalize về [0, 1].

        Phạt theo khoảng cách thứ tự giữa hai level:
          dist = |LEVEL_ORDER[pred] − LEVEL_ORDER[actual]|
          loss = dist / MAX_LEVEL_DIST

        Level không xác định → penalty tối đa = 1.0.
        """
        pred_ord   = LEVEL_ORDER.get((predicted_level or "").upper().strip(), -1)
        actual_ord = LEVEL_ORDER.get((actual_level    or "").upper().strip(), -1)
        if pred_ord < 0 or actual_ord < 0:
            return 1.0
        return abs(pred_ord - actual_ord) / MAX_LEVEL_DIST

    def confidence_calibration_loss(
        self, confidence: float, is_correct: bool
    ) -> float:
        """
        Expected Calibration Error cho một sample — trong [0, 1].

        |confidence − outcome|

        Model tự tin 0.9 nhưng sai → loss = 0.9  (rất cao)
        Model tự tin 0.3 nhưng đúng → loss = 0.7 (cũng cao, under-confident)
        Model tự tin 0.8 và đúng    → loss = 0.2  (tốt)
        """
        conf = max(0.0, min(1.0, float(confidence or 0.5)))
        return abs(conf - float(is_correct))

    # ─── Composite loss ──────────────────────────────────────────────────────

    def compute(
        self,
        predicted_depth: Optional[float],
        actual_depth:    Optional[float],
        predicted_level: Optional[str],
        actual_level:    Optional[str],
        confidence:      Optional[float] = 0.5,
    ) -> Dict:
        """
        Tính composite loss cho một prediction đã có ground truth.

        Returns
        -------
        dict với các key:
          composite_loss  float [0,1] — loss tổng hợp (weighted sum)
          depth_loss      float [0,1] — Huber depth loss (normalized)
          level_loss      float [0,1] — Ordinal level loss
          confidence_loss float [0,1] — Calibration loss
          depth_error_cm  float       — |predicted − actual| in cm
          level_correct   bool        — predicted_level == actual_level
          level_distance  int         — ordinal distance (−1 nếu unknown)
        """
        pred_d = float(predicted_depth or 0)
        act_d  = float(actual_depth    or 0)
        pred_l = (predicted_level or "").upper().strip()
        act_l  = (actual_level    or "").upper().strip()
        conf   = float(confidence  or 0.5)

        l_depth = self.depth_huber_loss(pred_d, act_d)
        l_level = self.level_ordinal_loss(pred_l, act_l)
        is_correct = bool(pred_l and act_l and pred_l == act_l)
        l_conf  = self.confidence_calibration_loss(conf, is_correct)

        composite = (
            self.alpha * l_depth
            + self.beta  * l_level
            + self.gamma * l_conf
        )

        pred_ord = LEVEL_ORDER.get(pred_l, -1)
        act_ord  = LEVEL_ORDER.get(act_l,  -1)
        level_dist = (
            abs(pred_ord - act_ord)
            if pred_ord >= 0 and act_ord >= 0 else -1
        )

        return {
            "composite_loss":  round(composite, 6),
            "depth_loss":      round(l_depth,   6),
            "level_loss":      round(l_level,   6),
            "confidence_loss": round(l_conf,    6),
            "depth_error_cm":  round(abs(pred_d - act_d), 1),
            "level_correct":   is_correct,
            "level_distance":  level_dist,
        }

    # ─── Batch analysis ──────────────────────────────────────────────────────

    def batch_loss(self, records: List[Dict]) -> Dict:
        """
        Tính trung bình loss trên một batch records.

        Mỗi record cần có:
          predicted_depth, actual_depth,
          predicted_level, actual_level,
          confidence (optional, default 0.5),
          image_path (optional, dùng để báo worst cases)
        """
        if not records:
            return {
                "n": 0,
                "avg_composite":  0.0,
                "avg_depth_loss": 0.0,
                "avg_level_loss": 0.0,
                "avg_conf_loss":  0.0,
                "depth_mae_cm":   0.0,
                "level_accuracy": 0.0,
                "worst_cases":    [],
            }

        computed = []
        for r in records:
            loss = self.compute(
                r.get("predicted_depth"),
                r.get("actual_depth"),
                r.get("predicted_level"),
                r.get("actual_level"),
                r.get("confidence", 0.5),
            )
            loss["_path"] = r.get("image_path", "")
            computed.append(loss)

        n = len(computed)
        avg_composite = sum(c["composite_loss"]  for c in computed) / n
        avg_depth     = sum(c["depth_loss"]      for c in computed) / n
        avg_level     = sum(c["level_loss"]      for c in computed) / n
        avg_conf      = sum(c["confidence_loss"] for c in computed) / n
        depth_mae     = sum(c["depth_error_cm"]  for c in computed) / n
        level_acc     = sum(1 for c in computed if c["level_correct"]) / n

        worst = sorted(computed, key=lambda x: -x["composite_loss"])[:5]
        worst_cases = [
            {
                "path":          w["_path"],
                "loss":          w["composite_loss"],
                "depth_err_cm":  w["depth_error_cm"],
                "level_dist":    w["level_distance"],
            }
            for w in worst
        ]

        return {
            "n":              n,
            "avg_composite":  round(avg_composite, 4),
            "avg_depth_loss": round(avg_depth,     4),
            "avg_level_loss": round(avg_level,     4),
            "avg_conf_loss":  round(avg_conf,      4),
            "depth_mae_cm":   round(depth_mae,     1),
            "level_accuracy": round(level_acc,     3),
            "worst_cases":    worst_cases,
        }

    # ─── Utilities ───────────────────────────────────────────────────────────

    def loss_to_priority_score(self, composite_loss: float) -> float:
        """
        Convert composite loss [0,1] → priority score [0,100].
        Cộng thêm vào base score trong ActiveLearner.score_case().
        """
        return round(min(composite_loss * 100.0, 100.0), 1)

    def get_threshold_recommendations(self, batch_stats: Dict) -> Dict[str, str]:
        """
        Từ batch loss stats, đưa ra khuyến nghị điều chỉnh thresholds.
        Dùng trong adaptive_thresholds.adjust() và update_learning().
        """
        recs: Dict[str, str] = {}
        avg_c = batch_stats.get("avg_composite",  0.0)
        avg_d = batch_stats.get("avg_depth_loss", 0.0)
        avg_l = batch_stats.get("avg_level_loss", 0.0)
        avg_k = batch_stats.get("avg_conf_loss",  0.0)
        acc   = batch_stats.get("level_accuracy", 1.0)
        n     = batch_stats.get("n", 0)

        if n < 5:
            recs["overall"] = "⚠️ Chưa đủ data để đánh giá (< 5 reviewed cases)"
            return recs

        # Tổng hợp
        if avg_c < 0.10:
            recs["overall"] = "✅ Pipeline hoạt động tốt — loss thấp"
        elif avg_c < 0.25:
            recs["overall"] = "➡️ Pipeline ổn định — tiếp tục theo dõi"
        elif avg_c < 0.45:
            recs["overall"] = "⚠️ Loss trung bình — nên thu thập thêm review"
        else:
            recs["overall"] = "🔴 Loss cao — pipeline cần cải thiện ngay"

        # Depth
        if avg_d > 0.45:
            recs["depth"] = (
                "Depth error rất cao — thử: (1) tăng yolo_conf, "
                "(2) đổi depth_model, (3) thu thập thêm ảnh có người"
            )
        elif avg_d > 0.25:
            recs["depth"] = "Depth error trên trung bình — xem lại reference objects"

        # Level classification
        if avg_l > 0.35:
            recs["level"] = (
                "Level classification sai nhiều — thêm training data "
                "cho các level bị nhầm lẫn"
            )
        elif avg_l > 0.20:
            recs["level"] = "Có nhầm lẫn giữa một số level kề nhau"

        # Confidence calibration
        if avg_k > 0.40:
            recs["confidence"] = (
                "Model kém calibrated — tăng min_confidence "
                "hoặc xem lại logic tính confidence"
            )
        elif avg_k > 0.25:
            recs["confidence"] = "Model hơi over-confident — theo dõi thêm"

        # Level accuracy
        if acc < 0.55:
            recs["level_accuracy"] = (
                f"Level accuracy thấp ({acc:.0%}) — xem lại boundary "
                "giữa ANKLE/KNEE/WAIST"
            )

        return recs

    def describe(self) -> str:
        """Mô tả ngắn gọn cấu hình hàm mất mát hiện tại."""
        return (
            f"FloodLoss(α={self.alpha} depth-Huber[δ={self.huber_delta}cm], "
            f"β={self.beta} level-Ordinal, "
            f"γ={self.gamma} conf-Calibration)"
        )


# ─── Module-level singleton ───────────────────────────────────────────────────
_default_loss_fn: Optional[FloodLossFunction] = None


def get_loss_function(
    alpha:       float = DEFAULT_ALPHA,
    beta:        float = DEFAULT_BETA,
    gamma:       float = DEFAULT_GAMMA,
    huber_delta: float = DEFAULT_HUBER_DELTA,
) -> FloodLossFunction:
    """
    Trả về instance singleton của FloodLossFunction.
    Tạo mới nếu chưa có; dùng default weights nếu không truyền tham số.
    """
    global _default_loss_fn
    if _default_loss_fn is None:
        _default_loss_fn = FloodLossFunction(alpha, beta, gamma, huber_delta)
        log.info(f"[LossFunction] {_default_loss_fn.describe()}")
    return _default_loss_fn


# ─── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    fn = FloodLossFunction()
    print(fn.describe())
    print()

    test_cases = [
        # (pred_depth, act_depth, pred_level, act_level, conf, desc)
        (45.0,  40.0, "ANKLE",     "ANKLE",     0.90, "Đúng, confident"),
        (45.0,  40.0, "KNEE",      "ANKLE",     0.85, "Level sai 1 bậc"),
        (120.0, 40.0, "ANKLE",     "WAIST",     0.75, "Depth + level đều sai"),
        (0.0,  80.0,  "NO_FLOOD",  "KNEE",      0.95, "Miss flood hoàn toàn"),
        (50.0, 50.0,  "KNEE",      "KNEE",      0.35, "Đúng nhưng under-confident"),
    ]

    print(f"{'Case':<35} {'Comp':>6} {'Depth':>6} {'Level':>6} {'Conf':>6}  {'DepthErr':>9}")
    print("─" * 75)
    for pred_d, act_d, pred_l, act_l, conf, desc in test_cases:
        r = fn.compute(pred_d, act_d, pred_l, act_l, conf)
        print(
            f"{desc:<35} {r['composite_loss']:>6.3f} "
            f"{r['depth_loss']:>6.3f} {r['level_loss']:>6.3f} "
            f"{r['confidence_loss']:>6.3f}  {r['depth_error_cm']:>7.1f}cm"
        )

    print()
    records = [
        {"predicted_depth": pd, "actual_depth": ad,
         "predicted_level": pl, "actual_level": al,
         "confidence": c, "image_path": f"img_{i}.jpg"}
        for i, (pd, ad, pl, al, c, _) in enumerate(test_cases)
    ]
    batch = fn.batch_loss(records)
    print(f"Batch ({batch['n']} samples):")
    print(f"  avg_composite = {batch['avg_composite']:.4f}")
    print(f"  depth_mae_cm  = {batch['depth_mae_cm']:.1f} cm")
    print(f"  level_accuracy= {batch['level_accuracy']:.0%}")
    print()
    recs = fn.get_threshold_recommendations(batch)
    for k, v in recs.items():
        print(f"  [{k}] {v}")
