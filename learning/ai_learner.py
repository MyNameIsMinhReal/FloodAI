# -*- coding: utf-8 -*-
"""
AI Learner — Học từ correction của người review
=================================================
Ba cơ chế học kết hợp:

  1. k-NN Correction Engine
     Tìm K ảnh đã-review giống nhất (dựa trên feature vector)
     → áp dụng weighted average của các correction đó lên prediction mới.

  2. Confidence Calibrator
     Histogram binning: track P(đúng | confidence_bin, level)
     → calibrate lại confidence score để nó phản ánh đúng thực tế.

  3. Level Bias Corrector
     Confusion matrix: track model hay nhầm level nào sang level nào
     → correction tự động khi phát hiện systematic bias.

Không cần sklearn / scipy — chỉ dùng numpy + sqlite3 (đã có sẵn).

Integration:
    # Sau khi ReferenceEstimator predict xong:
    from learning.ai_learner import AiLearner
    al = AiLearner()
    result = al.correct(result, image_features)

    # Khi người review submit correction:
    al.invalidate()  # model tự retrain lần sau

Run standalone để xem stats:
    python learning/ai_learner.py --stats
    python learning/ai_learner.py --train
"""

import json
import logging
import math
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
REVIEW_DB_PATH = "learning/review_queue.db"
MODEL_CACHE    = "learning/ai_model_cache.json"

# Flood level index map (consistent với codebase)
LEVEL_IDX = {
    "NO_FLOOD":  0,
    "PUDDLE":    1,
    "ANKLE":     2,
    "KNEE":      3,
    "WAIST":     4,
    "CHEST":     5,
    "SUBMERGED": 6,
}
IDX_LEVEL = {v: k for k, v in LEVEL_IDX.items()}

# k-NN
K_NEIGHBORS         = 7      # số neighbors
MIN_SIMILARITY      = 0.25   # similarity tối thiểu để tính correction
MIN_TRAINING_CASES  = 5      # cần ít nhất N reviewed cases để kích hoạt

# Confidence calibration
N_CONF_BINS         = 10     # số bin cho histogram calibration

# Bias correction
MIN_BIAS_CASES      = 8      # cần ít nhất N case để phát hiện bias trong 1 level
BIAS_THRESHOLD      = 0.40   # >= 40% case nhầm cùng 1 hướng → có bias

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_feature_vector(
    features: dict,
    predicted_depth: float,
    predicted_level: str,
    confidence: float,
    dino_embedding: Optional[np.ndarray] = None,  # [IMPROVE] DINOv2 semantic embedding
) -> np.ndarray:
    """
    Chuyển features dict + prediction → numpy vector chuẩn hóa.

    Vector gốc 10 chiều + DINOv2 embedding (384/768 dim):
      [0-9]   Hand-crafted features (brightness, blur, aspect, night, objects, people, pose, depth, level, confidence)
      [10...] DINOv2 embedding (semantic understanding of scene)
    """
    brightness   = float(features.get("brightness",   128.0))
    blur_score   = float(features.get("blur_score",   100.0))
    aspect_ratio = float(features.get("aspect_ratio", 1.33))
    is_night     = float(bool(features.get("is_night", False)))
    num_objects  = float(features.get("num_reference_objects", 0))
    num_people   = float(features.get("num_people", 0))
    has_pose     = float(bool(features.get("has_pose", False)))

    lvl_idx = LEVEL_IDX.get(str(predicted_level).upper(), 0)

    handcrafted = np.array([
        min(brightness / 255.0, 1.0),
        min(math.log1p(blur_score) / math.log1p(2000), 1.0),
        min(aspect_ratio / 3.0, 1.0),
        is_night,
        min(num_objects / 10.0, 1.0),
        min(num_people / 5.0, 1.0),
        has_pose,
        min(float(predicted_depth) / 500.0, 1.0),
        lvl_idx / 6.0,
        min(float(confidence), 1.0),
    ], dtype=np.float32)

    # [IMPROVE] Append DINOv2 embedding for semantic understanding
    if dino_embedding is not None:
        # Normalize embedding to unit length
        dino_norm = dino_embedding / (np.linalg.norm(dino_embedding) + 1e-6)
        return np.concatenate([handcrafted, dino_norm.astype(np.float32)])
    
    return handcrafted


# [IMPROVE] DINOv2 embedding extraction helper
def extract_dino_embedding(img_rgb: np.ndarray, model_name: str = "dinov2_vits14") -> Optional[np.ndarray]:
    """
    Trích xuất DINOv2 embedding từ ảnh.
    Returns: (384,) vector cho dinov2_vits14 hoặc (768,) cho dinov2_vitb14
    """
    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image
        
        # Load model (cache globally)
        global _DINO_MODEL, _DINO_TRANSFORM, _DINO_DEVICE
        if '_DINO_MODEL' not in globals() or _DINO_MODEL is None:
            _DINO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            _DINO_MODEL = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True).to(_DINO_DEVICE).eval()
            _DINO_TRANSFORM = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        pil_img = Image.fromarray(img_rgb).convert("RGB")
        tensor = _DINO_TRANSFORM(pil_img).unsqueeze(0).to(_DINO_DEVICE)
        
        with torch.no_grad():
            embedding = _DINO_MODEL(tensor).cpu().numpy().squeeze()
        
        return embedding
    except Exception as e:
        log.debug(f"DINOv2 embedding failed: {e}")
        return None


# ── kNN Correction Engine ─────────────────────────────────────────────────────

class KNNCorrectionEngine:
    """
    Học correction pattern từ reviewed cases.

    Mỗi training point = (feature_vector, depth_correction, level_correction)
    Khi inference: tìm K neighbors gần nhất → weighted average correction.
    
    [IMPROVE] Hỗ trợ variable-dim features (handcrafted + DINOv2).
    Dùng cosine similarity cho DINO part, Euclidean cho handcrafted.
    """

    def __init__(self, handcrafted_dim: int = 10, dino_dim: int = 384):
        self.handcrafted_dim = handcrafted_dim
        self.dino_dim = dino_dim
        self.total_dim = handcrafted_dim + dino_dim
        
        # Separate storage for efficient similarity computation
        self.X_hc: np.ndarray = np.empty((0, handcrafted_dim), dtype=np.float32)  # handcrafted
        self.X_dino: np.ndarray = np.empty((0, dino_dim), dtype=np.float32)       # DINO embeddings
        self.d_corr: np.ndarray = np.empty(0, dtype=np.float32)
        self.l_corr: np.ndarray = np.empty(0, dtype=np.int8)
        self.trained_at: Optional[str] = None
        self.n_cases: int = 0
        
        # [IMPROVE] Approximate NN index (simple ball tree for dino embeddings)
        self._dino_tree = None

    def fit(self, cases: List[dict], images_rgb: Optional[List[np.ndarray]] = None) -> int:
        """Train từ danh sách reviewed cases.
        
        [IMPROVE] images_rgb: optional list of RGB images để extract DINOv2 embeddings
        """
        rows_hc, rows_dino, rows_d, rows_l = [], [], [], []

        for i, c in enumerate(cases):
            try:
                features = json.loads(c.get("features") or "{}")
                pred_dep = float(c.get("predicted_depth") or 0)
                pred_lv  = str(c.get("predicted_level") or "NO_FLOOD")
                conf     = float(c.get("confidence")     or 0)
                act_dep  = float(c.get("actual_depth")   or 0)
                act_lv   = str(c.get("actual_level")     or "NO_FLOOD")

                # Extract DINOv2 embedding if images provided
                dino_emb = None
                if images_rgb is not None and i < len(images_rgb):
                    dino_emb = extract_dino_embedding(images_rgb[i])
                
                vec = extract_feature_vector(features, pred_dep, pred_lv, conf, dino_emb)
                d_c  = act_dep - pred_dep
                l_c  = LEVEL_IDX.get(act_lv.upper(), 0) - LEVEL_IDX.get(pred_lv.upper(), 0)

                # Split into handcrafted + DINO
                hc = vec[:self.handcrafted_dim]
                dino = vec[self.handcrafted_dim:] if len(vec) > self.handcrafted_dim else np.zeros(self.dino_dim, dtype=np.float32)
                
                rows_hc.append(hc)
                rows_dino.append(dino)
                rows_d.append(d_c)
                rows_l.append(l_c)
            except Exception as e:
                log.debug(f"[kNN] Skip case: {e}")
                continue

        if not rows_hc:
            return 0

        self.X_hc     = np.array(rows_hc, dtype=np.float32)
        self.X_dino   = np.array(rows_dino, dtype=np.float32)
        self.d_corr   = np.array(rows_d, dtype=np.float32)
        self.l_corr   = np.array(rows_l, dtype=np.int8)
        self.trained_at = datetime.now().isoformat()
        self.n_cases    = len(rows_hc)
        
        # [IMPROVE] Build simple ball tree for DINO embeddings (approximate NN)
        if self.n_cases > 20:
            self._build_dino_index()
            
        log.info(f"[kNN] Trained on {self.n_cases} reviewed cases (dim={self.total_dim})")
        return self.n_cases

    def _build_dino_index(self):
        """Build simple KD-tree for DINO embeddings (approximate NN)."""
        try:
            from sklearn.neighbors import KDTree
            self._dino_tree = KDTree(self.X_dino, leaf_size=10)
        except ImportError:
            # Fallback: linear scan
            self._dino_tree = None

    def predict_correction(
        self,
        vec: np.ndarray,
        k: int = K_NEIGHBORS,
    ) -> Tuple[float, int, float, int]:
        """
        Trả về (depth_correction, level_correction, avg_similarity, n_used).
        
        [IMPROVE] Hybrid similarity: handcrafted (Euclidean) + DINOv2 (cosine)
        """
        if self.n_cases < MIN_TRAINING_CASES:
            return 0.0, 0, 0.0, 0

        # Split query vector
        hc_q = vec[:self.handcrafted_dim]
        dino_q = vec[self.handcrafted_dim:] if len(vec) > self.handcrafted_dim else np.zeros(self.dino_dim, dtype=np.float32)
        
        # [IMPROVE] Hybrid similarity
        # Handcrafted: L2 distance
        diffs_hc = self.X_hc - hc_q[np.newaxis, :]
        dists_hc = np.linalg.norm(diffs_hc, axis=1)
        
        # DINOv2: Cosine similarity
        if self._dino_tree is not None and len(dino_q) > 0:
            # Query KD-tree for top-K DINO neighbors
            dino_dists, dino_idx = self._dino_tree.query(dino_q.reshape(1, -1), k=min(k * 3, self.n_cases))
            dino_dists = dino_dists.flatten()
            dino_idx = dino_idx.flatten()
            # Convert L2 to cosine-ish (assuming normalized)
            dino_sim = 1.0 / (1.0 + dino_dists)
            # Combine: use DINO indices as candidates, re-rank with combined score
            candidate_idx = dino_idx
        else:
            candidate_idx = np.arange(self.n_cases)
            # Compute DINO cosine for all (fallback)
            dino_norm_q = dino_q / (np.linalg.norm(dino_q) + 1e-6)
            dino_norm_db = self.X_dino / (np.linalg.norm(self.X_dino, axis=1, keepdims=True) + 1e-6)
            dino_sim = dino_norm_db @ dino_norm_q

        # Combined score: 0.6 * handcrafted_inv + 0.4 * dino_sim
        hc_inv = 1.0 / (1.0 + dists_hc[candidate_idx])
        dino_s = dino_sim if self._dino_tree is not None else dino_sim[candidate_idx]
        
        combined_scores = 0.6 * hc_inv + 0.4 * dino_s
        
        # Top-K by combined score
        top_k_local = np.argsort(combined_scores)[-k:][::-1]
        top_k = candidate_idx[top_k_local]
        
        # Weighted average correction
        weights = combined_scores[top_k_local]
        weights = weights / (weights.sum() + 1e-6)
        
        d_corr = float(np.sum(self.d_corr[top_k] * weights))
        l_corr = int(round(np.sum(self.l_corr[top_k] * weights)))
        avg_sim = float(weights.max())  # best similarity
        
        return d_corr, l_corr, avg_sim, len(top_k)
        dists = np.sqrt((diffs ** 2).sum(axis=1))    # (N,)

        # Chuyển distance → similarity (Gaussian kernel)
        sigma  = 0.5
        sims   = np.exp(-(dists ** 2) / (2 * sigma ** 2))

        # Lọc neighbors có similarity đủ cao
        mask = sims >= MIN_SIMILARITY
        if mask.sum() == 0:
            return 0.0, 0, 0.0, 0

        k_actual  = min(k, mask.sum())
        top_idx   = np.argsort(-sims[mask])[:k_actual]
        top_sims  = sims[mask][top_idx]
        top_d     = self.d_corr[mask][top_idx]
        top_l     = self.l_corr[mask][top_idx].astype(np.float32)

        w = top_sims / (top_sims.sum() + 1e-9)
        d_correction  = float(np.dot(w, top_d))
        l_correction  = int(round(float(np.dot(w, top_l))))
        avg_sim       = float(top_sims.mean())

        return d_correction, l_correction, avg_sim, int(k_actual)

    def to_dict(self) -> dict:
        return {
            "X":          self.X.tolist()      if self.n_cases else [],
            "d_corr":     self.d_corr.tolist() if self.n_cases else [],
            "l_corr":     self.l_corr.tolist() if self.n_cases else [],
            "trained_at": self.trained_at,
            "n_cases":    self.n_cases,
        }

    def from_dict(self, d: dict):
        x = d.get("X", [])
        if x:
            self.X      = np.array(x,         dtype=np.float32)
            self.d_corr = np.array(d["d_corr"], dtype=np.float32)
            self.l_corr = np.array(d["l_corr"], dtype=np.int8)
        self.trained_at = d.get("trained_at")
        self.n_cases    = d.get("n_cases", 0)


# ── Confidence Calibrator ─────────────────────────────────────────────────────

class ConfidenceCalibrator:
    """
    Histogram binning per-level:
      Chia confidence [0,1] thành N_CONF_BINS bin.
      Mỗi bin lưu: (n_total, n_correct).
      Calibrated confidence = n_correct / n_total trong bin đó.
    """

    def __init__(self):
        # bins[level][bin_idx] = [n_total, n_correct]
        self.bins: Dict[str, List[List[int]]] = {}
        self.n_cases: int = 0

    def _get_bin(self, conf: float) -> int:
        return min(int(conf * N_CONF_BINS), N_CONF_BINS - 1)

    def fit(self, cases: List[dict]) -> int:
        self.bins = {}
        n = 0
        for c in cases:
            try:
                pred_lv = str(c.get("predicted_level") or "NO_FLOOD").upper()
                act_lv  = str(c.get("actual_level")    or "NO_FLOOD").upper()
                conf    = float(c.get("confidence")     or 0)

                if pred_lv not in self.bins:
                    self.bins[pred_lv] = [[0, 0] for _ in range(N_CONF_BINS)]

                b  = self._get_bin(conf)
                self.bins[pred_lv][b][0] += 1
                if pred_lv == act_lv:
                    self.bins[pred_lv][b][1] += 1
                n += 1
            except Exception:
                continue

        self.n_cases = n
        log.info(f"[Calibrator] Fitted {n} cases across {len(self.bins)} levels")
        return n

    def calibrate(self, predicted_level: str, raw_confidence: float) -> Tuple[float, float]:
        """
        Trả về (calibrated_confidence, reliability_score).
        reliability_score: 0–1, cao = bin có nhiều data → đáng tin.
        """
        lvl = predicted_level.upper()
        if lvl not in self.bins:
            return raw_confidence, 0.0

        b = self._get_bin(raw_confidence)
        n_total, n_correct = self.bins[lvl][b]

        if n_total < 3:  # không đủ data trong bin này
            return raw_confidence, 0.0

        calibrated   = (n_correct + 0.5) / (n_total + 1.0)   # Laplace smoothing
        reliability  = min(n_total / 30.0, 1.0)               # tối đa 30 cases = full trust
        return float(calibrated), float(reliability)

    def get_calibration_curve(self, level: str) -> List[dict]:
        """Trả về calibration curve của 1 level để hiển thị."""
        lvl = level.upper()
        if lvl not in self.bins:
            return []
        result = []
        for i, (n_total, n_correct) in enumerate(self.bins[lvl]):
            bin_center = (i + 0.5) / N_CONF_BINS
            empirical  = n_correct / n_total if n_total > 0 else None
            result.append({
                "conf_range": f"{i/N_CONF_BINS:.1f}–{(i+1)/N_CONF_BINS:.1f}",
                "empirical_accuracy": round(empirical, 3) if empirical is not None else None,
                "n_samples":  n_total,
            })
        return result

    def to_dict(self) -> dict:
        return {"bins": self.bins, "n_cases": self.n_cases}

    def from_dict(self, d: dict):
        self.bins    = d.get("bins", {})
        self.n_cases = d.get("n_cases", 0)


# ── Level Bias Corrector ──────────────────────────────────────────────────────

class LevelBiasCorrector:
    """
    Confusion matrix approach:
      Track systematic bias: model hay dự đoán level X nhưng thực ra là level Y.
      Nếu >= BIAS_THRESHOLD của các case predict=X thực ra là Y → recommend shift.
    """

    def __init__(self):
        # confusion[pred_level][actual_level] = count
        self.confusion: Dict[str, Dict[str, int]] = {}
        # bias_map[pred_level] = recommended_actual (nếu có bias rõ ràng)
        self.bias_map:  Dict[str, Optional[str]] = {}
        self.n_cases: int = 0

    def fit(self, cases: List[dict]) -> int:
        self.confusion = {}
        n = 0
        for c in cases:
            try:
                pred_lv = str(c.get("predicted_level") or "NO_FLOOD").upper()
                act_lv  = str(c.get("actual_level")    or "NO_FLOOD").upper()
                if pred_lv not in self.confusion:
                    self.confusion[pred_lv] = {}
                self.confusion[pred_lv][act_lv] = self.confusion[pred_lv].get(act_lv, 0) + 1
                n += 1
            except Exception:
                continue

        self.n_cases = n
        self._compute_bias_map()
        log.info(f"[BiasCorrector] Fitted {n} cases, found bias in: "
                 f"{[k for k,v in self.bias_map.items() if v]}")
        return n

    def _compute_bias_map(self):
        self.bias_map = {}
        for pred_lv, actual_counts in self.confusion.items():
            total = sum(actual_counts.values())
            if total < MIN_BIAS_CASES:
                self.bias_map[pred_lv] = None
                continue

            # Tìm actual level phổ biến nhất khác pred_lv
            wrong_counts = {k: v for k, v in actual_counts.items() if k != pred_lv}
            if not wrong_counts:
                self.bias_map[pred_lv] = None
                continue

            top_wrong = max(wrong_counts, key=lambda k, wc=wrong_counts: wc[k])
            if wrong_counts[top_wrong] / total >= BIAS_THRESHOLD:
                self.bias_map[pred_lv] = top_wrong
            else:
                self.bias_map[pred_lv] = None

    def get_correction(self, predicted_level: str) -> Tuple[Optional[str], float]:
        """
        Trả về (suggested_level, bias_strength).
        suggested_level = None nếu không có bias đủ mạnh.
        """
        lvl = predicted_level.upper()
        suggestion = self.bias_map.get(lvl)
        if suggestion is None:
            return None, 0.0

        total      = sum(self.confusion.get(lvl, {}).values())
        wrong_cnt  = self.confusion.get(lvl, {}).get(suggestion, 0)
        strength   = wrong_cnt / max(total, 1)
        return suggestion, float(strength)

    def get_report(self) -> List[dict]:
        """Báo cáo tất cả bias được phát hiện."""
        report = []
        for pred_lv, suggestion in self.bias_map.items():
            if suggestion:
                total     = sum(self.confusion.get(pred_lv, {}).values())
                wrong_cnt = self.confusion.get(pred_lv, {}).get(suggestion, 0)
                report.append({
                    "predicted":    pred_lv,
                    "actual":       suggestion,
                    "bias_pct":     round(wrong_cnt / max(total, 1) * 100, 1),
                    "total_cases":  total,
                })
        return sorted(report, key=lambda x: -x["bias_pct"])

    def to_dict(self) -> dict:
        return {"confusion": self.confusion, "bias_map": self.bias_map, "n_cases": self.n_cases}

    def from_dict(self, d: dict):
        self.confusion = d.get("confusion", {})
        self.bias_map  = d.get("bias_map", {})
        self.n_cases   = d.get("n_cases", 0)


# ── AiLearner — Main Class ────────────────────────────────────────────────────

class AiLearner:
    """
    Hệ thống tự học từ correction của người review.

    Usage:
        al = AiLearner()

        # Áp dụng correction vào kết quả live_predict:
        corrected = al.correct(
            predicted_depth=85.0,
            predicted_level="WAIST",
            confidence=0.62,
            image_features=features_dict,
        )

        # Khi có review mới:
        al.invalidate()

        # Train lại thủ công:
        al.train()
    """

    def __init__(
        self,
        review_db: str  = REVIEW_DB_PATH,
        cache_path: str = MODEL_CACHE,
    ):
        self.review_db  = Path(review_db)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.knn        = KNNCorrectionEngine()
        self.calibrator = ConfidenceCalibrator()
        self.bias       = LevelBiasCorrector()

        self._trained   = False
        self._load_cache()

    # ── Public API ────────────────────────────────────────────────────────────

    def correct(
        self,
        predicted_depth: float,
        predicted_level: str,
        confidence: float,
        image_features: dict,
    ) -> dict:
        """
        Áp dụng 3 lớp correction và trả về dict kết quả.

        Returns:
            {
              "depth_cm":             float,   # depth sau correction
              "level":                str,     # level sau correction
              "confidence_raw":       float,
              "confidence_cal":       float,   # calibrated confidence
              "depth_correction":     float,   # delta áp dụng
              "level_correction":     int,     # level steps shifted
              "bias_detected":        bool,
              "bias_suggestion":      str|None,
              "n_neighbors":          int,     # neighbors dùng để correct
              "avg_similarity":       float,
              "correction_strength":  float,   # 0–1, mức độ tin correction
              "explanation":          str,
              "ai_active":            bool,
            }
        """
        if not self._trained:
            self._load_or_train()

        result = {
            "depth_cm":            predicted_depth,
            "level":               predicted_level,
            "confidence_raw":      confidence,
            "confidence_cal":      confidence,
            "depth_correction":    0.0,
            "level_correction":    0,
            "bias_detected":       False,
            "bias_suggestion":     None,
            "n_neighbors":         0,
            "avg_similarity":      0.0,
            "correction_strength": 0.0,
            "explanation":         "AI chưa đủ data để học",
            "ai_active":           False,
        }

        n_total = max(
            self.knn.n_cases,
            self.calibrator.n_cases,
            self.bias.n_cases,
        )

        if n_total < MIN_TRAINING_CASES:
            result["explanation"] = (
                f"AI cần ít nhất {MIN_TRAINING_CASES} reviewed cases "
                f"(hiện có {n_total}). Hãy review thêm ảnh."
            )
            return result

        result["ai_active"] = True
        explanations = []

        # ── Layer 1: k-NN depth & level correction ────────────────────────────
        vec = extract_feature_vector(image_features, predicted_depth, predicted_level, confidence)
        d_corr, l_corr, avg_sim, n_used = self.knn.predict_correction(vec)

        if n_used > 0 and avg_sim >= MIN_SIMILARITY:
            # Dampen correction: càng ít neighbors và thấp similarity → correction nhẹ hơn
            strength    = avg_sim * min(n_used / K_NEIGHBORS, 1.0)
            d_applied   = d_corr * strength
            new_depth   = max(0.0, predicted_depth + d_applied)

            new_level_idx = LEVEL_IDX.get(predicted_level.upper(), 0) + l_corr
            new_level_idx = max(0, min(6, new_level_idx))
            new_level     = IDX_LEVEL[new_level_idx]

            result["depth_cm"]            = round(new_depth, 1)
            result["level"]               = new_level
            result["depth_correction"]    = round(d_applied, 1)
            result["level_correction"]    = l_corr
            result["n_neighbors"]         = n_used
            result["avg_similarity"]      = round(avg_sim, 3)
            result["correction_strength"] = round(strength, 3)

            if abs(d_applied) > 1.0 or l_corr != 0:
                explanations.append(
                    f"kNN ({n_used} neighbors, sim={avg_sim:.2f}): "
                    f"depth {d_corr:+.1f}cm → áp dụng {d_applied:+.1f}cm"
                    + (f", level {predicted_level}→{new_level}" if l_corr != 0 else "")
                )
            else:
                explanations.append(
                    f"kNN ({n_used} neighbors): correction nhỏ, giữ nguyên prediction"
                )

        # ── Layer 2: Confidence calibration ──────────────────────────────────
        cal_conf, reliability = self.calibrator.calibrate(
            result["level"], confidence
        )
        if reliability > 0.3:
            result["confidence_cal"] = round(cal_conf, 3)
            diff = cal_conf - confidence
            if abs(diff) > 0.05:
                direction = "tăng" if diff > 0 else "giảm"
                explanations.append(
                    f"Calibration: confidence {direction} "
                    f"{confidence:.2f}→{cal_conf:.2f} "
                    f"(dựa trên {int(reliability*30)} cases tương tự)"
                )

        # ── Layer 3: Bias correction ──────────────────────────────────────────
        bias_suggestion, bias_strength = self.bias.get_correction(result["level"])
        if bias_suggestion and bias_strength >= BIAS_THRESHOLD:
            result["bias_detected"]  = True
            result["bias_suggestion"] = bias_suggestion
            explanations.append(
                f"Bias: model hay nhầm {result['level']}→{bias_suggestion} "
                f"({bias_strength:.0%} cases). Cân nhắc kiểm tra."
            )

        result["explanation"] = " | ".join(explanations) if explanations else "Prediction ổn định"
        return result

    def train(self, force: bool = False) -> dict:
        """Load reviewed cases từ DB và train cả 3 model."""
        cases = self._load_reviewed_cases()
        if not cases and not force:
            log.info("[AiLearner] Không có reviewed cases")
            return {"n_cases": 0, "status": "no_data"}

        t0 = time.time()
        n_knn  = self.knn.fit(cases)
        n_cal  = self.calibrator.fit(cases)
        n_bias = self.bias.fit(cases)

        self._trained = True
        self._save_cache()

        elapsed = time.time() - t0
        log.info(f"[AiLearner] Trained in {elapsed:.2f}s: "
                 f"kNN={n_knn}, calibrator={n_cal}, bias={n_bias}")
        return {
            "n_cases":         len(cases),
            "n_knn":           n_knn,
            "n_calibrator":    n_cal,
            "n_bias":          n_bias,
            "elapsed_sec":     round(elapsed, 2),
            "status":          "ok",
        }

    def invalidate(self):
        """Gọi sau khi có review mới — force retrain lần sau."""
        self._trained = False
        if self.cache_path.exists():
            self.cache_path.unlink()
        log.info("[AiLearner] Cache invalidated, will retrain on next use")

    def get_stats(self) -> dict:
        """Thống kê tổng quan."""
        if not self._trained:
            self._load_or_train()

        bias_report = self.bias.get_report()
        level_cal   = {}
        for lvl in LEVEL_IDX:
            curve = self.calibrator.get_calibration_curve(lvl)
            if any(c["n_samples"] > 0 for c in curve):
                level_cal[lvl] = curve

        return {
            "n_reviewed_cases": self.knn.n_cases,
            "trained_at":       self.knn.trained_at,
            "ai_active":        self.knn.n_cases >= MIN_TRAINING_CASES,
            "min_cases_needed": MIN_TRAINING_CASES,
            "knn": {
                "n_training_points": self.knn.n_cases,
                "k_neighbors":       K_NEIGHBORS,
            },
            "calibrator": {
                "n_cases":     self.calibrator.n_cases,
                "levels_with_data": list(self.calibrator.bins.keys()),
            },
            "bias_corrector": {
                "n_cases":     self.bias.n_cases,
                "biases_found": bias_report,
            },
            "calibration_curves": level_cal,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_or_train(self):
        """Load từ cache hoặc train mới."""
        if self._load_cache():
            return
        self.train()

    def _load_reviewed_cases(self) -> List[dict]:
        """Đọc tất cả cases từ:
          1. review_queue.db  — cases đã review qua UI
          2. training_images.db — ảnh upload thủ công qua tab 'Upload Ảnh Học'
        """
        rows: List[dict] = []

        # ── Nguồn 1: review_queue.db ─────────────────────────────────────────
        if self.review_db.exists():
            try:
                conn = sqlite3.connect(str(self.review_db))
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT predicted_depth, predicted_level, confidence,
                           actual_depth, actual_level, features, timestamp
                    FROM review_queue
                    WHERE status = 'reviewed'
                      AND actual_depth   IS NOT NULL
                      AND actual_level   IS NOT NULL
                      AND predicted_depth IS NOT NULL
                    ORDER BY timestamp DESC
                """)
                rq_rows = [dict(r) for r in cur.fetchall()]
                conn.close()
                rows.extend(rq_rows)
                log.info(f"[AiLearner] review_queue: {len(rq_rows)} reviewed cases")
            except Exception as e:
                log.warning(f"[AiLearner] review_queue read failed: {e}")

        # ── Nguồn 2: training_images.db (ảnh upload thủ công / live feedback) ─
        # Tìm DB dựa trên vị trí review_db (cùng thư mục learning/)
        train_db = self.review_db.parent / "training_images.db"
        if not train_db.exists():
            # fallback: tìm theo review_db path pattern
            train_db = Path("learning/training_images.db")
        if train_db.exists():
            try:
                conn = sqlite3.connect(str(train_db))
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT actual_depth, actual_level, added_at AS timestamp
                    FROM training_images
                    WHERE verified = 1
                      AND actual_depth IS NOT NULL
                      AND actual_level IS NOT NULL
                    ORDER BY id DESC
                """)
                ti_rows = cur.fetchall()
                conn.close()
                # Chuyển sang format giống review_queue (không có predicted → dùng actual)
                for r in ti_rows:
                    rows.append({
                        "predicted_depth":  float(r["actual_depth"]),
                        "predicted_level":  r["actual_level"],
                        "confidence":       0.75,   # giá trị trung tính
                        "actual_depth":     float(r["actual_depth"]),
                        "actual_level":     r["actual_level"],
                        "features":         None,
                        "timestamp":        r["timestamp"],
                    })
                log.info(f"[AiLearner] training_images.db: {len(ti_rows)} manual images")
            except Exception as e:
                log.warning(f"[AiLearner] training_images.db read failed: {e}")

        log.info(f"[AiLearner] Tổng cases để train: {len(rows)}")
        return rows

    def _save_cache(self):
        payload = {
            "version":    "1.0",
            "saved_at":   datetime.now().isoformat(),
            "knn":        self.knn.to_dict(),
            "calibrator": self.calibrator.to_dict(),
            "bias":       self.bias.to_dict(),
        }
        self.cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info(f"[AiLearner] Cache saved → {self.cache_path}")

    def _load_cache(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))

            # Kiểm tra cache còn mới không (nếu DB mới hơn cache → retrain)
            if self.review_db.exists():
                cache_time = datetime.fromisoformat(payload.get("saved_at", "2000-01-01"))
                db_mtime   = datetime.fromtimestamp(self.review_db.stat().st_mtime)
                if db_mtime > cache_time:
                    log.info("[AiLearner] DB mới hơn cache → retrain")
                    return False

            self.knn.from_dict(payload["knn"])
            self.calibrator.from_dict(payload["calibrator"])
            self.bias.from_dict(payload["bias"])
            self._trained = True
            log.info(f"[AiLearner] Loaded from cache ({self.knn.n_cases} cases)")
            return True
        except Exception as e:
            log.warning(f"[AiLearner] Cache load failed: {e}")
            return False


# ── Integration helper: wrap ReferenceFloodResult ────────────────────────────

def apply_ai_correction(result, image_features: dict) -> Tuple[object, dict]:
    """
    Convenience wrapper: nhận ReferenceFloodResult, trả về (result_modified, correction_info).

    Dùng trong live_predict:
        from learning.ai_learner import apply_ai_correction
        result, ai_info = apply_ai_correction(result, features)
    """
    al = AiLearner()

    pred_depth = float(getattr(result, "water_height_cm", 0) or 0)
    pred_level = str(getattr(result, "flood_level", "NO_FLOOD") or "NO_FLOOD")
    conf       = float(getattr(result, "confidence", 0) or 0)

    correction = al.correct(
        predicted_depth=pred_depth,
        predicted_level=pred_level,
        confidence=conf,
        image_features=image_features,
    )

    if correction["ai_active"]:
        # Áp dụng correction vào result object
        if hasattr(result, "water_height_cm"):
            result.water_height_cm = correction["depth_cm"]
        if hasattr(result, "flood_level"):
            result.flood_level = correction["level"]
        if hasattr(result, "confidence"):
            result.confidence = correction["confidence_cal"]

    return result, correction


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_stats(stats: dict):
    ai_on = stats["ai_active"]
    status = "✅ ĐANG HOẠT ĐỘNG" if ai_on else f"⏳ CẦN THÊM DATA ({stats['n_reviewed_cases']}/{stats['min_cases_needed']} cases)"

    print(f"\n{'='*60}")
    print(f"  AI Learner — {status}")
    print(f"{'='*60}")
    print(f"  Reviewed cases : {stats['n_reviewed_cases']}")
    if stats.get('trained_at'):
        print(f"  Trained at     : {stats['trained_at'][:19]}")

    print(f"\n  📐 k-NN Correction Engine")
    print(f"     Training points: {stats['knn']['n_training_points']}")
    print(f"     k neighbors    : {stats['knn']['k_neighbors']}")

    print(f"\n  🎯 Confidence Calibrator")
    print(f"     Cases used : {stats['calibrator']['n_cases']}")
    lvls = stats['calibrator']['levels_with_data']
    print(f"     Levels     : {', '.join(lvls) if lvls else 'chưa có'}")

    print(f"\n  ⚠️  Level Bias Corrector")
    biases = stats['bias_corrector']['biases_found']
    if biases:
        for b in biases:
            print(f"     {b['predicted']:12s} → {b['actual']:12s}  "
                  f"{b['bias_pct']:.0f}% ({b['total_cases']} cases)")
    else:
        print(f"     Chưa phát hiện bias rõ ràng")

    print()


if __name__ == "__main__":
    import argparse
    import sys

    # Thêm root vào path
    _root = Path(__file__).parent.parent
    sys.path.insert(0, str(_root))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    p = argparse.ArgumentParser(description="AI Learner")
    p.add_argument("--stats",  action="store_true", help="Hiển thị thống kê")
    p.add_argument("--train",  action="store_true", help="Train lại model từ DB")
    p.add_argument("--reset",  action="store_true", help="Xóa cache, train mới")
    p.add_argument("--test",   action="store_true", help="Test correction với dummy data")
    args = p.parse_args()

    al = AiLearner()

    if args.reset:
        al.invalidate()
        print("✓ Cache đã xóa")
        result = al.train(force=True)
        print(f"✓ Trained: {result}")

    elif args.train:
        result = al.train(force=True)
        print(f"✓ Train result: {result}")

    elif args.test:
        dummy_features = {
            "brightness": 90, "blur_score": 150, "aspect_ratio": 1.5,
            "is_night": False, "num_reference_objects": 2, "num_people": 1,
            "has_pose": True, "water_level_pct": 0.4,
        }
        corr = al.correct(
            predicted_depth=80.0,
            predicted_level="WAIST",
            confidence=0.55,
            image_features=dummy_features,
        )
        print(json.dumps(corr, indent=2, ensure_ascii=False))

    else:
        _print_stats(al.get_stats())
