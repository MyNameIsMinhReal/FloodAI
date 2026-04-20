# -*- coding: utf-8 -*-
"""
agent/flood_agent.py
====================
FloodAgent — agent phân tích lũ lụt thuần nội bộ, không dùng external LLM API.

Kiến trúc v2 (nâng cấp toàn diện):
  AgentConfig     — structured config, type-safe (thay raw Dict)
  EventBus        — event system: on("low_confidence", handler)
  FeedbackParser  — hiểu ý kiến người dùng (tiếng Việt + English)
  AgentMemory     — bộ nhớ session + lịch sử correction + persistent JSON
                    + online calibration bias
  PipelineTools   — tool registry với calibration + simulation mode
  PolicyEngine    — scoring-based decision tree (thay hard-code if-else)
  Planner         — multi-step planning (adjust → rerun → compare → decide)
  FloodAgent      — điều phối toàn bộ + confidence-aware + self-reflection
                    + context-aware response + decision trace logging
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("flood_agent")

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT STORAGE PATH
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_DIR  = Path(__file__).parent / "_agent_memory"
_MEMORY_FILE = _AGENT_DIR / "long_term_memory.json"


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """
    Structured config cho FloodAgent — type-safe, dễ maintain.
    Thay thế raw Dict trước đây.
    """
    flood_threshold:   float = 0.45
    yolo_model:        str   = "yolov8n.pt"
    depth_model:       str   = "depth-anything/Depth-Anything-V2-Small-hf"
    dino_model:        str   = "IDEA-Research/grounding-dino-tiny"
    pose_model:        str   = "ultralytics/assets"
    output_dir:        str   = "output"
    depth_chunk_size:  int   = 4
    yolo_conf:         float = 0.35
    use_dino:          bool  = True
    use_pose:          bool  = True
    use_segformer:     bool  = True
    # Confidence thresholds cho decision-making
    conf_auto_confirm: float = 0.80   # trên ngưỡng này → tự xác nhận
    conf_ask_user:     float = 0.50   # 0.5–0.8 → hỏi user
    conf_auto_rerun:   float = 0.30   # dưới ngưỡng này → tự rerun
    # Simulation
    sim_thresholds: List[float] = field(default_factory=lambda: [0.30, 0.40, 0.50, 0.60, 0.70])
    # Delta comparison
    delta_min_significant_cm: float = 5.0   # |delta| < 5cm → "không đổi đáng kể"

    @classmethod
    def from_dict(cls, d: Dict) -> "AgentConfig":
        """Tạo từ dict (tương thích với config.yaml cũ)."""
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

    def to_dict(self) -> Dict:
        return asdict(self)

    # Dict-like access để tương thích code cũ
    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def update(self, d: Dict):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, val):
        setattr(self, key, val)


@dataclass
class ParsedFeedback:
    """Kết quả parse ý kiến người dùng."""
    intent:      str            # confirm | increase | decrease | no_flood | rerun
                                # | upload | status | help | unknown
    depth_hint:  Optional[float] = None   # cm nếu user nói "khoảng 50cm"
    level_hint:  Optional[str]   = None   # flood level nếu user nói "ngang eo"
    magnitude:   str = "medium"           # small | medium | large
    raw:         str = ""
    image_id:    Optional[int] = None     # ID ảnh user muốn sửa: "ảnh #2 sai"


@dataclass
class ToolResult:
    """Kết quả trả về từ một tool."""
    tool:    str
    success: bool
    data:    Dict[str, Any] = field(default_factory=dict)
    error:   str = ""


@dataclass
class PlanStep:
    """Một bước trong multi-step plan."""
    tool:        str
    kwargs:      Dict = field(default_factory=dict)
    description: str = ""
    optional:    bool = False   # nếu True, thất bại không dừng plan


@dataclass
class Plan:
    """Multi-step plan cho agent."""
    steps: List[PlanStep]
    goal:  str = ""


@dataclass
class Action:
    """Hành động mà agent sẽ thực thi (single-step hoặc plan)."""
    type:         str           # analyze | adjust | adjust_rerun | queue | retrain
                                # | confirm | clarify | status | help | simulate | plan
    tool_calls:   List[Dict]    = field(default_factory=list)
    plan:         Optional[Plan] = None
    message_hint: str = ""
    decision_log: Dict = field(default_factory=dict)  # scores + reasons


@dataclass
class AgentResponse:
    """Response trả về cho UI / caller."""
    message:      str
    success:      bool   = True
    action:       str    = ""
    results:      List[Dict] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    rerun:        bool   = False
    memory:       Dict   = field(default_factory=dict)
    decision_log: Dict   = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "message":      self.message,
            "success":      self.success,
            "action":       self.action,
            "results":      self.results,
            "tool_results": [asdict(t) for t in self.tool_results],
            "rerun":        self.rerun,
            "memory":       self.memory,
            "decision_log": self.decision_log,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EVENT BUS
# ─────────────────────────────────────────────────────────────────────────────

class EventBus:
    """
    Event system đơn giản: on("event", handler) / emit("event", data).

    Dùng để mở rộng behavior của agent mà không sửa core logic.
    Ví dụ:
        bus.on("low_confidence",  lambda d: queue_for_review(d))
        bus.on("user_correction", lambda d: trigger_calibration(d))
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def on(self, event: str, handler: Callable) -> None:
        """Đăng ký handler cho event."""
        with self._lock:
            self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """Huỷ đăng ký handler."""
        with self._lock:
            if event in self._handlers:
                self._handlers[event] = [h for h in self._handlers[event] if h is not handler]

    def emit(self, event: str, data: Optional[Dict] = None) -> None:
        """Phát event — gọi tất cả handler đã đăng ký."""
        data = data or {}
        with self._lock:
            handlers = list(self._handlers.get(event, []))
        for h in handlers:
            try:
                h(data)
            except Exception as exc:
                log.warning(f"[EventBus] Handler lỗi ở event '{event}': {exc}")

    def events(self) -> List[str]:
        with self._lock:
            return list(self._handlers.keys())


# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK PARSER
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackParser:
    """
    Hiểu ý kiến người dùng — không cần LLM.

    Hỗ trợ:
      - Tiếng Việt + English
      - Trích xuất số: "khoảng 50cm", "~0.5m", "1 mét"
      - Trích xuất level: "ngang gối", "ngập eo", "waist deep"
      - Magnitude: "một chút", "nhiều", "rất cao"
    """

    _INTENTS: Dict[str, List[str]] = {
        "confirm": [
            "đúng rồi", "chính xác", "ok", "đúng", "ừ", "yes", "correct",
            "đúng vậy", "chuẩn", "chuẩn rồi", "chính xác rồi", "tốt lắm",
        ],
        "increase": [
            "cao hơn", "sâu hơn", "nhiều hơn", "higher", "deeper", "more",
            "thêm", "tăng", "lên", "nước cao hơn", "ngập nhiều hơn",
        ],
        "decrease": [
            "thấp hơn", "nông hơn", "ít hơn", "lower", "shallower", "less",
            "giảm", "xuống", "bớt", "nước thấp hơn", "ngập ít hơn",
        ],
        "no_flood": [
            "không có lũ", "không ngập", "khô ráo", "bình thường",
            "no flood", "dry", "không có nước", "không lũ",
        ],
        "rerun": [
            "chạy lại", "thử lại", "retry", "rerun", "phân tích lại",
            "chạy lại pipeline", "làm lại",
        ],
        "simulate": [
            "mô phỏng", "thử nhiều", "simulate", "thử các ngưỡng", "so sánh ngưỡng",
        ],
        "status": [
            "status", "trạng thái", "thông tin", "memory", "bộ nhớ", "info",
        ],
        "help": [
            "help", "giúp", "hướng dẫn", "hướng dẫn sử dụng", "cách dùng",
        ],
        "calibrate": [
            "hiệu chỉnh", "calibrate", "điều chỉnh bias", "sửa sai lệch",
        ],
        "save_response": [
            "lưu cách trả lời này", "nhớ cách này", "lưu lại cách này",
            "trả lời như vậy nữa", "tôi thích cách này", "phong cách này tốt",
            "save this response", "save this style", "remember this format",
            "giữ cách này", "cách này hay", "lưu phong cách",
            "lưu kiểu trả lời", "nhớ phong cách này",
        ],
        "greeting": [
            "hello", "hi", "hey", "xin chào", "chào", "chào bạn",
            "helo", "good morning", "good afternoon", "good evening",
            "alo", "yo", "hii", "helo", "howdy", "sup",
        ],
    }

    _LEVEL_KEYWORDS: Dict[str, str] = {
        "mắt cá":   "ANKLE",  "cổ chân": "ANKLE",  "ankle": "ANKLE",
        "đầu gối":  "KNEE",   "gối":     "KNEE",   "knee":  "KNEE",
        "ngang eo":  "WAIST", "hông":    "WAIST",  "waist": "WAIST",
        "ngực":     "CHEST",  "chest":   "CHEST",
        "ngập hoàn toàn": "SUBMERGED", "submerged": "SUBMERGED",
        "vũng nước": "PUDDLE", "puddle": "PUDDLE",
    }

    _MAGNITUDE: Dict[str, List[str]] = {
        "small":  ["một chút", "một tí", "nhẹ", "slightly", "a bit", "chút"],
        "large":  ["nhiều", "rất", "lắm", "a lot", "much", "significantly", "đáng kể"],
    }

    _NUM_RE = re.compile(
        r'(?:khoảng|approximately|~|about)?\s*'
        r'(\d+(?:[.,]\d+)?)\s*'
        r'(cm|m|mét|meter|meters|centimeter|centimeters)',
        re.IGNORECASE,
    )

    @staticmethod
    def _strip_diacritics(text: str) -> str:
        """Chuyển tiếng Việt có dấu → không dấu để so sánh linh hoạt.
        Ví dụ: 'trạng thái' → 'trang thai', 'chạy lại' → 'chay lai'.
        """
        import unicodedata
        # đ/Đ không decompose qua NFD, xử lý riêng trước
        text = text.replace('đ', 'd').replace('Đ', 'D')
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')

    def parse(self, text: str) -> ParsedFeedback:
        t      = text.lower().strip()
        t_norm = self._strip_diacritics(t)   # bản không dấu để match fallback

        intent = "unknown"
        for name, kws in self._INTENTS.items():
            kws_norm = [self._strip_diacritics(kw) for kw in kws]
            # Khớp với dấu: substring match
            if any(kw in t for kw in kws):
                intent = name
                break
            # Khớp không dấu: word-boundary để tránh false positive
            # (vd: "ừ" → "u" không được match "nuoc")
            if any(re.search(r'\b' + re.escape(kn) + r'\b', t_norm) for kn in kws_norm):
                intent = name
                break

        depth_hint: Optional[float] = None
        m = self._NUM_RE.search(t)
        if m:
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2).lower()
            depth_hint = val if unit.startswith("cm") else val * 100

        level_hint: Optional[str] = None
        for kw, lv in self._LEVEL_KEYWORDS.items():
            kw_norm = self._strip_diacritics(kw)
            if kw in t or kw_norm in t_norm:
                level_hint = lv
                break

        magnitude = "medium"
        for mag, kws in self._MAGNITUDE.items():
            kws_norm = [self._strip_diacritics(kw) for kw in kws]
            if any(kw in t for kw in kws) or any(kw in t_norm for kw in kws_norm):
                magnitude = mag
                break

        # Detect image ID: "ảnh #2", "ảnh số 2", "ảnh 2", "image 2", "#2"
        image_id: Optional[int] = None
        _id_m = re.search(
            r'(?:(?:anh|ảnh|image|img)\s*(?:so|số|#)?\s*#?\s*(\d+)'
            r'|#\s*(\d+)\b)',
            t_norm, re.IGNORECASE,
        )
        if _id_m:
            image_id = int(_id_m.group(1) or _id_m.group(2))

        return ParsedFeedback(
            intent=intent,
            depth_hint=depth_hint,
            level_hint=level_hint,
            magnitude=magnitude,
            raw=text,
            image_id=image_id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT MEMORY  (enhanced: persistence + calibration)
# ─────────────────────────────────────────────────────────────────────────────

class AgentMemory:
    """
    Bộ nhớ của agent:
      session_state     — trạng thái hiện tại (kết quả, ảnh,…)
      chat_history      — lịch sử hội thoại (session)
      correction_log    — các lần user sửa prediction (persistent)
      error_counter     — thống kê loại lỗi (persistent)
      action_history    — lịch sử action + outcome → self-reflection
      global_bias       — calibration bias tích lũy từ correction_log
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self.chat_history:   List[Dict] = []
        self.correction_log: List[Dict] = []
        self.error_counter:  Counter    = Counter()
        self.action_history: List[Dict] = []
        self.image_history:  List[Dict] = []   # lịch sử ảnh đã phân tích với ID
        self.last_result:    Optional[Dict] = None
        self.last_images:    List[Path] = []
        self.last_action_type: str = "analyze"
        self.n_sessions:     int = 0
        self.response_patterns: Dict[str, Dict] = {}
        self._lock = threading.RLock()   # RLock cho phép re-entrant (to_summary gọi các method khác cũng dùng lock)

        # Long-term storage
        self._persist_path = persist_path or _MEMORY_FILE
        self._load_from_disk()

    # ── Write ────────────────────────────────────────────────────

    def add_message(self, role: str, content: str, meta: Optional[Dict] = None):
        with self._lock:
            self.chat_history.append({
                "role":      role,
                "content":   content,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "meta":      meta or {},
            })

    def add_correction(self, original_level: str, corrected_level: str,
                       original_depth: float, corrected_depth: Optional[float],
                       feedback: str):
        with self._lock:
            entry = {
                "timestamp":       datetime.now().isoformat(timespec="seconds"),
                "original_level":  original_level,
                "corrected_level": corrected_level,
                "original_depth":  original_depth,
                "corrected_depth": corrected_depth,
                "feedback":        feedback,
            }
            self.correction_log.append(entry)
        # Lưu xuống disk ngay sau mỗi correction để không mất data
        self._save_to_disk()

    def record_error(self, error_type: str):
        with self._lock:
            self.error_counter[error_type] += 1

    def record_action_outcome(self, action_type: str, before: Optional[Dict],
                               after: Optional[Dict], success: bool):
        """Ghi lại outcome của action → dùng cho self-reflection."""
        with self._lock:
            entry: Dict = {
                "timestamp":   datetime.now().isoformat(timespec="seconds"),
                "action":      action_type,
                "success":     success,
            }
            if before and after:
                old_conf = float(before.get("confidence", 0) or 0)
                new_conf = float(after.get("confidence", 0) or 0)
                entry["conf_delta"]  = round(new_conf - old_conf, 3)
                entry["depth_delta"] = round(
                    float(after.get("water_height_cm", 0) or 0) -
                    float(before.get("water_height_cm", 0) or 0), 1
                )
                entry["improved"] = new_conf > old_conf
            self.action_history.append(entry)

    def set_last_result(self, result: Dict, images: Optional[List[Path]] = None):
        with self._lock:
            self.last_result = result
            if images:
                self.last_images = images

    # ── Image History ────────────────────────────────────────────

    def add_image_to_history(self, result: Dict,
                              images: Optional[List[Path]] = None) -> int:
        """Gán ID tuần tự cho ảnh, lưu vào history. Trả về image_id (1-based)."""
        with self._lock:
            image_id = len(self.image_history) + 1
            entry = {
                "id":        image_id,
                "filename":  Path(result.get("original_path", "")).name
                             or f"anh-{image_id}",
                "result":    dict(result),
                "images":    [str(p) for p in (images or [])],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            self.image_history.append(entry)
        return image_id

    def get_image_by_id(self, image_id: int) -> Optional[Dict]:
        """Tra cứu entry theo ID."""
        with self._lock:
            for entry in self.image_history:
                if entry["id"] == image_id:
                    return entry
        return None

    def get_image_history_summary(self, n: int = 10) -> List[Dict]:
        """Trả về n ảnh gần nhất để hiển thị."""
        with self._lock:
            recent = list(self.image_history[-n:])
        return [
            {
                "id":       e["id"],
                "filename": e["filename"],
                "level":    e["result"].get("flood_level", "UNKNOWN"),
                "depth_cm": e["result"].get("water_height_cm", 0),
                "conf_pct": round(float(e["result"].get("confidence", 0) or 0) * 100),
                "timestamp": e["timestamp"],
            }
            for e in recent
        ]

    # ── Calibration ──────────────────────────────────────────────

    def get_calibration_bias(self) -> Dict:
        """
        Tính bias từ correction_log:
          bias = mean(corrected_depth - original_depth)

        Dùng để cộng vào prediction: final = predicted + bias
        """
        with self._lock:
            log_copy = list(self.correction_log)

        if len(log_copy) < 3:
            return {"depth_bias_cm": 0.0, "by_level": {}, "n_samples": 0, "reliable": False}

        deltas: List[float] = []
        by_level: Dict[str, List[float]] = {}

        for c in log_copy:
            orig = c.get("original_depth", 0) or 0
            corr = c.get("corrected_depth")
            if corr is not None:
                delta = float(corr) - float(orig)
                deltas.append(delta)
                lvl = c.get("original_level", "UNKNOWN")
                by_level.setdefault(lvl, []).append(delta)

        if not deltas:
            return {"depth_bias_cm": 0.0, "by_level": {}, "n_samples": 0, "reliable": False}

        bias = sum(deltas) / len(deltas)
        level_bias = {k: round(sum(v) / len(v), 2) for k, v in by_level.items()}

        # Chỉ reliable nếu có ≥5 samples và bias đáng kể (>3cm)
        reliable = len(deltas) >= 5 and abs(bias) > 3.0

        return {
            "depth_bias_cm": round(bias, 2),
            "by_level":      level_bias,
            "n_samples":     len(deltas),
            "reliable":      reliable,
        }

    def apply_calibration(self, result: Dict) -> Dict:
        """Áp dụng calibration bias lên 1 kết quả dự đoán."""
        bias_info = self.get_calibration_bias()
        if not bias_info["reliable"]:
            return result

        result = dict(result)
        level = result.get("flood_level", "UNKNOWN")
        # Ưu tiên level-specific bias, fallback về global bias
        bias = bias_info["by_level"].get(level, bias_info["depth_bias_cm"])
        old_depth = float(result.get("water_height_cm", 0) or 0)
        result["water_height_cm"] = round(max(0.0, old_depth + bias), 1)
        result["calibration_applied"] = True
        result["calibration_bias_cm"] = bias
        return result

    def get_action_performance(self) -> Dict:
        """Thống kê hiệu quả của từng loại action → self-reflection."""
        with self._lock:
            hist = list(self.action_history)

        perf: Dict[str, Dict] = {}
        for entry in hist:
            act = entry["action"]
            if act not in perf:
                perf[act] = {"total": 0, "improved": 0, "failed": 0}
            perf[act]["total"] += 1
            if not entry.get("success"):
                perf[act]["failed"] += 1
            elif entry.get("improved", False):
                perf[act]["improved"] += 1

        return perf

    # ── Response Pattern Learning ────────────────────────────────

    def save_response_pattern(self, context_key: str, template: str, raw_text: str) -> None:
        """
        Lưu template phản hồi cho ngữ cảnh cụ thể.

        context_key: e.g. "analyze_KNEE_mid" (action_level_confidence_band)
        template: phiên bản đã trừu tượng hóa của response (có placeholder)
        raw_text: response gốc để tham khảo
        """
        with self._lock:
            existing = self.response_patterns.get(context_key, {})
            self.response_patterns[context_key] = {
                "template":   template,
                "raw_text":   raw_text,
                "saved_at":   datetime.now().isoformat(timespec="seconds"),
                "times_used": existing.get("times_used", 0),
            }
        self._save_to_disk()
        log.info(f"[AgentMemory] Saved response pattern for context: {context_key}")

    def find_matching_pattern(self, context_key: str) -> Optional[Dict]:
        """
        Tìm pattern phù hợp cho ngữ cảnh.
        Thử exact match trước, sau đó fuzzy match (bỏ confidence band).
        """
        with self._lock:
            # Exact match
            if context_key in self.response_patterns:
                p = self.response_patterns[context_key]
                self.response_patterns[context_key]["times_used"] = p.get("times_used", 0) + 1
                return p

            # Fuzzy match: bỏ phần confidence band (phần cuối sau dấu _)
            parts = context_key.rsplit("_", 1)
            if len(parts) == 2:
                fuzzy_key = parts[0]
                for k, v in self.response_patterns.items():
                    if k.startswith(fuzzy_key):
                        v["times_used"] = v.get("times_used", 0) + 1
                        return v
        return None

    def get_response_patterns_summary(self) -> List[Dict]:
        """Trả về danh sách các pattern đã lưu (để hiển thị)."""
        with self._lock:
            return [
                {
                    "context": k,
                    "saved_at": v.get("saved_at", ""),
                    "times_used": v.get("times_used", 0),
                    "preview": (v.get("raw_text") or "")[:80] + "…",
                }
                for k, v in self.response_patterns.items()
            ]

    # ── Read ─────────────────────────────────────────────────────

    def get_recent_history(self, n: int = 10) -> List[Dict]:
        with self._lock:
            return list(self.chat_history[-n:])

    def get_correction_trend(self) -> str:
        """Phân tích xu hướng correction gần nhất (underestimate / overestimate)."""
        with self._lock:
            recent = self.correction_log[-5:]
        if not recent:
            return "none"
        under = sum(1 for c in recent
                    if (c.get("corrected_depth") or 0) > (c.get("original_depth") or 0))
        over  = len(recent) - under
        if under > over:
            return "underestimate"
        if over > under:
            return "overestimate"
        return "mixed"

    def to_summary(self) -> Dict:
        with self._lock:
            lr = self.last_result or {}
            bias = self.get_calibration_bias()
            trend = self.get_correction_trend()
            perf  = self.get_action_performance()
            context_lines = []
            if lr:
                conf_pct = round(float(lr.get("confidence", 0) or 0) * 100)
                context_lines.append(
                    f"Kết quả gần nhất: level={lr.get('flood_level','?')}, "
                    f"depth={lr.get('water_height_cm','?')}cm, conf={conf_pct}%"
                )
            if self.correction_log:
                last_c = self.correction_log[-1]
                context_lines.append(
                    f"Lần sửa cuối: '{last_c['feedback'][:60]}' "
                    f"({last_c['original_level']} → {last_c['corrected_level']})"
                )
            if bias["reliable"]:
                context_lines.append(
                    f"Calibration bias: {bias['depth_bias_cm']:+.1f}cm "
                    f"({bias['n_samples']} mẫu, xu hướng: {trend})"
                )
            top_errors = self.error_counter.most_common(3)
            if top_errors:
                context_lines.append(
                    "Lỗi thường gặp: " +
                    ", ".join(f"{k}({v})" for k, v in top_errors)
                )
            return {
                "n_messages":    len(self.chat_history),
                "n_corrections": len(self.correction_log),
                "n_sessions":    self.n_sessions,
                "last_result":   lr,
                "context":       "\n".join(context_lines) or "Chưa có lịch sử.",
                "calibration":   bias,
                "correction_trend": trend,
                "action_performance": perf,
            }

    # ── Persistence ──────────────────────────────────────────────

    def _save_to_disk(self):
        """Lưu correction_log, error_counter và response_patterns xuống JSON."""
        try:
            _AGENT_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "correction_log":   self.correction_log,
                "error_counter":    dict(self.error_counter),
                "action_history":   self.action_history[-100:],
                "image_history":    self.image_history[-200:],   # giữ 200 ảnh gần nhất
                "response_patterns": self.response_patterns,
                "saved_at":         datetime.now().isoformat(timespec="seconds"),
            }
            tmp = self._persist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._persist_path)
        except Exception as e:
            log.warning(f"[AgentMemory] Không lưu được xuống disk: {e}")

    def _load_from_disk(self):
        """Load long-term memory khi khởi tạo."""
        try:
            if self._persist_path.exists():
                data = json.loads(self._persist_path.read_text(encoding="utf-8"))
                self.correction_log    = data.get("correction_log", [])
                self.error_counter     = Counter(data.get("error_counter", {}))
                self.action_history    = data.get("action_history", [])
                self.image_history     = data.get("image_history", [])
                self.response_patterns = data.get("response_patterns", {})
                n_patterns = len(self.response_patterns)
                log.info(
                    f"[AgentMemory] Loaded {len(self.correction_log)} corrections, "
                    f"{len(self.action_history)} action records, "
                    f"{n_patterns} response patterns from disk"
                )
        except Exception as e:
            log.warning(f"[AgentMemory] Không load được từ disk: {e}")

    def clear_long_term(self):
        """Xóa bộ nhớ dài hạn (correction_log + response_patterns)."""
        with self._lock:
            self.correction_log.clear()
            self.error_counter.clear()
            self.action_history.clear()
            self.response_patterns.clear()
        self._save_to_disk()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE TOOLS  (enhanced: calibration tool + simulation)
# ─────────────────────────────────────────────────────────────────────────────

class PipelineTools:
    """
    Các "tool" mà agent có thể gọi.

    Tool mới so với v1:
      calibrate_result — áp dụng calibration bias lên kết quả
      simulate         — thử nhiều threshold, chọn tốt nhất
    """

    def __init__(self, config: AgentConfig, memory: AgentMemory):
        self.cfg    = config
        self.memory = memory
        self._registry: Dict[str, Callable] = {}
        self._estimator = None

        self._register("analyze",           self._tool_analyze)
        self._register("adjust_threshold",  self._tool_adjust_threshold)
        self._register("rerun",             self._tool_rerun)
        self._register("queue_review",      self._tool_queue_review)
        self._register("trigger_retrain",   self._tool_trigger_retrain)
        self._register("system_status",     self._tool_system_status)
        self._register("calibrate_result",  self._tool_calibrate_result)  # NEW
        self._register("simulate",          self._tool_simulate)           # NEW

    def _register(self, name: str, fn: Callable):
        self._registry[name] = fn

    def call(self, tool_name: str, **kwargs) -> ToolResult:
        fn = self._registry.get(tool_name)
        if fn is None:
            return ToolResult(tool=tool_name, success=False,
                              error=f"Tool '{tool_name}' không tồn tại")
        try:
            return fn(**kwargs)
        except Exception as e:
            log.exception(f"[Tool:{tool_name}] lỗi")
            return ToolResult(tool=tool_name, success=False, error=str(e))

    # ── Tool: analyze images ─────────────────────────────────────

    def _tool_analyze(self, images: List[Path], extra_cfg: Optional[Dict] = None) -> ToolResult:
        t0 = time.time()
        try:
            estimator = self._get_estimator()
            results_raw = estimator.analyze_batch(
                [str(p) for p in images],
                chunk_size=self.cfg.depth_chunk_size,
            )
            results = []
            for r in results_raw:
                result = {
                    "filename":         getattr(r, "filename", ""),
                    "original_path":    getattr(r, "original_path", ""),
                    "overlay_path":     getattr(r, "overlay_path", ""),
                    "flood_level":      getattr(r, "flood_level", "UNKNOWN"),
                    "flood_level_desc": getattr(r, "flood_level_desc", ""),
                    "water_height_cm":  round(float(getattr(r, "water_height_cm", 0) or 0), 1),
                    "confidence":       round(float(getattr(r, "confidence", 0) or 0), 3),
                    "detected_objects": getattr(r, "detected_objects", []) or [],
                    "notes":            getattr(r, "notes", ""),
                }
                # Áp dụng calibration bias ngay tại đây
                result = self.memory.apply_calibration(result)
                results.append(result)

            return ToolResult(
                tool="analyze", success=True,
                data={"results": results, "duration_s": round(time.time() - t0, 1)},
            )
        except Exception as e:
            return ToolResult(tool="analyze", success=False, error=str(e))

    # ── Tool: adjust threshold ───────────────────────────────────

    def _tool_adjust_threshold(self, direction: str, amount: float = 0.05) -> ToolResult:
        current = self.cfg.flood_threshold
        if direction == "up":
            new_val = min(0.95, current + amount)
        elif direction == "down":
            new_val = max(0.05, current - amount)
        else:
            new_val = current
        self.cfg.flood_threshold = new_val
        log.info(f"[Decision] adjust_threshold {direction}: {current:.2f} → {new_val:.2f}")
        return ToolResult(
            tool="adjust_threshold", success=True,
            data={"key": "flood_threshold", "old": current, "new": new_val, "direction": direction},
        )

    # ── Tool: rerun pipeline ─────────────────────────────────────

    def _tool_rerun(self, images: List[Path], new_params: Optional[Dict] = None) -> ToolResult:
        if new_params:
            self.cfg.update(new_params)
        return self._tool_analyze(images=images)

    # ── Tool: queue review ───────────────────────────────────────

    def _tool_queue_review(self, result: Dict, reason: str = "agent_flagged") -> ToolResult:
        try:
            from learning.active_learner import ActiveLearnerV2, ReviewCase
            learner = ActiveLearnerV2()
            case = ReviewCase(
                image_path      = result.get("original_path", ""),
                predicted_depth = result.get("water_height_cm"),
                predicted_level = result.get("flood_level"),
                confidence      = result.get("confidence", 0.0),
                review_reason   = reason,
            )
            learner.add_to_queue(case)
            learner.close()
            return ToolResult(tool="queue_review", success=True, data={"reason": reason})
        except Exception as e:
            return ToolResult(tool="queue_review", success=False, error=str(e))

    # ── Tool: trigger retrain ────────────────────────────────────

    def _tool_trigger_retrain(self) -> ToolResult:
        def _work():
            try:
                from learning.ai_learner import AiLearner
                al = AiLearner()
                al.invalidate()
                result = al.train()
                log.info(f"[Agent:retrain] xong — {result.get('n_cases',0)} cases")
            except Exception as e:
                log.warning(f"[Agent:retrain] lỗi: {e}")
        threading.Thread(target=_work, daemon=True, name="agent-retrain").start()
        return ToolResult(tool="trigger_retrain", success=True,
                          data={"message": "Retrain đang chạy trong nền"})

    # ── Tool: system status ──────────────────────────────────────

    def _tool_system_status(self) -> ToolResult:
        try:
            from utils.memory_manager import get_memory_usage
            mem = get_memory_usage()
        except Exception:
            mem = {}
        cfg_snap = self.cfg.to_dict()
        bias     = self.memory.get_calibration_bias()
        perf     = self.memory.get_action_performance()
        return ToolResult(
            tool="system_status", success=True,
            data={
                "cfg":              cfg_snap,
                "memory":           mem,
                "calibration_bias": bias,
                "action_perf":      perf,
                "timestamp":        datetime.now().isoformat(timespec="seconds"),
            },
        )

    # ── Tool: calibrate_result (NEW) ─────────────────────────────

    def _tool_calibrate_result(self, result: Dict) -> ToolResult:
        """Áp dụng calibration bias lên 1 kết quả."""
        bias_info = self.memory.get_calibration_bias()
        if not bias_info["reliable"]:
            return ToolResult(
                tool="calibrate_result", success=True,
                data={"result": result, "bias_applied": False,
                      "reason": f"Chưa đủ dữ liệu ({bias_info['n_samples']} mẫu < 5)"},
            )
        calibrated = self.memory.apply_calibration(result)
        return ToolResult(
            tool="calibrate_result", success=True,
            data={
                "result":       calibrated,
                "bias_applied": True,
                "bias_cm":      bias_info["depth_bias_cm"],
                "n_samples":    bias_info["n_samples"],
            },
        )

    # ── Tool: simulate (NEW) ─────────────────────────────────────

    def _tool_simulate(self, images: List[Path],
                       thresholds: Optional[List[float]] = None) -> ToolResult:
        """
        Thử nhiều threshold → chọn ngưỡng cho confidence trung bình cao nhất.
        Giống gradient descent / optimization loop.
        """
        thresholds = thresholds or self.cfg.sim_thresholds
        original_thresh = self.cfg.flood_threshold

        best_result: Optional[ToolResult] = None
        best_avg_conf = -1.0
        best_thresh   = original_thresh
        sim_log: List[Dict] = []

        for thresh in thresholds:
            self.cfg.flood_threshold = thresh
            tr = self._tool_analyze(images=images)
            if tr.success and tr.data.get("results"):
                results = tr.data["results"]
                avg_conf = sum(r.get("confidence", 0) for r in results) / len(results)
                entry = {"threshold": thresh, "avg_confidence": round(avg_conf, 3),
                         "n_images": len(results)}
                sim_log.append(entry)
                log.info(f"[Simulate] threshold={thresh:.2f} → avg_conf={avg_conf:.3f}")
                if avg_conf > best_avg_conf:
                    best_avg_conf = avg_conf
                    best_result   = tr
                    best_thresh   = thresh
            else:
                sim_log.append({"threshold": thresh, "avg_confidence": 0.0, "error": tr.error})

        # Áp dụng ngưỡng tốt nhất vĩnh viễn
        self.cfg.flood_threshold = best_thresh
        log.info(f"[Simulate] Best threshold: {best_thresh:.2f} (conf={best_avg_conf:.3f}). "
                 f"Original: {original_thresh:.2f}")

        if best_result is None:
            self.cfg.flood_threshold = original_thresh
            return ToolResult(tool="simulate", success=False, error="Không có kết quả hợp lệ")

        best_result.data["sim_log"]        = sim_log
        best_result.data["best_threshold"] = best_thresh
        best_result.data["original_threshold"] = original_thresh
        best_result.data["improvement"]    = round(best_avg_conf - 0.5, 3)  # vs baseline
        return ToolResult(tool="simulate", success=True, data=best_result.data)

    # ── Lazy estimator ───────────────────────────────────────────

    def _get_estimator(self):
        if self._estimator is None:
            from depth_analysis.reference_estimator import ReferenceEstimator
            self._estimator = ReferenceEstimator(
                yolo_model    = self.cfg.yolo_model,
                depth_model   = self.cfg.depth_model,
                output_dir    = Path(self.cfg.output_dir) / "_agent_tmp",
                conf_thresh   = self.cfg.yolo_conf,
                use_dino      = self.cfg.use_dino,
                dino_model    = self.cfg.dino_model,
                use_pose      = self.cfg.use_pose,
                pose_model    = self.cfg.pose_model,
                use_segformer = self.cfg.use_segformer,
            )
        return self._estimator


# ─────────────────────────────────────────────────────────────────────────────
# POLICY ENGINE  (scoring-based, thay thế ActionEngine)
# ─────────────────────────────────────────────────────────────────────────────

class PolicyEngine:
    """
    Quyết định action bằng scoring system — không còn hard-code if-else.

    Điểm số được tích lũy từ 4 nguồn:
      1. Intent của user (trọng số cao nhất)
      2. Lịch sử lỗi trong memory (error_counter)
      3. Confidence của kết quả gần nhất
      4. Hiệu quả action trong quá khứ (self-reflection)

    Action có điểm cao nhất được chọn.
    """

    _THRESH_AMOUNT = {"small": 0.03, "medium": 0.07, "large": 0.15}

    # Điểm cơ bản từ intent → action
    _INTENT_SCORES: Dict[str, Dict[str, float]] = {
        "increase":  {"adjust_up": 2.0, "rerun": 0.5},
        "decrease":  {"adjust_down": 2.0, "rerun": 0.5},
        "no_flood":  {"adjust_up": 1.5, "queue_review": 0.5},
        "confirm":   {"confirm": 3.0},
        "rerun":     {"rerun": 3.0},
        "simulate":  {"simulate": 3.0},
        "calibrate": {"calibrate": 3.0},
        "status":    {"status": 5.0},   # explicit → không override
        "help":      {"help": 5.0},
        "greeting":  {"greeting": 5.0},
        "unknown":   {"clarify": 1.0},
    }

    def score(self, parsed: ParsedFeedback, memory: AgentMemory,
              config: AgentConfig) -> Dict[str, float]:
        """Tính điểm cho từng action. Trả về dict {action: score}."""
        scores: Dict[str, float] = {
            "adjust_up":    0.0,
            "adjust_down":  0.0,
            "rerun":        0.0,
            "simulate":     0.0,
            "queue_review": 0.0,
            "confirm":      0.0,
            "calibrate":    0.0,
            "clarify":      0.0,
            "status":       0.0,
            "help":         0.0,
            "greeting":     0.0,
        }

        last = memory.last_result or {}
        conf = float(last.get("confidence", 0.5) or 0.5)
        ec   = memory.error_counter
        perf = memory.get_action_performance()
        trend = memory.get_correction_trend()

        # ── 1. Điểm từ intent ────────────────────────────────────
        intent_boosts = self._INTENT_SCORES.get(parsed.intent, {})
        for action, pts in intent_boosts.items():
            scores[action] = scores.get(action, 0.0) + pts

        # ── 2. Điểm từ error history ─────────────────────────────
        if ec["underestimate"] > 2:
            scores["adjust_up"]  += 0.5
        if ec["overestimate"] > 2:
            scores["adjust_down"] += 0.5
        # Nếu trend liên tục underestimate → gợi ý rerun mạnh hơn
        if trend == "underestimate" and ec["underestimate"] > 3:
            scores["adjust_up"] += 0.3
            scores["rerun"]     += 0.3

        # ── 3. Điểm từ confidence ────────────────────────────────
        if conf < config.conf_auto_rerun:      # < 0.3 → rerun mạnh
            scores["rerun"]        += 1.5
            scores["queue_review"] += 0.5
        elif conf < config.conf_ask_user:      # 0.3–0.5 → queue + rerun nhẹ
            scores["rerun"]        += 0.8
            scores["queue_review"] += 0.3
        elif conf > config.conf_auto_confirm:  # > 0.8 → confirm nhẹ
            scores["confirm"]      += 0.3

        # ── 4. Điểm từ self-reflection (action performance) ──────
        # Giảm điểm action hay thất bại trong quá khứ
        for action, stats in perf.items():
            if stats["total"] >= 3:
                fail_rate = stats["failed"] / stats["total"]
                if fail_rate > 0.5:
                    scores[action] = scores.get(action, 0.0) * (1.0 - fail_rate * 0.5)

        # ── 5. Calibration bonus ─────────────────────────────────
        bias = memory.get_calibration_bias()
        if bias["reliable"] and abs(bias["depth_bias_cm"]) > 5:
            scores["calibrate"] += 0.4

        log.info(
            f"[PolicyEngine] intent={parsed.intent} conf={conf:.2f} "
            f"trend={trend} "
            f"top_actions={sorted(scores.items(), key=lambda x: -x[1])[:3]}"
        )
        return scores

    def decide(self, parsed: ParsedFeedback, memory: AgentMemory,
                config: AgentConfig) -> Action:
        """Chọn action có điểm cao nhất → tạo Action object."""
        scores = self.score(parsed, memory, config)
        amount = self._THRESH_AMOUNT.get(parsed.magnitude, 0.07)
        last   = memory.last_result or {}

        # Chọn top action
        top_action = max(scores, key=lambda k: scores[k])
        top_score  = scores[top_action]

        decision_log = {
            "scores":     {k: round(v, 3) for k, v in scores.items()},
            "chosen":     top_action,
            "score":      round(top_score, 3),
            "intent":     parsed.intent,
            "confidence": float(last.get("confidence", 0) or 0),
        }

        log.info(f"[Decision] chosen={top_action} (score={top_score:.2f}) intent={parsed.intent}")

        # ── Xây dựng tool_calls theo action đã chọn ──────────────

        if top_action == "confirm":
            return Action(
                type="confirm",
                tool_calls=[{"tool": "queue_review",
                             "kwargs": {"result": last, "reason": "user_confirmed"}}],
                message_hint="✅ Đã xác nhận — kết quả được lưu vào hệ thống học.",
                decision_log=decision_log,
            )

        if top_action in ("adjust_up", "adjust_down"):
            direction = "up" if top_action == "adjust_up" else "down"
            calls = [{"tool": "adjust_threshold",
                      "kwargs": {"direction": direction, "amount": amount}}]
            # Nếu có ảnh và score rerun cũng cao → kết hợp rerun
            if memory.last_images and scores["rerun"] > 0.3:
                calls.append({"tool": "rerun",
                              "kwargs": {"images": memory.last_images}})
                return Action(type="adjust_rerun", tool_calls=calls,
                              decision_log=decision_log)
            return Action(type="adjust", tool_calls=calls, decision_log=decision_log)

        if top_action == "rerun":
            if memory.last_images:
                return Action(
                    type="analyze",
                    tool_calls=[{"tool": "rerun",
                                 "kwargs": {"images": memory.last_images}}],
                    message_hint="🔄 Đang chạy lại phân tích…",
                    decision_log=decision_log,
                )
            return Action(type="clarify",
                          message_hint="Chưa có ảnh nào. Hãy upload ảnh để phân tích.",
                          decision_log=decision_log)

        if top_action == "simulate":
            if memory.last_images:
                return Action(
                    type="simulate",
                    tool_calls=[{"tool": "simulate",
                                 "kwargs": {"images": memory.last_images}}],
                    message_hint="🧪 Đang mô phỏng nhiều ngưỡng để tìm cấu hình tốt nhất…",
                    decision_log=decision_log,
                )
            return Action(type="clarify",
                          message_hint="Cần có ảnh để chạy simulation.",
                          decision_log=decision_log)

        if top_action == "queue_review":
            return Action(
                type="queue",
                tool_calls=[{"tool": "queue_review",
                             "kwargs": {"result": last, "reason": "agent_low_conf"}}],
                message_hint="📋 Đã đưa vào hàng đợi review do độ tin cậy thấp.",
                decision_log=decision_log,
            )

        if top_action == "calibrate":
            return Action(
                type="calibrate",
                tool_calls=[{"tool": "calibrate_result",
                             "kwargs": {"result": last}}] if last else [],
                message_hint="🎯 Áp dụng hiệu chỉnh từ lịch sử correction…",
                decision_log=decision_log,
            )

        if top_action == "greeting":
            return Action(type="greeting", decision_log=decision_log)

        if top_action == "status":
            return Action(type="status", decision_log=decision_log)

        if top_action == "help":
            return Action(type="help", decision_log=decision_log)

        # clarify hoặc unknown
        return Action(
            type="clarify",
            message_hint=(
                "Bạn có thể nói:\n"
                '  "Đúng rồi" — xác nhận kết quả\n'
                '  "Nước thấp hơn" / "Nước cao hơn" — điều chỉnh\n'
                '  "Không có lũ ở đây" — không ngập\n'
                '  "Chạy lại" — phân tích lại ảnh\n'
                '  "Mô phỏng" — thử nhiều ngưỡng tự động\n'
                '  "Trạng thái" — xem thông tin hệ thống'
            ),
            decision_log=decision_log,
        )


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

_LEVEL_VI = {
    "NO_FLOOD":  "Không ngập",
    "PUDDLE":    "Vũng nước nhỏ (<15cm)",
    "ANKLE":     "Ngập mắt cá (15-40cm)",
    "KNEE":      "Ngập đầu gối (40-70cm)",
    "WAIST":     "Ngập ngang hông (70-120cm)",
    "CHEST":     "Ngập ngang ngực (120-200cm)",
    "SUBMERGED": "Ngập hoàn toàn (>200cm)",
    "UNKNOWN":   "Không xác định",
}

_LEVEL_EMOJI = {
    "NO_FLOOD": "✅", "PUDDLE": "💧", "ANKLE": "🌊",
    "KNEE": "🌊", "WAIST": "⚠️", "CHEST": "🚨", "SUBMERGED": "🆘",
    "UNKNOWN": "❓",
}


def _format_analysis_response(results: List[Dict], duration_s: float = 0,
                               calib_applied: bool = False) -> str:
    if not results:
        return "ℹ️ Không có kết quả phân tích."
    calib_note = "  _(đã hiệu chỉnh bias)_" if calib_applied else ""
    lines = [f"✅ Phân tích xong {len(results)} ảnh ({duration_s:.1f}s){calib_note}\n"]
    for i, r in enumerate(results[:5], 1):
        lvl    = r.get("flood_level", "UNKNOWN")
        emoji  = _LEVEL_EMOJI.get(lvl, "❓")
        img_id = r.get("image_id")
        name   = Path(r.get("original_path", "")).name or f"ảnh-{i}"
        id_tag = f" `[#ID {img_id}]`" if img_id else ""
        depth  = r.get("water_height_cm", 0)
        conf   = round(float(r.get("confidence", 0)) * 100)
        n_obj  = len(r.get("detected_objects", []))
        lines.append(
            f"{emoji} **{name}**{id_tag}\n"
            f"   Mức lũ : {_LEVEL_VI.get(lvl, lvl)}\n"
            f"   Độ sâu : {depth:.0f} cm\n"
            f"   Độ tin cậy: {conf}%"
            + (f"  |  {n_obj} vật thể" if n_obj else "")
        )
    if len(results) > 5:
        lines.append(f"… và {len(results)-5} ảnh khác.")
    return "\n".join(lines)


def _context_aware_message(results: List[Dict], memory: AgentMemory,
                            action_type: str, duration_s: float = 0) -> str:
    """
    Tạo response thông minh dựa trên context:
      - Số lần đã sửa
      - Confidence của kết quả
      - Trend lỗi
    """
    base = _format_analysis_response(
        results, duration_s,
        calib_applied=any(r.get("calibration_applied") for r in results),
    )

    n_corr = len(memory.correction_log)
    conf   = float(results[0].get("confidence", 0.5) if results else 0.5)
    trend  = memory.get_correction_trend()

    # Thêm nhận xét context-aware
    notes: List[str] = []

    if n_corr >= 3 and trend == "underestimate":
        notes.append(
            "\n💡 _Có vẻ model đang liên tục ước tính thấp hơn thực tế. "
            "Mình sẽ điều chỉnh ngưỡng tự động lần sau._"
        )
    elif n_corr >= 5:
        notes.append(
            "\n⚠️ _Đã sửa nhiều lần — xem xét chạy 'mô phỏng' để tìm ngưỡng tốt hơn._"
        )

    if conf < 0.42 and action_type != "simulate":
        notes.append(
            f"\n🔍 _Độ tin cậy thấp ({round(conf*100)}%). "
            "Bạn có muốn thử 'mô phỏng' để tối ưu ngưỡng không?_"
        )
    elif conf > 0.80:
        notes.append(f"\n✨ _Độ tin cậy cao ({round(conf*100)}%). Kết quả đáng tin cậy._")

    return base + "".join(notes)


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PATTERN LEARNING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_CONF_BANDS = ((0.70, "high"), (0.40, "mid"), (0.0, "low"))


def _make_context_key(results: List[Dict], action_type: str) -> str:
    """
    Tạo context key từ kết quả phân tích và loại action.

    Format: "{action_type}_{flood_level}_{conf_band}"
    Ví dụ:  "analyze_KNEE_mid", "rerun_WAIST_high"
    """
    if not results:
        return f"{action_type}_unknown_mid"
    r    = results[0]
    lvl  = str(r.get("flood_level", "UNKNOWN")).upper()
    conf = float(r.get("confidence", 0.5) or 0.5)
    band = next(b for (thresh, b) in _CONF_BANDS if conf >= thresh)
    return f"{action_type}_{lvl}_{band}"


def _templatize_response(text: str) -> str:
    """
    Chuyển response text → template có placeholder.

    Thay thế các giá trị cụ thể bằng {placeholder}:
      - số ảnh / duration
      - mức lũ (tên tiếng Việt)
      - độ sâu cm
      - confidence %
      - số vật thể
    """
    import re as _re

    t = text
    # Số ảnh + duration: "Phân tích xong 3 ảnh (2.5s)"
    t = _re.sub(r'(\d+) ảnh \([\d.]+s\)', r'{n_images} ảnh ({dur}s)', t)
    t = _re.sub(r'(\d+) ảnh',             r'{n_images} ảnh',           t)
    # Độ sâu
    t = _re.sub(r'\d+(?:\.\d+)? cm',      '{depth_cm} cm',             t)
    # Confidence %
    t = _re.sub(r'\d+%',                  '{conf_pct}%',               t)
    # Số vật thể
    t = _re.sub(r'\|\s*(\d+) vật thể',    '| {n_obj} vật thể',        t)
    # Tên mức lũ tiếng Việt + emoji
    for lvl_code, lvl_name in _LEVEL_VI.items():
        if lvl_name in t:
            t = t.replace(lvl_name, '{level_vi}')
    for lvl_code, emoji in _LEVEL_EMOJI.items():
        # Chỉ replace emoji ở đầu dòng (phần hiển thị chính)
        t = _re.sub(r'^' + _re.escape(emoji), '{level_emoji}', t, flags=_re.MULTILINE)
    return t


def _fill_response_template(template: str, results: List[Dict],
                             duration_s: float) -> str:
    """
    Điền giá trị thực vào template đã templatize.
    Nếu template có lỗi format → raise ValueError.
    """
    if not results:
        raise ValueError("No results to fill template")
    r    = results[0]
    lvl  = str(r.get("flood_level", "UNKNOWN")).upper()
    vals = {
        "n_images":    len(results),
        "dur":         round(duration_s, 1),
        "depth_cm":    int(round(float(r.get("water_height_cm", 0) or 0))),
        "conf_pct":    round(float(r.get("confidence", 0) or 0) * 100),
        "n_obj":       len(r.get("detected_objects", []) or []),
        "level_vi":    _LEVEL_VI.get(lvl, lvl),
        "level_emoji": _LEVEL_EMOJI.get(lvl, "❓"),
    }
    return template.format_map(vals)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT
# ─────────────────────────────────────────────────────────────────────────────

class FloodAgent:
    """
    Agent phân tích lũ lụt v2 — thuần nội bộ, không dùng external API.

    Tính năng mới so với v1:
      - PolicyEngine: scoring-based decision (không hard-code if-else)
      - AgentConfig: type-safe config với AgentConfig dataclass
      - EventBus: on("low_confidence", handler) / emit("event", data)
      - AgentMemory: persistent JSON + calibration bias + self-reflection
      - Online calibration: final_depth = predicted + bias (từ correction_log)
      - Confidence-aware: tự xác nhận (>80%), hỏi (50-80%), rerun (<30%)
      - Delta comparison: so sánh trước/sau rerun, rollback nếu tệ hơn
      - Simulation mode: thử nhiều threshold, chọn tốt nhất
      - Context-aware response: message thay đổi theo n_corrections, conf
      - Self-reflection: ghi lại outcome action → giảm điểm action hay thất bại
      - Decision trace: log đầy đủ điểm số + lý do → dễ debug

    Sử dụng:
        agent = FloodAgent()
        resp  = agent.process_images([Path("flood1.jpg")])
        resp  = agent.chat("Nước thấp hơn, khoảng 40cm")
    """

    def __init__(self, cfg: Optional[Dict] = None):
        # Config
        if isinstance(cfg, AgentConfig):
            self.config = cfg
        else:
            self.config = AgentConfig.from_dict(cfg or {})

        # Core components
        self.memory  = AgentMemory()
        self.parser  = FeedbackParser()
        self.engine  = PolicyEngine()
        self.events  = EventBus()
        self.tools   = PipelineTools(self.config, self.memory)
        self._lock   = threading.Lock()

        # LLM Enhancer — dùng model đã fine-tune nếu có, fallback về FeedbackParser
        self._llm = None
        self._init_llm()

        # Đăng ký default event handlers
        self._register_default_events()

    def _init_llm(self):
        """Load LLMEnhancer nếu model đã fine-tune tồn tại."""
        try:
            from pathlib import Path as _Path
            merged_dir = _Path(__file__).parent.parent / "models" / "flood-agent-merged"
            gguf_path  = _Path(__file__).parent.parent / "models" / "flood-agent-q4.gguf"

            if gguf_path.exists():
                from learning.flood_llm import LLMEnhancer
                self._llm = LLMEnhancer.from_gguf(str(gguf_path))
                log.info("[Agent] LLM loaded: GGUF (CPU fast mode)")
            elif merged_dir.exists():
                from learning.flood_llm import LLMEnhancer
                self._llm = LLMEnhancer.from_lora_dir(str(merged_dir))
                log.info("[Agent] LLM loaded: merged model")
            else:
                log.info("[Agent] Không tìm thấy LLM model — dùng FeedbackParser")
        except Exception as e:
            log.warning(f"[Agent] Không load được LLM: {e} — fallback FeedbackParser")

    # ── Default Event Handlers ────────────────────────────────────

    def _register_default_events(self):
        self.events.on("low_confidence", self._on_low_confidence)
        self.events.on("user_correction", self._on_user_correction)
        self.events.on("action_failed", self._on_action_failed)

    def _on_low_confidence(self, data: Dict):
        result = data.get("result", {})
        conf   = float(result.get("confidence", 0) or 0)
        log.info(f"[Event:low_confidence] conf={conf:.2f} → auto queue_review")
        self.tools.call("queue_review", result=result, reason="low_confidence_auto")

    def _on_user_correction(self, data: Dict):
        log.info(f"[Event:user_correction] intent={data.get('intent')} → record error")
        # Tự động tăng error counter theo loại lỗi
        intent = data.get("intent", "unknown")
        if intent == "increase":
            self.memory.record_error("underestimate")
        elif intent == "decrease":
            self.memory.record_error("overestimate")
        elif intent == "no_flood":
            self.memory.record_error("false_positive")

    def _on_action_failed(self, data: Dict):
        log.warning(f"[Event:action_failed] tool={data.get('tool')} err={data.get('error')}")
        self.memory.record_error(f"{data.get('tool','unknown')}_fail")

    # ── Public API ────────────────────────────────────────────────

    def process_images(self, image_paths: List[Path]) -> AgentResponse:
        """
        Nhận danh sách ảnh → phân tích → confidence-aware decision.
        Entry point chính khi user upload ảnh.
        """
        with self._lock:
            self.memory.n_sessions += 1

        self.memory.add_message(
            "user", f"[Upload] {len(image_paths)} ảnh: "
            + ", ".join(p.name for p in image_paths[:5])
        )
        log.info(f"[Agent] process_images: {len(image_paths)} ảnh")

        tr = self.tools.call("analyze", images=image_paths)

        if not tr.success:
            msg = f"❌ Lỗi phân tích: {tr.error}"
            self.memory.add_message("agent", msg)
            self.memory.record_error("analyze_fail")
            self.events.emit("action_failed", {"tool": "analyze", "error": tr.error})
            return AgentResponse(message=msg, success=False, action="analyze",
                                 tool_results=[tr])

        results  = tr.data.get("results", [])
        duration = tr.data.get("duration_s", 0)
        extra_msg = ""

        if results:
            # Gán image_id cho từng kết quả và lưu vào history
            for r in results:
                img_id = self.memory.add_image_to_history(r, image_paths)
                r["image_id"] = img_id
            first = results[0]
            conf  = float(first.get("confidence", 0.5) or 0.5)
            self.memory.last_action_type = "analyze"
            self.memory.set_last_result(first, image_paths)

            # ── Confidence-aware decision ─────────────────────────
            if conf > self.config.conf_auto_confirm:
                # Auto confirm: đưa vào training ngay
                self.tools.call("queue_review", result=first, reason="auto_high_conf")
                extra_msg = f"\n✨ _Độ tin cậy cao ({round(conf*100)}%) — tự động xác nhận._"
                log.info(f"[Agent] Auto-confirm (conf={conf:.2f})")

            elif conf < self.config.conf_auto_rerun:
                # Auto rerun: tự chạy lại không cần hỏi
                log.info(f"[Agent] Auto-rerun do conf={conf:.2f} < {self.config.conf_auto_rerun}")
                rerun_tr = self.tools.call("rerun", images=image_paths)
                if rerun_tr.success and rerun_tr.data.get("results"):
                    new_results = rerun_tr.data["results"]
                    delta = self._compare_results(first, new_results[0])
                    results = new_results
                    self.memory.set_last_result(new_results[0])
                    extra_msg = (
                        f"\n🔄 _Tự động chạy lại do độ tin cậy rất thấp ({round(conf*100)}%). "
                        f"Kết quả mới: {'+' if delta['depth_delta'] >= 0 else ''}"
                        f"{delta['depth_delta']}cm, conf thay đổi {'+' if delta['conf_delta'] >= 0 else ''}"
                        f"{round(delta['conf_delta']*100)}%._"
                    )
                    self.events.emit("action_failed" if not delta["improved"] else "low_confidence",
                                     {"result": new_results[0]})

            elif conf < self.config.conf_ask_user:
                # Hỏi user
                extra_msg = (
                    f"\n🔍 _Kết quả này chưa chắc chắn ({round(conf*100)}%). "
                    "Bạn có thể xác nhận hoặc điều chỉnh?_"
                )
                self.events.emit("low_confidence", {"result": first})
                log.info(f"[Agent] Ask user (conf={conf:.2f})")

        # Kiểm tra phong cách trả lời đã học
        ctx_key = _make_context_key(results, "analyze")
        pattern = self.memory.find_matching_pattern(ctx_key)
        if pattern and not extra_msg:
            try:
                base_msg = _fill_response_template(pattern["template"], results, duration)
                base_msg += "\n_💾 [Đang dùng phong cách trả lời đã lưu]_"
                log.info(f"[Agent] process_images: using learned pattern {ctx_key}")
            except Exception:
                base_msg = _context_aware_message(results, self.memory, "analyze", duration)
        else:
            base_msg = _context_aware_message(results, self.memory, "analyze", duration)

        msg = base_msg + extra_msg
        self.memory.add_message("agent", msg)

        return AgentResponse(
            message=msg, success=True, action="analyze",
            results=results, tool_results=[tr], memory=self.memory.to_summary(),
        )

    def chat(self, user_text: str) -> AgentResponse:
        """
        Nhận phản hồi text từ user → parse → policy score → quyết định → thực thi.
        Entry point cho chat/feedback loop.
        """
        self.memory.add_message("user", user_text)

        # Dùng LLM nếu có, fallback về FeedbackParser
        if self._llm is not None:
            parsed = self._llm.parse_intent(
                user_text,
                self.memory.last_result,
                chat_history=self.memory.get_recent_history(6),
            )
            self._llm_response_hint = parsed.response or ""
        else:
            parsed = self.parser.parse(user_text)
            self._llm_response_hint = ""

        # Nếu user nhắc đến ID ảnh cụ thể ("ảnh #2 sai") → switch context về ảnh đó
        if parsed.image_id is not None:
            entry = self.memory.get_image_by_id(parsed.image_id)
            if entry:
                self.memory.last_result = entry["result"]
                self.memory.last_images = [Path(p) for p in entry.get("images", [])]
                log.info(f"[Agent] Context switched → ảnh #{parsed.image_id}: {entry['filename']}")
            else:
                msg = (f"⚠️ Không tìm thấy ảnh #{parsed.image_id} trong lịch sử. "
                       f"Hiện có {len(self.memory.image_history)} ảnh — "
                       f"dùng 'lịch sử' để xem danh sách.")
                self.memory.add_message("agent", msg)
                return AgentResponse(message=msg, action="clarify",
                                     memory=self.memory.to_summary())

        # Xử lý lệnh xem lịch sử ảnh
        if any(kw in user_text.lower() for kw in
               ("lịch sử", "lich su", "history", "danh sách ảnh", "xem ảnh đã")):
            return self._handle_image_history()

        log.info(f"[Agent] chat: intent={parsed.intent} depth={parsed.depth_hint} "
                 f"level={parsed.level_hint} magnitude={parsed.magnitude} "
                 f"image_id={parsed.image_id} source={'llm' if self._llm else 'parser'}")

        # Save response pattern: xử lý trực tiếp, không qua scoring
        if parsed.intent == "save_response":
            return self._handle_save_response()

        # Emit user_correction event nếu là intent sửa lỗi
        if parsed.intent in ("increase", "decrease", "no_flood"):
            self.events.emit("user_correction", {
                "intent": parsed.intent,
                "depth_hint": parsed.depth_hint,
                "level_hint": parsed.level_hint,
            })

        action = self.engine.decide(parsed, self.memory, self.config)
        return self._execute_action(action, parsed)

    def _handle_save_response(self) -> AgentResponse:
        """
        Lưu phong cách trả lời gần nhất vào memory.
        Được gọi khi user nói "lưu cách trả lời này" hoặc tương tự.
        """
        # Lấy message gần nhất của agent (trước lệnh "save" của user)
        last_agent_msgs = [
            m for m in reversed(self.memory.chat_history)
            if m["role"] == "agent"
        ]
        if not last_agent_msgs:
            msg = "⚠️ Chưa có phản hồi nào để lưu. Hãy phân tích ảnh trước."
            self.memory.add_message("agent", msg)
            return AgentResponse(message=msg, action="save_response",
                                 memory=self.memory.to_summary())

        last_msg  = last_agent_msgs[0]["content"]
        last      = self.memory.last_result or {}
        ctx_key   = _make_context_key(
            [last] if last else [],
            self.memory.last_action_type,
        )
        template  = _templatize_response(last_msg)
        self.memory.save_response_pattern(ctx_key, template, last_msg)

        log.info(f"[Agent] Response pattern saved: key={ctx_key}")

        n_patterns = len(self.memory.response_patterns)
        msg = (
            f"✅ Đã lưu phong cách trả lời!\n"
            f"  Ngữ cảnh: `{ctx_key}`\n"
            f"  Tổng số phong cách đã lưu: {n_patterns}\n\n"
            f"_Mình sẽ dùng phong cách này khi gặp tình huống tương tự._"
        )
        self.memory.add_message("agent", msg)
        return AgentResponse(message=msg, action="save_response",
                             memory=self.memory.to_summary())

    def _handle_image_history(self) -> AgentResponse:
        """Hiển thị lịch sử các ảnh đã phân tích kèm ID."""
        items = self.memory.get_image_history_summary(n=20)
        if not items:
            msg = "📋 Chưa có ảnh nào được phân tích trong session này."
            self.memory.add_message("agent", msg)
            return AgentResponse(message=msg, action="history",
                                 memory=self.memory.to_summary())

        lines = [f"📋 **Lịch sử {len(items)} ảnh gần nhất:**\n"]
        for e in items:
            lvl   = e["level"]
            emoji = _LEVEL_EMOJI.get(lvl, "❓")
            lines.append(
                f"{emoji} **#ID {e['id']}** — {e['filename']}\n"
                f"   {_LEVEL_VI.get(lvl, lvl)} | {e['depth_cm']:.0f}cm | {e['conf_pct']}%\n"
                f"   _{e['timestamp']}_"
            )
        lines.append(
            "\n💡 _Để sửa ảnh bất kỳ: \"ảnh #2 nước thấp hơn\" "
            "hoặc \"ảnh #3, thực ra không ngập\"_"
        )
        msg = "\n".join(lines)
        self.memory.add_message("agent", msg)
        return AgentResponse(message=msg, action="history",
                             memory=self.memory.to_summary())

    def get_memory(self) -> Dict:
        """Snapshot bộ nhớ hiện tại."""
        return self.memory.to_summary()

    def reset_session(self):
        """Xóa bộ nhớ session (giữ correction_log và calibration)."""
        with self._lock:
            self.memory.chat_history.clear()
            self.memory.last_result  = None
            self.memory.last_images  = []
        log.info("[Agent] Session reset (long-term memory preserved)")

    def reset_all(self):
        """Xóa toàn bộ bộ nhớ kể cả long-term."""
        with self._lock:
            self.memory.chat_history.clear()
            self.memory.last_result  = None
            self.memory.last_images  = []
            self.memory.clear_long_term()
        log.info("[Agent] Full reset including long-term memory")

    # ── Internal: Execute Action ──────────────────────────────────

    def _execute_action(self, action: Action, parsed: ParsedFeedback) -> AgentResponse:
        tool_results: List[ToolResult] = []
        rerun    = False
        llm_hint = getattr(self, "_llm_response_hint", "")
        # Chỉ dùng LLM response cho clarify/unknown — action khác giữ nguyên template
        msg = llm_hint or action.message_hint
        before   = self.memory.last_result

        # ── Không có tool call ────────────────────────────────────
        if action.type in ("clarify", "help", "status", "greeting"):
            if action.type == "status":
                tr = self.tools.call("system_status")
                tool_results.append(tr)
                msg = self._format_status(tr)
            elif action.type == "help":
                msg = self._help_text()
            elif action.type == "greeting":
                if not llm_hint:
                    msg = self._greeting_text()
            self.memory.add_message("agent", msg)
            return AgentResponse(
                message=msg, action=action.type,
                tool_results=tool_results,
                memory=self.memory.to_summary(),
                decision_log=action.decision_log,
            )

        # ── Gọi từng tool ─────────────────────────────────────────
        final_results: List[Dict] = []

        for call in action.tool_calls:
            tr = self.tools.call(call["tool"], **call.get("kwargs", {}))
            tool_results.append(tr)

            if not tr.success:
                self.memory.record_error(f"{call['tool']}_fail")
                self.events.emit("action_failed",
                                 {"tool": call["tool"], "error": tr.error})
                log.warning(f"[Agent] Tool {call['tool']} thất bại: {tr.error}")
                continue

            # Kết quả analyze / rerun
            if call["tool"] in ("analyze", "rerun", "simulate") and tr.success:
                results = tr.data.get("results", [])
                if results:
                    after  = results[0]
                    delta  = self._compare_results(before, after) if before else None
                    rerun  = call["tool"] in ("rerun", "simulate")

                    # Delta comparison: rollback nếu kết quả tệ hơn đáng kể
                    if delta and not delta["improved"] and abs(delta["conf_delta"]) > 0.05:
                        log.info(
                            f"[Agent] Delta check: conf {delta['conf_delta']:+.3f} "
                            f"depth {delta['depth_delta']:+.1f}cm → rollback"
                        )
                        # Giữ kết quả cũ nhưng thông báo
                        msg_suffix = (
                            f"\n⚠️ _Kết quả sau rerun thấp hơn (conf giảm "
                            f"{abs(round(delta['conf_delta']*100))}%). "
                            "Giữ lại kết quả cũ._"
                        )
                        msg = (msg or "") + msg_suffix
                    else:
                        final_results = results
                        self.memory.last_action_type = action.type
                        self.memory.set_last_result(after)

                        # Ghi correction nếu user đã sửa
                        if parsed.intent in ("increase", "decrease", "no_flood") and before:
                            self.memory.add_correction(
                                original_level  = before.get("flood_level", "?"),
                                corrected_level = after.get("flood_level", "?"),
                                original_depth  = float(before.get("water_height_cm", 0) or 0),
                                corrected_depth = parsed.depth_hint,
                                feedback        = parsed.raw,
                            )

                        # Self-reflection
                        self.memory.record_action_outcome(
                            action_type = action.type,
                            before      = before,
                            after       = after,
                            success     = True,
                        )

                        # Simulation summary
                        if call["tool"] == "simulate" and tr.data.get("sim_log"):
                            sim_summary = self._format_simulation(tr.data)
                            msg = (msg or "") + "\n" + sim_summary

            # Adjust threshold → thêm vào message nếu chưa có
            if call["tool"] == "adjust_threshold" and tr.success:
                d = tr.data
                if not msg:
                    dir_vi = "tăng" if d["direction"] == "up" else "giảm"
                    msg = (f"Đã {dir_vi} ngưỡng: "
                           f"{d['old']:.2f} → {d['new']:.2f}.")

            # Calibrate result
            if call["tool"] == "calibrate_result" and tr.success:
                cal = tr.data
                if cal.get("bias_applied") and not msg:
                    msg = (
                        f"🎯 Đã hiệu chỉnh kết quả:\n"
                        f"  Bias: {cal['bias_cm']:+.1f}cm ({cal['n_samples']} mẫu)\n"
                        f"  Độ sâu mới: {cal['result'].get('water_height_cm', '?')}cm"
                    )
                elif not cal.get("bias_applied") and not msg:
                    msg = f"ℹ️ {cal.get('reason', 'Không đủ dữ liệu để hiệu chỉnh.')}"

        # ── Tạo message cuối ─────────────────────────────────────
        if not msg and final_results:
            dur = next((tr.data.get("duration_s", 0) for tr in tool_results
                        if tr.tool in ("analyze", "rerun", "simulate")), 0)

            # Thử dùng phong cách trả lời đã học
            ctx_key = _make_context_key(final_results, action.type)
            pattern = self.memory.find_matching_pattern(ctx_key)
            if pattern:
                try:
                    msg = _fill_response_template(pattern["template"], final_results, dur)
                    msg += "\n_💾 [Đang dùng phong cách trả lời đã lưu]_"
                    log.info(f"[Agent] Using learned response pattern: {ctx_key}")
                except Exception as e:
                    log.debug(f"[Agent] Pattern fill failed ({e}), using default")
                    msg = _context_aware_message(final_results, self.memory, action.type, dur)
            else:
                msg = _context_aware_message(final_results, self.memory, action.type, dur)

        elif not msg:
            msg = "✅ Đã hoàn thành." if all(t.success for t in tool_results) \
                  else "⚠️ Có lỗi xảy ra. Kiểm tra log để biết thêm."

        self.memory.add_message("agent", msg)
        return AgentResponse(
            message=msg,
            success=all(t.success for t in tool_results),
            action=action.type,
            results=final_results,
            tool_results=tool_results,
            rerun=rerun,
            memory=self.memory.to_summary(),
            decision_log=action.decision_log,
        )

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _compare_results(old: Optional[Dict], new: Optional[Dict]) -> Dict:
        """So sánh kết quả trước–sau rerun."""
        if not old or not new:
            return {"depth_delta": 0.0, "conf_delta": 0.0, "improved": True, "significant": False}
        old_depth = float(old.get("water_height_cm", 0) or 0)
        new_depth = float(new.get("water_height_cm", 0) or 0)
        old_conf  = float(old.get("confidence", 0) or 0)
        new_conf  = float(new.get("confidence", 0) or 0)
        delta_d   = new_depth - old_depth
        delta_c   = new_conf  - old_conf
        return {
            "depth_delta":  round(delta_d, 1),
            "conf_delta":   round(delta_c, 3),
            "improved":     delta_c > -0.02,   # improved nếu conf không giảm đáng kể
            "significant":  abs(delta_d) >= 5.0,
        }

    @staticmethod
    def _format_simulation(data: Dict) -> str:
        sim_log   = data.get("sim_log", [])
        best_t    = data.get("best_threshold", "?")
        orig_t    = data.get("original_threshold", "?")
        lines     = ["\n🧪 **Kết quả Simulation:**"]
        for entry in sim_log:
            marker = " ← best" if entry["threshold"] == best_t else ""
            lines.append(
                f"  threshold={entry['threshold']:.2f} → "
                f"avg_conf={entry['avg_confidence']:.1%}{marker}"
            )
        lines.append(
            f"\n✅ Ngưỡng tốt nhất: **{best_t:.2f}** "
            f"(trước: {orig_t:.2f})"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_status(tr: ToolResult) -> str:
        if not tr.success:
            return f"❌ Lỗi status: {tr.error}"
        d    = tr.data
        cfg  = d.get("cfg", {})
        mem  = d.get("memory", {})
        bias = d.get("calibration_bias", {})
        perf = d.get("action_perf", {})
        lines = ["📊 **Trạng thái hệ thống**"]
        lines.append(f"  Threshold : {cfg.get('flood_threshold', 'N/A')}")
        lines.append(f"  YOLO model: {cfg.get('yolo_model', 'N/A')}")
        if mem:
            lines.append(f"  RAM       : {mem.get('ram_used_mb', '?')} / "
                         f"{mem.get('ram_total_mb', '?')} MB")
        if bias.get("reliable"):
            lines.append(f"  Bias      : {bias['depth_bias_cm']:+.1f}cm "
                         f"({bias['n_samples']} mẫu)")
        if perf:
            lines.append("  Action perf:")
            for act, s in perf.items():
                impr_rate = s['improved'] / max(s['total'], 1)
                lines.append(f"    {act}: {s['total']} lần, cải thiện {impr_rate:.0%}")
        return "\n".join(lines)

    @staticmethod
    def _greeting_text() -> str:
        return (
            "Xin chào! Mình là FloodAgent — trợ lý phân tích lũ lụt.\n\n"
            "Bạn có thể:\n"
            "• **Upload ảnh** để mình phân tích mức độ ngập\n"
            "• Gõ _\"Hướng dẫn\"_ để xem đầy đủ các lệnh\n"
            "• Gõ _\"Trạng thái\"_ để xem thông tin hệ thống"
        )

    @staticmethod
    def _help_text() -> str:
        return (
            "📖 **Hướng dẫn sử dụng FloodAgent v2**\n\n"
            "**Upload ảnh**: gửi ảnh lũ để phân tích tự động\n\n"
            "**Phản hồi sau khi có kết quả:**\n"
            '• _"Đúng rồi"_ — xác nhận, lưu vào training\n'
            '• _"Nước thấp hơn"_ / _"Nước cao hơn"_ — điều chỉnh + chạy lại\n'
            '• _"Khoảng 50cm"_ — gợi ý độ sâu cụ thể\n'
            '• _"Không có lũ ở đây"_ — tăng ngưỡng phát hiện\n'
            '• _"Chạy lại"_ — phân tích lại ảnh\n'
            '• _"Mô phỏng"_ — thử nhiều ngưỡng, chọn tốt nhất tự động\n'
            '• _"Hiệu chỉnh"_ — áp dụng bias từ lịch sử correction\n'
            '• _"Trạng thái"_ — xem system info + calibration + perf\n\n'
            "**Tính năng AI v2:**\n"
            "• PolicyEngine tự tính điểm và chọn action tốt nhất\n"
            "• Tự động hiệu chỉnh độ sâu dựa trên lịch sử sửa lỗi\n"
            "• Tự động chạy lại nếu confidence < 30%\n"
            "• Simulation mode: thử 5 ngưỡng, pick best\n"
            "• Bộ nhớ dài hạn: không mất khi tắt server\n"
        )

    # ── Convenience: load config từ file ─────────────────────────

    @classmethod
    def from_config_file(cls, config_path: str = "config.yaml") -> "FloodAgent":
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        return cls(cfg)
