# -*- coding: utf-8 -*-
"""
agent/flood_agent.py
====================
FloodAgent — agent phân tích lũ lụt thuần nội bộ, không dùng external LLM API.

Kiến trúc v3 (nâng cấp toàn diện):
  ReportState       — state machine chuẩn: received→analyzed→verified→pending→published
  AgentDecision     — JSON-first structured output (decision + reasons + advisory)
  ConfidenceTranslator — map số kỹ thuật → tiếng Việt cho user/admin
  EditorialPolicy   — guardrail nội dung: tránh giật tít, dùng ngôn ngữ chuẩn
  PermissionLevel   — phân quyền agent: analyst/editor/admin
  AuditLog          — ghi log mọi quyết định (agent, input, decision, reasons)
  FallbackEngine    — rule-based fallback khi LLM/API lỗi
  EventClusterer    — gom báo cáo gần nhau thành 1 event
  AlertLifecycle    — vòng đời cảnh báo: new→active→worsening→stable→resolved
  FloodAdvisory     — khuyến cáo theo đối tượng (xe máy, ô tô, người đi bộ)
  MissingInfoChecker — phát hiện thiếu thông tin và sinh câu hỏi hỏi lại
  AgentConfig       — structured config, type-safe (thay raw Dict)
  EventBus          — event system: on("low_confidence", handler)
  FeedbackParser    — hiểu ý kiến người dùng (tiếng Việt + English)
  AgentMemory       — bộ nhớ session + lịch sử correction + persistent JSON
  PipelineTools     — tool registry với calibration + simulation mode
  PolicyEngine      — scoring-based decision tree
  FloodAgent        — điều phối toàn bộ

Response layer (v3 mới — xem agent/response_rewriter.py):
  ResponseRewriter  — Decision JSON → natural language (5 modes)
  ResponseValidator — Kiểm tra từ kỹ thuật rò rỉ + giật tít
  ResponseMode      — public_user | admin_review | news_writer | alert_message | debug
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
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("flood_agent")

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT STORAGE PATH
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_DIR  = Path(__file__).parent / "_agent_memory"
_MEMORY_FILE = _AGENT_DIR / "long_term_memory.json"
_AUDIT_FILE  = _AGENT_DIR / "audit_log.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# PERMISSION SYSTEM  (#17)
# ─────────────────────────────────────────────────────────────────────────────

class PermissionLevel(Enum):
    """Phân quyền cho agent — agent không được tự làm những gì vượt quyền."""
    READ_ONLY = "read_only"
    ANALYST   = "analyst"    # phân tích, tạo nháp, queue review
    EDITOR    = "editor"     # + đăng mức thấp, gửi cảnh báo nhẹ
    ADMIN     = "admin"      # toàn quyền


class AgentPermissions:
    """
    Kiểm soát những gì agent có thể làm theo level.

    Dùng:
        AgentPermissions.can(PermissionLevel.ANALYST, "publish_high_alert")  # → False
        AgentPermissions.can(PermissionLevel.ADMIN,   "publish_high_alert")  # → True
    """

    _ALLOWED: Dict[PermissionLevel, set] = {
        PermissionLevel.READ_ONLY: set(),
        PermissionLevel.ANALYST:  {
            "create_draft", "queue_review", "add_to_map_draft",
            "ask_followup", "update_state", "send_to_review",
        },
        PermissionLevel.EDITOR: {
            "create_draft", "queue_review", "add_to_map_draft",
            "ask_followup", "update_state", "send_to_review",
            "publish_low", "publish_medium", "send_low_alert", "send_medium_alert",
        },
        PermissionLevel.ADMIN: {"*"},  # tất cả
    }

    # Những action này luôn cần ADMIN, kể cả Editor không được
    _ALWAYS_REQUIRE_ADMIN = {
        "publish_high_alert",
        "publish_critical_alert",
        "delete_source_data",
        "modify_audit_log",
        "change_production_config",
        "override_verified",
    }

    @classmethod
    def can(cls, level: PermissionLevel, action: str) -> bool:
        if action in cls._ALWAYS_REQUIRE_ADMIN and level != PermissionLevel.ADMIN:
            return False
        allowed = cls._ALLOWED.get(level, set())
        return "*" in allowed or action in allowed

    @classmethod
    def require(cls, level: PermissionLevel, action: str) -> None:
        """Raise PermissionError nếu không đủ quyền."""
        if not cls.can(level, action):
            raise PermissionError(
                f"Agent ({level.value}) không có quyền '{action}'. "
                f"Cần ADMIN hoặc level cao hơn."
            )


# ─────────────────────────────────────────────────────────────────────────────
# REPORT STATE MACHINE  (#2)
# ─────────────────────────────────────────────────────────────────────────────

_REPORT_TRANSITIONS = {
    "received":       ["analyzed", "rejected"],
    "analyzed":       ["verified", "pending_review", "rejected"],
    "verified":       ["drafted",  "pending_review", "rejected"],
    "drafted":        ["pending_review", "rejected"],
    "pending_review": ["published", "rejected"],
    "published":      ["archived"],
    "rejected":       [],
    "archived":       [],
}


@dataclass
class ReportState:
    """
    State machine cho một báo cáo lũ.
    Agent KHÔNG tự đăng — chỉ advance đến pending_review, admin mới publish.
    """
    report_id:          str
    source:             str          # "citizen_upload" | "camera" | "api"
    image_path:         str
    raw_description:    str  = ""
    raw_location:       str  = ""
    resolved_location:  Optional[str]  = None
    analysis_result:    Optional[Dict] = None
    verification_result: Optional[Dict] = None
    news_draft:         Optional[str]  = None
    alert_decision:     Optional[Dict] = None
    status:             str  = "received"
    missing_info:       List[str] = field(default_factory=list)
    created_at:         str  = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at:         str  = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def advance(self, new_status: str) -> None:
        """Chuyển trạng thái hợp lệ — raise nếu transition không được phép."""
        allowed = _REPORT_TRANSITIONS.get(self.status, [])
        if new_status not in allowed:
            raise ValueError(
                f"Không thể chuyển từ '{self.status}' → '{new_status}'. "
                f"Cho phép: {allowed}"
            )
        self.status     = new_status
        self.updated_at = datetime.now().isoformat(timespec="seconds")

    def can_advance_to(self, new_status: str) -> bool:
        return new_status in _REPORT_TRANSITIONS.get(self.status, [])

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# ALERT LIFECYCLE  (#11)
# ─────────────────────────────────────────────────────────────────────────────

class AlertLifecycle(Enum):
    NEW        = "new"
    ACTIVE     = "active"
    WORSENING  = "worsening"
    STABLE     = "stable"
    IMPROVING  = "improving"
    RESOLVED   = "resolved"
    EXPIRED    = "expired"


def infer_alert_lifecycle(prev_depth: float, curr_depth: float,
                           minutes_since_update: float) -> AlertLifecycle:
    """
    Suy luận vòng đời cảnh báo từ delta mực nước và thời gian.

    Dùng khi có nhiều báo cáo cùng khu vực:
        lifecycle = infer_alert_lifecycle(prev_depth, curr_depth, elapsed_min)
    """
    if minutes_since_update > 180:   # >3h không cập nhật
        return AlertLifecycle.EXPIRED

    delta = curr_depth - prev_depth
    if curr_depth < 5:
        return AlertLifecycle.RESOLVED

    if delta > 10:
        return AlertLifecycle.WORSENING
    if delta < -10:
        return AlertLifecycle.IMPROVING
    if abs(delta) <= 5:
        return AlertLifecycle.STABLE

    return AlertLifecycle.ACTIVE


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE TRANSLATOR  (#9)
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceTranslator:
    """
    Map số kỹ thuật (0.0–1.0) → ngôn ngữ thân thiện cho 2 đối tượng:
      - public: chỉ thấy label chung
      - admin: thấy số + label
    """

    _BANDS: List[Tuple[float, str, str]] = [
        # (ngưỡng_tối_thiểu, label_vi, key)
        (0.85, "Độ tin cậy cao",        "HIGH"),
        (0.65, "Cần xác minh thêm",     "MEDIUM"),
        (0.0,  "Chưa đủ cơ sở kết luận","LOW"),
    ]

    @classmethod
    def to_public(cls, conf: float) -> str:
        """Dùng cho UI người dân — không hiện số."""
        for threshold, label, _ in cls._BANDS:
            if conf >= threshold:
                return label
        return cls._BANDS[-1][1]

    @classmethod
    def to_admin(cls, conf: float) -> str:
        """Dùng cho admin dashboard — hiện số kèm label."""
        for threshold, label, _ in cls._BANDS:
            if conf >= threshold:
                return f"{round(conf * 100)}% ({label})"
        return f"{round(conf * 100)}% (Chưa đủ cơ sở)"

    @classmethod
    def status_label(cls, conf: float) -> str:
        """Dùng cho trạng thái trên map / feed."""
        for threshold, _, key in cls._BANDS:
            if conf >= threshold:
                return {
                    "HIGH":   "Đã xác minh",
                    "MEDIUM": "Đang chờ xác minh",
                    "LOW":    "Cần kiểm tra thêm",
                }.get(key, "Không rõ")
        return "Không rõ"


# ─────────────────────────────────────────────────────────────────────────────
# EDITORIAL POLICY  (#8)
# ─────────────────────────────────────────────────────────────────────────────

class EditorialPolicy:
    """
    Guardrail nội dung cho News Agent / bất kỳ text nào đăng công khai.

    Không cho dùng ngôn ngữ giật tít chưa xác minh.
    Tự động gợi ý thay thế.
    """

    _FORBIDDEN: List[str] = [
        "kinh hoàng", "thảm họa", "chấn động", "nguy hiểm chết người",
        "sốc:", "cực kỳ nguy hiểm", "chìm trong biển nước", "nhấn chìm",
        "hãi hùng", "tang thương", "hoảng loạn",
    ]

    _REPLACEMENTS: Dict[str, str] = {
        "kinh hoàng":             "đáng lo ngại",
        "thảm họa":               "sự cố nghiêm trọng",
        "chấn động":              "đáng chú ý",
        "nguy hiểm chết người":   "nguy hiểm, cần thận trọng",
        "chìm trong biển nước":   "ngập sâu",
        "nhấn chìm":              "gây ngập",
        "hãi hùng":               "nghiêm trọng",
        "tang thương":            "thiệt hại",
        "hoảng loạn":             "lo lắng",
    }

    # Từ nên dùng thay vì khẳng định chắc chắn khi chưa xác minh
    _PREFERRED_PREFIXES = [
        "Ghi nhận",
        "Ước tính",
        "Cần chú ý",
        "Khuyến cáo",
        "Đang chờ xác minh",
    ]

    @classmethod
    def check(cls, text: str) -> Tuple[bool, List[str]]:
        """
        Kiểm tra text có vi phạm policy không.
        Returns: (is_ok, violations)
        """
        violations = [w for w in cls._FORBIDDEN if w.lower() in text.lower()]
        return len(violations) == 0, violations

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Thay thế từ vi phạm bằng từ phù hợp."""
        result = text
        for bad, good in cls._REPLACEMENTS.items():
            result = re.sub(re.escape(bad), good, result, flags=re.IGNORECASE)
        return result

    @classmethod
    def generate_title(cls, result: Dict, location: str = "",
                       verified: bool = False) -> str:
        """
        Sinh tiêu đề chuẩn từ kết quả phân tích.
        Không giật tít, dùng 'Ghi nhận' thay vì khẳng định.
        """
        depth    = float(result.get("water_height_cm", 0) or 0)
        level    = result.get("flood_level", "UNKNOWN")
        loc_part = f" tại {location}" if location else ""
        status   = "" if verified else " (đang chờ xác minh)"

        if level in ("NO_FLOOD", "UNKNOWN"):
            return f"Kiểm tra tình trạng ngập{loc_part}"
        if depth > 0:
            lo = int(depth * 0.85)
            hi = int(depth * 1.15)
            return f"Ghi nhận ngập khoảng {lo}–{hi}cm{loc_part}{status}"
        return f"Ghi nhận tình trạng ngập{loc_part}{status}"

    @classmethod
    def generate_summary(cls, result: Dict, n_reports: int = 1) -> str:
        """Sinh tóm tắt ngắn chuẩn mực."""
        depth = float(result.get("water_height_cm", 0) or 0)
        conf  = ConfidenceTranslator.to_public(float(result.get("confidence", 0) or 0))
        multi = f" ({n_reports} báo cáo)" if n_reports > 1 else ""
        return (
            f"Mực nước ước tính khoảng {int(depth)}cm{multi}. "
            f"{conf}. Cần tiếp tục xác minh."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FLOOD ADVISORY — khuyến cáo theo đối tượng  (#12)
# ─────────────────────────────────────────────────────────────────────────────

class FloodAdvisory:
    """
    Sinh khuyến cáo cụ thể cho từng đối tượng dựa trên độ sâu.

    Output:
        {
            "motorbike": "Không nên di chuyển...",
            "car":       "Ô tô gầm thấp...",
            "pedestrian": "Người đi bộ...",
        }
    """

    # (depth_cm, motorbike_ok, car_low_ok, car_high_ok, pedestrian_note)
    _THRESHOLDS = {
        "motorbike":  30,   # >= 30cm → không nên
        "car_low":    40,   # ô tô gầm thấp >= 40cm → không nên
        "car_high":   80,   # ô tô gầm cao >= 80cm → không nên
        "pedestrian": 15,   # >= 15cm → cần chú ý hố ga
    }

    @classmethod
    def generate(cls, depth_cm: float) -> Dict[str, str]:
        recs: Dict[str, str] = {}

        # Xe máy
        if depth_cm >= cls._THRESHOLDS["motorbike"]:
            recs["motorbike"] = "Không nên di chuyển qua khu vực này bằng xe máy."
        elif depth_cm >= 15:
            recs["motorbike"] = "Xe máy có thể qua nhưng cần đi chậm, tránh tắt máy giữa chừng."
        else:
            recs["motorbike"] = "Xe máy có thể qua bình thường, nên đi chậm."

        # Ô tô
        if depth_cm >= cls._THRESHOLDS["car_high"]:
            recs["car"] = "Ô tô không nên đi qua khu vực ngập sâu này."
        elif depth_cm >= cls._THRESHOLDS["car_low"]:
            recs["car"] = "Ô tô gầm thấp nên chọn tuyến thay thế."
        else:
            recs["car"] = "Ô tô có thể qua với tốc độ thấp và thận trọng."

        # Người đi bộ
        if depth_cm >= 60:
            recs["pedestrian"] = "Người đi bộ không nên lội qua, nguy cơ bị cuốn nếu nước chảy mạnh."
        elif depth_cm >= cls._THRESHOLDS["pedestrian"]:
            recs["pedestrian"] = (
                "Người đi bộ cần chú ý miệng cống và dòng nước chảy. "
                "Không để trẻ em và người cao tuổi đi một mình."
            )
        else:
            recs["pedestrian"] = "Người đi bộ qua được, cần chú ý bề mặt trơn."

        # Học sinh / phụ huynh
        if depth_cm >= 20:
            recs["school"]  = "Phụ huynh/học sinh nên chọn tuyến khác hoặc chờ nước rút."

        return recs

    @classmethod
    def to_text(cls, depth_cm: float) -> str:
        recs  = cls.generate(depth_cm)
        lines = ["**Khuyến cáo di chuyển:**"]
        labels = {
            "motorbike": "🛵 Xe máy",
            "car":       "🚗 Ô tô",
            "pedestrian":"🚶 Người đi bộ",
            "school":    "🎒 Học sinh",
        }
        for key, label in labels.items():
            if key in recs:
                lines.append(f"• {label}: {recs[key]}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING INFO CHECKER  (#6)
# ─────────────────────────────────────────────────────────────────────────────

class MissingInfoChecker:
    """
    Kiểm tra báo cáo có đủ thông tin không.
    Trả về danh sách câu hỏi cần hỏi thêm.
    """

    @staticmethod
    def check(report: Dict) -> List[Dict]:
        """
        Returns: list of {field, question, options}
        Options = None → câu hỏi mở, List → câu hỏi có lựa chọn.
        """
        missing: List[Dict] = []

        # Thiếu vị trí
        if not report.get("location") and not report.get("gps") and not report.get("exif_gps"):
            missing.append({
                "field":    "location",
                "question": (
                    "📍 Bạn có thể cho biết ảnh này chụp ở khu vực nào không?\n"
                    "Ví dụ: tên đường, cổng trường, chợ, phường/xã."
                ),
                "options":  None,
                "required": True,
            })

        # Thiếu thời gian
        if not report.get("timestamp") and not report.get("exif_datetime"):
            missing.append({
                "field":    "timestamp",
                "question": "🕐 Ảnh này được chụp khi nào?",
                "options":  ["Vừa chụp", "Trong vòng 1 giờ", "Hôm nay", "Không rõ"],
                "required": False,
            })

        # Ảnh chất lượng kém (nếu có quality score)
        quality = report.get("image_quality", {})
        if quality.get("is_blurry"):
            missing.append({
                "field":    "image_quality",
                "question": (
                    "📷 Ảnh hơi mờ nên hệ thống khó xác định mức ngập.\n"
                    "Bạn có thể gửi thêm một ảnh rõ hơn hoặc chụp từ góc rộng hơn không?"
                ),
                "options":  None,
                "required": False,
            })

        return missing

    @staticmethod
    def format_question(item: Dict) -> str:
        """Format câu hỏi thành text đẹp."""
        q = item["question"]
        if item.get("options"):
            opts = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(item["options"]))
            return f"{q}\n{opts}"
        return q


# ─────────────────────────────────────────────────────────────────────────────
# AGENT DECISION — JSON-first structured output  (#4, #5, #16)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentDecision:
    """
    Quyết định có cấu trúc của agent — JSON-first, render thành text sau.

    Thiết kế:
      - Agent KHÔNG bao giờ tự publish high alert
      - Luôn phân biệt "đã xác minh" / "đang chờ xác minh"
      - Lý do minh bạch → admin duyệt nhanh hơn
    """
    decision:         str   # "publish" | "needs_review" | "ask_for_more_info"
                            # | "reject" | "update_existing_event"
    alert_level:      str   # "none" | "low" | "medium" | "high" | "critical"
    should_publish:   bool
    needs_review:     bool
    publish_target:   List[str]      # ["map", "news", "alert", "social"]
    title:            str            # Tiêu đề chuẩn (qua EditorialPolicy)
    summary:          str            # Tóm tắt 1–2 câu
    reasons:          List[str]      # Lý do quyết định (cho admin)
    required_actions: List[str]      # Việc cần làm trước khi đăng
    public_message:   str            # Hiện cho người dân
    admin_note:       str            # Ghi chú nội bộ cho admin
    confidence_display: str          # VD: "Cần xác minh thêm"
    recommendations:  Dict[str, str] = field(default_factory=dict)  # xe/người
    report_id:        str  = ""
    event_id:         str  = ""      # nếu gộp vào event có sẵn
    alert_lifecycle:  str  = AlertLifecycle.NEW.value
    created_at:       str  = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_admin_report(self) -> str:
        """
        Render reasoning report cho admin (#5).
        Hiện đầy đủ lý do + việc cần làm.
        """
        icon = {"none": "✅", "low": "💧", "medium": "⚠️",
                "high": "🚨", "critical": "🆘"}.get(self.alert_level, "❓")
        lines = [
            f"{icon} **Agent đề xuất: {self._decision_vi()}**\n",
            f"📰 **Tiêu đề:** {self.title}",
            f"📋 **Tóm tắt:** {self.summary}",
            f"📊 **Mức cảnh báo:** {self.alert_level.upper()} | {self.confidence_display}\n",
            "**Lý do:**",
        ]
        for r in self.reasons:
            lines.append(f"  • {r}")
        if self.required_actions:
            lines.append("\n**Cần làm trước khi đăng:**")
            for a in self.required_actions:
                lines.append(f"  □ {a}")
        if self.recommendations:
            lines.append("\n" + FloodAdvisory.to_text(0))  # placeholder, gọi riêng nếu cần
        if self.admin_note:
            lines.append(f"\n📝 _Ghi chú nội bộ: {self.admin_note}_")
        return "\n".join(lines)

    def _decision_vi(self) -> str:
        return {
            "publish":              "Có thể đăng",
            "needs_review":         "Cần duyệt trước khi đăng",
            "ask_for_more_info":    "Cần thêm thông tin",
            "reject":               "Từ chối",
            "update_existing_event": "Cập nhật sự kiện hiện có",
        }.get(self.decision, self.decision)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG  (#18)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    """Một bản ghi audit cho mọi quyết định của agent."""
    agent:          str
    action:         str
    decision:       str
    input_snapshot: Dict
    reasons:        List[str]
    created_at:     str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    model_version:  str = "agent-v3"
    prompt_version: str = "v1"
    report_id:      str = ""
    user_id:        str = ""


class AuditLog:
    """
    Ghi log mọi quyết định agent xuống file JSONL.
    Mỗi dòng = 1 AuditEntry.
    Không bao giờ sửa hay xóa log cũ.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self._path = log_path or _AUDIT_FILE
        self._lock = threading.Lock()

    def log(self, entry: AuditEntry) -> None:
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            except Exception as e:
                log.warning(f"[AuditLog] Không ghi được: {e}")

    def log_decision(self, agent_name: str, action: str, decision: AgentDecision,
                     input_snap: Dict, report_id: str = "") -> None:
        """Shortcut tạo AuditEntry từ AgentDecision."""
        self.log(AuditEntry(
            agent          = agent_name,
            action         = action,
            decision       = decision.decision,
            input_snapshot = input_snap,
            reasons        = decision.reasons,
            report_id      = report_id,
        ))

    def recent(self, n: int = 20) -> List[Dict]:
        """Đọc n bản ghi gần nhất."""
        entries: List[Dict] = []
        try:
            if not self._path.exists():
                return []
            with open(self._path, encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines[-n:]):
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        except Exception as e:
            log.warning(f"[AuditLog] Đọc lỗi: {e}")
        return list(reversed(entries))


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK ENGINE — rule-based khi LLM/API lỗi  (#19)
# ─────────────────────────────────────────────────────────────────────────────

def fallback_alert_decision(result: Dict) -> Dict:
    """
    Quyết định rule-based thuần túy — dùng khi LLM/model lỗi.
    Hệ thống cảnh báo KHÔNG phụ thuộc hoàn toàn vào AI.
    """
    depth = float(result.get("water_height_cm", 0) or 0)
    conf  = float(result.get("confidence",      0) or 0)

    if conf < 0.65:
        return {
            "decision":    "needs_review",
            "alert_level": "none",
            "reason":      "low_confidence",
            "fallback":    True,
        }
    if depth >= 120:
        return {
            "decision":    "needs_review",   # cao → bắt buộc duyệt
            "alert_level": "high",
            "reason":      "high_depth_requires_verification",
            "fallback":    True,
        }
    if depth >= 60:
        return {
            "decision":    "needs_review",
            "alert_level": "medium",
            "reason":      "medium_depth",
            "fallback":    True,
        }
    if depth >= 15:
        return {
            "decision":    "publish",
            "alert_level": "low",
            "reason":      "minor_flood",
            "fallback":    True,
        }
    return {
        "decision":    "needs_review",
        "alert_level": "none",
        "reason":      "insufficient_depth",
        "fallback":    True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVENT CLUSTERER — gom báo cáo thành sự kiện  (#10)
# ─────────────────────────────────────────────────────────────────────────────

class EventClusterer:
    """
    Tránh tạo 10 bài cho 10 ảnh cùng điểm ngập.
    Nếu báo cáo mới gần cùng vị trí + gần thời gian → gộp vào event cũ.
    """

    TIME_WINDOW_MINUTES = 60
    LOCATION_OVERLAP_THRESHOLD = 0.5   # Tỷ lệ overlap từ khóa vị trí

    @classmethod
    def should_update_existing(
        cls,
        new_location: str,
        new_depth: float,
        existing_events: List[Dict],
    ) -> Optional[str]:
        """
        Trả về event_id nếu nên cập nhật event cũ, None nếu nên tạo mới.

        existing_events: list of {"event_id", "location", "depth_cm", "updated_at"}
        """
        for event in existing_events:
            if cls._same_area(new_location, event.get("location", "")):
                # Kiểm tra thời gian
                try:
                    evt_time = datetime.fromisoformat(event.get("updated_at", ""))
                    elapsed  = (datetime.now() - evt_time).total_seconds() / 60
                    if elapsed <= cls.TIME_WINDOW_MINUTES:
                        return event["event_id"]
                except Exception:
                    pass
        return None

    @classmethod
    def _same_area(cls, loc1: str, loc2: str) -> bool:
        """So sánh hai vị trí text — đơn giản theo từ khóa chung."""
        if not loc1 or not loc2:
            return False
        tokens1 = set(loc1.lower().split())
        tokens2 = set(loc2.lower().split())
        # Bỏ stop words ngắn
        tokens1 = {t for t in tokens1 if len(t) > 2}
        tokens2 = {t for t in tokens2 if len(t) > 2}
        if not tokens1 or not tokens2:
            return False
        overlap = len(tokens1 & tokens2) / min(len(tokens1), len(tokens2))
        return overlap >= cls.LOCATION_OVERLAP_THRESHOLD

    @classmethod
    def build_update_message(cls, event: Dict, new_depth: float,
                              new_result: Dict) -> Dict:
        """Sinh message cập nhật sự kiện."""
        old_depth = float(event.get("depth_cm", 0) or 0)
        delta     = new_depth - old_depth
        if delta > 5:
            change = f"tăng từ {int(old_depth)}cm lên khoảng {int(new_depth)}cm"
        elif delta < -5:
            change = f"giảm từ {int(old_depth)}cm xuống còn khoảng {int(new_depth)}cm"
        else:
            change = f"ổn định ở khoảng {int(new_depth)}cm"

        return {
            "action":     "update_event",
            "event_id":   event["event_id"],
            "update":     f"Mực nước {change}.",
            "new_depth":  new_depth,
            "lifecycle":  infer_alert_lifecycle(
                old_depth, new_depth,
                (datetime.now() - datetime.fromisoformat(
                    event.get("updated_at", datetime.now().isoformat())
                )).total_seconds() / 60
            ).value,
        }


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

    def __init__(self, cfg: Optional[Dict] = None,
                 permission_level: PermissionLevel = PermissionLevel.ANALYST):
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

        # v3 components
        self.permission  = permission_level
        self.audit_log   = AuditLog()
        self.clusterer   = EventClusterer()
        self._active_events: List[Dict] = []   # in-memory event store (thay bằng DB nếu cần)

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

    def process_images(self, image_paths: List[Path],
                       location: str = "", description: str = "",
                       report_id: str = "", has_gps: bool = False,
                       has_privacy_risk: bool = False) -> AgentResponse:
        """
        Nhận danh sách ảnh → phân tích → confidence-aware decision → AgentDecision.
        Entry point chính khi user upload ảnh.

        Tham số mới (v3):
            location:         Vị trí text từ user
            description:      Mô tả của user
            report_id:        ID báo cáo (sinh tự động nếu trống)
            has_gps:          Ảnh có GPS EXIF không
            has_privacy_risk: Có biển số / khuôn mặt không cần blur
        """
        with self._lock:
            self.memory.n_sessions += 1

        # Sinh report_id nếu chưa có
        if not report_id:
            report_id = f"rep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Kiểm tra thông tin thiếu trước khi xử lý
        report_meta = {
            "location": location, "gps": has_gps,
            "description": description,
        }
        missing = MissingInfoChecker.check(report_meta)
        required_missing = [m for m in missing if m.get("required")]
        if required_missing:
            questions = "\n\n".join(
                MissingInfoChecker.format_question(m) for m in required_missing
            )
            msg = f"⚠️ Cần thêm thông tin trước khi phân tích:\n\n{questions}"
            self.memory.add_message("agent", msg)
            # Vẫn phân tích nhưng báo thiếu thông tin
            log.info(f"[Agent] Missing required info: {[m['field'] for m in required_missing]}")

        self.memory.add_message(
            "user", f"[Upload] {len(image_paths)} ảnh: "
            + ", ".join(p.name for p in image_paths[:5])
            + (f" | vị trí: {location}" if location else "")
        )
        log.info(f"[Agent] process_images: {len(image_paths)} ảnh, report_id={report_id}")

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

        # Cảnh báo thông tin thiếu (optional fields)
        optional_missing = [m for m in missing if not m.get("required")]
        if optional_missing and not required_missing:
            q = MissingInfoChecker.format_question(optional_missing[0])
            extra_msg += f"\n\n💬 _{q}_"

        if results:
            # Gán image_id và lưu vào history
            for r in results:
                img_id = self.memory.add_image_to_history(r, image_paths)
                r["image_id"] = img_id
            first = results[0]
            conf  = float(first.get("confidence", 0.5) or 0.5)
            self.memory.last_action_type = "analyze"
            self.memory.set_last_result(first, image_paths)

            # ── Tạo AgentDecision (v3) ────────────────────────────
            try:
                decision = self.make_decision(
                    result           = first,
                    location         = location,
                    n_reports_nearby = len(self._active_events) + 1,
                    has_privacy_risk = has_privacy_risk,
                    has_gps          = has_gps,
                    report_id        = report_id,
                )
                extra_msg += "\n\n" + decision.to_admin_report()

                # Cập nhật ReportState
                state = self.get_report_state(report_id, str(image_paths[0]),
                                              location=location, description=description)
                state.analysis_result = first
                state.alert_decision  = decision.to_dict()
                if decision.decision == "needs_review":
                    state.advance("pending_review")
                elif decision.decision == "update_existing_event":
                    state.advance("pending_review")
                else:
                    state.advance("analyzed")

                # Cập nhật _active_events nếu là event mới
                if decision.event_id:
                    # Cập nhật event hiện có
                    for ev in self._active_events:
                        if ev.get("event_id") == decision.event_id:
                            ev.update({"depth_cm": first.get("water_height_cm", 0),
                                       "updated_at": datetime.now().isoformat()})
                else:
                    # Thêm event mới
                    self._active_events.append({
                        "event_id":  report_id,
                        "location":  location,
                        "depth_cm":  first.get("water_height_cm", 0),
                        "updated_at": datetime.now().isoformat(),
                    })

            except Exception as e:
                log.warning(f"[Agent] make_decision lỗi ({e}), dùng fallback")
                fb = fallback_alert_decision(first)
                extra_msg += (
                    f"\n\n⚠️ _Dùng fallback rule-based_: "
                    f"decision={fb['decision']} level={fb['alert_level']}"
                )

            # ── Confidence-aware behavior (giữ từ v2) ─────────────
            if conf > self.config.conf_auto_confirm:
                self.tools.call("queue_review", result=first, reason="auto_high_conf")
                log.info(f"[Agent] Auto-confirm (conf={conf:.2f})")

            elif conf < self.config.conf_auto_rerun:
                log.info(f"[Agent] Auto-rerun do conf={conf:.2f}")
                rerun_tr = self.tools.call("rerun", images=image_paths)
                if rerun_tr.success and rerun_tr.data.get("results"):
                    new_results = rerun_tr.data["results"]
                    delta = self._compare_results(first, new_results[0])
                    results = new_results
                    self.memory.set_last_result(new_results[0])
                    extra_msg += (
                        f"\n🔄 _Tự động chạy lại (conf={round(conf*100)}%). "
                        f"Kết quả mới: {'+' if delta['depth_delta'] >= 0 else ''}"
                        f"{delta['depth_delta']}cm._"
                    )

            elif conf < self.config.conf_ask_user:
                self.events.emit("low_confidence", {"result": first})

        # Build base message — dùng ResponseRewriter (v3) thay vì format cũ
        if results:
            first_result = results[0]
            missing_field_names = [m["field"] for m in missing if not m.get("required")] \
                                  if missing else []

            # Thử dùng phong cách trả lời đã học trước
            ctx_key = _make_context_key(results, "analyze")
            pattern = self.memory.find_matching_pattern(ctx_key)
            if pattern:
                try:
                    base_msg = _fill_response_template(pattern["template"], results, duration)
                    base_msg += "\n_💾 [Đang dùng phong cách trả lời đã lưu]_"
                except Exception:
                    base_msg = self.build_natural_response(
                        result          = first_result,
                        decision        = None,
                        mode            = "public_user",
                        location        = location,
                        missing_fields  = missing_field_names or None,
                    )
            else:
                base_msg = self.build_natural_response(
                    result          = first_result,
                    decision        = None,
                    mode            = "public_user",
                    location        = location,
                    missing_fields  = missing_field_names or None,
                )
        else:
            base_msg = "ℹ️ Không có kết quả phân tích."

        msg = base_msg + extra_msg

        # Cảnh báo thiếu thông tin bắt buộc ở đầu
        if required_missing:
            questions = "\n".join(
                MissingInfoChecker.format_question(m) for m in required_missing
            )
            msg = f"⚠️ **Thiếu thông tin quan trọng:**\n{questions}\n\n---\n\n" + msg

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

    # ── v3 Public API ─────────────────────────────────────────────

    def make_decision(
        self,
        result: Dict,
        location: str = "",
        n_reports_nearby: int = 1,
        has_privacy_risk: bool = False,
        has_gps: bool = False,
        report_id: str = "",
    ) -> AgentDecision:
        """
        Tạo AgentDecision có cấu trúc từ kết quả phân tích.
        Đây là JSON-first output — render thành text qua to_admin_report().

        Không bao giờ tự publish high alert mà không có admin duyệt.
        """
        conf    = float(result.get("confidence",      0) or 0)
        depth   = float(result.get("water_height_cm", 0) or 0)
        level   = result.get("flood_level", "UNKNOWN")
        reasons:          List[str] = []
        required_actions: List[str] = []
        publish_target:   List[str] = ["map"]

        # === Phân tích lý do ===
        if depth > 0:
            reasons.append(f"AI nhận diện mực nước ước tính {int(depth)}cm ({level}).")
        if conf >= 0.85:
            reasons.append("Độ tin cậy cao — có vật tham chiếu rõ ràng trong ảnh.")
        elif conf >= 0.65:
            reasons.append("Độ tin cậy trung bình — cần xác minh thêm từ nguồn khác.")
        else:
            reasons.append(f"Độ tin cậy thấp ({round(conf*100)}%) — chưa đủ bằng chứng.")

        if not location and not has_gps:
            reasons.append("Vị trí chỉ được nhập bằng text, chưa có GPS.")
            required_actions.append("Xác minh vị trí chính xác (GPS hoặc địa chỉ đầy đủ).")

        if has_privacy_risk:
            reasons.append("Ảnh có thể chứa thông tin nhận dạng cá nhân (biển số xe, khuôn mặt).")
            required_actions.append("Dùng bản ảnh đã blur trước khi đăng công khai.")

        if n_reports_nearby > 1:
            reasons.append(f"Có {n_reports_nearby} báo cáo gần khu vực này.")

        # Kiểm tra event hiện có
        existing_event_id = self.clusterer.should_update_existing(
            location, depth, self._active_events
        )

        # === Quyết định chính ===
        try:
            if depth >= 120 or (depth >= 60 and not has_gps):
                # Mức cao hoặc chưa có GPS + mức trung → bắt buộc duyệt
                AgentPermissions.require(self.permission, "publish_high_alert")
                # Nếu đến đây → ADMIN mode
                decision    = "publish"
                needs_review = False
                alert_level = "high" if depth >= 120 else "medium"
                publish_target = ["map", "news", "alert"]
            elif conf < 0.65:
                decision    = "needs_review"
                needs_review = True
                alert_level = "none"
            elif existing_event_id:
                decision    = "update_existing_event"
                needs_review = True
                alert_level = "low" if depth >= 30 else "none"
            elif required_actions:
                decision    = "needs_review"
                needs_review = True
                alert_level = "medium" if depth >= 60 else "low"
                publish_target.append("news")
            else:
                decision    = "publish"
                needs_review = False
                alert_level = "low" if depth < 60 else "medium"
                publish_target.append("news")

        except PermissionError:
            # Không đủ quyền → bắt buộc đưa vào review
            decision     = "needs_review"
            needs_review = True
            alert_level  = "high"
            reasons.append("Agent không đủ quyền tự đăng cảnh báo mức cao — cần admin duyệt.")
            required_actions.append("Admin duyệt trước khi đăng cảnh báo.")

        # Fallback nếu logic lỗi
        should_publish = (decision == "publish") and not required_actions

        # Editorial policy
        title   = EditorialPolicy.generate_title(result, location, verified=not needs_review)
        summary = EditorialPolicy.generate_summary(result, n_reports_nearby)

        decision_obj = AgentDecision(
            decision          = decision,
            alert_level       = alert_level,
            should_publish    = should_publish,
            needs_review      = needs_review,
            publish_target    = publish_target,
            title             = title,
            summary           = summary,
            reasons           = reasons,
            required_actions  = required_actions,
            public_message    = ConfidenceTranslator.status_label(conf),
            admin_note        = f"confidence={round(conf*100)}%, depth={depth}cm, level={level}",
            confidence_display = ConfidenceTranslator.to_admin(conf),
            recommendations   = FloodAdvisory.generate(depth),
            report_id         = report_id,
            event_id          = existing_event_id or "",
        )

        # Ghi audit log
        self.audit_log.log_decision(
            agent_name = "FloodAgent.make_decision",
            action     = "make_decision",
            decision   = decision_obj,
            input_snap = {
                "depth":    depth, "conf": conf, "level": level,
                "location": location, "has_gps": has_gps,
            },
            report_id  = report_id,
        )

        log.info(
            f"[Decision] {decision} | alert={alert_level} | "
            f"depth={depth}cm conf={round(conf*100)}% "
            f"{'→ update event' if existing_event_id else ''}"
        )
        return decision_obj

    def get_recommendations(self, depth_cm: float) -> Dict[str, str]:
        """Trả về khuyến cáo theo đối tượng."""
        return FloodAdvisory.generate(depth_cm)

    def get_report_state(self, report_id: str,
                          image_path: str, source: str = "citizen_upload",
                          location: str = "", description: str = "") -> ReportState:
        """Tạo ReportState mới cho một báo cáo đến."""
        return ReportState(
            report_id       = report_id,
            source          = source,
            image_path      = image_path,
            raw_description = description,
            raw_location    = location,
        )

    def check_missing_info(self, report: Dict) -> List[Dict]:
        """Trả về danh sách thông tin còn thiếu và câu hỏi hỏi lại."""
        return MissingInfoChecker.check(report)

    def get_audit_trail(self, n: int = 20) -> List[Dict]:
        """Lấy n bản ghi audit gần nhất."""
        return self.audit_log.recent(n)

    def build_natural_response(
        self,
        result:         Dict,
        decision:       Any  = None,
        mode:           str  = "public_user",
        location:       str  = "",
        missing_fields: Optional[List[str]] = None,
    ) -> str:
        """
        Public helper: Decision JSON → natural language response.
        Tích hợp ResponseRewriter + ResponseValidator với 1 lần retry.

        Flow:
            rewrite(decision, result, mode)
            → validate(natural_response, mode)
            → nếu fail: rewrite lại với simplify=True
            → trả về natural_response cuối

        Dùng ở bất kỳ đâu cần text thân thiện:
            msg = agent.build_natural_response(result, decision, mode="public_user")
        """
        try:
            from agent.response_rewriter  import ResponseRewriter, ResponseMode
            from agent.response_validator import ResponseValidator
        except ImportError:
            # Fallback: dùng format cũ nếu file chưa có
            return _format_analysis_response([result])

        rewritten = ResponseRewriter.rewrite(
            decision        = decision or {},
            result          = result,
            mode            = mode,
            location        = location,
            missing_fields  = missing_fields,
        )

        # Validate — retry 1 lần với simplify=True nếu fail
        valid = ResponseValidator.validate(rewritten.natural_response, mode=mode)
        if not valid and valid.severity == "fail":
            log.debug(f"[Rewriter] Validate fail ({valid.issues}), retry với simplify=True")
            rewritten = ResponseRewriter.rewrite(
                decision        = decision or {},
                result          = result,
                mode            = mode,
                location        = location,
                missing_fields  = missing_fields,
                simplify        = True,
            )

        return rewritten.natural_response

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
            "📖 **Hướng dẫn sử dụng FloodAgent v3**\n\n"
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
            "**Tính năng v3 mới:**\n"
            "• AgentDecision JSON: quyết định minh bạch + lý do đầy đủ\n"
            "• EditorialPolicy: tự động kiểm tra ngôn ngữ trước khi đăng\n"
            "• ReportState: theo dõi trạng thái từng báo cáo\n"
            "• FloodAdvisory: khuyến cáo theo xe máy / ô tô / người đi bộ\n"
            "• EventClusterer: gộp báo cáo gần nhau thành 1 sự kiện\n"
            "• AuditLog: ghi log mọi quyết định vào file\n"
            "• PermissionLevel: phân quyền analyst/editor/admin\n"
            "• Fallback rule-based: hoạt động ngay cả khi AI lỗi\n"
            "• Hỏi lại khi thiếu vị trí hoặc thời gian\n"
        )

    # ── Convenience: load config từ file ─────────────────────────

    @classmethod
    def from_config_file(cls, config_path: str = "config.yaml",
                         permission_level: PermissionLevel = PermissionLevel.ANALYST
                         ) -> "FloodAgent":
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        return cls(cfg, permission_level=permission_level)
