# -*- coding: utf-8 -*-
"""
agent/conversation_state.py
============================
3 class cốt lõi để agent có "bộ nhớ ngữ cảnh" trong 1 cuộc trò chuyện:

  ConversationState — theo dõi những gì đã biết / đã hỏi trong phiên
  ResponsePlanner   — quyết định kiểu phản hồi phù hợp trước khi viết
  ResponseSchema    — cấu trúc intent trước khi render thành text (#13)

Thêm:
  ResponseLength    — short | normal | detail (#14)

Flow chuẩn:

    state   = ConversationState()
    planner = ResponsePlanner()

    # User gửi ảnh
    state.mark_image_received()

    # Agent chọn kiểu trả lời
    rtype = planner.plan(state, result, user_role="public_user")

    # Agent tạo schema
    schema = planner.build_schema(rtype, result, state, location="cổng trường")

    # ResponseRewriter render schema thành text
    text = ResponseRewriter.render_schema(schema, mode="public_user")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE TYPE  (#4)
# ─────────────────────────────────────────────────────────────────────────────

class ResponseType(Enum):
    INFORM          = "inform"           # Thông báo kết quả
    ASK_LOCATION    = "ask_location"     # Hỏi địa điểm
    ASK_TIME        = "ask_time"         # Hỏi thời gian
    ASK_BETTER_IMG  = "ask_better_image" # Yêu cầu ảnh rõ hơn
    WARN            = "warn"             # Cảnh báo nhanh
    SUMMARIZE       = "summarize"        # Tóm tắt cho admin
    DRAFT_NEWS      = "draft_news"       # Viết bản tin
    REJECT          = "reject"           # Từ chối / không đủ dữ liệu
    UPDATE_EVENT    = "update_event"     # Cập nhật sự kiện cũ
    EXPLAIN         = "explain"          # Giải thích quyết định (#16)


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE LENGTH  (#14)
# ─────────────────────────────────────────────────────────────────────────────

class ResponseLength(Enum):
    SHORT  = "short"   # 1–2 câu, không lý do
    NORMAL = "normal"  # 3–5 câu, có mức chắc + khuyến cáo
    DETAIL = "detail"  # Đầy đủ: lý do + hành động + advisory


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE SCHEMA  (#13)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResponseSchema:
    """
    Cấu trúc intent trước khi render thành text.
    Agent tạo schema → ResponseRewriter render → text tự nhiên.

    Ưu điểm:
      - Dễ test: kiểm tra schema thay vì text
      - Dễ debug: thấy ngay intent là gì
      - Dễ override: admin sửa schema trước khi render

    Ví dụ:
        ResponseSchema(
            intent      = ResponseType.INFORM,
            audience    = "public_user",
            tone        = "friendly_calm",
            main_message= "Khu vực này ngập khoảng tới đầu gối.",
            question    = None,
            safety_note = "Xe máy nên hạn chế đi qua.",
            action_hint = None,
        )
    """
    intent:       ResponseType
    audience:     str                     # "public_user" | "admin_review" | "news_writer"
    tone:         str                     # "friendly_calm" | "professional" | "neutral_journalistic"
    main_message: str                     # Câu kết luận chính
    question:     Optional[str]   = None  # Câu hỏi lại (chỉ 1)
    safety_note:  Optional[str]   = None  # Khuyến cáo an toàn
    action_hint:  Optional[str]   = None  # Gợi ý hành động (cho admin)
    length:       ResponseLength  = ResponseLength.NORMAL
    metadata:     Dict            = field(default_factory=dict)

    def render(self) -> str:
        """Render schema thành text tự nhiên."""
        parts: List[str] = []

        if self.main_message:
            parts.append(self.main_message)

        if self.safety_note:
            parts.append(self.safety_note)

        if self.action_hint and self.audience in ("admin_review", "debug"):
            parts.append(self.action_hint)

        if self.question:
            parts.append(self.question)

        return " ".join(p.strip() for p in parts if p.strip())

    def to_dict(self) -> Dict:
        return {
            "intent":       self.intent.value,
            "audience":     self.audience,
            "tone":         self.tone,
            "main_message": self.main_message,
            "question":     self.question,
            "safety_note":  self.safety_note,
            "action_hint":  self.action_hint,
            "length":       self.length.value,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION STATE  (#5)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationState:
    """
    Theo dõi trạng thái trong một cuộc trò chuyện.
    Tránh hỏi lại thứ đã biết, tránh cảnh báo trùng.

    Thường 1 instance / session user. Reset khi user mới.
    """
    # Thông tin đã nhận
    has_image:          bool = False
    location_provided:  bool = False
    time_provided:      bool = False
    description_provided: bool = False
    has_gps:            bool = False

    # Những gì agent đã hỏi
    asked_location:     bool = False
    asked_time:         bool = False
    asked_better_image: bool = False

    # Những gì agent đã thông báo
    warned_high_level:  bool = False
    gave_advisory:      bool = False

    # Context hiện tại
    location:           str  = ""
    last_level:         str  = "UNKNOWN"
    last_confidence:    float = 0.0
    last_depth_cm:      float = 0.0
    n_images_received:  int  = 0
    n_turns:            int  = 0
    user_role:          str  = "public_user"  # "public_user" | "admin" | "news_writer"

    created_at:  str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at:  str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def _touch(self) -> None:
        self.updated_at = datetime.now().isoformat(timespec="seconds")
        self.n_turns   += 1

    # ── Setters ──────────────────────────────────────────────────

    def mark_image_received(self, quality_ok: bool = True) -> None:
        self.has_image          = True
        self.n_images_received += 1
        self._touch()

    def mark_location_provided(self, location: str) -> None:
        self.location_provided = True
        self.location          = location
        self._touch()

    def mark_time_provided(self) -> None:
        self.time_provided = True
        self._touch()

    def mark_gps_available(self) -> None:
        self.has_gps = True
        self._touch()

    def mark_asked_location(self) -> None:
        self.asked_location = True
        self._touch()

    def mark_asked_time(self) -> None:
        self.asked_time = True
        self._touch()

    def mark_asked_better_image(self) -> None:
        self.asked_better_image = True
        self._touch()

    def mark_warned(self) -> None:
        self.warned_high_level = True
        self._touch()

    def mark_advisory_given(self) -> None:
        self.gave_advisory = True
        self._touch()

    def update_analysis(self, level: str, confidence: float, depth_cm: float) -> None:
        self.last_level      = level
        self.last_confidence = confidence
        self.last_depth_cm   = depth_cm
        self._touch()

    # ── Queries ──────────────────────────────────────────────────

    @property
    def missing_fields(self) -> List[str]:
        """Danh sách field còn thiếu (chưa hỏi)."""
        missing = []
        if not self.location_provided and not self.asked_location:
            missing.append("location")
        if not self.time_provided and not self.asked_time:
            missing.append("timestamp")
        return missing

    @property
    def is_high_severity(self) -> bool:
        return self.last_level in ("WAIST", "CHEST", "SUBMERGED")

    @property
    def should_give_advisory(self) -> bool:
        return self.last_depth_cm >= 15 and not self.gave_advisory

    @property
    def next_question_field(self) -> Optional[str]:
        """Field quan trọng nhất cần hỏi tiếp theo (chỉ 1)."""
        if not self.location_provided and not self.asked_location:
            return "location"
        if not self.time_provided and not self.asked_time and self.n_turns >= 2:
            return "timestamp"
        return None

    def to_dict(self) -> Dict:
        return {
            "has_image":         self.has_image,
            "location_provided": self.location_provided,
            "time_provided":     self.time_provided,
            "has_gps":           self.has_gps,
            "asked_location":    self.asked_location,
            "asked_time":        self.asked_time,
            "warned_high":       self.warned_high_level,
            "location":          self.location,
            "last_level":        self.last_level,
            "last_depth_cm":     self.last_depth_cm,
            "last_confidence":   round(self.last_confidence, 2),
            "n_turns":           self.n_turns,
            "user_role":         self.user_role,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PLANNER  (#4)
# ─────────────────────────────────────────────────────────────────────────────

# Khuyến cáo an toàn theo level
_SAFETY_NOTES: Dict[str, str] = {
    "PUDDLE":    "Người đi bộ chú ý trơn trượt.",
    "ANKLE":     "Xe máy qua được nhưng nên đi thật chậm.",
    "KNEE":      "Xe máy nên hạn chế đi qua. Ô tô gầm thấp cũng cần thận trọng.",
    "WAIST":     "Người đi bộ và xe máy không nên đi qua khu vực này.",
    "CHEST":     "Tuyệt đối không đi qua. Nguy hiểm cho cả người lẫn phương tiện.",
    "SUBMERGED": "Không được đi qua trong bất kỳ trường hợp nào.",
}

# Câu hỏi lại chuẩn theo field
_FOLLOW_UP_QUESTIONS: Dict[str, str] = {
    "location": (
        "Bạn cho mình biết ảnh này chụp ở đâu được không? "
        "Tên đường, cổng trường, chợ hoặc phường/xã đều được."
    ),
    "timestamp": "Ảnh này vừa chụp hay là ảnh cũ vậy bạn?",
    "image_quality": (
        "Ảnh hơi mờ nên mình chưa đủ chắc để xác định mức ngập. "
        "Bạn có thể gửi thêm một ảnh rộng hơn không?"
    ),
}


class ResponsePlanner:
    """
    Quyết định kiểu phản hồi trước khi viết — không viết text.

    Logic:
        1. Không có ảnh → hướng dẫn gửi ảnh
        2. Ảnh mờ → hỏi ảnh tốt hơn
        3. Thiếu địa điểm + chưa hỏi → hỏi địa điểm
        4. Confidence thấp → reject / xin thêm info
        5. Mức cao + admin → summarize
        6. Mức cao + public + chưa warn → warn
        7. Mức bình thường → inform
        8. Admin xem báo cáo → summarize
        9. News writer → draft_news
    """

    @classmethod
    def plan(
        cls,
        state:      ConversationState,
        result:     Optional[Dict] = None,
        user_role:  str = "public_user",
        image_ok:   bool = True,
        decision:   Optional[Dict] = None,
    ) -> ResponseType:
        """
        Chọn ResponseType phù hợp nhất.

        Args:
            state:     ConversationState hiện tại
            result:    kết quả phân tích (có thể None nếu chưa phân tích)
            user_role: "public_user" | "admin" | "news_writer"
            image_ok:  ảnh có chất lượng đủ dùng không
            decision:  AgentDecision dict (có thể None)
        """
        # Không có ảnh → inform / hướng dẫn
        if not state.has_image:
            return ResponseType.INFORM

        # Ảnh mờ → hỏi ảnh tốt hơn (chỉ hỏi 1 lần)
        if not image_ok and not state.asked_better_image:
            return ResponseType.ASK_BETTER_IMG

        conf  = float((result or {}).get("confidence", 0) or 0) if result else 0
        level = str((result or {}).get("flood_level", "UNKNOWN") or "UNKNOWN").upper()
        depth = float((result or {}).get("water_height_cm", 0) or 0) if result else 0

        # Confidence quá thấp → reject
        if conf < 0.4 and result:
            return ResponseType.REJECT

        # News writer → draft
        if user_role == "news_writer":
            return ResponseType.DRAFT_NEWS

        # Admin
        if user_role == "admin":
            dec_val = (decision or {}).get("decision", "")
            if dec_val == "update_existing_event":
                return ResponseType.UPDATE_EVENT
            if dec_val == "needs_review":
                return ResponseType.EXPLAIN
            return ResponseType.SUMMARIZE

        # Public user — hỏi địa điểm nếu chưa có (ưu tiên hơn cảnh báo)
        if not state.location_provided and not state.asked_location:
            return ResponseType.ASK_LOCATION

        # Public — cảnh báo nếu mức cao và chưa warn
        if level in ("WAIST", "CHEST", "SUBMERGED") and not state.warned_high_level:
            return ResponseType.WARN

        # Public — hỏi thời gian (sau lượt 2 nếu chưa hỏi)
        if state.n_turns >= 2 and not state.time_provided and not state.asked_time:
            return ResponseType.ASK_TIME

        return ResponseType.INFORM

    @classmethod
    def build_schema(
        cls,
        rtype:    ResponseType,
        result:   Optional[Dict],
        state:    ConversationState,
        location: str = "",
        decision: Optional[Dict] = None,
        length:   ResponseLength = ResponseLength.NORMAL,
    ) -> ResponseSchema:
        """
        Xây dựng ResponseSchema từ ResponseType + context.
        ResponseRewriter sẽ render schema → text.
        """
        from agent.response_rewriter import confidence_label, depth_range as _dr

        result   = result or {}
        decision = decision or {}
        conf     = float(result.get("confidence", 0) or 0)
        depth    = float(result.get("water_height_cm", 0) or 0)
        level    = str(result.get("flood_level", "UNKNOWN") or "UNKNOWN").upper()
        loc      = location or state.location or ""
        conf_lbl = confidence_label(conf)
        d_range  = _dr(depth) if depth > 0 else ""
        audience = state.user_role if state.user_role != "public_user" else "public_user"

        # ── Tone theo audience ─────────────────────────────────────
        _TONES = {
            "public_user": "friendly_calm",
            "admin":       "professional",
            "news_writer": "neutral_journalistic",
        }
        tone = _TONES.get(audience, "friendly_calm")

        # ── Xây dựng main_message theo ResponseType ───────────────
        _LEVEL_NATURAL = {
            "NO_FLOOD":  "không có dấu hiệu ngập",
            "PUDDLE":    "có vũng nước nhỏ",
            "ANKLE":     "ngập khoảng mắt cá chân",
            "KNEE":      "ngập khoảng tới đầu gối",
            "WAIST":     "ngập khoảng tới thắt lưng",
            "CHEST":     "ngập rất sâu, khoảng tới ngực",
            "SUBMERGED": "ngập hoàn toàn",
            "UNKNOWN":   "chưa xác định được mức ngập",
        }
        level_nat = _LEVEL_NATURAL.get(level, level)
        loc_part  = f" tại {loc}" if loc else ""

        if rtype == ResponseType.INFORM:
            if depth > 0:
                main = (
                    f"Ảnh cho thấy{loc_part} {level_nat}, ước tính {d_range}. "
                    f"Kết quả {conf_lbl}."
                )
            else:
                main = f"Ảnh cho thấy{loc_part} {level_nat}."

        elif rtype == ResponseType.WARN:
            main = (
                f"⚠️ Lưu ý:{loc_part} {level_nat}, ước tính {d_range}. "
                f"Đây là mức ngập đáng lo ngại."
            )

        elif rtype == ResponseType.SUMMARIZE:
            reasons = decision.get("reasons", [])
            req     = decision.get("required_actions", [])
            reason_str = (" ".join(f"• {r}" for r in reasons[:3])) if reasons else ""
            action_str = (" ".join(f"□ {a}" for a in req[:2])) if req else ""
            main = (
                f"Hệ thống ghi nhận{loc_part} {level_nat}, ước tính {d_range}. "
                f"Độ tin cậy {conf_lbl} ({round(conf*100)}%)."
            )
            schema = ResponseSchema(
                intent       = rtype,
                audience     = audience,
                tone         = tone,
                main_message = main,
                action_hint  = f"Đề xuất: {decision.get('decision','needs_review')}.\n"
                               + (reason_str + "\n" + action_str).strip(),
                length       = length,
            )
            return schema

        elif rtype == ResponseType.DRAFT_NEWS:
            verified = decision.get("decision") == "publish"
            status   = "" if verified else " (đang chờ xác minh)"
            main = (
                f"Ghi nhận{loc_part} {level_nat}, ước tính {d_range}{status}."
            )

        elif rtype == ResponseType.UPDATE_EVENT:
            main = (
                f"Cập nhật{loc_part}: mực nước {level_nat}, ước tính {d_range}."
            )

        elif rtype == ResponseType.EXPLAIN:
            reasons = decision.get("reasons", ["Chưa đủ thông tin để đăng công khai."])
            main = (
                f"Mình chưa khuyên đăng tin này ngay. "
                + " ".join(reasons[:2])
            )
            req_acts = decision.get("required_actions", [])
            action_str = (
                "Có thể " + req_acts[0].lower() + " trước."
                if req_acts else "Đưa vào hàng chờ xác minh."
            )
            schema = ResponseSchema(
                intent       = rtype,
                audience     = audience,
                tone         = tone,
                main_message = main,
                action_hint  = action_str,
                length       = length,
            )
            return schema

        elif rtype == ResponseType.REJECT:
            main = (
                "Mình chưa đủ cơ sở để kết luận mức ngập từ ảnh này. "
                f"Kết quả {conf_lbl}."
            )

        elif rtype == ResponseType.ASK_LOCATION:
            level_desc = (
                f"có dấu hiệu {level_nat}"
                if level not in ('UNKNOWN', 'NO_FLOOD') else "đã nhận được"
            )
            return ResponseSchema(
                intent       = ResponseType.ASK_LOCATION,
                audience     = audience,
                tone         = tone,
                main_message = f"Ảnh {level_desc}.",
                question     = _FOLLOW_UP_QUESTIONS["location"],
                length       = ResponseLength.SHORT,
            )

        elif rtype == ResponseType.ASK_TIME:
            return ResponseSchema(
                intent       = ResponseType.ASK_TIME,
                audience     = audience,
                tone         = tone,
                main_message = "",
                question     = _FOLLOW_UP_QUESTIONS["timestamp"],
                length       = ResponseLength.SHORT,
            )

        elif rtype == ResponseType.ASK_BETTER_IMG:
            return ResponseSchema(
                intent       = ResponseType.ASK_BETTER_IMG,
                audience     = audience,
                tone         = tone,
                main_message = "",
                question     = _FOLLOW_UP_QUESTIONS["image_quality"],
                length       = ResponseLength.SHORT,
            )

        else:
            main = f"Ảnh cho thấy{loc_part} {level_nat}."

        # Safety note
        safety = _SAFETY_NOTES.get(level, "") if depth >= 15 else ""

        # Question (chỉ thêm nếu chưa hỏi)
        question = None
        if rtype == ResponseType.INFORM and state.next_question_field:
            field_name = state.next_question_field
            question   = _FOLLOW_UP_QUESTIONS.get(field_name)

        return ResponseSchema(
            intent       = rtype,
            audience     = audience,
            tone         = tone,
            main_message = main,
            question     = question,
            safety_note  = safety if safety else None,
            length       = length,
        )
