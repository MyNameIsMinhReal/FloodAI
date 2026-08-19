# -*- coding: utf-8 -*-
"""
agent/response_rewriter.py
===========================
Tách biệt hoàn toàn giữa:
  Decision Agent  → đưa ra kết luận JSON (AgentDecision)
  ResponseRewriter → viết lại thành ngôn ngữ tự nhiên

Flow:
    decision = flood_agent.make_decision(result, ...)
    rewritten = ResponseRewriter.rewrite(decision, result, mode="public_user")
    ok, issues = ResponseValidator.validate(rewritten.natural_response, mode="public_user")
    if not ok:
        rewritten = ResponseRewriter.rewrite(decision, result, mode="public_user", simplify=True)

Tính năng:
  - 5 response modes: public_user | admin_review | news_writer | alert_message | debug
  - Template variants per level — random nhẹ để tránh lặp
  - Cấu trúc 4 phần: xác nhận → kết luận → mức chắc → khuyến cáo
  - confidence_label() tự nhiên thay vì số
  - Hedging phrases tự động: "ảnh cho thấy", "ước tính", "có khả năng"
  - Câu hỏi lại tự nhiên khi thiếu thông tin
  - ResponseMode.DEBUG vẫn hiện đủ kỹ thuật
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODES
# ─────────────────────────────────────────────────────────────────────────────

class ResponseMode(Enum):
    PUBLIC_USER    = "public_user"    # Người dân: đơn giản, thân thiện
    ADMIN_REVIEW   = "admin_review"   # Admin: rõ lý do, có thông tin kỹ thuật nhẹ
    NEWS_WRITER    = "news_writer"    # Văn phong tin tức, trung lập
    ALERT_MESSAGE  = "alert_message"  # Ngắn, rõ, ưu tiên an toàn
    DEBUG          = "debug"          # Kỹ thuật, đầy đủ chi tiết


# ─────────────────────────────────────────────────────────────────────────────
# NATURAL LANGUAGE MAPS
# ─────────────────────────────────────────────────────────────────────────────

_LEVEL_NATURAL: Dict[str, str] = {
    "NO_FLOOD":  "không có dấu hiệu ngập",
    "PUDDLE":    "có vũng nước nhỏ",
    "ANKLE":     "ngập khoảng mắt cá chân",
    "KNEE":      "ngập khoảng tới đầu gối",
    "WAIST":     "ngập khoảng tới thắt lưng",
    "CHEST":     "ngập rất sâu, khoảng tới ngực",
    "SUBMERGED": "ngập hoàn toàn",
    "UNKNOWN":   "chưa xác định được mức ngập",
}

_LEVEL_EMOJI: Dict[str, str] = {
    "NO_FLOOD":  "✅",
    "PUDDLE":    "💧",
    "ANKLE":     "🌊",
    "KNEE":      "🌊",
    "WAIST":     "⚠️",
    "CHEST":     "🚨",
    "SUBMERGED": "🆘",
    "UNKNOWN":   "❓",
}

_ALERT_LEVEL_VI: Dict[str, str] = {
    "none":     "Không cần cảnh báo",
    "low":      "Cảnh báo mức nhẹ",
    "medium":   "Cảnh báo mức trung bình",
    "high":     "Cảnh báo nghiêm trọng",
    "critical": "Cảnh báo khẩn cấp",
}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE LABEL (natural language)
# ─────────────────────────────────────────────────────────────────────────────

def confidence_label(score: float) -> str:
    """
    Map điểm kỹ thuật → tiếng Việt tự nhiên.

    Dùng trong câu trả lời thay vì hiện số:
        confidence_label(0.78) → "tương đối tin cậy"
    """
    if score >= 0.85:
        return "khá chắc chắn"
    if score >= 0.65:
        return "tương đối tin cậy"
    if score >= 0.45:
        return "chưa thật sự chắc chắn"
    return "chưa đủ cơ sở kết luận"


def depth_range(depth_cm: float, margin_pct: float = 0.15) -> str:
    """
    Chuyển độ sâu đơn lẻ → dải ước tính (tự nhiên hơn).

    depth_range(52) → "45–60cm"
    """
    lo = max(0, int(depth_cm * (1 - margin_pct)))
    hi = int(depth_cm * (1 + margin_pct))
    # Làm tròn đến 5cm gần nhất
    lo = round(lo / 5) * 5
    hi = round(hi / 5) * 5
    if lo == hi:
        return f"khoảng {lo}cm"
    return f"{lo}–{hi}cm"


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE VARIANTS  (item #5)
# ─────────────────────────────────────────────────────────────────────────────

# Câu mở đầu (xác nhận ảnh đã nhận)
_ACKNOWLEDGEMENTS: List[str] = [
    "Mình đã xem ảnh rồi.",
    "Ảnh đã được phân tích.",
    "Mình đã nhận được ảnh.",
    "Mình vừa xem qua ảnh.",
    "Đã xem xong ảnh bạn gửi.",
    "Ảnh bạn gửi mình đã xem hết rồi.",
]

# Template chính theo level — có {depth_range} và {location} placeholder
_LEVEL_TEMPLATES: Dict[str, List[str]] = {
    "NO_FLOOD": [
        "Khu vực {location}trông khô ráo, không có dấu hiệu ngập đáng kể.",
        "Ảnh cho thấy {location}chưa có dấu hiệu ngập.",
        "Hệ thống không ghi nhận nước ngập tại {location}.",
        "Nhìn ảnh thì {location}hơi khô ráo, không thấy ngập đâu.",
        "{location}trông khô khô, không thấy dấu hiệu ngập.",
    ],
    "PUDDLE": [
        "Ảnh cho thấy {location}có một số vũng nước nhỏ, chưa đến mức ngập đáng kể.",
        "Có vũng nước trên mặt đường tại {location}, mực nước khoảng {depth_range}.",
        "Hệ thống ghi nhận vũng nước nhỏ tại {location}, khoảng {depth_range}.",
        "Nhìn thì {location}có vài vũng nước nhỏ, mực nước khoảng {depth_range}, chưa ngập đáng kể.",
        "{location}có vài vũng nước lẻ tẻ, mực nước khoảng {depth_range}.",
    ],
    "ANKLE": [
        "Ảnh cho thấy {location}có dấu hiệu ngập nhẹ, nước khoảng mắt cá chân — ước tính {depth_range}.",
        "Khu vực {location}ngập khoảng mắt cá, ước tính {depth_range}.",
        "Hệ thống ghi nhận mực nước tại {location}khoảng mắt cá chân, ước tính {depth_range}.",
        "{location}hơi ngập, nước khoảng mắt cá chân thôi — mực nước ước tính {depth_range}.",
        "Nhìn thì {location}ngập nhẹ, khoảng mắt cá chân (khoảng {depth_range}).",
    ],
    "KNEE": [
        "Ảnh cho thấy {location}có dấu hiệu ngập khá rõ, mực nước khoảng tới đầu gối — ước tính {depth_range}.",
        "Khu vực {location}ngập khoảng đầu gối, ước tính {depth_range}.",
        "Hệ thống ghi nhận {location}ngập đầu gối, ước tính {depth_range}.",
        "{location}ngập đến đầu gối, mực nước ước tính {depth_range}.",
        "{location}ngập khoảng đầu gối thôi, khoảng {depth_range}.",
    ],
    "WAIST": [
        "Ảnh cho thấy {location}ngập khá sâu, mực nước ước tính khoảng tới thắt lưng — {depth_range}.",
        "Khu vực {location}ngập sâu, ước tính {depth_range} (ngang hông).",
        "Hệ thống ghi nhận mực nước tại {location}đã đến mức thắt lưng, khoảng {depth_range}.",
        "{location}ngập sâu, nước đến hông, ước tính {depth_range}.",
        "{location}ngập sâu lắm, nước tới hông, khoảng {depth_range}.",
    ],
    "CHEST": [
        "Ảnh cho thấy {location}ngập rất sâu, ước tính {depth_range} — gần tới ngực.",
        "Khu vực {location}ngập nguy hiểm, mực nước ước tính {depth_range}.",
        "{location}ngập rất sâu, nước gần ngực, khoảng {depth_range}.",
        "Nhìn ảnh thì {location}ngập ngực, mực nước ước tính {depth_range}.",
    ],
    "SUBMERGED": [
        "Ảnh cho thấy {location}ngập hoàn toàn, mực nước ước tính {depth_range}.",
        "Khu vực {location}ngập toàn bộ, tình trạng nghiêm trọng.",
        "{location}ngập hoàn toàn, mực nước ước tính {depth_range}.",
        "{location}ngập kín, nước lên cao, khoảng {depth_range}.",
    ],
    "UNKNOWN": [
        "Mình chưa xác định được mức ngập chính xác tại {location}.",
        "Ảnh chưa đủ rõ để kết luận mức ngập tại {location}.",
        "Khó kết luận mức ngập tại {location} từ ảnh này.",
        "Mình chưa thể xác định mức ngập tại {location} từ ảnh này.",
    ],
}

# Câu về mức tin cậy
_CONFIDENCE_PHRASES: Dict[str, List[str]] = {
    "khá chắc chắn": [
        "Kết quả này khá chắc chắn.",
        "Độ tin cậy cao.",
        "Mình khá tự tin về kết quả này.",
        "Kết quả này tin cậy lắm.",
    ],
    "tương đối tin cậy": [
        "Kết quả tương đối tin cậy, nhưng vẫn nên kiểm tra thêm.",
        "Kết quả có thể tham khảo, nhưng nên xác minh trước khi đăng công khai.",
        "Thông tin này đáng tin, tuy nhiên vẫn nên được kiểm tra lại.",
        "Kết quả tin cậy ở mức trung bình, nên kiểm tra thêm cho chắc.",
    ],
    "chưa thật sự chắc chắn": [
        "Thông tin này chưa thật sự chắc chắn, nên xem lại trước khi đăng.",
        "Kết quả chưa ổn định, cần kiểm tra thêm.",
        "Mình chưa đủ chắc về kết quả này.",
        "Kết quả hơi bất ổn, nên xem lại kỹ hơn.",
    ],
    "chưa đủ cơ sở kết luận": [
        "Mình chưa đủ cơ sở để kết luận — cần thêm thông tin.",
        "Kết quả chưa đủ tin cậy để kết luận.",
        "Không đủ cơ sở để xác nhận, cần kiểm tra lại.",
        "Cơ sở chưa đủ để chắc chắn, cần thêm thông tin.",
    ],
}

# Câu khuyến cáo theo level
_RECOMMENDATIONS: Dict[str, str] = {
    "NO_FLOOD":  "",
    "PUDDLE":    "Người đi bộ chú ý trơn trượt.",
    "ANKLE":     "Xe máy qua được nhưng nên đi chậm.",
    "KNEE":      "Xe máy nên hạn chế đi qua, ô tô gầm thấp cũng cần thận trọng.",
    "WAIST":     "Người đi bộ và xe máy không nên đi qua khu vực này.",
    "CHEST":     "Tuyệt đối không đi qua — nguy hiểm cho cả người lẫn xe.",
    "SUBMERGED": "Không được đi qua khu vực này trong bất kỳ trường hợp nào.",
    "UNKNOWN":   "",
}

# Câu hỏi lại tự nhiên theo field thiếu
_FOLLOWUP_QUESTIONS: Dict[str, str] = {
    "location": (
        "Bạn cho mình biết ảnh này chụp ở đâu được không? "
        "Tên đường, khu vực hoặc mốc gần nhất là được."
    ),
    "timestamp": (
        "Ảnh này chụp khi nào vậy bạn? Vừa chụp hay đã lâu rồi?"
    ),
    "image_quality": (
        "Ảnh hơi mờ nên mình khó xác định mực nước. "
        "Bạn có thể gửi thêm ảnh rõ hơn không?"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# REWRITTEN RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RewrittenResponse:
    """
    Output của ResponseRewriter — dual format:
      internal_result: JSON kỹ thuật đầy đủ (dùng nội bộ/lưu DB)
      natural_response: Câu trả lời tự nhiên cho user/admin/news
    """
    internal_result:   Dict
    natural_response:  str
    mode:              str
    template_used:     str = ""    # debug info
    confidence_label:  str = ""


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE REWRITER
# ─────────────────────────────────────────────────────────────────────────────

class ResponseRewriter:
    """
    Chuyển AgentDecision + analysis result → natural language response.

    Không chứa logic phân tích — chỉ làm nhiệm vụ dịch thuật/viết lại.
    Tách biệt khỏi FloodAgent để dễ test và maintain độc lập.

    Dùng:
        from agent.response_rewriter import ResponseRewriter, ResponseMode

        rewritten = ResponseRewriter.rewrite(
            decision = my_decision,
            result   = raw_result_dict,
            mode     = ResponseMode.PUBLIC_USER,
        )
        print(rewritten.natural_response)
        print(rewritten.internal_result)
    """

    # Seed cho random variants — set để test deterministic
    _rng = random.Random()

    @classmethod
    def rewrite(
        cls,
        decision: Any,           # AgentDecision (import lazy để tránh circular)
        result:   Dict,
        mode:     ResponseMode | str = ResponseMode.PUBLIC_USER,
        location: str = "",
        missing_fields: Optional[List[str]] = None,
        n_reports: int = 1,
        simplify: bool = False,  # True → viết đơn giản hơn (fallback khi validate fail)
    ) -> RewrittenResponse:
        """
        Entry point chính.

        Args:
            decision:       AgentDecision object (hoặc dict)
            result:         Raw analysis result dict
            mode:           ResponseMode enum hoặc string
            location:       Tên địa điểm (tự nhiên hơn nếu có)
            missing_fields: List field đang thiếu ["location", "timestamp"]
            n_reports:      Số báo cáo cùng khu vực
            simplify:       Fallback mode — viết ngắn hơn, tránh lỗi validate
        """
        if isinstance(mode, str):
            try:
                mode = ResponseMode(mode)
            except ValueError:
                mode = ResponseMode.PUBLIC_USER

        # Extract từ decision (hỗ trợ cả object và dict)
        if hasattr(decision, "to_dict"):
            dec_dict = decision.to_dict()
        elif isinstance(decision, dict):
            dec_dict = decision
        else:
            dec_dict = {}

        # Extract từ result
        depth   = float(result.get("water_height_cm", 0) or 0)
        conf    = float(result.get("confidence",      0) or 0)
        level   = str(result.get("flood_level", "UNKNOWN")).upper()
        conf_lbl = confidence_label(conf)
        d_range = depth_range(depth) if depth > 0 else ""

        # Internal result (JSON đầy đủ — lưu DB / debug)
        internal = {
            "level":       level,
            "depth_cm":    round(depth, 1),
            "depth_range": d_range,
            "confidence":  round(conf, 3),
            "conf_label":  conf_lbl,
            "decision":    dec_dict.get("decision", ""),
            "alert_level": dec_dict.get("alert_level", ""),
            "location":    location,
            "n_reports":   n_reports,
        }

        # Route sang method theo mode
        if mode == ResponseMode.DEBUG:
            natural = cls._rewrite_debug(result, dec_dict, conf_lbl, d_range)
        elif mode == ResponseMode.ADMIN_REVIEW:
            natural = cls._rewrite_admin(result, dec_dict, conf_lbl, d_range,
                                          level, location, n_reports, missing_fields)
        elif mode == ResponseMode.NEWS_WRITER:
            natural = cls._rewrite_news(result, dec_dict, conf_lbl, d_range,
                                         level, location, n_reports)
        elif mode == ResponseMode.ALERT_MESSAGE:
            natural = cls._rewrite_alert(level, location, d_range, depth)
        else:  # PUBLIC_USER (default)
            natural = cls._rewrite_public(result, dec_dict, conf_lbl, d_range,
                                           level, location, missing_fields,
                                           simplify=simplify)

        return RewrittenResponse(
            internal_result  = internal,
            natural_response = natural,
            mode             = mode.value,
            confidence_label = conf_lbl,
        )

    # ── PUBLIC USER  (#1, #3, #7, #13) ───────────────────────────

    @classmethod
    def _rewrite_public(
        cls, result: Dict, dec: Dict, conf_lbl: str, d_range: str,
        level: str, location: str, missing_fields: Optional[List[str]],
        simplify: bool = False,
    ) -> str:
        """
        Viết lại cho người dùng thường — 4 phần:
          1. Xác nhận ngắn
          2. Kết luận dễ hiểu
          3. Mức chắc chắn
          4. Khuyến cáo
        """
        parts: List[str] = []

        # (1) Xác nhận ngắn
        parts.append(cls._rng.choice(_ACKNOWLEDGEMENTS))

        # (2) Kết luận — dùng template theo level
        loc_prefix = f"khu vực {location} " if location else ""
        templates = _LEVEL_TEMPLATES.get(level, _LEVEL_TEMPLATES["UNKNOWN"])
        chosen = cls._rng.choice(templates)
        conclusion = chosen.format(
            location    = loc_prefix,
            depth_range = d_range if d_range else "không xác định",
        )
        parts.append(conclusion)

        if simplify:
            # Simplified mode: chỉ 3 phần, bỏ confidence phrase phức tạp
            if conf_lbl not in ("khá chắc chắn",):
                parts.append("Thông tin này nên được kiểm tra thêm.")
        else:
            # (3) Mức chắc chắn — thêm từ nối tự nhiên
            conf_phrases = _CONFIDENCE_PHRASES.get(
                conf_lbl, _CONFIDENCE_PHRASES["chưa thật sự chắc chắn"]
            )
            # Thêm từ nối tự nhiên trước câu confidence
            connectors = [
                "Ngoài ra, ",
                "Thêm vào đó, ",
                "Bên cạnh đó, ",
                "Ngoài ra thì, ",
                ""
            ]
            connector = cls._rng.choice(connectors)
            conf_phrase = cls._rng.choice(_CONFIDENCE_PHRASES.get(
                conf_lbl, _CONFIDENCE_PHRASES["chưa thật sự chắc chắn"]
            ))
            parts.append(f"{connector}{conf_phrase}")

        # (4) Khuyến cáo — thêm từ nối
        rec = _RECOMMENDATIONS.get(level, "")
        if rec:
            rec_connectors = [
                "Vì vậy, ",
                "Do đó, ",
                "Nên nhớ rằng, ",
                "Lưu ý: ",
                ""
            ]
            rec_connector = cls._rng.choice(rec_connectors)
            parts.append(f"{rec_connector}{rec}")

        # Câu hỏi lại nếu thiếu thông tin
        if missing_fields:
            # Chỉ hỏi 1 field quan trọng nhất
            priority = ["location", "image_quality", "timestamp"]
            for field_name in priority:
                if field_name in missing_fields:
                    question = _FOLLOWUP_QUESTIONS.get(
                        field_name,
                        f"Bạn có thể cung cấp thêm thông tin về {field_name} không?"
                    )
                    # Thêm từ nối mềm dẻo
                    question_connectors = [
                        "Còn một chút, ",
                        "Một chút nữa, ",
                        "Thêm chút này, ",
                        ""
                    ]
                    parts.append(f"{cls._rng.choice(question_connectors)}{question}")
                    break

        return " ".join(p.strip() for p in parts if p.strip())

    # ── ADMIN REVIEW ──────────────────────────────────────────────

    @classmethod
    def _rewrite_admin(
        cls, result: Dict, dec: Dict, conf_lbl: str, d_range: str,
        level: str, location: str, n_reports: int,
        missing_fields: Optional[List[str]],
    ) -> str:
        """
        Viết lại cho admin — đề xuất + lý do + hành động.
        Được phép dùng một số thuật ngữ kỹ thuật nhẹ.
        """
        decision  = dec.get("decision", "needs_review")
        alert_lv  = dec.get("alert_level", "none")
        reasons   = dec.get("reasons", [])
        req_acts  = dec.get("required_actions", [])
        conf_pct  = round(float(result.get("confidence", 0) or 0) * 100)
        depth_val = float(result.get("water_height_cm", 0) or 0)

        decision_vi = {
            "publish":              "✅ Có thể đăng",
            "needs_review":         "🔍 Đề xuất đưa vào hàng chờ duyệt",
            "ask_for_more_info":    "💬 Cần thêm thông tin",
            "reject":               "🚫 Đề xuất từ chối",
            "update_existing_event":"🔄 Cập nhật sự kiện hiện có",
        }.get(decision, decision)

        lines = [f"**{decision_vi}**\n"]

        # Phần phân tích
        loc_part = f" tại {location}" if location else ""
        level_natural = _LEVEL_NATURAL.get(level, level)
        lines.append(
            f"Hệ thống ghi nhận{loc_part} {level_natural}, "
            f"ước tính {d_range}. "
            f"Độ tin cậy {conf_lbl} ({conf_pct}%)."
        )

        if n_reports > 1:
            lines.append(f"Có {n_reports} báo cáo từ khu vực này.")

        # Lý do
        if reasons:
            lines.append("\n**Lý do:**")
            for r in reasons:
                lines.append(f"• {r}")

        # Hành động cần làm
        if req_acts:
            lines.append("\n**Cần làm trước khi đăng:**")
            for a in req_acts:
                lines.append(f"□ {a}")
        elif decision == "publish":
            lines.append("\nKhông có vấn đề gì đặc biệt — có thể đăng sau khi admin xem qua.")

        # Missing info
        if missing_fields:
            lines.append("\n**Thông tin còn thiếu:** " + ", ".join(missing_fields))

        return "\n".join(lines)

    # ── NEWS WRITER ───────────────────────────────────────────────

    @classmethod
    def _rewrite_news(
        cls, result: Dict, dec: Dict, conf_lbl: str, d_range: str,
        level: str, location: str, n_reports: int,
    ) -> str:
        """
        Viết lại dạng tin tức ngắn — trung lập, dùng "ghi nhận", "ước tính".
        """
        depth_val = float(result.get("water_height_cm", 0) or 0)
        level_natural = _LEVEL_NATURAL.get(level, level).replace("ngập khoảng", "ngập")
        loc_part = f" tại {location}" if location else ""
        multi    = f" ({n_reports} báo cáo)" if n_reports > 1 else ""

        # Quyết định có thêm "đang chờ xác minh" không
        verified = dec.get("decision") == "publish"
        status   = "" if verified else " (đang chờ xác minh)"

        rec = _RECOMMENDATIONS.get(level, "")
        rec_sentence = f" {rec}" if rec else ""

        # Thêm hedging phrases tự nhiên
        hedging_intro = cls._rng.choice([
            "Theo hệ thống, ",
            "Theo kết quả phân tích, ",
            "Hệ thống ghi nhận, ",
            "Kết quả cho thấy, ",
        ])

        if depth_val > 0:
            text = (
                f"{hedging_intro}{loc_part} {level_natural}, "
                f"ước tính {d_range}{multi}{status}.{rec_sentence}"
            )
        else:
            text = f"{hedging_intro}ghi nhận tình trạng ngập{loc_part}{multi}{status}.{rec_sentence}"

        return text

    # ── ALERT MESSAGE ─────────────────────────────────────────────

    @classmethod
    def _rewrite_alert(cls, level: str, location: str,
                        d_range: str, depth: float) -> str:
        """
        Cảnh báo cực ngắn — tối đa 2 câu, rõ ràng, ưu tiên an toàn.
        """
        emoji    = _LEVEL_EMOJI.get(level, "⚠️")
        loc_part = f"khu vực {location} " if location else "khu vực này "
        level_natural = _LEVEL_NATURAL.get(level, "có dấu hiệu ngập")

        # Severity prefix
        if level in ("CHEST", "SUBMERGED"):
            prefix = f"{emoji} Cảnh báo nghiêm trọng:"
        elif level in ("WAIST", "KNEE"):
            prefix = f"{emoji} Cảnh báo:"
        else:
            prefix = f"{emoji} Lưu ý:"

        rec = _RECOMMENDATIONS.get(level, "")
        rec_short = rec.split(".")[0] + "." if rec else ""

        d_part = f" ({d_range})" if d_range else ""
        
        # Thêm từ nối mềm dẻo
        transition = cls._rng.choice([
            "Do đó, ",
            "Vì vậy, ",
            "Nên nhớ: ",
            "Cần lưu ý: ",
        ])
        
        return f"{prefix} {loc_part}{level_natural}{d_part}. {transition}{rec_short}".strip()

    # ── DEBUG ─────────────────────────────────────────────────────

    @classmethod
    def _rewrite_debug(cls, result: Dict, dec: Dict,
                        conf_lbl: str, d_range: str) -> str:
        """Debug mode — hiện toàn bộ kỹ thuật."""
        import json as _json
        lines = [
            "=== DEBUG: RAW ANALYSIS RESULT ===",
            _json.dumps(result, ensure_ascii=False, indent=2),
            "",
            "=== DEBUG: AGENT DECISION ===",
            _json.dumps(dec, ensure_ascii=False, indent=2),
            "",
            f"confidence_label: {conf_lbl}",
            f"depth_range:      {d_range}",
        ]
        return "\n".join(lines)

    # ── Convenience: rewrite from raw result only (without decision) ──

    @classmethod
    def rewrite_simple(
        cls,
        result:   Dict,
        mode:     ResponseMode | str = ResponseMode.PUBLIC_USER,
        location: str = "",
        missing_fields: Optional[List[str]] = None,
    ) -> RewrittenResponse:
        """
        Shortcut khi không có AgentDecision đầy đủ.
        Sinh decision giả từ kết quả.
        """
        conf  = float(result.get("confidence", 0) or 0)
        depth = float(result.get("water_height_cm", 0) or 0)
        fake_decision = {
            "decision":    "needs_review" if conf < 0.65 else "publish",
            "alert_level": "none" if depth < 15 else ("low" if depth < 60 else "medium"),
            "reasons":     [],
            "required_actions": [],
        }
        return cls.rewrite(
            decision        = fake_decision,
            result          = result,
            mode            = mode,
            location        = location,
            missing_fields  = missing_fields,
        )

    # ── render_schema  (#13) ──────────────────────────────────────

    @classmethod
    def render_schema(
        cls,
        schema: Any,   # ResponseSchema from conversation_state.py
        mode:   ResponseMode | str = ResponseMode.PUBLIC_USER,
        length: Any = None,        # ResponseLength (optional override)
    ) -> str:
        """
        Render ResponseSchema → natural language text.
        Tách biệt: ConversationState/Planner tạo schema,
                   ResponseRewriter render thành text.

        Ví dụ:
            schema = planner.build_schema(ResponseType.INFORM, result, state)
            text   = ResponseRewriter.render_schema(schema, mode="public_user")
        """
        if schema is None:
            return ""

        # Import lazy để tránh circular
        try:
            from agent.conversation_state import ResponseLength
        except ImportError:
            ResponseLength = None

        # Kiểm tra schema có phương thức render không
        if hasattr(schema, "render"):
            base = schema.render()
        else:
            # Fallback: schema là dict
            parts = [
                schema.get("main_message", ""),
                schema.get("safety_note",  ""),
                schema.get("action_hint",  "") if str(mode).endswith("admin_review") else "",
                schema.get("question",     ""),
            ]
            base = " ".join(p.strip() for p in parts if p.strip())

        # Validate sau khi render
        try:
            from agent.response_validator import ResponseValidator
            mode_str = mode.value if isinstance(mode, ResponseMode) else str(mode)
            valid = ResponseValidator.validate(base, mode=mode_str)
            if not valid and valid.severity == "fail":
                # Thay từ kỹ thuật đơn giản
                base = ResponseValidator.filter_tech_words(base)
        except ImportError:
            pass

        return base

    # ── explain_decision  (#16) ───────────────────────────────────

    @classmethod
    def explain_decision(
        cls,
        decision: Any,    # AgentDecision hoặc dict
        result:   Dict,
        mode:     ResponseMode | str = ResponseMode.PUBLIC_USER,
    ) -> str:
        """
        Giải thích quyết định của agent bằng ngôn ngữ tự nhiên (#16).

        Dùng khi:
          - Admin hỏi tại sao không đăng
          - User muốn hiểu kết quả phân tích

        Ví dụ output (admin_review):
            "Mình chưa khuyên đăng tin này ngay, vì ảnh chưa có vị trí rõ ràng
             và chất lượng hơi thấp. Có thể đưa vào hàng chờ xác minh hoặc
             yêu cầu người gửi bổ sung địa điểm."

        Ví dụ output (public_user):
            "Ảnh này có dấu hiệu ngập nhưng mình chưa chắc chắn lắm về kết quả.
             Bạn cho mình biết địa điểm chụp ảnh để mình có thể xác nhận rõ hơn."
        """
        if hasattr(decision, "to_dict"):
            dec = decision.to_dict()
        elif isinstance(decision, dict):
            dec = decision
        else:
            dec = {}

        dec_type   = dec.get("decision", "needs_review")
        reasons    = dec.get("reasons", [])
        req_acts   = dec.get("required_actions", [])
        conf       = float(result.get("confidence", 0) or 0)
        depth      = float(result.get("water_height_cm", 0) or 0)
        level      = str(result.get("flood_level", "UNKNOWN") or "UNKNOWN").upper()
        conf_lbl   = confidence_label(conf)
        d_range    = depth_range(depth) if depth > 0 else ""
        is_admin   = str(mode).endswith("admin_review") or mode == ResponseMode.ADMIN_REVIEW

        if dec_type == "publish" and not req_acts:
            if is_admin:
                return (
                    f"Báo cáo này có thể đăng. "
                    f"Hệ thống ghi nhận ngập {d_range}, kết quả {conf_lbl}. "
                    f"Không có vấn đề gì đặc biệt cần xử lý trước khi xuất bản."
                )
            else:
                return (
                    f"Ảnh cho thấy khu vực này {_LEVEL_NATURAL.get(level, 'có dấu hiệu ngập')}, "
                    f"ước tính {d_range}. Kết quả {conf_lbl}."
                )

        # needs_review hoặc có required_actions
        reason_str = reasons[0] if reasons else "Thông tin chưa đủ để kết luận."

        if is_admin:
            action_str = (
                f"Có thể {req_acts[0][0].lower() + req_acts[0][1:]} trước khi đăng."
                if req_acts else "Đưa vào hàng chờ xác minh."
            )
            return (
                f"Mình chưa khuyên đăng tin này ngay, vì {reason_str[0].lower() + reason_str[1:]}. "
                f"{action_str}"
            )
        else:
            followup = (
                "Bạn cho mình biết địa điểm chụp ảnh để mình xác nhận rõ hơn."
                if "vị trí" in reason_str.lower() or "gps" in reason_str.lower()
                else "Thông tin này nên được kiểm tra thêm trước khi chia sẻ rộng rãi."
            )
            return (
                f"Ảnh có dấu hiệu ngập nhưng kết quả {conf_lbl}. "
                f"{followup}"
            )
