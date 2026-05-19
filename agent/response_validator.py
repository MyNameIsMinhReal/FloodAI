# -*- coding: utf-8 -*-
"""
agent/response_validator.py
============================
Kiểm tra câu trả lời của agent trước khi gửi đến user.

Hai loại kiểm tra:
  1. TechnicalLeakCheck  — phát hiện từ kỹ thuật rò rỉ sang public response
  2. ToneCheck           — phát hiện ngôn ngữ quá chắc chắn, giật tít

Dùng sau ResponseRewriter:

    rewritten = rewriter.rewrite(decision, mode="public_user")
    ok, issues = ResponseValidator.validate(rewritten.natural_response, mode="public_user")
    if not ok:
        rewritten = rewriter.rewrite(decision, mode="public_user", simplify=True)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# BAD WORD LISTS
# ─────────────────────────────────────────────────────────────────────────────

# Thuật ngữ kỹ thuật tuyệt đối không được xuất hiện trong public response
_TECH_FORBIDDEN: List[str] = [
    "confidence",
    "pipeline",
    "inference",
    "bbox",
    "bounding box",
    "segmentation",
    "json",
    "model output",
    "detection output",
    "flood_level",
    "water_height",
    "depth_cm",
    "flood_prob",
    "stage",
    "threshold",
    "yolo",
    "dino",
    "resnet",
    "huggingface",
    "tensor",
    "softmax",
]

# Ngôn ngữ giật tít / phóng đại (tất cả modes)
_SENSATIONAL: List[str] = [
    "kinh hoàng",
    "thảm họa",
    "chấn động",
    "nguy hiểm chết người",
    "chìm trong biển nước",
    "hãi hùng",
    "hoảng loạn",
    "tang thương",
    "nhấn chìm",
    "sốc:",
    "cực kỳ nguy hiểm",
]

# Câu nói quá chắc chắn (không phù hợp với ước tính AI)
_OVERCONFIDENT_PATTERNS: List[str] = [
    r"đang ngập \d+cm",          # "đang ngập 52cm" → nên dùng "ước tính"
    r"chính xác là \d+",
    r"ngập đúng \d+",
    r"xác nhận ngập",
    r"chắc chắn ngập",
]

# Câu nghe quá "AI / robot"
_ROBOTIC_PHRASES: List[str] = [
    "dựa trên dữ liệu đầu vào được cung cấp",
    "hệ thống đã thực hiện phân tích",
    "tôi đã xử lý đầu vào",
    "mô hình cho thấy confidence",
    "kết quả inference",
    "đối tượng water được detect",
    "pipeline trả về",
    "tôi không thể xác nhận",
    "tôi không có khả năng",
    "as an ai",
    "as an assistant",
    "i cannot",
    "i am unable to",
]


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    is_valid:  bool
    issues:    List[str] = field(default_factory=list)
    severity:  str = "ok"   # "ok" | "warn" | "fail"

    def __bool__(self) -> bool:
        return self.is_valid


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class ResponseValidator:
    """
    Kiểm tra câu trả lời theo mode.

    Mode "public_user" → kiểm tra kỹ nhất (không để lọt từ kỹ thuật).
    Mode "admin_review" → cho phép một số thuật ngữ nhẹ.
    Mode "news_writer"  → kiểm tra giật tít.
    Mode "debug"        → không kiểm tra gì.
    """

    @classmethod
    def validate(cls, text: str, mode: str = "public_user") -> ValidationResult:
        """
        Kiểm tra câu trả lời.

        Returns:
            ValidationResult(is_valid, issues, severity)
        """
        if mode == "debug":
            return ValidationResult(is_valid=True, severity="ok")

        issues: List[str] = []
        t_lower = text.lower()

        # 1. Kiểm tra từ kỹ thuật (chỉ public_user và news_writer)
        if mode in ("public_user", "news_writer", "alert_message"):
            for word in _TECH_FORBIDDEN:
                if word.lower() in t_lower:
                    issues.append(f"Từ kỹ thuật rò rỉ: '{word}'")

        # 2. Kiểm tra giật tít (tất cả modes trừ debug)
        for phrase in _SENSATIONAL:
            if phrase.lower() in t_lower:
                issues.append(f"Ngôn ngữ phóng đại: '{phrase}'")

        # 3. Kiểm tra quá chắc chắn (public + news)
        if mode in ("public_user", "news_writer", "alert_message"):
            for pattern in _OVERCONFIDENT_PATTERNS:
                if re.search(pattern, t_lower):
                    issues.append(f"Diễn đạt quá chắc chắn: pattern='{pattern}'")

        # 4. Kiểm tra ngôn ngữ robot (tất cả modes trừ debug)
        for phrase in _ROBOTIC_PHRASES:
            if phrase.lower() in t_lower:
                issues.append(f"Ngôn ngữ robot: '{phrase}'")

        # 5. Kiểm tra độ dài (alert_message không nên quá dài)
        if mode == "alert_message" and len(text) > 200:
            issues.append(f"Alert message quá dài ({len(text)} ký tự > 200)")

        # 6. Kiểm tra câu hỏi lại — public_user không nên có quá 1 câu hỏi
        if mode == "public_user":
            n_questions = text.count("?")
            if n_questions > 2:
                issues.append(f"Có {n_questions} câu hỏi trong 1 response — nên ≤ 2")

        severity = "ok" if not issues else ("fail" if len(issues) >= 2 else "warn")
        return ValidationResult(
            is_valid  = len(issues) == 0,
            issues    = issues,
            severity  = severity,
        )

    @classmethod
    def filter_tech_words(cls, text: str) -> str:
        """
        Thay thế từ kỹ thuật phổ biến bằng từ thân thiện.
        Dùng như bước pre-filter nhẹ trước khi validate.
        """
        replacements = {
            r'\bconfidence\b':      "độ tin cậy",
            r'\binference\b':       "phân tích",
            r'\bpipeline\b':        "hệ thống",
            r'\bdetection\b':       "nhận diện",
            r'\bmodel\b':           "hệ thống",
            r'\bthreshold\b':       "ngưỡng",
            r'\bbbox\b':            "vùng ảnh",
            r'\bneeds_review\b':    "cần xem lại",
            r'\bno_flood\b':        "không ngập",
            r'\blevel\s*=\s*\w+':   "",   # xóa "level=knee" dạng raw
        }
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# QUICK CHECK FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def is_public_safe(text: str) -> bool:
    """Kiểm tra nhanh text có an toàn để hiện cho người dùng thường không."""
    result = ResponseValidator.validate(text, mode="public_user")
    return result.is_valid


def has_sensational_language(text: str) -> bool:
    """Kiểm tra text có ngôn ngữ giật tít không."""
    t_lower = text.lower()
    return any(p in t_lower for p in _SENSATIONAL)


def extract_issues(text: str, mode: str = "public_user") -> List[str]:
    """Trả về danh sách vấn đề trong text."""
    return ResponseValidator.validate(text, mode).issues
