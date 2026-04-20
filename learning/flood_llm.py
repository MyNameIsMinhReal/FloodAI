# -*- coding: utf-8 -*-
"""
learning/flood_llm.py
======================
Inference wrapper cho FloodAgent LLM — chạy trên CPU hoặc GPU.

Hỗ trợ 2 backend:
  llama_cpp   — dùng file .gguf (tốt nhất cho CPU, nhẹ, nhanh)
  transformers — dùng thư mục model HuggingFace (CPU/GPU)

Tích hợp vào FloodAgent:
  enhancer = LLMEnhancer.from_gguf("models/flood-agent-q4.gguf")
  # hoặc base model chưa fine-tune (để test trước):
  enhancer = LLMEnhancer.base_model_cpu()

  # Thay FeedbackParser:
  parsed = enhancer.parse_intent(user_text, context=last_result)

  # Thay _context_aware_message:
  msg = enhancer.generate_response(results, action_type, memory)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("flood_llm")

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (giống với flood_data_generator.py)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "Bạn là FloodAgent — AI phân tích lũ lụt thông minh, hỗ trợ tiếng Việt và tiếng Anh.\n"
    "Chức năng chính: phân tích ảnh lũ, đọc kết quả và phản hồi người dùng.\n\n"
    "Khi người dùng gửi tin nhắn, hãy trả lời theo đúng định dạng sau:\n"
    "INTENT: <intent>\n"
    "DEPTH_HINT: <số cm hoặc null>\n"
    "LEVEL_HINT: <ANKLE/KNEE/WAIST/CHEST/SUBMERGED/NO_FLOOD hoặc null>\n\n"
    "<phản hồi tự nhiên bằng tiếng Việt>\n\n"
    "Các intent hợp lệ: confirm | increase | decrease | no_flood | "
    "rerun | simulate | status | help | calibrate | greeting | unknown\n"
    "Dùng 'greeting' khi người dùng chào hỏi (hello, hi, xin chào, chào,…).\n\n"
    "QUAN TRỌNG:\n"
    "- Nếu người dùng hỏi câu hỏi thông thường KHÔNG liên quan đến lũ lụt "
    "(toán học, kiến thức chung, hỏi thăm,…), hãy trả lời ĐÚNG và tự nhiên "
    "như một trợ lý thông thường — KHÔNG gắn kết quả vào ngữ cảnh lũ.\n"
    "- Chỉ nói về lũ lụt khi người dùng hỏi về lũ hoặc đang trong ngữ cảnh phân tích ảnh.\n"
    "- KHÔNG tự ý thêm 'sẽ phân tích ảnh lũ' vào câu trả lời không liên quan."
)

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

VALID_INTENTS = {
    "confirm", "increase", "decrease", "no_flood",
    "rerun", "simulate", "status", "help", "calibrate", "greeting", "unknown",
}


# ─────────────────────────────────────────────────────────────────────────────
# PARSED RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMParsedFeedback:
    intent:      str
    depth_hint:  Optional[float]
    level_hint:  Optional[str]
    response:    str
    raw_output:  str = ""

    # Tương thích với ParsedFeedback cũ
    @property
    def magnitude(self) -> str:
        return "medium"

    @property
    def raw(self) -> str:
        return self.raw_output

    @property
    def image_id(self) -> Optional[int]:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_output(text: str) -> LLMParsedFeedback:
    """
    Parse output của LLM theo định dạng:
      INTENT: decrease
      DEPTH_HINT: 30
      LEVEL_HINT: null

      <response text>
    """
    intent     = "unknown"
    depth_hint: Optional[float] = None
    level_hint: Optional[str]   = None
    response   = ""

    lines      = text.strip().split("\n")
    body_start = 0

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith("INTENT:"):
            val = line_stripped.split(":", 1)[1].strip().lower()
            if val in VALID_INTENTS:
                intent = val
            body_start = i + 1
        elif line_stripped.startswith("DEPTH_HINT:"):
            val = line_stripped.split(":", 1)[1].strip()
            if val.lower() not in ("null", "none", ""):
                try:
                    depth_hint = float(val)
                except ValueError:
                    pass
            body_start = i + 1
        elif line_stripped.startswith("LEVEL_HINT:"):
            val = line_stripped.split(":", 1)[1].strip().upper()
            if val not in ("NULL", "NONE", ""):
                level_hint = val
            body_start = i + 1

    # Response = phần còn lại sau các header lines
    response_lines = lines[body_start:]
    # Bỏ dòng trống đầu
    while response_lines and not response_lines[0].strip():
        response_lines = response_lines[1:]
    response = "\n".join(response_lines).strip()

    if not response:
        response = text.strip()

    return LLMParsedFeedback(
        intent=intent,
        depth_hint=depth_hint,
        level_hint=level_hint,
        response=response,
        raw_output=text,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM ENHANCER
# ─────────────────────────────────────────────────────────────────────────────

class LLMEnhancer:
    """
    Wrapper inference cho FloodAgent LLM.

    Dùng thay cho:
      - FeedbackParser.parse()        → self.parse_intent()
      - _context_aware_message()      → self.generate_response()
    """

    def __init__(
        self,
        backend: str = "llama_cpp",
        model_path: Optional[str] = None,
        n_ctx: int = 1024,
        n_threads: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ):
        """
        backend:
          "llama_cpp"    — dùng GGUF file, tốt nhất cho CPU
          "transformers" — dùng HuggingFace model dir, cần nhiều RAM hơn

        model_path:
          llama_cpp:    đường dẫn tới file .gguf
          transformers: đường dẫn tới thư mục model
        """
        self.backend        = backend
        self.model_path     = model_path
        self.n_ctx          = n_ctx
        self.n_threads      = n_threads
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self._model: Any    = None
        self._tokenizer: Any = None

        self._load()

    # ── Load ──────────────────────────────────────────────────────────

    def _load(self):
        if self.backend == "llama_cpp":
            self._load_llama_cpp()
        elif self.backend == "transformers":
            self._load_transformers()
        else:
            raise ValueError(f"backend không hợp lệ: {self.backend}")

    def _load_llama_cpp(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "Cần cài llama-cpp-python:\n"
                "  pip install llama-cpp-python"
            )
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Không tìm thấy GGUF model: {self.model_path}\n"
                "Xem hướng dẫn download ở README hoặc chạy flood_trainer.py trước."
            )
        log.info(f"[LLM] Loading GGUF: {self.model_path}")
        self._model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
            chat_format="chatml",
        )
        log.info("[LLM] GGUF model loaded (CPU mode)")

    def _load_transformers(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "Cần cài transformers:\n"
                "  pip install transformers accelerate"
            )
        if not self.model_path:
            raise ValueError("Cần chỉ định model_path cho backend transformers")

        log.info(f"[LLM] Loading transformers model: {self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,  # CPU dùng float32
            device_map="cpu",
            trust_remote_code=True,
        )
        self._model.eval()
        log.info("[LLM] Transformers model loaded (CPU mode)")

    # ── Inference ─────────────────────────────────────────────────────

    def _infer(self, messages: List[Dict]) -> str:
        if self.backend == "llama_cpp":
            return self._infer_llama_cpp(messages)
        return self._infer_transformers(messages)

    def _infer_llama_cpp(self, messages: List[Dict]) -> str:
        result = self._model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return result["choices"][0]["message"]["content"]

    def _infer_transformers(self, messages: List[Dict]) -> str:
        import torch
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── Public API ────────────────────────────────────────────────────

    def parse_intent(
        self,
        user_text: str,
        context: Optional[Dict] = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> LLMParsedFeedback:
        """
        Hiểu ý người dùng → trả về LLMParsedFeedback.
        Thay thế FeedbackParser.parse().

        context: last_result dict từ AgentMemory (optional)
        chat_history: lịch sử hội thoại gần nhất để LLM hiểu ngữ cảnh
        """
        user_content = self._build_user_content(user_text, context)
        messages: List[Dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for entry in (chat_history or [])[-6:]:
            role = "assistant" if entry.get("role") == "agent" else "user"
            messages.append({"role": role, "content": entry.get("content", "")})
        messages.append({"role": "user", "content": user_content})
        try:
            raw_output = self._infer(messages)
            return _parse_llm_output(raw_output)
        except Exception as e:
            log.warning(f"[LLM] parse_intent lỗi: {e}")
            return LLMParsedFeedback(
                intent="unknown", depth_hint=None, level_hint=None,
                response="Xin lỗi, mình chưa hiểu ý bạn. Có thể nói rõ hơn không?",
                raw_output="",
            )

    def generate_response(
        self,
        results: List[Dict],
        action_type: str,
        memory_summary: Optional[Dict] = None,
        duration_s: float = 0,
    ) -> str:
        """
        Tạo response tự nhiên từ kết quả phân tích.
        Thay thế _context_aware_message().
        """
        if not results:
            return "ℹ️ Không có kết quả phân tích."

        result_text = self._format_results_for_prompt(results, duration_s)
        context_text = ""
        if memory_summary:
            n_corr = memory_summary.get("n_corrections", 0)
            trend  = memory_summary.get("correction_trend", "none")
            if n_corr > 0:
                context_text = (
                    f"\n[Lịch sử: {n_corr} lần sửa, xu hướng: {trend}]"
                )

        prompt = (
            f"Bạn vừa phân tích lũ xong (action: {action_type}).\n"
            f"{result_text}{context_text}\n\n"
            "Hãy viết phản hồi tự nhiên bằng tiếng Việt cho người dùng. "
            "Không cần dùng định dạng INTENT/DEPTH_HINT, chỉ viết response thôi."
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = self._infer(messages)
            # Nếu model vẫn xuất header → chỉ lấy phần response
            parsed = _parse_llm_output(raw)
            return parsed.response if parsed.response else raw
        except Exception as e:
            log.warning(f"[LLM] generate_response lỗi: {e}")
            return self._fallback_response(results, duration_s)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_user_content(user_text: str, context: Optional[Dict]) -> str:
        if context:
            level_code = context.get("flood_level", "UNKNOWN")
            level_name = _LEVEL_VI.get(level_code, level_code)
            depth      = context.get("water_height_cm", 0)
            conf       = round(float(context.get("confidence", 0) or 0) * 100)
            ctx = (
                f"[Kết quả phân tích hiện tại]\n"
                f"- Mức lũ   : {level_name}\n"
                f"- Độ sâu   : {depth}cm\n"
                f"- Độ tin cậy: {conf}%\n\n"
            )
        else:
            ctx = ""
        return f"{ctx}Người dùng nói: \"{user_text}\""

    @staticmethod
    def _format_results_for_prompt(results: List[Dict], duration_s: float) -> str:
        lines = [f"Kết quả phân tích ({len(results)} ảnh, {duration_s:.1f}s):"]
        for i, r in enumerate(results[:3], 1):
            lvl  = r.get("flood_level", "UNKNOWN")
            name = _LEVEL_VI.get(lvl, lvl)
            depth = r.get("water_height_cm", 0)
            conf  = round(float(r.get("confidence", 0) or 0) * 100)
            lines.append(f"  Ảnh {i}: {name}, {depth}cm, confidence {conf}%")
        if len(results) > 3:
            lines.append(f"  ... và {len(results) - 3} ảnh khác")
        return "\n".join(lines)

    @staticmethod
    def _fallback_response(results: List[Dict], duration_s: float) -> str:
        """Response dự phòng nếu LLM lỗi."""
        if not results:
            return "ℹ️ Không có kết quả."
        r    = results[0]
        lvl  = r.get("flood_level", "UNKNOWN")
        name = _LEVEL_VI.get(lvl, lvl)
        depth = r.get("water_height_cm", 0)
        conf  = round(float(r.get("confidence", 0) or 0) * 100)
        return (
            f"✅ Phân tích xong {len(results)} ảnh ({duration_s:.1f}s)\n"
            f"Mức lũ: {name}\n"
            f"Độ sâu: {depth}cm  |  Độ tin cậy: {conf}%"
        )

    # ── Factory methods ───────────────────────────────────────────────

    @classmethod
    def from_gguf(cls, gguf_path: str, **kwargs) -> "LLMEnhancer":
        """Load từ file GGUF (sau khi fine-tune + convert)."""
        return cls(backend="llama_cpp", model_path=gguf_path, **kwargs)

    @classmethod
    def base_model_cpu(
        cls,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        **kwargs,
    ) -> "LLMEnhancer":
        """
        Load base model từ HuggingFace (chưa fine-tune).
        Dùng để test inference trên CPU trước khi có model đã train.
        """
        return cls(backend="transformers", model_path=model_name, **kwargs)

    @classmethod
    def from_lora_dir(cls, lora_dir: str, **kwargs) -> "LLMEnhancer":
        """Load model đã fine-tune (thư mục LoRA adapter đã merge)."""
        return cls(backend="transformers", model_path=lora_dir, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST  — chạy để kiểm tra inference
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== FloodAgent LLM — Quick Test ===\n")

    # Xác định model path từ args
    if len(sys.argv) > 1:
        model_arg = sys.argv[1]
        if model_arg.endswith(".gguf"):
            print(f"Loading GGUF: {model_arg}")
            enhancer = LLMEnhancer.from_gguf(model_arg)
        else:
            print(f"Loading transformers model: {model_arg}")
            enhancer = LLMEnhancer.from_lora_dir(model_arg)
    else:
        print("Dùng base model (Qwen2.5-0.5B, chưa fine-tune)...")
        print("Lưu ý: cần kết nối internet để download lần đầu (~1GB)\n")
        enhancer = LLMEnhancer.base_model_cpu()

    # Test cases
    test_cases = [
        {
            "text": "Đúng rồi, chính xác!",
            "context": {"flood_level": "KNEE", "water_height_cm": 55, "confidence": 0.75},
        },
        {
            "text": "Nước thấp hơn nhiều, khoảng 30cm thôi.",
            "context": {"flood_level": "KNEE", "water_height_cm": 55, "confidence": 0.60},
        },
        {
            "text": "Không có lũ ở đây bạn ơi.",
            "context": {"flood_level": "ANKLE", "water_height_cm": 25, "confidence": 0.45},
        },
        {
            "text": "Chạy lại đi.",
            "context": {"flood_level": "WAIST", "water_height_cm": 90, "confidence": 0.55},
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"--- Test {i} ---")
        print(f"User: {tc['text']}")
        parsed = enhancer.parse_intent(tc["text"], tc.get("context"))
        print(f"Intent    : {parsed.intent}")
        print(f"Depth hint: {parsed.depth_hint}")
        print(f"Level hint: {parsed.level_hint}")
        print(f"Response  : {parsed.response}")
        print()
