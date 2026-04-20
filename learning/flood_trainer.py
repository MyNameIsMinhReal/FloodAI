# -*- coding: utf-8 -*-
"""
learning/flood_trainer.py
==========================
Fine-tune Qwen2.5 với LoRA trên training data flood.

Chạy trên GPU server:
  pip install transformers peft datasets accelerate bitsandbytes trl
  python -m learning.flood_trainer

Sau khi train xong, copy thư mục models/flood-agent-lora/ về máy.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from pathlib import Path

# ── Fix Windows encoding bug trong trl ───────────────────────────────────────
# trl đọc file .jinja bằng pathlib.Path.read_text() không chỉ định encoding,
# dẫn đến lỗi cp932 trên Windows. Monkey-patch để ép UTF-8 trước khi trl import.
_orig_read_text = pathlib.Path.read_text

def _utf8_read_text(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors)

pathlib.Path.read_text = _utf8_read_text  # type: ignore[method-assign]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("flood_trainer")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — chỉnh ở đây nếu cần
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
ROOT_DIR   = BASE_DIR.parent

# Model constants theo VRAM
MODEL_0_5B = "Qwen/Qwen2.5-0.5B-Instruct"   # < 6GB VRAM hoặc CPU
MODEL_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"   # 6–10GB VRAM  (RTX 5050 8GB → chọn đây)
MODEL_3B   = "Qwen/Qwen2.5-3B-Instruct"     # 10–20GB VRAM
MODEL_7B   = "Qwen/Qwen2.5-7B-Instruct"     # 20GB+ VRAM
BASE_MODEL = MODEL_0_5B                      # default, ghi đè khi detect GPU

TRAIN_FILE = BASE_DIR / "training_data" / "flood_conversations.jsonl"
EVAL_FILE  = BASE_DIR / "training_data" / "flood_conversations_eval.jsonl"
OUTPUT_DIR = ROOT_DIR / "models" / "flood-agent-lora"

TRAIN_CFG = {
    # ── 5 epoch: GPU nhàn rỗi → học kỹ hơn, ít overfitting với data ~900+
    "num_train_epochs":            5,
    # ── Batch 8 × grad_accum 2 = effective batch 16, GPU utilization cao hơn
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size":  8,
    "gradient_accumulation_steps": 2,
    # ── Learning rate: 1.5e-4 an toàn hơn cho 1.5B model, cosine decay dần
    "learning_rate":               1.5e-4,
    "warmup_ratio":                0.06,
    "lr_scheduler_type":           "cosine",
    # ── Log/eval thường xuyên để theo dõi loss
    "logging_steps":               5,
    "eval_steps":                  20,
    "save_steps":                  40,
    "save_total_limit":            3,        # chỉ giữ 3 checkpoint gần nhất
    # ── Sequence dài hơn → hiểu context tốt hơn (RTX 5050 đủ VRAM)
    "max_seq_length":              1024,
    # ── bf16 cho RTX 5050 (Blackwell native bf16, không cần GradScaler)
    "fp16":                        False,
    "bf16":                        True,
    # ── Gradient checkpointing: đổi tốc độ lấy VRAM → cho phép seq dài hơn
    "gradient_checkpointing":      True,
    # ── paged_adamw_8bit: optimizer tối ưu RAM/VRAM khi dùng bitsandbytes
    "optim":                       "paged_adamw_8bit",
    "load_best_model_at_end":      True,
    "report_to":                   "none",
}

LORA_CFG = {
    # r=32: rank cao → học được pattern phức tạp hơn (phù hợp 1.5B model)
    "r":              32,
    "lora_alpha":     64,   # = 2×r là chuẩn → scale đúng gradient
    "lora_dropout":   0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "bias":           "none",
    "task_type":      "CAUSAL_LM",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path):
    """Load JSONL file thành list of dicts."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_chat(example: dict, tokenizer) -> dict:
    """
    Chuyển messages format → tokenized input cho causal LM.
    Chỉ tính loss trên phần assistant response.
    """
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def _build_trainer(trl_new, sft_trainer_cls, training_args_cls, model, tokenizer,
                   train_dataset, eval_dataset, merged_cfg, eval_strategy_key, inspect):
    """
    Tạo SFTTrainer tương thích với mọi phiên bản TRL.
    Dùng inspect để chỉ truyền parameters mà class thực sự hỗ trợ.
    """
    max_seq = TRAIN_CFG["max_seq_length"]

    if trl_new:
        from trl import SFTConfig
        cfg_params = inspect.signature(SFTConfig.__init__).parameters

        seq_key = "max_seq_length" if "max_seq_length" in cfg_params else "max_length"
        extra_cfg = {seq_key: max_seq}
        if "dataset_text_field" in cfg_params:
            extra_cfg["dataset_text_field"] = "text"
        if "packing" in cfg_params:
            extra_cfg["packing"] = False

        sft_args = SFTConfig(
            output_dir=str(OUTPUT_DIR),
            **{eval_strategy_key: "steps"},
            **extra_cfg,
            **merged_cfg,
        )

        trainer_params = inspect.signature(sft_trainer_cls.__init__).parameters
        tok_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"

        trainer_kwargs = {
            "model":         model,
            tok_key:         tokenizer,
            "train_dataset": train_dataset,
            "eval_dataset":  eval_dataset,
            "args":          sft_args,
        }
        if "dataset_text_field" in trainer_params and "dataset_text_field" not in extra_cfg:
            trainer_kwargs["dataset_text_field"] = "text"

        return sft_trainer_cls(**trainer_kwargs)

    # TRL < 0.12: API cũ
    training_args = training_args_cls(
        output_dir=str(OUTPUT_DIR),
        **{eval_strategy_key: "steps"},
        **merged_cfg,
    )
    return sft_trainer_cls(
        model=model, tokenizer=tokenizer,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        dataset_text_field="text", max_seq_length=max_seq,
        args=training_args, packing=False,
    )


def _detect_device(torch) -> tuple[bool, str]:
    """Phát hiện GPU/CPU và chọn model phù hợp. Trả về (use_gpu, model_name)."""
    if not torch.cuda.is_available():
        log.warning("=" * 60)
        log.warning("Không có GPU — chạy trên CPU (chậm, mất 1-4 giờ).")
        log.warning("Model cố định: Qwen2.5-0.5B (nhỏ nhất)")
        log.warning("=" * 60)
        return False, MODEL_0_5B

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    if vram_gb < 6:
        model_name = MODEL_0_5B   # <6GB
    elif vram_gb < 10:
        model_name = MODEL_1_5B   # 6–10GB: RTX 5050 (8GB) → 1.5B + QLoRA 4-bit
    elif vram_gb < 20:
        model_name = MODEL_3B     # 10–20GB
    else:
        model_name = MODEL_7B     # 20GB+
    log.info(f"Tự động chọn model: {model_name}")
    return True, model_name


def _load_model(torch, model_cls, model_name: str, use_gpu: bool):
    """Load model với QLoRA (GPU) hoặc float32 (CPU)."""
    log.info(f"Loading model: {model_name}")
    if not use_gpu:
        return model_cls.from_pretrained(
            model_name, device_map="cpu",
            torch_dtype=torch.float32, trust_remote_code=True,
        )
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        return model_cls.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
    except Exception:
        return model_cls.from_pretrained(
            model_name, device_map="auto",
            torch_dtype=torch.float16, trust_remote_code=True,
        )


def train():
    try:
        import torch
    except ImportError:
        log.error("Thiếu torch. Chạy: pip install torch")
        return

    use_gpu, model_name = _detect_device(torch)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
        from datasets import Dataset
    except (ImportError, RuntimeError) as e:
        log.error(
            f"Thiếu hoặc lỗi thư viện: {e}\n"
            "Chạy: pip install transformers peft datasets accelerate trl\n"
            "Nếu lỗi encoding: set PYTHONUTF8=1 && python -m learning.flood_trainer"
        )
        return

    if not TRAIN_FILE.exists():
        log.error(f"Không tìm thấy training data: {TRAIN_FILE}")
        log.error("Chạy trước: python -m learning.flood_data_generator")
        return

    train_raw = load_jsonl(TRAIN_FILE)
    eval_raw  = load_jsonl(EVAL_FILE) if EVAL_FILE.exists() else train_raw[:20]
    log.info(f"Train: {len(train_raw)} | Eval: {len(eval_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _apply_template(ex):
        return tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )

    train_dataset = Dataset.from_dict({"text": [_apply_template(e) for e in train_raw]})
    eval_dataset  = Dataset.from_dict({"text": [_apply_template(e) for e in eval_raw]})

    model = _load_model(torch, AutoModelForCausalLM, model_name, use_gpu)  # noqa: F821
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**LORA_CFG))
    if use_gpu and TRAIN_CFG.get("gradient_checkpointing"):
        model.enable_input_require_grads()   # bắt buộc khi dùng gradient_checkpointing + PEFT
    model.print_trainable_parameters()

    import trl as _trl
    import transformers as _tf
    _trl_ver  = tuple(int(x) for x in _trl.__version__.split(".")[:2])
    _tf_ver   = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    _trl_new  = _trl_ver >= (0, 12)   # SFTConfig thay TrainingArguments
    _tf_new   = _tf_ver  >= (4, 46)   # eval_strategy thay evaluation_strategy

    # warmup_ratio deprecated → tính warmup_steps thủ công
    n_steps_per_epoch = max(1, len(train_dataset) //
                            TRAIN_CFG["per_device_train_batch_size"])
    total_steps = n_steps_per_epoch * TRAIN_CFG["num_train_epochs"]
    warmup_steps = max(1, int(total_steps * TRAIN_CFG["warmup_ratio"]))

    cpu_overrides = {} if use_gpu else {
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size":  1,
        "gradient_accumulation_steps": 8,
        "fp16":                        False,
        "bf16":                        False,
        "gradient_checkpointing":      False,  # không cần trên CPU
        "optim":                       "adamw_torch",  # paged_adamw_8bit cần bitsandbytes GPU
        "use_cpu":                     True,
        "dataloader_num_workers":      0,
        "eval_steps":                  200,
        "save_steps":                  200,
        "logging_steps":               20,
    }

    # Build config dict — loại bỏ các key không còn dùng
    exclude = {"max_seq_length", "warmup_ratio"}
    merged_cfg = {k: v for k, v in TRAIN_CFG.items() if k not in exclude}
    merged_cfg["warmup_steps"] = warmup_steps
    merged_cfg.update(cpu_overrides)
    eval_strategy_key = "eval_strategy" if _tf_new else "evaluation_strategy"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dùng inspect để detect chính xác parameters được hỗ trợ
    # → không bị break khi TRL thay đổi API
    import inspect
    trainer = _build_trainer(
        trl_new=_trl_new,
        sft_trainer_cls=SFTTrainer,
        training_args_cls=TrainingArguments,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        merged_cfg=merged_cfg,
        eval_strategy_key=eval_strategy_key,
        inspect=inspect,
    )

    if not use_gpu:
        log.info("CPU mode — có thể mất 1-4 giờ. Giảm num_train_epochs=1 để nhanh hơn.")
    log.info("=== Bắt đầu training ===")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────
    log.info(f"Lưu LoRA adapter → {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    log.info("=== Training hoàn thành! ===")
    log.info(f"Adapter saved: {OUTPUT_DIR}")
    log.info("")
    log.info("Bước tiếp theo — merge adapter + export GGUF:")
    log.info("  python -m learning.flood_trainer --merge")
    log.info("")
    log.info("Hoặc copy thư mục models/flood-agent-lora/ về máy CPU")
    log.info("rồi dùng flood_llm.py với backend='transformers'")


# ─────────────────────────────────────────────────────────────────────────────
# MERGE  — gộp LoRA adapter vào base model để export GGUF
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_save():
    """
    Merge LoRA adapter với base model → lưu full model.
    Sau đó có thể convert sang GGUF bằng llama.cpp.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError as e:
        log.error(f"Thiếu thư viện: {e}")
        return

    merged_dir = ROOT_DIR / "models" / "flood-agent-merged"

    # Đọc base_model_name_or_path từ adapter_config.json để tránh mismatch
    import json
    adapter_cfg = OUTPUT_DIR / "adapter_config.json"
    if adapter_cfg.exists():
        with open(adapter_cfg) as f:
            _cfg = json.load(f)
        actual_base_model = _cfg.get("base_model_name_or_path", BASE_MODEL)
    else:
        actual_base_model = BASE_MODEL

    log.info(f"Loading base model: {actual_base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        actual_base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR), trust_remote_code=True)

    log.info(f"Loading LoRA adapter: {OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR))

    log.info("Merging LoRA into base model...")
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving merged model → {merged_dir}")
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    log.info("=== Merge xong! ===")
    log.info("")
    log.info("Convert sang GGUF (chạy trên GPU server):")
    log.info("  git clone https://github.com/ggerganov/llama.cpp")
    log.info("  pip install -r llama.cpp/requirements.txt")
    log.info(f"  python llama.cpp/convert_hf_to_gguf.py {merged_dir} --outfile models/flood-agent-q4.gguf --outtype q4_k_m")
    log.info("")
    log.info("Copy file models/flood-agent-q4.gguf về máy CPU")
    log.info("rồi dùng flood_llm.py với backend='llama_cpp'")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--merge" in sys.argv:
        merge_and_save()
    else:
        train()
