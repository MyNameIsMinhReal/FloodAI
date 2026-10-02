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
MODEL_7B   = "Qwen/Qwen2.5-7B-Instruct"     # 20–35GB VRAM
MODEL_14B  = "Qwen/Qwen2.5-14B-Instruct"    # 35–55GB VRAM  (bf16 full, không cần QLoRA)
MODEL_32B  = "Qwen/Qwen2.5-32B-Instruct"    # 55GB+ VRAM    (QLoRA 4-bit ≈ 20GB, còn 60GB cho batch lớn)
MODEL_72B  = "Qwen/Qwen2.5-72B-Instruct"    # 80GB+ VRAM    (QLoRA 4-bit ≈ 40GB, cần A100/H100)
BASE_MODEL = MODEL_0_5B                      # default, ghi đè khi detect GPU

TRAIN_FILE = BASE_DIR / "training_data" / "flood_conversations.jsonl"
EVAL_FILE  = BASE_DIR / "training_data" / "flood_conversations_eval.jsonl"
OUTPUT_DIR = ROOT_DIR / "models" / "flood-agent-lora"

TRAIN_CFG = {
    "num_train_epochs":            5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size":  16,
    "gradient_accumulation_steps": 2,
    "learning_rate":               1.5e-4,
    "warmup_ratio":                0.06,
    "lr_scheduler_type":           "cosine",
    "logging_steps":               5,
    "eval_steps":                  20,
    "save_steps":                  20,
    "save_total_limit":            3,
    "max_seq_length":              2048,
    "fp16":                        False,
    "bf16":                        True,
    "gradient_checkpointing":      False,
    "optim":                       "adamw_torch",
    "load_best_model_at_end":      True,   # giữ checkpoint tốt nhất theo eval_loss
    "metric_for_best_model":       "eval_loss",
    "greater_is_better":           False,
    "eval_delay":                  0,
    "report_to":                   "none",
}

# Flag toàn cục: True → dùng bf16 thuần (không QLoRA), False → dùng QLoRA 4-bit
# Sẽ được set lại trong _detect_device() dựa trên VRAM thực tế
_USE_BF16_FULL = False

LORA_CFG = {
    # r=32: rank mặc định; _detect_device() sẽ nâng lên 64/128 nếu đủ VRAM
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


def _split_prompt_completion(messages: list, tokenizer) -> dict:
    """Chia messages → (prompt, completion) chuỗi cho TRL prompt-completion dataset.

    - prompt     = apply_chat_template(messages[:-1], add_generation_prompt=True)
                   (kết thúc bằng header assistant '<|im_start|>assistant\n')
    - completion = phần còn lại của full conversation (assistant content + <|im_end|>),
                   lấy = full[len(prompt):] để ĐẢM BẢO prefix → completion_mask chính xác.

    KHÔNG dùng apply_chat_template([assistant]) vì sẽ render thêm header trùng.
    Lý do format prompt/completion (thay vì messages + completion_only_loss):
    TRL dùng return_assistant_tokens_mask=True dựa trên template có `{% generation %}`
    keyword — template Qwen2.5 KHÔNG có → assistant_masks không được sinh → loss trên
    toàn sequence (bug cũ lặp lại âm thầm). Với prompt/completion chuỗi, TRL tự build
    completion_mask thủ công ([0]*prompt + [1]*completion) — không phụ thuộc template.
    """
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return {"prompt": prompt, "completion": full[len(prompt):]}


def _make_early_stopping(patience: int = 4, min_delta: float = 0.0):
    """Tạo EarlyStoppingCallback (transformers cung cấp, không cần đúng TRL)."""
    try:
        from transformers import EarlyStoppingCallback
        log.info(f"[EarlyStopping] patience={patience}, min_delta={min_delta}")
        return [EarlyStoppingCallback(early_stopping_patience=patience,
                                      early_stopping_delta=min_delta)]
    except Exception as e:
        log.warning(f"Không tạo được EarlyStoppingCallback: {e}")
        return []


def _build_trainer(trl_new, sft_trainer_cls, training_args_cls, model, tokenizer,
                   train_dataset, eval_dataset, merged_cfg, eval_strategy_key, inspect, sft_config_cls=None):
    """
    Tạo SFTTrainer tương thích với mọi phiên bản TRL.
    Dùng inspect để chỉ truyền parameters mà class thực sự hỗ trợ.
    """
    max_seq = TRAIN_CFG["max_seq_length"]

    trainer_params = inspect.signature(sft_trainer_cls.__init__).parameters

    # Tokenizer/processor param: trl 1.x = "processing_class", trl < 1.x = "tokenizer"
    tok_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"

    # ── Nhánh mới: SFTConfig (trl >= 0.12) ─────────────────────────────────
    if trl_new and sft_config_cls is not None:
        cfg_params = inspect.signature(sft_config_cls.__init__).parameters

        seq_key = "max_seq_length" if "max_seq_length" in cfg_params else "max_length"
        extra_cfg = {seq_key: max_seq}
        # completion_only_loss=True: labels = -100 trên mọi token prompt,
        # loss CHỈ trên completion. Với dataset prompt/completion chuỗi, TRL
        # tự build completion_mask ([0]*prompt + [1]*completion) KHÔNG cần
        # tokenizer template có `{% generation %}` (template Qwen2.5 thiếu).
        # KHÔNG đặt dataset_text_field ở đây: prompt-completion dataset không dùng.
        if "completion_only_loss" in cfg_params:
            extra_cfg["completion_only_loss"] = True
            log.info("[Loss] completion_only_loss=True → loss chỉ trên completion")
        else:
            log.warning("[Loss] SFTConfig thiếu completion_only_loss → TRL tự detect theo prompt/completion")
        if "packing" in cfg_params:
            extra_cfg["packing"] = False

        sft_args = sft_config_cls(
            output_dir=str(OUTPUT_DIR),
            **{eval_strategy_key: "steps"},
            **extra_cfg,
            **merged_cfg,
        )

        trainer_kwargs = {
            "model":         model,
            tok_key:         tokenizer,
            "train_dataset": train_dataset,
            "eval_dataset":  eval_dataset,
            "args":          sft_args,
        }
        if "callbacks" in trainer_params:
            trainer_kwargs["callbacks"] = _make_early_stopping()
        if "dataset_text_field" in trainer_params and "dataset_text_field" not in extra_cfg:
            trainer_kwargs["dataset_text_field"] = "text"

        return sft_trainer_cls(**trainer_kwargs)

    # ── Nhánh cũ (TRL < 0.12): dùng TrainingArguments + inspect để tránh
    #    hardcode "tokenizer="/"max_seq_length=" vốn không còn trong trl mới.
    #    TRL cũ không hỗ trợ completion_only_loss; nó có thể hiểu dataset
    #    prompt/completion chuỗi. Nếu không, gộp prompt+completion thành "text".
    _cols = getattr(train_dataset, "column_names", []) if train_dataset is not None else []
    if "prompt" in _cols and "completion" in _cols:
        log.warning("[Loss] TRL cũ: SFTTrainer có thể không tự mask completion → chuyển sang text (loss toàn sequence)")
        def _to_text(ex):
            return {"text": ex["prompt"] + " " + ex["completion"]}
        if train_dataset is not None:
            train_dataset = train_dataset.map(_to_text, remove_columns=["prompt", "completion"])
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(_to_text, remove_columns=["prompt", "completion"])

    training_args = training_args_cls(
        output_dir=str(OUTPUT_DIR),
        **{eval_strategy_key: "steps"},
        **merged_cfg,
    )
    trainer_kwargs = {
        "model":         model,
        tok_key:         tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset":  eval_dataset,
        "args":          training_args,
    }
    if "callbacks" in trainer_params:
        trainer_kwargs["callbacks"] = _make_early_stopping()
    # TRL cũ: nếu dataset đã convert thành text thì chỉ cần dataset_text_field.
    # TRL mới (nhánh trên) đã truyền args.ds_text_field nên không cần ở đây.
    for key, val in (("dataset_text_field", "text"),
                     ("max_seq_length", max_seq),
                     ("packing", False)):
        if key in trainer_params:
            trainer_kwargs[key] = val

    return sft_trainer_cls(**trainer_kwargs)


def _parse_cli_overrides() -> dict:
    """Đọc các override từ CLI: --model, --lora-rank, --epochs, --batch, --seq.
    Cho phép benchmark nhiều cấu hình mà KHÔNG cần sửa code (fix review #1, #5)."""
    args = sys.argv[1:]
    out = {}
    for flag, key, cast in (
        ("--model", "model", str),
        ("--lora-rank", "lora_rank", int),
        ("--epochs", "epochs", int),
        ("--batch", "batch_size", int),
        ("--seq", "max_seq_length", int),
    ):
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args):
                try:
                    out[key] = cast(args[i + 1])
                except ValueError:
                    log.warning(f"Ignore invalid {flag}: {args[i+1]}")
    return out


def _apply_cli_overrides(overrides: dict, vram_gb: float, use_gpu: bool):
    """Ghi override vào TRAIN_CFG / LORA_CFG dựa trên CLI (nếu có)."""
    if "lora_rank" in overrides:
        r = overrides["lora_rank"]
        LORA_CFG["r"] = r
        LORA_CFG["lora_alpha"] = 2 * r
        log.info(f"[CLI] LoRA rank override → r={r}, alpha={2*r}")
    if "epochs" in overrides:
        TRAIN_CFG["num_train_epochs"] = overrides["epochs"]
        log.info(f"[CLI] Epochs override → {overrides['epochs']}")
    if "batch_size" in overrides:
        TRAIN_CFG["per_device_train_batch_size"] = overrides["batch_size"]
        TRAIN_CFG["per_device_eval_batch_size"] = overrides["batch_size"]
        log.info(f"[CLI] Batch size override → {overrides['batch_size']}")
    if "max_seq_length" in overrides:
        TRAIN_CFG["max_seq_length"] = overrides["max_seq_length"]
        log.info(f"[CLI] Max seq length override → {overrides['max_seq_length']}")


def _pick_best_gpu() -> int:
    """Dùng nvidia-smi chọn GPU free nhất — KHÔNG cần torch, gọi trước khi import torch."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        free_mems = [int(x.strip()) for x in out.strip().split("\n") if x.strip()]
        best = max(range(len(free_mems)), key=lambda i: free_mems[i])
        log.info(f"nvidia-smi free VRAM (MB): {free_mems} → chọn GPU {best} ({free_mems[best]} MB free)")
        return best
    except Exception as e:
        log.warning(f"nvidia-smi không khả dụng ({e}), dùng GPU 0")
        return 0


def _detect_device(torch) -> tuple[bool, str, float]:
    """Phát hiện GPU/CPU và chọn model. CUDA_VISIBLE_DEVICES đã được set trước khi torch init."""
    global _USE_BF16_FULL, LORA_CFG

    if not torch.cuda.is_available():
        log.warning("=" * 60)
        log.warning("Không có GPU — chạy trên CPU (chậm, mất 1-4 giờ).")
        log.warning("Model cố định: Qwen2.5-0.5B (nhỏ nhất)")
        log.warning("=" * 60)
        return False, MODEL_0_5B, 0.0

    # Sau khi set CUDA_VISIBLE_DEVICES, chỉ có 1 GPU → luôn là index 0
    gpu_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_gb  = torch.cuda.mem_get_info(0)[0] / 1e9
    log.info(f"GPU 0: {gpu_name} | {free_gb:.1f}/{total_gb:.1f} GB free")

    # Kiểm tra disk space để tránh download model lớn hơn dung lượng trống
    import shutil
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    disk_free_gb = shutil.disk_usage(cache_dir if cache_dir.exists() else Path.home()).free / 1e9
    log.info(f"Disk free: {disk_free_gb:.1f} GB")

    # Dung lượng disk tối thiểu để download (bf16 weights, chưa tính overhead)
    # QLoRA 4-bit chiếm khoảng 50% so với bf16
    MODEL_DISK_REQ = {
        MODEL_72B: 150, MODEL_32B: 65, MODEL_14B: 30,
        MODEL_7B: 15,   MODEL_3B: 7,   MODEL_1_5B: 3, MODEL_0_5B: 1,
    }

    # ── Chọn model theo free VRAM ─────────────────────────────────────────────
    # QLoRA 4-bit: VRAM ≈ params × 0.5 bytes + LoRA overhead + activations
    # bf16 full  : VRAM ≈ params × 2 bytes  (không dùng BitsAndBytesConfig)
    #
    # VRAM thresholds (dư thoải mái cho batch lớn + activations):
    #   ≥ 70 GB → 72B QLoRA  (~40GB weights + 30GB activations/batch)
    #   ≥ 50 GB → 32B QLoRA  (~20GB weights + 30GB cho batch lớn)
    #   ≥ 30 GB → 14B bf16   (~28GB weights, không cần QLoRA)
    #   ≥ 16 GB → 14B QLoRA  (~7GB  weights + overhead)
    #   ≥ 13 GB → 7B  bf16   (~14GB weights)
    #   ≥ 7 GB  → 3B  (QLoRA hoặc bf16)
    #   ≥ 4 GB  → 1.5B
    #   < 4 GB  → 0.5B
    if free_gb >= 70:
        model_name = MODEL_72B
        _USE_BF16_FULL = False          # QLoRA 4-bit (bf16 thuần cần >140GB)
        LORA_CFG["r"]          = 128
        LORA_CFG["lora_alpha"] = 256
        log.info(f"[Config] {free_gb:.0f}GB free → 72B QLoRA | LoRA r=128")
    elif free_gb >= 50:
        model_name = MODEL_32B
        _USE_BF16_FULL = False          # QLoRA 4-bit (~20GB weights, dư 30GB cho batch)
        LORA_CFG["r"]          = 128
        LORA_CFG["lora_alpha"] = 256
        log.info(f"[Config] {free_gb:.0f}GB free → 32B QLoRA | LoRA r=128")
    elif free_gb >= 30:
        model_name = MODEL_14B
        _USE_BF16_FULL = True           # bf16 thuần (~28GB), không cần QLoRA
        LORA_CFG["r"]          = 64
        LORA_CFG["lora_alpha"] = 128
        log.info(f"[Config] {free_gb:.0f}GB free → 14B bf16-full | LoRA r=64")
    elif free_gb >= 16:
        model_name = MODEL_14B
        _USE_BF16_FULL = False          # QLoRA 4-bit (~7GB weights)
        LORA_CFG["r"]          = 64
        LORA_CFG["lora_alpha"] = 128
        log.info(f"[Config] {free_gb:.0f}GB free → 14B QLoRA | LoRA r=64")
    elif free_gb >= 13:
        model_name = MODEL_7B
        _USE_BF16_FULL = True           # bf16 thuần (~14GB)
        LORA_CFG["r"]          = 64
        LORA_CFG["lora_alpha"] = 128
        log.info(f"[Config] {free_gb:.0f}GB free → 7B bf16-full | LoRA r=64")
    elif free_gb >= 7:
        model_name = MODEL_3B
        _USE_BF16_FULL = False
        log.info(f"[Config] {free_gb:.0f}GB free → 3B QLoRA | LoRA r=32")
    elif free_gb >= 4:
        model_name = MODEL_1_5B
        _USE_BF16_FULL = False
        log.info(f"[Config] {free_gb:.0f}GB free → 1.5B QLoRA | LoRA r=32")
    else:
        model_name = MODEL_0_5B
        _USE_BF16_FULL = False
        log.info(f"[Config] {free_gb:.0f}GB free → 0.5B QLoRA | LoRA r=32")

    # ── Kiểm tra disk và fallback ─────────────────────────────────────────────
    fallback_order = [model_name, MODEL_32B, MODEL_14B, MODEL_7B,
                      MODEL_3B, MODEL_1_5B, MODEL_0_5B]
    for candidate in fallback_order:
        cached = cache_dir / f"models--{candidate.replace('/', '--')}"
        if cached.exists():
            if candidate != model_name:
                log.info(f"Model {model_name} chưa cache, dùng {candidate} (đã có sẵn)")
                model_name = candidate
            break
        if disk_free_gb >= MODEL_DISK_REQ[candidate] + 5:
            if candidate != model_name:
                log.warning(f"Disk không đủ cho {model_name} → fallback {candidate}")
                model_name = candidate
            break
        log.warning(f"Disk không đủ cho {candidate} ({MODEL_DISK_REQ[candidate]}GB cần) → thử nhỏ hơn")

    log.info(
        f"✓ Model cuối cùng: {model_name} "
        f"(VRAM {free_gb:.1f}GB, Disk {disk_free_gb:.1f}GB free, "
        f"bf16_full={_USE_BF16_FULL}, LoRA r={LORA_CFG['r']})"
    )
    return True, model_name, free_gb


def _load_model(torch, model_cls, model_name: str, use_gpu: bool):
    """Load model với bf16-full / QLoRA 4-bit (GPU) hoặc float32 (CPU).
    
    _USE_BF16_FULL=True  → load bf16 thuần, không dùng BitsAndBytes (tốt hơn khi đủ VRAM)
    _USE_BF16_FULL=False → load QLoRA 4-bit NF4 + double quant (tiết kiệm VRAM)
    """
    log.info(f"Loading model: {model_name} | bf16_full={_USE_BF16_FULL}")
    if not use_gpu:
        return model_cls.from_pretrained(
            model_name, device_map="cpu",
            torch_dtype=torch.float32, trust_remote_code=True,
        )

    # CUDA_VISIBLE_DEVICES đã được set → chỉ 1 GPU visible
    if _USE_BF16_FULL:
        # bf16 thuần — chất lượng gradient tốt hơn QLoRA, không cần bitsandbytes
        log.info("Dùng bf16 full precision (không QLoRA)")
        return model_cls.from_pretrained(
            model_name,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    # QLoRA 4-bit NF4 + double quantization — tiết kiệm VRAM tối đa
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,   # bfloat16 > float16 cho stability
            bnb_4bit_use_double_quant=True,           # double quant tiết kiệm thêm ~0.4 bits/param
        )
        log.info("Dùng QLoRA 4-bit NF4 + double quant")
        return model_cls.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda:0",
            trust_remote_code=True,
        )
    except Exception as e:
        log.warning(f"QLoRA thất bại ({e}), fallback bf16")
        return model_cls.from_pretrained(
            model_name, device_map="cuda:0",
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        )


def train():
    import os as _os

    # QUAN TRỌNG: set CUDA_VISIBLE_DEVICES TRƯỚC KHI import torch
    # Nếu set sau khi torch đã init CUDA context thì không có hiệu lực
    if "CUDA_VISIBLE_DEVICES" not in _os.environ:
        best_gpu = _pick_best_gpu()
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        log.info(f"CUDA_VISIBLE_DEVICES={best_gpu} (set trước torch init)")
    else:
        log.info(f"CUDA_VISIBLE_DEVICES={_os.environ['CUDA_VISIBLE_DEVICES']} (từ môi trường)")

    cli = _parse_cli_overrides()
    use_gpu, model_name, vram_gb = _detect_device(torch)
    if "model" in cli:
        model_name = cli["model"]
        log.info(f"[CLI] Model override → {model_name}")
    # _apply_cli_overrides() được gọi SAU khối VRAM override (bên dưới) để
    # CLI luôn thắng; chỉ model được áp trước vì cần cho load tokenizer/model.

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model

        # ── Import SFTConfig / SFTTrainer — thử nhiều path để tương thích
        #    mọi phiên bản TRL (0.12 → 1.x): submodule riêng, rồi top-level.
        SFTConfig = SFTTrainer = None
        for cfg_path, tr_path in (
            # trl >= 1.x: submodule trl.trainer.sft_*  (cùng path là ổn định nhất)
            ("trl.trainer.sft_config", "trl.trainer.sft_trainer"),
            # trl top-level exports (1.x đôi khi chỉ có ở đây)
            ("trl.SFTConfig", "trl.SFTTrainer"),
        ):
            try:
                import importlib
                SFTConfig = getattr(importlib.import_module(cfg_path), "SFTConfig", None)
                SFTTrainer = getattr(importlib.import_module(tr_path), "SFTTrainer", None)
                if SFTConfig is not None and SFTTrainer is not None:
                    break
            except (ImportError, AttributeError):
                SFTConfig = SFTTrainer = None

        if SFTTrainer is None:
            # Cuối cùng: thử import trực tiếp (bắt mọi exception, không chỉ ImportError)
            try:
                from trl import SFTTrainer
            except Exception:
                SFTTrainer = None
            if SFTConfig is None:
                try:
                    from trl import SFTConfig
                except Exception:
                    SFTConfig = None

        if SFTTrainer is None:
            raise ImportError("Không import được SFTTrainer từ trl")

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

# QUAN TRỌNG (fix completion-only loss):
    # - KHÔNG pre-apply_chat_template thành cột "text" → TRL coi là language modeling →
    #   loss trên TOÀN BỘ sequence (bug cũ).
    # - KHÔNG dùng cột messages + completion_only_loss=True: cách này dựa vào
    #   tokenizer template có keyword `{% generation %}` để sinh assistant_tokens_mask;
    #   template Qwen2.5 KHÔNG có → TRL âm thầm fallback về loss toàn sequence.
    # - ĐÚNG: chia thành cột `prompt`/`completion` (chuỗi). TRL detect đây là
    #   prompt-completion dataset → tự build completion_mask thủ công ([0]*prompt +
    #   [1]*completion) → labels = -100 trên mọi token prompt, loss CHỈ trên completion.
    train_dataset = Dataset.from_dict([
        _split_prompt_completion(e["messages"], tokenizer) for e in train_raw
    ])
    eval_dataset = Dataset.from_dict([
        _split_prompt_completion(e["messages"], tokenizer) for e in eval_raw
    ])

    model = _load_model(torch, AutoModelForCausalLM, model_name, use_gpu)  # noqa: F821
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**LORA_CFG))
    # Báo Trainer rằng model tự quản lý device_map → không wrap DataParallel
    # Fix: Don't set boolean on tensor attributes
    # model.is_parallelizable = True
    # model.model_parallel = True
    # These attributes may not exist or may be tensors - skip them
    pass
    if use_gpu and TRAIN_CFG.get("gradient_checkpointing"):
        # Fix: enable_input_require_grads may be a property or a tensor in some model versions.
        # Only invoke it when it's actually a callable method.
        value = getattr(model, 'enable_input_require_grads', None)
        if callable(value) and not isinstance(value, torch.Tensor):
            value()
        elif hasattr(model, 'enable_input_require_grads'):
            # It's a property, just access it to trigger the setter without calling it.
            _ = getattr(model, 'enable_input_require_grads')
    model.print_trainable_parameters()

    import trl as _trl
    import transformers as _tf
    _trl_ver  = tuple(int(x) for x in _trl.__version__.split(".")[:2])
    _tf_ver   = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    _trl_new  = _trl_ver >= (0, 12)   # SFTConfig thay TrainingArguments
    _tf_new   = _tf_ver  >= (4, 46)   # eval_strategy thay evaluation_strategy

    # warmup_ratio deprecated → tính warmup_steps sau khi đã apply VRAM/CLI overrides
    # (batch size quyết định số steps/epoch → phải tính sau override, không phải trước)

    # Override config theo FREE VRAM thực tế (quan trọng trên server dùng chung)
    if use_gpu:
        if vram_gb >= 70:          # A100/H100 80GB → 32B QLoRA, batch an toàn
            TRAIN_CFG["max_seq_length"]              = 2048
            TRAIN_CFG["gradient_checkpointing"]      = True
            TRAIN_CFG["num_train_epochs"]            = 5
            TRAIN_CFG["per_device_train_batch_size"] = 4
            TRAIN_CFG["per_device_eval_batch_size"]  = 4
            TRAIN_CFG["gradient_accumulation_steps"] = 8   # effective batch = 32
            TRAIN_CFG["learning_rate"]               = 8e-5
            TRAIN_CFG["optim"]                       = "paged_adamw_8bit"
            log.info(f"[Config] {vram_gb:.0f}GB free → seq=2048, epochs=5, batch=4 (grad_ckpt=True, eff_batch=32)")
        elif vram_gb >= 50:        # 50–70GB → 32B QLoRA, batch lớn
            TRAIN_CFG["max_seq_length"]              = 4096
            TRAIN_CFG["gradient_checkpointing"]      = False
            TRAIN_CFG["num_train_epochs"]            = 10
            TRAIN_CFG["per_device_train_batch_size"] = 16
            TRAIN_CFG["per_device_eval_batch_size"]  = 16
            TRAIN_CFG["gradient_accumulation_steps"] = 1
            TRAIN_CFG["learning_rate"]               = 8e-5
            TRAIN_CFG["optim"]                       = "paged_adamw_8bit"
            log.info(f"[Config] {vram_gb:.0f}GB free → seq=4096, epochs=10, batch=16 (32B QLoRA mode)")
        elif vram_gb >= 30:        # 30–50GB → 14B bf16, batch vừa
            TRAIN_CFG["max_seq_length"]              = 4096
            TRAIN_CFG["gradient_checkpointing"]      = False
            TRAIN_CFG["num_train_epochs"]            = 8
            TRAIN_CFG["per_device_train_batch_size"] = 8
            TRAIN_CFG["per_device_eval_batch_size"]  = 8
            TRAIN_CFG["gradient_accumulation_steps"] = 2
            TRAIN_CFG["learning_rate"]               = 1e-4
            log.info(f"[Config] {vram_gb:.0f}GB free → seq=4096, epochs=8, batch=8 (14B bf16 mode)")
        elif vram_gb >= 13:        # 13–30GB → 7B/14B, seq dài
            TRAIN_CFG["max_seq_length"]              = 2048
            TRAIN_CFG["gradient_checkpointing"]      = False
            TRAIN_CFG["num_train_epochs"]            = 7
            TRAIN_CFG["per_device_train_batch_size"] = 8
            TRAIN_CFG["learning_rate"]               = 1.2e-4
            log.info(f"[Config] {vram_gb:.0f}GB free → seq=2048, epochs=7, no grad_ckpt")
        elif vram_gb >= 7:         # 7–13GB free → 3B/1.5B model với grad_ckpt
            TRAIN_CFG["max_seq_length"]              = 1024
            TRAIN_CFG["gradient_checkpointing"]      = True
            TRAIN_CFG["num_train_epochs"]            = 5
            TRAIN_CFG["per_device_train_batch_size"] = 4
            log.info(f"[Config] {vram_gb:.0f}GB free → seq=1024, epochs=5, grad_ckpt=on")
        elif vram_gb >= 4:         # 4–7GB free → 1.5B/0.5B, batch nhỏ
            TRAIN_CFG["max_seq_length"]              = 512
            TRAIN_CFG["gradient_checkpointing"]      = True
            TRAIN_CFG["num_train_epochs"]            = 3
            TRAIN_CFG["per_device_train_batch_size"] = 2
            log.info(f"[Config] {vram_gb:.0f}GB free → seq=512, epochs=3, batch=2")
        else:                      # <4GB free → tối thiểu
            TRAIN_CFG["max_seq_length"]              = 256
            TRAIN_CFG["gradient_checkpointing"]      = True
            TRAIN_CFG["num_train_epochs"]            = 2
            TRAIN_CFG["per_device_train_batch_size"] = 1
            log.info(f"[Config] {vram_gb:.0f}GB free → seq=256, epochs=2 (minimal)")

    # CLI overrides (epochs/batch/seq/lora-rank) áp SAU VRAM block → luôn thắng
    _apply_cli_overrides(cli, vram_gb, use_gpu)

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
    # Tính warmup_steps dựa trên batch size CUỐI CÙNG (sau override)
    n_steps_per_epoch = max(1, len(train_dataset) //
                            TRAIN_CFG["per_device_train_batch_size"])
    total_steps = n_steps_per_epoch * TRAIN_CFG["num_train_epochs"]
    warmup_steps = max(1, int(total_steps * TRAIN_CFG["warmup_ratio"]))
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
    # Fix: merge_and_unload should always be called as a method
    merge_and_unload_fn = getattr(model, "merge_and_unload", None)
    if callable(merge_and_unload_fn):
        model = merge_and_unload_fn()
    else:
        log.warning("merge_and_unload not found, returning base model")
        model = base_model

    merged_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving merged model → {merged_dir}")
    if hasattr(model, "save_pretrained") and callable(getattr(model, "save_pretrained", None)):
        model.save_pretrained(str(merged_dir))  # type: ignore
    else:
        log.error("Model does not have save_pretrained method")
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
