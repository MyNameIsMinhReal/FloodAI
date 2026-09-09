# -*- coding: utf-8 -*-
"""
depth_analysis/vlm_verifier.py
===============================
VLM (Vision-Language Model) verifier — bước kiểm định CUỐI CÙNG của pipeline.

Vai trò:
    Nhận ảnh gốc + kết quả đo từ CV pipeline (mực nước, chiều cao người,
    số lượng phương tiện...) → dùng VLM (Qwen2.5-VL) "nhìn" ảnh và xác minh
    từng tuyên bố. Ví dụ: pipeline đo người cao 1.7m nhưng dáng người trong
    ảnh chỉ tương đương 1.5m → flag MISMATCH, đề xuất giá trị sửa.

Flow trong pipeline:
    ... → raincoat → **vlm_verify** → postprocess → store → learn

Thiết kế an toàn:
    - Không bao giờ làm crash pipeline: mọi lỗi → verdict "skipped"/"error".
    - Chỉ auto-correct khi VLM tự tin (confidence >= ngưỡng config).
    - Model load lười (lazy) — không tải nếu stage bị tắt.

Sử dụng trực tiếp:
    from depth_analysis.vlm_verifier import VLMVerifier
    v = VLMVerifier(cfg)
    verdict = v.verify(image_path, result_dict)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("depth.vlm_verifier")

# Chiều cao người Việt trưởng thành hợp lý (cm) — dùng để sanity-check VLM
_PERSON_H_MIN, _PERSON_H_MAX = 120.0, 210.0
_WATER_CM_MAX = 400.0          # mực nước > 4m gần như chắc chắn sai
_MAX_IMAGE_SIDE = 1024         # downscale ảnh trước khi đưa vào VLM


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Lấy field từ dict HOẶC object (dataclass) — pipeline có cả 2 dạng."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    else:
        try:
            setattr(obj, key, value)
        except Exception:
            pass


class VLMVerifier:
    """
    Verifier dùng Vision-Language Model (mặc định Qwen2.5-VL).

    Args:
        cfg: config toàn cục; đọc section `vlm_verify` và `models.vlm`.

    Config keys (config.yaml):
        models.vlm:                    tên model HF (Qwen/Qwen2.5-VL-7B-Instruct)
        vlm_verify.device:             "auto" | "cuda" | "cpu"
        vlm_verify.max_new_tokens:     số token sinh tối đa
        vlm_verify.tolerance_pct:      % sai lệch cho phép trước khi flag mismatch
        vlm_verify.min_abs_diff_cm:    chênh tuyệt đối tối thiểu (cm) để flag
        vlm_verify.auto_correct:       tự sửa water_height_cm khi mismatch + tự tin
        vlm_verify.min_confidence_to_correct: ngưỡng confidence để được sửa
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.cfg = cfg
        vcfg: Dict[str, Any] = cfg.get("vlm_verify", {}) or {}
        # Chuỗi rỗng trong config = chưa cấu hình → giữ nguyên để stage check
        self.model_name: str = (
            (cfg.get("models", {}) or {}).get("vlm")
            or vcfg.get("model")
            or ""
        )
        self.device_pref: str   = vcfg.get("device", "auto")
        self.max_new_tokens:int = int(vcfg.get("max_new_tokens", 512))
        self.tolerance_pct: float   = float(vcfg.get("tolerance_pct", 30))
        self.min_abs_diff_cm: float = float(vcfg.get("min_abs_diff_cm", 12))
        self.auto_correct: bool     = bool(vcfg.get("auto_correct", True))
        self.min_conf_to_fix: float = float(vcfg.get("min_confidence_to_correct", 0.65))

        self._model = None
        self._processor = None
        self._device = None
        self._load_lock = threading.Lock()
        self._load_failed = False

    # ── Loading ────────────────────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        """Giải phóng model khỏi RAM/VRAM."""
        self._model = None
        self._processor = None
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("  [VLMVerifier] Model unloaded")

    def _ensure_loaded(self) -> None:
        if self.is_loaded():
            return
        if self._load_failed:
            raise RuntimeError("VLM đã fail load trước đó — bỏ qua")
        with self._load_lock:
            if self.is_loaded():
                return
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor

        device = self._resolve_device(torch)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        log.info(f"  [VLMVerifier] Loading {self.model_name} → {device} ({dtype}) …")

        processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

        model = self._load_architecture(torch, dtype)
        model.to(device)
        model.eval()

        self._model, self._processor, self._device = model, processor, device
        log.info("  [VLMVerifier] Model sẵn sàng")

    def _resolve_device(self, torch):
        pref = self.device_pref.lower()
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("vlm_verify.device='cuda' nhưng không có GPU")
            return torch.device("cuda:0")
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_architecture(self, torch, dtype):
        name = self.model_name
        # 1) Qwen2.5-VL / Qwen2-VL (ưu tiên — class riêng, hỗ trợ tốt box ảo)
        for cls_path in ("Qwen2_5_VLForConditionalGeneration",
                         "Qwen2VLForConditionalGeneration"):
            try:
                import transformers as tf
                cls = getattr(tf, cls_path, None)
                if cls is None:
                    continue
                return cls.from_pretrained(name, torch_dtype=dtype)
            except Exception as exc:
                log.debug(f"  [VLMVerifier] {cls_path} fail: {exc}")
        # 2) Generic vision-to-text
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq.from_pretrained(name, torch_dtype=dtype)

    # ── Public API ─────────────────────────────────────────────────────────────

    def verify(self, image_path: Any, result: Any) -> Dict[str, Any]:
        """
        Xác minh kết quả CV pipeline cho 1 ảnh.

        Args:
            image_path: đường dẫn ảnh gốc (ưu tiên) — fallback overlay.
            result:     ReferenceFloodResult (dataclass) hoặc dict.

        Returns:
            verdict dict:
            {
              "verdict":  "agree" | "disagree" | "uncertain" | "skipped" | "error",
              "confidence": float,
              "water":    {"claimed_cm","vlm_cm","status","diff_cm"},
              "persons":  [{"claimed_height_cm","vlm_height_cm","status"}],
              "counts":   {...},       # đếm thực tế VLM nhìn thấy
              "corrected_water_cm": float|None,   # đề xuất sửa
              "applied_correction": bool,
              "notes": str,
              "raw":    str (500 ký tự đầu),
            }
        """
        claims = self.build_claims(result)
        base = {
            "verdict": "uncertain", "confidence": 0.0,
            "water": None, "persons": [], "counts": {},
            "corrected_water_cm": None, "applied_correction": False,
            "notes": "", "raw": "",
        }

        img_file = self._pick_image(image_path, result)
        if img_file is None:
            base["verdict"], base["notes"] = "skipped", "Không tìm thấy file ảnh"
            return base

        has_claims = (
            claims["water_height_cm"] is not None or claims["persons"]
            or claims["vehicles"]
        )
        if not has_claims:
            base["verdict"], base["notes"] = "skipped", "Không có tuyên bố nào để xác minh"
            return base

        try:
            self._ensure_loaded()
            raw = self._generate(img_file, claims)
        except Exception as exc:
            log.warning(f"  [VLMVerifier] Lỗi generate ({type(exc).__name__}): {exc}")
            base["verdict"] = "error"
            base["notes"] = f"{type(exc).__name__}: {exc}"[:200]
            return base

        parsed = self._parse_json(raw)
        base["raw"] = (raw or "")[:500]
        if parsed is None:
            base["verdict"] = "error"
            base["notes"] = "VLM trả về JSON không hợp lệ"
            return base

        return self._compare(claims, parsed)

    def build_claims(self, result: Any) -> Dict[str, Any]:
        """Trích xuất các tuyên bố cần xác minh từ result."""
        objects: List[dict] = _get(result, "detected_objects", []) or []
        persons, others = [], []
        for o in objects:
            cls = str(_get(o, "class_name", "")).lower()
            entry = {
                "class_name": _get(o, "class_name", ""),
                "ref_height_cm": float(_get(o, "ref_height_cm", 0) or 0),
                "estimated_height_cm": float(_get(o, "estimated_height_cm", 0) or 0),
                "water_height_cm": float(_get(o, "water_height_cm", 0) or 0),
                "pose_factor": float(_get(o, "pose_factor", 1) or 1),
            }
            if cls == "person":
                persons.append(entry)
            else:
                others.append(entry)

        vehicles = _get(result, "vehicles_detected", []) or []

        return {
            "water_height_cm": _get(result, "water_height_cm"),
            "flood_level": _get(result, "flood_level", ""),
            "persons": persons,
            "other_objects": others,
            "vehicles": [
                {"vehicle_type": _get(v, "vehicle_type", _get(v, "class_name", "")),
                 "submerged_pct": _get(v, "submerged_pct", None)}
                for v in vehicles
            ],
            "has_raincoat": _get(result, "has_raincoat", None),
        }

    # ── Prompt & generation ───────────────────────────────────────────────────

    def _build_prompt(self, claims: Dict[str, Any]) -> str:
        claims_json = {
            "water_height_cm": claims["water_height_cm"],
            "flood_level":     claims["flood_level"],
            "n_persons":       len(claims["persons"]),
            "person_heights_cm": [p["estimated_height_cm"] or p["ref_height_cm"]
                                  for p in claims["persons"]],
            "vehicles":        claims["vehicles"],
        }
        return (
            "Bạn là chuyên gia kiểm định đo ngập lụt bằng thị giác máy tính.\n"
            "Hệ thống CV đã phân tích ảnh này và đưa ra các kết quả sau:\n"
            f"{json.dumps(claims_json, ensure_ascii=False)}\n\n"
            "Nhiệm vụ: QUAN SÁT ẢNH và xác minh từng con số trên.\n"
            "Gợi ý đối chiếu:\n"
            "- Mực nước: so với mắt cá chân (~10cm), đầu gối (~50cm), đùi (~80cm), "
            "thắt lưng (~100cm), ngực (~130cm); với xe máy nhìn bánh xe/báo số.\n"
            "- Chiều cao người trưởng thành Việt Nam thường 155-175cm; "
            "trẻ em thấp hơn rõ rệt. Nếu người đang ngồi/chụp xa, ghi chú uncertain.\n"
            "- Đếm số người, ô tô, xe máy thực sự thấy được.\n\n"
            "Chỉ trả về DUY NHẤT một JSON hợp lệ, không giải thích thêm:\n"
            '{"verdict":"agree|disagree|uncertain",'
            '"confidence":<0.0-1.0>,'
            '"vlm_water_height_cm":<số hoặc null>,'
            '"vlm_persons_height_cm":[<số...>],'
            '"counts":{"person":<n>,"car":<n>,"motorbike":<n>},'
            '"notes":"nhận xét ngắn <=30 từ"}'
        )

    def _generate(self, image_file: Path, claims: Dict[str, Any]) -> str:
        from PIL import Image

        prompt = self._build_prompt(claims)
        image = Image.open(image_file).convert("RGB")
        image = self._downscale(image)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        proc = self._processor
        text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = proc(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with __import__("torch").no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,           # deterministic cho verification
                temperature=None,
                top_p=None,
            )
        gen = out_ids[:, inputs["input_ids"].shape[1]:]
        return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _downscale(image, max_side: int = _MAX_IMAGE_SIDE):
        w, h = image.size
        if max(w, h) <= max_side:
            return image
        scale = max_side / max(w, h)
        return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # ── Parsing & so sánh ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """Trích JSON từ text VLM (chống code fence / chữ thừa)."""
        if not raw:
            return None
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        blob = cleaned[start:end + 1]
        try:
            data = json.loads(blob)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            try:  # fix phổ biến: dấu phẩy thừa trước }
                fixed = re.sub(r",\s*([}\]])", r"\1", blob)
                data = json.loads(fixed)
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f == f else None   # loại NaN

    def _status_for(self, claimed: Optional[float], vlm: Optional[float]) -> tuple:
        """→ (status, diff_cm). mismatch khi vượt CẢ % lẫn chênh tuyệt đối."""
        if claimed is None or vlm is None:
            return "unsure", None
        diff = abs(vlm - claimed)
        tol = max(claimed * self.tolerance_pct / 100.0, self.min_abs_diff_cm)
        return ("mismatch" if diff > tol else "ok"), round(diff, 1)

    def _compare(self, claims: Dict[str, Any], parsed: dict) -> Dict[str, Any]:
        conf = min(max(self._to_float(parsed.get("confidence")) or 0.0, 0.0), 1.0)
        verdict_raw = str(parsed.get("verdict", "uncertain")).lower().strip()
        if verdict_raw not in ("agree", "disagree", "uncertain"):
            verdict_raw = "uncertain"

        # ── Water ──────────────────────────────────────────────────────────────
        claimed_w = self._to_float(claims["water_height_cm"])
        vlm_w     = self._to_float(parsed.get("vlm_water_height_cm"))
        if vlm_w is not None and not (0 <= vlm_w <= _WATER_CM_MAX):
            vlm_w = None                      # ngoài vùng hợp lý → bỏ qua
        w_status, w_diff = self._status_for(claimed_w, vlm_w)
        water = {
            "claimed_cm": claimed_w, "vlm_cm": vlm_w,
            "status": w_status, "diff_cm": w_diff,
        }

        # ── Persons ────────────────────────────────────────────────────────────
        vlm_persons = parsed.get("vlm_persons_height_cm") or []
        if not isinstance(vlm_persons, list):
            vlm_persons = []
        person_checks = []
        corrected_persons: List[float] = []
        for i, p in enumerate(claims["persons"]):
            claimed_h = p["estimated_height_cm"] or p["ref_height_cm"] or None
            vlm_h = self._to_float(
                vlm_persons[i] if i < len(vlm_persons) else None
            )
            if vlm_h is not None and not (_PERSON_H_MIN <= vlm_h <= _PERSON_H_MAX):
                vlm_h = None
            status, diff = self._status_for(claimed_h, vlm_h)
            person_checks.append({
                "claimed_height_cm": claimed_h, "vlm_height_cm": vlm_h,
                "status": status, "diff_cm": diff,
            })
            corrected_persons.append(vlm_h if vlm_h is not None else claimed_h)

        counts = parsed.get("counts") if isinstance(parsed.get("counts"), dict) else {}

        # ── Tổng hợp verdict ───────────────────────────────────────────────────
        any_mismatch = (w_status == "mismatch") or \
                       any(pc["status"] == "mismatch" for pc in person_checks)
        all_ok = (w_status in ("ok", None)) and \
                 all(pc["status"] == "ok" for pc in person_checks)

        if verdict_raw == "disagree" or any_mismatch:
            verdict = "disagree"
        elif all_ok and verdict_raw == "agree":
            verdict = "agree"
        else:
            verdict = "uncertain"

        # Đề xuất sửa mực nước: chỉ khi mismatch và VLM có số hợp lý
        corrected_water = None
        if w_status == "mismatch" and vlm_w is not None:
            corrected_water = vlm_w

        return {
            "verdict":    verdict,
            "confidence": conf,
            "water":      water,
            "persons":    person_checks,
            "corrected_persons": corrected_persons,
            "counts":     {k: v for k, v in counts.items()
                           if isinstance(v, (int, float))},
            "corrected_water_cm": corrected_water,
            "applied_correction": False,     # stage quyết định và set lại
            "notes":     str(parsed.get("notes", ""))[:200],
            "raw":       "",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_image(image_path: Any, result: Any) -> Optional[Path]:
        for cand in (image_path,
                     _get(result, "original_path"),
                     _get(result, "overlay_path")):
            if not cand:
                continue
            p = Path(str(cand))
            if p.exists():
                return p
        return None

    def apply_corrections(self, result: Any, verdict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Áp dụng correction lên result (in-place) nếu đủ điều kiện.
        Trả về verdict đã cập nhật applied_correction.
        """
        if verdict.get("verdict") != "disagree" or not self.auto_correct:
            return verdict
        if verdict.get("confidence", 0.0) < self.min_conf_to_fix:
            return verdict

        changed = False
        new_w = verdict.get("corrected_water_cm")
        if new_w is not None:
            old_w = _get(result, "water_height_cm")
            try:
                from utils.constants import classify_level
                lvl, desc = classify_level(float(new_w))
                _set(result, "water_height_cm", round(float(new_w), 1))
                _set(result, "flood_level", lvl)
                _set(result, "flood_level_desc", desc)
                _set(result, "water_height_range",
                     f"{max(0, int(new_w - 10))}-{int(new_w + 10)} cm")
                log.info(
                    f"  [VLMVerifier] Sửa mực nước {old_w}cm → {new_w}cm ({lvl})"
                )
                changed = True
            except Exception as exc:
                log.debug(f"  [VLMVerifier] classify_level fail: {exc}")

        corrected_persons = verdict.get("corrected_persons") or []
        objects = _get(result, "detected_objects", []) or []
        persons_idx = [
            i for i, o in enumerate(objects)
            if str(_get(o, "class_name", "")).lower() == "person"
        ]
        for j, i in enumerate(persons_idx):
            if j < len(corrected_persons):
                new_h = corrected_persons[j]
                if new_h and abs(new_h - (_get(objects[i], "estimated_height_cm") or 0)) > 0.5:
                    _set(objects[i], "estimated_height_cm", round(float(new_h), 1))
                    changed = True

        verdict["applied_correction"] = changed
        return verdict


# ── Singleton tiện dụng ───────────────────────────────────────────────────────

_instance: Optional[VLMVerifier] = None
_inst_lock = threading.Lock()


def get_vlm_verifier(cfg: Optional[dict] = None) -> VLMVerifier:
    global _instance
    if _instance is None:
        with _inst_lock:
            if _instance is None:
                _instance = VLMVerifier(cfg)
    return _instance
