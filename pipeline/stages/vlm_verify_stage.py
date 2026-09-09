# -*- coding: utf-8 -*-
"""
pipeline/stages/vlm_verify_stage.py
====================================
Stage kiểm định chéo bằng VLM — chạy SAU CÙNG trước khi xuất kết quả.

Vị trí trong flow:
    input → analyze → depth → raincoat → **vlm_verify** → postprocess → store → learn

Nhiệm vụ:
    Với mỗi ảnh đã đo:
      1. Gửi ảnh gốc + các con số pipeline đo được cho VLM (Qwen2.5-VL).
      2. VLM "nhìn" ảnh và xác minh: mực nước, chiều cao từng người,
         số người/xe...
      3. Flag MISMATCH (vd: người nhìn ~1.5m nhưng pipeline ra 1.7m).
      4. Nếu `auto_correct` bật và VLM đủ tự tin → sửa số trực tiếp trên
         result (kèm lưu verdict để truy vết), nếu không chỉ ghi nhận.
      5. Kết quả verify gắn vào từng result (`vlm_verification`) và tổng hợp
         vào `state.vlm_verifications` để xuất báo cáo.

Config:
    skip_vlm_verify: true          # tắt stage
    vlm_verify:
      enabled: true
      auto_correct: true
      unload_after: false          # giải phóng VRAM sau khi chạy xong

An toàn:
    - Stage KHÔNG BAO GIỜ raise làm hỏng pipeline — lỗi từng ảnh chỉ ghi log.
    - Model load fail (máy yếu / chưa download) → toàn bộ verdict = "skipped".
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List

from pipeline.stage_base import StageBase

if TYPE_CHECKING:
    from pipeline.orchestrator import PipelineState

log = logging.getLogger("pipeline.vlm_verify")


class VLMVerifyStage(StageBase):
    """Xác minh kết quả đo bằng Vision-Language Model."""

    name = "vlm_verify"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        vcfg = cfg.get("vlm_verify", {}) or {}
        # StageBase._enabled dựa vào skip_<name>; cộng thêm enabled flag riêng
        self._enabled = self._enabled and bool(vcfg.get("enabled", True))
        self.unload_after = bool(vcfg.get("unload_after", False))
        self._verifier = None

    # ── StageBase API ──────────────────────────────────────────────────────────

    def process(self, state: "PipelineState") -> "PipelineState":
        if not state.depth_results:
            log.info("  [VLMVerify] Không có depth results — bỏ qua")
            return state

        self._verifier = self._get_verifier()
        if self._verifier is None:
            for r in state.depth_results:
                self._attach(r, {
                    "verdict": "skipped", "confidence": 0.0,
                    "notes": "VLM không khả dụng", "applied_correction": False,
                })
            state.vlm_verifications = []
            return state

        verdicts: List[dict] = []
        n_ok = n_fix = n_bad = 0

        for i, result in enumerate(state.depth_results):
            img = self._image_for(state, result, i)
            try:
                verdict = self._verifier.verify(img, result)
            except Exception as exc:
                log.warning(f"  [VLMVerify] Lỗi ảnh {i}: {exc}")
                verdict = {"verdict": "error", "confidence": 0.0,
                           "notes": str(exc)[:200], "applied_correction": False}

            if verdict.get("verdict") == "disagree":
                try:
                    verdict = self._verifier.apply_corrections(result, verdict)
                except Exception as exc:
                    log.debug(f"  [VLMVerify] apply_corrections fail: {exc}")

            self._attach(result, verdict)

            v = verdict.get("verdict")
            if verdict.get("applied_correction"):
                n_fix += 1
            elif v == "agree":
                n_ok += 1
            elif v in ("disagree", "uncertain"):
                n_bad += 1

            verdicts.append({
                "index": i,
                "filename": self._filename(result),
                **{k: verdict.get(k) for k in (
                    "verdict", "confidence", "water", "persons",
                    "counts", "corrected_water_cm",
                    "applied_correction", "notes")},
            })

        state.vlm_verifications = verdicts
        total = len(verdicts)
        log.info(
            f"  [VLMVerify] {total} ảnh | agree={n_ok} "
            f"flag/sửa={n_fix} mismatch/unsure={n_bad}"
        )

        if self.unload_after:
            self._verifier.unload()

        return state

    def health_check(self) -> dict:
        vcfg = self.cfg.get("vlm_verify", {}) or {}
        if not vcfg.get("enabled", True):
            return {"ok": True, "message": "vlm_verify disabled trong config"}
        model_name = (self.cfg.get("models", {}) or {}).get("vlm")
        return {"ok": True,
                "message": f"VLM verifier sẵn sàng (model={model_name or 'default'})"}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_verifier(self):
        """
        Lazy tạo VLMVerifier; fail → None (stage vẫn chạy, verdict=skipped).
        Chưa set models.vlm trong config → coi như chưa cấu hình, không tải model.
        """
        try:
            from depth_analysis.vlm_verifier import get_vlm_verifier
            verifier = get_vlm_verifier(self.cfg)
            if not verifier.model_name:
                log.info(
                    "  [VLMVerify] Chưa cấu hình models.vlm trong config.yaml "
                    "— bỏ qua xác minh (verdict=skipped). "
                    "Bật bằng cách đặt:  models: vlm: \"Qwen/Qwen2.5-VL-7B-Instruct\""
                )
                return None
            return verifier
        except Exception as exc:
            log.warning(f"  [VLMVerify] Không khởi tạo được VLMVerifier: {exc}")
            return None

    @staticmethod
    def _image_for(state: "PipelineState", result: Any, idx: int):
        """Ưu tiên ảnh gốc cùng folder input, fallback path trong result."""
        from depth_analysis.vlm_verifier import _get
        orig = _get(result, "original_path")
        if orig:
            from pathlib import Path
            p = Path(str(orig))
            if p.exists():
                return p
        # input_images song song với depth_results theo thứ tự
        try:
            return state.input_images[idx]
        except (IndexError, TypeError):
            return None

    @staticmethod
    def _attach(result: Any, verdict: dict) -> None:
        from depth_analysis.vlm_verifier import _set
        _set(result, "vlm_verification", verdict)

    @staticmethod
    def _filename(result: Any) -> str:
        from depth_analysis.vlm_verifier import _get
        import os
        orig = _get(result, "original_path") or ""
        return os.path.basename(orig) if orig else _get(result, "filename", "?")
