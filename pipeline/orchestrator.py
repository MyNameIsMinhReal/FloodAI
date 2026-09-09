# -*- coding: utf-8 -*-
"""
FloodPipeline Orchestrator
===========================
Pipeline đã cắt bỏ Crawl và Filter. Flow mới:

    Input (folder ảnh) → Analyze → Depth → Raincoat → VLM Verify → Postprocess → Store → Learn

Cách dùng:
    pipeline = FloodPipeline(cfg)
    state    = pipeline.run(images)           # truyền thẳng list[Path]
    # hoặc
    state    = pipeline.run_from_dir("/path") # tự quét folder
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.stages.analyze_stage import AnalyzeStage
from pipeline.stages.depth_stage import DepthStage
from pipeline.stages.raincoat_stage import RaincoatStage
from pipeline.stages.vlm_verify_stage import VLMVerifyStage
from pipeline.stages.postprocess_stage import PostprocessStage
from pipeline.stages.store_stage import StoreStage
from pipeline.stages.learn_stage import LearnStage
from pipeline.confidence import ConfidenceScorer
from utils.constants import (
    OUTPUT_FOLDER_ORIGINAL,
    OUTPUT_FOLDER_OVERLAY, OUTPUT_FOLDER_DEPTHMAP,
    PIPELINE_SUMMARY_JSON,
)

log = logging.getLogger("pipeline.orchestrator")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def validate_images(images: List[Path]) -> List[Path]:
    """
    Lọc danh sách ảnh đầu vào: bỏ file không tồn tại, sai định dạng, hoặc rỗng.
    Trả về danh sách ảnh hợp lệ.
    """
    valid = []
    for p in images:
        if not p.exists():
            log.warning(f"  [Validate] Bỏ qua (không tồn tại): {p}")
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            log.warning(f"  [Validate] Bỏ qua (định dạng không hợp lệ): {p}")
            continue
        size = p.stat().st_size
        if size == 0:
            log.warning(f"  [Validate] Bỏ qua (file rỗng): {p}")
            continue
        if size > MAX_IMAGE_SIZE_BYTES:
            log.warning(f"  [Validate] Bỏ qua (file quá lớn {size // 1024 // 1024}MB): {p}")
            continue
        valid.append(p)
    return valid


# ─── Pipeline State ────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    """Trạng thái chạy qua từng stage."""
    run_id:       str = ""
    input_dir:    Optional[Path] = None

    input_images:    List[Path] = field(default_factory=list)
    analyzed_images: List[Path] = field(default_factory=list)
    depth_results:   List[Any]  = field(default_factory=list)
    location_map:    Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    vlm_verifications: List[Dict[str, Any]] = field(default_factory=list)

    timings:         Dict[str, float] = field(default_factory=dict)
    errors:          List[str]        = field(default_factory=list)
    drive_folder_id: Optional[str]   = None
    output_dir:      Optional[Path]  = None
    version_meta:    Dict[str, Any]  = field(default_factory=dict)  # [v4] pipeline version traceability

    # Optional callback(stage, current, total, message) cho progress realtime
    progress_callback: Any = field(default=None, repr=False)

    def log_stage(self, stage: str, duration: float, count: int):
        self.timings[stage] = duration
        log.info(f"  ✓ [{stage}] {count} items — {duration:.1f}s")

    def report_progress(self, stage: str, current: int, total: int, message: str = ""):
        """Gọi progress_callback nếu có — dùng bởi các stage để báo tiến độ."""
        if callable(self.progress_callback):
            try:
                self.progress_callback(stage, current, total, message)
            except Exception:
                pass

    def to_dict(self) -> dict:
        return {
            "run_id":          self.run_id,
            "input_dir":       str(self.input_dir or ""),
            "input_images":    [str(p) for p in self.input_images],
            "depth_data":      self.depth_results,
            "location_map":    self.location_map,
            "drive_folder_id": self.drive_folder_id,
            "timings":         self.timings,
            "errors":          self.errors,
        }


# ─── Pipeline Orchestrator ─────────────────────────────────────────────────────

class FloodPipeline:
    """
    Orchestrator chính — flow không có Crawl / Filter:

        input images → analyze → depth → raincoat → vlm_verify → postprocess → store → learn

    Ví dụ:
        pipeline = FloodPipeline(cfg)
        state = pipeline.run([Path("a.jpg"), Path("b.jpg")])
        state = pipeline.run_from_dir("/data/floods")
    """

    def __init__(self, cfg: dict, progress_callback=None):
        self.cfg = cfg
        self.progress_callback = progress_callback
        self.confidence_scorer = ConfidenceScorer()

        self.analyzer  = AnalyzeStage(cfg)
        self.depth     = DepthStage(cfg)
        self.raincoat  = RaincoatStage(cfg)
        self.vlm_verify = VLMVerifyStage(cfg)
        self.postproc  = PostprocessStage(cfg)
        self.storage   = StoreStage(cfg)
        self.learner   = LearnStage(cfg)

        log.info("FloodPipeline initialized — flow: input→analyze→depth→raincoat→vlm_verify→postprocess→store→learn")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, images: List[Path]) -> "PipelineState":
        """
        Chạy pipeline với danh sách ảnh đầu vào.

        Args:
            images: list[Path] — các file ảnh đã tồn tại trên disk

        Returns:
            PipelineState đầy đủ kết quả
        """
        images = validate_images(list(images))
        if not images:
            raise ValueError("Không có ảnh hợp lệ. Kiểm tra định dạng, kích thước và đường dẫn file.")

        run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.cfg.get("output_dir", "output")) / run_id
        self._prepare_dirs(base_dir)

        state = PipelineState(
            run_id=run_id,
            input_images=list(images),
            output_dir=base_dir,
            progress_callback=self.progress_callback,
        )

        # [v4 #5] Build version metadata (depth_model, yolo, config_hash)
        try:
            from utils.pipeline_version import build_version_meta
            state.version_meta = build_version_meta(self.cfg)
        except Exception:
            state.version_meta = {}

        log.info(f"\n{'='*60}")
        log.info(f"  FLOOD PIPELINE — run_id={run_id}")
        log.info(f"  Input: {len(images)} ảnh")
        log.info(f"{'='*60}\n")

        t_start = time.time()

        # ── Stage 1: Analyze ──────────────────────────────────────────
        state = self._run_stage("analyze", self._stage_analyze, state)

        # ── Stage 2: Depth ────────────────────────────────────────────
        if not self.cfg.get("skip_depth", False):
            state = self._run_stage("depth", self._stage_depth, state)
        else:
            log.info("  [SKIP] Depth stage bị tắt trong config.")

        # ── Stage 3: Raincoat ─────────────────────────────────────────
        if not self.cfg.get("skip_raincoat", False):
            state = self._run_stage("raincoat", self._stage_raincoat, state)
        else:
            log.info("  [SKIP] Raincoat stage bị tắt trong config.")

        # ── Stage 3.5: VLM Verify (kiểm định chéo trước khi xuất) ────
        if not self.cfg.get("skip_vlm_verify", False):
            state = self._run_stage("vlm_verify", self._stage_vlm_verify, state)
        else:
            log.info("  [SKIP] VLM verify stage bị tắt trong config.")

        # ── Stage 4: Postprocess ──────────────────────────────────────
        state = self._run_stage("postprocess", self._stage_postprocess, state)

        # ── Stage 5: Store ────────────────────────────────────────────
        state = self._run_stage("store", self._stage_store, state)

        # ── Stage 6: Learn ────────────────────────────────────────────
        if not self.cfg.get("skip_learning", False):
            state = self._run_stage("learn", self._stage_learn, state)

        total = time.time() - t_start
        log.info(f"\n{'='*60}")
        log.info(f"  PIPELINE HOÀN THÀNH — {total:.1f}s tổng cộng")
        self._print_timing_report(state)
        log.info(f"{'='*60}\n")

        # ── [v4 #5] Pipeline version manifest ──────────────────────────────
        try:
            from utils.pipeline_version import attach_version_batch, save_run_manifest
            attach_version_batch(state.depth_results, state.version_meta)
            save_run_manifest(state.run_id, self.cfg, state)
        except Exception as exc:
            log.debug(f"  [Version] Manifest skip: {exc}")

        return state

    def run_from_dir(self, folder: "str | Path") -> "PipelineState":
        """
        Quét toàn bộ ảnh trong folder rồi chạy pipeline.

        Args:
            folder: đường dẫn thư mục (hỗ trợ nested)

        Returns:
            PipelineState
        """
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder không tồn tại: {folder}")

        images = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise ValueError(f"Không tìm thấy ảnh nào trong: {folder}")

        log.info(f"  Quét folder: {folder} → {len(images)} ảnh")
        state = self.run(images)
        state.input_dir = folder
        return state

    # ── Stage runners ──────────────────────────────────────────────────────────

    def _run_stage(self, name: str, fn, state: "PipelineState") -> "PipelineState":
        log.info(f"\n── Stage: {name.upper()} ──────────────────────────────")
        t0 = time.time()
        try:
            state = fn(state)
        except Exception as exc:
            log.error(f"  [ERROR] Stage '{name}' thất bại sau {time.time()-t0:.1f}s: {exc}", exc_info=True)
            state.errors.append(f"{name}: {exc}")
        else:
            state.timings[name] = time.time() - t0
        return state

    def _stage_analyze(self, state: "PipelineState") -> "PipelineState":
        if self.cfg.get("detect_location", True):
            location_map = self.analyzer.run(state.input_images) or {}
            state.location_map = location_map
            # [BUG FIX v2] Add null check before calling .values()
            if location_map:
                located = sum(1 for v in location_map.values() if v.get("method") != "none")
                log.info(f"  Location: {located}/{len(state.input_images)} ảnh được định vị")
            else:
                log.warning("  Location: analyzer returned None/empty")

        state.analyzed_images = state.input_images
        state.log_stage("analyze", state.timings.get("analyze", 0), len(state.analyzed_images))
        return state

    def _stage_depth(self, state: "PipelineState") -> "PipelineState":
        if state.output_dir is None:
            raise ValueError("output_dir is not set")
        overlay_dir  = state.output_dir / OUTPUT_FOLDER_OVERLAY
        depthmap_dir = state.output_dir / OUTPUT_FOLDER_DEPTHMAP
        overlay_dir.mkdir(exist_ok=True)
        depthmap_dir.mkdir(exist_ok=True)

        images_to_process, cached_results = state.analyzed_images, []

        if self.cfg.get("use_output_cache", True):
            images_to_process, cached_results = self._filter_cached_depth(
                state.analyzed_images, depthmap_dir
            )
            if cached_results:
                log.info(f"  [Cache] Reused {len(cached_results)} depth results")

        depth_results = self.depth.compute(
            images=images_to_process,
            overlay_dir=overlay_dir,
            depthmap_dir=depthmap_dir,
        )

        state.depth_results = cached_results + (depth_results or [])
        state.log_stage("depth", state.timings.get("depth", 0), len(state.depth_results))
        return state

    def _stage_raincoat(self, state: "PipelineState") -> "PipelineState":
        if not state.depth_results:
            return state

        overlay_dir = state.output_dir / OUTPUT_FOLDER_OVERLAY if state.output_dir else None
        t0 = time.time()

        state.depth_results = self.raincoat.compute(
            depth_results=state.depth_results,
            overlay_dir=overlay_dir,
        )

        rc_count = sum(
            1 for r in state.depth_results
            if (r.get("has_raincoat") if isinstance(r, dict) else getattr(r, "has_raincoat", False))
        )
        state.log_stage("raincoat", time.time() - t0, rc_count)
        log.info(f"  [Raincoat] {rc_count}/{len(state.depth_results)} ảnh có áo mưa")
        return state

    def _stage_vlm_verify(self, state: "PipelineState") -> "PipelineState":
        if not state.depth_results:
            return state
        return self.vlm_verify.run(state)

    def _stage_postprocess(self, state: "PipelineState") -> "PipelineState":
        if not state.depth_results:
            return state

        for result in state.depth_results:
            confidence = self.confidence_scorer.compute(result)
            img_key = getattr(result, "original_path", str(result))
            state.confidence_scores[img_key] = confidence

        # ── [v4 Gap B] Temporal aggregation: gộp nhiều ảnh cùng vị trí ────
        agg_cfg = self.cfg.get("temporal_agg", {})
        if agg_cfg.get("enable", False) and state.location_map:
            try:
                from utils.temporal_aggregator import aggregate_by_location
                n_before = len(state.depth_results)
                state.depth_results = aggregate_by_location(
                    state.depth_results, state.location_map,
                    min_group_size=2,
                )
                log.info(
                    f"  [Agg] Temporal aggregation: {n_before} ảnh → "
                    f"{len(set(getattr(r, 'original_path','') for r in state.depth_results))} vị trí"
                )
            except Exception as exc:
                log.warning(f"  [Agg] Bỏ qua: {exc}")

        if self.cfg.get("predict_routes", False):
            self.postproc.predict_routes(state)

        self.postproc.check_alerts(state)

        # ── [v4 Gap #4] FloodAlerter (Telegram/email/Discord) ──────────────
        try:
            from utils.alerting import FloodAlerter
            alerter = FloodAlerter(self.cfg)
            sent = alerter.check_and_alert(state)
            if sent:
                log.info(f"  [Alert] Sent via: {[ch for ch, ok in sent if ok]}")
        except Exception as exc:
            log.debug(f"  [Alert] FloodAlerter skip: {exc}")

        self.postproc.check_disagreements(state)
        state.log_stage("postprocess", state.timings.get("postprocess", 0), len(state.depth_results))
        return state

    def _stage_store(self, state: "PipelineState") -> "PipelineState":
        self.storage.save_reports(state)

        if not self.cfg.get("skip_drive", True):
            state.drive_folder_id = self.storage.upload_drive(state)

        state.log_stage("store", state.timings.get("store", 0), 1)
        return state

    def _stage_learn(self, state: "PipelineState") -> "PipelineState":
        if not state.depth_results:
            return state

        self.learner.update(
            depth_results=state.depth_results,
            cfg=self.cfg,
            image_paths=state.analyzed_images,
        )
        state.log_stage("learn", state.timings.get("learn", 0), len(state.depth_results))
        return state

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _prepare_dirs(self, base_dir: Path):
        for folder in [OUTPUT_FOLDER_ORIGINAL, OUTPUT_FOLDER_OVERLAY, OUTPUT_FOLDER_DEPTHMAP]:
            (base_dir / folder).mkdir(parents=True, exist_ok=True)

    def _filter_cached_depth(self, images: List[Path], depthmap_dir: Path):
        import json as _json
        to_process, cached = [], []
        for img_path in images:
            meta  = depthmap_dir / f"{img_path.stem}_meta.json"
            depth = depthmap_dir / f"{img_path.stem}_depth.png"
            if meta.exists() and depth.exists():
                try:
                    cached.append(_json.loads(meta.read_text(encoding="utf-8")))
                    continue
                except Exception:
                    pass
            to_process.append(img_path)
        return to_process, cached

    def _print_timing_report(self, state: "PipelineState"):
        log.info("\n  Timing breakdown:")
        total = sum(state.timings.values()) or 1
        for stage, t in state.timings.items():
            pct = t / total * 100
            bar = "█" * int(pct / 5)
            log.info(f"    {stage:12s} {t:6.1f}s  {bar} {pct:.0f}%")
        log.info(f"    {'TOTAL':12s} {total:6.1f}s")

        if state.errors:
            log.warning(f"\n  ⚠ Errors ({len(state.errors)}):")
            for err in state.errors:
                log.warning(f"    - {err}")
