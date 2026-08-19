# -*- coding: utf-8 -*-
"""
pipeline/orchestrator_async.py  —  Async Event-Driven Orchestrator
====================================================================
Nâng cấp từ linear orchestrator → event-driven / DAG pipeline.

Vấn đề của linear pipeline:
  - Tất cả stages chạy tuần tự
  - Stage A xong hoàn toàn mới chạy Stage B
  - Không tận dụng được I/O overlap

Pipeline mới (DAG):
                    ┌→ depth_stage ─────────────┐
  analyze_stage ───┤                            ├→ postprocess → store → learn
                    └→ raincoat_stage ──────────┘

  Depth + Raincoat chạy SONG SONG (nếu cấu hình cho phép)

Thêm:
  - Confidence-aware routing: low confidence → ensemble / human review
  - Event bus: stage emit event → other stages react
  - Batch processing: xử lý nhiều ảnh theo batch thay vì từng cái một

Sử dụng:
    # Drop-in replacement cho FloodPipeline:
    pipeline = AsyncFloodPipeline(cfg)
    state = asyncio.run(pipeline.run_async(images))

    # Hoặc dùng sync wrapper:
    state = pipeline.run(images)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pipeline.orchestrator import PipelineState  # reuse existing state
from pipeline.confidence import ConfidenceScorer

log = logging.getLogger("pipeline.orchestrator_async")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ── Event Bus ──────────────────────────────────────────────────────────────────

class EventBus:
    """
    Simple in-process event bus cho pipeline stages.

    Stages có thể emit events và subscribe để react.

    Ví dụ:
        bus = EventBus()
        bus.subscribe("low_confidence", send_to_review_queue)
        bus.emit("low_confidence", result)
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}

    def subscribe(self, event: str, handler: Callable) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, data: Any = None) -> None:
        for handler in self._handlers.get(event, []):
            try:
                handler(data)
            except Exception as exc:
                log.warning(f"  [EventBus] Handler error for '{event}': {exc}")

    def emit_async(self, event: str, data: Any = None) -> asyncio.Task:
        """Emit event async."""
        async def _emit():
            self.emit(event, data)
        return asyncio.create_task(_emit())


# ── Routing decision ───────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """Quyết định routing sau khi tính confidence."""
    result: Any
    confidence: float
    route: str          # "normal" | "ensemble" | "human_review" | "skip"
    reason: str = ""


class ConfidenceRouter:
    """
    Confidence-aware routing engine.

    Thay vì chỉ compute confidence và lưu lại,
    Router còn quyết định HÀNH ĐỘNG tiếp theo dựa trên confidence:

      HIGH (>= 0.70) → normal flow
      MEDIUM (0.40-0.70) → trigger multi-model ensemble
      LOW (< 0.40) → send to human review queue
    """

    def __init__(
        self,
        high_threshold: float = 0.70,
        low_threshold:  float = 0.40,
    ):
        self.scorer = ConfidenceScorer(
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
        self.high = high_threshold
        self.low  = low_threshold

    def route(self, result: Any) -> RoutingDecision:
        """Tính confidence và quyết định route."""
        conf = self.scorer.compute(result)
        label = self.scorer.label(conf)

        if conf >= self.high:
            return RoutingDecision(
                result=result, confidence=conf,
                route="normal",
                reason=f"Confidence cao ({conf:.2f}) → normal flow",
            )
        elif conf >= self.low:
            return RoutingDecision(
                result=result, confidence=conf,
                route="ensemble",
                reason=f"Confidence trung bình ({conf:.2f}) → trigger ensemble",
            )
        else:
            return RoutingDecision(
                result=result, confidence=conf,
                route="human_review",
                reason=f"Confidence thấp ({conf:.2f}) → yêu cầu human review",
            )

    def route_batch(self, results: list) -> Dict[str, List]:
        """Route cả batch, nhóm theo route type."""
        groups: Dict[str, List] = {"normal": [], "ensemble": [], "human_review": []}
        for r in results:
            d = self.route(r)
            groups[d.route].append(d)
            if d.route != "normal":
                log.info(f"  [Router] {d.reason}")
        return groups


# ── Async Pipeline ─────────────────────────────────────────────────────────────

class AsyncFloodPipeline:
    """
    Async event-driven flood pipeline.

    Nâng cấp so với FloodPipeline (linear):
      1. Depth + Raincoat có thể chạy song song (DAG)
      2. Confidence-aware routing: low conf → ensemble/review
      3. Event bus: stages giao tiếp qua events
      4. Batch processing: process theo chunks thay vì từng ảnh

    Backward-compatible: có sync wrapper `.run()` dùng được như cũ.

    Ví dụ:
        pipeline = AsyncFloodPipeline(cfg)
        state = pipeline.run(images)                     # sync
        state = await pipeline.run_async(images)         # async
    """

    def __init__(self, cfg: dict):
        self.cfg  = cfg
        self.bus  = EventBus()
        self.router = ConfidenceRouter(
            high_threshold=cfg.get("confidence_high", 0.70),
            low_threshold=cfg.get("confidence_low",  0.40),
        )

        # Lazy-import stages để tránh circular import
        self._stages_initialized = False
        self._setup_event_handlers()

        log.info(
            "AsyncFloodPipeline initialized — "
            "flow: analyze → [depth ‖ raincoat] → route → postprocess → store → learn"
        )

    def _init_stages(self):
        if self._stages_initialized:
            return
        from pipeline.stages.analyze_stage    import AnalyzeStage
        from pipeline.stages.depth_stage      import DepthStage
        from pipeline.stages.raincoat_stage   import RaincoatStage
        from pipeline.stages.postprocess_stage import PostprocessStage
        from pipeline.stages.store_stage      import StoreStage
        from pipeline.stages.learn_stage      import LearnStage

        self.analyzer  = AnalyzeStage(self.cfg)
        self.depth     = DepthStage(self.cfg)
        self.raincoat  = RaincoatStage(self.cfg)
        self.postproc  = PostprocessStage(self.cfg)
        self.storage   = StoreStage(self.cfg)
        self.learner   = LearnStage(self.cfg)
        self._stages_initialized = True

    def _setup_event_handlers(self):
        """Đăng ký event handlers."""
        self.bus.subscribe("low_confidence", self._handle_low_confidence)
        self.bus.subscribe("ensemble_needed", self._handle_ensemble)
        self.bus.subscribe("stage_error", self._handle_stage_error)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, images: List[Path]) -> PipelineState:
        """Sync wrapper — backward-compatible với FloodPipeline.run()."""
        # Avoid the optional nest_asyncio dependency. In an already-running
        # event loop, use a worker thread with its own event loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(images))

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, self.run_async(images))
            return future.result()

    def run_from_dir(self, folder: "str | Path") -> PipelineState:
        """Quét folder và chạy pipeline."""
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

    async def run_async(self, images: List[Path]) -> PipelineState:
        """Async entry point — chạy DAG pipeline."""
        self._init_stages()

        if not images:
            raise ValueError("Không có ảnh đầu vào")

        run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(self.cfg.get("output_dir", "output")) / run_id
        self._prepare_dirs(base_dir)

        state = PipelineState(
            run_id=run_id,
            input_images=list(images),
            output_dir=base_dir,
        )

        log.info(f"\n{'='*60}")
        log.info(f"  ASYNC FLOOD PIPELINE — run_id={run_id}")
        log.info(f"  Input: {len(images)} ảnh")
        log.info(f"{'='*60}\n")

        t_start = time.time()

        # Stage 1: Analyze (must be first)
        state = await self._async_stage("analyze", self._stage_analyze, state)

        # Stages 2+3: Depth & Raincoat có thể chạy song song
        if self.cfg.get("parallel_depth_raincoat", True):
            state = await self._run_parallel_stages(state)
        else:
            if not self.cfg.get("skip_depth", False):
                state = await self._async_stage("depth", self._stage_depth, state)
            if not self.cfg.get("skip_raincoat", False):
                state = await self._async_stage("raincoat", self._stage_raincoat, state)

        # Stage 4: Confidence routing
        state = await self._async_stage("confidence_routing", self._stage_routing, state)

        # Stage 5: Postprocess
        state = await self._async_stage("postprocess", self._stage_postprocess, state)

        # Stage 6: Store
        state = await self._async_stage("store", self._stage_store, state)

        # Stage 7: Learn
        if not self.cfg.get("skip_learning", False):
            state = await self._async_stage("learn", self._stage_learn, state)

        total = time.time() - t_start
        log.info(f"\n{'='*60}")
        log.info(f"  PIPELINE HOÀN THÀNH — {total:.1f}s tổng cộng")
        self._print_timing_report(state)
        log.info(f"{'='*60}\n")

        return state

    # ── DAG: parallel stages ───────────────────────────────────────────────────

    async def _run_parallel_stages(self, state: PipelineState) -> PipelineState:
        """
        Chạy Depth và Raincoat song song.
        Note: cả hai đều cần input từ Analyze nhưng độc lập với nhau.
        """
        log.info("  ⚡ Parallel mode: Depth ‖ Raincoat")
        t0 = time.time()

        # Clone state nhẹ cho từng branch
        depth_state    = _shallow_copy_state(state)
        raincoat_state = _shallow_copy_state(state)

        skip_depth    = self.cfg.get("skip_depth", False)
        skip_raincoat = self.cfg.get("skip_raincoat", False)

        tasks = []
        if not skip_depth:
            tasks.append(self._async_stage("depth", self._stage_depth, depth_state))
        if not skip_raincoat:
            tasks.append(self._async_stage("raincoat", self._stage_raincoat, raincoat_state))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge results back to main state
            for result in results:
                if isinstance(result, Exception):
                    log.error(f"  [Parallel] Stage error: {result}")
                    state.errors.append(str(result))
                elif isinstance(result, PipelineState):
                    _merge_state(state, result)
        else:
            log.info("  [Parallel] Cả Depth và Raincoat đều bị skip")

        log.info(f"  ⚡ Parallel stages hoàn thành: {time.time()-t0:.1f}s")
        return state

    # ── Stage implementations ──────────────────────────────────────────────────

    async def _stage_analyze(self, state: PipelineState) -> PipelineState:
        loop = asyncio.get_event_loop()
        if self.cfg.get("detect_location", True):
            location_map = await loop.run_in_executor(
                None, self.analyzer.run, state.input_images
            )
            state.location_map = location_map
        state.analyzed_images = state.input_images
        return state

    async def _stage_depth(self, state: PipelineState) -> PipelineState:
        from utils.constants import OUTPUT_FOLDER_OVERLAY, OUTPUT_FOLDER_DEPTHMAP
        if state.output_dir is None:
            raise ValueError("output_dir is not set")
        overlay_dir  = state.output_dir / OUTPUT_FOLDER_OVERLAY
        depthmap_dir = state.output_dir / OUTPUT_FOLDER_DEPTHMAP
        overlay_dir.mkdir(exist_ok=True)
        depthmap_dir.mkdir(exist_ok=True)

        loop = asyncio.get_event_loop()
        depth_results = await loop.run_in_executor(
            None,
            lambda: self.depth.compute(
                images=state.analyzed_images,
                overlay_dir=overlay_dir,
                depthmap_dir=depthmap_dir,
            )
        )
        state.depth_results = depth_results or []
        return state

    async def _stage_raincoat(self, state: PipelineState) -> PipelineState:
        if not state.depth_results:
            return state
        from utils.constants import OUTPUT_FOLDER_OVERLAY
        overlay_dir = state.output_dir / OUTPUT_FOLDER_OVERLAY if state.output_dir else None

        loop = asyncio.get_event_loop()
        depth_results = await loop.run_in_executor(
            None,
            lambda: self.raincoat.compute(
                depth_results=state.depth_results,
                overlay_dir=overlay_dir,
            )
        )
        state.depth_results = depth_results
        return state

    async def _stage_routing(self, state: PipelineState) -> PipelineState:
        """
        Confidence-aware routing.
        - HIGH → normal
        - MEDIUM → emit "ensemble_needed"
        - LOW → emit "low_confidence" → human review queue
        """
        if not state.depth_results:
            return state

        groups = self.router.route_batch(state.depth_results)

        n_normal  = len(groups["normal"])
        n_ensemble = len(groups["ensemble"])
        n_review  = len(groups["human_review"])

        log.info(
            f"  [Router] {n_normal} normal | "
            f"{n_ensemble} ensemble | {n_review} human_review"
        )

        # Emit events
        for d in groups["ensemble"]:
            self.bus.emit("ensemble_needed", d)
        for d in groups["human_review"]:
            self.bus.emit("low_confidence", d)

        # Lưu routing info vào state
        for group_name, decisions in groups.items():
            for d in decisions:
                img_key = getattr(d.result, "original_path", str(d.result))
                state.confidence_scores[img_key] = d.confidence

        return state

    async def _stage_postprocess(self, state: PipelineState) -> PipelineState:
        if self.cfg.get("predict_routes", False):
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.postproc.predict_routes, state)
        self.postproc.check_alerts(state)
        return state

    async def _stage_store(self, state: PipelineState) -> PipelineState:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.storage.save_reports, state)
        if not self.cfg.get("skip_drive", True):
            state.drive_folder_id = await loop.run_in_executor(
                None, self.storage.upload_drive, state
            )
        return state

    async def _stage_learn(self, state: PipelineState) -> PipelineState:
        if not state.depth_results:
            return state
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.learner.update(
                depth_results=state.depth_results,
                cfg=self.cfg,
                image_paths=state.analyzed_images,
            )
        )
        return state

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _handle_low_confidence(self, decision: RoutingDecision) -> None:
        """Gửi kết quả confidence thấp vào human review queue."""
        log.warning(
            f"  [Review] Low confidence ({decision.confidence:.2f}) → "
            f"gửi vào review queue: {decision.reason}"
        )
        try:
            # [BUG FIX v2] Make learner class configurable instead of hardcoded
            learner_class_name = self.cfg.get("learner_class", "ActiveLearnerV2")
            from learning import active_learner
            from learning.active_learner import ReviewCase
            learner_class = getattr(active_learner, learner_class_name)
            learner = learner_class()
            r = decision.result
            case = ReviewCase(
                image_path      = str(getattr(r, "image_path", "")),
                predicted_depth = getattr(r, "flood_depth_cm", None),
                predicted_level = getattr(r, "flood_level", None),
                confidence      = decision.confidence,
                review_reason   = decision.reason,
            )
            learner.add_to_queue(case)
        except Exception as exc:
            log.debug(f"  [Review] add_to_queue failed: {exc}")

    def _handle_ensemble(self, decision: RoutingDecision) -> None:
        """Trigger multi-model ensemble cho kết quả confidence trung bình."""
        log.info(
            f"  [Ensemble] Medium confidence ({decision.confidence:.2f}) → "
            f"trigger ensemble: {decision.reason}"
        )
        # Placeholder: trong production sẽ dispatch sang ensemble worker
        # Ví dụ: Celery task, hay asyncio task pool

    def _handle_stage_error(self, error: dict) -> None:
        """Xử lý lỗi từ stage."""
        log.error(f"  [Error] Stage '{error.get('stage')}': {error.get('message')}")

    # ── Async stage runner ─────────────────────────────────────────────────────

    async def _async_stage(
        self,
        name: str,
        fn: Callable,
        state: PipelineState,
    ) -> PipelineState:
        """Wrapper async cho stage: timing + error handling."""
        log.info(f"\n── Stage: {name.upper()} ──────────────────────────────")
        t0 = time.time()
        try:
            state = await fn(state)
            elapsed = time.time() - t0
            state.timings[name] = elapsed
            log.info(f"  ✓ [{name}] {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.time() - t0
            log.error(
                f"  [ERROR] Stage '{name}' thất bại sau {elapsed:.1f}s: {exc}",
                exc_info=True,
            )
            state.errors.append(f"{name}: {exc}")
            self.bus.emit("stage_error", {"stage": name, "message": str(exc)})
        return state

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _prepare_dirs(self, base_dir: Path):
        from utils.constants import (
            OUTPUT_FOLDER_ORIGINAL, OUTPUT_FOLDER_OVERLAY, OUTPUT_FOLDER_DEPTHMAP,
        )
        for folder in [OUTPUT_FOLDER_ORIGINAL, OUTPUT_FOLDER_OVERLAY, OUTPUT_FOLDER_DEPTHMAP]:
            (base_dir / folder).mkdir(parents=True, exist_ok=True)

    def _print_timing_report(self, state: PipelineState):
        log.info("\n  Timing breakdown:")
        total = sum(state.timings.values()) or 1
        for stage, t in state.timings.items():
            pct = t / total * 100
            bar = "█" * int(pct / 5)
            log.info(f"    {stage:20s} {t:6.1f}s  {bar} {pct:.0f}%")
        log.info(f"    {'TOTAL':20s} {total:6.1f}s")
        if state.errors:
            log.warning(f"\n  ⚠ Errors ({len(state.errors)}):")
            for err in state.errors:
                log.warning(f"    - {err}")


# ── State helpers ──────────────────────────────────────────────────────────────

def _shallow_copy_state(state: PipelineState) -> PipelineState:
    """Tạo shallow copy để chạy stages song song."""
    from dataclasses import replace
    return replace(
        state,
        depth_results=list(state.depth_results),
        errors=list(state.errors),
        timings=dict(state.timings),
    )


def _merge_state(target: PipelineState, source: PipelineState) -> None:
    """Merge source state vào target sau khi chạy song song."""
    if source.depth_results:
        target.depth_results = source.depth_results
    target.timings.update(source.timings)
    target.errors.extend(source.errors)
