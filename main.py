# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import time
from pathlib import Path

from utils.structured_logger import setup_logging
setup_logging(level=logging.INFO)

log = logging.getLogger("main")


def load_config(config_path: str = "config.yaml") -> dict:
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning(f"  Config '{config_path}' không tìm thấy — dùng defaults")
        return {}
    except ImportError:
        log.warning("  PyYAML chưa cài — dùng empty config")
        return {}


def run_pipeline(args, cfg: dict):
    from datetime import datetime
    input_path = Path(args.input)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Structured Logger ─────────────────────────────────────────────────────
    from utils.structured_logger import PipelineLogger
    plog = PipelineLogger(
        run_id=run_id,
        log_dir=cfg.get("log_dir", "output/logs"),
        component="main",
    )
    plog.emit("pipeline_start", input=str(input_path))

    # ── Model Loader: preload ─────────────────────────────────────────────────
    from core.model_loader import ModelLoader
    loader = ModelLoader.instance()
    loader.register_all_from_config(cfg)
    preload = cfg.get("preload_models", [])
    if preload:
        log.info(f"  Preloading: {preload}")
        loader.preload(preload)
        loader.log_status()

    # ── Intermediate cache ────────────────────────────────────────────────────
    from utils.memory_manager import IntermediateCache, log_memory
    cache = IntermediateCache.instance(max_entries=cfg.get("cache_max_entries", 200))
    log_memory("startup")

    # ── Model Versioning ──────────────────────────────────────────────────────
    from learning.model_versioning import ModelRegistry
    registry = ModelRegistry(base_dir=cfg.get("model_versions_dir", "learning/models"))
    current = registry.current_version()
    if current:
        log.info(f"  Model version: {current.version_id} ({current.created_at[:10]})")

    # ── Choose pipeline mode ──────────────────────────────────────────────────
    if getattr(args, "async_mode", False) or cfg.get("use_async_pipeline", False):
        log.info("  Mode: AsyncFloodPipeline (event-driven DAG)")
        from pipeline.orchestrator_async import AsyncFloodPipeline
        pipeline = AsyncFloodPipeline(cfg)
    else:
        log.info("  Mode: FloodPipeline (linear)")
        from pipeline.orchestrator import FloodPipeline
        pipeline = FloodPipeline(cfg)

    # ── Run ───────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        state = (pipeline.run_from_dir(input_path)
                 if input_path.is_dir()
                 else pipeline.run([input_path]))
    except Exception as exc:
        plog.error("pipeline", str(exc), exc=exc)
        log.error(f"  Pipeline lỗi: {exc}", exc_info=True)
        return 1

    total_time = time.time() - t0

    # ── Hard Example Mining ───────────────────────────────────────────────────
    if not cfg.get("skip_hard_mining", False):
        from learning.hard_example_mining import HardExampleMiner
        miner = HardExampleMiner(db_path=cfg.get("hard_examples_db", "learning/hard_examples.db"))
        n_hard = 0
        for result in state.depth_results:
            conf = state.confidence_scores.get(
                getattr(result, "original_path", str(result)), 0.5
            )
            score = miner.score_hardness(result, confidence=conf)
            if score.is_hard:
                miner.add_to_hard_queue(result, score, confidence=conf)
                n_hard += 1
        if n_hard:
            log.info(f"  [HardMining] {n_hard} hard examples queued")
        miner.log_stats()

    # ── Save model version if needed ──────────────────────────────────────────
    if not cfg.get("skip_versioning", False) and state.depth_results:
        try:
            _maybe_save_model_version(registry, state, cfg)
        except Exception as exc:
            log.warning(f"  [Versioning] Bỏ qua: {exc}")

    # ── Cleanup & summary ─────────────────────────────────────────────────────
    log_memory("after_pipeline")
    cache.log_stats("final")
    loader.unload_heavy()

    plog.emit("pipeline_end", total_time_s=round(total_time, 2),
              n_processed=len(state.depth_results), n_errors=len(state.errors))
    plog.log_summary()
    plog.close()

    log.info(f"\n  ✅ Hoàn thành: {len(state.depth_results)} ảnh — {total_time:.1f}s")
    if state.output_dir:
        log.info(f"  Output: {state.output_dir}")
    return 0


def show_status(cfg: dict):
    from learning.model_versioning import ModelRegistry
    from utils.memory_manager import log_memory
    from core.model_loader import ModelLoader
    ModelRegistry(base_dir=cfg.get("model_versions_dir", "learning/models")).print_history()
    log_memory("status")
    ModelLoader.instance().log_status()


def _maybe_save_model_version(registry, state, cfg: dict) -> None:
    import json
    ai_cache = Path("learning/ai_model_cache.json")
    if not ai_cache.exists():
        return
    model_data = json.loads(ai_cache.read_text())
    n_cases = model_data.get("knn", {}).get("n_cases", 0)
    min_cases = cfg.get("min_cases_for_version", 10)
    if n_cases >= min_cases:
        current = registry.current_version()
        if n_cases > (current.n_training if current else 0):
            v = registry.save_version(
                model_data=model_data, n_training=n_cases,
                metrics={"n_cases": n_cases},
                notes=f"Auto-save: {len(state.depth_results)} images",
                auto_promote=True,
            )
            log.info(f"  [Versioning] Saved {v.version_id} (n_cases={n_cases})")


def main():
    parser = argparse.ArgumentParser(description="Flood Pipeline v2")
    parser.add_argument("--input",  "-i", help="Folder ảnh hoặc file ảnh")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--output", "-o", help="Override output dir")
    parser.add_argument("--async",  dest="async_mode", action="store_true",
                        help="Dùng AsyncFloodPipeline")
    parser.add_argument("--status", action="store_true",
                        help="Xem trạng thái system")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    if args.debug:
        setup_logging(level=logging.DEBUG)

    cfg = load_config(args.config)
    if args.output:
        cfg["output_dir"] = args.output

    if args.status:
        show_status(cfg)
        return 0

    if not args.input:
        parser.print_help()
        return 1

    return run_pipeline(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
