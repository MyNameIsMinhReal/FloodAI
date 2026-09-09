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
    from utils.config_loader import load_config as load_yaml_config, apply_config_to_cfg
    raw = load_yaml_config(config_path)
    cfg = apply_config_to_cfg(raw, {})
    return cfg


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
    # [v4] IntermediateCache singleton được khởi tạo ở đây nếu cần dùng.
    # Hiện tại chưa có module nào gọi get_cache(), giữ lại để tương lai
    # cache depth/segmentation results giữa các batch.
    from utils.memory_manager import IntermediateCache, log_memory
    IntermediateCache.instance(max_entries=cfg.get("cache_max_entries", 200))
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


def run_watch(args, cfg: dict):
    """
    Watch mode: theo dõi folder, tự tạo job khi có ảnh mới.
    python main.py watch --input data/incoming --interval 5
    """
    import time as _time
    from pipeline.orchestrator import IMAGE_EXTENSIONS

    watch_dir = Path(args.input)
    interval  = getattr(args, "interval", 10)

    if not watch_dir.exists():
        log.error("Folder không tồn tại: %s", watch_dir)
        return 1

    log.info("  [Watch] Đang theo dõi %s (interval=%ds)", watch_dir, interval)
    seen: set = set()

    while True:
        current = {
            p for p in watch_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }
        new_files = sorted(current - seen)
        if new_files:
            log.info("  [Watch] %d ảnh mới phát hiện", len(new_files))
            try:
                run_pipeline_images(new_files, cfg)
            except Exception as exc:
                log.error("  [Watch] Pipeline lỗi: %s", exc)
            seen.update(new_files)
        _time.sleep(interval)


def run_pipeline_images(images, cfg: dict) -> int:
    """Chạy pipeline với danh sách ảnh (helper cho watch mode)."""
    from pipeline.orchestrator import FloodPipeline
    pipeline = FloodPipeline(cfg)
    state = pipeline.run(images)
    log.info("  [Watch] Xử lý xong %d ảnh → %s", len(state.depth_results), state.output_dir)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Flood Pipeline v2")

    sub = parser.add_subparsers(dest="command")

    # ── run (mặc định) ────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Phân tích ảnh (mặc định)")
    run_p.add_argument("--input",  "-i", required=True, help="Folder hoặc file ảnh")
    run_p.add_argument("--config", "-c", default="config.yaml")
    run_p.add_argument("--output", "-o", help="Override output dir")
    run_p.add_argument("--safe",   action="store_true", help="Safe mode")
    run_p.add_argument("--async",  dest="async_mode", action="store_true")
    run_p.add_argument("--debug",  action="store_true")

    # ── watch ─────────────────────────────────────────────────────────────────
    watch_p = sub.add_parser("watch", help="Theo dõi folder, tự xử lý ảnh mới")
    watch_p.add_argument("--input",    "-i", required=True, help="Folder cần theo dõi")
    watch_p.add_argument("--config",   "-c", default="config.yaml")
    watch_p.add_argument("--interval", "-n", type=int, default=10, help="Giây giữa các lần kiểm tra")
    watch_p.add_argument("--safe",     action="store_true")
    watch_p.add_argument("--debug",    action="store_true")

    # ── status ────────────────────────────────────────────────────────────────
    stat_p = sub.add_parser("status", help="Xem trạng thái system")
    stat_p.add_argument("--config", "-c", default="config.yaml")

    # Tương thích ngược: không có subcommand → chạy như run
    parser.add_argument("--input",  "-i", help=argparse.SUPPRESS)
    parser.add_argument("--config", "-c", default="config.yaml", help=argparse.SUPPRESS)
    parser.add_argument("--output", "-o", help=argparse.SUPPRESS)
    parser.add_argument("--async",  dest="async_mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--status", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--safe",   action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug",  action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.debug:
        setup_logging(level=logging.DEBUG)

    cfg_file = getattr(args, "config", "config.yaml")
    cfg = load_config(cfg_file)
    if hasattr(args, "output") and args.output:
        cfg["output_dir"] = args.output

    if args.safe or cfg.get("safe_mode"):
        cfg.update({"skip_drive": True, "skip_learning": True,
                    "skip_hard_mining": True, "skip_versioning": True})
        log.info("  [Safe Mode] Drive/Learning/Mining bị tắt.")

    # Dispatch subcommands
    if args.command == "watch":
        return run_watch(args, cfg)
    if args.command == "status" or getattr(args, "status", False):
        show_status(cfg)
        return 0
    if args.command == "run" or (args.command is None and args.input):
        return run_pipeline(args, cfg)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
