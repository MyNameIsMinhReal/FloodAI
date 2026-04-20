# -*- coding: utf-8 -*-
"""
core/model_loader.py  —  Singleton Model Loader
=================================================
Giải quyết vấn đề: mỗi stage load model riêng → tốn RAM + chậm startup.

Pattern: Singleton + Lazy Loading + Preload mode

Sử dụng:
    # Preload tất cả khi start pipeline
    loader = ModelLoader.instance()
    loader.preload(["yolo", "depth", "pose"])

    # Trong mỗi stage: lấy model đã cached
    yolo = loader.get("yolo")
    depth_model = loader.get("depth")

    # Sau khi dùng xong: giải phóng model nặng
    loader.unload("depth")

Model Lifecycle:
    preload → cached in memory → auto-unload if memory low
                                  └→ re-load on next get()
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Set

log = logging.getLogger("core.model_loader")


# ── Model Registry ─────────────────────────────────────────────────────────────

class _ModelEntry:
    """Metadata cho một model trong cache."""
    def __init__(self, name: str, loader_fn: Callable, config: dict):
        self.name       = name
        self.loader_fn  = loader_fn
        self.config     = config
        self.model: Any = None
        self.loaded_at: Optional[float] = None
        self.last_used: Optional[float] = None
        self.load_count: int = 0
        self.load_time_s: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> Any:
        if self.is_loaded:
            self.last_used = time.time()
            return self.model

        log.info(f"  [ModelLoader] Loading '{self.name}'…")
        t0 = time.time()
        self.model = self.loader_fn(self.config)
        self.load_time_s = time.time() - t0
        self.loaded_at = time.time()
        self.last_used = time.time()
        self.load_count += 1
        log.info(f"  [ModelLoader] '{self.name}' loaded in {self.load_time_s:.2f}s")
        return self.model

    def unload(self) -> None:
        if not self.is_loaded:
            return
        try:
            import torch
            if hasattr(self.model, "cpu"):
                self.model.cpu()
        except ImportError:
            pass
        del self.model
        self.model = None
        gc.collect()
        _free_gpu()
        log.debug(f"  [ModelLoader] '{self.name}' unloaded")


# ── Singleton Loader ────────────────────────────────────────────────────────────

class ModelLoader:
    """
    Singleton model loader: đảm bảo mỗi model chỉ load 1 lần duy nhất.

    Thread-safe: dùng lock riêng cho từng model entry.

    Ví dụ:
        loader = ModelLoader.instance()
        loader.register("yolo", _load_yolo, cfg)
        loader.register("depth", _load_depth, cfg)

        loader.preload(["yolo", "depth"])      # load ngay khi start
        model = loader.get("yolo")             # lấy từ cache
    """

    _instance: Optional["ModelLoader"] = None
    _init_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ModelLoader":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._registry: Dict[str, _ModelEntry] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._loaded_names: Set[str] = set()

    # ── Registration ───────────────────────────────────────────────────────────

    def register(self, name: str, loader_fn: Callable, config: dict | None = None) -> None:
        """
        Đăng ký một model với loader function.

        Args:
            name:      Tên model (e.g., "yolo", "depth", "pose")
            loader_fn: fn(config) → model object
            config:    Dict config truyền vào loader_fn
        """
        with self._global_lock:
            if name not in self._registry:
                self._registry[name] = _ModelEntry(name, loader_fn, config or {})
                self._locks[name] = threading.Lock()
                log.debug(f"  [ModelLoader] Registered '{name}'")

    def register_all_from_config(self, cfg: dict) -> None:
        """
        Tự động register các model phổ biến từ config.yaml.
        Chỉ register nếu chưa có.
        """
        model_cfg = cfg.get("models", {})

        # YOLO detector
        yolo_key = model_cfg.get("detector", cfg.get("yolo_model", "yolov8n.pt"))
        self.register("yolo", _loader_yolo, {"model_path": yolo_key,
                                              "conf": cfg.get("yolo_conf", 0.35)})

        # Pose model
        pose_key = model_cfg.get("pose", cfg.get("pose_model", "yolov8n-pose.pt"))
        self.register("pose", _loader_yolo, {"model_path": pose_key,
                                              "task": "pose"})

        # Depth model
        depth_key = model_cfg.get("depth", "depth-anything/Depth-Anything-V2-Small-hf")
        self.register("depth", _loader_depth, {"model_name": depth_key})

        # DINOv2
        dino_key = model_cfg.get("dino", "facebook/dinov2-small")
        self.register("dino", _loader_dino, {"model_name": dino_key})

    # ── Loading ────────────────────────────────────────────────────────────────

    def preload(self, names: list) -> None:
        """
        Preload một danh sách models ngay khi khởi động pipeline.
        Chạy tuần tự để tránh OOM khi load nhiều model cùng lúc.
        """
        log.info(f"  [ModelLoader] Preloading {len(names)} models: {names}")
        for name in names:
            try:
                self.get(name)
            except Exception as exc:
                log.warning(f"  [ModelLoader] Preload '{name}' failed: {exc}")

    def get(self, name: str) -> Any:
        """
        Lấy model (từ cache hoặc load mới nếu chưa có).
        Thread-safe: chỉ 1 thread load cùng lúc cho mỗi model.

        Raises:
            KeyError: nếu model chưa được register
        """
        if name not in self._registry:
            raise KeyError(
                f"Model '{name}' chưa được register. "
                f"Các model available: {list(self._registry.keys())}"
            )
        lock = self._locks[name]
        with lock:
            entry = self._registry[name]
            model = entry.load()
            self._loaded_names.add(name)
            return model

    def get_or_none(self, name: str) -> Optional[Any]:
        """Lấy model, trả về None nếu không có / load thất bại."""
        try:
            return self.get(name)
        except Exception as exc:
            log.warning(f"  [ModelLoader] get_or_none('{name}'): {exc}")
            return None

    # ── Unloading ──────────────────────────────────────────────────────────────

    def unload(self, name: str) -> None:
        """Giải phóng một model khỏi memory."""
        if name in self._registry:
            with self._locks[name]:
                self._registry[name].unload()
                self._loaded_names.discard(name)

    def unload_heavy(self) -> None:
        """
        Giải phóng các model nặng (depth, dino) sau khi xử lý xong.
        Giữ lại các model nhẹ (yolo) cho các stages tiếp theo.
        """
        HEAVY = {"depth", "dino"}
        for name in list(self._loaded_names):
            if name in HEAVY:
                self.unload(name)
                log.info(f"  [ModelLoader] Unloaded heavy model: '{name}'")

    def unload_all(self) -> None:
        """Giải phóng tất cả models."""
        for name in list(self._loaded_names):
            self.unload(name)

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Trả về trạng thái loading của tất cả models."""
        return {
            name: {
                "loaded":      entry.is_loaded,
                "load_count":  entry.load_count,
                "load_time_s": round(entry.load_time_s, 2),
                "last_used":   entry.last_used,
            }
            for name, entry in self._registry.items()
        }

    def log_status(self) -> None:
        """Log trạng thái loading."""
        log.info("  [ModelLoader] Status:")
        for name, info in self.status().items():
            icon = "✅" if info["loaded"] else "⬜"
            log.info(
                f"    {icon} {name:12s} loaded={info['loaded']} "
                f"count={info['load_count']} time={info['load_time_s']}s"
            )


# ── Default loaders ────────────────────────────────────────────────────────────

def _loader_yolo(config: dict) -> Any:
    """Load YOLO model."""
    from ultralytics import YOLO
    model_path = config.get("model_path", "yolov8n.pt")
    model = YOLO(model_path)
    log.debug(f"  YOLO loaded: {model_path}")
    return model


def _loader_depth(config: dict) -> Any:
    """Load Depth Anything V2 từ HuggingFace."""
    from transformers import pipeline as hf_pipeline
    model_name = config.get("model_name", "depth-anything/Depth-Anything-V2-Small-hf")
    pipe = hf_pipeline(task="depth-estimation", model=model_name)
    log.debug(f"  Depth model loaded: {model_name}")
    return pipe


def _loader_dino(config: dict) -> Any:
    """Load DINOv2 feature extractor."""
    from transformers import AutoModel, AutoImageProcessor
    model_name = config.get("model_name", "facebook/dinov2-small")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    log.debug(f"  DINOv2 loaded: {model_name}")
    return {"model": model, "processor": processor}


def _free_gpu():
    """Giải phóng GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
