# -*- coding: utf-8 -*-
"""
utils/memory_manager.py  —  Memory Manager + Intermediate Cache
================================================================
Nâng cấp từ phiên bản cũ (chỉ có GPU free) → thêm:

  1. Intermediate cache: lưu kết quả tính toán trung gian
     - segmentation maps, pose keypoints, YOLO detections
     Tránh compute lại khi cùng ảnh chạy qua nhiều stages

  2. GPU/RAM tracking có pct usage

  3. LRU eviction: tự động xóa cache cũ khi đầy

Sử dụng:
    cache = IntermediateCache.instance()
    cache.set("pose:img001.jpg", keypoints)
    keypoints = cache.get("pose:img001.jpg")   # None nếu miss
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

_lock = threading.Lock()


# ── GPU / RAM utilities (giữ nguyên từ bản cũ, thêm gpu_pct) ──────────────────

def free_gpu_memory():
    """Giải phóng GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log.debug("  GPU memory cleared")
    except ImportError:
        pass


def free_model(model_ref):
    """Giải phóng 1 model khỏi bộ nhớ."""
    try:
        import torch
        if hasattr(model_ref, "cpu"):
            model_ref.cpu()
        del model_ref
    except Exception:
        pass
    gc.collect()
    free_gpu_memory()


def get_memory_usage() -> dict:
    """Lấy thông tin bộ nhớ hiện tại."""
    info = {}
    try:
        import psutil
        proc = psutil.Process()
        info["ram_mb"] = proc.memory_info().rss / 1024 / 1024
    except ImportError:
        info["ram_mb"] = 0

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_mb_used"]  = torch.cuda.memory_allocated() / 1024 / 1024
            info["gpu_mb_total"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            info["gpu_pct"] = info["gpu_mb_used"] / (info["gpu_mb_total"] + 1e-9) * 100
    except ImportError:
        pass

    return info


def log_memory(label: str = ""):
    """Log bộ nhớ hiện tại."""
    info = get_memory_usage()
    ram  = info.get("ram_mb", 0)
    gpu  = info.get("gpu_mb_used", 0)
    total_gpu = info.get("gpu_mb_total", 0)
    if gpu > 0:
        log.info(
            f"  Memory {label}: RAM={ram:.0f}MB "
            f"GPU={gpu:.0f}/{total_gpu:.0f}MB ({info.get('gpu_pct', 0):.0f}%)"
        )
    else:
        log.info(f"  Memory {label}: RAM={ram:.0f}MB")


# ── Intermediate Cache (NEW) ───────────────────────────────────────────────────

class _CacheEntry:
    def __init__(self, value: Any):
        self.value    = value
        self.created  = time.time()
        self.accessed = time.time()
        self.hits     = 0

    def touch(self):
        self.accessed = time.time()
        self.hits += 1


class IntermediateCache:
    """
    LRU cache cho các kết quả tính toán trung gian trong pipeline.

    Lưu: segmentation maps, pose keypoints, YOLO detections, depth maps.
    Tránh compute lại cùng một ảnh khi chạy qua nhiều stages.
    Thread-safe. Singleton.

    Key convention:
        "pose:{stem}"  → pose keypoints
        "seg:{stem}"   → segmentation mask
        "yolo:{stem}"  → YOLO detections
        "depth:{stem}" → depth map
        "water:{stem}" → water mask
    """

    _instance: Optional["IntermediateCache"] = None
    _init_lock = threading.Lock()

    @classmethod
    def instance(cls, max_entries: int = 200) -> "IntermediateCache":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(max_entries=max_entries)
        return cls._instance

    def __init__(self, max_entries: int = 200):
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._max    = max_entries
        self._lock   = threading.Lock()
        self._hits   = 0
        self._misses = 0

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = _CacheEntry(value)
            while len(self._cache) > self._max:
                evicted, _ = self._cache.popitem(last=False)
                log.debug(f"  [Cache] Evicted: {evicted}")

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.touch()
                self._cache.move_to_end(key)
                self._hits += 1
                return entry.value
            self._misses += 1
            return None

    def get_or_compute(self, key: str, fn: Callable, *args, **kwargs) -> Any:
        """
        Lấy từ cache hoặc compute nếu miss, rồi cache lại.

        Ví dụ:
            kps = cache.get_or_compute("pose:img1", pose_model.run, img)
        """
        result = self.get(key)
        if result is not None:
            return result
        result = fn(*args, **kwargs)
        if result is not None:
            self.set(key, result)
        return result

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear_prefix(self, prefix: str) -> int:
        """Xóa tất cả entries bắt đầu bằng prefix."""
        with self._lock:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                del self._cache[k]
            return len(keys)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = self._misses = 0
        gc.collect()
        log.info("  [Cache] Cleared all entries")

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries":  len(self._cache),
                "max":      self._max,
                "hits":     self._hits,
                "misses":   self._misses,
                "hit_rate": f"{self.hit_rate:.1%}",
            }

    def log_stats(self, label: str = "") -> None:
        s = self.stats()
        log.info(
            f"  [Cache{' ' + label if label else ''}] "
            f"entries={s['entries']}/{s['max']} "
            f"hit_rate={s['hit_rate']} "
            f"({s['hits']} hits, {s['misses']} misses)"
        )


# ── Convenience ────────────────────────────────────────────────────────────────

def get_cache() -> IntermediateCache:
    """Lấy global cache singleton."""
    return IntermediateCache.instance()


def cache_key(stage: str, image_path: str) -> str:
    """Tạo cache key chuẩn: 'stage:stem'."""
    from pathlib import Path
    return f"{stage}:{Path(image_path).stem}"
