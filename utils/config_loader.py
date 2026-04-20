# -*- coding: utf-8 -*-
"""
utils/config_loader.py
------------------------
Doc va merge config tu config.yaml vao cfg dict.
Cho phep override moi gia tri mac dinh bang file config.
"""

import logging
from pathlib import Path
from typing import Any

from .constants import DEFAULT_YOLO_MODEL, DEFAULT_POSE_MODEL, DEFAULT_DEPTH_MODEL, DEFAULT_DINO_MODEL, DEFAULT_QUERY

log = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Doc config.yaml, tra ve dict phat trien.
    Neu file khong ton tai, tra ve dict rong.
    """
    p = Path(config_path)
    if not p.exists():
        log.debug(f"  config.yaml not found at {config_path}, using defaults")
        return {}

    try:
        import yaml
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        log.info(f"  Config loaded from {config_path}")
        return raw
    except ImportError:
        log.warning("  PyYAML not installed: pip install pyyaml")
        return {}
    except Exception as e:
        log.warning(f"  Cannot parse config.yaml: {e}")
        return {}


def apply_config_to_cfg(yaml_config: dict, cfg: dict) -> dict:
    """
    Ap dung gia tri tu config.yaml vao cfg dict.
    Uu tien: cfg (interactive input) > config.yaml > default code.
    Tuc la config.yaml chi set DEFAULT, nguoi dung van co the override.
    """
    if not yaml_config:
        return cfg

    # ── Crawl defaults ──────────────────────────────────────────────
    crawl = yaml_config.get("crawl", {})
    if "query" not in cfg or cfg["query"] == DEFAULT_QUERY:
        cfg.setdefault("query",       crawl.get("default_query", DEFAULT_QUERY))
    cfg.setdefault("max_images",  crawl.get("max_images", 50))
    cfg.setdefault("sources",     crawl.get("default_sources", ["bing","google"]))

    # ── Filter defaults ─────────────────────────────────────────────
    filt = yaml_config.get("filter", {})
    cfg.setdefault("blur_thresh",        filt.get("blur_threshold",      100.0))
    cfg.setdefault("hash_thresh",        filt.get("hash_threshold",      10))
    cfg.setdefault("max_text_density",   filt.get("max_text_density",    0.15))
    cfg.setdefault("require_flood_water",filt.get("require_flood_water", False))
    cfg.setdefault("max_banner_ratio",   filt.get("max_banner_ratio",    0.50))

    # ── Enhance defaults ─────────────────────────────────────────────
    enh = yaml_config.get("enhance", {})
    cfg.setdefault("enhance_images",      enh.get("enable",          True))
    cfg.setdefault("dark_threshold",      enh.get("dark_threshold",  80.0))
    cfg.setdefault("use_sharpen",         enh.get("sharpen",         False))

    # Fix borders from config
    borders = enh.get("fix_borders", {})
    cfg.setdefault("fix_borders",        borders.get("enable",       True))
    cfg.setdefault("border_method",      borders.get("method",       "lama"))
    cfg.setdefault("border_black_thresh",borders.get("black_thresh", 15))
    cfg.setdefault("border_min_pct",     borders.get("min_bar_width_pct", 0.03))

    # ── Watermark defaults ──────────────────────────────────────────
    wm = yaml_config.get("watermark", {})
    cfg.setdefault("check_watermark",    wm.get("enable",             True))
    cfg.setdefault("remove_watermark",   wm.get("remove",             True))
    cfg.setdefault("watermark_method",   wm.get("method",             "lama"))
    cfg.setdefault("known_wm_patterns",  wm.get("known_patterns",     []))

    # ── Depth defaults ──────────────────────────────────────────────
    dep = yaml_config.get("depth", {})
    cfg.setdefault("depth_model",  dep.get("model",      DEFAULT_DEPTH_MODEL))
    cfg.setdefault("yolo_model",   dep.get("yolo_model", DEFAULT_YOLO_MODEL))
    cfg.setdefault("yolo_conf",    dep.get("yolo_conf",  0.35))
    cfg.setdefault("use_dino",     dep.get("use_dino",   True))
    cfg.setdefault("dino_model",   dep.get("dino_model", DEFAULT_DINO_MODEL))

    # Custom flood level thresholds
    fl = dep.get("flood_levels", {})
    if fl:
        cfg["custom_flood_levels"] = fl

    # ── Drive defaults ───────────────────────────────────────────────
    drv = yaml_config.get("drive", {})
    cfg.setdefault("drive_folder",  drv.get("folder_name",            "FloodAnalysis"))
    cfg.setdefault("delete_local",  drv.get("delete_local_default",   False))

    # ── Output defaults ──────────────────────────────────────────────
    out = yaml_config.get("output", {})
    cfg.setdefault("output_dir",    out.get("base_dir", "output"))

    # ── UI / Labels ─────────────────────────────────────────────────
    ui = yaml_config.get("ui", {})
    cfg["_ui_labels"]    = ui.get("step_labels", {})
    cfg["_ui_questions"] = ui.get("questions", {})
    cfg["_ui_language"]  = ui.get("language", "vi")

    return cfg


def get_label(cfg: dict, key: str, default: str) -> str:
    """Lay label tu config, fallback ve default."""
    return cfg.get("_ui_labels", {}).get(key, default)


def get_question(cfg: dict, key: str, default: str) -> str:
    """Lay cau hoi tu config, fallback ve default."""
    q = cfg.get("_ui_questions", {}).get(key, "")
    return q if q else default
