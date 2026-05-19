# -*- coding: utf-8 -*-
"""
utils/pipeline_version.py
==========================
Gắn version metadata vào mỗi kết quả pipeline để traceability:

    {
      "pipeline_version": "2.1.0",
      "depth_model": "depth-anything-v2-small",
      "yolo_model":  "yolov8n.pt",
      "config_hash": "a3f9b12c",
      "run_time":    "2026-05-19T10:30:00",
    }

Lý do: nếu đổi model, kết quả cũ và mới sẽ khác. Không lưu version thì không
biết kết quả nào được tạo bởi bản nào, cũng không thể reproduce.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("utils.version")

PIPELINE_VERSION = "2.1.0"


def _hash_cfg(cfg: dict) -> str:
    """SHA-8 của config dict (stable hash)."""
    try:
        canonical = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:8]
    except Exception:
        return "unknown"


def build_version_meta(cfg: dict) -> Dict[str, str]:
    """
    Tạo version metadata từ config hiện tại.

    Returns:
        dict chứa pipeline_version, model names, config_hash, run_time
    """
    return {
        "pipeline_version": PIPELINE_VERSION,
        "depth_model":      _model_shortname(cfg.get("depth_model", "")),
        "yolo_model":       _model_shortname(cfg.get("yolo_model", "")),
        "dino_model":       _model_shortname(cfg.get("dino_model", "")) if cfg.get("use_dino") else "disabled",
        "config_hash":      _hash_cfg(cfg),
        "run_time":         datetime.now().isoformat(timespec="seconds"),
    }


def attach_version(result: Any, version_meta: Dict[str, str]):
    """Gắn version_meta vào result (dict hoặc object)."""
    if isinstance(result, dict):
        result["_version"] = version_meta
    else:
        try:
            object.__setattr__(result, "_version", version_meta)
        except Exception:
            pass


def attach_version_batch(results: list, version_meta: Dict[str, str]):
    """Gắn version vào toàn bộ batch results."""
    for r in results:
        attach_version(r, version_meta)


def save_run_manifest(
    run_id: str,
    cfg: dict,
    state: Any,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Lưu file run_manifest.json vào output_dir.
    Chứa đầy đủ version, config hash, timing, stats.

    Returns:
        Path đến file manifest đã lưu
    """
    out_dir = output_dir or getattr(state, "output_dir", None)
    if not out_dir:
        raise ValueError("output_dir chưa được set")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id":           run_id,
        **build_version_meta(cfg),
        "total_input":      len(getattr(state, "input_images", [])),
        "total_analyzed":   len(getattr(state, "depth_results", [])),
        "total_errors":     len(getattr(state, "errors", [])),
        "timings":          getattr(state, "timings", {}),
        "config_snapshot":  {
            k: v for k, v in cfg.items()
            if k not in ("_ui_labels", "_ui_questions")  # bỏ UI labels
        },
    }

    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("[Version] Manifest saved → %s", manifest_path)
    return manifest_path


def _model_shortname(model_path: str) -> str:
    """Rút gọn HuggingFace path thành tên ngắn gọn."""
    if not model_path:
        return "unknown"
    # depth-anything/Depth-Anything-V2-Small-hf → depth-anything-v2-small
    name = Path(model_path).name.lower()
    name = name.replace("-hf", "").replace("_", "-")
    return name or model_path
