# -*- coding: utf-8 -*-
"""
utils/model_registry.py
========================
Quản lý model tập trung từ models/registry.yaml.

Pipeline chỉ đọc model từ registry, không hard-code path rải rác.

Dùng:
    from utils.model_registry import ModelRegistry
    reg = ModelRegistry()
    model_cfg = reg.get("depth_small")     # dict
    profile   = reg.profile("cpu")        # resolve profile → cfg patch
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("utils.model_registry")

_DEFAULT_REGISTRY = Path(__file__).parent.parent / "models" / "registry.yaml"


class ModelRegistry:
    """
    Đọc models/registry.yaml và cung cấp:
      - get(model_id)     → dict config của model
      - profile(name)     → patch cfg theo profile (cpu/gpu/demo/prod)
      - list_models()     → danh sách models
      - resolve_cfg(cfg)  → tự điền model paths vào cfg từ registry
    """

    def __init__(self, registry_path: "str | Path" = _DEFAULT_REGISTRY):
        self._path = Path(registry_path)
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if not self._path.exists():
            log.warning("[Registry] registry.yaml không tìm thấy tại %s", self._path)
            return
        try:
            import yaml
            with open(self._path, encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
            n = len(self._data.get("models", {}))
            log.debug("[Registry] Loaded %d models từ %s", n, self._path)
        except ImportError:
            log.warning("[Registry] PyYAML chưa cài — model registry không hoạt động")
        except Exception as exc:
            log.warning("[Registry] Không đọc được registry: %s", exc)

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Lấy config của một model theo ID."""
        return self._data.get("models", {}).get(model_id)

    def path(self, model_id: str) -> Optional[str]:
        """Lấy path của model."""
        m = self.get(model_id)
        return m["path"] if m else None

    def list_models(self, model_type: Optional[str] = None) -> List[str]:
        """Danh sách model IDs, có thể lọc theo type."""
        models = self._data.get("models", {})
        if model_type:
            return [k for k, v in models.items() if v.get("type") == model_type]
        return list(models.keys())

    def profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Trả về profile config (depth/detector/feature model IDs).

        Ví dụ:
            reg.profile("cpu") → {"depth": "depth_small", "detector": "yolo_nano", ...}
        """
        profiles = self._data.get("profiles", {})
        if profile_name not in profiles:
            log.warning("[Registry] Profile '%s' không tồn tại", profile_name)
            return {}
        return profiles[profile_name]

    def resolve_cfg(self, cfg: dict) -> dict:
        """
        Tự điền model paths vào cfg từ registry.

        Nếu cfg có:
            model_profile: "cpu"
        thì tự điền:
            depth_model: <path từ registry>
            yolo_model:  <path từ registry>
            use_dino:    false
        """
        profile_name = cfg.get("model_profile")
        if not profile_name:
            return cfg

        p = self.profile(profile_name)
        if not p:
            return cfg

        # Depth
        if "depth" in p:
            m = self.get(p["depth"])
            if m:
                cfg.setdefault("depth_model", m["path"])

        # Detector
        if "detector" in p:
            m = self.get(p["detector"])
            if m:
                cfg.setdefault("yolo_model", m["path"])

        # Feature (DINO)
        if "feature" in p:
            m = self.get(p["feature"])
            if m:
                cfg.setdefault("dino_model", m["path"])

        if "use_dino" in p:
            cfg.setdefault("use_dino", p["use_dino"])

        log.info("[Registry] Resolved profile '%s' → depth=%s yolo=%s",
                 profile_name,
                 cfg.get("depth_model", "?"),
                 cfg.get("yolo_model", "?"))
        return cfg

    def validate(self) -> List[str]:
        """Kiểm tra model files nào cần download (chỉ local paths)."""
        missing = []
        for model_id, m in self._data.get("models", {}).items():
            p = m.get("path", "")
            if "/" not in p and not Path(p).exists():
                missing.append(f"{model_id}: {p}")
        return missing
