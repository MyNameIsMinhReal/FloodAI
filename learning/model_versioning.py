# -*- coding: utf-8 -*-
"""
learning/model_versioning.py  —  Model Versioning System
=========================================================
Theo dõi, lưu trữ, và quản lý các phiên bản model theo thời gian.

Vấn đề hiện tại:
  - Không biết prediction nào được tạo bởi model version nào
  - Không có audit trail khi retrain
  - Khó rollback nếu model mới kém hơn

Giải pháp:
  - ModelRegistry: quản lý tất cả versions
  - Mỗi prediction gắn với version cụ thể
  - So sánh metrics giữa các versions
  - Rollback về version tốt hơn nếu cần

Structure:
    models/
      registry.json        ← metadata của tất cả versions
      v1/                  ← model files
        ai_model_cache.json
        metadata.json
      v2/
        ...
      current -> v2/       ← symlink đến version hiện tại

Sử dụng:
    registry = ModelRegistry()

    # Lưu model version mới
    v = registry.save_version(model_data, metrics={"accuracy": 0.82})
    print(v.version_id)  # "v3"

    # Lấy version hiện tại
    current = registry.current_version()

    # Promote model tốt nhất
    registry.promote(version_id="v3")

    # Rollback nếu cần
    registry.rollback()
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("learning.model_versioning")

REGISTRY_FILE = "registry.json"
METADATA_FILE = "metadata.json"
CURRENT_LINK  = "current"


# ── Version metadata ───────────────────────────────────────────────────────────

@dataclass
class ModelVersion:
    """Metadata của một model version."""
    version_id:   str
    created_at:   str
    n_training:   int        # số training samples
    metrics:      Dict[str, Any]  = field(default_factory=dict)
    promoted_at:  Optional[str]  = None
    promoted_by:  str = "auto"
    notes:        str = ""
    is_current:   bool = False

    @property
    def version_num(self) -> int:
        """Lấy số version: 'v3' → 3."""
        return int(self.version_id.lstrip("v")) if self.version_id.startswith("v") else 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelVersion":
        return cls(**d)

    def __str__(self) -> str:
        promoted = f" [CURRENT]" if self.is_current else ""
        metrics_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in self.metrics.items())
        return f"  {self.version_id}{promoted}: {self.created_at} | n={self.n_training} | {metrics_str}"


# ── Model Registry ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Quản lý tất cả versions của AI model.

    Ví dụ sử dụng:
        registry = ModelRegistry(base_dir="learning/models")

        # Sau mỗi lần retrain:
        v = registry.save_version(
            model_data={"knn": knn_data, "calibrator": calib_data},
            metrics={"accuracy": 0.85, "n_cases": 150},
            notes="Retrain sau lũ tháng 4"
        )

        # Promote version tốt nhất:
        registry.promote(v.version_id)

        # Xem lịch sử:
        registry.print_history()
    """

    def __init__(self, base_dir: str = "learning/models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, ModelVersion] = {}
        self._load_registry()

    # ── Core API ───────────────────────────────────────────────────────────────

    def save_version(
        self,
        model_data: dict,
        metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
        n_training: int = 0,
        auto_promote: bool = False,
    ) -> ModelVersion:
        """
        Lưu một model version mới.

        Args:
            model_data:   Dict chứa model state (ai_model_cache format)
            metrics:      Dict metrics đánh giá (accuracy, n_cases, v.v.)
            notes:        Ghi chú (vd: "retrain sau lũ tháng 4")
            n_training:   Số training samples đã dùng
            auto_promote: Tự động promote nếu metrics tốt hơn hiện tại

        Returns:
            ModelVersion mới được tạo
        """
        # Tạo version ID mới
        version_id = self._next_version_id()
        version_dir = self.base_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Lưu model data
        model_path = version_dir / "ai_model_cache.json"
        model_path.write_text(
            json.dumps(model_data, indent=2, default=str),
            encoding="utf-8"
        )

        # Tạo version metadata
        version = ModelVersion(
            version_id=version_id,
            created_at=datetime.now().isoformat(),
            n_training=n_training,
            metrics=metrics or {},
            notes=notes,
        )

        # Lưu metadata
        meta_path = version_dir / METADATA_FILE
        meta_path.write_text(
            json.dumps(version.to_dict(), indent=2),
            encoding="utf-8"
        )

        self._registry[version_id] = version
        self._save_registry()

        log.info(f"  [Versioning] Saved {version_id} — n={n_training}, metrics={metrics}")

        # Auto-promote nếu được yêu cầu và tốt hơn current
        if auto_promote and self._is_better(version):
            self.promote(version_id, promoted_by="auto")

        return version

    def promote(self, version_id: str, promoted_by: str = "manual") -> None:
        """
        Promote một version thành current.

        Tạo/cập nhật symlink 'current' → version_dir.
        """
        if version_id not in self._registry:
            raise ValueError(f"Version '{version_id}' không tồn tại trong registry")

        # Reset current flag
        for v in self._registry.values():
            v.is_current = False

        self._registry[version_id].is_current = True
        self._registry[version_id].promoted_at = datetime.now().isoformat()
        self._registry[version_id].promoted_by = promoted_by

        # Update symlink
        current_link = self.base_dir / CURRENT_LINK
        if current_link.is_symlink() or current_link.exists():
            current_link.unlink()

        target = self.base_dir / version_id
        try:
            current_link.symlink_to(target)
        except (OSError, NotImplementedError):
            # Trên Windows có thể không support symlink → copy thay thế
            if (self.base_dir / CURRENT_LINK).exists():
                shutil.rmtree(self.base_dir / CURRENT_LINK)
            shutil.copytree(str(target), str(self.base_dir / CURRENT_LINK))

        self._save_registry()
        log.info(f"  [Versioning] Promoted {version_id} as current (by={promoted_by})")

    def rollback(self, steps: int = 1) -> Optional[ModelVersion]:
        """
        Rollback về version trước đó.

        Args:
            steps: Số bước rollback (default 1 = về version liền trước)

        Returns:
            Version mới được activate, hoặc None nếu không có
        """
        versions = self._sorted_versions()
        if len(versions) < 2:
            log.warning("  [Versioning] Không đủ versions để rollback")
            return None

        current = self.current_version()
        if current is None:
            log.warning("  [Versioning] Không có current version")
            return None

        # Tìm index của current
        version_ids = [v.version_id for v in versions]
        try:
            idx = version_ids.index(current.version_id)
        except ValueError:
            idx = len(version_ids) - 1

        target_idx = max(0, idx - steps)
        if target_idx == idx:
            log.warning("  [Versioning] Đang ở version cũ nhất, không thể rollback thêm")
            return None

        target = versions[target_idx]
        self.promote(target.version_id, promoted_by="rollback")
        log.info(f"  [Versioning] Rolled back từ {current.version_id} → {target.version_id}")
        return target

    # ── Query ──────────────────────────────────────────────────────────────────

    def current_version(self) -> Optional[ModelVersion]:
        """Lấy version đang được dùng."""
        for v in self._registry.values():
            if v.is_current:
                return v
        # Fallback: version mới nhất
        versions = self._sorted_versions()
        return versions[-1] if versions else None

    def current_model_path(self) -> Optional[Path]:
        """Lấy path đến model file của version hiện tại."""
        current = self.current_version()
        if current is None:
            # Thử dùng file gốc
            legacy = Path("learning/ai_model_cache.json")
            if legacy.exists():
                return legacy
            return None

        model_path = self.base_dir / current.version_id / "ai_model_cache.json"
        return model_path if model_path.exists() else None

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        return self._registry.get(version_id)

    def list_versions(self) -> List[ModelVersion]:
        """Danh sách tất cả versions, mới nhất trước."""
        return self._sorted_versions(reverse=True)

    def compare(self, v1_id: str, v2_id: str, metric: str = "accuracy") -> dict:
        """So sánh metrics giữa 2 versions."""
        v1 = self._registry.get(v1_id)
        v2 = self._registry.get(v2_id)
        if not v1 or not v2:
            return {}
        return {
            "metric":  metric,
            v1_id:     v1.metrics.get(metric),
            v2_id:     v2.metrics.get(metric),
            "winner":  v1_id if (v1.metrics.get(metric, 0) > v2.metrics.get(metric, 0)) else v2_id,
        }

    # ── Display ────────────────────────────────────────────────────────────────

    def print_history(self) -> None:
        """In lịch sử các versions."""
        log.info("  [Versioning] Model History:")
        for v in self._sorted_versions(reverse=True):
            log.info(str(v))

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_registry(self) -> None:
        registry_path = self.base_dir / REGISTRY_FILE
        if not registry_path.exists():
            return
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            self._registry = {
                k: ModelVersion.from_dict(v)
                for k, v in data.items()
            }
            log.debug(f"  [Versioning] Loaded {len(self._registry)} versions")
        except Exception as exc:
            log.warning(f"  [Versioning] Cannot load registry: {exc}")

    def _save_registry(self) -> None:
        registry_path = self.base_dir / REGISTRY_FILE
        try:
            data = {k: v.to_dict() for k, v in self._registry.items()}
            registry_path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8"
            )
        except Exception as exc:
            log.warning(f"  [Versioning] Cannot save registry: {exc}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _next_version_id(self) -> str:
        """Tạo version ID tiếp theo: v1, v2, v3, ..."""
        if not self._registry:
            return "v1"
        max_num = max(v.version_num for v in self._registry.values())
        return f"v{max_num + 1}"

    def _sorted_versions(self, reverse: bool = False) -> List[ModelVersion]:
        """Sắp xếp versions theo số thứ tự."""
        return sorted(
            self._registry.values(),
            key=lambda v: v.version_num,
            reverse=reverse,
        )

    def _is_better(self, candidate: ModelVersion, metric: str = "accuracy") -> bool:
        """Kiểm tra xem candidate có tốt hơn current không."""
        current = self.current_version()
        if current is None:
            return True
        current_metric = current.metrics.get(metric, 0)
        candidate_metric = candidate.metrics.get(metric, 0)
        return candidate_metric > current_metric
