# -*- coding: utf-8 -*-
"""
learning/config_promotion.py
=============================
Human-in-the-loop config promotion — tránh tự học và thay đổi ngay mà
không kiểm chứng.

Luồng an toàn:
    1. Pipeline dự đoán với current_config
    2. Người review sửa kết quả → evaluation chạy
    3. Nếu bản mới (candidate) tốt hơn → promote lên current
    4. Nếu tệ hơn → rollback (giữ nguyên hoặc về rollback_config)

4 trạng thái config:
    candidate_config  — đang được thử nghiệm, chưa promote
    current_config    — đang dùng trong production
    best_config       — tốt nhất từ trước đến nay (không xóa)
    rollback_config   — config trước khi promote lần cuối

Dùng:
    pm = ConfigPromoter()
    pm.propose_candidate(new_cfg, reason="Better threshold after 50 reviews")
    ...
    pm.promote_candidate()   # nếu đánh giá tốt hơn
    pm.rollback()            # nếu tệ hơn
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("learning.config_promotion")

_DEFAULT_DIR = Path("learning/config_versions")

CONFIG_STATES = ("candidate", "current", "best", "rollback")


class ConfigPromoter:
    """
    Quản lý vòng đời config: candidate → current → rollback.

    Tất cả config được lưu dưới dạng JSON, có timestamp và lý do.
    """

    def __init__(self, base_dir: Path = _DEFAULT_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ── Read ──────────────────────────────────────────────────────────────────

    def load(self, state: str) -> Optional[Dict]:
        """Đọc config theo state (candidate/current/best/rollback)."""
        p = self._path(state)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("[Promoter] Không đọc được %s: %s", state, exc)
            return None

    def current_config(self) -> Optional[Dict]:
        """Trả về current config (đang dùng)."""
        entry = self.load("current")
        return entry.get("config") if entry else None

    def best_config(self) -> Optional[Dict]:
        """Trả về config tốt nhất từ trước đến nay."""
        entry = self.load("best")
        return entry.get("config") if entry else None

    def status(self) -> Dict[str, Any]:
        """Tóm tắt trạng thái hiện tại."""
        result = {}
        for state in CONFIG_STATES:
            entry = self.load(state)
            if entry:
                result[state] = {
                    "timestamp": entry.get("timestamp", "?"),
                    "reason":    entry.get("reason", ""),
                    "metrics":   entry.get("metrics", {}),
                }
        return result

    # ── Write ─────────────────────────────────────────────────────────────────

    def propose_candidate(
        self,
        config: dict,
        reason: str = "",
        metrics: Optional[dict] = None,
    ):
        """
        Đề xuất config mới để thử nghiệm (candidate).
        Không ảnh hưởng current config cho đến khi promote.

        Args:
            config: dict config mới
            reason: lý do thay đổi (ví dụ "Threshold update sau 50 reviews")
            metrics: kết quả evaluation của candidate (nếu đã có)
        """
        self._save("candidate", config, reason, metrics)
        log.info("[Promoter] Candidate proposed: %s", reason)

    def promote_candidate(self, force: bool = False) -> bool:
        """
        Promote candidate lên current (nếu tốt hơn).

        Args:
            force: nếu True, promote dù không có metrics so sánh

        Returns:
            True nếu promote thành công
        """
        candidate = self.load("candidate")
        if not candidate:
            log.warning("[Promoter] Không có candidate để promote")
            return False

        current = self.load("current")

        # Kiểm tra candidate có tốt hơn không
        if not force and current and candidate.get("metrics") and current.get("metrics"):
            cand_acc = candidate["metrics"].get("accuracy", 0)
            curr_acc = current["metrics"].get("accuracy", 0)
            if cand_acc <= curr_acc:
                log.warning(
                    "[Promoter] Candidate (acc=%.3f) không tốt hơn current (acc=%.3f) — dùng force=True để override",
                    cand_acc, curr_acc,
                )
                return False

        # Lưu current vào rollback trước khi replace
        if current:
            self._save("rollback", current["config"],
                       f"Saved before promote: {candidate.get('reason', '')}")

        # Promote candidate → current
        self._save("current", candidate["config"],
                   f"Promoted: {candidate.get('reason', '')}", candidate.get("metrics"))

        # Nếu tốt hơn best thì update best
        best = self.load("best")
        cand_acc = (candidate.get("metrics") or {}).get("accuracy", 0)
        best_acc = (best.get("metrics") or {}).get("accuracy", 0) if best else 0
        if cand_acc > best_acc:
            self._save("best", candidate["config"],
                       f"New best: acc={cand_acc:.3f}", candidate.get("metrics"))
            log.info("[Promoter] New best config! acc=%.3f", cand_acc)

        # Xóa candidate cũ
        self._path("candidate").unlink(missing_ok=True)
        log.info("[Promoter] Promoted candidate → current")
        return True

    def rollback(self) -> bool:
        """
        Rollback current về rollback_config (config trước promote lần cuối).

        Returns:
            True nếu rollback thành công
        """
        rollback = self.load("rollback")
        if not rollback:
            log.warning("[Promoter] Không có rollback config")
            return False

        current = self.load("current")
        if current:
            self._save("candidate", current["config"],
                       "Saved as candidate after rollback")

        self._save("current", rollback["config"],
                   "Rolled back", rollback.get("metrics"))
        self._path("rollback").unlink(missing_ok=True)
        log.info("[Promoter] Rolled back to previous config")
        return True

    def update_metrics(self, state: str, metrics: dict):
        """Cập nhật metrics cho một config state sau khi evaluation chạy."""
        entry = self.load(state)
        if not entry:
            log.warning("[Promoter] Không tìm thấy state '%s'", state)
            return
        entry["metrics"] = metrics
        entry["metrics_updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._path(state).write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("[Promoter] Updated metrics for '%s': %s", state, metrics)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _path(self, state: str) -> Path:
        return self.base_dir / f"{state}_config.json"

    def _save(self, state: str, config: dict, reason: str = "",
              metrics: Optional[dict] = None):
        entry = {
            "state":     state,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reason":    reason,
            "metrics":   metrics or {},
            "config":    config,
        }
        self._path(state).write_text(
            json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
