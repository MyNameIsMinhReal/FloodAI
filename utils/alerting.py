# -*- coding: utf-8 -*-
"""
utils/alerting.py
==================
Gửi cảnh báo tự động khi phát hiện khu vực ngập nặng.

Config:
    alerts:
      enabled: true
      min_level: waist
      min_confidence: 0.75
      channels:
        - telegram
        - email
        - discord

Kênh hỗ trợ:
    telegram: cần TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
    email:    cần SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO
    discord:  cần DISCORD_WEBHOOK_URL

Dùng:
    alerter = FloodAlerter(cfg)
    alerter.check_and_alert(state)
"""

import logging
import os
import smtplib
import urllib.request
import json
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.orchestrator import PipelineState

log = logging.getLogger("utils.alerting")

LEVEL_ORDER = ["dry", "ankle", "knee", "waist", "chest", "submerged"]
LEVEL_VI = {
    "dry":       "khô ráo",
    "ankle":     "ngập mắt cá (~20cm)",
    "knee":      "ngập đầu gối (~50cm)",
    "waist":     "ngập ngang eo (~90cm)",
    "chest":     "ngập ngang ngực (~130cm)",
    "submerged": "ngập hoàn toàn",
}


class FloodAlerter:
    """
    Kiểm tra kết quả pipeline và gửi cảnh báo nếu đạt ngưỡng.

    Dùng:
        alerter = FloodAlerter(cfg)
        sent = alerter.check_and_alert(state)
        # sent = [(channel, message), ...]
    """

    def __init__(self, cfg: dict):
        alert_cfg = cfg.get("alerts", {})
        self.enabled        = alert_cfg.get("enabled", False)
        self.min_level      = alert_cfg.get("min_level", "waist")
        self.min_confidence = alert_cfg.get("min_confidence", 0.75)
        self.channels       = alert_cfg.get("channels", [])
        self.min_level_idx  = LEVEL_ORDER.index(self.min_level) if self.min_level in LEVEL_ORDER else 3

    def check_and_alert(self, state: "PipelineState") -> List[tuple]:
        """
        Kiểm tra depth_results và gửi cảnh báo nếu cần.

        Returns:
            list of (channel_name, success_bool)
        """
        if not self.enabled:
            return []

        danger = self._find_danger_results(state.depth_results)
        if not danger:
            return []

        message = self._format_message(danger, state)
        log.warning("[Alert] %d vùng nguy hiểm phát hiện — gửi cảnh báo", len(danger))

        results = []
        for channel in self.channels:
            try:
                ok = self._send(channel, message)
                results.append((channel, ok))
            except Exception as exc:
                log.error("[Alert] Lỗi gửi %s: %s", channel, exc)
                results.append((channel, False))
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_danger_results(self, depth_results: List[Any]) -> List[Any]:
        out = []
        for r in depth_results:
            level = _get(r, "flood_level", "dry") or "dry"
            conf  = _get(r, "confidence", 0.0) or 0.0
            try:
                level_idx = LEVEL_ORDER.index(level)
            except ValueError:
                continue
            if level_idx >= self.min_level_idx and conf >= self.min_confidence:
                out.append(r)
        return out

    def _format_message(self, danger: List[Any], state: "PipelineState") -> str:
        lines = [
            f"🚨 CẢNH BÁO LŨ LỤT — {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            f"Run ID: {state.run_id}",
            f"Phát hiện {len(danger)} khu vực nguy hiểm:\n",
        ]
        for r in danger[:5]:
            level    = _get(r, "flood_level", "?")
            depth_cm = _get(r, "depth_cm", 0) or 0
            conf     = _get(r, "confidence", 0) or 0
            img      = _get(r, "original_path", "") or _get(r, "image_path", "?")
            level_vi = LEVEL_VI.get(level, level)
            lines.append(f"  • {level_vi}, sâu ~{depth_cm:.0f}cm (conf {conf:.0%})")
            lines.append(f"    Ảnh: {img}")
        if len(danger) > 5:
            lines.append(f"  ... và {len(danger)-5} khu vực khác")
        lines.append(f"\nOutput: {state.output_dir or 'N/A'}")
        return "\n".join(lines)

    def _send(self, channel: str, message: str) -> bool:
        if channel == "telegram":
            return self._send_telegram(message)
        if channel == "email":
            return self._send_email(message)
        if channel == "discord":
            return self._send_discord(message)
        log.warning("[Alert] Kênh không hỗ trợ: %s", channel)
        return False

    def _send_telegram(self, message: str) -> bool:
        token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            log.warning("[Alert/Telegram] Thiếu TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID")
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": message}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status == 200
        log.info("[Alert/Telegram] Gửi %s", "✓" if ok else "✗")
        return ok

    def _send_email(self, message: str) -> bool:
        smtp_host = os.environ.get("SMTP_HOST", "")
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASS", "")
        to_addr   = os.environ.get("ALERT_EMAIL_TO", "")
        if not all([smtp_host, smtp_user, smtp_pass, to_addr]):
            log.warning("[Alert/Email] Thiếu SMTP config trong .env")
            return False
        msg = MIMEText(message, "plain", "utf-8")
        msg["Subject"] = f"🚨 [FloodAI] Cảnh báo lũ lụt {datetime.now().strftime('%d/%m %H:%M')}"
        msg["From"]    = smtp_user
        msg["To"]      = to_addr
        with smtplib.SMTP_SSL(smtp_host, 465) as smtp:
            smtp.login(smtp_user, smtp_pass)
            smtp.send_message(msg)
        log.info("[Alert/Email] Đã gửi → %s", to_addr)
        return True

    def _send_discord(self, message: str) -> bool:
        webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
        if not webhook:
            log.warning("[Alert/Discord] Thiếu DISCORD_WEBHOOK_URL")
            return False
        payload = json.dumps({"content": message}).encode()
        req = urllib.request.Request(
            webhook, data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status in (200, 204)
        log.info("[Alert/Discord] Gửi %s", "✓" if ok else "✗")
        return ok


def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
