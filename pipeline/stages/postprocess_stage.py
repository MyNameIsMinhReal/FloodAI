# -*- coding: utf-8 -*-
"""
Stage 5 — Post-process
=======================
Các bước xử lý sau depth estimation:
  - Route prediction (nếu có GPS)
  - Alert system (nếu flood > threshold)
  - Temporal tracking (nếu có nhiều ảnh cùng location)
"""
import logging
from typing import TYPE_CHECKING, Any

log = logging.getLogger("pipeline.postprocess")

if TYPE_CHECKING:
    from pipeline.orchestrator import PipelineState


class PostprocessStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def predict_routes(self, state: "PipelineState") -> None:
        """Dự báo tuyến đường an toàn từ kết quả depth + GPS."""
        try:
            from utils.flood_route_predictor import FloodRoutePredictor, depth_results_to_flood_points
            from types import SimpleNamespace

            loc_map  = state.location_map
            loc_list = []
            for dr in state.depth_results:
                key     = getattr(dr, "original_path", "")
                loc_obj = loc_map.get(key, {})
                loc_list.append(SimpleNamespace(
                    latitude  = loc_obj.get("latitude"),
                    longitude = loc_obj.get("longitude"),
                    address   = loc_obj.get("address", "") or "",
                ))

            flood_points = depth_results_to_flood_points(state.depth_results, loc_list)
            if not flood_points:
                log.warning("  [Route] Không có GPS → bỏ qua route prediction")
                return

            predictor = FloodRoutePredictor(output_dir=str(state.output_dir / "routes"))
            map_path = predictor.build_flood_map(flood_points, output_name="flood_overview")
            if map_path:
                log.info(f"  [Route] Flood map: {map_path}")

        except Exception as exc:
            log.warning(f"  [Route] Lỗi: {exc}")

    def check_alerts(self, state: "PipelineState") -> None:
        """
        Gửi alert nếu có ảnh ngập sâu vượt ngưỡng.
        Cấu hình trong config.yaml:
            alerts:
              flood_threshold: KNEE   # PUDDLE | ANKLE | KNEE | WAIST | CHEST
              send_email: true
        """
        alert_cfg   = self.cfg.get("alerts", {})
        threshold   = alert_cfg.get("flood_threshold", "WAIST")
        level_order = ["NO_FLOOD", "PUDDLE", "ANKLE", "KNEE", "WAIST", "CHEST", "SUBMERGED"]

        if threshold not in level_order:
            return

        threshold_idx = level_order.index(threshold)
        critical = [
            r for r in state.depth_results
            if level_order.index(getattr(r, "flood_level", "NO_FLOOD"))
            >= threshold_idx
        ]

        if not critical:
            return

        log.warning(f"  ⚠ ALERT: {len(critical)} ảnh ngập >= {threshold}!")
        if alert_cfg.get("send_email", False):
            self._send_email_alert(critical, alert_cfg)

    def _send_email_alert(self, results: list, alert_cfg: dict) -> None:
        """Gửi email alert — cần cấu hình SMTP trong .env."""
        try:
            import smtplib, os
            from email.message import EmailMessage

            smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
            smtp_port = int(os.environ.get("SMTP_PORT", "587"))
            sender    = os.environ.get("ALERT_EMAIL_FROM", "")
            receiver  = alert_cfg.get("email_to", os.environ.get("ALERT_EMAIL_TO", ""))
            password  = os.environ.get("ALERT_EMAIL_PASSWORD", "")

            if not all([sender, receiver, password]):
                log.warning("  [Alert] Thiếu SMTP config trong .env")
                return

            msg = EmailMessage()
            msg["Subject"] = f"[FloodPipeline] ALERT: {len(results)} điểm ngập nghiêm trọng"
            msg["From"]    = sender
            msg["To"]      = receiver
            msg.set_content(
                f"Phát hiện {len(results)} ảnh có mức ngập nghiêm trọng.\n"
                f"Vui lòng kiểm tra dashboard.\n"
            )

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
            log.info(f"  [Alert] Email đã gửi đến {receiver}")
        except Exception as exc:
            log.warning(f"  [Alert] Gửi email thất bại: {exc}")
