# -*- coding: utf-8 -*-
"""
Stage 5 — Post-process
=======================
Các bước xử lý sau depth estimation:
  - Route prediction (nếu có GPS)
  - Alert system (nếu flood > threshold)
  - Temporal tracking (nếu có nhiều ảnh cùng location)
  - Disagreement flagging → review queue cho active learner [v4]
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

            output_dir = getattr(state, "output_dir", None)
            routes_dir = output_dir / "routes" if output_dir is not None else "routes"
            predictor = FloodRoutePredictor(output_dir=str(routes_dir))
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

    # ── [v4] Disagreement → review queue cho active learner ────────────────

    def check_disagreements(self, state: "PipelineState") -> None:
        """
        Quét kết quả depth, tìm các ảnh có bất đồng giữa phương pháp đo →
        đưa vào review_queue (ActiveLearnerV2) để con người xác minh.

        Ưu tiên review: ảnh ngập sâu (>30cm) + bất đồng lớn → quan trọng
        hơn ảnh nông có bất đồng nhỏ.
        """
        measurement = self.cfg.get("measurement", {})
        if not measurement.get("flag_disagreement", True):
            return

        try:
            from learning.active_learner import ActiveLearnerV2, ReviewCase
        except ImportError:
            log.debug("  [Disagreement] ActiveLearnerV2 không sẵn sàng → bỏ qua")
            return

        reviewer = ActiveLearnerV2(self.cfg)
        n_added = 0

        for dr in state.depth_results:
            needs_review = False
            review_reason = ""
            water_cm = getattr(dr, "water_height_cm", 0.0) or 0.0

            # ── Case 1: Có FinalMeasurement với needs_review flag ─────
            fm = getattr(dr, "final_measurement", None)
            if fm is not None:
                if getattr(fm, "needs_review", False):
                    needs_review = True
                    review_reason = getattr(fm, "review_reason", "method_disagreement")

            # ── Case 2: ReferenceFloodResult (không có FinalMeasurement)
            #    → dùng standalone assess_disagreement ─────────────────
            if not needs_review and water_cm > 0:
                try:
                    from depth_analysis.measurement_engine import assess_disagreement
                    flagged, dev_pct, reason = assess_disagreement(dr)
                    if flagged:
                        needs_review = True
                        review_reason = reason or "method_disagreement"
                except ImportError:
                    pass

            if not needs_review:
                continue

            img_path = getattr(dr, "original_path", "") or ""
            case = ReviewCase(
                timestamp="",
                image_path=img_path,
                predicted_depth=water_cm,
                predicted_level=getattr(dr, "flood_level", "UNKNOWN"),
                confidence=getattr(dr, "confidence", 0.0),
                review_reason=review_reason,
                priority=2 if water_cm > 30 else 1,
                score=60.0 + min(40.0, water_cm / 2.0),
            )
            reviewer.add_to_queue(case)
            n_added += 1
            log.debug(
                f"  [Disagreement] → review queue: {img_path} "
                f"({water_cm:.0f}cm, reason={review_reason})"
            )

        if n_added > 0:
            log.info(f"  [Disagreement] Đã đưa {n_added} ảnh vào review queue")
        reviewer.close()

    # ── Email helpers ──────────────────────────────────────────────────────

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
