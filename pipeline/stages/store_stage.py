# -*- coding: utf-8 -*-
"""
Stage 6 — Store
================
Lưu kết quả: report (HTML/CSV/Excel), Google Drive upload, pipeline summary.
"""
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from utils.constants import PIPELINE_SUMMARY_JSON

log = logging.getLogger("pipeline.store")

if TYPE_CHECKING:
    from pipeline.orchestrator import PipelineState


class StoreStage:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def save_reports(self, state: "PipelineState") -> None:
        """Tạo báo cáo HTML + CSV + Excel và lưu summary JSON."""
        self._save_summary(state)
        self._generate_reports(state)

    def upload_drive(self, state: "PipelineState") -> Optional[str]:
        """Upload toàn bộ output lên Google Drive."""
        from uploader.drive_uploader import DriveUploader
        from utils.constants import DRIVE_UPLOAD_EXTENSIONS
        try:
            output_dir = state.output_dir
            if output_dir is None:
                log.error("  [Drive] Không thể upload: chưa có thư mục output")
                return None
            folder_name = f"{self.cfg.get('drive_folder', 'FloodAnalysis')}/{state.run_id}"
            fid = DriveUploader().upload_folder(
                local_dir=output_dir,
                folder_name=folder_name,
                extensions=DRIVE_UPLOAD_EXTENSIONS,
            )
            log.info(f"  [Drive] https://drive.google.com/drive/folders/{fid}")
            return fid
        except Exception as exc:
            log.error(f"  [Drive] Upload thất bại: {exc}")
            return None

    def _generate_reports(self, state: "PipelineState") -> None:
        output_dir = state.output_dir
        if output_dir is None:
            log.warning("  [Store] Không thể tạo báo cáo: chưa có thư mục output")
            return
        results_dict = state.to_dict()
        try:
            from utils.report_generator import ReportGeneratorV2
            csv_p, html_p = ReportGeneratorV2(output_dir=output_dir).generate(results_dict)
            log.info(f"  [Store] HTML → {html_p}")
            log.info(f"  [Store] CSV  → {csv_p}")
        except Exception as exc:
            log.warning(f"  [Store] Report lỗi: {exc}")

        try:
            from utils.excel_reporter import ExcelReporter
            xlsx_p = ExcelReporter(output_dir=output_dir).generate(results_dict)
            if xlsx_p:
                log.info(f"  [Store] XLSX → {xlsx_p}")
        except Exception as exc:
            log.warning(f"  [Store] Excel report lỗi: {exc}")

    def _save_summary(self, state: "PipelineState") -> None:
        from pipeline.orchestrator import PipelineState

        def _count_levels(depth_data) -> dict:
            counts: dict = {}
            for r in depth_data:
                lvl = getattr(r, "flood_level", "?")
                counts[lvl] = counts.get(lvl, 0) + 1
            return counts

        summary = {
            "run_id":          state.run_id,
            "input_dir":       str(state.input_dir or ""),
            "total_input":     len(state.input_images),
            "total_analyzed":  len(state.depth_results),
            "drive_folder":    state.drive_folder_id or "N/A",
            "flood_summary":   _count_levels(state.depth_results),
            "timings":         state.timings,
            "errors":          state.errors,
        }
        try:
            output_dir = state.output_dir
            if output_dir is None:
                log.warning("  [Store] Không ghi được summary: chưa có thư mục output")
                return
            out = output_dir / PIPELINE_SUMMARY_JSON
            out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            log.info(f"  [Store] Summary → {out}")
        except Exception as exc:
            log.warning(f"  [Store] Không ghi được summary: {exc}")
