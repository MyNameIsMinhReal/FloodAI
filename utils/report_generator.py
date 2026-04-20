# -*- coding: utf-8 -*-
"""
utils/report_generator_v2.py
-----------------------------
Tao bao cao HTML + CSV voi thong tin do muc nuoc chi tiet
tu ReferenceEstimator (YOLO + Depth Anything V2).
"""

import csv
import json
import logging
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, List

from .constants import REPORT_CSV, REPORT_HTML

log = logging.getLogger(__name__)

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #0a0f1e;
  color: #e2e8f0;
  padding: 24px;
  min-height: 100vh;
}
h1 { color: #38bdf8; text-align: center; font-size: 1.9em; margin-bottom: 4px; }
.subtitle { text-align: center; color: #64748b; font-size: .9em; margin-bottom: 28px; }
.stats {
  display: flex; flex-wrap: wrap; gap: 14px;
  justify-content: center; margin-bottom: 32px;
}
.stat {
  background: #111827; border: 1px solid #1e3a5f;
  border-radius: 14px; padding: 14px 22px;
  text-align: center; min-width: 110px;
}
.stat .val { font-size: 2em; font-weight: 700; line-height: 1.1; }
.stat .lbl { font-size: .78em; color: #64748b; margin-top: 3px; }
.section-title {
  font-size: 1.1em; font-weight: 600; color: #94a3b8;
  margin: 28px 0 12px; padding-left: 6px;
  border-left: 3px solid #0284c7;
}
table {
  width: 100%; border-collapse: collapse;
  background: #111827; border-radius: 12px;
  overflow: hidden; font-size: .86em; margin-bottom: 36px;
}
thead th {
  background: #0c4a6e; color: #bae6fd;
  padding: 11px 10px; text-align: left;
  font-weight: 600; letter-spacing: .3px;
}
tbody td {
  padding: 9px 10px; border-bottom: 1px solid #1e293b;
  vertical-align: middle;
}
tbody tr:hover td { background: #162032; }
img.thumb {
  width: 90px; height: 65px; object-fit: cover;
  border-radius: 8px; display: block;
}
.badge {
  display: inline-block; padding: 3px 10px;
  border-radius: 20px; font-size: .82em;
  font-weight: 700; white-space: nowrap;
}
.PUDDLE    { background: #052e16; color: #4ade80; }
.ANKLE     { background: #052e16; color: #86efac; }
.KNEE      { background: #422006; color: #fbbf24; }
.WAIST     { background: #431407; color: #fb923c; }
.CHEST     { background: #450a0a; color: #f87171; }
.SUBMERGED { background: #3b0764; color: #e879f9; }
.UNKNOWN   { background: #1e293b; color: #94a3b8; }
.bar-bg { background: #1e293b; border-radius: 4px; height: 8px; width: 100px; display: inline-block; vertical-align: middle; }
.bar    { height: 8px; border-radius: 4px; display: block; }
.obj-chip {
  display: inline-block; margin: 2px 3px 2px 0;
  background: #1e3a5f; color: #7dd3fc;
  border-radius: 10px; padding: 2px 8px; font-size: .78em;
}
.note { color: #64748b; font-size: .82em; max-width: 260px; line-height: 1.4; }
footer { text-align: center; color: #334155; font-size: .8em; margin-top: 40px; }
"""

LEVEL_ORDER = ["PUDDLE", "ANKLE", "KNEE", "WAIST", "CHEST", "SUBMERGED", "UNKNOWN"]
LEVEL_COLORS = {
    "PUDDLE": "#4ade80", "ANKLE": "#86efac", "KNEE": "#fbbf24",
    "WAIST": "#fb923c",  "CHEST": "#f87171", "SUBMERGED": "#e879f9",
    "UNKNOWN": "#64748b",
}
BAR_COLORS = {
    "PUDDLE": "#22c55e", "ANKLE": "#4ade80", "KNEE": "#f59e0b",
    "WAIST": "#f97316",  "CHEST": "#ef4444", "SUBMERGED": "#a855f7",
    "UNKNOWN": "#475569",
}


class ReportGeneratorV2:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, results: dict) -> Tuple[Path, Path]:
        csv_path  = self._write_csv(results)
        html_path = self._write_html(results)
        return csv_path, html_path

    # -------------------------------------------------------------
    def _write_csv(self, results: dict) -> Path:
        path = self.output_dir / REPORT_CSV
        depth_data = results.get("depth_data", [])

        fields = [
            "filename", "flood_level", "flood_level_desc",
            "water_height_cm", "water_height_range",
            "confidence", "depth_flood_pct",
            "detected_objects_count", "detected_object_types",
            "notes", "original_path", "overlay_path", "depth_map_path",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in depth_data:
                d = r.__dict__ if hasattr(r, "__dict__") else r
                objs = d.get("detected_objects", [])
                writer.writerow({
                    "filename":              d.get("filename", ""),
                    "flood_level":           d.get("flood_level", ""),
                    "flood_level_desc":      d.get("flood_level_desc", ""),
                    "water_height_cm":       d.get("water_height_cm", ""),
                    "water_height_range":    d.get("water_height_range", ""),
                    "confidence":            d.get("confidence", ""),
                    "depth_flood_pct":       d.get("depth_flood_pct", ""),
                    "detected_objects_count":len(objs),
                    "detected_object_types": ", ".join(
                        set(o.get("class_name","") for o in objs)
                    ),
                    "notes":       d.get("notes", ""),
                    "original_path":   d.get("original_path", ""),
                    "overlay_path":    d.get("overlay_path", ""),
                    "depth_map_path":  d.get("depth_map_path", ""),
                })

        if not depth_data:
            # Chi ghi danh sach anh filtered neu khong co depth data
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "path"], extrasaction="ignore")
                writer.writeheader()
                for p in results.get("filtered", []):
                    writer.writerow({"filename": Path(p).name, "path": str(p)})

        return path

    # -------------------------------------------------------------
    def _write_html(self, results: dict) -> Path:
        path       = self.output_dir / REPORT_HTML
        depth_data = results.get("depth_data", [])
        run_id     = results.get("run_id", "")
        query      = results.get("query", "")
        sources    = ", ".join(results.get("sources", []))
        now        = datetime.now().strftime("%Y-%m-%d %H:%M")

        # -- Stats ---------------------------------------------------
        level_counts = {l: 0 for l in LEVEL_ORDER}
        total_water_cm = []
        all_obj_types  = {}

        for r in depth_data:
            d = r.__dict__ if hasattr(r, "__dict__") else r
            lvl = d.get("flood_level", "UNKNOWN")
            if lvl in level_counts:
                level_counts[lvl] += 1
            cm = d.get("water_height_cm", 0)
            if cm:
                total_water_cm.append(cm)
            for obj in d.get("detected_objects", []):
                cn = obj.get("class_name", "")
                all_obj_types[cn] = all_obj_types.get(cn, 0) + 1

        avg_water = sum(total_water_cm) / len(total_water_cm) if total_water_cm else 0
        max_water = max(total_water_cm) if total_water_cm else 0

        # -- Stats HTML ----------------------------------------------
        stats_html = ""
        color_map  = {"crawled": "#38bdf8", "filtered": "#a78bfa",
                      "analyzed": "#34d399"}
        stats_html += self._stat_card(len(results.get("raw",[])),      "Crawled",   "#38bdf8")
        stats_html += self._stat_card(len(results.get("filtered",[])), "Filtered",  "#a78bfa")
        stats_html += self._stat_card(len(depth_data),                 "Analyzed",  "#34d399")
        stats_html += self._stat_card(f"{avg_water:.0f} cm",           "Avg Depth", "#f59e0b")
        stats_html += self._stat_card(f"{max_water:.0f} cm",           "Max Depth", "#ef4444")
        for lvl in LEVEL_ORDER:
            if level_counts[lvl]:
                stats_html += self._stat_card(
                    level_counts[lvl], lvl, LEVEL_COLORS.get(lvl, "#fff")
                )

        # -- Table rows ----------------------------------------------
        rows_html = ""
        for r in sorted(
            depth_data,
            key=lambda x: -(x.__dict__ if hasattr(x,"__dict__") else x).get("water_height_cm", 0)
        ):
            d    = r.__dict__ if hasattr(r, "__dict__") else r
            lvl  = d.get("flood_level", "UNKNOWN")
            cm   = d.get("water_height_cm", 0)
            rng  = d.get("water_height_range", "")
            conf = d.get("confidence", 0)
            desc = d.get("flood_level_desc", "")
            note = d.get("notes", "")
            dfp  = d.get("depth_flood_pct", 0)
            objs = d.get("detected_objects", [])
            name = d.get("filename", "")

            overlay_src = d.get("overlay_path", "")
            depth_src   = d.get("depth_map_path", "")

            # Object chips
            obj_chips = "".join(
                f'<span class="obj-chip">{o.get("class_name","")}'
                f' ({o.get("water_height_cm",0):.0f}cm)</span>'
                for o in objs
            ) or '<span style="color:#475569;font-size:.8em">none detected</span>'

            bar_pct   = min(cm / 250 * 100, 100)
            bar_color = BAR_COLORS.get(lvl, "#475569")

            rows_html += f"""
            <tr>
              <td>
                <a href="{overlay_src}" target="_blank">
                  <img class="thumb" src="{overlay_src}"
                       onerror="this.style.opacity='.3'"
                       title="Click to view overlay">
                </a>
              </td>
              <td style="font-size:.82em;color:#94a3b8;word-break:break-all">{name}</td>
              <td>
                <span class="badge {lvl}">{lvl}</span>
                <div style="font-size:.75em;color:#64748b;margin-top:3px">{desc}</div>
              </td>
              <td>
                <strong style="color:{bar_color}">{cm:.0f} cm</strong>
                <div style="font-size:.75em;color:#64748b">{rng}</div>
                <span class="bar-bg"><span class="bar" style="width:{bar_pct:.0f}%;background:{bar_color}"></span></span>
              </td>
              <td style="color:#94a3b8">{conf*100:.0f}%</td>
              <td>{obj_chips}</td>
              <td>
                <a href="{depth_src}" target="_blank" style="color:#38bdf8;font-size:.8em">depth map</a>
              </td>
              <td class="note">{note}</td>
            </tr>"""

        if not rows_html:
            rows_html = '<tr><td colspan="8" style="text-align:center;color:#475569;padding:30px">No depth analysis data available</td></tr>'

        html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flood Analysis - {run_id}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>đ Flood Image Analysis Report</h1>
  <p class="subtitle">
    Query: <strong style="color:#e2e8f0">{query}</strong>
    &nbsp;|&nbsp; Sources: {sources}
    &nbsp;|&nbsp; Run: {run_id}
    &nbsp;|&nbsp; {now}
  </p>

  <div class="stats">{stats_html}</div>

  <div class="section-title">đ Water Level Analysis (sorted by depth)</div>
  <table>
    <thead>
      <tr>
        <th>Overlay</th>
        <th>Filename</th>
        <th>Flood Level</th>
        <th>Water Depth</th>
        <th>Conf.</th>
        <th>Reference Objects (measured)</th>
        <th>Depth Map</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>

  <footer>
    Generated by Flood Image Analysis Pipeline &nbsp;|&nbsp;
    Depth Anything V2 + YOLOv8 Reference Method &nbsp;|&nbsp; {now}
  </footer>
</body>
</html>"""

        path.write_text(html, encoding="utf-8")
        return path

    def _stat_card(self, val, label: str, color: str) -> str:
        return f"""
        <div class="stat">
          <div class="val" style="color:{color}">{val}</div>
          <div class="lbl">{label}</div>
        </div>"""
