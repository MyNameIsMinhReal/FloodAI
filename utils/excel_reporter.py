# -*- coding: utf-8 -*-
"""
utils/excel_reporter.py
────────────────────────
Tạo file Excel báo cáo phân tích ảnh lũ.

Sheet 1: Summary   — thống kê tổng hợp + biểu đồ
Sheet 2: Chi tiết  — từng ảnh, thumbnail, kết quả đo
Sheet 3: Google Maps — hyperlink mở bản đồ (GPS từ EXIF hoặc tìm kiếm)

Cài: pip install openpyxl Pillow
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image as PILImage

from .constants import FLOOD_LEVEL_KNEE, FLOOD_LEVEL_HIP, FLOOD_LEVEL_CHEST, FLOOD_LEVEL_COMPLETE

from .constants import REPORT_XLSX

log = logging.getLogger(__name__)

LEVEL_COLORS = {
    "NO_FLOOD":  "C8E6C9",
    "PUDDLE":    "A5D6A7",
    "ANKLE":     "FFF9C4",
    "KNEE":      "FFE082",
    "WAIST":     "FFCC80",
    "CHEST":     "FF8A65",
    "SUBMERGED": "EF9A9A",
    "UNKNOWN":   "E0E0E0",
}
LEVEL_VN = {
    "NO_FLOOD":  "Không ngập",
    "PUDDLE":    "Vũng nước nhỏ (<15cm)",
    "ANKLE":     "Ngập mắt cá (15-40cm)",
    "KNEE":      FLOOD_LEVEL_KNEE,
    "WAIST":     FLOOD_LEVEL_HIP,
    "CHEST":     FLOOD_LEVEL_CHEST,
    "SUBMERGED": FLOOD_LEVEL_COMPLETE,
    "UNKNOWN":   "Không xác định",
}


class ExcelReporter:

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    def generate(self, results: dict) -> Optional[Path]:
        try:
            import openpyxl
        except ImportError:
            log.warning("openpyxl chưa cài: pip install openpyxl")
            return None

        wb   = openpyxl.Workbook()
        dest = self.output_dir / REPORT_XLSX

        ws1 = wb.active
        if ws1 is None:
            raise RuntimeError("Workbook không có worksheet mặc định")
        ws1.title = "Summary"
        self._build_summary(ws1, results, openpyxl)

        ws2 = wb.create_sheet("Chi tiết ảnh")
        self._build_detail(ws2, results, openpyxl)

        ws3 = wb.create_sheet("Google Maps")
        self._build_maps(ws3, results, openpyxl)

        wb.save(str(dest))
        log.info(f"  Excel → {dest}")
        return dest

    # ──────────────────────────────────────────────────────────────────
    def _build_summary(self, ws, results, openpyxl):
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter

        depth_data = results.get("depth_data", [])
        query      = results.get("query", "")
        run_id     = results.get("run_id", "")

        def fill(color): return PatternFill("solid", fgColor=color)
        def font(size=10, bold=False, color="000000", italic=False):
            return Font(name="Arial", size=size, bold=bold,
                        color=color, italic=italic)
        def align(h="center", v="center", wrap=False):
            return Alignment(horizontal=h, vertical=v, wrap_text=wrap)

        # ── Tiêu đề ──────────────────────────────────────────────────
        ws.merge_cells("A1:H1")
        ws["A1"] = "BÁO CÁO PHÂN TÍCH ẢNH LŨ LỤT"
        ws["A1"].font      = font(18, bold=True, color="FFFFFF")
        ws["A1"].fill      = fill("1565C0")
        ws["A1"].alignment = align()
        ws.row_dimensions[1].height = 38

        ws.merge_cells("A2:H2")
        ws["A2"] = (f"Từ khóa: {query}  |  Run: {run_id}  |  "
                    f"Tạo lúc: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        ws["A2"].font      = font(10, italic=True, color="546E7A")
        ws["A2"].alignment = align()
        ws.row_dimensions[2].height = 22
        ws.row_dimensions[3].height = 8

        # ── Stats cards ───────────────────────────────────────────────
        stats = [
            ("Tổng crawl",    len(results.get("raw", [])),       "1E88E5"),
            ("Sau lọc",       len(results.get("filtered", [])),  "43A047"),
            ("Đã phân tích",  len(depth_data),                   "FB8C00"),
            ("TB nước (cm)",  self._avg_water(depth_data),       "E53935"),
            ("Max nước (cm)", self._max_water(depth_data),       "8E24AA"),
        ]
        for i, (label, val, color) in enumerate(stats, 1):
            ws.cell(row=4, column=i, value=val).font      = font(22, bold=True, color=color)
            ws.cell(row=4, column=i).alignment             = align()
            ws.cell(row=5, column=i, value=label).font    = font(9, color="78909C")
            ws.cell(row=5, column=i).alignment             = align()
            ws.column_dimensions[get_column_letter(i)].width = 18
        ws.row_dimensions[4].height = 34
        ws.row_dimensions[5].height = 18
        ws.row_dimensions[6].height = 8

        # ── Bảng phân bố ─────────────────────────────────────────────
        ws.merge_cells("A7:E7")
        ws["A7"] = "PHÂN BỐ MỨC NGẬP"
        ws["A7"].font      = font(12, bold=True, color="FFFFFF")
        ws["A7"].fill      = fill("37474F")
        ws["A7"].alignment = align("left")
        ws.row_dimensions[7].height = 24

        hdrs = ["Cấp độ", "Mô tả", "Số lượng", "Tỉ lệ", "Màu"]
        hwidths = [14, 30, 12, 12, 8]
        for i, (h, w) in enumerate(zip(hdrs, hwidths), 1):
            c = ws.cell(row=8, column=i, value=h)
            c.font      = font(10, bold=True, color="FFFFFF")
            c.fill      = fill("546E7A")
            c.alignment = align()
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.row_dimensions[8].height = 20

        counts = self._count_levels(depth_data)
        total  = sum(counts.values()) or 1
        row = 9
        for level, hex_color in LEVEL_COLORS.items():
            count = counts.get(level, 0)
            if count == 0 and level not in ("NO_FLOOD", "UNKNOWN"):
                continue
            pct = f"{count / total * 100:.1f}%"
            bg  = "F5F5F5" if row % 2 == 0 else "FFFFFF"

            for ci, val in enumerate([level, LEVEL_VN.get(level,""), count, pct, ""], 1):
                c = ws.cell(row=row, column=ci, value=val)
                c.alignment = align()
                c.fill      = fill(hex_color if ci == 5 else bg)
                c.font      = font(10, bold=(ci == 1))
            ws.row_dimensions[row].height = 18
            row += 1

        # ── Biểu đồ ──────────────────────────────────────────────────
        chart = openpyxl.chart.BarChart()
        chart.type           = "col"
        chart.title          = "Phân bố mức ngập"
        chart.y_axis.title   = "Số ảnh"
        chart.x_axis.title   = "Cấp độ"
        chart.style          = 10
        chart.width          = 16
        chart.height         = 11

        data_ref = openpyxl.chart.Reference(ws, min_col=3, min_row=8, max_row=row-1)
        cats     = openpyxl.chart.Reference(ws, min_col=1, min_row=9, max_row=row-1)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats)
        ws.add_chart(chart, "G7")

    # ──────────────────────────────────────────────────────────────────
    def _build_detail(self, ws, results, openpyxl):
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter
        from openpyxl.drawing.image import Image as XLImage

        depth_data = results.get("depth_data", [])

        hdrs = [
            ("STT",       5),  ("Tên file",   28), ("Mức ngập",  14),
            ("Nước (cm)", 12), ("Khoảng",     16), ("Tin cậy",   11),
            # [v4] Scene quality columns
            ("Chất lượng", 12), ("Đêm", 6), ("Cần review", 11),
            ("Vật thể",   22), ("Ghi chú",    40), ("Ảnh",       16),
        ]

        ws.row_dimensions[1].height = 26
        for ci, (name, width) in enumerate(hdrs, 1):
            c = ws.cell(row=1, column=ci, value=name)
            c.font      = Font(name="Arial", size=11, bold=True, color="FFFFFF")
            c.fill      = PatternFill("solid", fgColor="0D47A1")
            c.alignment = Alignment(horizontal="center", vertical="center")
            ws.column_dimensions[get_column_letter(ci)].width = width
        ws.freeze_panes = "A2"

        for idx, r in enumerate(depth_data, 1):
            d   = r.__dict__ if hasattr(r, "__dict__") else (r if isinstance(r, dict) else {})
            row = idx + 1
            ws.row_dimensions[row].height = 58

            level = d.get("flood_level", "UNKNOWN")
            lc    = LEVEL_COLORS.get(level, "E0E0E0")
            objs  = ", ".join(dict.fromkeys(
                o.get("class_name","") for o in d.get("detected_objects", [])
            )) or "—"

            row_data = [
                idx,
                d.get("filename", ""),
                level,
                d.get("water_height_cm", 0),
                d.get("water_height_range", ""),
                f"{d.get('confidence', 0)*100:.0f}%",
                # [v4] Scene quality data
                f"{float(d.get('scene_score', 0) or 0)*100:.0f}%",
                "Đêm" if d.get("is_night") else "",
                "Có" if d.get("needs_review") else "",
                objs,
                (d.get("notes", "") or "")[:150],
            ]
            for ci, val in enumerate(row_data, 1):
                c = ws.cell(row=row, column=ci, value=val)
                c.font      = Font(name="Arial", size=9)
                c.alignment = Alignment(
                    horizontal="center", vertical="center",
                    wrap_text=(ci == 8),
                )
                if ci == 3:
                    c.fill = PatternFill("solid", fgColor=lc)
                    c.font = Font(name="Arial", size=9, bold=True)
                if ci == 4:
                    c.fill = PatternFill("solid", fgColor=self._cm_color(float(val or 0)))

            # Thumbnail
            ov = d.get("overlay_path", "")
            if ov and Path(ov).exists():
                try:
                    buf = self._thumb(ov, 110, 80)
                    if buf:
                        img = XLImage(buf)
                        img.width = 110; img.height = 80
                        ws.add_image(img, f"{get_column_letter(9)}{row}")
                except Exception as e:
                    log.debug(f"  thumbnail: {e}")

    # ──────────────────────────────────────────────────────────────────
    def _build_maps(self, ws, results, openpyxl):
        """
        Sheet Google Maps với hyperlink ĐÚNG.

        LỖI CŨ: set c.hyperlink = url  rồi c.value = "Mở bản đồ"
                → openpyxl ghi đè display text = url, link bị mất

        FIX: Dùng openpyxl.styles.Font với underline="single" + color xanh
             Giữ c.value = url_text, KHÔNG ghi đè sau khi set hyperlink.
             Hoặc dùng =HYPERLINK(url, text) formula — hoạt động 100%.
        """
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter

        depth_data = results.get("depth_data", [])
        query      = results.get("query", "lũ lụt Việt Nam")

        # Tiêu đề
        ws.merge_cells("A1:G1")
        ws["A1"] = "GOOGLE MAPS — VỊ TRÍ ẢNH"
        ws["A1"].font      = Font(name="Arial", size=14, bold=True, color="FFFFFF")
        ws["A1"].fill      = PatternFill("solid", fgColor="1B5E20")
        ws["A1"].alignment = Alignment(horizontal="center")
        ws.row_dimensions[1].height = 30

        hdrs   = ["STT", "Tên file", "Mức ngập", "Nước (cm)", "GPS", "Tọa độ", "Link Google Maps"]
        widths = [6,      28,         14,          12,          8,     20,        50]
        for ci, (h, w) in enumerate(zip(hdrs, widths), 1):
            c = ws.cell(row=2, column=ci, value=h)
            c.font      = Font(name="Arial", size=10, bold=True, color="FFFFFF")
            c.fill      = PatternFill("solid", fgColor="2E7D32")
            c.alignment = Alignment(horizontal="center")
            ws.column_dimensions[get_column_letter(ci)].width = w
        ws.row_dimensions[2].height = 22
        ws.freeze_panes = "A3"

        link_font = Font(name="Arial", size=9, color="1565C0", underline="single")
        no_gps_font = Font(name="Arial", size=9, color="9E9E9E", italic=True)

        for idx, r in enumerate(depth_data, 1):
            d        = r.__dict__ if hasattr(r, "__dict__") else (r if isinstance(r, dict) else {})
            row      = idx + 2
            ws.row_dimensions[row].height = 20

            filename  = d.get("filename", "")
            level     = d.get("flood_level", "UNKNOWN")
            water_cm  = d.get("water_height_cm", 0)
            orig_path = d.get("original_path", "")

            lat, lon, gps_source = self._extract_gps(orig_path)

            if lat is not None and lon is not None:
                has_gps  = True
                coord    = f"{lat:.6f}, {lon:.6f}"
                maps_url = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
                gps_icon = "📍"
            else:
                has_gps  = False
                coord    = "—"
                # Tìm kiếm theo query — URL encode đúng cách
                import urllib.parse
                q_enc    = urllib.parse.quote(query)
                maps_url = f"https://www.google.com/maps/search/{q_enc}"
                gps_icon = "🔍"

            # Cột 1-6: dữ liệu
            row_data = [idx, filename, level, water_cm, gps_icon, coord]
            for ci, val in enumerate(row_data, 1):
                c = ws.cell(row=row, column=ci, value=val)
                c.font      = Font(name="Arial", size=9)
                c.alignment = Alignment(horizontal="center", vertical="center")
                if ci == 3:
                    c.fill = PatternFill("solid", fgColor=LEVEL_COLORS.get(level, "E0E0E0"))
                    c.font = Font(name="Arial", size=9, bold=True)

            # Cột 7: hyperlink — CÁCH ĐÚNG
            # Dùng công thức =HYPERLINK(url, text) thay vì openpyxl hyperlink API
            # Công thức này hoạt động trong mọi phiên bản Excel/LibreOffice
            if has_gps:
                display_text = f"📍 {lat:.4f}, {lon:.4f}"
            else:
                display_text = "🔍 Tìm theo query"

            c_link = ws.cell(row=row, column=7)
            # =HYPERLINK(url, text) — cách đáng tin cậy nhất
            safe_url = maps_url.replace('"', '%22')
            c_link.value     = f'=HYPERLINK("{safe_url}","{display_text}")'
            c_link.font      = link_font if has_gps else no_gps_font
            c_link.alignment = Alignment(horizontal="left", vertical="center")

        # Ghi chú cuối
        note_row = len(depth_data) + 4
        ws.merge_cells(f"A{note_row}:G{note_row}")
        note = ws.cell(row=note_row, column=1,
                       value="📌 GPS lấy từ EXIF metadata. "
                             "📍 = tọa độ chính xác.  🔍 = tìm kiếm theo từ khóa.")
        note.font      = Font(name="Arial", size=8, italic=True, color="78909C")
        note.alignment = Alignment(horizontal="left")

    # ──────────────────────────────────────────────────────────────────
    def _extract_gps(self, image_path: str):
        """
        Trích xuất GPS từ EXIF.
        Trả về (lat, lon, source) hoặc (None, None, "none").
        """
        if not image_path or not Path(image_path).exists():
            return None, None, "no_file"
        try:
            from PIL import Image as PILImg
            from PIL.ExifTags import TAGS, GPSTAGS

            img  = PILImg.open(image_path)
            exif = img.getexif()
            if not exif:
                return None, None, "no_exif"

            gps = {}
            for tag_id, val in exif.items():
                if TAGS.get(tag_id) == "GPSInfo":
                    for k, v in val.items():
                        gps[GPSTAGS.get(k, k)] = v
            if not gps:
                return None, None, "no_gps_tag"

            def to_deg(coord, ref):
                if not coord or not ref: return None
                try:
                    d, m, s = float(coord[0]), float(coord[1]), float(coord[2])
                    v = d + m/60 + s/3600
                    return -v if ref in ("S","W") else v
                except Exception:
                    return None

            lat = to_deg(gps.get("GPSLatitude"),  gps.get("GPSLatitudeRef"))
            lon = to_deg(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))

            if lat is None or lon is None:
                return None, None, "incomplete"
            return lat, lon, "exif"

        except Exception as e:
            log.debug(f"  GPS extract: {e}")
            return None, None, "error"

    # ──────────────────────────────────────────────────────────────────
    def _thumb(self, path: str, w: int, h: int) -> io.BytesIO:
        img = PILImage.open(path)
        img.thumbnail((w, h), PILImage.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=72)
        buf.seek(0)
        return buf

    def _cm_color(self, cm: float) -> str:
        if cm == 0:  return "C8E6C9"
        if cm < 15:  return "A5D6A7"
        if cm < 40:  return "FFF9C4"
        if cm < 70:  return "FFE082"
        if cm < 120: return "FFCC80"
        if cm < 200: return "FF8A65"
        return "EF9A9A"

    def _avg_water(self, depth_data) -> str:
        vals = [self._d(r).get("water_height_cm", 0) for r in depth_data]
        return f"{float(np.mean(vals)):.0f}" if vals else "0"

    def _max_water(self, depth_data) -> str:
        vals = [self._d(r).get("water_height_cm", 0) for r in depth_data]
        return f"{float(max(vals)):.0f}" if vals else "0"

    def _count_levels(self, depth_data) -> dict:
        out = {}
        for r in depth_data:
            lvl = self._d(r).get("flood_level", "UNKNOWN")
            out[lvl] = out.get(lvl, 0) + 1
        return out

    def _d(self, r) -> dict:
        return r.__dict__ if hasattr(r, "__dict__") else (r if isinstance(r, dict) else {})
