# -*- coding: utf-8 -*-
"""
utils/article_generator.py
===========================
Sinh bài viết tiếng Việt tự động từ kết quả pipeline.

Thay vì hiển thị `{"level":"knee","depth_cm":52,"confidence":0.81}`,
sinh thành bài báo ngắn theo mẫu:

    Tiêu đề: Ghi nhận ngập ~50cm tại khu vực X
    Nội dung: Vào lúc 08:35, hệ thống ghi nhận mực nước ước tính...
    Khuyến cáo: Người dân nên hạn chế di chuyển...

Dùng:
    gen = ArticleGenerator()
    article = gen.from_result(result, location="Đường Nguyễn Trãi", time="08:35")
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("utils.article_gen")

# ── Level metadata ─────────────────────────────────────────────────────────────
LEVEL_META = {
    "dry": {
        "label":      "Không ngập",
        "label_short":"Khô ráo",
        "depth":      "0 cm",
        "severity":   "ok",
        "icon":       "✅",
        "advice":     "Khu vực không có dấu hiệu ngập nước.",
        "recommend":  "Lưu thông bình thường.",
    },
    "ankle": {
        "label":      "Ngập mắt cá chân",
        "label_short":"Ngập nhẹ",
        "depth":      "15–25 cm",
        "severity":   "low",
        "icon":       "🟡",
        "advice":     "Nước ngập nhẹ, khoảng ngang mắt cá chân.",
        "recommend":  "Người đi bộ nên chú ý. Xe máy đi qua được nhưng cẩn thận.",
    },
    "knee": {
        "label":      "Ngập đầu gối",
        "label_short":"Ngập vừa",
        "depth":      "40–60 cm",
        "severity":   "medium",
        "icon":       "🟠",
        "advice":     "Nước ngập đến khoảng đầu gối, gây khó khăn cho giao thông.",
        "recommend":  "Hạn chế di chuyển bằng xe máy. Ô tô thấp gầm nên tránh tuyến đường này.",
    },
    "waist": {
        "label":      "Ngập ngang eo",
        "label_short":"Ngập nặng",
        "depth":      "80–100 cm",
        "severity":   "high",
        "icon":       "🔴",
        "advice":     "Nước ngập sâu ngang eo, rất nguy hiểm cho người đi bộ và xe máy.",
        "recommend":  "Không nên đi qua khu vực này. Sơ tán nếu cần thiết.",
    },
    "chest": {
        "label":      "Ngập ngang ngực",
        "label_short":"Rất nguy hiểm",
        "depth":      "120–140 cm",
        "severity":   "critical",
        "icon":       "🚨",
        "advice":     "Nước ngập cực kỳ sâu, nguy hiểm đến tính mạng.",
        "recommend":  "Không được tiếp cận khu vực này. Liên hệ lực lượng cứu hộ ngay.",
    },
    "submerged": {
        "label":      "Ngập hoàn toàn",
        "label_short":"Cực kỳ nguy hiểm",
        "depth":      "> 150 cm",
        "severity":   "critical",
        "icon":       "🚨",
        "advice":     "Khu vực bị ngập hoàn toàn, cực kỳ nguy hiểm.",
        "recommend":  "Tuyệt đối không tiếp cận. Cần sơ tán khẩn cấp.",
    },
    "unknown": {
        "label":      "Chưa xác định",
        "label_short":"Cần xác minh",
        "depth":      "Không rõ",
        "severity":   "unknown",
        "icon":       "⚪",
        "advice":     "Mức ngập chưa được xác định rõ ràng.",
        "recommend":  "Cần thêm thông tin để xác minh.",
    },
}

CONFIDENCE_LABEL = {
    "HIGH":   "Cao",
    "MEDIUM": "Trung bình",
    "LOW":    "Cần xác minh",
}


class ArticleGenerator:
    """
    Sinh bài viết từ depth_result của pipeline.

    Dùng:
        gen = ArticleGenerator()
        article = gen.from_result(result, location="Đường Nguyễn Trãi")
        print(article["title"])    # tiêu đề
        print(article["body"])     # nội dung bài
        print(article["severity"]) # ok/low/medium/high/critical
    """

    def from_result(
        self,
        result: Any,
        location: str = "khu vực ghi nhận",
        time: Optional[str] = None,
        source: str = "Người dân gửi",
        run_id: str = "",
    ) -> Dict:
        """
        Sinh bài viết từ depth_result.

        Returns:
            dict với các key: id, title, headline, body, summary,
            recommendations, level_label, depth_range, severity,
            confidence_label, time, location, source, verified
        """
        if time is None:
            time = datetime.now().strftime("%H:%M")

        level      = _get(result, "flood_level", "unknown") or "unknown"
        depth_cm   = _get(result, "depth_cm", 0) or 0
        confidence = _get(result, "confidence", 0.5) or 0.5
        ci         = _get(result, "confidence_info", {}) or {}
        conf_label_raw = ci.get("label", "MEDIUM") if ci else "MEDIUM"

        meta       = LEVEL_META.get(level, LEVEL_META["unknown"])
        conf_label = CONFIDENCE_LABEL.get(conf_label_raw, "Trung bình")

        # Depth range (±10cm)
        if depth_cm > 0:
            low  = max(0, int(depth_cm) - 10)
            high = int(depth_cm) + 10
            depth_range = f"{low}–{high} cm"
        else:
            depth_range = meta["depth"]

        # Article ID
        article_id = run_id or hashlib.sha256(
            f"{location}{time}{level}".encode()
        ).hexdigest()[:10]

        title    = self._gen_title(level, depth_range, location)
        headline = self._gen_headline(level, depth_range, location)
        body     = self._gen_body(level, depth_range, location, time, source, depth_cm)
        summary  = self._gen_summary(level, depth_range, location, meta)
        warnings = (ci.get("warnings") or []) if ci else []

        return {
            "id":               article_id,
            "title":            title,
            "headline":         headline,
            "body":             body,
            "summary":          summary,
            "recommendations":  meta["recommend"],
            "advice":           meta["advice"],
            "warnings":         warnings,
            # Display fields
            "level":            level,
            "level_label":      meta["label"],
            "level_label_short":meta["label_short"],
            "depth_range":      depth_range,
            "severity":         meta["severity"],
            "icon":             meta["icon"],
            "confidence_label": conf_label,
            "confidence_raw":   round(confidence, 2),
            "time":             time,
            "location":         location,
            "source":           source,
            "verified":         conf_label_raw == "HIGH",
            "auto_generated":   True,
        }

    def from_pipeline_summary(self, summary_path: Path) -> List[Dict]:
        """Sinh danh sách bài viết từ pipeline_summary.json."""
        try:
            data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        except Exception:
            return []

        run_id = data.get("run_id", "")
        articles = []

        # Nếu có depth_results embedded (không phải lúc nào cũng có)
        depth_results = data.get("depth_results", [])
        if not depth_results:
            # Tạo 1 bài tổng hợp từ flood_summary
            flood_summary = data.get("flood_summary", {})
            if flood_summary:
                # Lấy mức ngập cao nhất
                level = _dominant_level(flood_summary)
                fake_result = {"flood_level": level, "depth_cm": 0, "confidence": 0.6}
                art = self.from_result(fake_result, location="khu vực phân tích", run_id=run_id)
                articles.append(art)
        else:
            for r in depth_results[:3]:  # tối đa 3 bài từ 1 run
                art = self.from_result(r, location=r.get("location", "khu vực"), run_id=run_id)
                articles.append(art)

        return articles

    # ── Private ───────────────────────────────────────────────────────────────

    def _gen_title(self, level: str, depth_range: str, location: str) -> str:
        meta = LEVEL_META.get(level, LEVEL_META["unknown"])
        if level in ("dry", "unknown"):
            return f"Cập nhật tình hình tại {location}"
        if level == "ankle":
            return f"Ghi nhận ngập nhẹ tại {location}"
        if level == "knee":
            return f"Nước ngập khoảng {depth_range} tại {location}"
        if level == "waist":
            return f"Cảnh báo: ngập sâu tại {location}"
        return f"🚨 Cảnh báo khẩn cấp: ngập nghiêm trọng tại {location}"

    def _gen_headline(self, level: str, depth_range: str, location: str) -> str:
        if level in ("dry", "unknown"):
            return f"Hệ thống ghi nhận thông tin tại khu vực {location}."
        return (
            f"Mực nước ước tính {depth_range} tại khu vực {location}. "
            f"{LEVEL_META.get(level, LEVEL_META['unknown'])['advice']}"
        )

    def _gen_body(
        self,
        level: str,
        depth_range: str,
        location: str,
        time: str,
        source: str,
        depth_cm: float,
    ) -> str:
        meta = LEVEL_META.get(level, LEVEL_META["unknown"])
        if level in ("dry", "unknown"):
            return (
                f"Vào lúc {time}, hệ thống phân tích hình ảnh từ {source.lower()} "
                f"tại khu vực {location}. Hiện chưa ghi nhận dấu hiệu ngập nước rõ ràng "
                f"hoặc thông tin đang được xác minh.\n\n"
                f"Hệ thống sẽ tiếp tục cập nhật khi có thông tin mới."
            )

        return (
            f"Vào lúc {time}, hệ thống ghi nhận hình ảnh ngập nước tại khu vực {location} "
            f"do {source.lower()} cung cấp. "
            f"Dựa trên phân tích hình ảnh hiện trường, mực nước được ước tính ở mức "
            f"**{meta['label'].lower()}**, khoảng {depth_range}.\n\n"
            f"{meta['advice']}\n\n"
            f"{meta['recommend']}"
        )

    def _gen_summary(self, level: str, depth_range: str, location: str, meta: dict) -> List[str]:
        points = []
        if level not in ("dry", "unknown"):
            points.append(f"Nước ngập khoảng {depth_range} tại {location}.")
        points.append(meta["advice"])
        if meta["recommend"] and meta["recommend"] != points[-1]:
            points.append(meta["recommend"])
        if level in ("waist", "chest", "submerged"):
            points.append("Liên hệ đường dây hỗ trợ thiên tai: 1800 599 926.")
        return points


def generate_article(
    result: Any,
    location: str = "khu vực",
    time: Optional[str] = None,
    source: str = "Người dân gửi",
    run_id: str = "",
) -> Dict:
    """Shorthand helper."""
    return ArticleGenerator().from_result(result, location, time, source, run_id)


def load_recent_articles(output_dir: Path = Path("output"), limit: int = 20) -> List[Dict]:
    """
    Quét output/ folder, đọc pipeline_summary.json, sinh danh sách bài viết gần đây.
    """
    gen   = ArticleGenerator()
    articles: List[Dict] = []

    if not output_dir.exists():
        return []

    runs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )[:limit * 2]

    for run_dir in runs:
        summary = run_dir / "pipeline_summary.json"
        if not summary.exists():
            continue
        arts = gen.from_pipeline_summary(summary)
        articles.extend(arts)
        if len(articles) >= limit:
            break

    return articles[:limit]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _dominant_level(flood_summary: dict) -> str:
    ORDER = ["submerged", "chest", "waist", "knee", "ankle", "dry", "unknown"]
    for lvl in ORDER:
        if flood_summary.get(lvl, 0) > 0:
            return lvl
    return "unknown"
