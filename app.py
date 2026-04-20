# -*- coding: utf-8 -*-
"""
FloodAI Web — Main Entry Point
5 pages: Home / News / Map / Chat / Dashboard (admin)
"""

import base64
import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.request
from functools import wraps
from pathlib import Path

import yaml

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

# ── Bootstrap ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("floodai_web")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "floodai-s3cr3t-k3y-change-in-prod")

# ── Config ─────────────────────────────────────────────────────────────────────
ADMIN_USER      = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS_HASH = hashlib.sha256(
    os.environ.get("ADMIN_PASS", "123456").encode()
).hexdigest()
GOOGLE_MAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
MERGED_MODEL    = str(BASE_DIR / "models" / "flood-agent-merged")
_IMG_PATTERNS   = ("*.jpg", "*.png")

# ── LLM Singleton ─────────────────────────────────────────────────────────────
_llm            = None
_llm_ready      = False
_llm_loading    = False
_llm_error: str = ""
_llm_lock       = threading.Lock()
_infer_lock     = threading.Lock()  # serialise inference (CPU model)


def _load_llm_bg():
    global _llm, _llm_ready, _llm_loading, _llm_error
    try:
        from learning.flood_llm import LLMEnhancer
        _llm = LLMEnhancer.from_lora_dir(MERGED_MODEL, max_new_tokens=512, temperature=0.7)
        _llm_ready = True
        log.info("[LLM] flood-agent-merged loaded ✓")
    except Exception as exc:
        _llm_error = str(exc)
        log.warning(f"[LLM] load failed: {exc}")
    finally:
        _llm_loading = False


def start_llm():
    global _llm_loading
    with _llm_lock:
        if _llm_loading or _llm_ready:
            return
        _llm_loading = True
    threading.Thread(target=_load_llm_bg, daemon=True).start()


def get_llm():
    return _llm if _llm_ready else None


def llm_infer(messages: list, max_new_tokens: int = None) -> str:
    """Thread-safe inference. Raises on error."""
    llm = get_llm()
    if llm is None:
        raise RuntimeError("Model chưa sẵn sàng")
    with _infer_lock:
        return llm._infer(messages, max_new_tokens=max_new_tokens)


# ── News Cache ─────────────────────────────────────────────────────────────────
_news_cache       = {"articles": [], "ts": 0.0}
_news_lock        = threading.Lock()
_news_summarizing = False
NEWS_TTL          = 900  # 15 min

RSS_FEEDS = [
    ("VnExpress",  "https://vnexpress.net/rss/thoi-su.rss"),
    ("VnExpress",  "https://vnexpress.net/rss/moi-truong.rss"),
    ("Tuổi Trẻ",   "https://tuoitre.vn/rss/tin-moi-nhat.rss"),
    ("Tuổi Trẻ",   "https://tuoitre.vn/rss/moi-truong.rss"),
    ("Thanh Niên", "https://thanhnien.vn/rss/thoi-su.rss"),
    ("Dân Trí",    "https://dantri.com.vn/xa-hoi.rss"),
    ("Tiền Phong", "https://tienphong.vn/rss/xa-hoi.rss"),
    ("Zing News",  "https://zingnews.vn/xa-hoi.rss"),
    ("Nhân Dân",   "https://nhandan.vn/rss/thoi-su.rss"),
    ("Nhân Dân",   "https://nhandan.vn/rss/xa-hoi.rss"),
    ("Lao Động",   "https://laodong.vn/rss/thoi-su.rss"),
    ("Lao Động",   "https://laodong.vn/rss/moi-truong.rss"),
    ("Phụ Nữ VN",  "https://phunuvietnam.vn/rss/thoi-su.rss"),
    ("Phụ Nữ VN",  "https://phunuvietnam.vn/rss/xa-hoi.rss"),
    ("Báo Mới",    "https://baomoi.com/thien-tai.epi/rss.xml"),
]
FLOOD_KW = [
    # lũ lụt cốt lõi
    "lũ", "lụt", "lũ lụt", "lũ quét", "mưa lũ", "lũ ống",
    "ngập lụt", "ngập úng", "ngập sâu", "nước ngập", "triều cường", "nước dâng",
    # thiên tai
    "bão", "siêu bão", "áp thấp nhiệt đới", "lốc", "lốc xoáy", "vòi rồng",
    "sạt lở", "sụt lún", "vỡ đê", "vỡ đập",
    "cảnh báo lũ", "cảnh báo bão", "báo động lũ",
    # ứng phó thiên tai
    "sơ tán", "di dời dân",
    # môi trường nước
    "hạn hán", "xâm nhập mặn", "mực nước sông", "đỉnh lũ",
    # english
    "flood", "typhoon", "flash flood", "storm surge",
]

SUMMARIZE_SYS = (
    "Bạn là trợ lý tóm tắt tin tức thiên tai bằng tiếng Việt. "
    "Tóm tắt ngắn gọn, trung thực, không thêm thông tin ngoài bài."
)
REWRITE_SYS = (
    "Bạn là phóng viên báo thiên tai Việt Nam chuyên nghiệp. "
    "Viết lại bài báo thành bài viết hoàn chỉnh, rõ ràng, tiếng Việt chuẩn. "
    "Cấu trúc gồm: đoạn mở (nêu sự kiện chính), 2-3 đoạn thân bài (chi tiết, bối cảnh), "
    "1 đoạn kết (ý nghĩa hoặc khuyến nghị an toàn). "
    "Không thêm thông tin ngoài bài gốc. Không dùng bullet points."
)
CHAT_SYS = (
    "Bạn là FloodAgent — chuyên gia AI về lũ lụt và thiên tai, hỗ trợ tiếng Việt.\n"
    "Nhiệm vụ: tư vấn an toàn lũ lụt, giải thích nguyên nhân thiên tai, "
    "hướng dẫn ứng phó khẩn cấp, phân tích ảnh lũ nếu được gửi kèm.\n"
    "Trả lời ngắn gọn, rõ ràng, ưu tiên an toàn tính mạng. "
    "KHÔNG dùng format INTENT:/DEPTH_HINT: — chỉ viết câu trả lời tự nhiên."
)


def _summarize(title: str, body: str) -> str:
    try:
        prompt = (
            f"Tóm tắt trong 1-2 câu ngắn bằng tiếng Việt. "
            f"Chỉ viết câu tóm tắt:\nTiêu đề: {title}\nNội dung: {body[:500]}"
        )
        return llm_infer([
            {"role": "system", "content": SUMMARIZE_SYS},
            {"role": "user",   "content": prompt},
        ], max_new_tokens=100).strip()
    except Exception as exc:
        log.debug(f"[summarize] {exc}")
        return (body[:250] + "…") if len(body) > 250 else body


def _parse_entry_ts(e) -> int:
    import calendar
    try:
        if e.get("published_parsed"):
            return calendar.timegm(e.published_parsed)
    except Exception:
        pass
    return 0


def _format_ts(ts: int, fallback: str) -> str:
    import datetime as _dt
    if ts:
        return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%d/%m/%Y %H:%M")
    return fallback[:16]


def _entry_to_article(src: str, e) -> dict:
    ts = _parse_entry_ts(e)
    return {
        "source":    src,
        "title":     e.get("title", ""),
        "link":      e.get("link", ""),
        "published": _format_ts(ts, e.get("published", "")),
        "ts":        ts,
        "body":      e.get("summary", "")[:800],
        "summary":   None,
    }


def _fetch_news():
    try:
        import feedparser
    except ImportError:
        return [{"title": "Cần cài feedparser", "source": "", "link": "",
                 "published": "", "ts": 0, "summary": "pip install feedparser"}]

    articles = []
    seen_links: set = set()

    for src, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                link = e.get("link", "")
                if link in seen_links:
                    continue
                txt = (e.get("title", "") + " " + e.get("summary", "")).lower()
                if not any(kw in txt for kw in FLOOD_KW):
                    continue
                seen_links.add(link)
                articles.append(_entry_to_article(src, e))
        except Exception as ex:
            log.warning(f"[news] feed {url}: {ex}")

    articles.sort(key=lambda x: x["ts"], reverse=True)
    return articles


def _summarize_articles_bg():
    """Generate LLM summaries in background; updates cache in-place."""
    global _news_summarizing
    try:
        with _news_lock:
            snapshot = list(_news_cache["articles"])
        for i, art in enumerate(snapshot[:10]):
            if art.get("summary"):
                continue
            try:
                s = _summarize(art["title"], art.get("body", ""))
                with _news_lock:
                    if i < len(_news_cache["articles"]):
                        _news_cache["articles"][i]["summary"] = s
            except Exception as exc:
                log.debug(f"[summarize_bg] idx={i}: {exc}")
    finally:
        _news_summarizing = False


# ── Image Analysis (ReferenceEstimator, same backend as /live_predict) ────────
_chat_estimator      = None
_chat_estimator_lock = threading.Lock()


def _get_chat_estimator():
    global _chat_estimator
    with _chat_estimator_lock:
        if _chat_estimator is not None:
            return _chat_estimator, None
        try:
            from depth_analysis.reference_estimator import ReferenceEstimator
            cfg_path = BASE_DIR / "config.yaml"
            raw_cfg  = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
            _chat_estimator = ReferenceEstimator(
                yolo_model    = raw_cfg.get("yolo_model",   "yolov8n.pt"),
                depth_model   = raw_cfg.get("depth_model",  "depth-anything/Depth-Anything-V2-Small-hf"),
                output_dir    = BASE_DIR / "_chat_tmp",
                conf_thresh   = float(raw_cfg.get("yolo_conf", 0.35)),
                use_dino      = raw_cfg.get("use_dino",     True),
                dino_model    = raw_cfg.get("dino_model",   "facebook/dinov2-small"),
                use_pose      = raw_cfg.get("use_pose",     True),
                pose_model    = raw_cfg.get("pose_model",   "yolov8n-pose.pt"),
                use_segformer = raw_cfg.get("use_segformer", True),
            )
            log.info("[estimator] ReferenceEstimator loaded for chat")
            return _chat_estimator, None
        except Exception as exc:
            log.warning(f"[estimator] {exc}")
            return None, str(exc)


def _analyze_image(image_b64: str) -> str:
    """Run flood analysis on an uploaded image. Returns a text summary for the LLM."""
    import base64, os, tempfile
    # Always start with this so LLM knows an image was received
    header = "[Người dùng đã gửi ảnh lũ lụt để phân tích]"
    try:
        raw = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(raw)
            tmp_path = f.name
        try:
            estimator, err = _get_chat_estimator()
            if estimator is None:
                log.warning(f"[analyze_image] estimator unavailable: {err}")
                return (
                    f"{header}\n"
                    "Pipeline phân tích ảnh chưa khả dụng (model chưa tải hoặc thiếu file).\n"
                    "Hãy mô tả những gì bạn thấy trong ảnh để tôi tư vấn thêm."
                )
            results = estimator.analyze_batch([Path(tmp_path)], chunk_size=1)
            if not results:
                return f"{header}\nKhông trích xuất được kết quả từ ảnh."
            r      = results[0]
            level  = getattr(r, "flood_level", "UNKNOWN") or "UNKNOWN"
            depth  = round(float(getattr(r, "water_height_cm", 0) or 0), 1)
            conf   = round(float(getattr(r, "confidence", 0) or 0) * 100, 1)
            n_objs = len(getattr(r, "detected_objects", []) or [])
            vehs   = getattr(r, "vehicles_detected", []) or []
            lines  = [
                header,
                "[KẾT QUẢ PHÂN TÍCH TỰ ĐỘNG]",
                f"- Mức độ lũ: {level}",
                f"- Độ sâu nước ước tính: {depth} cm",
                f"- Độ tin cậy: {conf}%",
                f"- Số đối tượng phát hiện: {n_objs}",
            ]
            if vehs:
                lines.append(f"- Phương tiện: {', '.join(str(v) for v in vehs)}")
            return "\n".join(lines)
        finally:
            try: os.unlink(tmp_path)
            except Exception: pass
    except Exception as exc:
        log.warning(f"[analyze_image] {exc}")
        return f"{header}\nLỗi khi xử lý ảnh: {exc}"


# ── Chat Sessions ──────────────────────────────────────────────────────────────
_sessions: dict = {}
_sess_lock      = threading.Lock()
MAX_HIST        = 8


# ── Auth ───────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            if request.is_json or request.method != "GET":
                return jsonify({"error": "Chưa đăng nhập"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return wrapped


# ── API error handlers (always return JSON, never HTML) ────────────────────────
_API_PREFIX = "/api/"

@app.errorhandler(404)
def _err404(e):
    if request.path.startswith(_API_PREFIX):
        return jsonify({"error": "not found"}), 404
    return e

@app.errorhandler(405)
def _err405(e):
    if request.path.startswith(_API_PREFIX):
        return jsonify({"error": "method not allowed"}), 405
    return e

@app.errorhandler(500)
def _err500(e):
    if request.path.startswith(_API_PREFIX):
        return jsonify({"error": "internal server error"}), 500
    return e


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", active="home",
                           logged_in=session.get("logged_in", False))


@app.route("/news", methods=["GET"])
def news():
    return render_template("news.html", active="news",
                           logged_in=session.get("logged_in", False))


@app.route("/map", methods=["GET"])
def map_page():
    return render_template("map.html", active="map",
                           maps_key=GOOGLE_MAPS_KEY,
                           logged_in=session.get("logged_in", False))


@app.route("/chat", methods=["GET"])
def chat():
    return render_template("chat.html", active="chat",
                           logged_in=session.get("logged_in", False))


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    output_dir = BASE_DIR / "output"
    runs = []
    if output_dir.exists():
        for d in sorted(output_dir.iterdir(), reverse=True)[:10]:
            if d.is_dir():
                imgs = sum(len(list(d.rglob(p))) for p in _IMG_PATTERNS)
                runs.append({"name": d.name, "images": imgs})
    return render_template(
        "dashboard.html", active="dashboard", logged_in=True,
        runs=runs,
        llm_ready=_llm_ready, llm_loading=_llm_loading, llm_error=_llm_error,
    )


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        user    = request.form.get("username", "")
        pw_hash = hashlib.sha256(
            request.form.get("password", "").encode()
        ).hexdigest()
        if user == ADMIN_USER and pw_hash == ADMIN_PASS_HASH:
            session["logged_in"] = True
            session["is_admin"]  = True
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Sai tên đăng nhập hoặc mật khẩu")
    return render_template("login.html")


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect(url_for("home"))


# ══════════════════════════════════════════════════════════════════════════════
# API — NEWS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/news/fetch", methods=["GET"])
def api_news_fetch():
    global _news_summarizing
    force = request.args.get("refresh") == "1"
    with _news_lock:
        if not force and time.time() - _news_cache["ts"] < NEWS_TTL:
            return jsonify({"articles": _news_cache["articles"], "cached": True,
                            "summarizing": _news_summarizing})

    articles = _fetch_news()
    with _news_lock:
        _news_cache.update({"articles": articles, "ts": time.time()})

    if not _news_summarizing and get_llm():
        _news_summarizing = True
        threading.Thread(target=_summarize_articles_bg, daemon=True).start()

    return jsonify({"articles": articles, "cached": False,
                    "summarizing": _news_summarizing})


@app.route("/api/news/summaries", methods=["GET"])
def api_news_summaries():
    """Return current summary status for polling (no LLM call)."""
    with _news_lock:
        summaries = [a.get("summary") for a in _news_cache["articles"]]
    return jsonify({"summaries": summaries, "done": not _news_summarizing})


@app.route("/api/news/rewrite", methods=["POST"])
def api_news_rewrite():
    try:
        data  = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            data = {}
        title = data.get("title", "").strip()
        body  = data.get("body", "").strip()
        if not title:
            return jsonify({"error": "missing title"}), 400

        try:
            rewritten = llm_infer([
                {"role": "system", "content": REWRITE_SYS},
                {"role": "user",   "content":
                 f"Tiêu đề: {title}\nNội dung gốc: {body[:800]}\n\nViết lại thành bài báo hoàn chỉnh:"},
            ], max_new_tokens=350).strip()
        except Exception as exc:
            log.warning(f"[rewrite] {exc}")
            rewritten = body or title
        return jsonify({"rewritten": rewritten})
    except Exception as exc:
        log.error(f"[rewrite] unhandled: {exc}")
        return jsonify({"error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# API — CHAT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat/message", methods=["POST"])
def api_chat_message():
    data      = request.get_json() or {}
    user_msg  = data.get("message", "").strip()
    sid       = data.get("session_id", "default")
    image_b64 = data.get("image_b64")

    if not user_msg and not image_b64:
        return jsonify({"error": "Thiếu nội dung"}), 400

    if not _llm_ready:
        status = "đang tải model…" if _llm_loading else "chưa khởi động"
        return jsonify({"reply": f"⏳ FloodAgent {status}. Vui lòng thử lại sau 30 giây.",
                        "loading": True})

    with _sess_lock:
        hist = _sessions.setdefault(sid, [])

    content = user_msg
    if image_b64:
        analysis = _analyze_image(image_b64)
        content  = (analysis + ("\n" + user_msg if user_msg else "")).strip()

    messages = [{"role": "system", "content": CHAT_SYS}]
    for h in hist[-MAX_HIST:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": content})

    try:
        reply = llm_infer(messages, max_new_tokens=256)
        # Strip structured format if model still outputs it
        if "INTENT:" in reply:
            from learning.flood_llm import _parse_llm_output
            reply = _parse_llm_output(reply).response or reply
    except Exception as exc:
        return jsonify({"error": f"Lỗi inference: {exc}"}), 500

    with _sess_lock:
        hist.append({"role": "user",      "content": content})
        hist.append({"role": "assistant", "content": reply})
        _sessions[sid] = hist[-MAX_HIST:]

    return jsonify({"reply": reply})


@app.route("/api/chat/clear", methods=["POST"])
def api_chat_clear():
    sid = (request.get_json() or {}).get("session_id", "default")
    with _sess_lock:
        _sessions.pop(sid, None)
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════════════════════
# API — MAP
# ══════════════════════════════════════════════════════════════════════════════

_DEMO_POINTS = [
    {"lat": 16.464, "lng": 107.591, "level": "KNEE",   "depth": 45,  "ts": "Demo"},
    {"lat": 15.880, "lng": 108.338, "level": "ANKLE",  "depth": 20,  "ts": "Demo"},
    {"lat": 10.823, "lng": 106.630, "level": "PUDDLE", "depth": 8,   "ts": "Demo"},
    {"lat": 20.845, "lng": 106.688, "level": "WAIST",  "depth": 85,  "ts": "Demo"},
    {"lat": 17.460, "lng": 106.620, "level": "CHEST",  "depth": 140, "ts": "Demo"},
]


def _parse_map_points(run_dir) -> list:
    points = []
    for jf in run_dir.glob("*.json"):
        try:
            items = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(items, dict):
                items = [items]
            for item in items:
                lat = item.get("latitude") or item.get("lat")
                lng = item.get("longitude") or item.get("lon") or item.get("lng")
                if lat and lng:
                    points.append({
                        "lat":   float(lat),
                        "lng":   float(lng),
                        "level": item.get("flood_level", "UNKNOWN"),
                        "depth": item.get("flood_depth_cm", 0),
                        "ts":    item.get("timestamp", run_dir.name),
                    })
        except Exception:
            pass
    return points


@app.route("/api/map/data", methods=["GET"])
def api_map_data():
    output_dir = BASE_DIR / "output"
    points = []

    if output_dir.exists():
        for run_dir in sorted(output_dir.iterdir(), reverse=True)[:5]:
            points.extend(_parse_map_points(run_dir))

    return jsonify({"points": points or _DEMO_POINTS})


@app.route("/api/map/weather", methods=["GET"])
def api_map_weather():
    """Rain forecast from Open-Meteo (free, no API key)."""
    lat = request.args.get("lat", "16.05")
    lon = request.args.get("lon", "108.20")
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=precipitation,precipitation_probability,weathercode"
            f"&forecast_days=3&timezone=Asia%2FHo_Chi_Minh"
        )
        with urllib.request.urlopen(url, timeout=8) as resp:
            return jsonify(json.loads(resp.read()))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# API — DASHBOARD / STATS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Public lightweight stats for home page."""
    output_dir = BASE_DIR / "output"
    total_runs = 0
    total_imgs = 0
    if output_dir.exists():
        for d in output_dir.iterdir():
            if d.is_dir():
                total_runs += 1
                total_imgs += sum(len(list(d.rglob(p))) for p in _IMG_PATTERNS)
    return jsonify({"total_runs": total_runs, "total_images": total_imgs})


@app.route("/api/dashboard/stats", methods=["GET"])
@login_required
def api_dashboard_stats():
    output_dir = BASE_DIR / "output"
    runs = []
    if output_dir.exists():
        for d in sorted(output_dir.iterdir(), reverse=True)[:10]:
            if d.is_dir():
                imgs = sum(len(list(d.rglob(p))) for p in _IMG_PATTERNS)
                runs.append({"name": d.name, "images": imgs})
    return jsonify({"total_runs": len(runs), "recent_runs": runs})


@app.route("/api/llm/status", methods=["GET"])
def api_llm_status():
    return jsonify({"ready": _llm_ready, "loading": _llm_loading, "error": _llm_error})


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY REDIRECTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/live", methods=["GET"])
def legacy_live():
    return redirect(url_for("chat"))


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL (review queue, pipeline, training, active learning)
# ══════════════════════════════════════════════════════════════════════════════

try:
    from learning.admin_routes import register_admin
    register_admin(app)
    log.info("[admin] Admin routes registered at /admin")
except Exception as _adm_exc:
    log.warning(f"[admin] Could not load admin routes: {_adm_exc}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start_llm()
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    log.info(f"FloodAI Web → http://{host}:{port}")
    app.run(debug=False, port=port, host=host)
