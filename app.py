# -*- coding: utf-8 -*-
"""
FloodAI Web — Main Entry Point
5 pages: Home / News / Map / Chat / Dashboard (admin)
"""

import base64
import hashlib
import html
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

# ── Secret key: bắt buộc phải set trong production ────────────────────────────
_secret_key = os.environ.get("SECRET_KEY")
if not _secret_key:
    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError(
            "SECRET_KEY chưa được set. Không được chạy production với key mặc định."
        )
    _secret_key = "floodai-s3cr3t-k3y-change-in-prod"
    log.warning("  [Auth] Dùng SECRET_KEY mặc định — chỉ chấp nhận trong môi trường dev.")

app.secret_key = _secret_key
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV") == "production",
)

# ── Config ─────────────────────────────────────────────────────────────────────
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")

_admin_pass = os.environ.get("ADMIN_PASS")
if not _admin_pass:
    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError(
            "ADMIN_PASS chưa được set. Không được chạy production với password mặc định."
        )
    _admin_pass = "123456"
    log.warning("  [Auth] Dùng ADMIN_PASS mặc định — chỉ chấp nhận trong môi trường dev.")

ADMIN_PASS_HASH = hashlib.sha256(_admin_pass.encode()).hexdigest()
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
        "title":     html.unescape(e.get("title", "")),
        "link":      e.get("link", ""),
        "published": _format_ts(ts, e.get("published", "")),
        "ts":        ts,
        "body":      html.unescape(e.get("summary", "")[:800]),
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


@app.route("/alerts", methods=["GET"])
def alerts_page():
    return render_template("alerts.html", active="alerts",
                           logged_in=session.get("logged_in", False))


@app.route("/submit", methods=["GET"])
def submit_page():
    return render_template("submit.html", active="submit",
                           logged_in=session.get("logged_in", False))


@app.route("/roads", methods=["GET"])
def roads_page():
    return render_template("roads.html", active="roads",
                           logged_in=session.get("logged_in", False))


@app.route("/article/<article_id>", methods=["GET"])
def article_page(article_id: str):
    """Trang chi tiết bài viết / kết quả phân tích."""
    from utils.article_generator import ArticleGenerator
    article_data = None
    try:
        from pipeline.job_queue import JobQueue
        status = JobQueue.instance().get(article_id)
        if status and status.result_summary:
            res = status.result_summary
            gen = ArticleGenerator()
            article_data = gen.from_result(
                res,
                location=res.get("location", "khu vực phân tích"),
                time=res.get("time", ""),
                source=res.get("source", "Hệ thống"),
                run_id=article_id,
            )
    except Exception as exc:
        log.debug("article_page: %s", exc)
    return render_template("article.html", active="news",
                           article=article_data,
                           logged_in=session.get("logged_in", False))


@app.route("/chat", methods=["GET"])
def chat():
    return render_template("chat.html", active="chat",
                           logged_in=session.get("logged_in", False))


@app.route("/dashboard", methods=["GET"])
@app.route("/admin", methods=["GET"])
@login_required
def dashboard():
    return render_template("admin.html", active="admin",
                           logged_in=True)


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

@app.route("/api/map/data", methods=["GET"])
def api_map_data():
    """
    Trả về danh sách điểm ngập có tọa độ để hiện trên map.
    Kết hợp: job_meta DB (geo) + job_queue DB (flood level).
    """
    from pipeline.job_queue import JobQueue
    geo_rows  = _get_all_meta(limit=200)
    geo_index = {r["job_id"]: r for r in geo_rows}

    points = []
    try:
        all_jobs = JobQueue.instance().list_recent(200)
        for job in all_jobs:
            if job.status != "done":
                continue
            meta = geo_index.get(job.job_id)
            if not meta or meta.get("lat") is None:
                continue
            result = job.result_summary or {}
            points.append({
                "job_id":   job.job_id,
                "lat":      meta["lat"],
                "lon":      meta["lon"],
                "radius_m": meta["radius_m"],
                "method":   meta["method"],
                "address":  meta["address"] or result.get("location", ""),
                "level":    result.get("level", "unknown"),
                "depth_range": result.get("depth_range", ""),
                "confidence": result.get("confidence_raw", 0),
                "time":     job.created_at,
                "source":   result.get("source", "Người dân gửi"),
            })
    except Exception as exc:
        log.warning("[map/data] %s", exc)

    return jsonify({"points": points})


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
# PIPELINE API  (POST /api/analyze, GET /api/jobs/*, GET /api/health)
# ══════════════════════════════════════════════════════════════════════════════

def _get_analysis_service():
    """Lazy-init FloodAnalysisService với config từ config.yaml."""
    from pipeline.service import get_service
    from utils.config_loader import load_config as _load_yaml, apply_config_to_cfg
    raw = _load_yaml(str(BASE_DIR / "config.yaml"))
    cfg = apply_config_to_cfg(raw, {})
    return get_service(cfg)


# ── Job location metadata (SQLite) ────────────────────────────────────────────

_META_DB = BASE_DIR / "output" / "job_meta.db"

def _meta_conn():
    import sqlite3
    _META_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_META_DB), check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS job_meta (
        job_id TEXT PRIMARY KEY,
        lat REAL, lon REAL, radius_m INTEGER,
        method TEXT, address TEXT,
        province TEXT, district TEXT, street TEXT,
        confidence REAL, created_at TEXT
    )""")
    conn.commit()
    return conn

def _save_job_meta(job_id: str, geo: "GeoEstimate"):
    try:
        import sqlite3
        from datetime import datetime
        conn = _meta_conn()
        conn.execute("""INSERT OR REPLACE INTO job_meta
            (job_id,lat,lon,radius_m,method,address,province,district,street,confidence,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (job_id, geo.lat, geo.lon, geo.radius_m, geo.method,
             geo.address, geo.province, geo.district, geo.street,
             geo.confidence, datetime.now().isoformat(timespec="seconds")))
        conn.commit()
        conn.close()
    except Exception as exc:
        log.debug("[meta] save error: %s", exc)

def _get_job_meta(job_id: str) -> Optional[dict]:
    try:
        conn = _meta_conn()
        row  = conn.execute(
            "SELECT * FROM job_meta WHERE job_id=?", (job_id,)
        ).fetchone()
        conn.close()
        if row:
            cols = ["job_id","lat","lon","radius_m","method","address",
                    "province","district","street","confidence","created_at"]
            return dict(zip(cols, row))
    except Exception:
        pass
    return None

def _get_all_meta(limit: int = 100) -> list:
    try:
        conn  = _meta_conn()
        rows  = conn.execute(
            "SELECT job_id,lat,lon,radius_m,method,address,province,district,street,confidence "
            "FROM job_meta WHERE lat IS NOT NULL ORDER BY rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        cols = ["job_id","lat","lon","radius_m","method","address",
                "province","district","street","confidence"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception:
        return []


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check — không cần login."""
    try:
        svc = _get_analysis_service()
        info = svc.health()
    except Exception as exc:
        info = {"status": "degraded", "error": str(exc)}
    return jsonify(info)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Upload ảnh — public, không cần đăng nhập.
    Form data: images, browser_lat, browser_lon, browser_accuracy, safe
    """
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "Không có ảnh nào được upload"}), 400

    # Browser geolocation (gửi kèm từ JS)
    browser_lat = request.form.get("browser_lat", type=float)
    browser_lon = request.form.get("browser_lon", type=float)
    browser_acc = request.form.get("browser_accuracy", type=float)
    safe_mode   = request.form.get("safe", "false").lower() == "true"
    client_ip   = request.remote_addr or request.headers.get("X-Forwarded-For", "")

    # Lưu ảnh vào thư mục tạm
    import uuid as _uuid
    job_tmp = BASE_DIR / "_upload_tmp" / _uuid.uuid4().hex
    job_tmp.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        if not f or not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        dest = job_tmp / (_uuid.uuid4().hex + ext)
        f.save(str(dest))
        if dest.stat().st_size > 30 * 1024 * 1024:
            dest.unlink(); continue
        saved.append(dest)

    if not saved:
        try: job_tmp.rmdir()
        except Exception: pass
        return jsonify({"error": "Không có ảnh hợp lệ (jpg/png/webp, ≤30MB)"}), 400

    # Geo-localize ngay từ ảnh đầu tiên
    from utils.geo_localizer import GeoLocalizer
    geo = GeoLocalizer().localize(
        image_path=saved[0],
        browser_lat=browser_lat,
        browser_lon=browser_lon,
        browser_accuracy=browser_acc,
        client_ip=client_ip,
    )
    log.info("[analyze] geo: method=%s lat=%s lon=%s r=%sm",
             geo.method, geo.lat, geo.lon, geo.radius_m)

    try:
        from utils.config_loader import load_config as _load_yaml, apply_config_to_cfg
        raw = _load_yaml(str(BASE_DIR / "config.yaml"))
        cfg = apply_config_to_cfg(raw, {})
        if safe_mode:
            cfg.update({"skip_drive": True, "skip_learning": True,
                        "skip_hard_mining": True, "skip_versioning": True})

        svc    = _get_analysis_service()
        job_id = svc.submit_async(saved, input_label=geo.address or str(job_tmp))
        _save_job_meta(job_id, geo)
    except Exception as exc:
        log.error("[api_analyze] %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "job_id":       job_id,
        "queued_count": len(saved),
        "geo":          geo.to_dict(),
        "status_url":   f"/api/jobs/{job_id}",
        "stream_url":   f"/api/jobs/{job_id}/stream",
    }), 202


@app.route("/api/drive/analyze", methods=["POST"])
@login_required
def api_drive_analyze():
    """
    Admin paste link Google Drive folder → pull ảnh → chạy pipeline.
    JSON body: { "drive_url": "https://drive.google.com/drive/folders/...", "safe": true }
    """
    data      = request.get_json(force=True) or {}
    drive_url = (data.get("drive_url") or "").strip()
    safe_mode = data.get("safe", True)
    max_files = int(data.get("max_files", 200))

    if not drive_url:
        return jsonify({"error": "Thiếu drive_url"}), 400

    # Kiểm tra URL hợp lệ
    if "drive.google.com" not in drive_url and len(drive_url) < 10:
        return jsonify({"error": "URL Drive không hợp lệ"}), 400

    import uuid as _uuid
    job_tmp = BASE_DIR / "_drive_tmp" / _uuid.uuid4().hex
    job_tmp.mkdir(parents=True, exist_ok=True)

    def _pull_and_submit():
        try:
            from uploader.drive_uploader import DriveUploader
            uploader = DriveUploader()
            log.info("[Drive] Downloading from: %s", drive_url)
            images = uploader.download_folder(
                drive_folder=drive_url,
                local_dir=job_tmp,
                max_files=max_files,
            )
            if not images:
                log.warning("[Drive] Không tìm thấy ảnh trong folder")
                return

            log.info("[Drive] Downloaded %d images → submitting job", len(images))
            from utils.config_loader import load_config as _load_yaml, apply_config_to_cfg
            raw = _load_yaml(str(BASE_DIR / "config.yaml"))
            cfg = apply_config_to_cfg(raw, {})
            if safe_mode:
                cfg.update({"skip_drive": True, "skip_learning": True,
                            "skip_hard_mining": True, "skip_versioning": True})

            svc    = _get_analysis_service()
            job_id = svc.submit_async(images, input_label=f"Drive: {drive_url[-40:]}")
            log.info("[Drive] Job submitted: %s", job_id)

        except Exception as exc:
            log.error("[Drive] Pull failed: %s", exc, exc_info=True)

    # Pull chạy nền — không block request
    threading.Thread(target=_pull_and_submit, daemon=True).start()

    return jsonify({
        "status":    "pulling",
        "drive_url": drive_url,
        "message":   f"Đang tải ảnh từ Drive (tối đa {max_files} file). Kiểm tra Jobs sau vài phút.",
        "jobs_url":  "/admin",
    }), 202
    """Trả về trạng thái hiện tại của một job."""
    from pipeline.job_queue import JobQueue
    status = JobQueue.instance().get(job_id)
    if not status:
        return jsonify({"error": "Job không tồn tại"}), 404
    return jsonify({
        "job_id":        status.job_id,
        "status":        status.status,
        "stage":         status.stage,
        "progress":      status.progress,
        "message":       status.message,
        "error":         status.error,
        "created_at":    status.created_at,
        "finished_at":   status.finished_at,
        "output_dir":    status.output_dir,
        "result":        status.result_summary,
    })


@app.route("/api/jobs", methods=["GET"])
@login_required
def api_jobs_list():
    """Danh sách jobs gần đây."""
    limit = min(int(request.args.get("limit", 20)), 100)
    from pipeline.job_queue import JobQueue
    jobs = JobQueue.instance().list_recent(limit)
    return jsonify([{
        "job_id":      j.job_id,
        "status":      j.status,
        "progress":    j.progress,
        "message":     j.message,
        "created_at":  j.created_at,
        "finished_at": j.finished_at,
    } for j in jobs])


@app.route("/api/jobs/<job_id>/stream", methods=["GET"])
@login_required
def api_job_stream(job_id: str):
    """
    Server-Sent Events stream cho tiến độ realtime của job.

    Dùng từ JS:
        const es = new EventSource('/api/jobs/<id>/stream');
        es.onmessage = e => {
            const d = JSON.parse(e.data);
            updateProgressBar(d.progress, d.message);
        };
    """
    from flask import Response, stream_with_context
    from pipeline.job_queue import JobQueue

    def generate():
        for chunk in JobQueue.instance().stream(job_id):
            yield chunk

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/runs/<run_id>/report", methods=["GET"])
@login_required
def api_run_report(run_id: str):
    """Trả về nội dung report JSON của một run cụ thể."""
    from utils.constants import PIPELINE_SUMMARY_JSON
    summary_path = BASE_DIR / "output" / run_id / PIPELINE_SUMMARY_JSON
    if not summary_path.exists():
        return jsonify({"error": "Report không tìm thấy"}), 404
    import json as _json
    return jsonify(_json.loads(summary_path.read_text(encoding="utf-8")))


# ── /api/metrics — số liệu tổng hợp ─────────────────────────────────────────

@app.route("/api/metrics", methods=["GET"])
@login_required
def api_metrics():
    """
    Số liệu tổng hợp về job và pipeline.
    Dùng cho dashboard biểu đồ.
    """
    from pipeline.job_queue import JobQueue
    import time as _time

    jq = JobQueue.instance()
    jobs = jq.list_recent(200)

    total      = len(jobs)
    done       = [j for j in jobs if j.status == "done"]
    failed     = [j for j in jobs if j.status == "failed"]
    pending    = sum(1 for j in jobs if j.status == "pending")
    running    = sum(1 for j in jobs if j.status == "running")

    # Thời gian xử lý trung bình (từ created → finished)
    durations = []
    for j in done:
        if j.created_at and j.finished_at:
            try:
                from datetime import datetime as _dt
                t1 = _dt.fromisoformat(j.created_at)
                t2 = _dt.fromisoformat(j.finished_at)
                durations.append((t2 - t1).total_seconds())
            except Exception:
                pass

    avg_time = sum(durations) / len(durations) if durations else 0

    # Tổng ảnh xử lý
    total_analyzed = sum(
        j.result_summary.get("total_analyzed", 0) for j in done
    )

    # Lý do failed phổ biến
    fail_reasons: Dict[str, int] = {}
    for j in failed:
        reason = j.error[:60] if j.error else "unknown"
        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    return jsonify({
        "jobs": {
            "total":    total,
            "done":     len(done),
            "failed":   len(failed),
            "pending":  pending,
            "running":  running,
        },
        "performance": {
            "avg_processing_s": round(avg_time, 1),
            "total_images_analyzed": total_analyzed,
        },
        "fail_reasons": sorted(fail_reasons.items(), key=lambda x: -x[1])[:5],
        "timestamp": __import__("time").time(),
    })


# ── /api/v1/* — versioned API aliases ────────────────────────────────────────

@app.route("/api/v1/jobs", methods=["POST"])
@login_required
def api_v1_create_job():
    """Alias: POST /api/v1/jobs → POST /api/analyze"""
    return api_analyze()


@app.route("/api/v1/jobs/<job_id>", methods=["GET"])
@login_required
def api_v1_job(job_id: str):
    return api_job_status(job_id)


@app.route("/api/v1/jobs/<job_id>/result", methods=["GET"])
@login_required
def api_v1_job_result(job_id: str):
    """Trả về kết quả chi tiết của job (nếu đã done)."""
    from pipeline.job_queue import JobQueue
    status = JobQueue.instance().get(job_id)
    if not status:
        return jsonify({"error": "Job không tồn tại"}), 404
    if status.status != "done":
        return jsonify({
            "error": f"Job chưa xong (status={status.status})",
            "job_id": job_id,
            "status": status.status,
        }), 202
    return jsonify(status.result_summary)


@app.route("/api/v1/jobs/<job_id>/report", methods=["GET"])
@login_required
def api_v1_job_report(job_id: str):
    """Trả về report JSON của job."""
    from pipeline.job_queue import JobQueue
    status = JobQueue.instance().get(job_id)
    if not status or not status.output_dir:
        return jsonify({"error": "Report không tìm thấy"}), 404
    from utils.constants import PIPELINE_SUMMARY_JSON
    import json as _j
    p = Path(status.output_dir) / PIPELINE_SUMMARY_JSON
    if not p.exists():
        return jsonify({"error": "Report chưa được tạo"}), 404
    return jsonify(_j.loads(p.read_text(encoding="utf-8")))


@app.route("/api/v1/metrics", methods=["GET"])
@login_required
def api_v1_metrics():
    return api_metrics()


@app.route("/api/v1/health", methods=["GET"])
def api_v1_health():
    return api_health()


@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    """Minimal OpenAPI 3.0 spec cho pipeline API."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "FloodAI Pipeline API",
            "version": "1.0.0",
            "description": "API phân tích ảnh lũ lụt",
        },
        "paths": {
            "/api/v1/health":           {"get":  {"summary": "Health check",               "tags": ["system"]}},
            "/api/v1/metrics":          {"get":  {"summary": "Số liệu tổng hợp",           "tags": ["system"]}},
            "/api/v1/jobs":             {"post": {"summary": "Tạo job phân tích ảnh",       "tags": ["jobs"]}},
            "/api/v1/jobs/{job_id}":    {"get":  {"summary": "Trạng thái job",              "tags": ["jobs"]}},
            "/api/v1/jobs/{job_id}/result": {"get": {"summary": "Kết quả job",             "tags": ["jobs"]}},
            "/api/v1/jobs/{job_id}/report": {"get": {"summary": "Report JSON",             "tags": ["jobs"]}},
            "/api/jobs/{job_id}/stream":{"get":  {"summary": "SSE progress stream",        "tags": ["jobs"]}},
        },
    }
    return jsonify(spec)


@app.route("/api/docs", methods=["GET"])
def api_docs():
    """Redirect sang Swagger UI (cần swagger-ui-bundle CDN)."""
    swagger_html = """<!DOCTYPE html>
<html><head><title>FloodAI API Docs</title>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui.min.css"/>
</head><body>
<div id="swagger-ui"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.11.0/swagger-ui-bundle.min.js"></script>
<script>
SwaggerUIBundle({url:"/openapi.json",dom_id:"#swagger-ui",presets:[SwaggerUIBundle.presets.apis],layout:"BaseLayout"});
</script></body></html>"""
    from flask import Response
    return Response(swagger_html, mimetype="text/html")


# ── Khởi động JobQueue khi app start ──────────────────────────────────────────
try:
    from pipeline.job_queue import JobQueue
    JobQueue.instance()
    log.info("[JobQueue] Initialized and worker thread started")
except Exception as _jq_exc:
    log.warning("[JobQueue] Could not initialize: %s", _jq_exc)


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
