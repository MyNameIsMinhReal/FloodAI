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
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from utils.geo_localizer import GeoEstimate

import yaml

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

# Bootstrap
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Đảm bảo UTF-8 cho console tránh lỗi Unicode trên mọi hệ điều hành
for _stream in (sys.stdout, sys.stderr):
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
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

# ── Auth blueprints ────────────────────────────────────────────────────────────
try:
    from auth.routes import auth_bp, users_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(users_bp)
    log.info("[Auth] Blueprints registered: /api/auth/*, /api/users/*")
except Exception as _auth_exc:
    log.warning(f"[Auth] Could not load auth routes: {_auth_exc}")

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
LORA_MODEL      = str(BASE_DIR / "models" / "flood-agent-lora")
DEFAULT_VL_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
_IMG_PATTERNS   = ("*.jpg", "*.png")

# ── LLM / VLM Singleton ───────────────────────────────────────────────────────
_llm            = None
_llm_ready      = False
_llm_loading    = False
_llm_error: str = ""
_llm_lock       = threading.Lock()
_infer_lock     = threading.Lock()  # serialise inference


def _resolve_llm_model_source() -> str:
    """Ưu tiên: flood-agent-lora (nhẹ, load 4-bit nhanh) → flood-agent-merged → models.vlm trong config.yaml."""
    lora_p = BASE_DIR / "models" / "flood-agent-lora"
    if (lora_p / "adapter_config.json").exists():
        return str(lora_p)
    merged_p = BASE_DIR / "models" / "flood-agent-merged"
    if (merged_p / "config.json").exists():
        return str(merged_p)
    try:
        cfg_path = BASE_DIR / "config.yaml"
        if cfg_path.exists():
            raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            vlm_name = (raw_cfg.get("models", {}) or {}).get("vlm") or raw_cfg.get("vlm_model")
            if vlm_name:
                return str(vlm_name)
    except Exception:
        pass
    return DEFAULT_VL_MODEL


def _load_llm_bg():
    global _llm, _llm_ready, _llm_loading, _llm_error
    try:
        from learning.flood_llm import LLMEnhancer
        model_src = _resolve_llm_model_source()
        log.info(f"[LLM] Loading VLM/LLM from: {model_src}")
        try:
            _llm = LLMEnhancer.from_lora_dir(model_src, max_new_tokens=512, temperature=0.7)
        except Exception as first_exc:
            log.warning(f"[LLM] Primary model {model_src} failed ({first_exc}), clearing CUDA cache and falling back to {DEFAULT_VL_MODEL}")
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if model_src != DEFAULT_VL_MODEL:
                _llm = LLMEnhancer.from_lora_dir(DEFAULT_VL_MODEL, max_new_tokens=512, temperature=0.7)
            else:
                raise
        _llm_ready = True
        log.info(f"[LLM] Model loaded ✓ (is_vl={getattr(_llm, '_is_vl', False)})")
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


def llm_infer(messages: list, max_new_tokens: Optional[int] = None, image=None) -> str:
    """Thread-safe inference (supports text-only and multimodal image+text for Qwen-VL)."""
    llm = get_llm()
    if llm is None:
        raise RuntimeError("Model chưa sẵn sàng")
    with _infer_lock:
        return llm._infer(messages, max_new_tokens=max_new_tokens, image=image)


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
    "Bạn là FloodAgent — chuyên gia AI về lũ lụt và cứu hộ khẩn cấp thuộc Hệ thống FloodAI GIRC Thái Nguyên.\n"
    "Nhiệm vụ: tư vấn an toàn lũ lụt, giải thích nguyên nhân thiên tai, "
    "hướng dẫn ứng phó cứu hộ khẩn cấp, phân tích hiện trạng ngập lũ nếu có ảnh gửi kèm.\n"
    "QUY TẮC BẮT BUỘC:\n"
    "- LUÔN LUÔN trả lời 100% bằng TIẾNG VIỆT tự nhiên, chuẩn mực, rõ ràng.\n"
    "- TUYỆT ĐỐI KHÔNG dùng tiếng Trung, chữ Hán, tiếng Nga hay bất kỳ từ ngữ ngoại lai nào.\n"
    "- Trả lời ngắn gọn, thiết thực, ưu tiên bảo vệ an toàn tính mạng con người.\n"
    "- KHÔNG dùng format INTENT:/DEPTH_HINT: — chỉ viết câu trả lời tự nhiên."
)


def _clean_vietnamese_reply(text: str) -> str:
    """Xóa bỏ hoặc dịch các cụm từ tiếng Trung/ngoại lai nếu mô hình gốc bị code-switching."""
    import re
    replacements = {
        "原地等待救援": "ở nguyên vị trí an toàn chờ cứu hộ",
        "等待救援": "chờ lực lượng cứu hộ",
        "注意安全": "chú ý an toàn",
        "撤离": "sơ tán khẩn cấp",
        "水位": "mực nước",
        "洪水": "ngập lụt",
        "危险": "nguy hiểm",
        "安全": "an toàn",
        "救援": "cứu hộ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Loại bỏ triệt để các ký tự chữ Hán (CJK) nếu còn sót lại
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    # Dọn dẹp khoảng trắng thừa
    text = re.sub(r' +', ' ', text)
    return text.strip()


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
                txt = (str(e.get("title") or "") + " " + str(e.get("summary") or "")).lower()
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
                use_dino      = raw_cfg.get("use_dino",     False),
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
            # ── [MỚI] Pre-validate ảnh trước khi chạy pipeline nặng ─────
            try:
                from depth_analysis.junk_classifier import JunkClassifier
                jc = JunkClassifier()
                screen = jc.classify(tmp_path)
                if screen.image_type in ("junk", "dry"):
                    log.info(f"[analyze_image] pre-validated "
                             f"type={screen.image_type} reason={screen.reason}")
                    return (
                        f"{header}\n"
                        f"{screen.vietnamese_message()}\n"
                        "Vì vậy tôi không chạy phân tích độ sâu cho ảnh này."
                        " Hãy gửi ảnh chụp ngoài trời lúc ngập để tôi phân tích."
                    )
            except Exception as exc:
                log.warning(f"[analyze_image] pre-validate bỏ qua: {exc}")
            results = estimator.analyze_batch([Path(tmp_path)], chunk_size=1)
            if not results:
                return f"{header}\nKhông trích xuất được kết quả từ ảnh."
            r      = results[0]
            level  = getattr(r, "flood_level", "UNKNOWN") or "UNKNOWN"
            depth  = round(float(getattr(r, "water_height_cm", 0) or 0), 1)
            conf   = round(float(getattr(r, "confidence", 0) or 0) * 100, 1)
            n_objs = len(getattr(r, "detected_objects", []) or [])
            vehs   = getattr(r, "vehicles_detected", []) or []
            # ── [v4] Thêm thông tin scene quality + night mode ──────────
            scene_sc = getattr(r, "scene_score", None)
            is_night = getattr(r, "is_night", None)
            needs_rev = getattr(r, "needs_review", None)
            spread   = getattr(r, "method_spread_cm", None)
            lines  = [
                header,
                "[KẾT QUẢ PHÂN TÍCH TỰ ĐỘNG]",
                f"- Mức độ lũ: {level}",
                f"- Độ sâu nước ước tính: {depth} cm",
                f"- Độ tin cậy: {conf}%",
                f"- Số đối tượng phát hiện: {n_objs}",
            ]
            if scene_sc is not None:
                q_tag = "✅" if scene_sc >= 0.6 else "⚠" if scene_sc >= 0.4 else "❌"
                lines.append(f"- Chất lượng ảnh: {q_tag} {scene_sc:.0%}")
            if is_night is not None:
                lines.append(f"- Điều kiện ánh sáng: {'Ban đêm' if is_night else 'Ban ngày'}")
            if needs_rev:
                lines.append(f"- ⚠ Cần xác minh: bất đồng đo lường (spread={spread:.0f}cm)" if spread else "- ⚠ Cần xác minh: bất đồng đo lường")
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
# PAGE ROUTES (Single-Page App: tất cả đều dùng chung templates/home.html)
# ══════════════════════════════════════════════════════════════════════════════

def _render_spa(initial_view: str = "home", **extra):
    return render_template(
        "home.html",
        active=initial_view,
        initial_view=initial_view,
        maps_key=GOOGLE_MAPS_KEY,
        logged_in=session.get("logged_in", False),
        user_role=session.get("role", ""),
        user_email=session.get("email", ""),
        user_name=session.get("name", ""),
        **extra,
    )


@app.route("/", methods=["GET"])
def home():
    return _render_spa("home")


@app.route("/news", methods=["GET"])
def news():
    return _render_spa("home")


@app.route("/map", methods=["GET"])
def map_page():
    return _render_spa("map")


@app.route("/alerts", methods=["GET"])
def alerts_page():
    return _render_spa("home")


@app.route("/submit", methods=["GET"])
def submit_page():
    return _render_spa("sos")


@app.route("/roads", methods=["GET"])
def roads_page():
    return _render_spa("map")


@app.route("/article/<article_id>", methods=["GET"])
def article_page(article_id: str):
    return _render_spa("home")


@app.route("/chat", methods=["GET"])
def chat():
    return _render_spa("chat")


@app.route("/dashboard", methods=["GET"])
@app.route("/admin", methods=["GET"])
def dashboard():
    return _render_spa("admin")


@app.route("/login", methods=["GET", "POST"])
def login_page():
    return _render_spa("admin")


@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect(url_for("home"))


# ══════════════════════════════════════════════════════════════════════════════
# API — SOS REPORTS, BROADCAST & INSTANT AI DEPTH (cho SPA home.html)
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_SOS_REPORTS = [
    {
        "id": 1,
        "title": "Cứu hộ khẩn 4 người mắc kẹt gác xép",
        "type": "emergency",
        "lat": 21.5850,
        "lng": 105.8500,
        "location": "Phường Túc Duyên, TP. Thái Nguyên",
        "time": "10 phút trước",
        "phone": "0982***112",
        "detail": "Nước dâng tràn hơn 1.8m, cắt điện hoàn toàn. Cần xuồng máy hỗ trợ đón người già và em nhỏ.",
        "aiAnalysis": "Nước dâng 180cm • Mức độ 3 (Khẩn cấp)",
        "status": "pending",
    },
    {
        "id": 2,
        "title": "Ngập sâu 1.2m ngã tư Bến Oánh",
        "type": "flood",
        "lat": 21.6050,
        "lng": 105.8350,
        "location": "Phường Quang Vinh, TP. Thái Nguyên",
        "time": "25 phút trước",
        "phone": "0915***388",
        "detail": "Các phương tiện ô tô, xe máy không thể lưu thông qua khu vực này. Nhiều xe ô tô con bị ngập.",
        "aiAnalysis": "Nước dâng 120cm • Xe gầm cao bị cản trở",
        "status": "pending",
    },
    {
        "id": 3,
        "title": "Trạm Sơ Tán & Kho Nhu Yếu Phẩm ĐH Nông Lâm (TUAF)",
        "type": "shelter",
        "lat": 21.5880,
        "lng": 105.8180,
        "location": "Trường ĐH Nông Lâm Thái Nguyên, Quyết Thắng",
        "time": "Đang hoạt động",
        "phone": "0208.3855.112",
        "detail": "Khu vực địa hình cao ráo, cung cấp chỗ ở tạm thời, nước sạch, sạc pin điện thoại và cơm nóng miễn phí.",
        "aiAnalysis": "An toàn tuyệt đối • Sức chứa 500 người",
        "status": "resolved",
    },
    {
        "id": 4,
        "title": "Cần viện trợ nước sạch & áo phao",
        "type": "emergency",
        "lat": 21.5720,
        "lng": 105.8580,
        "location": "Phường Gia Sàng, TP. Thái Nguyên",
        "time": "40 phút trước",
        "phone": "0977***901",
        "detail": "Xóm bị cô lập 2 ngày qua, hết nước uống và pin dự phòng.",
        "aiAnalysis": "Cần nhu yếu phẩm khẩn cấp",
        "status": "pending",
    },
    {
        "id": 5,
        "title": "Trạm Cứu Hộ & Y Tế Đồng Hỷ",
        "type": "shelter",
        "lat": 21.6120,
        "lng": 105.8650,
        "location": "Trung tâm Y tế Huyện Đồng Hỷ",
        "time": "Đang hoạt động",
        "phone": "0208.3822.115",
        "detail": "Trực 24/24 tiếp nhận bệnh nhân và các ca sơ cứu thiên tai.",
        "aiAnalysis": "Điểm y tế hỗ trợ",
        "status": "resolved",
    },
    {
        "id": 6,
        "title": "Trạm Quan Trắc Mực Nước Cầu Gia Bẩy (Sông Cầu)",
        "type": "sensor",
        "lat": 21.5980,
        "lng": 105.8480,
        "location": "Cầu Gia Bẩy, Sông Cầu, Thái Nguyên",
        "time": "Vừa cập nhật",
        "phone": "112",
        "detail": "Mực nước Sông Cầu vượt Báo động III (+2.4m). Cảnh báo ngập lụt diện rộng.",
        "aiAnalysis": "Mực nước Báo động III",
        "status": "resolved",
    },
]


def _get_sos_reports() -> list:
    data = _load_json("sos_reports.json", default=None)
    if not data:
        _save_json("sos_reports.json", _DEFAULT_SOS_REPORTS)
        return list(_DEFAULT_SOS_REPORTS)
    return data


@app.route("/api/sos/reports", methods=["GET"])
def api_sos_reports_list():
    reports = _get_sos_reports()
    bcast = _load_json("broadcast.json", default={"active": False, "text": ""})
    return jsonify({"reports": reports, "broadcast": bcast})


@app.route("/api/sos/report", methods=["POST"])
def api_sos_report_create():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "Người dân").strip()
    phone = (data.get("phone") or "").strip()
    address = (data.get("address") or "TP. Thái Nguyên").strip()
    rtype = (data.get("type") or "emergency").strip()
    detail = (data.get("detail") or "Tín hiệu cứu hộ gửi trực tiếp từ ứng dụng FloodAI.").strip()
    lat = data.get("lat")
    lng = data.get("lng")
    image_b64 = data.get("image_b64")

    if lat is None or lng is None:
        import random
        lat = round(21.5928 + (random.random() - 0.5) * 0.025, 5)
        lng = round(105.8442 + (random.random() - 0.5) * 0.025, 5)

    ai_label = "Mới gửi • Đã xác minh tọa độ GPS"
    depth_cm = None
    if image_b64:
        try:
            d_res = _run_instant_depth(image_b64)
            if d_res.get("depth_cm") is not None:
                depth_cm = d_res["depth_cm"]
                ai_label = f"AI đo: {depth_cm}cm • {d_res.get('risk_label', 'Đã phân tích')}"
        except Exception as exc:
            log.debug(f"[sos/report] image depth error: {exc}")

    new_item = {
        "id": int(time.time() * 1000),
        "title": f"[{ 'Cứu hộ khẩn' if rtype == 'emergency' else 'Báo ngập' if rtype == 'flood' else 'Điểm hỗ trợ' }] {name}",
        "type": rtype,
        "lat": float(lat),
        "lng": float(lng),
        "location": address,
        "time": "Vừa xong",
        "phone": phone,
        "detail": detail,
        "aiAnalysis": ai_label,
        "depth_cm": depth_cm,
        "status": "pending",
    }
    reports = _get_sos_reports()
    reports.insert(0, new_item)
    _save_json("sos_reports.json", reports[:200])
    return jsonify({"ok": True, "report": new_item})


@app.route("/api/sos/resolve/<int:report_id>", methods=["POST"])
def api_sos_resolve(report_id: int):
    reports = _get_sos_reports()
    for r in reports:
        if int(r.get("id", 0)) == report_id:
            r["status"] = "resolved"
            _save_json("sos_reports.json", reports)
            return jsonify({"ok": True, "report": r})
    return jsonify({"error": "Không tìm thấy"}), 404


@app.route("/api/sos/broadcast", methods=["POST"])
def api_sos_broadcast():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    active = bool(data.get("active", bool(text)))
    payload = {"active": active, "text": text, "updated_at": _now_iso()}
    _save_json("broadcast.json", payload)
    return jsonify({"ok": True, "broadcast": payload})


def _run_instant_depth(image_b64: str) -> dict:
    """Chạy nhanh JunkClassifier + ReferenceEstimator trên ảnh base64."""
    import base64
    import tempfile

    raw = base64.b64decode(image_b64)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(raw)
        tmp_path = f.name

    try:
        # 1. Kiểm tra ảnh rác / khô trước
        try:
            from depth_analysis.junk_classifier import JunkClassifier
            jc = JunkClassifier()
            screen = jc.classify(tmp_path)
            if screen.image_type in ("junk", "dry"):
                return {
                    "ok": True,
                    "image_type": screen.image_type,
                    "depth_cm": 0.0,
                    "flood_level": "NO_FLOOD",
                    "confidence": round(float(screen.confidence) * 100, 1),
                    "risk_label": "KHÔNG NGẬP / ẢNH KHÔNG HỢP LỆ",
                    "advice": screen.vietnamese_message(),
                }
        except Exception as exc:
            log.debug(f"[instant_depth] junk check skipped: {exc}")

        # 2. Chạy ReferenceEstimator (SegFormer + Depth Anything V2 + YOLO/Pose)
        estimator, err = _get_chat_estimator()
        if estimator is None:
            raise RuntimeError(f"Model chưa sẵn sàng: {err}")

        results = estimator.analyze_batch([Path(tmp_path)], chunk_size=1)
        if not results:
            raise RuntimeError("Không trích xuất được kết quả từ ảnh")

        r = results[0]
        depth_cm = round(float(getattr(r, "water_height_cm", 0) or 0), 1)
        level = str(getattr(r, "flood_level", "UNKNOWN") or "UNKNOWN").upper()
        conf = round(float(getattr(r, "confidence", 0) or 0) * 100, 1)
        vehs = getattr(r, "vehicles_detected", []) or []

        if depth_cm >= 100 or level in ("CHEST", "SUBMERGED"):
            risk_label = f"CẢNH BÁO MỨC 3 — RẤT NGUY HIỂM ({level})"
            advice = "⚠️ Mực nước ngập rất sâu. Cần ngắt cầu dao điện khẩn cấp, di chuyển lên vị trí cao và gọi xuồng cứu hộ 112 / 0208.3855.112."
        elif depth_cm >= 40 or level in ("KNEE", "WAIST"):
            risk_label = f"CẢNH BÁO MỨC 2 — NGUY HIỂM ({level})"
            advice = "⚠️ Nước ngập qua bánh xe máy / gầm ô tô. Tuyệt đối không cố đi xe qua vùng ngập, ưu tiên sơ tán người già và trẻ nhỏ."
        elif depth_cm >= 10 or level in ("PUDDLE", "ANKLE"):
            risk_label = f"CẢNH BÁO MỨC 1 — NGẬP NHẸ ({level})"
            advice = "ℹ️ Mặt đường có nước ngập nhẹ (~mắt cá chân). Các phương tiện di chuyển chậm, chú ý nắp cống và ổ gà."
        else:
            risk_label = "AN TOÀN — KHÔNG PHÁT HIỆN NGẬP SÂU"
            advice = "✅ Không phát hiện mực nước ngập nguy hiểm trong ảnh."

        if vehs:
            advice += f" (Đối tượng tham chiếu phát hiện: {', '.join(str(v) for v in vehs)})."

        return {
            "ok": True,
            "image_type": "flood" if depth_cm >= 10 else "dry",
            "depth_cm": depth_cm,
            "flood_level": level,
            "confidence": max(conf, 50.0) if depth_cm > 0 else conf,
            "risk_label": risk_label,
            "advice": advice,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route("/api/depth/instant", methods=["POST"])
def api_depth_instant():
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image_b64") or data.get("data_b64")
    if not image_b64:
        return jsonify({"error": "Thiếu dữ liệu ảnh (image_b64)"}), 400
    try:
        res = _run_instant_depth(image_b64)
        return jsonify(res)
    except Exception as exc:
        log.warning(f"[api/depth/instant] {exc}")
        return jsonify({"error": str(exc)}), 500


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


def _detect_location(image_b64: Optional[str] = None,
                     browser_lat: Optional[float] = None,
                     browser_lon: Optional[float] = None,
                     browser_acc: Optional[float] = None,
                     client_ip: Optional[str] = None) -> Optional[dict]:
    """Chạy GeoLocalizer qua GPS EXIF ảnh -> WiFi/Browser -> OCR -> IP."""
    import tempfile, base64, os
    tmp_path = None
    try:
        from utils.geo_localizer import GeoLocalizer
        if image_b64:
            raw = base64.b64decode(image_b64)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(raw)
                tmp_path = f.name
        
        loc = GeoLocalizer()
        est = loc.localize(
            image_path=Path(tmp_path) if tmp_path else None,
            browser_lat=browser_lat,
            browser_lon=browser_lon,
            browser_accuracy=browser_acc,
            client_ip=client_ip
        )
        if est and est.has_coords():
            method_labels = {
                "gps_exif": "GPS chụp từ ảnh (Độ chính xác cao)",
                "browser": "Định vị thiết bị / WiFi",
                "ocr_street": "Nhận diện tên đường từ ảnh",
                "ocr_district": "Nhận diện quận/huyện từ ảnh",
                "geoclip": "AI dự đoán từ cảnh quan ảnh",
                "ip": "Vị trí mạng Internet (IP)",
            }
            return {
                "lat": round(est.lat, 5),
                "lon": round(est.lon, 5),
                "address": est.address or f"Khu vực Thái Nguyên ({est.lat:.4f}, {est.lon:.4f})",
                "method": est.method,
                "method_label": method_labels.get(est.method, "Định vị tự động"),
                "radius_m": est.radius_m,
                "confidence": est.confidence,
            }
    except Exception as exc:
        log.warning(f"[detect_location] {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except Exception: pass
    return None


@app.route("/api/geo/detect", methods=["POST"])
def api_geo_detect():
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image_b64")
    browser_lat = data.get("browser_lat")
    browser_lon = data.get("browser_lon")
    browser_acc = data.get("browser_acc")
    client_ip = request.remote_addr or request.headers.get("X-Forwarded-For", "")
    
    geo = _detect_location(
        image_b64=image_b64,
        browser_lat=float(browser_lat) if browser_lat is not None else None,
        browser_lon=float(browser_lon) if browser_lon is not None else None,
        browser_acc=float(browser_acc) if browser_acc is not None else None,
        client_ip=client_ip
    )
    if geo:
        return jsonify({"ok": True, "geo": geo})
    return jsonify({"ok": False, "msg": "Không xác định được tọa độ"})


# ══════════════════════════════════════════════════════════════════════════════
# API — CHAT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat/message", methods=["POST"])
def api_chat_message():
    data      = request.get_json() or {}
    user_msg  = data.get("message", "").strip()
    sid       = data.get("session_id", "default")
    image_b64 = data.get("image_b64")
    browser_lat = data.get("browser_lat")
    browser_lon = data.get("browser_lon")
    browser_acc = data.get("browser_acc")
    client_ip = request.remote_addr or request.headers.get("X-Forwarded-For", "")

    if not user_msg and not image_b64:
        return jsonify({"error": "Thiếu nội dung"}), 400

    if not _llm_ready:
        status = "đang tải model…" if _llm_loading else "chưa khởi động"
        return jsonify({"reply": f"⏳ FloodAgent {status}. Vui lòng thử lại sau 30 giây.",
                        "loading": True})

    with _sess_lock:
        hist = _sessions.setdefault(sid, [])

    content = user_msg
    pil_img = None
    geo_info = None
    if image_b64:
        try:
            import base64
            import io
            from PIL import Image as _PILImage
            pil_img = _PILImage.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        except Exception as _img_exc:
            log.debug(f"[chat] decode PIL image failed: {_img_exc}")
        analysis = _analyze_image(image_b64)
        content  = (analysis + ("\n" + user_msg if user_msg else "")).strip()

        # Tự động trích xuất vị trí của người gửi ảnh
        geo_info = _detect_location(
            image_b64=image_b64,
            browser_lat=float(browser_lat) if browser_lat is not None else None,
            browser_lon=float(browser_lon) if browser_lon is not None else None,
            browser_acc=float(browser_acc) if browser_acc is not None else None,
            client_ip=client_ip
        )

    messages = [{"role": "system", "content": CHAT_SYS}]
    for h in hist[-MAX_HIST:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": content})

    try:
        reply = llm_infer(messages, max_new_tokens=256, image=pil_img)
        # Strip structured format if model still outputs it
        if "INTENT:" in reply:
            from learning.flood_llm import _parse_llm_output
            reply = _parse_llm_output(reply).response or reply
        reply = _clean_vietnamese_reply(reply)
    except Exception as exc:
        return jsonify({"error": f"Lỗi inference: {exc}"}), 500

    with _sess_lock:
        hist.append({"role": "user",      "content": content})
        hist.append({"role": "assistant", "content": reply})
        _sessions[sid] = hist[-MAX_HIST:]

    return jsonify({"reply": reply, "geo": geo_info})


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
                # [v4] Scene quality metadata
                "scene_score": result.get("scene_score", None),
                "is_night":    result.get("is_night", None),
                "needs_review": result.get("needs_review", False),
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


@app.route("/api/jobs/<job_id>", methods=["GET"])
@login_required
def api_job_status(job_id: str):
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
# ADMIN API — Data helpers
# ══════════════════════════════════════════════════════════════════════════════
_DATA_DIR = BASE_DIR / "data"

def _has_role(*roles: str) -> bool:
    r = session.get("role", "")
    return r == "admin" or r in roles

def _forbid():
    return jsonify({"error": "Không đủ quyền"}), 403

def _load_json(filename: str, default=None):
    path = _DATA_DIR / filename
    if not path.exists():
        return [] if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [] if default is None else default

def _save_json(filename: str, data) -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    (_DATA_DIR / filename).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def _now_iso() -> str:
    from datetime import datetime as _dt
    return _dt.now().isoformat(timespec="seconds")


# ── Alerts ─────────────────────────────────────────────────────────────────
@app.route("/api/admin/alerts", methods=["GET"])
@login_required
def api_admin_alerts_list():
    return jsonify(_load_json("alerts.json"))

@app.route("/api/admin/alerts", methods=["POST"])
@login_required
def api_admin_alerts_create():
    if not _has_role("alert_mgr", "editor"):
        return _forbid()
    data = request.get_json(silent=True) or {}
    alerts = _load_json("alerts.json")
    alert = {
        "id":              str(int(time.time() * 1000)),
        "location":        data.get("location", ""),
        "level":           data.get("level", "knee"),
        "message":         data.get("message", ""),
        "recommendations": data.get("recommendations", ""),
        "status":          "active",
        "created_at":      _now_iso(),
        "updated_at":      _now_iso(),
        "expires_at":      data.get("expires_at", ""),
        "created_by":      session.get("email", "admin"),
        "job_id":          data.get("job_id", ""),
    }
    alerts.insert(0, alert)
    _save_json("alerts.json", alerts)
    return jsonify({"ok": True, "alert": alert})

@app.route("/api/admin/alerts/<alert_id>", methods=["PATCH"])
@login_required
def api_admin_alerts_update(alert_id: str):
    if not _has_role("alert_mgr", "editor"):
        return _forbid()
    data = request.get_json(silent=True) or {}
    alerts = _load_json("alerts.json")
    for a in alerts:
        if a.get("id") == alert_id:
            for k, v in data.items():
                if k not in ("id", "created_at", "created_by"):
                    a[k] = v
            a["updated_at"] = _now_iso()
            _save_json("alerts.json", alerts)
            return jsonify({"ok": True, "alert": a})
    return jsonify({"error": "Không tìm thấy"}), 404

@app.route("/api/admin/alerts/<alert_id>", methods=["DELETE"])
@login_required
def api_admin_alerts_delete(alert_id: str):
    if not _has_role("alert_mgr"):
        return _forbid()
    alerts = [a for a in _load_json("alerts.json") if a.get("id") != alert_id]
    _save_json("alerts.json", alerts)
    return jsonify({"ok": True})


# ── Flood Events ────────────────────────────────────────────────────────────
@app.route("/api/admin/events", methods=["GET"])
@login_required
def api_admin_events_list():
    return jsonify(_load_json("events.json"))

@app.route("/api/admin/events", methods=["POST"])
@login_required
def api_admin_events_create():
    if not _has_role("editor", "alert_mgr"):
        return _forbid()
    data = request.get_json(silent=True) or {}
    events = _load_json("events.json")
    event = {
        "id":            str(int(time.time() * 1000)),
        "name":          data.get("name", "Sự kiện ngập"),
        "location":      data.get("location", ""),
        "status":        data.get("status", "active"),
        "max_level":     data.get("max_level", "unknown"),
        "current_level": data.get("current_level", "unknown"),
        "report_count":  int(data.get("report_count", 1)),
        "notes":         data.get("notes", ""),
        "created_at":    _now_iso(),
        "updated_at":    _now_iso(),
        "created_by":    session.get("email", "admin"),
        "job_ids":       data.get("job_ids", []),
    }
    events.insert(0, event)
    _save_json("events.json", events)
    return jsonify({"ok": True, "event": event})

@app.route("/api/admin/events/<event_id>", methods=["PATCH"])
@login_required
def api_admin_events_update(event_id: str):
    if not _has_role("editor", "alert_mgr"):
        return _forbid()
    data = request.get_json(silent=True) or {}
    events = _load_json("events.json")
    for e in events:
        if e.get("id") == event_id:
            for k, v in data.items():
                if k not in ("id", "created_at", "created_by"):
                    e[k] = v
            e["updated_at"] = _now_iso()
            _save_json("events.json", events)
            return jsonify({"ok": True, "event": e})
    return jsonify({"error": "Không tìm thấy"}), 404

@app.route("/api/admin/events/<event_id>", methods=["DELETE"])
@login_required
def api_admin_events_delete(event_id: str):
    events = [e for e in _load_json("events.json") if e.get("id") != event_id]
    _save_json("events.json", events)
    return jsonify({"ok": True})


# ── Admin Copilot ───────────────────────────────────────────────────────────
_COPILOT_SYS = (
    "Bạn là AI Copilot hỗ trợ admin kiểm duyệt tin tức ngập lụt tại Việt Nam. "
    "Phân tích dữ liệu AI và đưa ra đề xuất cụ thể, ngắn gọn bằng tiếng Việt. "
    "Tối đa 3–4 câu. Không bịa thêm thông tin ngoài dữ liệu được cung cấp."
)
_COPILOT_FALLBACKS: Dict[str, str] = {
    "summarize": "Báo cáo ngập tại khu vực được ghi nhận. Mức độ và độ tin cậy từ kết quả AI.",
    "publish":   "Kiểm tra: vị trí đã xác minh, ảnh đã blur, mức ngập hợp lý, có khuyến cáo.",
    "title":     "Ngập lụt tại khu vực — cập nhật từ hệ thống AI",
    "alert":     "Cảnh báo ngập lụt tại khu vực. Người dân chú ý và hạn chế di chuyển.",
    "compare":   "So sánh với báo cáo trước để xác định xu hướng ngập tăng hay giảm.",
}

@app.route("/api/admin/copilot", methods=["POST"])
@login_required
def api_admin_copilot():
    data    = request.get_json(silent=True) or {}
    action  = data.get("action", "summarize")
    job_raw = data.get("job", {})
    custom  = data.get("custom_prompt", "")

    if not _llm_ready:
        return jsonify({"reply": _COPILOT_FALLBACKS.get(action, "Model chưa sẵn sàng."), "fallback": True})

    job_txt = json.dumps(job_raw, ensure_ascii=False)[:500]
    prompts = {
        "summarize": f"Tóm tắt ngắn báo cáo ngập này:\n{job_txt}",
        "publish":   f"Có nên đăng báo cáo này lên công khai không? Giải thích:\n{job_txt}",
        "title":     f"Viết tiêu đề báo tin (dưới 15 từ) cho báo cáo:\n{job_txt}",
        "alert":     f"Viết nội dung cảnh báo ngắn (1–2 câu) từ báo cáo:\n{job_txt}",
        "compare":   f"Đề xuất hành động dựa trên báo cáo này so với tình hình chung:\n{job_txt}",
    }
    prompt = custom if custom else prompts.get(action, f"Phân tích báo cáo:\n{job_txt}")

    try:
        reply = llm_infer([
            {"role": "system", "content": _COPILOT_SYS},
            {"role": "user",   "content": prompt},
        ], max_new_tokens=200).strip()
        return jsonify({"reply": reply})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Config ──────────────────────────────────────────────────────────────────
@app.route("/api/admin/config", methods=["GET"])
@login_required
def api_admin_config_get():
    try:
        cfg_path = BASE_DIR / "config.yaml"
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {} if cfg_path.exists() else {}
        return jsonify(raw)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/api/admin/config", methods=["POST"])
@login_required
def api_admin_config_update():
    if not _has_role():  # admin only
        return _forbid()
    data = request.get_json(silent=True) or {}
    try:
        cfg_path = BASE_DIR / "config.yaml"
        current  = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {} if cfg_path.exists() else {}
        current.update(data)
        cfg_path.write_text(yaml.dump(current, allow_unicode=True, default_flow_style=False), encoding="utf-8")
        log.info("[Config] Updated by %s: %s", session.get("email"), list(data.keys()))
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Sources ─────────────────────────────────────────────────────────────────
@app.route("/api/admin/sources", methods=["GET"])
@login_required
def api_admin_sources():
    if not _has_role("editor", "alert_mgr"):
        return _forbid()
    from pipeline.job_queue import JobQueue
    jobs = JobQueue.instance().list_recent(300)
    sources: Dict[str, dict] = {}
    for j in jobs:
        src = (j.result_summary or {}).get("source") or "Hệ thống"
        if src not in sources:
            sources[src] = {"name": src, "total": 0, "done": 0, "failed": 0, "last_at": ""}
        sources[src]["total"] += 1
        if j.status == "done":   sources[src]["done"]   += 1
        if j.status == "failed": sources[src]["failed"] += 1
        if (j.created_at or "") > sources[src]["last_at"]:
            sources[src]["last_at"] = j.created_at or ""
    return jsonify(sorted(sources.values(), key=lambda x: -x["total"]))


# ── Job review actions ───────────────────────────────────────────────────────
_job_review_states: Dict[str, str] = {}

@app.route("/api/admin/jobs/<job_id>/review", methods=["POST"])
@login_required
def api_admin_job_review(job_id: str):
    data   = request.get_json(silent=True) or {}
    action = data.get("action", "")
    if action not in ("approve", "reject", "verify", "alert", "update"):
        return jsonify({"error": "action không hợp lệ"}), 400
    _job_review_states[job_id] = action
    log.info("[Review] Job %s → %s by %s", job_id, action, session.get("email", "?"))
    return jsonify({"ok": True, "job_id": job_id, "action": action})

@app.route("/api/admin/review-states", methods=["GET"])
@login_required
def api_admin_review_states():
    return jsonify(_job_review_states)


# ── Audit log ───────────────────────────────────────────────────────────────
@app.route("/api/admin/audit", methods=["GET"])
@login_required
def api_admin_audit():
    try:
        from auth.user_manager import UserManager
        um = UserManager.instance() if hasattr(UserManager, "instance") else None
        if um and hasattr(um, "get_audit_log"):
            return jsonify(um.get_audit_log())
    except Exception:
        pass
    return jsonify([])


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
    import argparse
    parser = argparse.ArgumentParser(description="FloodAI Web Application")
    parser.add_argument("--port", "-p", type=int, default=int(os.environ.get("PORT", 5000)),
                        help="Port để chạy web server (mặc định: 5000 hoặc biến môi trường PORT)")
    parser.add_argument("--host", "-H", type=str, default=os.environ.get("HOST", "0.0.0.0"),
                        help="Host để lắng nghe (mặc định: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Bật chế độ debug của Flask")
    args, _ = parser.parse_known_args()

    port = args.port
    host = args.host

    start_llm()
    log.info(f"FloodAI Web → http://{host}:{port}")
    app.run(debug=args.debug, port=port, host=host)
