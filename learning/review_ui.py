# -*- coding: utf-8 -*-

import base64
import hashlib
import json
import logging
import shutil
import sqlite3
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from functools import wraps

from flask import (Flask, g, jsonify, redirect, render_template,
                   request, send_file, session, url_for)

# ── Path bootstrap ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent          # learning/
ROOT_DIR   = BASE_DIR.parent               # project root
sys.path.insert(0, str(ROOT_DIR))

from learning.active_learner import ActiveLearnerV2 as ActiveLearner

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT_DIR / "review_ui.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("review_ui")

app = Flask(__name__)
app.secret_key = "floodai-s3cr3t-k3y-change-in-prod"

# ── Auth ───────────────────────────────────────────────────────────────────────
_ADMIN_USER      = "admin"
_ADMIN_PASS_HASH = hashlib.sha256(b"123456").hexdigest()


def login_required(f):
    @wraps(f)
    def _wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            if request.is_json or request.method != "GET":
                return jsonify({"error": "Chưa đăng nhập"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return _wrapped


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_DB_PATH = BASE_DIR / "training_images.db"
TRAIN_IMG_DIR = BASE_DIR / "training_images"
LIVE_DIR      = BASE_DIR / "_live_results"

def get_learner() -> ActiveLearner:
    if "learner" not in g:
        g.learner = ActiveLearner()
    return g.learner

@app.teardown_appcontext
def close_resources(exc=None):
    lrn = g.pop("learner", None)
    if lrn:
        try: lrn.close()
        except Exception: pass
    tc = g.pop("train_conn", None)
    if tc:
        try: tc.close()
        except Exception: pass

def get_train_conn():
    if "train_conn" not in g:
        conn = sqlite3.connect(str(TRAIN_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS training_images (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                added_at    TEXT    NOT NULL,
                image_path  TEXT    NOT NULL,
                image_hash  TEXT    UNIQUE,
                actual_depth REAL   NOT NULL,
                actual_level TEXT   NOT NULL,
                notes        TEXT   DEFAULT '',
                source       TEXT   DEFAULT 'manual',
                verified     INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_ti_level ON training_images(actual_level);
        """)
        conn.commit()
        g.train_conn = conn
    return g.train_conn

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STATE
# ══════════════════════════════════════════════════════════════════════════════

_pl = {
    "status":    "idle",   # idle | running | done | error
    "run_id":    None,
    "log":       [],       # [{"text":str, "level":str, "ts":str}]
    "step":      0,        # current step index (0-4)
    "stop_flag": False,
}
_pl_lock = threading.Lock()

STEPS = [
    "Tải ảnh",
    "Sao chép",
    "Địa điểm",
    "Depth AI",
    "Báo cáo",
]

class _WebLogHandler(logging.Handler):
    """Captures root logger records → pipeline log."""
    def emit(self, record):
        try:
            level = record.levelname
            text  = self.format(record)
            ts    = datetime.now().strftime("%H:%M:%S")
            with _pl_lock:
                _pl["log"].append({"text": text, "level": level, "ts": ts})
        except Exception:
            pass

def _plog(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    with _pl_lock:
        _pl["log"].append({"text": msg, "level": level, "ts": ts})

def _pstep(idx: int):
    with _pl_lock:
        _pl["step"] = idx

def _stopped() -> bool:
    with _pl_lock:
        return _pl["stop_flag"]

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE THREAD
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(cfg: dict):
    web_handler = _WebLogHandler()
    web_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    ))
    root = logging.getLogger()
    root.addHandler(web_handler)

    try:
        from utils.constants import (
            IMAGE_EXTENSIONS, DRIVE_UPLOAD_EXTENSIONS,
            OUTPUT_FOLDER_ORIGINAL, OUTPUT_FOLDER_OVERLAY,
            OUTPUT_FOLDER_DEPTHMAP, OUTPUT_FOLDER_TMP_DEPTH,
            PIPELINE_SUMMARY_JSON,
        )
        from utils.constants import DEFAULT_DINO_MODEL, DEFAULT_POSE_MODEL

        run_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root     = Path(cfg.get("output_dir", "output")) / run_id
        original_dir = out_root / OUTPUT_FOLDER_ORIGINAL
        overlay_dir  = out_root / OUTPUT_FOLDER_OVERLAY
        depthmap_dir = out_root / OUTPUT_FOLDER_DEPTHMAP

        for d in (original_dir, overlay_dir, depthmap_dir):
            d.mkdir(parents=True, exist_ok=True)

        with _pl_lock:
            _pl["run_id"] = run_id

        results = {
            "run_id": run_id, "query": "N/A", "sources": [],
            "raw": [], "filtered": [], "depth_data": [],
        }

        # ── STEP 0: Load images ────────────────────────────────────────────
        _pstep(0)
        mode = cfg.get("mode", "local")
        _plog(f"═══ STEP 1/5 · Tải ảnh [{mode.upper()}] ═══")
        images = []

        if mode == "local":
            img_dir = Path(cfg.get("input_path", ""))
            if not img_dir.exists():
                _plog(f"❌ Folder không tồn tại: {img_dir}", "ERROR")
                with _pl_lock: _pl["status"] = "error"
                return
            images = sorted([
                f for f in img_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ])
            _plog(f"✅ Tải {len(images)} ảnh từ {img_dir}", "SUCCESS")

        elif mode == "drive":
            from uploader.drive_uploader import DriveUploader
            _plog("☁️  Đang kết nối Google Drive...")
            images = DriveUploader().download_folder(
                drive_folder=cfg.get("input_path", ""),
                local_dir=original_dir,
                extensions=IMAGE_EXTENSIONS,
            )
            _plog(f"✅ Tải {len(images)} ảnh từ Drive", "SUCCESS")

        results["raw"] = results["filtered"] = images

        if not images:
            _plog("❌ Không có ảnh nào để xử lý.", "ERROR")
            with _pl_lock: _pl["status"] = "error"
            return

        if _stopped(): _plog("⏹ Đã dừng.", "WARNING"); return

        # ── STEP 1: Copy to output ─────────────────────────────────────────
        _pstep(1)
        _plog(f"═══ STEP 2/5 · Sao chép {len(images)} ảnh sang output ═══")
        original_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for src in images:
            dst = original_dir / Path(src).name
            # Bỏ qua nếu src và dst là cùng một file
            if Path(src).resolve() == dst.resolve():
                copied.append(dst)
                continue
            # Retry nếu file bị lock bởi process khác (Windows WinError 32)
            for attempt in range(5):
                try:
                    shutil.copy2(str(src), str(dst))
                    break
                except PermissionError as e:
                    if attempt < 4:
                        import time as _t; _t.sleep(0.5)
                    else:
                        _plog(f"⚠️  Bỏ qua {Path(src).name}: file đang bị khóa ({e})", "WARNING")
                        dst = src  # fallback: dùng file gốc
            copied.append(dst)
        images = copied
        results["filtered"] = images
        _plog(f"✅ Sao chép xong: {len(images)} ảnh", "SUCCESS")

        if _stopped(): _plog("⏹ Đã dừng.", "WARNING"); return

        # ── STEP 2: Location ───────────────────────────────────────────────
        _pstep(2)
        location_map = {}
        if cfg.get("detect_location", True):
            _plog("═══ STEP 3/5 · Nhận diện địa điểm ═══")
            from utils.location_detector import LocationDetector
            loc_list = LocationDetector(
                google_maps_key=cfg.get("google_maps_key", ""),
                use_ocr=True, use_plate=True, use_exif=True,
            ).detect_batch([Path(p) for p in images])
            located      = sum(1 for r in loc_list if r.method != "none")
            location_map = {r.image_path: r for r in loc_list}
            results["location_map"] = {k: vars(v) for k, v in location_map.items()}
            _plog(f"✅ Địa điểm: {located}/{len(images)} ảnh có vị trí", "SUCCESS")
        else:
            _plog("⏭  Bỏ qua nhận diện địa điểm.")

        if _stopped(): _plog("⏹ Đã dừng.", "WARNING"); return

        # ── STEP 5: Depth Analysis ─────────────────────────────────────────
        _pstep(3)
        depth_results = []
        if not cfg.get("skip_depth", False):
            _plog("═══ STEP 4/5 · Depth Anything V2 + YOLO ═══")
            from depth_analysis.reference_estimator import ReferenceEstimator
            tmp = out_root / OUTPUT_FOLDER_TMP_DEPTH
            tmp.mkdir(exist_ok=True)

            estimator = ReferenceEstimator(
                yolo_model   = cfg.get("yolo_model",  "yolov8n.pt"),
                depth_model  = cfg.get("depth_model", "depth-anything/Depth-Anything-V2-Small-hf"),
                output_dir   = tmp,
                conf_thresh  = float(cfg.get("yolo_conf", 0.35)),
                use_dino     = cfg.get("use_dino", True),
                dino_model   = cfg.get("dino_model",  DEFAULT_DINO_MODEL),
                use_pose     = cfg.get("use_pose",  True),
                pose_model   = cfg.get("pose_model",  DEFAULT_POSE_MODEL),
                use_segformer= cfg.get("use_segformer", True),
            )
            chunk = cfg.get("depth_chunk_size", 8)
            _plog(f"🔍 Phân tích {len(images)} ảnh, chunk_size={chunk}…")
            # [FIX] Truyền _stopped vào để Stop button có thể dừng giữa chừng
            depth_results = estimator.analyze_batch(
                images, chunk_size=chunk, stop_check=_stopped
            )
            estimator.unload_heavy_models()

            for r in depth_results:
                for attr, dest in [("overlay_path", overlay_dir), ("depth_map_path", depthmap_dir)]:
                    src = Path(getattr(r, attr, "") or "")
                    if src.exists():
                        dst = dest / src.name
                        shutil.move(str(src), str(dst))
                        setattr(r, attr, str(dst))
            shutil.rmtree(tmp, ignore_errors=True)
            results["depth_data"] = depth_results
            _plog(f"✅ Depth xong: {len(depth_results)}/{len(images)} ảnh", "SUCCESS")

            # Self-learning
            try:
                from learning_update import SelfLearningPipeline
                sl = SelfLearningPipeline()
                sl.process_results(depth_results=depth_results, cfg=cfg, image_paths=images)
                sl.close()
                _plog("🧠 Self-learning: dữ liệu đã được xếp hàng review.", "SUCCESS")
            except Exception as e:
                _plog(f"⚠️  Self-learning skipped: {e}", "WARNING")
        else:
            _plog("⏭  Bỏ qua Depth Analysis.")

        if _stopped(): _plog("⏹ Đã dừng.", "WARNING"); return

        # ── STEP 6: Report + Drive ─────────────────────────────────────────
        _pstep(4)
        _plog("═══ STEP 5/5 · Tạo báo cáo ═══")
        try:
            from utils.report_generator import ReportGeneratorV2
            csv_p, html_p = ReportGeneratorV2(output_dir=out_root).generate(results)
            _plog(f"✅ HTML → {html_p}", "SUCCESS")
            _plog(f"✅ CSV  → {csv_p}",  "SUCCESS")
        except Exception as e:
            _plog(f"⚠️  HTML/CSV report failed: {e}", "WARNING")

        try:
            from utils.excel_reporter import ExcelReporter
            xlsx_p = ExcelReporter(output_dir=out_root).generate(results)
            if xlsx_p:
                _plog(f"✅ XLSX → {xlsx_p}", "SUCCESS")
        except Exception as e:
            _plog(f"⚠️  Excel report skipped: {e}", "WARNING")

        if not cfg.get("skip_drive", True) and not _stopped():
            _plog("☁️  Đang upload lên Google Drive…")
            try:
                from uploader.drive_uploader import DriveUploader
                fid = DriveUploader().upload_folder(
                    local_dir   = out_root,
                    folder_name = f"{cfg.get('drive_folder','FloodAnalysis')}/{run_id}",
                    extensions  = DRIVE_UPLOAD_EXTENSIONS,
                )
                results["drive_folder_id"] = fid
                _plog(f"✅ Drive → https://drive.google.com/drive/folders/{fid}", "SUCCESS")
            except Exception as e:
                _plog(f"❌ Drive upload thất bại: {e}", "ERROR")

        # ── Summary JSON ───────────────────────────────────────────────────
        dd     = results.get("depth_data", [])
        counts = {}
        for r in dd:
            lvl = r.flood_level if hasattr(r, "flood_level") else r.get("flood_level", "?")
            counts[lvl] = counts.get(lvl, 0) + 1

        summary = {
            "run_id":         run_id,
            "query":          "N/A",
            "sources":        [],
            "total_crawled":  len(results.get("raw", [])),
            "total_filtered": len(results.get("filtered", [])),
            "total_analyzed": len(dd),
            "drive_folder":   results.get("drive_folder_id", "N/A"),
            "flood_summary":  counts,
        }
        try:
            (out_root / PIPELINE_SUMMARY_JSON).write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)
            )
        except Exception:
            pass

        _plog("─" * 52)
        _plog(f"  Run ID       : {run_id}")
        _plog(f"  Output       : {out_root}")
        _plog(f"  Ảnh đầu vào : {len(results.get('raw', []))}")
        _plog(f"  Đã phân tích : {len(dd)}")
        for lvl, cnt in sorted(counts.items()):
            _plog(f"    {lvl:<14}: {cnt}")
        _plog("═══ ✅ PIPELINE HOÀN THÀNH ═══", "SUCCESS")

        with _pl_lock:
            _pl["status"] = "done"
            _pl["step"]   = len(STEPS)

    except Exception as exc:
        _plog(f"❌ Lỗi pipeline: {exc}", "ERROR")
        _plog(traceback.format_exc(), "ERROR")
        with _pl_lock:
            _pl["status"] = "error"
    finally:
        root.removeHandler(web_handler)

# ══════════════════════════════════════════════════════════════════════════════
# LIVE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

_live_estimator = None

def _invalidate_live_estimator():
    global _live_estimator
    _live_estimator = None

# ══════════════════════════════════════════════════════════════════════════════
# AI LEARNER SINGLETON  (dùng chung toàn bộ server, không tạo mới mỗi request)
# ══════════════════════════════════════════════════════════════════════════════

_ai_singleton      = None
_ai_singleton_lock = threading.Lock()

# ══════════════════════════════════════════════════════════════════════════════
# FLOOD AGENT SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_flood_agent      = None
_flood_agent_lock = threading.Lock()

def get_flood_agent():
    """Trả về instance FloodAgent duy nhất, khởi tạo lazy."""
    global _flood_agent
    with _flood_agent_lock:
        if _flood_agent is None:
            try:
                from agent.flood_agent import FloodAgent
                _flood_agent = FloodAgent()
                log.info("[FloodAgent-singleton] Khởi tạo xong")
            except Exception as e:
                log.warning(f"[FloodAgent-singleton] Không khởi tạo được: {e}")
        return _flood_agent

def get_ai_singleton():
    """Trả về instance AiLearner duy nhất, khởi tạo lazy."""
    global _ai_singleton
    with _ai_singleton_lock:
        if _ai_singleton is None:
            try:
                from learning.ai_learner import AiLearner
                _ai_singleton = AiLearner()
                log.info("[AI-singleton] Khởi tạo xong")
            except Exception as e:
                log.warning(f"[AI-singleton] Không khởi tạo được: {e}")
        return _ai_singleton

def _trigger_ai_retrain():
    """Invalidate cache và retrain AI ngay trong background thread.
    Gọi sau mỗi lần có data mới (upload / feedback / review).
    """
    def _worker():
        with _ai_singleton_lock:
            ai = _ai_singleton
        if ai is None:
            return
        try:
            ai.invalidate()          # xóa cache file
            result = ai.train()      # retrain ngay lập tức
            log.info(f"[AI-retrain] Xong — {result.get('n_cases', 0)} cases, "
                     f"kNN={result.get('n_knn', 0)}")
        except Exception as e:
            log.warning(f"[AI-retrain] Lỗi: {e}")

    threading.Thread(target=_worker, daemon=True, name="ai-retrain").start()

def _get_live_estimator():
    global _live_estimator
    if _live_estimator is not None:
        return _live_estimator, None
    try:
        from depth_analysis.reference_estimator import ReferenceEstimator
        import yaml as _yaml
        cfg_path = ROOT_DIR / "config.yaml"
        raw_cfg  = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        # Apply adaptive thresholds
        try:
            from learning_update import SelfLearningPipeline
            sl      = SelfLearningPipeline()
            raw_cfg = sl.get_adaptive_config(raw_cfg)
            sl.close()
        except Exception:
            pass
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        _live_estimator = ReferenceEstimator(
            yolo_model   = raw_cfg.get("yolo_model",  "yolov8n.pt"),
            depth_model  = raw_cfg.get("depth_model", "depth-anything/Depth-Anything-V2-Small-hf"),
            output_dir   = BASE_DIR / "_live_tmp",
            conf_thresh  = float(raw_cfg.get("yolo_conf", 0.35)),
            use_dino     = raw_cfg.get("use_dino", True),
            dino_model   = raw_cfg.get("dino_model", "facebook/dinov2-small"),
            use_pose     = raw_cfg.get("use_pose", True),
            pose_model   = raw_cfg.get("pose_model", "yolov8n-pose.pt"),
            use_segformer= raw_cfg.get("use_segformer", True),
        )
        return _live_estimator, None
    except Exception as e:
        return None, str(e)






# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES — Review
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return '', 204


# ── Auth routes ────────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if (username == _ADMIN_USER and
                hashlib.sha256(password.encode()).hexdigest() == _ADMIN_PASS_HASH):
            session['logged_in'] = True
            session['username']  = username
            return redirect(url_for('index'))
        error = "Sai tên đăng nhập hoặc mật khẩu."
    return render_template('login.html', error=error)


@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    return redirect(url_for('login_page'))


@app.route('/live', methods=['GET'])
def live_public():
    """Trang Live Model công khai — không cần đăng nhập."""
    if session.get('logged_in'):
        return redirect(url_for('index'))
    return render_template('live.html')


@app.route('/')
@login_required
def index():
    learner = get_learner()
    stats   = {k: (v or 0) for k, v in learner.get_review_stats().items()}
    cases   = learner.get_pending_reviews(max_count=5, priority_filter=1)
    return render_template('dashboard.html', stats=stats, cases=cases)


@app.route('/review/<int:case_id>', methods=['POST'])
@login_required
def submit_review(case_id):
    d = request.json
    get_learner().submit_review(
        case_id      = case_id,
        actual_depth = d['actual_depth'],
        actual_level = d['actual_level'],
        reviewed_by  = 'web_ui',
        notes        = d.get('notes', ''),
    )
    # Đảm bảo singleton tồn tại rồi retrain ngay trong background
    get_ai_singleton()
    _trigger_ai_retrain()
    return jsonify({"status": "ok"})


@app.route('/skip/<int:case_id>', methods=['POST'])
@login_required
def skip_case(case_id):
    lrn = get_learner()
    lrn.conn.execute("UPDATE review_queue SET status='skipped' WHERE id=?", (case_id,))
    lrn.conn.commit()
    return jsonify({"status": "ok"})


@app.route('/image/<int:case_id>', methods=['GET'])
@login_required
def get_image(case_id):
    case = get_learner()._get_case_by_id(case_id)
    if case:
        p = Path(case.image_path)
        for candidate in (p, TRAIN_IMG_DIR / p.name):
            if candidate.exists():
                try: return send_file(str(candidate.resolve()))
                except Exception: pass
    return "Image not found", 404


@app.route('/run_image/<path:rel_path>', methods=['GET'])
@login_required
def run_image(rel_path):
    output_root = ROOT_DIR / "output"
    full = (output_root / rel_path).resolve()
    try: full.relative_to(output_root.resolve())
    except ValueError: return "Forbidden", 403
    if full.exists():
        try: return send_file(str(full))
        except Exception: pass
    return "Not found", 404


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES — Training data
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/upload_training', methods=['POST'])
@login_required
def upload_training():
    d = request.json or {}
    required = {'filename', 'data_b64', 'actual_depth', 'actual_level'}
    if not required.issubset(d):
        return jsonify({"error": "missing fields"}), 400
    if not d.get('actual_level'):
        return jsonify({"error": "actual_level required"}), 400

    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    raw       = base64.b64decode(d['data_b64'])
    img_hash  = hashlib.sha256(raw).hexdigest()[:12]
    suffix    = Path(d['filename']).suffix.lower() or '.jpg'
    save_path = TRAIN_IMG_DIR / f"{img_hash}{suffix}"
    save_path.write_bytes(raw)

    conn = get_train_conn()
    try:
        conn.execute(
            "INSERT INTO training_images (added_at,image_path,image_hash,actual_depth,actual_level,notes,source) "
            "VALUES (?,?,?,?,?,?,'manual')",
            (datetime.now().isoformat(), str(save_path), img_hash,
             float(d['actual_depth']), d['actual_level'], d.get('notes', '')),
        )
        conn.commit()
    except Exception as e:
        return jsonify({"status": "duplicate", "msg": str(e)}), 409

    # Retrain AI ngay (không cần tắt/bật server)
    get_ai_singleton()
    _trigger_ai_retrain()

    return jsonify({"status": "ok", "path": str(save_path)})


@app.route('/training_stats', methods=['GET'])
@login_required
def training_stats():
    conn     = get_train_conn()
    total    = conn.execute("SELECT COUNT(*) FROM training_images").fetchone()[0]
    by_level = dict(conn.execute(
        "SELECT actual_level, COUNT(*) FROM training_images GROUP BY actual_level"
    ).fetchall())
    recent   = [dict(r) for r in conn.execute(
        "SELECT * FROM training_images ORDER BY id DESC LIMIT 50"
    ).fetchall()]
    return jsonify({"total": total, "by_level": by_level, "recent": recent})


@app.route('/delete_training/<int:tid>', methods=['DELETE'])
@login_required
def delete_training(tid):
    conn = get_train_conn()
    row  = conn.execute("SELECT image_path FROM training_images WHERE id=?", (tid,)).fetchone()
    if row:
        p = Path(row['image_path'])
        p.unlink(missing_ok=True)
        conn.execute("DELETE FROM training_images WHERE id=?", (tid,))
        conn.commit()
    return jsonify({"status": "ok"})


@app.route('/stats', methods=['GET'])
@login_required
def stats_json():
    return jsonify({k: (v or 0) for k, v in get_learner().get_review_stats().items()})


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES — Live model
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/live_predict', methods=['POST'])
def live_predict():
    d = request.json or {}
    if 'data_b64' not in d or 'filename' not in d:
        return jsonify({"error": "Cần filename và data_b64"}), 400

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    raw      = base64.b64decode(d['data_b64'])
    img_hash = hashlib.sha256(raw).hexdigest()[:12]
    suffix   = Path(d['filename']).suffix.lower() or '.jpg'
    img_path = LIVE_DIR / f"live_{img_hash}{suffix}"
    img_path.write_bytes(raw)

    estimator, err = _get_live_estimator()
    if estimator is None:
        return jsonify({"error": f"Không thể tải model: {err}"}), 503

    try:
        results = estimator.analyze_batch([img_path], chunk_size=1)
        if not results:
            return jsonify({"error": "Không phân tích được ảnh"}), 500
        r = results[0]

        overlay_rel = None
        if getattr(r, "overlay_path", None) and Path(r.overlay_path).exists():
            ov_dst = LIVE_DIR / Path(r.overlay_path).name
            shutil.copy2(r.overlay_path, ov_dst)
            overlay_rel = ov_dst.name

        # ── AI Correction ──────────────────────────────────────────────────
        ai_info = {}
        try:
            _features = json.loads(d.get("features") or "{}")
            _al = get_ai_singleton()   # singleton — luôn có data mới nhất
            if _al is not None:
                ai_info = _al.correct(
                    predicted_depth = float(getattr(r, "water_height_cm", 0) or 0),
                    predicted_level = getattr(r, "flood_level", "NO_FLOOD") or "NO_FLOOD",
                    confidence      = float(getattr(r, "confidence", 0) or 0),
                    image_features  = _features,
                )
                if ai_info.get("ai_active"):
                    r.water_height_cm = ai_info["depth_cm"]
                    r.flood_level     = ai_info["level"]
                    r.confidence      = ai_info["confidence_cal"]
        except Exception as _ai_err:
            app.logger.debug(f"AI correction skipped: {_ai_err}")
        # ───────────────────────────────────────────────────────────────────

        return jsonify({
            "status":            "ok",
            "image_hash":        img_hash,
            "flood_depth_cm":    round(float(getattr(r, "water_height_cm", 0) or 0), 1),
            "flood_level":       getattr(r, "flood_level", "Không xác định"),
            "confidence":        round(float(getattr(r, "confidence", 0) or 0), 3),
            "method":            getattr(r, "method", ""),
            "overlay_name":      overlay_rel,
            "detections":        len(getattr(r, "detected_objects", []) or []),
            "vehicles_detected": getattr(r, "vehicles_detected", []) or [],
            # AI learning metadata
            "ai_active":         ai_info.get("ai_active", False),
            "ai_depth_correction": ai_info.get("depth_correction", 0.0),
            "ai_level_correction": ai_info.get("level_correction", 0),
            "ai_confidence_raw":   ai_info.get("confidence_raw", 0.0),
            "ai_n_neighbors":      ai_info.get("n_neighbors", 0),
            "ai_bias_detected":    ai_info.get("bias_detected", False),
            "ai_bias_suggestion":  ai_info.get("bias_suggestion"),
            "ai_explanation":      ai_info.get("explanation", ""),
        })
    except Exception as e:
        app.logger.exception("live_predict error")
        return jsonify({"error": str(e)}), 500


@app.route('/live_image/<path:fname>')
def live_image(fname):
    base = LIVE_DIR.resolve()
    full = (base / fname).resolve()
    try: full.relative_to(base)
    except ValueError: return "Forbidden", 403
    if full.exists():
        try: return send_file(str(full))
        except Exception: pass
    return "Not found", 404


@app.route('/live_feedback', methods=['POST'])
def live_feedback():
    d = request.json or {}
    if not {'filename', 'data_b64', 'actual_level'}.issubset(d):
        return jsonify({"error": "Thiếu trường bắt buộc"}), 400
    try:
        raw      = base64.b64decode(d['data_b64'])
        img_hash = hashlib.sha256(raw).hexdigest()[:12]
        suffix   = Path(d['filename']).suffix.lower() or '.jpg'
        img_path = LIVE_DIR / f"live_{img_hash}{suffix}"
        img_path.write_bytes(raw)

        conn = get_train_conn()
        conn.execute(
            "INSERT OR IGNORE INTO training_images "
            "(added_at,image_path,image_hash,actual_depth,actual_level,notes,source,verified) "
            "VALUES (?,?,?,?,?,?,?,1)",
            (datetime.now().isoformat(timespec='seconds'), str(img_path), img_hash,
             float(d['actual_depth']) if d.get('actual_depth') is not None else 0.0,
             d['actual_level'], d.get('notes', ''),
             'live_correct' if d.get('is_correct') else 'live_corrected'),
        )
        conn.commit()

        if not d.get('is_correct'):
            _invalidate_live_estimator()

        # Retrain AI ngay trong background (không cần restart server)
        get_ai_singleton()
        _trigger_ai_retrain()

        return jsonify({"status": "ok", "image_hash": img_hash})
    except Exception as e:
        app.logger.exception("live_feedback error")
        return jsonify({"error": str(e)}), 500


@app.route('/live_reload_estimator', methods=['POST'])
def live_reload_estimator():
    _invalidate_live_estimator()
    return jsonify({"status": "ok"})


@app.route('/live_history')
def live_history():
    """Danh sách ảnh đã chạy qua Live Model, kèm overlay nếu có."""
    items = []
    if LIVE_DIR.exists():
        for f in sorted(LIVE_DIR.iterdir(), reverse=True):
            if not (f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}):
                continue
            if '_overlay' in f.name or '_depth' in f.name:
                continue
            overlay_name = f.stem + '_overlay.jpg'
            overlay_path = LIVE_DIR / overlay_name
            items.append({
                "name":         f.name,
                "overlay":      overlay_name if overlay_path.exists() else None,
                "size_kb":      round(f.stat().st_size / 1024, 1),
                "mtime":        datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            })
    # Tìm thêm entry trong training_images.db (source=live_correct / live_corrected)
    db_map = {}
    try:
        conn = get_train_conn()
        rows = conn.execute(
            "SELECT image_hash, actual_depth, actual_level, notes, source, added_at "
            "FROM training_images WHERE source LIKE 'live%' ORDER BY id DESC"
        ).fetchall()
        for r in rows:
            db_map[r['image_hash']] = dict(r)
    except Exception:
        pass

    # Gắn thông tin DB vào item nếu hash khớp
    for item in items:
        h = item['name'].replace('live_', '').split('.')[0]
        if h in db_map:
            item['db'] = db_map[h]

    return jsonify({"items": items, "total": len(items)})


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES — Run history
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/run_history', methods=['GET'])
@login_required
def run_history():
    output_root = ROOT_DIR / "output"
    runs = []
    if output_root.exists():
        for run_dir in sorted(output_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            sf = run_dir / "pipeline_summary.json"
            try:    s = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
            except: s = {}
            s["run_dir"] = run_dir.name
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            for key, sub in [("overlay_images", "depth_overlays"), ("original_images", "original_images")]:
                sub_dir = run_dir / sub
                s[key]  = [f.name for f in sorted(sub_dir.glob("*")) if f.suffix.lower() in exts] \
                          if sub_dir.exists() else []
            runs.append(s)
    return jsonify(runs)


@app.route('/run_detail/<run_dir>', methods=['GET'])
@login_required
def run_detail(run_dir):
    """Return full analytics for a run: summary + CSV rows."""
    import csv as _csv
    run_path = ROOT_DIR / "output" / run_dir
    if not run_path.exists():
        return jsonify({"error": "Not found"}), 404
    sf = run_path / "pipeline_summary.json"
    try:
        summary = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
    except Exception:
        summary = {}
    rows = []
    for csv_name in ("flood_analysis_report.csv", "results.csv", "report.csv"):
        csv_path = run_path / csv_name
        if csv_path.exists():
            try:
                with open(csv_path, encoding="utf-8") as f:
                    rows = list(_csv.DictReader(f))
            except Exception:
                rows = []
            break
    return jsonify({"summary": summary, "rows": rows, "run_dir": run_dir})


@app.route('/run_csv/<run_dir>', methods=['GET'])
@login_required
def run_csv_download(run_dir):
    """Serve the CSV file for download."""
    run_path = ROOT_DIR / "output" / run_dir
    for csv_name in ("flood_analysis_report.csv", "results.csv", "report.csv"):
        csv_path = run_path / csv_name
        if csv_path.exists():
            return send_file(str(csv_path), as_attachment=True,
                             download_name=f"flood_{run_dir}.csv",
                             mimetype="text/csv")
    return jsonify({"error": "CSV not found"}), 404


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES — Pipeline
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/pipeline_defaults', methods=['GET'])
@login_required
def pipeline_defaults():
    """Return config.yaml settings for pre-filling the pipeline form."""
    try:
        import yaml as _yaml
        cfg_path = ROOT_DIR / "config.yaml"
        if cfg_path.exists():
            raw = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            return jsonify({
                "yolo_model":   raw.get("yolo_model",   "yolov8n.pt"),
                "depth_model":  raw.get("depth_model",  "depth-anything/Depth-Anything-V2-Small-hf"),
                "output_dir":   raw.get("output_dir",   "output"),
                "drive_folder": raw.get("drive_folder", "FloodAnalysis"),
            })
    except Exception:
        pass
    return jsonify({})


@app.route('/start_pipeline', methods=['POST'])
@login_required
def start_pipeline():
    with _pl_lock:
        if _pl["status"] == "running":
            return jsonify({"error": "Pipeline đang chạy"}), 409
        _pl["status"]    = "running"
        _pl["log"]       = []
        _pl["step"]      = 0
        _pl["stop_flag"] = False
        _pl["run_id"]    = None

    cfg = request.json or {}
    threading.Thread(target=_run_pipeline, args=(cfg,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route('/pipeline_log', methods=['GET'])
@login_required
def pipeline_log():
    offset = int(request.args.get("offset", 0))
    with _pl_lock:
        lines  = _pl["log"][offset:]
        status = _pl["status"]
        step   = _pl["step"]
        nxt    = len(_pl["log"])
    return jsonify({"lines": lines, "next_offset": nxt, "status": status, "step": step})


@app.route('/stop_pipeline', methods=['POST'])
@login_required
def stop_pipeline():
    with _pl_lock:
        _pl["stop_flag"] = True
        _pl["status"]    = "idle"
    return jsonify({"status": "ok"})


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES — Nhận xét & Học (Review Tab)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/review_cases', methods=['GET'])
@login_required
def review_cases():
    """Trả về danh sách ReviewCase cần review từ ActiveLearner queue."""
    try:
        learner = get_learner()
        cases   = learner.get_pending_reviews(max_count=20)
        result  = []
        for c in cases:
            d = c.to_dict() if hasattr(c, 'to_dict') else {}
            # Chỉ giữ các field cần thiết cho UI
            result.append({
                'id':              d.get('id'),
                'timestamp':       d.get('timestamp', ''),
                'image_hash':      d.get('image_hash', ''),
                'image_path':      d.get('image_path', ''),
                'predicted_depth': d.get('predicted_depth'),
                'predicted_level': d.get('predicted_level'),
                'confidence':      d.get('confidence', 0.0),
                'review_reason':   d.get('review_reason', ''),
                'score':           d.get('score', 0),
            })

        # Gợi ý các level thiếu dữ liệu
        underrepresented = []
        try:
            underrepresented = learner.get_underrepresented_levels() if hasattr(learner, 'get_underrepresented_levels') else []
        except Exception:
            pass

        return jsonify({'cases': result, 'underrepresented': underrepresented})
    except Exception as e:
        app.logger.exception('review_cases error')
        return jsonify({'cases': [], 'underrepresented': [], 'error': str(e)})


@app.route('/case_image/<int:case_id>', methods=['GET'])
@login_required
def case_image(case_id):
    """Serve ảnh của ReviewCase theo ID."""
    try:
        case = get_learner()._get_case_by_id(case_id)
        if case:
            p = Path(case.image_path)
            for candidate in (p, TRAIN_IMG_DIR / p.name, LIVE_DIR / p.name):
                if candidate.exists():
                    return send_file(str(candidate.resolve()))
    except Exception:
        pass
    return "Image not found", 404


@app.route('/training_summary', methods=['GET'])
@login_required
def training_summary():
    """Tổng hợp số lượng ảnh training theo từng flood level."""
    try:
        conn     = get_train_conn()
        total    = conn.execute("SELECT COUNT(*) FROM training_images").fetchone()[0]
        by_level = dict(conn.execute(
            "SELECT actual_level, COUNT(*) FROM training_images GROUP BY actual_level"
        ).fetchall())
        return jsonify({'total': total, 'by_level': by_level})
    except Exception as e:
        return jsonify({'total': 0, 'by_level': {}, 'error': str(e)})


@app.route('/annotate_live_image', methods=['POST'])
@login_required
def annotate_live_image():
    """
    Lưu nhãn thực tế cho ảnh từ Live History vào training_images.db
    và kích hoạt retrain AI ngay.

    Body JSON: { image_name, actual_depth, actual_level, notes }
    """
    d = request.json or {}
    image_name   = d.get('image_name', '')
    actual_level = d.get('actual_level', '')
    actual_depth = float(d.get('actual_depth') or 0)
    notes        = d.get('notes', '')

    if not image_name or not actual_level:
        return jsonify({'error': 'Thiếu image_name hoặc actual_level'}), 400

    # Tìm file ảnh trong LIVE_DIR
    img_path = LIVE_DIR / image_name
    if not img_path.exists():
        return jsonify({'error': f'Không tìm thấy ảnh: {image_name}'}), 404

    img_bytes = img_path.read_bytes()
    img_hash  = hashlib.sha256(img_bytes).hexdigest()[:12]

    # Copy sang thư mục training nếu chưa có
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    suffix    = img_path.suffix.lower() or '.jpg'
    train_dst = TRAIN_IMG_DIR / f"{img_hash}{suffix}"
    if not train_dst.exists():
        import shutil as _sh
        _sh.copy2(str(img_path), str(train_dst))

    # Ghi vào DB
    try:
        conn = get_train_conn()
        conn.execute(
            "INSERT OR REPLACE INTO training_images "
            "(added_at, image_path, image_hash, actual_depth, actual_level, notes, source, verified) "
            "VALUES (?,?,?,?,?,?,'history_annotate',1)",
            (datetime.now().isoformat(timespec='seconds'), str(train_dst), img_hash,
             actual_depth, actual_level, notes),
        )
        conn.commit()
    except Exception as e:
        app.logger.exception('annotate_live_image DB error')
        return jsonify({'error': str(e)}), 500

    # Kích hoạt retrain AI trong background
    get_ai_singleton()
    _trigger_ai_retrain()

    log.info(f'[annotate] {image_name} → {actual_level} {actual_depth}cm  hash={img_hash}')
    return jsonify({'status': 'ok', 'image_hash': img_hash, 'train_path': str(train_dst)})


@app.route('/trigger_retrain', methods=['POST'])
@login_required
def trigger_retrain():
    """Kích hoạt retrain AI thủ công ngay lập tức."""
    get_ai_singleton()
    _trigger_ai_retrain()
    return jsonify({'status': 'ok', 'message': 'Retrain đã được kích hoạt trong nền'})


# ══════════════════════════════════════════════════════════════════════════════
# 4.4 UNCERTAINTY SAMPLING + ACTIVE LEARNING LOOP (MỚI)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/uncertainty_ranking', methods=['GET'])
@login_required
def uncertainty_ranking():
    """
    Xếp hạng ảnh cần review theo uncertainty sampling.

    Thuật toán:
      - Entropy sampling: ảnh có predicted_confidence thấp → uncertainty cao → ưu tiên
      - Margin sampling: ảnh có confidence gần threshold → khó quyết định → ưu tiên
      - Diversity: tránh chọn quá nhiều ảnh cùng flood level
      - Cluster underrepresentation: ưu tiên flood level ít data training

    Returns:
      {
        "ranked_cases": [...sorted by priority...],
        "priority_breakdown": {entropy: N, margin: N, diversity: N},
        "recommendation": "Focus on HIGH/MEDIUM/LOW flood images"
      }
    """
    try:
        learner = get_learner()
        # Lấy nhiều hơn để có pool đủ lớn để rank
        cases = learner.get_pending_reviews(max_count=100)

        if not cases:
            return jsonify({
                'ranked_cases': [],
                'priority_breakdown': {},
                'recommendation': 'Không có ảnh cần review',
            })

        # Lấy training distribution để tính underrepresentation
        conn = get_train_conn()
        level_counts = dict(conn.execute(
            "SELECT actual_level, COUNT(*) FROM training_images GROUP BY actual_level"
        ).fetchall())

        all_levels = ['LOW', 'MEDIUM', 'HIGH', 'SEVERE']
        max_count  = max(level_counts.values()) if level_counts else 1

        scored_cases = []
        for c in cases:
            d = c.to_dict() if hasattr(c, 'to_dict') else {}
            conf     = float(d.get('confidence', 0.5) or 0.5)
            level    = str(d.get('predicted_level', 'MEDIUM') or 'MEDIUM')

            # ── Score 1: Entropy uncertainty (thấp confidence → cao entropy) ──
            # entropy = -p*log(p) - (1-p)*log(1-p), normalized to [0,1]
            p = max(1e-6, min(1 - 1e-6, conf))
            entropy = -(p * __import__('math').log2(p) + (1 - p) * __import__('math').log2(1 - p))
            entropy_score = float(entropy)   # max = 1.0 at p=0.5

            # ── Score 2: Margin sampling (gần threshold = khó quyết định) ──
            low_thresh  = 0.40
            high_thresh = 0.70
            # Distance from boundaries: 0 = ON boundary (hardest), 1 = far from boundary
            dist_low  = abs(conf - low_thresh)
            dist_high = abs(conf - high_thresh)
            margin_score = 1.0 - min(dist_low, dist_high) / max(high_thresh, 0.1)
            margin_score = float(max(0, margin_score))

            # ── Score 3: Underrepresentation bonus ──
            level_count  = level_counts.get(level, 0)
            # Levels với ít data → priority cao hơn
            repr_ratio   = level_count / max(max_count, 1)
            underrep_score = float(max(0, 1.0 - repr_ratio))

            # ── Combined priority score ──
            priority = (
                entropy_score   * 0.40 +
                margin_score    * 0.35 +
                underrep_score  * 0.25
            )

            scored_cases.append({
                'id':              d.get('id'),
                'timestamp':       d.get('timestamp', ''),
                'image_hash':      d.get('image_hash', ''),
                'image_path':      d.get('image_path', ''),
                'predicted_depth': d.get('predicted_depth'),
                'predicted_level': level,
                'confidence':      round(conf, 3),
                'review_reason':   d.get('review_reason', ''),
                'score':           d.get('score', 0),
                # Uncertainty metrics
                'priority_score':  round(priority, 4),
                'entropy_score':   round(entropy_score, 3),
                'margin_score':    round(margin_score, 3),
                'underrep_score':  round(underrep_score, 3),
                'uncertainty_tag': _get_uncertainty_tag(entropy_score, margin_score),
            })

        # Sort by priority descending
        scored_cases.sort(key=lambda x: x['priority_score'], reverse=True)

        # Priority breakdown
        high_prio  = sum(1 for c in scored_cases if c['priority_score'] > 0.65)
        med_prio   = sum(1 for c in scored_cases if 0.35 < c['priority_score'] <= 0.65)
        low_prio   = sum(1 for c in scored_cases if c['priority_score'] <= 0.35)

        # Recommendation
        underrep_levels = [
            lvl for lvl in all_levels
            if level_counts.get(lvl, 0) < max(level_counts.values(), default=1) * 0.3
        ]
        recommendation = (
            f"Tập trung review các ảnh mức {', '.join(underrep_levels)} "
            f"(thiếu dữ liệu training)"
            if underrep_levels
            else "Phân phối training data đồng đều — focus on high-entropy cases"
        )

        return jsonify({
            'ranked_cases': scored_cases[:50],   # top 50 sau khi rank
            'total_pending': len(scored_cases),
            'priority_breakdown': {
                'high_priority': high_prio,
                'medium_priority': med_prio,
                'low_priority': low_prio,
            },
            'level_distribution': level_counts,
            'underrepresented_levels': underrep_levels,
            'recommendation': recommendation,
        })

    except Exception as e:
        app.logger.exception('uncertainty_ranking error')
        return jsonify({'ranked_cases': [], 'error': str(e)}), 500


def _get_uncertainty_tag(entropy: float, margin: float) -> str:
    """Nhãn mô tả loại uncertainty cho UI."""
    if entropy > 0.85:
        return "🔴 Rất không chắc"
    if entropy > 0.65:
        return "🟠 Không chắc"
    if margin > 0.70:
        return "🟡 Gần ngưỡng"
    return "🟢 Tương đối chắc"


@app.route('/active_learning_status', methods=['GET'])
@login_required
def active_learning_status():
    """
    Trạng thái active learning loop.
    Trả về metrics để hiển thị trên UI.
    """
    try:
        learner = get_learner()
        stats   = learner.get_review_stats()
        conn    = get_train_conn()

        total_training = conn.execute(
            "SELECT COUNT(*) FROM training_images"
        ).fetchone()[0]

        by_level = dict(conn.execute(
            "SELECT actual_level, COUNT(*) FROM training_images GROUP BY actual_level"
        ).fetchall())

        # Lấy retrain history từ AI model cache
        retrain_count = 0
        cache_path = BASE_DIR / 'ai_model_cache.json'
        if cache_path.exists():
            import json as _j
            try:
                cache = _j.loads(cache_path.read_text())
                retrain_count = cache.get('retrain_count', 0)
            except Exception:
                pass

        # Ước tính data quality: std của confidence scores trong training
        conf_vals = conn.execute(
            "SELECT actual_depth FROM training_images ORDER BY added_at DESC LIMIT 100"
        ).fetchall()
        depth_values = [r[0] for r in conf_vals if r[0] is not None]
        data_quality = 1.0 - min(1.0, __import__('statistics').stdev(depth_values) / 100
                                 if len(depth_values) > 1 else 0.5)

        return jsonify({
            'status': 'active',
            'total_training_images': total_training,
            'training_by_level':     by_level,
            'pending_reviews':       stats.get('pending', 0),
            'reviewed_total':        stats.get('reviewed', 0),
            'retrain_count':         retrain_count,
            'data_quality_score':    round(data_quality, 3),
            'auto_retrain_threshold': _al_state.get('retrain_threshold', 10),
            'reviews_since_last_retrain': _al_state.get('reviews_since_retrain', 0),
            'next_retrain_in': max(
                0,
                _al_state.get('retrain_threshold', 10) - _al_state.get('reviews_since_retrain', 0)
            ),
        })
    except Exception as e:
        app.logger.exception('active_learning_status error')
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ── Active Learning Loop State ─────────────────────────────────────────────────

_al_state = {
    'reviews_since_retrain': 0,
    'retrain_threshold':     10,   # retrain sau mỗi 10 reviews
    'auto_retrain_enabled':  True,
    'last_retrain_ts':       None,
    'retrain_lock':          threading.Lock(),
}


@app.route('/configure_active_learning', methods=['POST'])
@login_required
def configure_active_learning():
    """Cấu hình active learning loop."""
    try:
        data = request.get_json() or {}
        if 'retrain_threshold' in data:
            _al_state['retrain_threshold'] = max(1, int(data['retrain_threshold']))
        if 'auto_retrain_enabled' in data:
            _al_state['auto_retrain_enabled'] = bool(data['auto_retrain_enabled'])
        return jsonify({
            'status': 'ok',
            'config': {
                'retrain_threshold':  _al_state['retrain_threshold'],
                'auto_retrain_enabled': _al_state['auto_retrain_enabled'],
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400


def _active_learning_on_review_complete(case_id: int, verified_depth: float, verified_level: str):
    """
    Hook gọi sau mỗi lần user complete 1 review.
    Tự động kích hoạt retrain nếu đủ số lượng.

    Args:
        case_id:        ID của case vừa review
        verified_depth: độ sâu đã xác nhận (cm)
        verified_level: flood level đã xác nhận
    """
    if not _al_state.get('auto_retrain_enabled', True):
        return

    with _al_state['retrain_lock']:
        _al_state['reviews_since_retrain'] = _al_state.get('reviews_since_retrain', 0) + 1
        current = _al_state['reviews_since_retrain']
        threshold = _al_state.get('retrain_threshold', 10)

        log.info(
            f"  [ActiveLearning] Review #{current}/{threshold} complete "
            f"→ case={case_id}, level={verified_level}, depth={verified_depth:.1f}cm"
        )

        if current >= threshold:
            # Reset counter
            _al_state['reviews_since_retrain'] = 0
            _al_state['last_retrain_ts'] = datetime.now().isoformat()

            # Trigger retrain trong background thread
            log.info(f"  [ActiveLearning] Auto-retrain triggered after {threshold} reviews!")
            retrain_thread = threading.Thread(
                target=_active_learning_retrain_job,
                args=(threshold,),
                daemon=True,
            )
            retrain_thread.start()


def _active_learning_retrain_job(n_reviews: int):
    """Background job: thực hiện retrain và log kết quả."""
    try:
        log.info(f"  [ActiveLearning] Starting retrain job (triggered by {n_reviews} reviews)")

        # Tăng retrain counter trong cache
        cache_path = BASE_DIR / 'ai_model_cache.json'
        if cache_path.exists():
            import json as _j
            try:
                cache = _j.loads(cache_path.read_text())
                cache['retrain_count'] = cache.get('retrain_count', 0) + 1
                cache['last_retrain']  = datetime.now().isoformat()
                cache['trigger']       = f'auto_after_{n_reviews}_reviews'
                cache_path.write_text(_j.dumps(cache, indent=2, ensure_ascii=False))
            except Exception as e:
                log.warning(f"  [ActiveLearning] Cache update failed: {e}")

        # Kích hoạt AI retrain pipeline
        get_ai_singleton()
        _trigger_ai_retrain()

        log.info("  [ActiveLearning] Retrain job completed")

        # Cập nhật model cache với thông tin retrain
        _plog(
            f"🤖 Active Learning: Auto-retrain hoàn thành sau {n_reviews} reviews mới",
            level="INFO"
        )

    except Exception as e:
        log.error(f"  [ActiveLearning] Retrain job failed: {e}")
        _plog(f"❌ Active Learning retrain thất bại: {e}", level="ERROR")


@app.route('/al_review_complete', methods=['POST'])
@login_required
def al_review_complete():
    """
    API endpoint báo hiệu 1 review đã hoàn thành.
    Kích hoạt active learning loop.

    Body: {case_id, verified_depth, verified_level}
    """
    try:
        data = request.get_json() or {}
        case_id        = int(data.get('case_id', 0))
        verified_depth = float(data.get('verified_depth', 0))
        verified_level = str(data.get('verified_level', 'MEDIUM'))

        # Chạy active learning hook trong background
        hook_thread = threading.Thread(
            target=_active_learning_on_review_complete,
            args=(case_id, verified_depth, verified_level),
            daemon=True,
        )
        hook_thread.start()

        reviews_left = max(
            0,
            _al_state.get('retrain_threshold', 10) -
            _al_state.get('reviews_since_retrain', 0) - 1
        )

        return jsonify({
            'status': 'ok',
            'reviews_since_last_retrain': _al_state.get('reviews_since_retrain', 0),
            'reviews_until_retrain': reviews_left,
            'auto_retrain_enabled': _al_state.get('auto_retrain_enabled', True),
        })

    except Exception as e:
        app.logger.exception('al_review_complete error')
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/uncertainty_heatmap', methods=['GET'])
@login_required
def uncertainty_heatmap():
    """
    Dữ liệu heatmap uncertainty cho visualization.
    Trả về list {confidence, predicted_level, timestamp} để vẽ chart.
    """
    try:
        learner = get_learner()
        cases   = learner.get_pending_reviews(max_count=200)

        bins = {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
        level_uncertainty = {}

        for c in cases:
            d    = c.to_dict() if hasattr(c, 'to_dict') else {}
            conf = float(d.get('confidence', 0.5) or 0.5)
            lvl  = str(d.get('predicted_level', 'MEDIUM') or 'MEDIUM')

            # Bin
            if conf < 0.2:   bins['0.0-0.2'] += 1
            elif conf < 0.4: bins['0.2-0.4'] += 1
            elif conf < 0.6: bins['0.4-0.6'] += 1
            elif conf < 0.8: bins['0.6-0.8'] += 1
            else:            bins['0.8-1.0'] += 1

            # Per-level average uncertainty
            if lvl not in level_uncertainty:
                level_uncertainty[lvl] = []
            level_uncertainty[lvl].append(1.0 - conf)

        level_avg_uncertainty = {
            lvl: round(sum(vals) / len(vals), 3)
            for lvl, vals in level_uncertainty.items()
            if vals
        }

        return jsonify({
            'confidence_distribution': bins,
            'level_uncertainty':       level_avg_uncertainty,
            'total_cases':             len(cases),
            'high_uncertainty_count':  bins.get('0.4-0.6', 0) + bins.get('0.2-0.4', 0) + bins.get('0.0-0.2', 0),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES — Agent Chat
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/agent_chat', methods=['POST'])
@login_required
def agent_chat():
    """
    Endpoint chat với FloodAgent.
    Body JSON: { "message": "...", "reset": false }
    Returns: { "message": "...", "action": "...", "success": bool, "memory": {...} }
    """
    d = request.json or {}
    user_msg = str(d.get('message', '')).strip()
    do_reset = bool(d.get('reset', False))

    if not user_msg and not do_reset:
        return jsonify({'error': 'Thiếu message'}), 400

    agent = get_flood_agent()
    if agent is None:
        return jsonify({
            'message': '⚠️ Agent chưa sẵn sàng. Vui lòng thử lại sau.',
            'success': False,
        }), 503

    try:
        if do_reset:
            agent.reset_session()
            return jsonify({'message': '✅ Phiên chat đã được reset.', 'success': True})

        resp = agent.chat(user_msg)
        data = resp.to_dict()
        data['results'] = data['results'][:3]   # giới hạn payload
        return jsonify(data)
    except Exception as e:
        app.logger.exception('agent_chat error')
        return jsonify({'message': f'❌ Lỗi: {e}', 'success': False}), 500


@app.route('/agent_reset', methods=['POST'])
@login_required
def agent_reset():
    """Reset phiên chat agent."""
    agent = get_flood_agent()
    if agent:
        agent.reset_session()
    return jsonify({'status': 'ok'})


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES — Easter Egg Music Player
# ══════════════════════════════════════════════════════════════════════════════

MUSIC_DIR  = BASE_DIR / "music"
MUSIC_EXTS = {'.mp3', '.ogg', '.wav', '.flac', '.m4a', '.aac'}


@app.route('/music_list', methods=['GET'])
@login_required
def music_list():
    MUSIC_DIR.mkdir(exist_ok=True)
    files = sorted(
        f.name for f in MUSIC_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in MUSIC_EXTS
    )
    return jsonify({"files": files})


@app.route('/music/<path:fname>', methods=['GET'])
@login_required
def music_serve(fname):
    base = MUSIC_DIR.resolve()
    full = (base / fname).resolve()
    try:
        full.relative_to(base)
    except ValueError:
        return "Forbidden", 403
    if full.exists() and full.suffix.lower() in MUSIC_EXTS:
        return send_file(str(full))
    return "Not found", 404


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Fix encoding on Windows consoles (cp932 can't print emojis)
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[union-attr]
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[union-attr]
        except Exception:
            pass

    # Quick stats for banner
    _tmp  = ActiveLearner()
    stats = {k: (v or 0) for k, v in _tmp.get_review_stats().items()}
    _tmp.close()

    try:
        _tc = sqlite3.connect(str(TRAIN_DB_PATH))
        _ti = _tc.execute("SELECT COUNT(*) FROM training_images").fetchone()[0]
        _tc.close()
    except Exception:
        _ti = 0

    print("\n" + "=" * 58)
    print("  FloodAI - Unified Web Console")
    print("=" * 58)
    print(f"\n  http://localhost:5000")
    print(f"  Pending reviews  : {stats.get('pending', 0)}")
    print(f"  Reviewed         : {stats.get('reviewed', 0)}")
    print(f"  Training images  : {_ti}")
    print(f"  Output directory : {ROOT_DIR / 'output'}")
    print("\n  Tabs: Review / Upload / Training / Live / History / Pipeline")
    print("\n  Ctrl+C to stop\n")
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(debug=False, host='0.0.0.0', port=5000)
