# -*- coding: utf-8 -*-
"""
learning/admin_routes.py
========================
Registers all admin/review/pipeline routes onto the main Flask app (app.py).
Replaces running learning/review_ui.py as a standalone server.

Usage in app.py:
    from learning.admin_routes import register_admin
    register_admin(app)

Routes added (no URL prefix — same as the original review_ui.py):
  GET  /admin                  — Full admin panel (old dashboard.html)
  GET  /admin/live             — Live model public page
  POST /review/<case_id>       — Submit review
  POST /skip/<case_id>         — Skip case
  GET  /image/<case_id>        — Case image
  GET  /run_image/<path>       — Run output image
  POST /upload_training        — Upload training image
  GET  /training_stats         — Training stats JSON
  GET  /training_summary       — Training summary
  DEL  /delete_training/<tid>  — Delete training entry
  GET  /review_stats           — Review queue stats
  POST /live_predict           — Live flood analysis
  GET  /live_image/<fname>     — Live result image
  POST /live_feedback          — User feedback on live result
  POST /live_reload_estimator  — Invalidate live estimator cache
  GET  /live_history           — Live prediction history
  GET  /run_history            — Pipeline run history
  GET  /run_detail/<run_dir>   — Run detail + CSV rows
  GET  /run_csv/<run_dir>      — Download run CSV
  GET  /pipeline_defaults      — Config defaults for pipeline form
  POST /start_pipeline         — Start pipeline thread
  GET  /pipeline_log           — Streaming pipeline log
  POST /stop_pipeline          — Stop pipeline
  GET  /review_cases           — Pending review cases
  GET  /case_image/<case_id>   — Review case image
  POST /annotate_live_image    — Annotate live history image
  POST /trigger_retrain        — Manual AI retrain
  GET  /uncertainty_ranking    — Uncertainty-ranked review cases
  GET  /active_learning_status — Active learning metrics
  POST /configure_active_learning — Configure AL loop
  POST /al_review_complete     — Hook: review completed
  GET  /uncertainty_heatmap    — Confidence histogram
"""

import base64
import hashlib
import json
import logging
import math
import shutil
import sqlite3
import sys
import threading
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path

log = logging.getLogger("admin_routes")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent          # learning/
ROOT_DIR  = BASE_DIR.parent               # project root
TMPL_DIR  = BASE_DIR / "templates"

sys.path.insert(0, str(ROOT_DIR))

TRAIN_DB_PATH  = BASE_DIR / "training_images.db"
TRAIN_IMG_DIR  = BASE_DIR / "training_images"
LIVE_DIR       = BASE_DIR / "_live_results"
DRIVE_PULL_DIR = BASE_DIR / "_drive_pull"

# ── Pipeline state (module-level, shared across requests) ──────────────────────
_pl = {
    "status":    "idle",
    "run_id":    None,
    "log":       [],
    "step":      0,
    "stop_flag": False,
}
_pl_lock = threading.Lock()

STEPS = ["Tải ảnh", "Sao chép", "Địa điểm", "Depth AI", "Báo cáo"]


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


# ── Live estimator singleton ───────────────────────────────────────────────────
_live_estimator = None


def _invalidate_live_estimator():
    global _live_estimator
    _live_estimator = None


# ── AI / Agent singletons ──────────────────────────────────────────────────────
_ai_singleton      = None
_ai_singleton_lock = threading.Lock()


def get_ai_singleton():
    global _ai_singleton
    with _ai_singleton_lock:
        if _ai_singleton is None:
            try:
                from learning.ai_learner import AiLearner
                _ai_singleton = AiLearner()
                log.info("[AI-singleton] Khởi tạo xong")
            except Exception as e:
                log.warning(f"[AI-singleton] {e}")
    return _ai_singleton


def _trigger_ai_retrain():
    def _worker():
        ai = get_ai_singleton()
        if ai is None:
            return
        try:
            ai.invalidate()
            result = ai.train()
            log.info(f"[AI-retrain] {result.get('n_cases', 0)} cases, kNN={result.get('n_knn', 0)}")
        except Exception as e:
            log.warning(f"[AI-retrain] {e}")
    threading.Thread(target=_worker, daemon=True, name="ai-retrain").start()


# ── Module-level string constants (avoid duplication) ─────────────────────────
_MSG_STOPPED   = "\u23f9 Đã dừng."
_MSG_NOT_FOUND = "Not found"
_SQL_COUNT_TI  = "SELECT COUNT(*) FROM training_images"
_SQL_LEVEL_TI  = "SELECT actual_level, COUNT(*) FROM training_images GROUP BY actual_level"
_SQL_IMG_BY_ID = "SELECT image_path FROM training_images WHERE id=?"

# ── Active learning state ──────────────────────────────────────────────────────
_al_state = {
    "reviews_since_retrain": 0,
    "retrain_threshold":     10,
    "auto_retrain_enabled":  True,
    "last_retrain_ts":       None,
    "retrain_lock":          threading.Lock(),
}


# ── Pipeline step helpers ──────────────────────────────────────────────────────

def _pl_load_images(cfg: dict, original_dir: Path):
    """Step 0: Load images from local folder or Google Drive. Returns (images, mode)."""
    from utils.constants import IMAGE_EXTENSIONS
    mode = cfg.get("mode", "local")
    _plog(f"═══ STEP 1/5 · Tải ảnh [{mode.upper()}] ═══")
    if mode == "local":
        img_dir = Path(cfg.get("input_path", ""))
        if not img_dir.exists():
            _plog(f"❌ Folder không tồn tại: {img_dir}", "ERROR")
            return None, mode
        images = sorted(
            f for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        _plog(f"✅ Tải {len(images)} ảnh từ {img_dir}", "SUCCESS")
        return images, mode
    if mode == "drive":
        from uploader.drive_uploader import DriveUploader
        _plog("☁️ Đang kết nối Google Drive...")
        images = DriveUploader().download_folder(
            drive_folder=cfg.get("input_path", ""),
            local_dir=original_dir,
            extensions=IMAGE_EXTENSIONS,
        )
        _plog(f"✅ Tải {len(images)} ảnh từ Drive", "SUCCESS")
        return images, mode
    return [], mode


def _pl_copy_file(src: Path, dst: Path) -> Path:
    """Copy one file with up to 5 retries on PermissionError."""
    import time as _t
    for attempt in range(5):
        try:
            shutil.copy2(str(src), str(dst))
            return dst
        except PermissionError as e:
            if attempt < 4:
                _t.sleep(0.5)
            else:
                _plog(f"⚠️ Bỏ qua {src.name}: file bị khóa ({e})", "WARNING")
                return src
    return dst


def _pl_copy_to_output(images: list, original_dir: Path) -> list:
    """Step 1: Copy images into original_dir, return list of destination paths."""
    _plog(f"═══ STEP 2/5 · Sao chép {len(images)} ảnh sang output ═══")
    original_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in images:
        dst = original_dir / Path(src).name
        copied.append(dst if Path(src).resolve() == dst.resolve() else _pl_copy_file(Path(src), dst))
    _plog(f"✅ Sao chép xong: {len(copied)} ảnh", "SUCCESS")
    return copied


def _pl_detect_location(images: list, cfg: dict, results: dict):
    """Step 2: Detect location for each image and store into results."""
    _pstep(2)
    if not cfg.get("detect_location", True):
        _plog("⏭ Bỏ qua nhận diện địa điểm.")
        return
    _plog("═══ STEP 3/5 · Nhận diện địa điểm ═══")
    from utils.location_detector import LocationDetector
    loc_list = LocationDetector(
        google_maps_key=cfg.get("google_maps_key", ""),
        use_ocr=True, use_plate=True, use_exif=True,
    ).detect_batch([Path(p) for p in images])
    located = sum(1 for r in loc_list if r.method != "none")
    results["location_map"] = {r.image_path: vars(r) for r in loc_list}
    _plog(f"✅ Địa điểm: {located}/{len(images)} ảnh có vị trí", "SUCCESS")


def _pl_move_depth_outputs(depth_results: list, overlay_dir: Path, depthmap_dir: Path):
    for r in depth_results:
        for attr, dest in [("overlay_path", overlay_dir), ("depth_map_path", depthmap_dir)]:
            src = Path(getattr(r, attr, "") or "")
            if src.exists():
                dst = dest / src.name
                shutil.move(str(src), str(dst))
                setattr(r, attr, str(dst))


def _pl_self_learning(depth_results: list, cfg: dict, images: list):
    try:
        from learning_update import SelfLearningPipeline
        sl = SelfLearningPipeline()
        sl.process_results(depth_results, cfg, images)
        sl.close()
        _plog("🧠 Self-learning: dữ liệu đã được xếp hàng review.", "SUCCESS")
    except Exception as e:
        _plog(f"⚠️ Self-learning skipped: {e}", "WARNING")


def _pl_depth_analysis(images: list, cfg: dict,
                       out_root: Path, overlay_dir: Path, depthmap_dir: Path) -> list:
    """Step 3: Run depth analysis, move outputs, trigger self-learning."""
    from utils.constants import (
        OUTPUT_FOLDER_TMP_DEPTH, DEFAULT_DINO_MODEL, DEFAULT_POSE_MODEL,
        DEFAULT_YOLO_MODEL, DEFAULT_DEPTH_MODEL,
    )
    _pstep(3)
    if cfg.get("skip_depth", False):
        _plog("⏭ Bỏ qua Depth Analysis.")
        return []
    _plog("═══ STEP 4/5 · Depth Anything V2 + YOLO ═══")
    from depth_analysis.reference_estimator import ReferenceEstimator
    tmp = out_root / OUTPUT_FOLDER_TMP_DEPTH
    tmp.mkdir(exist_ok=True)
    estimator = ReferenceEstimator(
        yolo_model   =cfg.get("yolo_model",  DEFAULT_YOLO_MODEL),
        depth_model  =cfg.get("depth_model", DEFAULT_DEPTH_MODEL),
        output_dir   =tmp,
        conf_thresh  =float(cfg.get("yolo_conf", 0.35)),
        use_dino     =cfg.get("use_dino",      True),
        dino_model   =cfg.get("dino_model",    DEFAULT_DINO_MODEL),
        use_pose     =cfg.get("use_pose",      True),
        pose_model   =cfg.get("pose_model",    DEFAULT_POSE_MODEL),
        use_segformer=cfg.get("use_segformer", True),
    )
    chunk = cfg.get("depth_chunk_size", 8)
    _plog(f"🔍 Phân tích {len(images)} ảnh, chunk_size={chunk}…")
    depth_results = estimator.analyze_batch(images, chunk_size=chunk, stop_check=_stopped)
    estimator.unload_heavy_models()
    _pl_move_depth_outputs(depth_results, overlay_dir, depthmap_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    _plog(f"✅ Depth xong: {len(depth_results)}/{len(images)} ảnh", "SUCCESS")
    _pl_self_learning(depth_results, cfg, images)
    return depth_results


def _pl_write_reports(results: dict, out_root: Path):
    try:
        from utils.report_generator import ReportGeneratorV2
        csv_p, html_p = ReportGeneratorV2(output_dir=out_root).generate(results)
        _plog(f"✅ HTML → {html_p}", "SUCCESS")
        _plog(f"✅ CSV  → {csv_p}",  "SUCCESS")
    except Exception as e:
        _plog(f"⚠️ HTML/CSV report failed: {e}", "WARNING")
    try:
        from utils.excel_reporter import ExcelReporter
        xlsx_p = ExcelReporter(output_dir=out_root).generate(results)
        if xlsx_p:
            _plog(f"✅ XLSX → {xlsx_p}", "SUCCESS")
    except Exception as e:
        _plog(f"⚠️ Excel report skipped: {e}", "WARNING")


def _pl_upload_drive(results: dict, out_root: Path, run_id: str, cfg: dict):
    from utils.constants import DRIVE_UPLOAD_EXTENSIONS
    _plog("☁️ Đang upload lên Google Drive…")
    try:
        from uploader.drive_uploader import DriveUploader
        fid = DriveUploader().upload_folder(
            local_dir=out_root,
            folder_name=f"{cfg.get('drive_folder', 'FloodAnalysis')}/{run_id}",
            extensions=DRIVE_UPLOAD_EXTENSIONS,
        )
        results["drive_folder_id"] = fid
        _plog(f"✅ Drive → https://drive.google.com/drive/folders/{fid}", "SUCCESS")
    except Exception as e:
        _plog(f"❌ Drive upload thất bại: {e}", "ERROR")


def _pl_generate_reports(results: dict, out_root: Path, run_id: str, cfg: dict):
    """Step 4: Write reports, optionally upload to Drive, save summary JSON."""
    from utils.constants import PIPELINE_SUMMARY_JSON
    _pstep(4)
    _plog("═══ STEP 5/5 · Tạo báo cáo ═══")
    _pl_write_reports(results, out_root)
    if not cfg.get("skip_drive", True) and not _stopped():
        _pl_upload_drive(results, out_root, run_id, cfg)
    dd = results.get("depth_data", [])
    counts: dict = {}
    for r in dd:
        lvl = r.flood_level if hasattr(r, "flood_level") else r.get("flood_level", "?")
        counts[lvl] = counts.get(lvl, 0) + 1
    summary = {
        "run_id": run_id, "query": "N/A", "sources": [],
        "total_crawled":  len(results.get("raw",      [])),
        "total_filtered": len(results.get("filtered", [])),
        "total_analyzed": len(dd),
        "flood_summary":  counts,
    }
    try:
        (out_root / PIPELINE_SUMMARY_JSON).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception:
        pass


def _pl_cleanup_input(cfg: dict, mode: str):
    """Auto-delete local input folder after successful run, if requested."""
    if not cfg.get("delete_input_folder") or mode != "local":
        return
    try:
        folder = Path(cfg.get("input_path", ""))
        if folder.exists() and folder.is_dir():
            shutil.rmtree(folder)
            _plog(f"🗑 Đã xóa folder input: {folder}", "SUCCESS")
    except Exception as e:
        _plog(f"⚠️ Không xóa được folder input: {e}", "WARNING")


# ── Pipeline thread ────────────────────────────────────────────────────────────

def _run_pipeline(cfg: dict):
    try:
        from utils.constants import (
            OUTPUT_FOLDER_ORIGINAL, OUTPUT_FOLDER_OVERLAY, OUTPUT_FOLDER_DEPTHMAP,
        )
        run_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root     = Path(cfg.get("output_dir", "output")) / run_id
        original_dir = out_root / OUTPUT_FOLDER_ORIGINAL
        overlay_dir  = out_root / OUTPUT_FOLDER_OVERLAY
        depthmap_dir = out_root / OUTPUT_FOLDER_DEPTHMAP
        for d in (original_dir, overlay_dir, depthmap_dir):
            d.mkdir(parents=True, exist_ok=True)
        with _pl_lock:
            _pl["run_id"] = run_id

        results = {"run_id": run_id, "query": "N/A", "sources": [],
                   "raw": [], "filtered": [], "depth_data": []}

        _pstep(0)
        images, mode = _pl_load_images(cfg, original_dir)
        if images is None:
            with _pl_lock: _pl["status"] = "error"
            return
        results["raw"] = results["filtered"] = images
        if not images:
            _plog("❌ Không có ảnh nào để xử lý.", "ERROR")
            with _pl_lock: _pl["status"] = "error"
            return
        if _stopped(): _plog(_MSG_STOPPED, "WARNING"); return

        _pstep(1)
        images = _pl_copy_to_output(images, original_dir)
        results["filtered"] = images
        if _stopped(): _plog(_MSG_STOPPED, "WARNING"); return

        _pl_detect_location(images, cfg, results)
        if _stopped(): _plog(_MSG_STOPPED, "WARNING"); return

        depth_results = _pl_depth_analysis(images, cfg, out_root, overlay_dir, depthmap_dir)
        results["depth_data"] = depth_results
        if _stopped(): _plog(_MSG_STOPPED, "WARNING"); return

        _pl_generate_reports(results, out_root, run_id, cfg)
        _pl_cleanup_input(cfg, mode)

        _plog("═══ ✅ PIPELINE HOÀN THÀNH ═══", "SUCCESS")
        with _pl_lock:
            _pl["status"] = "done"
            _pl["step"]   = len(STEPS)

    except Exception as exc:
        _plog(f"❌ Lỗi pipeline: {exc}", "ERROR")
        _plog(traceback.format_exc(), "ERROR")
        with _pl_lock:
            _pl["status"] = "error"


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE SUB-REGISTRARS
# ══════════════════════════════════════════════════════════════════════════════

def _reg_core(app, lr, gl, gtc, send_file):
    from flask import jsonify, redirect, render_template, request

    @app.route("/admin")
    @lr
    def admin_panel():
        return render_template("admin.html", active="admin", logged_in=True)

    @app.route("/review/<int:case_id>", methods=["POST"])
    @lr
    def submit_review(case_id):
        d = request.json or {}
        gl().submit_review(case_id=case_id, actual_depth=d["actual_depth"],
                           actual_level=d["actual_level"], reviewed_by="web_ui",
                           notes=d.get("notes", ""))
        get_ai_singleton()
        _trigger_ai_retrain()
        return jsonify({"status": "ok"})

    @app.route("/skip/<int:case_id>", methods=["POST"])
    @lr
    def skip_case(case_id):
        lrn = gl()
        lrn.conn.execute("UPDATE review_queue SET status='skipped' WHERE id=?", (case_id,))
        lrn.conn.commit()
        return jsonify({"status": "ok"})

    @app.route("/image/<int:case_id>")
    @lr
    def get_image(case_id):
        case = gl()._get_case_by_id(case_id)
        if case:
            p = Path(case.image_path)
            for candidate in (p, TRAIN_IMG_DIR / p.name):
                if candidate.exists():
                    try: return send_file(str(candidate.resolve()))
                    except Exception: pass
        return "Image not found", 404

    @app.route("/run_image/<path:rel_path>")
    @lr
    def run_image(rel_path):
        output_root = ROOT_DIR / "output"
        full = (output_root / rel_path).resolve()
        try: full.relative_to(output_root.resolve())
        except ValueError: return "Forbidden", 403
        if full.exists():
            try: return send_file(str(full))
            except Exception: pass
        return _MSG_NOT_FOUND, 404


def _reg_training(app, lr, gl, gtc, send_file):
    from flask import jsonify, request

    @app.route("/upload_training", methods=["POST"])
    @lr
    def upload_training():
        d = request.json or {}
        if not {"filename", "data_b64", "actual_depth", "actual_level"}.issubset(d):
            return jsonify({"error": "missing fields"}), 400
        TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
        raw       = base64.b64decode(d["data_b64"])
        img_hash  = hashlib.sha256(raw).hexdigest()[:12]
        suffix    = Path(d["filename"]).suffix.lower() or ".jpg"
        save_path = TRAIN_IMG_DIR / f"{img_hash}{suffix}"
        save_path.write_bytes(raw)
        try:
            conn = gtc()
            conn.execute(
                "INSERT INTO training_images "
                "(added_at,image_path,image_hash,actual_depth,actual_level,notes,source) "
                "VALUES (?,?,?,?,?,?,'manual')",
                (datetime.now().isoformat(), str(save_path), img_hash,
                 float(d["actual_depth"]), d["actual_level"], d.get("notes", "")),
            )
            conn.commit()
        except Exception as e:
            return jsonify({"status": "duplicate", "msg": str(e)}), 409
        get_ai_singleton()
        _trigger_ai_retrain()
        return jsonify({"status": "ok", "path": str(save_path)})

    @app.route("/pull_from_drive", methods=["POST"])
    @lr
    def pull_from_drive():
        import tempfile, shutil
        d = request.json or {}
        drive_link = (d.get("drive_link") or "").strip()
        if not drive_link:
            return jsonify({"error": "missing drive_link"}), 400

        try:
            from uploader.drive_uploader import DriveUploader
            uploader = DriveUploader()
        except Exception as e:
            return jsonify({"error": f"Drive auth failed: {e}"}), 500

        # Clear previous pull results
        if DRIVE_PULL_DIR.exists():
            shutil.rmtree(DRIVE_PULL_DIR)
        DRIVE_PULL_DIR.mkdir(parents=True)

        with tempfile.TemporaryDirectory() as tmp:
            try:
                files = uploader.download_folder(drive_link, Path(tmp))
            except Exception as e:
                return jsonify({"error": f"Download failed: {e}"}), 500

            saved = []
            filtered_fake = errors = 0

            for f in files:
                try:
                    raw = f.read_bytes()

                    # AI art / drawing filter: measure sensor noise via Gaussian residual.
                    # Real camera photos always have noise (std > 3).
                    # AI-generated images and drawings are unnaturally smooth (std < 3).
                    try:
                        import cv2
                        import numpy as np
                        arr = np.frombuffer(raw, np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            blurred   = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 0)
                            noise_std = (img.astype(np.float32) - blurred).std()
                            if noise_std < 3.0:
                                filtered_fake += 1
                                continue
                    except Exception:
                        pass  # if cv2 unavailable, don't reject

                    dest = DRIVE_PULL_DIR / f.name
                    dest.write_bytes(raw)
                    saved.append(f.name)
                except Exception:
                    errors += 1

        return jsonify({"saved": len(saved), "files": saved,
                        "filtered_fake": filtered_fake, "errors": errors})

    @app.route("/drive_pull_image/<path:fname>")
    @lr
    def drive_pull_image(fname):
        base = DRIVE_PULL_DIR.resolve()
        full = (base / fname).resolve()
        try: full.relative_to(base)
        except ValueError: return "Forbidden", 403
        if full.exists():
            try: return send_file(str(full))
            except Exception: pass
        return _MSG_NOT_FOUND, 404

    @app.route("/drive_pull_upload", methods=["POST"])
    @lr
    def drive_pull_upload():
        import shutil
        d           = request.json or {}
        folder_name = (d.get("folder_name") or "FloodAI_Filtered").strip()

        if not DRIVE_PULL_DIR.exists() or not any(DRIVE_PULL_DIR.iterdir()):
            return jsonify({"error": "Không có ảnh nào để upload. Hãy pull trước."}), 400

        try:
            from uploader.drive_uploader import DriveUploader
            uploader = DriveUploader()
        except Exception as e:
            return jsonify({"error": f"Drive auth failed: {e}"}), 500

        try:
            paths = list(DRIVE_PULL_DIR.iterdir())
            share_link = uploader.upload_images_public(paths, folder_name)
            shutil.rmtree(DRIVE_PULL_DIR, ignore_errors=True)
            return jsonify({"status": "ok", "share_link": share_link})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/training_stats")
    @lr
    def training_stats():
        conn     = gtc()
        total    = conn.execute(_SQL_COUNT_TI).fetchone()[0]
        by_level = dict(conn.execute(_SQL_LEVEL_TI).fetchall())
        recent   = [dict(r) for r in conn.execute(
            "SELECT * FROM training_images ORDER BY id DESC LIMIT 50"
        ).fetchall()]
        return jsonify({"total": total, "by_level": by_level, "recent": recent})

    @app.route("/training_summary")
    @lr
    def training_summary():
        conn     = gtc()
        total    = conn.execute(_SQL_COUNT_TI).fetchone()[0]
        by_level = dict(conn.execute(_SQL_LEVEL_TI).fetchall())
        return jsonify({"total": total, "by_level": by_level})

    @app.route("/delete_training/<int:tid>", methods=["DELETE"])
    @lr
    def delete_training(tid):
        conn = gtc()
        row  = conn.execute(_SQL_IMG_BY_ID, (tid,)).fetchone()
        if row:
            Path(row["image_path"]).unlink(missing_ok=True)
            conn.execute("DELETE FROM training_images WHERE id=?", (tid,))
            conn.commit()
        return jsonify({"status": "ok"})

    @app.route("/review_stats")
    @lr
    def review_stats_api():
        return jsonify({k: (v or 0) for k, v in gl().get_review_stats().items()})

    @app.route("/training_image/<int:tid>")
    @lr
    def training_image(tid):
        try:
            row = gtc().execute(_SQL_IMG_BY_ID, (tid,)).fetchone()
            if row:
                p = Path(row["image_path"])
                for candidate in (p, TRAIN_IMG_DIR / p.name, LIVE_DIR / p.name):
                    if candidate.exists():
                        return send_file(str(candidate.resolve()))
        except Exception:
            pass
        return _MSG_NOT_FOUND, 404


def _reg_live(app, gl, gtc, gle, send_file):
    from flask import jsonify, request

    @app.route("/live_predict", methods=["POST"])
    def live_predict():
        d = request.json or {}
        if "data_b64" not in d or "filename" not in d:
            return jsonify({"error": "Cần filename và data_b64"}), 400
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        raw      = base64.b64decode(d["data_b64"])
        img_hash = hashlib.sha256(raw).hexdigest()[:12]
        suffix   = Path(d["filename"]).suffix.lower() or ".jpg"
        img_path = LIVE_DIR / f"live_{img_hash}{suffix}"
        img_path.write_bytes(raw)
        estimator, err = gle()
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
            ai_info = {}
            try:
                _al = get_ai_singleton()
                if _al:
                    ai_info = _al.correct(
                        predicted_depth=float(getattr(r, "water_height_cm", 0) or 0),
                        predicted_level=getattr(r, "flood_level", "NO_FLOOD") or "NO_FLOOD",
                        confidence=float(getattr(r, "confidence", 0) or 0),
                        image_features=json.loads(d.get("features") or "{}"),
                    )
                    if ai_info.get("ai_active"):
                        r.water_height_cm = ai_info["depth_cm"]
                        r.flood_level     = ai_info["level"]
                        r.confidence      = ai_info["confidence_cal"]
            except Exception:
                pass
            return jsonify({
                "status":              "ok",
                "image_hash":          img_hash,
                "flood_depth_cm":      round(float(getattr(r, "water_height_cm", 0) or 0), 1),
                "flood_level":         getattr(r, "flood_level", "Không xác định"),
                "confidence":          round(float(getattr(r, "confidence", 0) or 0), 3),
                "method":              getattr(r, "method", ""),
                "overlay_name":        overlay_rel,
                "detections":          len(getattr(r, "detected_objects", []) or []),
                "vehicles_detected":   getattr(r, "vehicles_detected", []) or [],
                "ai_active":           ai_info.get("ai_active", False),
                "ai_depth_correction": ai_info.get("depth_correction", 0.0),
                "ai_level_correction": ai_info.get("level_correction", 0),
                "ai_confidence_raw":   ai_info.get("confidence_raw", 0.0),
                "ai_n_neighbors":      ai_info.get("n_neighbors", 0),
                "ai_bias_detected":    ai_info.get("bias_detected", False),
                "ai_bias_suggestion":  ai_info.get("bias_suggestion"),
                "ai_explanation":      ai_info.get("explanation", ""),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/live_image/<path:fname>")
    def live_image(fname):
        base = LIVE_DIR.resolve()
        full = (base / fname).resolve()
        try: full.relative_to(base)
        except ValueError: return "Forbidden", 403
        if full.exists():
            try: return send_file(str(full))
            except Exception: pass
        return _MSG_NOT_FOUND, 404

    @app.route("/live_feedback", methods=["POST"])
    def live_feedback():
        d = request.json or {}
        if not {"filename", "data_b64", "actual_level"}.issubset(d):
            return jsonify({"error": "Thiếu trường bắt buộc"}), 400
        try:
            raw      = base64.b64decode(d["data_b64"])
            img_hash = hashlib.sha256(raw).hexdigest()[:12]
            suffix   = Path(d["filename"]).suffix.lower() or ".jpg"
            img_path = LIVE_DIR / f"live_{img_hash}{suffix}"
            img_path.write_bytes(raw)
            conn = gtc()
            conn.execute(
                "INSERT OR IGNORE INTO training_images "
                "(added_at,image_path,image_hash,actual_depth,actual_level,notes,source,verified) "
                "VALUES (?,?,?,?,?,?,?,1)",
                (datetime.now().isoformat(timespec="seconds"), str(img_path), img_hash,
                 float(d["actual_depth"]) if d.get("actual_depth") is not None else 0.0,
                 d["actual_level"], d.get("notes", ""),
                 "live_correct" if d.get("is_correct") else "live_corrected"),
            )
            conn.commit()
            if not d.get("is_correct"):
                _invalidate_live_estimator()
            get_ai_singleton()
            _trigger_ai_retrain()
            return jsonify({"status": "ok", "image_hash": img_hash})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/live_reload_estimator", methods=["POST"])
    def live_reload_estimator():
        _invalidate_live_estimator()
        return jsonify({"status": "ok"})

    @app.route("/live_history")
    def live_history():
        items = []
        if LIVE_DIR.exists():
            for f in sorted(LIVE_DIR.iterdir(), reverse=True):
                if not (f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}):
                    continue
                if "_overlay" in f.name or "_depth" in f.name:
                    continue
                overlay_name = f.stem + "_overlay.jpg"
                items.append({
                    "name":    f.name,
                    "overlay": overlay_name if (LIVE_DIR / overlay_name).exists() else None,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "mtime":   datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
        db_map = {}
        try:
            rows = gtc().execute(
                "SELECT image_hash, actual_depth, actual_level, notes, source, added_at "
                "FROM training_images WHERE source LIKE 'live%' ORDER BY id DESC"
            ).fetchall()
            for row in rows:
                db_map[row["image_hash"]] = dict(row)
        except Exception:
            pass
        for item in items:
            h = item["name"].replace("live_", "").split(".")[0]
            if h in db_map:
                item["db"] = db_map[h]
        return jsonify({"items": items, "total": len(items)})


def _reg_history(app, lr, send_file):
    from flask import jsonify, request

    @app.route("/run_history")
    @lr
    def run_history():
        output_root = ROOT_DIR / "output"
        runs = []
        if output_root.exists():
            for run_dir in sorted(output_root.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue
                sf = run_dir / "pipeline_summary.json"
                try:    s = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
                except Exception: s = {}
                s["run_dir"] = run_dir.name
                exts = {".jpg", ".jpeg", ".png", ".webp"}
                for key, sub in [("overlay_images", "depth_overlays"),
                                  ("original_images", "original_images")]:
                    sub_dir = run_dir / sub
                    s[key] = [f.name for f in sorted(sub_dir.glob("*"))
                              if f.suffix.lower() in exts] if sub_dir.exists() else []
                runs.append(s)
        return jsonify(runs)

    @app.route("/run_detail/<run_dir>")
    @lr
    def run_detail(run_dir):
        import csv as _csv
        run_path = ROOT_DIR / "output" / run_dir
        if not run_path.exists():
            return jsonify({"error": _MSG_NOT_FOUND}), 404
        sf = run_path / "pipeline_summary.json"
        try:    summary = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
        except Exception: summary = {}
        rows = []
        for csv_name in ("flood_analysis_report.csv", "results.csv", "report.csv"):
            csv_path = run_path / csv_name
            if csv_path.exists():
                try:
                    with open(csv_path, encoding="utf-8") as f:
                        rows = list(_csv.DictReader(f))
                except Exception: rows = []
                break
        return jsonify({"summary": summary, "rows": rows, "run_dir": run_dir})

    @app.route("/run_csv/<run_dir>")
    @lr
    def run_csv_download(run_dir):
        run_path = ROOT_DIR / "output" / run_dir
        for csv_name in ("flood_analysis_report.csv", "results.csv", "report.csv"):
            csv_path = run_path / csv_name
            if csv_path.exists():
                return send_file(str(csv_path), as_attachment=True,
                                 download_name=f"flood_{run_dir}.csv",
                                 mimetype="text/csv")
        return jsonify({"error": "CSV not found"}), 404


def _reg_pipeline_ctrl(app, lr):
    from flask import jsonify, request

    @app.route("/pipeline_defaults")
    @lr
    def pipeline_defaults():
        try:
            import yaml as _yaml
            cfg_path = ROOT_DIR / "config.yaml"
            if cfg_path.exists():
                raw = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                from utils.constants import DEFAULT_YOLO_MODEL, DEFAULT_DEPTH_MODEL
                return jsonify({
                    "yolo_model":   raw.get("yolo_model",   DEFAULT_YOLO_MODEL),
                    "depth_model":  raw.get("depth_model",  DEFAULT_DEPTH_MODEL),
                    "output_dir":   raw.get("output_dir",   "output"),
                    "drive_folder": raw.get("drive_folder", "FloodAnalysis"),
                })
        except Exception: pass
        return jsonify({})

    @app.route("/start_pipeline", methods=["POST"])
    @lr
    def start_pipeline():
        with _pl_lock:
            if _pl["status"] == "running":
                return jsonify({"error": "Pipeline đang chạy"}), 409
            _pl.update({"status": "running", "log": [], "step": 0,
                        "stop_flag": False, "run_id": None})
        threading.Thread(target=_run_pipeline, args=(request.json or {},),
                         daemon=True).start()
        return jsonify({"status": "started"})

    @app.route("/pipeline_log")
    @lr
    def pipeline_log():
        offset = int(request.args.get("offset", 0))
        with _pl_lock:
            lines  = _pl["log"][offset:]
            status = _pl["status"]
            step   = _pl["step"]
            nxt    = len(_pl["log"])
        return jsonify({"lines": lines, "next_offset": nxt, "status": status, "step": step})

    @app.route("/stop_pipeline", methods=["POST"])
    @lr
    def stop_pipeline():
        with _pl_lock:
            _pl["stop_flag"] = True
            _pl["status"]    = "idle"
        return jsonify({"status": "ok"})


def _reg_cases(app, lr, gl, gtc, send_file):
    from flask import jsonify, request

    @app.route("/review_cases")
    @lr
    def review_cases():
        try:
            learner = gl()
            cases   = learner.get_pending_reviews(max_count=20)
            fields  = ["id", "timestamp", "image_hash", "image_path",
                       "predicted_depth", "predicted_level", "confidence",
                       "review_reason", "score"]
            result  = [{k: (c.to_dict() if hasattr(c, "to_dict") else {}).get(k)
                        for k in fields} for c in cases]
            underrepresented = []
            try:
                underrepresented = learner.get_underrepresented_levels() \
                    if hasattr(learner, "get_underrepresented_levels") else []
            except Exception: pass
            return jsonify({"cases": result, "underrepresented": underrepresented})
        except Exception as e:
            return jsonify({"cases": [], "underrepresented": [], "error": str(e)})

    @app.route("/case_image/<int:case_id>")
    @lr
    def case_image(case_id):
        try:
            case = gl()._get_case_by_id(case_id)
            if case:
                p = Path(case.image_path)
                for candidate in (p, TRAIN_IMG_DIR / p.name, LIVE_DIR / p.name):
                    if candidate.exists():
                        return send_file(str(candidate.resolve()))
        except Exception: pass
        return "Image not found", 404

    @app.route("/annotate_live_image", methods=["POST"])
    @lr
    def annotate_live_image():
        d            = request.json or {}
        image_name   = d.get("image_name", "")
        actual_level = d.get("actual_level", "")
        actual_depth = float(d.get("actual_depth") or 0)
        notes        = d.get("notes", "")
        if not image_name or not actual_level:
            return jsonify({"error": "Thiếu image_name hoặc actual_level"}), 400
        img_path = LIVE_DIR / image_name
        if not img_path.exists():
            return jsonify({"error": f"Không tìm thấy ảnh: {image_name}"}), 404
        img_hash  = hashlib.sha256(img_path.read_bytes()).hexdigest()[:12]
        suffix    = img_path.suffix.lower() or ".jpg"
        TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
        train_dst = TRAIN_IMG_DIR / f"{img_hash}{suffix}"
        if not train_dst.exists():
            shutil.copy2(str(img_path), str(train_dst))
        try:
            conn = gtc()
            conn.execute(
                "INSERT OR REPLACE INTO training_images "
                "(added_at,image_path,image_hash,actual_depth,actual_level,notes,source,verified) "
                "VALUES (?,?,?,?,?,?,'history_annotate',1)",
                (datetime.now().isoformat(timespec="seconds"), str(train_dst), img_hash,
                 actual_depth, actual_level, notes),
            )
            conn.commit()
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        get_ai_singleton()
        _trigger_ai_retrain()
        return jsonify({"status": "ok", "image_hash": img_hash, "train_path": str(train_dst)})

    @app.route("/trigger_retrain", methods=["POST"])
    @lr
    def trigger_retrain():
        get_ai_singleton()
        _trigger_ai_retrain()
        return jsonify({"status": "ok", "message": "Retrain đã được kích hoạt trong nền"})


def _reg_active_learning(app, lr, gl, gtc):
    from flask import jsonify, request

    def _score_case(c, level_counts, max_count):
        d       = c.to_dict() if hasattr(c, "to_dict") else {}
        conf    = float(d.get("confidence", 0.5) or 0.5)
        level   = str(d.get("predicted_level", "MEDIUM") or "MEDIUM")
        p       = max(1e-6, min(1 - 1e-6, conf))
        entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        margin  = max(0, 1.0 - min(abs(conf - 0.40), abs(conf - 0.70)) / 0.70)
        underrep = max(0, 1.0 - level_counts.get(level, 0) / max(max_count, 1))
        priority = entropy * 0.40 + margin * 0.35 + underrep * 0.25
        if entropy > 0.85:
            tag = "🔴 Rất không chắc"
        elif entropy > 0.65:
            tag = "🟠 Không chắc"
        elif margin > 0.70:
            tag = "🟡 Gần ngưỡng"
        else:
            tag = "🟢 Tương đối chắc"
        return {
            **{k: d.get(k) for k in ["id", "timestamp", "image_hash", "image_path",
                                      "predicted_depth", "review_reason", "score"]},
            "predicted_level": level, "confidence": round(conf, 3),
            "priority_score": round(priority, 4), "entropy_score": round(entropy, 3),
            "margin_score": round(margin, 3), "underrep_score": round(underrep, 3),
            "uncertainty_tag": tag,
        }

    @app.route("/uncertainty_ranking")
    @lr
    def uncertainty_ranking():
        try:
            learner = gl()
            cases   = learner.get_pending_reviews(max_count=100)
            if not cases:
                return jsonify({"ranked_cases": [], "total_pending": 0,
                                "recommendation": "Không có ảnh cần review"})
            level_counts = dict(gtc().execute(_SQL_LEVEL_TI).fetchall())
            max_count    = max(level_counts.values()) if level_counts else 1
            scored = [_score_case(c, level_counts, max_count) for c in cases]
            scored.sort(key=lambda x: x["priority_score"], reverse=True)
            return jsonify({"ranked_cases": scored[:50], "total_pending": len(scored)})
        except Exception as e:
            return jsonify({"ranked_cases": [], "error": str(e)}), 500

    @app.route("/active_learning_status")
    @lr
    def active_learning_status():
        try:
            stats    = gl().get_review_stats()
            conn     = gtc()
            total    = conn.execute(_SQL_COUNT_TI).fetchone()[0]
            by_level = dict(conn.execute(_SQL_LEVEL_TI).fetchall())
            retrain_count = 0
            cache_path = BASE_DIR / "ai_model_cache.json"
            if cache_path.exists():
                try: retrain_count = json.loads(cache_path.read_text()).get("retrain_count", 0)
                except Exception: pass
            return jsonify({
                "status": "active",
                "total_training_images":     total,
                "training_by_level":         by_level,
                "pending_reviews":           stats.get("pending", 0),
                "reviewed_total":            stats.get("reviewed", 0),
                "retrain_count":             retrain_count,
                "auto_retrain_threshold":    _al_state.get("retrain_threshold", 10),
                "reviews_since_last_retrain": _al_state.get("reviews_since_retrain", 0),
            })
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/configure_active_learning", methods=["POST"])
    @lr
    def configure_active_learning():
        try:
            data = request.get_json() or {}
            if "retrain_threshold" in data:
                _al_state["retrain_threshold"] = max(1, int(data["retrain_threshold"]))
            if "auto_retrain_enabled" in data:
                _al_state["auto_retrain_enabled"] = bool(data["auto_retrain_enabled"])
            return jsonify({"status": "ok", "config": {
                "retrain_threshold":    _al_state["retrain_threshold"],
                "auto_retrain_enabled": _al_state["auto_retrain_enabled"],
            }})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 400

    @app.route("/al_review_complete", methods=["POST"])
    @lr
    def al_review_complete():
        def _hook():
            if not _al_state.get("auto_retrain_enabled", True):
                return
            with _al_state["retrain_lock"]:
                _al_state["reviews_since_retrain"] += 1
                if _al_state["reviews_since_retrain"] >= _al_state.get("retrain_threshold", 10):
                    _al_state["reviews_since_retrain"] = 0
                    _al_state["last_retrain_ts"] = datetime.now().isoformat()
                    get_ai_singleton()
                    _trigger_ai_retrain()
        try:
            threading.Thread(target=_hook, daemon=True).start()
            return jsonify({
                "status": "ok",
                "reviews_since_last_retrain": _al_state.get("reviews_since_retrain", 0),
                "reviews_until_retrain": max(
                    0, _al_state.get("retrain_threshold", 10)
                       - _al_state.get("reviews_since_retrain", 0) - 1),
            })
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/uncertainty_heatmap")
    @lr
    def uncertainty_heatmap():
        try:
            cases = gl().get_pending_reviews(max_count=200)
            bins  = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
            level_conf: dict = {}
            for c in cases:
                d    = c.to_dict() if hasattr(c, "to_dict") else {}
                conf = float(d.get("confidence", 0.5) or 0.5)
                lvl  = str(d.get("predicted_level", "MEDIUM") or "MEDIUM")
                if   conf < 0.2: bins["0.0-0.2"] += 1
                elif conf < 0.4: bins["0.2-0.4"] += 1
                elif conf < 0.6: bins["0.4-0.6"] += 1
                elif conf < 0.8: bins["0.6-0.8"] += 1
                else:            bins["0.8-1.0"] += 1
                level_conf.setdefault(lvl, []).append(conf)
            return jsonify({
                "bins": bins,
                "level_uncertainty": {
                    k: round(sum(v) / len(v), 3) for k, v in level_conf.items() if v
                },
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500


def _reg_gallery(app, lr, gtc, send_file):
    from flask import jsonify, request

    @app.route("/gallery_images")
    @lr
    def gallery_images():
        try:
            rows = gtc().execute(
                "SELECT id, image_path, image_hash, actual_level, actual_depth, "
                "notes, source, added_at, category "
                "FROM training_images ORDER BY id DESC"
            ).fetchall()
            images = [{"id": r["id"], "filename": Path(r["image_path"]).name,
                       "actual_level": r["actual_level"], "actual_depth": r["actual_depth"],
                       "notes": r["notes"], "source": r["source"], "added_at": r["added_at"],
                       "category": r["category"] or "untagged",
                       "exists": Path(r["image_path"]).exists()} for r in rows]
            return jsonify({"images": images, "total": len(images)})
        except Exception as e:
            return jsonify({"images": [], "error": str(e)}), 500

    @app.route("/tag_image/<int:tid>", methods=["POST"])
    @lr
    def tag_image(tid):
        data    = request.get_json() or {}
        cat     = data.get("category", "untagged")
        allowed = {"ai", "hand_drawn", "oil_painting", "photo", "untagged"}
        if cat not in allowed:
            return jsonify({"error": f"category phải là một trong {allowed}"}), 400
        try:
            conn = gtc()
            conn.execute("UPDATE training_images SET category=? WHERE id=?", (cat, tid))
            conn.commit()
            return jsonify({"status": "ok", "id": tid, "category": cat})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/upload_gallery_to_drive", methods=["POST"])
    @lr
    def upload_gallery_to_drive():
        data        = request.get_json() or {}
        image_ids   = [int(i) for i in (data.get("image_ids") or [])]
        folder_name = str(data.get("folder_name") or "FloodAI_Gallery").strip() or "FloodAI_Gallery"
        if not image_ids:
            return jsonify({"error": "Không có ảnh nào được chọn"}), 400
        try:
            conn  = gtc()
            paths = []
            for tid in image_ids:
                row = conn.execute(_SQL_IMG_BY_ID, (tid,)).fetchone()
                if not row:
                    continue
                p = Path(row["image_path"])
                for candidate in (p, TRAIN_IMG_DIR / p.name, LIVE_DIR / p.name):
                    if candidate.exists():
                        paths.append(candidate)
                        break
            if not paths:
                return jsonify({"error": "Không tìm thấy file ảnh nào trên đĩa"}), 404
            from uploader.drive_uploader import DriveUploader
            share_link = DriveUploader().upload_images_public(paths, folder_name)
            return jsonify({"status": "ok", "share_link": share_link,
                            "uploaded": len(paths), "folder_name": folder_name})
        except FileNotFoundError as e:
            return jsonify({"error": f"Drive chưa cấu hình: {e}"}), 503
        except Exception as e:
            log.exception("[gallery_upload] %s", e)
            return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def register_admin(app):
    """Register all admin/review/pipeline routes onto a Flask app instance."""
    from flask import g, jsonify, redirect, request, send_file, session

    def login_required(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not session.get("logged_in"):
                if request.is_json or request.method != "GET":
                    return jsonify({"error": "Chưa đăng nhập"}), 401
                return redirect("/login")
            return f(*args, **kwargs)
        return wrapped

    def get_learner():
        if "adm_learner" not in g:
            from learning.active_learner import ActiveLearnerV2 as AL
            g.adm_learner = AL()
        return g.adm_learner

    def get_train_conn():
        if "adm_tc" not in g:
            conn = sqlite3.connect(str(TRAIN_DB_PATH), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS training_images (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    added_at     TEXT NOT NULL,
                    image_path   TEXT NOT NULL,
                    image_hash   TEXT UNIQUE,
                    actual_depth REAL NOT NULL,
                    actual_level TEXT NOT NULL,
                    notes        TEXT DEFAULT '',
                    source       TEXT DEFAULT 'manual',
                    verified     INTEGER DEFAULT 1,
                    category     TEXT DEFAULT 'untagged'
                );
                CREATE INDEX IF NOT EXISTS idx_ti_level    ON training_images(actual_level);
                CREATE INDEX IF NOT EXISTS idx_ti_category ON training_images(category);
            """)
            try:
                conn.execute("ALTER TABLE training_images ADD COLUMN category TEXT DEFAULT 'untagged'")
                conn.commit()
            except Exception:
                pass
            g.adm_tc = conn
        return g.adm_tc

    @app.teardown_appcontext
    def _close_admin(exc=None):
        lrn = g.pop("adm_learner", None)
        if lrn:
            try: lrn.close()
            except Exception: pass
        tc = g.pop("adm_tc", None)
        if tc:
            try: tc.close()
            except Exception: pass

    def _get_live_estimator():
        global _live_estimator
        if _live_estimator is not None:
            return _live_estimator, None
        try:
            from depth_analysis.reference_estimator import ReferenceEstimator
            import yaml as _yaml
            cfg_path = ROOT_DIR / "config.yaml"
            raw_cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
            try:
                from learning_update import SelfLearningPipeline
                sl = SelfLearningPipeline()
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
                use_dino     = raw_cfg.get("use_dino",  True),
                dino_model   = raw_cfg.get("dino_model", "facebook/dinov2-small"),
                use_pose     = raw_cfg.get("use_pose",  True),
                pose_model   = raw_cfg.get("pose_model", "yolov8n-pose.pt"),
                use_segformer= raw_cfg.get("use_segformer", True),
            )
            return _live_estimator, None
        except Exception as e:
            return None, str(e)

    lr, gl, gtc, gle = login_required, get_learner, get_train_conn, _get_live_estimator
    _reg_core(app, lr, gl, gtc, send_file)
    _reg_training(app, lr, gl, gtc, send_file)
    _reg_live(app, gl, gtc, gle, send_file)
    _reg_history(app, lr, send_file)
    _reg_pipeline_ctrl(app, lr)
    _reg_cases(app, lr, gl, gtc, send_file)
    _reg_active_learning(app, lr, gl, gtc)
    _reg_gallery(app, lr, gtc, send_file)
    log.info("[admin_routes] %d routes registered ✓", sum(1 for _ in app.url_map.iter_rules()))
