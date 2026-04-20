# -*- coding: utf-8 -*-
"""
setup_env.py
------------
Tự động kiểm tra và cài đặt thư viện + tải model lần đầu chạy.
Gọi từ main.py trước khi bắt đầu pipeline:

    from setup_env import ensure_setup
    ensure_setup(silent=False)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ── Các model file cần có (tải tự động qua ultralytics nếu thiếu) ─────────────
_REQUIRED_MODELS: list[dict] = [
    {
        "filename": "yolov8n.pt",
        "description": "YOLOv8n — object detection",
    },
    {
        "filename": "yolov8n-pose.pt",
        "description": "YOLOv8n-pose — human pose estimation",
    },
]

_ROOT = Path(__file__).parent


def _pip_install(requirements_path: Path, silent: bool) -> bool:
    """Cài tất cả package từ requirements.txt. Trả về True nếu thành công."""
    if not requirements_path.exists():
        if not silent:
            print(f"  ⚠️  Không tìm thấy {requirements_path}, bỏ qua cài đặt.")
        return False
    try:
        cmd = [
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_path),
            "--quiet", "--disable-pip-version-check",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0 and not silent:
            print(f"  ⚠️  pip install gặp lỗi (có thể bỏ qua nếu đã cài):\n{result.stderr[:400]}")
        return result.returncode == 0
    except Exception as exc:
        if not silent:
            print(f"  ⚠️  Không thể chạy pip: {exc}")
        return False


def _ensure_models(silent: bool) -> None:
    """Tải các YOLO model nếu chưa có (ultralytics tự tải khi import)."""
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        if not silent:
            print("  ⚠️  ultralytics chưa được cài — bỏ qua kiểm tra model.")
        return

    for m in _REQUIRED_MODELS:
        model_path = _ROOT / m["filename"]
        if model_path.exists():
            if not silent:
                print(f"  ✅ {m['filename']} — đã có")
            continue
        if not silent:
            print(f"  ⬇️  Đang tải {m['filename']} ({m['description']})...")
        try:
            YOLO(m["filename"])          # ultralytics tự tải về thư mục hiện tại
            # Di chuyển về root nếu tải vào cwd khác
            cwd_model = Path(m["filename"])
            if cwd_model.exists() and not model_path.exists():
                cwd_model.rename(model_path)
            if not silent:
                print(f"  ✅ {m['filename']} — đã tải xong")
        except Exception as exc:
            if not silent:
                print(f"  ⚠️  Không tải được {m['filename']}: {exc}")


def _check_critical_imports(silent: bool) -> list[str]:
    """Kiểm tra nhanh các package quan trọng. Trả về list package bị thiếu."""
    critical = {
        "cv2":           "opencv-python",
        "torch":         "torch",
        "ultralytics":   "ultralytics",
        "transformers":  "transformers",
        "flask":         "flask",
        "PIL":           "Pillow",
        "yaml":          "pyyaml",
        "openpyxl":      "openpyxl",
    }
    missing: list[str] = []
    for module, pkg in critical.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)
            if not silent:
                print(f"  ❌ Thiếu: {pkg}")
    return missing


def ensure_setup(silent: bool = True) -> None:
    """
    Điểm vào chính — gọi một lần từ main.py khi khởi động.

    Thực hiện theo thứ tự:
      1. Kiểm tra package thiếu
      2. Nếu có package thiếu → chạy pip install -r requirements.txt
      3. Kiểm tra và tải model YOLO nếu chưa có
    """
    if not silent:
        print("\n" + "─" * 50)
        print("🔧  AUTO SETUP — kiểm tra môi trường...")
        print("─" * 50)

    req_path = _ROOT / "requirements.txt"

    # Bước 1: kiểm tra package
    missing = _check_critical_imports(silent=True)   # im lặng lần đầu

    if missing:
        if not silent:
            print(f"\n  📦 Đang cài {len(missing)} package còn thiếu...")
        installed_ok = _pip_install(req_path, silent)
        if installed_ok and not silent:
            print("  ✅ Cài đặt package xong")
        elif not installed_ok and not silent:
            missing2 = _check_critical_imports(silent=True)
            if missing2:
                print(f"  ⚠️  Vẫn còn thiếu: {missing2}")
                print("       Hãy chạy thủ công: pip install -r requirements.txt")
    else:
        if not silent:
            print("  ✅ Tất cả package đã có")

    # Bước 2: kiểm tra / tải model
    if not silent:
        print("\n  🤖 Kiểm tra model files...")
    _ensure_models(silent)

    if not silent:
        print("─" * 50)
        print("✅  Setup hoàn tất — bắt đầu pipeline\n")
