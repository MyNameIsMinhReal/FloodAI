# -*- coding: utf-8 -*-
"""
Env Loader — Nạp biến môi trường bảo mật
==========================================
Thay thế credentials.json và token.json bằng .env.

Ưu tiên:
  1. Biến môi trường hệ thống (os.environ)
  2. File .env trong project root
  3. Fallback: credentials.json (backward-compat)

Cách dùng:
    from utils.env_loader import get_env, load_google_credentials

    api_key = get_env("GOOGLE_MAPS_API_KEY")
    creds   = load_google_credentials()
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("utils.env_loader")

_ENV_LOADED = False


def _ensure_loaded():
    """Nạp .env một lần duy nhất."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_file = Path(".env")
    if not env_file.exists():
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)  # override=False: env var hệ thống có ưu tiên cao hơn
        log.debug(f"  [Env] Nạp từ {env_file}")
    except ImportError:
        # Parse thủ công nếu không có python-dotenv
        _parse_env_file(env_file)


def _parse_env_file(env_file: Path) -> None:
    """Parser đơn giản cho .env khi không có python-dotenv."""
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as exc:
        log.warning(f"  [Env] Không đọc được .env: {exc}")


def get_env(key: str, default: str = "", required: bool = False) -> str:
    """
    Lấy giá trị từ environment.

    Args:
        key:      Tên biến môi trường
        default:  Giá trị fallback
        required: Nếu True → raise ValueError khi thiếu

    Returns:
        str: Giá trị của biến
    """
    _ensure_loaded()
    val = os.environ.get(key, default)
    if required and not val:
        raise ValueError(
            f"Thiếu biến môi trường bắt buộc: {key}\n"
            f"Thêm vào file .env hoặc set: export {key}=<value>"
        )
    return val


def get_env_int(key: str, default: int = 0) -> int:
    val = get_env(key, str(default))
    try:
        return int(val)
    except ValueError:
        return default


def get_env_bool(key: str, default: bool = False) -> bool:
    val = get_env(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


def load_google_credentials() -> Optional[Dict[str, Any]]:
    """
    Nạp Google credentials theo thứ tự ưu tiên:
    1. Biến môi trường (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN)
    2. File .env
    3. credentials.json (backward-compat)
    """
    _ensure_loaded()

    client_id     = get_env("GOOGLE_CLIENT_ID")
    client_secret = get_env("GOOGLE_CLIENT_SECRET")
    refresh_token = get_env("GOOGLE_REFRESH_TOKEN")

    if client_id and client_secret:
        return {
            "installed": {
                "client_id":     client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "token_uri":     "https://oauth2.googleapis.com/token",
                "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
            }
        }

    # Fallback: đọc từ file (backward-compat)
    cred_path = Path(get_env("GOOGLE_CREDENTIALS_FILE", "credentials.json"))
    if cred_path.exists():
        log.warning(
            "  [Env] Đang dùng credentials.json. "
            "Khuyến nghị chuyển sang .env để bảo mật hơn."
        )
        try:
            return json.loads(cred_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.error(f"  [Env] Không đọc được credentials.json: {exc}")

    return None


def get_maps_api_key() -> str:
    """Lấy Google Maps API key."""
    return get_env("GOOGLE_MAPS_API_KEY", "")


def check_secrets_security() -> None:
    """
    Cảnh báo nếu credentials bị commit vào repo.
    Gọi khi startup.
    """
    dangerous_files = ["credentials.json", "token.json", ".env"]
    gitignore = Path(".gitignore")

    if not gitignore.exists():
        log.warning("  [Security] Không tìm thấy .gitignore!")
        return

    content = gitignore.read_text(encoding="utf-8")
    for fname in dangerous_files:
        if fname not in content and Path(fname).exists():
            log.warning(
                f"  [Security] ⚠ '{fname}' tồn tại nhưng KHÔNG có trong .gitignore!\n"
                f"             Thêm '{fname}' vào .gitignore ngay để tránh lộ credentials."
            )
