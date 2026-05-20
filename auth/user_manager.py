# -*- coding: utf-8 -*-
"""
auth/user_manager.py
=====================
Quản lý users + roles dùng SQLite (cùng DB với pipeline).

Roles:
    admin       — toàn quyền, phân quyền được
    editor      — tạo/sửa/đăng bài
    alert_mgr   — tạo/hạ/gia hạn cảnh báo
    reviewer    — duyệt/từ chối báo cáo

Dùng trong app.py:
    from auth.user_manager import UserManager, require_role

    um = UserManager()

    @app.route("/api/users")
    @require_role("admin")
    def list_users(): ...
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flask import jsonify, session
from functools import wraps

log = logging.getLogger("auth")

# ── Constants ────────────────────────────────────────────────────────────────

ROLES = ["admin", "editor", "alert_mgr", "reviewer"]

ROLE_PERMISSIONS = {
    "reviewer":  ["approve_report", "reject_report", "view_queue"],
    "editor":    ["approve_report", "reject_report", "view_queue",
                  "create_article", "edit_article", "publish_article",
                  "create_event", "edit_event"],
    "alert_mgr": ["approve_report", "reject_report", "view_queue",
                  "create_article", "edit_article",
                  "create_alert", "update_alert", "send_notification"],
    "admin":     ["*"],  # toàn quyền
}

DB_PATH = Path(__file__).parent.parent / "_agent_memory" / "users.db"


# ── DB context ────────────────────────────────────────────────────────────────

@contextmanager
def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── UserManager ───────────────────────────────────────────────────────────────

class UserManager:
    """
    CRUD users + roles.
    Tự tạo bảng và tài khoản admin mặc định khi khởi động.
    """

    def __init__(self, default_admin_pass: Optional[str] = None):
        self._init_db()
        self._ensure_default_admin(default_admin_pass)

    # ── Setup ────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with _db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    email       TEXT    UNIQUE NOT NULL,
                    name        TEXT    NOT NULL,
                    pass_hash   TEXT    NOT NULL,
                    role        TEXT    NOT NULL DEFAULT 'reviewer',
                    active      INTEGER NOT NULL DEFAULT 0,
                    pending     INTEGER NOT NULL DEFAULT 1,
                    created_at  TEXT    NOT NULL,
                    last_login  TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor      TEXT    NOT NULL,
                    action     TEXT    NOT NULL,
                    target     TEXT,
                    detail     TEXT,
                    created_at TEXT    NOT NULL
                )
            """)

    def _ensure_default_admin(self, pw: Optional[str] = None) -> None:
        """Tạo admin mặc định nếu chưa có user nào."""
        with _db() as conn:
            row = conn.execute("SELECT id FROM users LIMIT 1").fetchone()
            if row:
                return
            password = pw or os.environ.get("ADMIN_PASS", "123456")
            admin_email = os.environ.get("ADMIN_USER", "admin@floodwatch.vn")
            conn.execute(
                "INSERT INTO users (email, name, pass_hash, role, active, pending, created_at) "
                "VALUES (?, ?, ?, 'admin', 1, 0, ?)",
                (admin_email, "Admin", _hash(password), _now())
            )
            log.info(f"[Auth] Tạo admin mặc định: {admin_email}")

    # ── Auth ─────────────────────────────────────────────────────────────

    def authenticate(self, email: str, password: str) -> Optional[Dict]:
        """
        Xác thực email + password.
        Returns: user dict hoặc None nếu sai/bị khóa.
        """
        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email=? AND active=1 AND pending=0",
                (email,)
            ).fetchone()
            if not row:
                return None
            if row["pass_hash"] != _hash(password):
                return None
            conn.execute(
                "UPDATE users SET last_login=? WHERE email=?",
                (_now(), email)
            )
            return dict(row)

    # ── User CRUD ─────────────────────────────────────────────────────────

    def register(self, email: str, name: str, password: str,
                 requested_role: str = "reviewer") -> Dict:
        """
        Đăng ký tài khoản mới — trạng thái pending, cần admin phê duyệt.
        """
        if requested_role not in ROLES or requested_role == "admin":
            requested_role = "reviewer"
        with _db() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE email=?", (email,)
            ).fetchone()
            if existing:
                return {"error": "Email đã tồn tại"}
            conn.execute(
                "INSERT INTO users (email, name, pass_hash, role, active, pending, created_at) "
                "VALUES (?, ?, ?, ?, 0, 1, ?)",
                (email, name, _hash(password), requested_role, _now())
            )
        log.info(f"[Auth] Đăng ký mới: {email} → {requested_role} (pending)")
        return {"ok": True, "message": "Đã gửi yêu cầu — chờ admin phê duyệt"}

    def approve(self, email: str, actor: str, role: Optional[str] = None) -> Dict:
        """Admin phê duyệt tài khoản pending, có thể đổi role ngay lúc duyệt."""
        with _db() as conn:
            row = conn.execute("SELECT role FROM users WHERE email=? AND pending=1", (email,)).fetchone()
            if not row:
                return {"error": "Không tìm thấy tài khoản pending"}
            final_role = role if role in ROLES else row["role"]
            conn.execute(
                "UPDATE users SET active=1, pending=0, role=? WHERE email=?",
                (final_role, email)
            )
            self._log(conn, actor, "approve_user", email, f"role={final_role}")
        return {"ok": True, "role": final_role}

    def reject(self, email: str, actor: str) -> Dict:
        """Admin từ chối và xóa tài khoản pending."""
        with _db() as conn:
            conn.execute("DELETE FROM users WHERE email=? AND pending=1", (email,))
            self._log(conn, actor, "reject_user", email)
        return {"ok": True}

    def change_role(self, email: str, new_role: str, actor: str) -> Dict:
        """Đổi role — chỉ admin được gọi."""
        if new_role not in ROLES:
            return {"error": f"Role không hợp lệ: {new_role}"}
        with _db() as conn:
            row = conn.execute("SELECT role FROM users WHERE email=?", (email,)).fetchone()
            if not row:
                return {"error": "User không tồn tại"}
            old_role = row["role"]
            conn.execute("UPDATE users SET role=? WHERE email=?", (new_role, email))
            self._log(conn, actor, "change_role", email,
                      f"{old_role} → {new_role}")
        log.info(f"[Auth] {actor} đổi quyền {email}: {old_role} → {new_role}")
        return {"ok": True}

    def set_active(self, email: str, active: bool, actor: str) -> Dict:
        """Khóa / mở khóa tài khoản."""
        with _db() as conn:
            conn.execute(
                "UPDATE users SET active=? WHERE email=?",
                (1 if active else 0, email)
            )
            action = "unlock_user" if active else "lock_user"
            self._log(conn, actor, action, email)
        return {"ok": True}

    def add_user(self, email: str, name: str, role: str,
                 password: str, actor: str) -> Dict:
        """Admin thêm user trực tiếp (không cần pending)."""
        if role not in ROLES:
            return {"error": f"Role không hợp lệ"}
        with _db() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE email=?", (email,)
            ).fetchone()
            if existing:
                return {"error": "Email đã tồn tại"}
            conn.execute(
                "INSERT INTO users (email, name, pass_hash, role, active, pending, created_at) "
                "VALUES (?, ?, ?, ?, 1, 0, ?)",
                (email, name, _hash(password), role, _now())
            )
            self._log(conn, actor, "add_user", email, f"role={role}")
        return {"ok": True}

    # ── Queries ───────────────────────────────────────────────────────────

    def list_users(self) -> List[Dict]:
        with _db() as conn:
            rows = conn.execute(
                "SELECT email, name, role, active, pending, created_at, last_login "
                "FROM users ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_pending(self) -> List[Dict]:
        with _db() as conn:
            rows = conn.execute(
                "SELECT email, name, role, created_at FROM users WHERE pending=1"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_user(self, email: str) -> Optional[Dict]:
        with _db() as conn:
            row = conn.execute(
                "SELECT email, name, role, active, pending FROM users WHERE email=?",
                (email,)
            ).fetchone()
        return dict(row) if row else None

    # ── Permissions ───────────────────────────────────────────────────────

    @staticmethod
    def can(role: str, action: str) -> bool:
        perms = ROLE_PERMISSIONS.get(role, [])
        return "*" in perms or action in perms

    # ── Audit log ─────────────────────────────────────────────────────────

    def _log(self, conn, actor: str, action: str,
             target: str = "", detail: str = "") -> None:
        conn.execute(
            "INSERT INTO auth_log (actor, action, target, detail, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (actor, action, target, detail, _now())
        )

    def get_auth_log(self, n: int = 50) -> List[Dict]:
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM auth_log ORDER BY created_at DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Flask decorators ──────────────────────────────────────────────────────────

def require_login(f):
    """Redirect về /login nếu chưa đăng nhập."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "Chưa đăng nhập"}), 401
            from flask import redirect, url_for, request as req
            return redirect(f"/login?next={req.path}")
        return f(*args, **kwargs)
    return wrapped


def require_role(*roles):
    """
    Chỉ cho phép các role nhất định.

    Dùng:
        @require_role("admin")
        @require_role("admin", "editor")
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not session.get("logged_in"):
                return jsonify({"error": "Chưa đăng nhập"}), 401
            user_role = session.get("role", "")
            if user_role not in roles and user_role != "admin":
                role_names = {
                    "admin": "System Admin",
                    "editor": "Editor",
                    "alert_mgr": "Alert Manager",
                    "reviewer": "Reviewer",
                }
                needed = " hoặc ".join(role_names.get(r, r) for r in roles)
                return jsonify({
                    "error": f"Không đủ quyền. Cần: {needed}"
                }), 403
            return f(*args, **kwargs)
        return wrapped
    return decorator
