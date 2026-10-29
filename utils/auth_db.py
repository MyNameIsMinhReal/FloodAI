# -*- coding: utf-8 -*-
"""
Database & Helper Functions for User Authentication in FloodAI
Uses SQLite for persistent and lightweight storage in data/users.db
"""

import os
import sqlite3
from typing import Optional, Dict, Any
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "users.db")


def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db() -> None:
    """Initialize users table if not exists"""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                province TEXT DEFAULT '',
                address TEXT DEFAULT '',
                role TEXT DEFAULT 'citizen',
                avatar_url TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def register_user(name: str, phone: str, email: str, password: str, province: str = "", address: str = "") -> Dict[str, Any]:
    """Register a new user account"""
    init_user_db()
    name = (name or "").strip()
    phone = (phone or "").strip()
    email = (email or "").strip().lower()
    province = (province or "").strip()
    address = (address or "").strip()

    if not name:
        return {"ok": False, "error": "Vui lòng nhập họ và tên"}
    if not phone and not email:
        return {"ok": False, "error": "Vui lòng cung cấp Số điện thoại hoặc Email để liên hệ"}
    if not password or len(password) < 6:
        return {"ok": False, "error": "Mật khẩu phải có ít nhất 6 ký tự"}

    pwd_hash = generate_password_hash(password)

    try:
        with get_db_connection() as conn:
            # Check duplicate phone or email
            if phone:
                cur = conn.execute("SELECT id FROM users WHERE phone = ?", (phone,))
                if cur.fetchone():
                    return {"ok": False, "error": f"Số điện thoại '{phone}' đã được đăng ký"}
            if email:
                cur = conn.execute("SELECT id FROM users WHERE email = ?", (email,))
                if cur.fetchone():
                    return {"ok": False, "error": f"Email '{email}' đã được đăng ký"}

            cursor = conn.execute("""
                INSERT INTO users (name, phone, email, password_hash, province, address, role)
                VALUES (?, ?, ?, ?, ?, ?, 'citizen')
            """, (name, phone if phone else None, email if email else None, pwd_hash, province, address))
            conn.commit()
            user_id = cursor.lastrowid

            return {
                "ok": True,
                "user": {
                    "id": user_id,
                    "name": name,
                    "phone": phone,
                    "email": email,
                    "province": province,
                    "address": address,
                    "role": "citizen"
                }
            }
    except Exception as e:
        return {"ok": False, "error": f"Lỗi cơ sở dữ liệu: {str(e)}"}


def authenticate_user(identifier: str, password: str) -> Dict[str, Any]:
    """Authenticate user with phone or email or admin username and password"""
    init_user_db()
    ident = (identifier or "").strip()
    if not ident:
        return {"ok": False, "error": "Vui lòng nhập Số điện thoại hoặc Email"}
    if not password:
        return {"ok": False, "error": "Vui lòng nhập Mật khẩu"}

    # Special case: Built-in Admin
    admin_user = os.environ.get("ADMIN_USER", "admin")
    admin_pass = os.environ.get("ADMIN_PASS", "123456")
    if ident.lower() == admin_user.lower() and password == admin_pass:
        return {
            "ok": True,
            "user": {
                "id": 999999,
                "name": "Quản Trị Viên (Admin)",
                "phone": "112",
                "email": "admin@floodai.vn",
                "province": "Ban Chỉ Đạo PCTT",
                "address": "Trung tâm Điều hành Quốc gia",
                "role": "admin"
            }
        }

    with get_db_connection() as conn:
        # Search by phone OR email
        cur = conn.execute("""
            SELECT id, name, phone, email, password_hash, province, address, role, created_at
            FROM users
            WHERE phone = ? OR email = ?
            LIMIT 1
        """, (ident, ident.lower()))
        row = cur.fetchone()
        if not row:
            return {"ok": False, "error": "Tài khoản hoặc mật khẩu không chính xác"}

        if not check_password_hash(row["password_hash"], password):
            return {"ok": False, "error": "Tài khoản hoặc mật khẩu không chính xác"}

        return {
            "ok": True,
            "user": {
                "id": row["id"],
                "name": row["name"],
                "phone": row["phone"] or "",
                "email": row["email"] or "",
                "province": row["province"] or "",
                "address": row["address"] or "",
                "role": row["role"] or "citizen"
            }
        }


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch user profile by ID"""
    if not user_id:
        return None
    if user_id == 999999:
        return {
            "id": 999999,
            "name": "Quản Trị Viên (Admin)",
            "phone": "112",
            "email": "admin@floodai.vn",
            "province": "Ban Chỉ Đạo PCTT",
            "address": "Trung tâm Điều hành Quốc gia",
            "role": "admin"
        }
    try:
        init_user_db()
        with get_db_connection() as conn:
            cur = conn.execute("""
                SELECT id, name, phone, email, province, address, role, created_at
                FROM users WHERE id = ?
            """, (user_id,))
            row = cur.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "name": row["name"],
                    "phone": row["phone"] or "",
                    "email": row["email"] or "",
                    "province": row["province"] or "",
                    "address": row["address"] or "",
                    "role": row["role"] or "citizen",
                    "created_at": row["created_at"]
                }
    except Exception:
        pass
    return None
