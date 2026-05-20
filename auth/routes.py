# -*- coding: utf-8 -*-
"""
auth/routes.py
==============
Flask Blueprint cho tất cả endpoints liên quan đến auth + user management.

Đăng ký vào app.py:
    from auth.routes import auth_bp
    app.register_blueprint(auth_bp)

Endpoints:
    POST /api/auth/login
    POST /api/auth/logout
    POST /api/auth/register
    GET  /api/auth/me

    GET    /api/users              [admin]
    GET    /api/users/pending      [admin]
    POST   /api/users              [admin]  — thêm user trực tiếp
    POST   /api/users/approve      [admin]
    POST   /api/users/reject       [admin]
    PATCH  /api/users/<email>/role [admin]
    PATCH  /api/users/<email>/lock [admin]
"""

from __future__ import annotations

import os
import logging
from flask import Blueprint, jsonify, request, session

from auth.user_manager import UserManager, require_role, ROLES, ROLE_PERMISSIONS

log = logging.getLogger("auth.routes")

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")
users_bp = Blueprint("users", __name__, url_prefix="/api/users")

_um = UserManager(default_admin_pass=os.environ.get("ADMIN_PASS"))


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@auth_bp.route("/login", methods=["POST"])
def login():
    """
    POST /api/auth/login
    Body: { email, password }
    Returns: { name, role, permissions }
    """
    data = request.get_json(silent=True) or {}
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Thiếu email hoặc mật khẩu"}), 400

    user = _um.authenticate(email, password)
    if not user:
        return jsonify({"error": "Sai tài khoản, mật khẩu, hoặc tài khoản bị khóa"}), 401

    session["logged_in"] = True
    session["email"]     = user["email"]
    session["name"]      = user["name"]
    session["role"]      = user["role"]
    session["is_admin"]  = user["role"] == "admin"

    log.info(f"[Auth] Login: {email} ({user['role']})")

    return jsonify({
        "ok":          True,
        "name":        user["name"],
        "email":       user["email"],
        "role":        user["role"],
        "is_admin":    user["role"] == "admin",
        "permissions": ROLE_PERMISSIONS.get(user["role"], []),
    })


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """POST /api/auth/logout"""
    email = session.get("email", "unknown")
    session.clear()
    log.info(f"[Auth] Logout: {email}")
    return jsonify({"ok": True})


@auth_bp.route("/register", methods=["POST"])
def register():
    """
    POST /api/auth/register
    Body: { email, name, password, role }
    → Tạo tài khoản pending, chờ admin phê duyệt
    """
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name  = (data.get("name") or "").strip()
    pw    = data.get("password") or ""
    role  = data.get("role", "reviewer")

    if not email or not name or not pw:
        return jsonify({"error": "Thiếu thông tin bắt buộc"}), 400
    if len(pw) < 6:
        return jsonify({"error": "Mật khẩu tối thiểu 6 ký tự"}), 400

    result = _um.register(email, name, pw, role)
    if "error" in result:
        return jsonify(result), 409
    return jsonify(result), 201


@auth_bp.route("/me", methods=["GET"])
def me():
    """GET /api/auth/me — trả về info user đang đăng nhập"""
    if not session.get("logged_in"):
        return jsonify({"logged_in": False}), 401
    return jsonify({
        "logged_in":   True,
        "email":       session.get("email"),
        "name":        session.get("name"),
        "role":        session.get("role"),
        "is_admin":    session.get("is_admin", False),
        "permissions": ROLE_PERMISSIONS.get(session.get("role", ""), []),
    })


# ══════════════════════════════════════════════════════════════════════════════
# USER MANAGEMENT — chỉ admin
# ══════════════════════════════════════════════════════════════════════════════

@users_bp.route("", methods=["GET"])
@require_role("admin")
def list_users():
    """GET /api/users — danh sách tất cả user"""
    return jsonify(_um.list_users())


@users_bp.route("/pending", methods=["GET"])
@require_role("admin")
def list_pending():
    """GET /api/users/pending — danh sách tài khoản chờ duyệt"""
    return jsonify(_um.list_pending())


@users_bp.route("", methods=["POST"])
@require_role("admin")
def add_user():
    """
    POST /api/users — admin thêm user trực tiếp (không qua pending)
    Body: { email, name, password, role }
    """
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name  = (data.get("name") or "").strip()
    pw    = data.get("password", "changeme123")
    role  = data.get("role", "reviewer")

    if not email or not name:
        return jsonify({"error": "Thiếu email hoặc tên"}), 400

    result = _um.add_user(email, name, role, pw, actor=session["email"])
    if "error" in result:
        return jsonify(result), 409
    return jsonify(result), 201


@users_bp.route("/approve", methods=["POST"])
@require_role("admin")
def approve_user():
    """
    POST /api/users/approve
    Body: { email, role? }
    Admin có thể chỉnh role ngay lúc phê duyệt.
    Nếu không truyền role → giữ nguyên role user đã chọn lúc đăng ký.
    """
    data  = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    role  = data.get("role")  # optional
    if not email:
        return jsonify({"error": "Thiếu email"}), 400
    return jsonify(_um.approve(email, actor=session["email"], role=role))


@users_bp.route("/reject", methods=["POST"])
@require_role("admin")
def reject_user():
    """POST /api/users/reject  Body: { email }"""
    email = (request.get_json(silent=True) or {}).get("email", "").strip().lower()
    if not email:
        return jsonify({"error": "Thiếu email"}), 400
    return jsonify(_um.reject(email, actor=session["email"]))


@users_bp.route("/<email>/role", methods=["PATCH"])
@require_role("admin")
def change_role(email: str):
    """PATCH /api/users/<email>/role  Body: { role }"""
    new_role = (request.get_json(silent=True) or {}).get("role", "")
    if new_role not in ROLES:
        return jsonify({"error": f"Role không hợp lệ. Chọn: {ROLES}"}), 400
    # Không được tự hạ quyền của chính mình
    if email.lower() == session.get("email", "").lower() and new_role != "admin":
        return jsonify({"error": "Không thể tự đổi quyền của chính mình"}), 403
    return jsonify(_um.change_role(email, new_role, actor=session["email"]))


@users_bp.route("/<email>/lock", methods=["PATCH"])
@require_role("admin")
def toggle_lock(email: str):
    """PATCH /api/users/<email>/lock  Body: { active: true/false }"""
    if email.lower() == session.get("email", "").lower():
        return jsonify({"error": "Không thể tự khóa chính mình"}), 403
    active = (request.get_json(silent=True) or {}).get("active", False)
    return jsonify(_um.set_active(email, active, actor=session["email"]))


@users_bp.route("/auth-log", methods=["GET"])
@require_role("admin")
def auth_log():
    """GET /api/users/auth-log — audit log cho user actions"""
    n = int(request.args.get("n", 50))
    return jsonify(_um.get_auth_log(n))
