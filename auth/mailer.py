# -*- coding: utf-8 -*-
"""
auth/mailer.py
==============
Gửi email OTP qua Gmail SMTP.

Dùng:
    from auth.mailer import send_otp, verify_otp

    ok, code = send_otp("user@gmail.com")   # gửi mail, trả về True + code đã gửi
    ok       = verify_otp("user@gmail.com", "123456")  # True nếu đúng và còn hạn
"""

from __future__ import annotations

import logging
import os
import random
import smtplib
import string
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Optional, Tuple

log = logging.getLogger("auth.mailer")

# OTP lưu tạm trong memory: { email: {code, expires_at, attempts} }
_otp_store: Dict[str, Dict] = {}

OTP_TTL_MINUTES = 10
OTP_MAX_ATTEMPTS = 5


# ── Generate & send ───────────────────────────────────────────────────────────

def send_otp(email: str) -> Tuple[bool, str]:
    """
    Sinh OTP 6 số, gửi qua Gmail, lưu vào store.
    Returns: (success, message)
    """
    code = _generate_code()
    _otp_store[email.lower()] = {
        "code":       code,
        "expires_at": datetime.now() + timedelta(minutes=OTP_TTL_MINUTES),
        "attempts":   0,
    }

    mail_from  = os.environ.get("MAIL_FROM", "")
    app_pass   = os.environ.get("MAIL_APP_PASS", "")

    if not mail_from or not app_pass or app_pass == "your-app-password-here":
        # Dev mode — log code ra console thay vì gửi mail
        log.warning(f"[Mailer] DEV MODE — OTP cho {email}: {code}")
        return True, f"[DEV] OTP: {code} (chưa config MAIL_APP_PASS)"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[FloodWatch] Mã xác nhận đăng ký: {code}"
        msg["From"]    = f"FloodWatch <{mail_from}>"
        msg["To"]      = email

        html_body = _email_template(code)
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp:
            smtp.login(mail_from, app_pass)
            smtp.sendmail(mail_from, email, msg.as_string())

        log.info(f"[Mailer] Đã gửi OTP đến {email}")
        return True, "Đã gửi mã xác nhận về email"

    except smtplib.SMTPAuthenticationError:
        log.error("[Mailer] Xác thực Gmail thất bại — kiểm tra MAIL_APP_PASS")
        return False, "Lỗi xác thực Gmail — kiểm tra App Password trong .env"
    except Exception as e:
        log.error(f"[Mailer] Lỗi gửi mail: {e}")
        return False, f"Không gửi được email: {e}"


def verify_otp(email: str, code: str) -> Tuple[bool, str]:
    """
    Xác minh OTP.
    Returns: (valid, message)
    """
    key  = email.lower()
    data = _otp_store.get(key)

    if not data:
        return False, "Mã xác nhận không tồn tại hoặc đã hết hạn"

    if datetime.now() > data["expires_at"]:
        _otp_store.pop(key, None)
        return False, f"Mã đã hết hạn (hạn {OTP_TTL_MINUTES} phút) — vui lòng đăng ký lại"

    data["attempts"] += 1
    if data["attempts"] > OTP_MAX_ATTEMPTS:
        _otp_store.pop(key, None)
        return False, "Nhập sai quá nhiều lần — vui lòng đăng ký lại"

    if data["code"] != code.strip():
        remaining = OTP_MAX_ATTEMPTS - data["attempts"]
        return False, f"Mã không đúng — còn {remaining} lần thử"

    # Đúng → xóa khỏi store
    _otp_store.pop(key, None)
    return True, "Xác nhận thành công"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_code(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


def _email_template(code: str) -> str:
    return f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#f4f4f0;font-family:-apple-system,sans-serif">
<div style="max-width:420px;margin:40px auto;background:#fff;border-radius:12px;
     border:1px solid #e8e6e0;overflow:hidden">
  <div style="background:#1a1a1a;padding:20px 28px;display:flex;align-items:center;gap:10px">
    <span style="font-size:18px">🌊</span>
    <span style="color:#fff;font-size:15px;font-weight:500">FloodWatch</span>
  </div>
  <div style="padding:28px">
    <p style="margin:0 0 8px;font-size:15px;font-weight:500;color:#1a1a1a">
      Mã xác nhận đăng ký
    </p>
    <p style="margin:0 0 24px;font-size:13px;color:#6b6b6b;line-height:1.6">
      Nhập mã dưới đây để hoàn tất đăng ký tài khoản FloodWatch Admin.
      Mã có hiệu lực trong <strong>{OTP_TTL_MINUTES} phút</strong>.
    </p>
    <div style="background:#f4f4f0;border-radius:8px;padding:18px;text-align:center;
         letter-spacing:8px;font-size:28px;font-weight:700;color:#1a1a1a;
         font-family:monospace;margin-bottom:20px">
      {code}
    </div>
    <p style="margin:0;font-size:12px;color:#999;line-height:1.5">
      Nếu bạn không yêu cầu đăng ký, hãy bỏ qua email này.<br>
      Không chia sẻ mã này với bất kỳ ai.
    </p>
  </div>
</div>
</body>
</html>
"""
