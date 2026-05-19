"""
get_google_refresh_token.py
---------------------------
Chay script nay de lay GOOGLE_REFRESH_TOKEN moi cho Google Drive.
Sau khi chay, copy gia tri in ra vao file .env cua flood_pipeline.

Yeu cau:
  pip install google-auth-oauthlib

Cach dung:
  1. Dat credentials.json cung thu muc voi script nay
     (tai tu Google Cloud Console -> Credentials -> OAuth 2.0 Client IDs -> Download JSON)
  2. python get_google_refresh_token.py
  3. Trinh duyet se mo ra, dang nhap Google va cho phep quyen Drive
  4. Copy 3 gia tri in ra vao .env
"""
import os
import json
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDS_FILE = Path(__file__).parent / "credentials.json"
TOKEN_FILE  = Path(__file__).parent / "token_new.json"


def main():
    if not CREDS_FILE.exists():
        print(f"[ERROR] Khong tim thay: {CREDS_FILE}")
        print()
        print("Cach tai credentials.json:")
        print("  1. Vao https://console.cloud.google.com")
        print("  2. Chon project -> APIs & Services -> Credentials")
        print("  3. Tao (hoac chon) OAuth 2.0 Client ID  ->  Desktop app")
        print("  4. Download JSON -> dat cung thu muc voi script nay")
        return

    from google_auth_oauthlib.flow import InstalledAppFlow

    print("Dang mo trinh duyet de xac thuc Google...")
    flow  = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)

    TOKEN_FILE.write_text(creds.to_json())

    print()
    print("=" * 60)
    print("COPY 3 DONG SAU VAO file .env cua flood_pipeline:")
    print("=" * 60)
    print(f"GOOGLE_CLIENT_ID={creds.client_id}")
    print(f"GOOGLE_CLIENT_SECRET={creds.client_secret}")
    print(f"GOOGLE_REFRESH_TOKEN={creds.refresh_token}")
    print("=" * 60)
    print(f"\n(Da luu backup token tai: {TOKEN_FILE})")


if __name__ == "__main__":
    main()
