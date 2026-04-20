# -*- coding: utf-8 -*-
"""
uploader/drive_uploader.py
--------------------------
Upload + Download tu Google Drive API v3.

Tinh nang moi:
  - download_folder(): tai anh tu Drive folder ve local
  - upload_folder(): co them tuy chon delete_local_after_upload=True
  - get_folder_id_from_url(): lay folder ID tu URL Google Drive
"""

import logging
import mimetypes
import os
import shutil
import time
from pathlib import Path
from typing import Optional, List
from utils.constants import IMAGE_EXTENSIONS, DOCUMENT_EXTENSIONS, DRIVE_UPLOAD_EXTENSIONS

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/drive",   # full access (can download + upload)
]


class DriveUploader:

    def __init__(
        self,
        credentials_path: str = "credentials.json",
        token_path:        str = "token.json",
        use_service_account: bool = False,
    ):
        self.credentials_path    = credentials_path
        self.token_path          = token_path
        self.use_service_account = use_service_account
        self._service            = None

    # ==========================================================================
    # AUTH
    # ==========================================================================
    def _get_service(self):
        if self._service:
            return self._service
        from googleapiclient.discovery import build
        self._service = build("drive", "v3", credentials=self._load_credentials())
        log.info("  Google Drive API connected")
        return self._service

    def _load_credentials(self):
        sa_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        if self.use_service_account or (sa_file and Path(sa_file).exists()):
            from google.oauth2 import service_account
            return service_account.Credentials.from_service_account_file(
                sa_file or "service-account.json", scopes=SCOPES
            )

        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request

        creds = None
        if Path(self.token_path).exists():
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        if not creds or not creds.valid:
            if not Path(self.credentials_path).exists():
                raise FileNotFoundError(
                    f"Khong tim thay {self.credentials_path}.\n"
                    "  1. Vao https://console.cloud.google.com\n"
                    "  2. Enable Google Drive API\n"
                    "  3. Tao OAuth 2.0 Client ID (Desktop app)\n"
                    "  4. Download -> dat vao thu muc project lam 'credentials.json'"
                )
            creds = InstalledAppFlow.from_client_secrets_file(
                self.credentials_path, SCOPES
            ).run_local_server(port=0)
            Path(self.token_path).write_text(creds.to_json())
        return creds

    # ==========================================================================
    # UTILITY
    # ==========================================================================
    @staticmethod
    def get_folder_id_from_url(url_or_id: str) -> str:
        """
        Lay folder ID tu nhieu dang input:
          - ID thuan: "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"
          - URL day du: "https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"
          - URL co tham so: "https://drive.google.com/drive/u/0/folders/1BxiMVs0..."
        """
        import re
        # Thu parse URL
        m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url_or_id)
        if m:
            return m.group(1)
        # Neu la ID thuan (khong co slash, khong phai URL)
        if "/" not in url_or_id and "đ" not in url_or_id:
            return url_or_id.strip()
        raise ValueError(
            f"Khong the phan tich Drive folder ID tu: {url_or_id}\n"
            "  Hay dung dang: https://drive.google.com/drive/folders/FOLDER_ID"
        )

    # ==========================================================================
    # FOLDER MANAGEMENT
    # ==========================================================================
    def create_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        """Tao folder (ho tro nested path 'Parent/Child'). Tra ve folder ID."""
        service = self._get_service()
        parts   = [p for p in name.replace("\\", "/").split("/") if p]
        current: Optional[str] = parent_id
        for part in parts:
            current = self._get_or_create_folder(service, part, current)
        return current or ""

    def _get_or_create_folder(self, service, name: str, parent_id: Optional[str]) -> str:
        q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            q += f" and '{parent_id}' in parents"
        resp  = service.files().list(q=q, spaces="drive", fields="files(id)").execute()
        files = resp.get("files", [])
        if files:
            return files[0]["id"]
        meta: dict = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent_id:
            meta["parents"] = [parent_id]
        f = service.files().create(body=meta, fields="id").execute()
        log.info(f"  Created Drive folder: '{name}'")
        return f["id"]

    def list_files(
        self,
        folder_id: str,
        extensions: Optional[set] = None,
        recursive: bool = False,
    ) -> List[dict]:
        """
        Liet ke file trong Drive folder.
        Tra ve list dict: {id, name, mimeType, size}
        """
        service = self._get_service()
        results = []

        q = f"'{folder_id}' in parents and trashed=false"
        page_token = None

        while True:
            resp = service.files().list(
                q=q,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageToken=page_token,
                pageSize=100,
            ).execute()

            for f in resp.get("files", []):
                mime = f.get("mimeType", "")
                if mime == "application/vnd.google-apps.folder":
                    if recursive:
                        results.extend(
                            self.list_files(f["id"], extensions, recursive)
                        )
                else:
                    if extensions:
                        ext = "." + f["name"].rsplit(".", 1)[-1].lower() if "." in f["name"] else ""
                        if ext not in extensions:
                            continue
                    results.append(f)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        return results

    # ==========================================================================
    # DOWNLOAD FROM DRIVE
    # ==========================================================================
    def download_folder(
        self,
        drive_folder: str,
        local_dir: Path,
        extensions: Optional[set] = None,
        max_files: Optional[int] = None,
    ) -> List[Path]:
        """
        Tai tat ca anh tu Google Drive folder ve local.

        Args:
            drive_folder: Folder ID hoac URL Google Drive
                          VD: "https://drive.google.com/drive/folders/1BxiM..."
                          hoac chi "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"
            local_dir:    Thu muc local de luu anh
            extensions:   Bo loc duoi file. Mac dinh: {.jpg, .jpeg, .png, .webp}
            max_files:    Gioi han so file tai ve

        Returns:
            List duong dan file da tai ve
        """
        from googleapiclient.http import MediaIoBaseDownload
        import io

        folder_id = self.get_folder_id_from_url(drive_folder)
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        if extensions is None:
            extensions = IMAGE_EXTENSIONS

        log.info(f"  Listing files in Drive folder: {folder_id}")
        files = self.list_files(folder_id, extensions=extensions, recursive=True)

        if max_files:
            files = files[:max_files]

        log.info(f"  Found {len(files)} files to download")
        service   = self._get_service()
        saved     = []
        errors    = 0

        for i, f in enumerate(files, 1):
            dest = local_dir / f["name"]
            # Skip neu da co file nay roi
            if dest.exists():
                log.debug(f"  Skip (exists): {f['name']}")
                saved.append(dest)
                continue

            try:
                request = service.files().get_media(fileId=f["id"])
                buf     = io.BytesIO()
                dl      = MediaIoBaseDownload(buf, request)

                done = False
                while not done:
                    _, done = dl.next_chunk()

                dest.write_bytes(buf.getvalue())
                saved.append(dest)

                if i % 10 == 0 or i == len(files):
                    log.info(f"  Downloaded {i}/{len(files)}: {f['name']}")

            except Exception as e:
                log.warning(f"  Download failed {f['name']}: {e}")
                errors += 1

        log.info(f"  Download complete: {len(saved)} files, {errors} errors -> {local_dir}")
        return saved

    # ==========================================================================
    # UPLOAD TO DRIVE
    # ==========================================================================
    def upload_file(
        self,
        local_path: Path,
        folder_id:  Optional[str] = None,
        filename:   Optional[str] = None,
    ) -> str:
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.errors import HttpError
        import requests

        service   = self._get_service()
        local_path= Path(local_path)
        fname     = filename or local_path.name
        mime_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        meta: dict = {"name": fname}
        if folder_id:
            meta["parents"] = [folder_id]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                media  = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
                result = service.files().create(body=meta, media_body=media, fields="id").execute()
                return result["id"]
            except (HttpError, requests.exceptions.RequestException, TimeoutError, ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    log.warning(f"  Upload failed (attempt {attempt+1}/{max_retries}) {fname}: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    log.error(f"  Upload failed after {max_retries} attempts {fname}: {e}")
                    raise
        raise RuntimeError(f"Upload failed: {fname}")

    def upload_folder(
        self,
        local_dir:   Path,
        folder_name: str,
        parent_id:   Optional[str] = None,
        extensions:  Optional[set] = None,
        delete_local_after_upload: bool = False,   # <-- MOI: xoa local sau khi upload
    ) -> str:
        """
        Upload toan bo thu muc len Drive.

        Args:
            delete_local_after_upload: Neu True, xoa toan bo local_dir
                                       SAU KHI upload thanh cong hoan toan.
                                       Chi xoa khi 0 loi.
        Returns:
            Drive folder ID
        """
        local_dir = Path(local_dir)
        folder_id = self.create_folder(folder_name, parent_id)

        if extensions is None:
            extensions = DRIVE_UPLOAD_EXTENSIONS

        # Thu thap tat ca file can upload
        files_to_upload = [
            f for f in sorted(local_dir.rglob("*"))
            if f.is_file() and f.suffix.lower() in extensions
        ]

        log.info(f"  Uploading {len(files_to_upload)} files to Drive ...")
        uploaded = 0
        errors   = 0

        # Cache subfolder IDs de tranh tao trung
        subfolder_cache: dict = {}

        for item in files_to_upload:
            rel     = item.relative_to(local_dir)
            parents = rel.parts[:-1]

            # Xac dinh folder dich tren Drive
            target_folder = folder_id
            if parents:
                sub_key = "/".join(parents)
                if sub_key not in subfolder_cache:
                    subfolder_cache[sub_key] = self.create_folder(sub_key, folder_id)
                target_folder = subfolder_cache[sub_key]

            try:
                self.upload_file(item, folder_id=target_folder)
                uploaded += 1
                if uploaded % 10 == 0:
                    log.info(f"  Uploaded {uploaded}/{len(files_to_upload)} ...")
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                log.warning(f"  Upload failed {item.name}: {e}")
                errors += 1

        log.info(
            f"  Upload complete: {uploaded} files, {errors} errors\n"
            f"  Drive folder: https://drive.google.com/drive/folders/{folder_id}"
        )

        # Xoa local chi khi upload thanh cong hoan toan (0 loi)
        if delete_local_after_upload:
            if errors == 0:
                log.info(f"  Deleting local folder: {local_dir}")
                try:
                    shutil.rmtree(str(local_dir))
                    log.info(f"  Deleted: {local_dir}")
                except Exception as e:
                    log.warning(f"  Could not delete local folder: {e}")
            else:
                log.warning(
                    f"  Skipped local delete: {errors} upload error(s) detected.\n"
                    f"  Local data is KEPT safe at: {local_dir}"
                )

        return folder_id
