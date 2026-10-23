"""
drive_store.py — Google Drive upload for the SRT translator.

Uploads finished translations to a `subtitles/` folder on Google Drive as
`<reel_name>_translated.srt`, so the output survives Streamlit Cloud session
wipes.

Account-agnostic: reads a personal Google OAuth token (refresh_token +
client_id + client_secret) and calls the Drive REST API directly with
`requests`, refreshing its own access token. No google-auth dependency.

Token shapes accepted:
    refresh_token, token_uri, client_id, client_secret   (google-auth format)
    refresh_token, client_id, client_secret, access_token (raw OAuth response)
"""

from __future__ import annotations

import json
import re
import time

import requests

DRIVE_API = "https://www.googleapis.com/drive/v3"
UPLOAD_API = "https://www.googleapis.com/upload/drive/v3"
TOKEN_URI = "https://oauth2.googleapis.com/token"
FOLDER_MIME = "application/vnd.google-apps.folder"
SRT_MIME = "text/plain"


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name or "srt")


class DriveStore:
    """Upload finished SRTs to a single Drive folder."""

    def __init__(self, token_data: dict, folder_name: str = "subtitles"):
        self.td = token_data or {}
        self.folder_name = folder_name
        self._folder_id = None
        self._access = None
        self._expiry = 0.0

    # ------------------------------------------------------------- auth

    @property
    def available(self) -> bool:
        return bool(self.td.get("refresh_token"))

    def _refresh(self) -> str:
        r = requests.post(
            self.td.get("token_uri") or TOKEN_URI,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.td["refresh_token"],
                "client_id": self.td["client_id"],
                "client_secret": self.td["client_secret"],
            },
            timeout=30,
        )
        r.raise_for_status()
        tok = r.json()
        self._access = tok["access_token"]
        self._expiry = time.time() + int(tok.get("expires_in", 3600)) - 60
        return self._access

    def _token(self) -> str:
        if self._access and time.time() < self._expiry:
            return self._access
        return self._refresh()

    def _auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self._token()}"}

    def _request(self, method: str, url: str, retry: bool = True, **kw):
        headers = kw.pop("headers", {})
        headers.update(self._auth_header())
        r = requests.request(method, url, timeout=60, headers=headers, **kw)
        if r.status_code == 401 and retry:
            self._refresh()
            headers["Authorization"] = f"Bearer {self._access}"
            r = requests.request(method, url, timeout=60, headers=headers, **kw)
        return r

    # ------------------------------------------------------------- folder

    def _ensure_folder(self) -> str:
        if self._folder_id:
            return self._folder_id
        q = (f"name='{self.folder_name}' and mimeType='{FOLDER_MIME}' "
             f"and trashed=false")
        r = self._request("GET", f"{DRIVE_API}/files",
                          params={"q": q, "fields": "files(id,name)"})
        r.raise_for_status()
        files = r.json().get("files", [])
        if files:
            self._folder_id = files[0]["id"]
        else:
            r = self._request(
                "POST", f"{DRIVE_API}/files",
                json={"name": self.folder_name, "mimeType": FOLDER_MIME})
            r.raise_for_status()
            self._folder_id = r.json()["id"]
        return self._folder_id

    def _find_file(self, name: str) -> str | None:
        fid = self._ensure_folder()
        q = f"name='{name}' and '{fid}' in parents and trashed=false"
        r = self._request("GET", f"{DRIVE_API}/files",
                          params={"q": q, "fields": "files(id,name)"})
        r.raise_for_status()
        files = r.json().get("files", [])
        return files[0]["id"] if files else None

    # ------------------------------------------------------------- primitives

    def upload(self, name: str, content: str, mime: str = SRT_MIME) -> str:
        """Create (or replace) a file by name inside the folder."""
        self._ensure_folder()
        existing = self._find_file(name)
        if existing:
            self._request("DELETE", f"{DRIVE_API}/files/{existing}")

        meta = json.dumps({"name": name, "parents": [self._folder_id],
                           "mimeType": mime})
        boundary = "----srtboundary"
        body = (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{meta}\r\n--{boundary}\r\nContent-Type: {mime}\r\n\r\n"
        ).encode("utf-8") + content.encode("utf-8") + f"\r\n--{boundary}--\r\n".encode("utf-8")

        r = self._request(
            "POST", f"{UPLOAD_API}/files?uploadType=multipart",
            data=body,
            headers={"Content-Type": f"multipart/related; boundary={boundary}"})
        r.raise_for_status()
        return r.json()["id"]

    def upload_translated(self, srt_content: str, out_name: str) -> str:
        """Upload the finished SRT as `<reel>_translated.srt`."""
        return self.upload(out_name, srt_content, SRT_MIME)


def get_drive_store(token_data) -> DriveStore | None:
    """Return a DriveStore if the token looks usable, else None."""
    try:
        if isinstance(token_data, str):
            token_data = json.loads(token_data)
        store = DriveStore(token_data)
        return store if store.available else None
    except Exception:
        return None
