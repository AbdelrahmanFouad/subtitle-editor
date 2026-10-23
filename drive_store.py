"""
drive_store.py — Google Drive persistence for the SRT translator.

Account-agnostic: it reads a personal Google OAuth token (refresh_token +
client_id + client_secret) and mirrors checkpoints to a Drive folder, so
Streamlit Cloud sessions (which wipe their local filesystem) can still recover
work.

No google-auth dependency — it calls the Drive REST API directly with
`requests`, refreshing the access token itself. This sidesteps the known
`~/.venv` gotcha where google-auth is absent.

Token shapes accepted (whichever key names the flow produced):
    refresh_token, token_uri, client_id, client_secret   (google-auth format)
    refresh_token, client_id, client_secret, access_token (raw OAuth response)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests

DRIVE_API = "https://www.googleapis.com/drive/v3"
UPLOAD_API = "https://www.googleapis.com/upload/drive/v3"
TOKEN_URI = "https://oauth2.googleapis.com/token"
FOLDER_MIME = "application/vnd.google-apps.folder"
JSON_MIME = "application/json"
SRT_MIME = "text/plain"


@dataclass
class Checkpoint:
    ref: str          # "drive:<id>" or "local:<path>" — opaque to the UI
    source: str
    saved: str
    blocks: int
    label: str


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name or "srt")


class DriveStore:
    """Mirror checkpoints to a single Drive folder."""

    def __init__(self, token_data: dict, folder_name: str = "srt-translator-checkpoints"):
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

    def upload(self, name: str, content: str, mime: str = JSON_MIME) -> str:
        """Create (or replace) a file by name inside the checkpoint folder."""
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

    def list_files(self) -> list[dict]:
        fid = self._ensure_folder()
        r = self._request(
            "GET", f"{DRIVE_API}/files",
            params={"q": f"'{fid}' in parents and trashed=false",
                    "fields": "files(id,name,modifiedTime)",
                    "orderBy": "modifiedTime desc", "pageSize": 200})
        r.raise_for_status()
        return r.json().get("files", [])

    def download(self, file_id: str) -> str:
        r = self._request("GET", f"{DRIVE_API}/files/{file_id}",
                          params={"alt": "media"})
        r.raise_for_status()
        return r.text

    def delete(self, file_id: str) -> None:
        self._request("DELETE", f"{DRIVE_API}/files/{file_id}")

    # ------------------------------------------------------------- checkpoints

    def save_checkpoint(self, subs, source_name: str, tag: str = "auto") -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        payload = {"source": source_name, "tag": tag, "saved": ts,
                   "blocks": len(subs), "subs": subs}
        content = json.dumps(payload, ensure_ascii=False)
        safe = _safe_name(source_name)
        # latest.json (overwritten) + one history .json + one portable .srt
        self.upload("latest.json", content, JSON_MIME)
        self.upload(f"{ts}_{safe}.json", content, JSON_MIME)
        srt = self._build_srt(subs)
        self.upload(f"{ts}_{safe}.srt", srt, SRT_MIME)
        self.prune(keep=50)
        return ts

    @staticmethod
    def _build_srt(subs) -> str:
        chunks = []
        for s in subs:
            body = (f"{s['arabic']}\n" if s.get("arabic") else "") + "\n".join(
                s.get("english_lines", []))
            chunks.append(f"{s['index']}\n{s['start']} --> {s['end']}\n{body}")
        return "\n\n".join(chunks)

    def latest_meta(self) -> dict | None:
        try:
            fid = self._find_file("latest.json")
            if not fid:
                return None
            return json.loads(self.download(fid))
        except Exception:
            return None

    def list_checkpoints(self) -> list[Checkpoint]:
        out = []
        for f in self.list_files():
            name = f["name"]
            if name == "latest.json" or not name.endswith(".json"):
                continue
            m = re.match(r"(\d{8}_\d{6})_(.*)\.json$", name)
            if not m:
                continue
            ts, src = m.group(1), m.group(2)
            out.append(Checkpoint(
                ref=f"drive:{f['id']}", source=src, saved=ts, blocks=0,
                label=f"{ts} · {src}"))
        return out

    def load_checkpoint(self, ref: str):
        file_id = ref.removeprefix("drive:")
        data = json.loads(self.download(file_id))
        return data["subs"], data.get("source", ""), data.get("saved", "")

    def load_latest(self):
        data = self.latest_meta()
        if not data:
            return None
        return data["subs"], data.get("source", ""), data.get("saved", "")

    def prune(self, keep: int = 50) -> None:
        """Drop oldest history .json/.srt beyond `keep` (never latest.json)."""
        files = self.list_files()
        hist = [f for f in files if re.match(r"\d{8}_\d{6}_", f["name"])]
        for f in hist[keep:]:
            try:
                self.delete(f["id"])
            except Exception:
                pass


def get_drive_store(token_data) -> DriveStore | None:
    """Return a DriveStore if the token looks usable, else None."""
    try:
        if isinstance(token_data, str):
            token_data = json.loads(token_data)
        store = DriveStore(token_data)
        return store if store.available else None
    except Exception:
        return None
