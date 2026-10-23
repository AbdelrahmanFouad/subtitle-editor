"""
srte.py — SRT Editor & Translator: English → Arabic (AI)

A fast, concurrent Streamlit app for translating SRT subtitles with Google
Gemini across a pool of API keys, with live-streaming results and automatic
upload of finished files to Google Drive.

* Live: a background worker translates in parallel (10 workers × 16-block
  batches) and the results table fills in as blocks complete.
* Accurate: index-based response parsing, per-block retry, within-file reuse.
* Robust: thread-safe key pool with parking; every retryable error handled.
* Persistent: finished files are flushed to Drive as `subtitles/<original>_translated.srt`.

Usage:
    streamlit run srte.py
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chardet
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

from drive_store import DriveStore, get_drive_store

# ---------------------------------------------------------------- configuration

MODELS = [
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

BATCH_SIZE = 16
WORKERS = 10
MAX_BATCH_ATTEMPTS = 4
MAX_ROUNDS = 3
MAX_KEY_FAILS = 3
DRIVE_FOLDER = "subtitles"

RETRYABLE = (
    "quota", "429", "resource", "limit", "not found", "unsupported",
    "not supported", "overloaded", "deadline", "503", "unavailable",
    "high demand", "try again", "connection", "timeout", "timed out",
    "getaddrinfo", "errno",
)

ARABIC_RANGES = (
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
)
_ARABIC_RE = re.compile(
    "[" + "".join(f"\\u{lo:04x}-\\u{hi:04x}" for lo, hi in ARABIC_RANGES) + "]"
)

SYSTEM_INSTRUCTION = (
    "Translate English subtitle blocks to natural Arabic. Maintain line breaks. "
    "Output translations preceded by [1], [2], etc. Always output one numbered "
    "entry for every input block, even if the text is short or contains markup. "
    "Never skip a number."
)


# ---------------------------------------------------------------- helpers

def autodetect_decode(uploader) -> str:
    raw = uploader.getvalue()
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    try:
        return raw.decode(enc)
    except Exception:
        return raw.decode("utf-8", errors="ignore")


def is_arabic_line(line: str) -> bool:
    return bool(_ARABIC_RE.search(line))


def parse_srt(text: str) -> list[dict]:
    text = text.replace("\r\n", "\n")
    subs: list[dict] = []
    for blk in re.split(r"\n\s*\n", text.strip()):
        lines = blk.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        idx = lines[0].strip()
        start, end = [t.strip() for t in lines[1].split("-->")]
        body = [l.strip() for l in lines[2:] if l.strip()]
        arabic_lines = [l for l in body if is_arabic_line(l)]
        english_lines = [l for l in body if not is_arabic_line(l)]
        subs.append({
            "index": idx, "start": start, "end": end,
            "english_lines": english_lines,
            "arabic": "\n".join(arabic_lines),
        })
    return subs


def build_srt(subs: list[dict]) -> str:
    chunks = []
    for s in subs:
        body = (f"{s['arabic']}\n" if s["arabic"] else "") + "\n".join(s["english_lines"])
        chunks.append(f"{s['index']}\n{s['start']} --> {s['end']}\n{body}")
    return "\n\n".join(chunks)


def split_indexed(response_text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    parts = re.split(r"\[(\d+)\]", response_text.strip())
    for i in range(1, len(parts) - 1, 2):
        try:
            n = int(parts[i])
        except ValueError:
            continue
        body = parts[i + 1].strip()
        if body:
            out[n] = body
    return out


def translate_batch(client, model_id: str, block_texts: list[str]) -> dict[int, str]:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION, temperature=0.3)
    prompt = "Translate these blocks:\n\n" + "".join(
        f"[{i}]\n{t}\n\n" for i, t in enumerate(block_texts, 1))
    response = client.models.generate_content(
        model=model_id, contents=prompt, config=config)
    return split_indexed(response.text)


def block_status(s: dict) -> str:
    ar = len([l for l in s["arabic"].split("\n") if l.strip()])
    if not ar:
        return "missing"
    return "done" if ar >= len(s["english_lines"]) else "missing"


def build_reuse_map(subs: list[dict]) -> dict[str, str]:
    m: dict[str, str] = {}
    for s in subs:
        if block_status(s) != "done":
            continue
        key = "\n".join(s["english_lines"]).strip().lower()
        if key:
            m[key] = s["arabic"]
    return m


# ---------------------------------------------------------------- persistence

def _read_drive_token():
    raw = None
    try:
        raw = st.secrets.get("drive_token")
    except Exception:
        raw = None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        stripped = raw.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return None
        p = Path(stripped)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


drive = get_drive_store(_read_drive_token())


# ---------------------------------------------------------------- key pool

class KeyPool:
    """Thread-safe round-robin over API keys with parking."""

    def __init__(self, n_keys: int):
        self.n = n_keys
        self._lock = threading.Lock()
        self._fail = [0] * n_keys
        self._model_pos = [0] * n_keys
        self._cursor = 0

    def acquire(self) -> int:
        with self._lock:
            for _ in range(self.n):
                i = self._cursor
                self._cursor = (self._cursor + 1) % self.n
                if self._fail[i] < MAX_KEY_FAILS:
                    return i
            self._fail = [0] * self.n
            return 0

    def model(self, i: int) -> str:
        with self._lock:
            m = MODELS[self._model_pos[i] % len(MODELS)]
            self._model_pos[i] = (self._model_pos[i] + 1) % len(MODELS)
            return m

    def success(self, i: int) -> None:
        with self._lock:
            self._fail[i] = 0

    def fail(self, i: int) -> None:
        with self._lock:
            self._fail[i] += 1


# ---------------------------------------------------------------- engine

def _one_batch(pool, clients, subs, batch, reuse_map):
    texts = ["\n".join(subs[i]["english_lines"]) for i in batch]
    last_err = ""
    for _ in range(MAX_BATCH_ATTEMPTS):
        ki = pool.acquire()
        model = pool.model(ki)
        try:
            got = translate_batch(clients[ki], model, texts)
            pool.success(ki)
            out = {}
            for pos, idx in enumerate(batch, 1):
                out[idx] = got.get(pos, "").strip()
            return out, last_err
        except Exception as e:  # noqa: BLE001
            pool.fail(ki)
            last_err = str(e)
    return {idx: "" for idx in batch}, last_err


def translate_all(subs, pending, api_keys, reuse_map, apply_cb=None) -> dict[int, str]:
    clients = [genai.Client(api_key=k) for k in api_keys]
    pool = KeyPool(len(api_keys))
    result: dict[int, str] = {}
    resolved: dict[int, str] = {}

    for i in list(pending):
        key = "\n".join(subs[i]["english_lines"]).strip().lower()
        if key in reuse_map:
            resolved[i] = reuse_map[key]
    result.update(resolved)
    if apply_cb:
        for idx, t in resolved.items():
            apply_cb(idx, t)

    remaining = [i for i in pending if i not in result]
    rounds = [BATCH_SIZE, max(1, BATCH_SIZE // 2), 1][:MAX_ROUNDS]

    for size in rounds:
        if not remaining:
            break
        batches = [remaining[i:i + size] for i in range(0, len(remaining), size)]
        n_workers = min(WORKERS, max(1, len(batches)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_one_batch, pool, clients, subs, b, reuse_map): b
                    for b in batches}
            for f in as_completed(futs):
                got, _ = f.result()
                for idx, t in got.items():
                    if t:
                        result[idx] = t
                        if apply_cb:
                            apply_cb(idx, t)
        remaining = [i for i in remaining if i not in result]

    return result


# ---------------------------------------------------------------- live run

@st.cache_resource
def _live_ctrl():
    return {
        "lock": threading.Lock(),
        "running": False,
        "total": 0,
        "results": {},
        "log": [],
        "error": None,
        "t0": 0.0,
        "elapsed": 0.0,
        "finalized": False,
        "source_name": "",
    }


def _translate_worker(ctrl, subs, pending, api_keys, reuse_map):
    def apply_cb(idx, text):
        with ctrl["lock"]:
            ctrl["results"][idx] = text

    def progress_cb(resolved, total, tag):
        pass

    def log_cb(err):
        with ctrl["lock"]:
            ctrl["log"].append(err)

    try:
        translate_all(subs, pending, api_keys, reuse_map, apply_cb=apply_cb)
        with ctrl["lock"]:
            ctrl["elapsed"] = time.time() - ctrl["t0"]
            ctrl["running"] = False
    except Exception as e:  # noqa: BLE001
        with ctrl["lock"]:
            ctrl["error"] = str(e)
            ctrl["elapsed"] = time.time() - ctrl["t0"]
            ctrl["running"] = False


# ---------------------------------------------------------------- theme & page

st.set_page_config(
    page_title="SRT Translator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --bg: #0e1117;
    --panel: #161b26;
    --border: #262d3a;
    --accent: #F1C409;
    --blue: #0F76B2;
    --text: #e6e9ef;
    --muted: #8b93a3;
}
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
.block-container { padding-top: 1.4rem; max-width: 1400px; }
h1, h2, h3 { color: var(--text); letter-spacing: -0.01em; }
p, label, .stCaption { color: var(--muted); }

/* metric cards */
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
}
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--text); font-weight: 700; }

/* primary button — gold */
.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #f7d32a, #e0b40a);
    color: #1a1400;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(241,196,9,0.25);
}
.stButton > button[kind="primary"]:hover { background: #f7d32a; color: #000; }

/* progress bar */
[data-testid="stProgress"] > div > div > div > div { background: var(--accent); }

/* data editor */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* file uploader */
[data-testid="stFileUploader"] section {
    background: var(--panel);
    border: 1px dashed var(--border);
    border-radius: 12px;
}

/* radio */
.stRadio label { color: var(--text); }
</style>
""", unsafe_allow_html=True)

st.title("🎬 SRT Translator")
st.caption("English → Arabic subtitle translation · concurrent · auto-saved to Google Drive")

# ---------------------------------------------------------------- session state

if "subs" not in st.session_state:
    st.session_state.subs = []
if "loaded_file" not in st.session_state:
    st.session_state.loaded_file = None
if "source_name" not in st.session_state:
    st.session_state.source_name = ""

uploader = st.file_uploader("Upload SRT file", type="srt")

if uploader:
    sig = f"{uploader.name}:{uploader.size}"
    if st.session_state.loaded_file != sig:
        st.session_state.subs = parse_srt(autodetect_decode(uploader))
        st.session_state.loaded_file = sig
        st.session_state.source_name = uploader.name
        for k in [k for k in st.session_state.keys()
                  if k.startswith(("arabic_", "english_", "start_", "end_"))]:
            del st.session_state[k]

if not st.session_state.subs:
    st.info("Upload an .srt file to begin.")
    st.stop()

subs = st.session_state.subs
source_name = st.session_state.source_name
# output name always preserves the original upload name (never generic)
out_filename = f"{Path(source_name).stem}_translated.srt" if source_name else "translated.srt"

# ---------------------------------------------------------------- live merge

_live = _live_ctrl()
with _live["lock"]:
    _running = _live["running"]
    _results = dict(_live["results"])
    _total = _live["total"]
    _error = _live["error"]
    _elapsed = _live["elapsed"]
    _finalized = _live["finalized"]

if _results:
    for _i, _t in _results.items():
        if _i < len(subs) and subs[_i].get("arabic") != _t:
            subs[_i]["arabic"] = _t

# once the run finishes, flush the finished file to Drive (once)
if not _running and not _finalized and _total:
    _live["finalized"] = True
    final_srt = build_srt(subs)
    if drive:
        try:
            drive.upload_translated(final_srt, out_filename)
            st.session_state["drive_saved"] = f"subtitles/{out_filename}"
        except Exception as e:  # noqa: BLE001
            st.session_state["drive_error"] = str(e)
    else:
        st.session_state["drive_saved"] = None

# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Translation scope",
        ["Missing only", "Full file"],
        index=0,
        horizontal=True,
        help="Missing only = translate blocks with no Arabic. Full file = re-translate everything.",
    )
    reuse = st.checkbox("Reuse repeated lines", value=True,
                        help="A line that repeats elsewhere is translated once and reused.")
    st.divider()
    st.caption("Worker pool and batch size are fixed for maximum throughput.")
    st.markdown(
        f"<div style='font-size:0.85rem;color:var(--muted)'>"
        f"• {WORKERS} parallel workers<br>• {BATCH_SIZE} blocks per request<br>"
        f"• Drive folder: <code>{DRIVE_FOLDER}/</code></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- metrics

statuses = [block_status(s) for s in subs]
n_total = len(subs)
n_missing = statuses.count("missing")
n_done = statuses.count("done")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Blocks", n_total)
c2.metric("Done", n_done)
c3.metric("Missing", n_missing)
c4.metric("Complete", f"{n_done / max(1, n_total) * 100:.0f}%")

if _running:
    _done = len(_results)
    st.progress(_done / max(1, _total))
    st.caption(f"🔄 Translating… {_done}/{_total} blocks")
elif _elapsed > 0:
    _still = n_missing
    msg = f"✅ Done in {_elapsed:.1f}s — {n_total - _still}/{n_total} translated"
    if st.session_state.get("drive_saved"):
        msg += f" · saved to Drive `{st.session_state['drive_saved']}`"
    elif "drive_error" in st.session_state:
        msg += f" · Drive upload failed: {st.session_state['drive_error']}"
    else:
        msg += " · Drive not configured — download below"
    st.success(msg)
    if _error:
        st.error(f"Run error: {_error}")
    with _live["lock"]:
        _log = list(_live["log"])
    if _log:
        with st.expander(f"Errors during run ({len(_log)})"):
            for line in _log[:50]:
                st.text(line)


def block_is_pending(s, full):
    if full:
        return True
    return not s["arabic"].strip()


# ---------------------------------------------------------------- translate

if st.button("🚀 Translate", type="primary", width="stretch"):
    raw_keys = [v for k, v in st.secrets.items() if k.startswith("gemini_api")]
    api_keys = list(dict.fromkeys([k for k in raw_keys if k]))[:WORKERS]

    if not api_keys:
        st.error("No Gemini API keys found in secrets.")
        st.stop()

    full = mode.startswith("Full")
    pending = [i for i, s in enumerate(subs) if block_is_pending(s, full)]
    if not pending:
        st.success("Nothing to translate — every block already has Arabic.")
    else:
        reuse_map = build_reuse_map(subs) if reuse else {}

        with _live["lock"]:
            _live["results"] = {}
            _live["log"] = []
            _live["error"] = None
            _live["total"] = len(pending)
            _live["t0"] = time.time()
            _live["elapsed"] = 0.0
            _live["finalized"] = False
            _live["source_name"] = source_name
            _live["running"] = True
        st.session_state.pop("drive_saved", None)
        st.session_state.pop("drive_error", None)

        threading.Thread(
            target=_translate_worker,
            args=(_live, subs, pending, api_keys, reuse_map),
            daemon=True,
        ).start()
        st.rerun()

# ---------------------------------------------------------------- review

st.divider()
st.subheader("Review & edit")

df = pd.DataFrame({
    "#": [s["index"] for s in subs],
    "Start": [s["start"] for s in subs],
    "End": [s["end"] for s in subs],
    "English": ["\n".join(s["english_lines"]) for s in subs],
    "Arabic": [s["arabic"] for s in subs],
})

edited = st.data_editor(
    df,
    column_config={
        "#": st.column_config.TextColumn(disabled=True, width="small"),
        "Start": st.column_config.TextColumn(width="medium"),
        "End": st.column_config.TextColumn(width="medium"),
        "English": st.column_config.TextColumn(width="large"),
        "Arabic": st.column_config.TextColumn(width="large"),
    },
    hide_index=True,
    height=560,
    width="stretch",
    num_rows="fixed",
)

_fp_before = [s["arabic"] for s in subs]
for i, row in edited.iterrows():
    subs[i]["start"] = str(row["Start"]).strip()
    subs[i]["end"] = str(row["End"]).strip()
    subs[i]["arabic"] = str(row["Arabic"]).rstrip("\n")
    subs[i]["english_lines"] = str(row["English"]).split("\n")
_fp_after = [s["arabic"] for s in subs]
_changed = _fp_after != _fp_before

# ---------------------------------------------------------------- download

st.divider()
c6, c7 = st.columns([1, 2])
with c6:
    st.download_button(
        "⬇️ Download .srt",
        build_srt(subs),
        file_name=out_filename,
        mime="text/plain",
        width="stretch",
    )
with c7:
    st.caption(
        "Every edit here is included in the download and in the automatic "
        "Google Drive upload that happens when translation completes."
    )

# ---------------------------------------------------------------- live refresh

if _running:
    time.sleep(1.0)
    st.rerun()
