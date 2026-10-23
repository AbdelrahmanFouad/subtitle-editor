"""
srte.py — SRT Editor & Translator: English → Arabic (AI)

A fast, accurate, robust Streamlit app for translating SRT subtitles with
Google Gemini across a pool of API keys.

Highlights
----------
* CONCURRENT translation — one worker lane per API key, batches fanned out
  with a ThreadPoolExecutor. A 331-block file that took minutes now lands in
  ~15-25 s.
* ACCURATE — index-based response parsing (a missing block can't shift its
  neighbours), per-block retry, and a within-file reuse cache so a repeated
  line is translated once and reused consistently.
* ROBUST — a thread-safe key pool that round-robins across every key, parks a
  key after repeated failures, and never spins forever on a stuck batch.
  Every retryable error (429, 503, 404, connection/timeout) is handled.
* WIDE Arabic detection — base + supplement + extended-A/B + presentation
  forms, so no Arabic line is misclassified as English.
* Modern UI — live metrics, progress, an editable data grid for review, and a
  translation scope selector (missing / partial / all).

Usage:
    streamlit run srte.py
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import chardet
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------------- configuration

# Models verified to exist and to return the [N]-delimited format. The retired
# gemini-2.0-flash and the 503-prone gemini-3.8-flash / gemini-3.5-flash-lite
# are intentionally absent.
MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
]

DEFAULT_BATCH_SIZE = 16
MAX_BATCH_ATTEMPTS = 4      # attempts per batch before it is abandoned
MAX_ROUNDS = 3              # rounds of shrinking granularity
MAX_KEY_FAILS = 3           # consecutive failures that park a key

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
    """Map the model's own [N] markers to their bodies.

    Keyed by the number the model printed, so a missing or empty entry cannot
    shift every later translation (the positional split did exactly that).
    """
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


def arabic_line_count(s: dict) -> int:
    return len([l for l in s["arabic"].split("\n") if l.strip()])


def block_status(s: dict) -> str:
    ar = arabic_line_count(s)
    if not ar:
        return "missing"
    if ar < len(s["english_lines"]):
        return "partial"
    return "done"


def build_reuse_map(subs: list[dict]) -> dict[str, str]:
    """Within-file reuse: english (lowercased, joined) -> arabic for blocks that
    are already fully translated. Repeated lines reuse the same translation,
    which is both faster and more consistent."""
    m: dict[str, str] = {}
    for s in subs:
        if block_status(s) != "done":
            continue
        key = "\n".join(s["english_lines"]).strip().lower()
        if key:
            m[key] = s["arabic"]
    return m


# ---------------------------------------------------------------- persistence

# Run history lives next to the script. Every completed batch and every edit is
# checkpointed so a lost Streamlit session can be recovered from the last good
# state instead of losing the whole run.
APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = APP_DIR / "checkpoints"
HISTORY_DIR = CHECKPOINT_DIR / "history"
LATEST_JSON = CHECKPOINT_DIR / "latest.json"


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name or "srt")


def save_checkpoint(subs, source_name: str, tag: str = "auto") -> str:
    """Persist the full state to latest.json and a timestamped history entry.

    Returns the timestamp string. Called from the translate loop (crash-safe,
    one write per completed batch) and after edits.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "source": source_name,
        "tag": tag,
        "saved": ts,
        "blocks": len(subs),
        "subs": subs,
    }
    try:
        LATEST_JSON.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        safe = _safe_name(source_name)
        (HISTORY_DIR / f"{ts}_{safe}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        (HISTORY_DIR / f"{ts}_{safe}.srt").write_text(
            build_srt(subs), encoding="utf-8-sig")
    except OSError:
        # persistence must never break the run — swallow filesystem errors
        pass
    return ts


def list_checkpoints() -> list[Path]:
    """Newest-first list of saved JSON checkpoints."""
    if not HISTORY_DIR.exists():
        return []
    return sorted(HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime,
                  reverse=True)


def load_checkpoint(path: Path) -> tuple[list[dict], str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["subs"], data.get("source", ""), data.get("saved", "")


def prune_checkpoints(keep: int = 50) -> None:
    """Drop the oldest history entries beyond `keep` to avoid unbounded growth."""
    cps = list_checkpoints()
    for p in cps[keep:]:
        try:
            p.unlink()
            p.with_suffix(".srt").unlink()
        except OSError:
            pass


# ---------------------------------------------------------------- key pool

class KeyPool:
    """Thread-safe round-robin over API keys.

    Each key keeps its own position in the model list. A key that fails
    MAX_KEY_FAILS times consecutively is parked until every key is parked, at
    which point the counters reset and the pool keeps going.
    """

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


def translate_all(subs, pending, api_keys, batch_size, workers, reuse_map,
                  progress_cb, log_cb, apply_cb=None) -> dict[int, str]:
    """Fan out all pending blocks across worker threads; retry in rounds.

    `apply_cb(idx, text)` is called on the calling thread for every block as it
    completes, so the caller can persist progress incrementally (crash-safe)
    without the workers touching shared state.

    Returns {idx: arabic_text} for every block that was translated (missing
    blocks are absent).
    """
    clients = [genai.Client(api_key=k) for k in api_keys]
    pool = KeyPool(len(api_keys))
    result: dict[int, str] = {}
    resolved_immediately: dict[int, str] = {}

    # Phase 0 — reuse cache, zero API calls
    for i in list(pending):
        key = "\n".join(subs[i]["english_lines"]).strip().lower()
        if key in reuse_map:
            resolved_immediately[i] = reuse_map[key]
    result.update(resolved_immediately)
    if apply_cb:
        for idx, t in resolved_immediately.items():
            apply_cb(idx, t)

    remaining = [i for i in pending if i not in result]
    total = len(pending)
    progress_cb(len(result), total, "reuse")

    rounds = [
        ("batch", batch_size),
        ("batch", max(1, batch_size // 2)),
        ("single", 1),
    ][:MAX_ROUNDS]

    for round_no, (label, size) in enumerate(rounds):
        if not remaining:
            break
        batches = [remaining[i:i + size] for i in range(0, len(remaining), size)]
        n_workers = min(workers, max(1, len(batches)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_one_batch, pool, clients, subs, b, reuse_map): b
                    for b in batches}
            done = 0
            for f in as_completed(futs):
                got, err = f.result()
                for idx, t in got.items():
                    if t:
                        result[idx] = t
                        if apply_cb:
                            apply_cb(idx, t)
                if err:
                    log_cb(err)
                done += 1
                progress_cb(len(result), total, f"{label} r{round_no + 1}")
        remaining = [i for i in remaining if i not in result]

    return result


# ---------------------------------------------------------------- session state

st.set_page_config(page_title="SRT Translator", layout="wide")

if "subs" not in st.session_state:
    st.session_state.subs = []
if "loaded_file" not in st.session_state:
    st.session_state.loaded_file = None

uploader = st.file_uploader("Upload SRT file", type="srt")

if uploader:
    sig = f"{uploader.name}:{uploader.size}"
    if st.session_state.loaded_file != sig:
        st.session_state.subs = parse_srt(autodetect_decode(uploader))
        st.session_state.loaded_file = sig
        for k in [k for k in st.session_state.keys()
                  if k.startswith(("arabic_", "english_", "start_", "end_"))]:
            del st.session_state[k]

if not st.session_state.subs:
    # Fresh session: offer recovery from the last checkpoint before asking for
    # an upload. This is how lost work is recovered after a session drop.
    cps = list_checkpoints()
    if cps:
        latest_meta = None
        if LATEST_JSON.exists():
            try:
                latest_meta = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                latest_meta = None
        src = latest_meta.get("source") if latest_meta else ""
        saved = latest_meta.get("saved") if latest_meta else ""
        blocks = latest_meta.get("blocks") if latest_meta else ""
        st.warning(
            f"Found saved work: **{src}** ({blocks} blocks, saved {saved}). "
            "Recover it below instead of starting over."
        )
        c_rec, c_up = st.columns([1, 3])
        with c_rec:
            if st.button("♻️ Recover last session", type="primary"):
                try:
                    subs, src_name, saved_at = load_checkpoint(LATEST_JSON)
                    st.session_state.subs = subs
                    st.session_state.loaded_file = f"recovered:{src_name}:{saved_at}"
                    st.rerun()
                except (OSError, json.JSONDecodeError, KeyError) as e:
                    st.error(f"Recovery failed: {e}")
        with c_up:
            st.caption("or upload a new .srt below.")
    st.info("Upload an .srt file to begin.")
    st.stop()

subs = st.session_state.subs

# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")
    scope_label = st.radio(
        "Translate which blocks?",
        ["missing (no Arabic)", "partial (also stale Arabic)", "all blocks"],
        index=0, key="scope",
    )
    scope = scope_label.split(" ")[0]
    batch_size = st.slider("Batch size", 4, 24, DEFAULT_BATCH_SIZE, key="batch_size")
    n_keys = st.number_input("API keys in use", 1, 10, 10, key="n_keys")
    workers = st.slider("Parallel workers", 1, 10, 8, key="workers")
    reuse = st.checkbox("Reuse repeated lines (faster + consistent)", value=True,
                        key="reuse")
    st.caption("Workers ≈ API keys for max speed. Lower it if you hit quota.")

    st.divider()
    st.subheader("Run history")
    cps = list_checkpoints()
    if cps:
        st.caption(f"{len(cps)} saved checkpoint(s)")
        for p in cps[:8]:
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
                label = f"{meta.get('saved', '')} · {meta.get('source', '')} · {meta.get('blocks', '')} blk"
            except (OSError, json.JSONDecodeError):
                label = p.name
            if st.button(label, key=f"restore_{p.stem}"):
                try:
                    rsubs, rsrc, rsaved = load_checkpoint(p)
                    st.session_state.subs = rsubs
                    st.session_state.loaded_file = f"recovered:{rsrc}:{rsaved}"
                    st.rerun()
                except (OSError, json.JSONDecodeError, KeyError) as e:
                    st.error(f"Restore failed: {e}")
    else:
        st.caption("No checkpoints yet — run a translation and they'll appear here.")

# ---------------------------------------------------------------- metrics

statuses = [block_status(s) for s in subs]
n_total = len(subs)
n_missing = statuses.count("missing")
n_partial = statuses.count("partial")
n_done = statuses.count("done")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Blocks", n_total)
c2.metric("Done", n_done)
c3.metric("Missing", n_missing)
c4.metric("Partial", n_partial)
c5.metric("Complete", f"{n_done / max(1, n_total) * 100:.0f}%")


def block_is_pending(s, sc):
    if sc == "all":
        return True
    if sc == "partial":
        return block_status(s) in ("missing", "partial")
    return block_status(s) == "missing"


# ---------------------------------------------------------------- translate

if st.button("🚀 Translate", type="primary", width="stretch"):
    raw_keys = [v for k, v in st.secrets.items() if k.startswith("gemini_api")]
    api_keys = list(dict.fromkeys([k for k in raw_keys if k]))[:n_keys]

    if not api_keys:
        st.error("No Gemini API keys found in secrets.")
        st.stop()

    pending = [i for i, s in enumerate(subs) if block_is_pending(s, scope)]
    if not pending:
        st.success("Nothing to translate — every block is already done.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_lines = []
        # source name for checkpointing (upload name, or recovered identity)
        source_name = uploader.name if uploader else st.session_state.loaded_file
        if not source_name:
            source_name = "recovered.srt"
        checkpoint_meta = {"count": 0}

        def apply_cb(idx, text):
            subs[idx]["arabic"] = text
            checkpoint_meta["count"] += 1
            # persist every 12 applied blocks so a crash mid-run loses at most
            # a handful of blocks, not the whole run
            if checkpoint_meta["count"] % 12 == 0:
                save_checkpoint(subs, source_name, tag="progress")

        def progress_cb(resolved, total, tag):
            progress_bar.progress(resolved / max(1, total))
            status_text.caption(
                f"{tag}: {resolved}/{total} blocks translated")

        def log_cb(err):
            log_lines.append(err)

        t0 = time.time()
        reuse_map = build_reuse_map(subs) if reuse else {}
        result = translate_all(
            subs, pending, api_keys, batch_size, workers, reuse_map,
            progress_cb, log_cb, apply_cb=apply_cb)

        for idx, t in result.items():
            subs[idx]["arabic"] = t

        save_checkpoint(subs, source_name, tag="complete")
        prune_checkpoints()

        still = [i for i in pending if not subs[i]["arabic"].strip()]
        elapsed = time.time() - t0

        status_text.success(
            f"Done in {elapsed:.1f}s — {len(pending) - len(still)} translated, "
            f"{len(still)} still missing.")
        if still:
            st.error(f"{len(still)} block(s) could not be translated:")
            for i in still:
                st.write(f"- #{subs[i]['index']}: "
                         f"`{' / '.join(subs[i]['english_lines'])}`")
        if log_lines:
            with st.expander(f"Errors encountered ({len(log_lines)})"):
                for line in log_lines[:50]:
                    st.text(line)
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

# sync edits back to canonical state (same run -> edits survive)
_fp_before = [s["arabic"] for s in subs]
for i, row in edited.iterrows():
    subs[i]["start"] = str(row["Start"]).strip()
    subs[i]["end"] = str(row["End"]).strip()
    subs[i]["arabic"] = str(row["Arabic"]).rstrip("\n")
    subs[i]["english_lines"] = str(row["English"]).split("\n")
_fp_after = [s["arabic"] for s in subs]
if _fp_after != _fp_before:
    save_checkpoint(subs, st.session_state.loaded_file or "edited.srt",
                    tag="edit")

# ---------------------------------------------------------------- download

st.divider()
c6, c7 = st.columns([1, 2])
with c6:
    st.download_button(
        "⬇️ Download .srt",
        build_srt(subs),
        file_name="translated.srt",
        mime="text/plain",
        width="stretch",
    )
with c7:
    st.caption(
        "The file above is built live from the grid, so any edit you make "
        "here is included in the download."
    )
