# 🌐 SRT AI Translator — English → Arabic

A Streamlit web app for **editing SRT subtitle files** and **translating them from English to Arabic** using Google Gemini AI — with multi-key rotation, model fallback, and batch processing.

> 🌐 **Live App:** [Deployed on Streamlit Community Cloud](https://subtitle-editor.streamlit.app)

## How It Works

```
Upload .srt file
       │
       ▼
┌──────────────────────────────────┐
│  1. Smart SRT Parser             │
│     • Auto-detect encoding       │  chardet → UTF-8, Arabic, BOM handling
│     • Splits English vs Arabic   │  Unicode range detection (\u0600–\u06FF)
│     • Preserves timestamps       │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  2. AI Batch Translation         │  Google Gemini API
│     • 12 blocks per API call     │
│     • Multi-key rotation         │  Cycles through N API keys
│     • Model fallback chain:      │
│       gemini-2.0-flash           │
│       → gemini-2.5-flash         │
│       → gemini-3-flash-preview   │
│     • Auto-retry on 429/quota    │
│     • Progress bar tracking      │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  3. Interactive Review           │
│     • Side-by-side EN → AR       │
│     • Edit any block inline      │
│     • Adjust timestamps          │
│     • Build & download .srt      │
└──────────────────────────────────┘
```

## ✨ Key Features

| Feature | Detail |
|---------|--------|
| **Multi-Key Rotation** | Dynamically reads all `gemini_api*` keys from Streamlit secrets — rotates automatically when one hits quota |
| **Model Fallback Chain** | `gemini-2.0-flash` → `gemini-2.5-flash` → `gemini-3-flash-preview` — if a model fails, tries the next |
| **Batch Translation** | Processes 12 subtitle blocks per API call for efficiency, with `[1]`, `[2]` indexed parsing |
| **Arabic Detection** | Uses Unicode range `U+0600–U+06FF` to separate existing Arabic lines from English lines |
| **Encoding Detection** | `chardet` auto-detects file encoding — handles BOM markers, UTF-8, Latin-1 |
| **Inline Editing** | Every block is editable — timestamps, English text, Arabic translation |
| **Rate Limit Handling** | Catches 429, quota, and "resource exhausted" errors → rotates key/model and retries |

---

### `sync_timestamps.py` — SRT Timestamp Synchronizer *(CLI)*

When you translate subtitles and the timestamps drift, this script **copies timestamps from a reference SRT** into a translated SRT by matching English text lines.

```bash
python sync_timestamps.py new_srt_dir/ translated_srt_dir/ --out-dir fixed/
```

**How it works:**
- Parses both SRT files, filters out Arabic lines for matching
- Normalizes whitespace and builds a text → timestamp lookup
- Replaces timestamps in the translated file where English text matches
- Handles duplicates and BOM markers gracefully

---

## 🚀 Quick Start

```bash
pip install streamlit chardet google-genai

# .streamlit/secrets.toml:
# gemini_api_1 = "key_1"
# gemini_api_2 = "key_2"
# gemini_api_3 = "key_3"

streamlit run srte.py
```

## 📋 Requirements

```
streamlit
chardet
google-genai
```

## 💡 Future Ideas

- [ ] Support for additional target languages (French, Spanish, Turkish)
- [ ] VTT and ASS subtitle format support
- [ ] Bulk file upload (process entire season of subtitles)
- [ ] Side-by-side video preview with synced subtitles

---

> Built for Arabic content localization — translating hundreds of subtitle files per week for music video and documentary distribution.
