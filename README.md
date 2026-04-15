# SRT Editor & AI Translator

A Streamlit web app for editing SRT subtitle files and translating them from **English → Arabic** using the Gemini AI API — with automatic key rotation, model fallback, and batch processing.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://subtitle-editor.streamlit.app)

## ✨ Features

- **Upload & Parse SRT** — load any `.srt` file with automatic encoding detection (handles UTF-8, Arabic encodings, etc.)
- **Side-by-side Editor** — review and manually edit both English source and Arabic translation for every subtitle block
- **AI Batch Translation** — translates the entire file in batches of 12 blocks per API call using Google Gemini
- **Multi-key Rotation** — automatically rotates between multiple Gemini API keys when quota is exhausted
- **Model Fallback Chain** — cycles through `gemini-2.0-flash → gemini-2.5-flash → gemini-3-flash` if a model fails
- **Build & Download** — reconstruct and download the final bilingual `.srt` file in one click
- **Stateful Session** — edits are preserved across re-runs via Streamlit session state

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Configure API Keys

Create `.streamlit/secrets.toml`:

```toml
gemini_api_1 = "AIzaSy..."
gemini_api_2 = "AIzaSy..."   # optional: add more for rotation
```

### Run

```bash
streamlit run srte.py
```

Then open [http://localhost:8501](http://localhost:8501) and upload your `.srt` file.

## 🔄 Translation Workflow

```
Upload .srt
     │
     ▼
parse_srt()          ← splits into blocks, detects Arabic/English lines
     │
     ▼
translate_batch()    ← sends 12 blocks per Gemini API call
     │
  ┌──┴──────────────────────────┐
  │ Quota error?                │
  │  → try next model           │
  │ All models failed?          │
  │  → rotate to next API key   │
  └─────────────────────────────┘
     │
     ▼
Session state updated → UI reflects new Arabic translations
     │
     ▼
build_srt() → bilingual .srt download
```

## 📁 File Structure

```
subtitle-editor/
├── srte.py              # Main Streamlit app
├── split.py             # SRT file splitter utility
├── requirements.txt
└── .streamlit/
    └── secrets.toml     # API keys (not committed)
```

## 📋 Requirements

```
streamlit
google-genai
chardet
```

## 💡 Future Ideas

- [ ] Support additional language pairs (Arabic → English, French → Arabic)
- [ ] Timestamp offset correction tool
- [ ] Batch upload multiple SRT files
- [ ] Export to `.vtt` (WebVTT) format
- [ ] Confidence scoring per translation block

---

> Originally built for translating Arabic music video subtitles at scale — handling hundreds of subtitle files with minimal manual effort.
