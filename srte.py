import os
import re
import time
import chardet
import streamlit as st
from google import genai
from google.genai import types

# --- Configuration ---
# Reverted to your futuristic 2026 model list
MODELS = [
    "gemini-2.5-flash-lite",  
    "gemini-2.5-flash", 
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview"
]
BATCH_SIZE = 12  # Optimal for balancing context window and speed
PAGE_SIZE = 25   # How many blocks to show in the UI at once to prevent lag

st.set_page_config(page_title="Pro SRT AI Translator", layout="wide")

# --- CSS Styling ---
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .subtitle-block { 
        padding: 15px; 
        margin-bottom: 10px; 
        border-left: 5px solid #4CAF50; 
        background: #f0f2f6; 
        border-radius: 5px; 
    }
    .block-idx { font-weight: bold; color: #1f77b4; margin-bottom: 5px; }
    textarea { font-family: 'Courier New', Courier, monospace !important; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def autodetect_decode(uploader):
    """Detects encoding and decodes the uploaded file safely."""
    raw = uploader.getvalue()
    result = chardet.detect(raw)
    encoding = result["encoding"] or "utf-8"
    try:
        return raw.decode(encoding)
    except:
        return raw.decode("utf-8", errors="ignore")

def parse_srt(text: str):
    """Robust SRT parser that handles malformed blocks and Arabic/English line splitting."""
    text = text.replace('\r\n', '\n')
    # Split by the standard SRT blank line pattern
    blocks = re.split(r'\n\s*\n', text.strip())
    subs = []
    
    for blk in blocks:
        lines = [l.strip() for l in blk.splitlines() if l.strip()]
        if len(lines) < 3: continue
        
        # Verify timestamp line exists
        if "-->" not in lines[1]: continue
        
        idx = lines[0]
        start, end = [t.strip() for t in lines[1].split("-->")]
        body_lines = lines[2:]
        
        # Separate existing Arabic and English lines
        arabic = [l for l in body_lines if any('\u0600' <= ch <= '\u06FF' for ch in l)]
        english = [l for l in body_lines if not any('\u0600' <= ch <= '\u06FF' for ch in l)]
        
        subs.append({
            "index": idx,
            "start": start,
            "end": end,
            "english_lines": english,
            "arabic": "\n".join(arabic)
        })
    return subs

def build_srt(subs):
    """Reconstructs the SRT file structure."""
    out = []
    for s in subs:
        combined = f"{s['arabic']}\n" if s.get('arabic') else ""
        combined += "\n".join(s["english_lines"])
        out.append(f"{s['index']}\n{s['start']} --> {s['end']}\n{combined}")
    return "\n\n".join(out)

def translate_batch(client, model_id, block_texts):
    """Calls Gemini API with strict system instructions and formatting."""
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a professional subtitle translator. Translate English to natural, cinematic Arabic. "
            "Maintain line breaks and speaker dashes (-). "
            "Output format MUST be: [1] Translation [2] Translation... No extra text."
        ),
        temperature=0.2,
    )
    
    prompt = "Translate these subtitle blocks:\n\n"
    for i, txt in enumerate(block_texts, 1):
        prompt += f"[{i}]\n{txt}\n\n"
    
    response = client.models.generate_content(
        model=model_id, 
        contents=prompt, 
        config=config,
    )
    
    # Extract blocks using regex
    results = re.split(r'\[\d+\]', response.text.strip())
    return [r.strip() for l in results if (r := l.strip())]

# --- Session Management ---
if "subs" not in st.session_state:
    st.session_state.subs = []
if "page" not in st.session_state:
    st.session_state.page = 0
if "uploaded_filename" not in st.session_state: # Track uploaded file
    st.session_state.uploaded_filename = None

# --- Sidebar & File Input ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploader = st.file_uploader("Upload English SRT", type="srt")
    if uploader:
        # Check if a new file is uploaded or if subs are empty
        if st.session_state.uploaded_filename != uploader.name:
            st.session_state.subs = [] # Clear existing if new file
            st.session_state.page = 0
            st.session_state.uploaded_filename = uploader.name
            st.session_state.subs = parse_srt(autodetect_decode(uploader))
            st.rerun() # Rerun to display data after upload
        elif not st.session_state.subs: # If no subs but filename matches (rerun behavior)
            st.session_state.subs = parse_srt(autodetect_decode(uploader))
            st.rerun() # Rerun to display data after upload
    
    if st.session_state.subs:
        st.divider()
        st.write(f"📊 Total Blocks: {len(st.session_state.subs)}")
        translated_count = sum(1 for s in st.session_state.subs if s["arabic"].strip())
        st.write(f"✅ Translated: {translated_count}")
        st.progress(translated_count / len(st.session_state.subs))

        st.divider()
        # --- Moved Download Section to Sidebar ---
        st.subheader("⬇️ Download Translated SRT")
        final_content = build_srt(st.session_state.subs)
        st.download_button(
            label="Download .srt File",
            data=final_content,
            file_name="translated_output.srt",
            mime="text/plain"
        )
        st.info("File ready for download! Make sure all translations are complete.")

# --- Main Logic ---
st.title("🌍 Pro SRT AI Translator")

if not st.session_state.subs:
    st.info("👋 Welcome! Please upload an SRT file in the sidebar to begin.")
    st.stop()

# --- Batch Translation Engine ---
col_action, col_empty = st.columns([1, 3])
if col_action.button("🚀 Auto-Translate All (Batch Mode)"):
    raw_keys = [val for key, val in st.secrets.items() if key.startswith("gemini_api")]
    api_keys = list(dict.fromkeys([k for k in raw_keys if k]))
    
    if not api_keys:
        st.error("No API keys found. Add gemini_api_1, gemini_api_2 etc. to Streamlit secrets.")
    else:
        pending = [i for i, s in enumerate(st.session_state.subs) if not s["arabic"].strip()]
        
        if not pending:
            st.success("All blocks are already translated!")
        else:
            progress_bar = st.progress(0)
            status = st.empty()
            
            key_idx = 0
            model_idx = 0 
            
            while pending:
                current_batch_idxs = pending[:BATCH_SIZE]
                texts = ["\n".join(st.session_state.subs[i]["english_lines"]) for i in current_batch_idxs]
                
                try:
                    client = genai.Client(api_key=api_keys[key_idx])
                    model_name = MODELS[model_idx]
                    status.info(f"⚡ Batch: {len(pending)} left | Key #{key_idx+1} | Model: {model_name}")
                    
                    results = translate_batch(client, model_name, texts)
                    
                    if not results:
                        raise ValueError("AI returned empty results.")

                    # Map results back to state
                    num_saved = 0
                    for i, trans_text in enumerate(results):
                        if i >= len(current_batch_idxs): break
                        real_idx = current_batch_idxs[i]
                        st.session_state.subs[real_idx]["arabic"] = trans_text
                        st.session_state[f"arabic_{real_idx}"] = trans_text
                        num_saved += 1
                    
                    # Update pending list only by successful count
                    pending = pending[num_saved:]
                    
                    # Update UI progress
                    total = len(st.session_state.subs)
                    done = total - len(pending)
                    progress_bar.progress(done / total)
                    
                    time.sleep(1) 
                    
                except Exception as e:
                    # Capture full error for debugging
                    err_detail = str(e)
                    status.warning(f"⚠️ Error with {MODELS[model_idx]} on Key #{key_idx+1}: {err_detail[:150]}...")
                    
                    # ROTATION LOGIC:
                    # 1. Try next model for current key
                    model_idx += 1
                    
                    # 2. If all 4 models exhausted, move to next key
                    if model_idx >= len(MODELS):
                        st.error(f"❌ Key #{key_idx+1} exhausted for all {len(MODELS)} models. Rotating Key...")
                        model_idx = 0
                        key_idx = (key_idx + 1) % len(api_keys)
                        time.sleep(5)
                    else:
                        time.sleep(2)
                    
                    # continue to retry the same batch with new setup
                    continue 

            status.success("✨ Translation complete!")
            st.rerun()

st.divider()

# --- Paginated Editor ---
total_pages = (len(st.session_state.subs) // PAGE_SIZE) + 1
page = st.select_slider("Page Navigation", options=range(total_pages), value=st.session_state.page)
st.session_state.page = page

start_idx = page * PAGE_SIZE
end_idx = min(start_idx + PAGE_SIZE, len(st.session_state.subs))

for i in range(start_idx, end_idx):
    s = st.session_state.subs[i]
    with st.container():
        st.markdown(f"<div class='subtitle-block'><div class='block-idx'># {s['index']} ({s['start']} → {s['end']})</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            eng_text = "\n".join(s["english_lines"])
            new_eng = st.text_area("English Source", eng_text, key=f"eng_{i}", height=100)
            st.session_state.subs[i]["english_lines"] = new_eng.splitlines()
        with c2:
            new_arabic = st.text_area("Arabic Translation", s["arabic"], key=f"arabic_{i}", height=100)
            st.session_state.subs[i]["arabic"] = new_arabic
        st.markdown("</div>", unsafe_allow_html=True)
