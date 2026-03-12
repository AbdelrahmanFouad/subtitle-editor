import os
import re
import time
import chardet
import streamlit as st
from google import genai
from google.genai import types

# --- Configuration ---
DEFAULT_MODEL = "gemini-2.5-flash" 

st.set_page_config(layout="wide", page_title="AI SRT Translator")
st.title("SRT Editor & Translator: English → Arabic (AI)")

# --- CSS for styling ---
st.markdown("""
<style>
.subtitle-block { padding: 10px; margin-bottom: 8px; border: 1px solid #ddd; border-radius: 8px; background: #fdfdfd; }
.block-heading { font-weight: bold; color: #1f77b4; margin-bottom: 5px; }
.status-box { padding: 15px; border-radius: 5px; background: #e1f5fe; border-left: 5px solid #0288d1; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def autodetect_decode(uploader):
    raw = uploader.getvalue()
    result = chardet.detect(raw)
    enc = result["encoding"] or "utf-8"
    try:
        return raw.decode(enc)
    except:
        return raw.decode("utf-8", errors="ignore")

def parse_srt(text: str):
    # Normalize line endings and split by empty lines
    text = text.replace('\r\n', '\n')
    blocks = re.split(r'\n\s*\n', text.strip())
    subs = []
    for blk in blocks:
        lines = blk.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        idx = lines[0].strip()
        times = lines[1].split("-->")
        start, end = times[0].strip(), times[1].strip()
        body = lines[2:]
        
        arabic_lines = [l for l in body if any('\u0600' <= ch <= '\u06FF' for ch in l)]
        english_lines = [l for l in body if not any('\u0600' <= ch <= '\u06FF' for ch in l)]
        
        subs.append({
            "index": idx, "start": start, "end": end,
            "english_lines": english_lines, "arabic": "\n".join(arabic_lines)
        })
    return subs

def translate_batch(client, model_id, block_texts):
    # Use System Instruction for better control
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a professional subtitle translator. Translate English to natural Arabic. "
            "Output format: Return each translation preceded by its index in brackets, like [1], [2], etc. "
            "Example output:\n[1]\nمرحبا كيف حالك؟\n\n[2]\nأنا بخير شكرا."
        ),
        temperature=0.3,
    )
    
    prompt = "Translate these blocks:\n\n"
    for i, txt in enumerate(block_texts, 1):
        prompt += f"[{i}]\n{txt}\n\n"
    
    response = client.models.generate_content(model=model_id, contents=prompt, config=config)
    raw_text = response.text.strip()
    
    # Parse based on bracket markers [1], [2]...
    results = []
    # Find all content between [n] markers
    parts = re.split(r'\[\d+\]', raw_text)
    for p in parts:
        if p.strip():
            results.append(p.strip())
    return results

# --- Session State ---
if "subs" not in st.session_state:
    st.session_state.subs = []

# --- UI: File Upload ---
uploader = st.file_uploader("Upload SRT file", type="srt")
if uploader:
    if not st.session_state.subs:
        text = autodetect_decode(uploader)
        st.session_state.subs = parse_srt(text)

if not st.session_state.subs:
    st.info("Please upload an SRT file to begin.")
    st.stop()

# --- Translation Logic ---
if st.button("🚀 Start Batch Translation"):
    # Deduplicate keys to avoid hitting the same limit twice
    raw_keys = [st.secrets.get(k) for k in ["gemini_api_key", "gemini_api_2", "gemini_api_3", "gemini_api_4"]]
    api_keys = list(dict.fromkeys([k for k in raw_keys if k])) # Keep unique, non-None keys
    
    pending_idxs = [i for i, s in enumerate(st.session_state.subs) if not s["arabic"].strip()]
    
    if not pending_idxs:
        st.success("All blocks are already translated!")
    else:
        key_idx = 0
        batch_size = 15 # Optimized for speed vs quota
        progress_bar = st.progress(0)
        status_area = st.empty()
        
        current_idx = 0
        while current_idx < len(pending_idxs):
            key = api_keys[key_idx]
            client = genai.Client(api_key=key)
            
            batch_indices = pending_idxs[current_idx : current_idx + batch_size]
            texts = ["\n".join(st.session_state.subs[i]["english_lines"]) for i in batch_indices]
            
            status_area.markdown(f"""
            <div class='status-box'>
                <b>Status:</b> Translating blocks {st.session_state.subs[batch_indices[0]]['index']} to {st.session_state.subs[batch_indices[-1]]['index']}<br>
                <b>Using Key:</b> #{key_idx + 1} ({len(api_keys)} total unique keys)<br>
                <b>Progress:</b> {current_idx}/{len(pending_idxs)} blocks
            </div>
            """, unsafe_allow_html=True)
            
            try:
                translations = translate_batch(client, DEFAULT_MODEL, texts)
                
                # Apply translations (handle potential model length mismatches)
                applied = 0
                for i, idx in enumerate(batch_indices):
                    if i < len(translations):
                        st.session_state.subs[idx]["arabic"] = translations[i]
                        applied += 1
                
                current_idx += applied
                progress_bar.progress(current_idx / len(pending_idxs))
                time.sleep(10) # Safety buffer
                
            except Exception as e:
                err = str(e).lower()
                if any(x in err for x in ["quota", "429", "resourceexhausted"]):
                    st.warning(f"⚠️ Key #{key_idx + 1} quota exceeded. Waiting 65s for reset...")
                    time.sleep(65)
                    key_idx = (key_idx + 1) % len(api_keys)
                else:
                    st.error(f"Unexpected error: {e}")
                    time.sleep(5)
        
        status_area.success("✅ Translation Process Finished!")

# --- Review and Edit ---
st.subheader("Edit Translations")
for i, s in enumerate(st.session_state.subs):
    with st.container():
        st.markdown(f"<div class='subtitle-block'><div class='block-heading'>Block {s['index']} ({s['start']} → {s['end']})</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.subs[i]["english_lines"] = st.text_area("English", "\n".join(s["english_lines"]), key=f"en_{i}", height=100).splitlines()
        with col2:
            st.session_state.subs[i]["arabic"] = st.text_area("Arabic", s["arabic"], key=f"ar_{i}", height=100)
        st.markdown("</div>", unsafe_allow_html=True)

# --- Download ---
if st.button("📦 Build & Download SRT"):
    out = []
    for s in st.session_state.subs:
        body = s["arabic"] + "\n" + "\n".join(s["english_lines"])
        out.append(f"{s['index']}\n{s['start']} --> {s['end']}\n{body}")
    final_srt = "\n\n".join(out)
    st.download_button("Click to Download", final_srt, file_name="translated_subs.srt")
