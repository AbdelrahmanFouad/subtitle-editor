import os
import re
import time
import chardet
import streamlit as st
from google import genai
from google.api_core import exceptions

# --- Configure Gemini API ---
# Note: Using the model ID provided in your error message
MODEL_ID = "gemini-2.0-flash-lite" 

if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=st.secrets["gemini_api_key"])

st.set_page_config(layout="wide")
st.title("SRT Editor & Translator: English → Arabic (AI)")

# --- CSS for general styling ---
st.markdown("""
<style>
.subtitle-block { 
    padding: 4px; 
    margin-bottom: 4px; 
    border: 1px solid #ddd; 
    border-radius: 4px; 
    background: #f9f9f9; 
}
.block-heading { 
    font-size: 1rem; 
    font-weight: 600; 
    margin-bottom: 3px; 
}
textarea {
    scrollbar-width: auto;
    scrollbar-color: #888 #f1f1f1;
}
textarea::-webkit-scrollbar {
    width: 16px;
}
textarea::-webkit-scrollbar-track {
    background: #f1f1f1;
}
textarea::-webkit-scrollbar-thumb {
    background-color: #888;
    border-radius: 10px;
    border: 3px solid #f1f1f1;
}
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def autodetect_decode(uploader):
    raw = uploader.getvalue()
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    try:
        return raw.decode(enc)
    except Exception as e:
        st.error(f"Could not decode file with encoding {enc}. Error: {e}")
        return ""

def parse_srt(text: str):
    blocks = re.split(r'\n\s*\n', text.strip())
    subs = []
    for blk in blocks:
        lines = blk.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        idx = lines[0].strip()
        start, end = [t.strip() for t in lines[1].split("-->")]
        body_lines = [l.strip() for l in lines[2:] if l.strip()]
        arabic_lines = []
        english_lines = []
        for line in body_lines:
            if any('\u0600' <= ch <= '\u06FF' for ch in line):
                arabic_lines.append(line)
            else:
                english_lines.append(line)
        arabic = "\n".join(arabic_lines)
        subs.append({
            "index": idx,
            "start": start,
            "end": end,
            "english_lines": english_lines,
            "arabic": arabic
        })
    return subs

def build_srt(subs):
    out = []
    for s in subs:
        combined = f"{s['arabic']}\n" if s['arabic'] else ""
        combined += "\n".join(s["english_lines"])
        out.append(f"{s['index']}\n{s['start']} --> {s['end']}\n{combined}")
    return "\n\n".join(out)

def translate_batch(block_texts: list[str]) -> list[str]:
    # Optimized prompt for batching
    prompt = (
        "Translate these English subtitle blocks to Arabic. "
        "Maintain line breaks. Output ONLY the Arabic text for each block, separated by double newlines. "
        "Do not include block numbers or labels.\n\n"
    )
    for i, blk in enumerate(block_texts, 1):
        prompt += f"[{i}]\n{blk}\n\n"
    
    response = st.session_state.client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    text = response.text.strip()
    # Handle various possible delimiters from the model
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts

# --- UI: File Upload ---
uploader = st.file_uploader("Upload SRT file", type="srt")
if not uploader:
    st.info("Please upload your SRT file.")
    st.stop()

text = autodetect_decode(uploader)
subs = parse_srt(text)
st.session_state.setdefault("subs", subs)

# --- Translation: Entire File in Batches ---
if st.button("Translate Entire File (Optimized Batches)"):
    api_keys = [
        st.secrets["gemini_api_key"],
        st.secrets["gemini_api_2"],
        st.secrets["gemini_api_3"],
        st.secrets["gemini_api_4"],
    ]
    key_index = 0
    st.session_state.client = genai.Client(api_key=api_keys[key_index])
    
    pending_idxs = [i for i, s in enumerate(st.session_state.subs) if not s["arabic"].strip()]
    total_pending = len(pending_idxs)
    batch_size = 12  # INCREASED: Fewer requests = fewer quota hits
    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_start = 0

    while batch_start < total_pending:
        current_batch_indices = pending_idxs[batch_start : batch_start + batch_size]
        texts_to_translate = ["\n".join(st.session_state.subs[i]["english_lines"]) for i in current_batch_indices]
        block_numbers = [st.session_state.subs[i]['index'] for i in current_batch_indices]

        try:
            status_text.info(f"Translating blocks: {block_numbers[0]} to {block_numbers[-1]} (Key #{key_index + 1})")
            
            translations = translate_batch(texts_to_translate)

            # Verification logic to handle model output mismatches
            if len(translations) < len(current_batch_indices):
                st.warning(f"Batch mismatch (expected {len(current_batch_indices)}, got {len(translations)}). Reducing batch size for this turn...")
                # Temporary fallback: process a smaller chunk if the model gets confused
                temp_batch_size = max(1, len(translations))
                current_batch_indices = current_batch_indices[:temp_batch_size]
                translations = translations[:temp_batch_size]

            for i, idx in enumerate(current_batch_indices):
                st.session_state.subs[idx]["arabic"] = translations[i]
            
            batch_start += len(current_batch_indices)
            progress = min(1.0, batch_start / total_pending)
            progress_bar.progress(progress)
            
            if batch_start < total_pending:
                time.sleep(10) # Safe buffer between successful calls

        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str or "resourceexhausted" in error_str:
                st.warning(f"Quota hit on key #{key_index + 1}. Entering 30s cooldown...")
                time.sleep(30) # COOLDOWN: Let the per-minute limit reset
                
                key_index = (key_index + 1) % len(api_keys) # LOOP BACK to key 1 if needed
                st.session_state.client = genai.Client(api_key=api_keys[key_index])
                st.info(f"Switched to Key #{key_index + 1}. Retrying...")
            else:
                st.error(f"Error: {e}. Retrying in 5s...")
                time.sleep(5)

    if batch_start >= total_pending and total_pending > 0:
        progress_bar.progress(1.0)
        status_text.success("Translation complete!")

# --- Review & Download ---
st.write("### Subtitle Blocks")
edited = []
for i, s in enumerate(st.session_state.subs):
    st.markdown("<div class='subtitle-block'>", unsafe_allow_html=True)
    st.markdown(f"<div class='block-heading'>Block {s['index']}</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        start_time = st.text_input("Start", s["start"], key=f"start_{i}")
        end_time = st.text_input("End", s["end"], key=f"end_{i}")
    with c2:
        arabic_text = st.text_area("Arabic", s["arabic"], key=f"arabic_{i}", height=80)
        english_text = st.text_area("English", "\n".join(s["english_lines"]), key=f"english_{i}", height=80)
    edited.append({
        "index": s["index"], "start": start_time, "end": end_time,
        "arabic": arabic_text, "english_lines": english_text.splitlines()
    })
    st.markdown("</div>", unsafe_allow_html=True)
st.session_state.subs = edited

if st.button("Build & Download SRT"):
    final = build_srt(st.session_state.subs)
    st.download_button("Download .srt", final, "translated.srt", "text/plain")
