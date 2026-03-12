import os
import re
import time
import chardet
import streamlit as st
from google import genai
from google.genai import types

# --- Configure Gemini API ---
DEFAULT_MODEL = "gemini-2.5-flash" 

st.set_page_config(layout="wide")
st.title("SRT Editor & Translator: English → Arabic (AI)")

# --- CSS for general styling ---
st.markdown("""
<style>
.subtitle-block { padding: 4px; margin-bottom: 4px; border: 1px solid #ddd; border-radius: 4px; background: #f9f9f9; }
.block-heading { font-size: 1rem; font-weight: 600; margin-bottom: 3px; }
textarea { scrollbar-width: auto; scrollbar-color: #888 #f1f1f1; }
.live-preview { background-color: #e3f2fd; padding: 10px; border-radius: 5px; border-left: 5px solid #2196f3; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def autodetect_decode(uploader):
    raw = uploader.getvalue()
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    try:
        return raw.decode(enc)
    except:
        return raw.decode("utf-8", errors="ignore")

def parse_srt(text: str):
    text = text.replace('\r\n', '\n')
    blocks = re.split(r'\n\s*\n', text.strip())
    subs = []
    for blk in blocks:
        lines = blk.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        idx = lines[0].strip()
        start, end = [t.strip() for t in lines[1].split("-->")]
        body_lines = [l.strip() for l in lines[2:] if l.strip()]
        arabic_lines = [l for l in body_lines if any('\u0600' <= ch <= '\u06FF' for ch in l)]
        english_lines = [l for l in body_lines if not any('\u0600' <= ch <= '\u06FF' for ch in l)]
        subs.append({
            "index": idx, "start": start, "end": end,
            "english_lines": english_lines, "arabic": "\n".join(arabic_lines)
        })
    return subs

def translate_batch(client, model_id, block_texts):
    config = types.GenerateContentConfig(
        system_instruction="Translate English subtitle blocks to natural Arabic. Maintain line breaks. Output translations preceded by [1], [2], etc.",
        temperature=0.3,
    )
    prompt = "Translate these blocks:\n\n"
    for i, txt in enumerate(block_texts, 1):
        prompt += f"[{i}]\n{txt}\n\n"
    
    response = client.models.generate_content(model=model_id, contents=prompt, config=config)
    raw_text = response.text.strip()
    results = [p.strip() for p in re.split(r'\[\d+\]', raw_text) if p.strip()]
    return results

# --- Session State Initialization ---
if "subs" not in st.session_state:
    st.session_state.subs = []

# --- UI: File Upload ---
uploader = st.file_uploader("Upload SRT file", type="srt")
if uploader and not st.session_state.subs:
    text = autodetect_decode(uploader)
    st.session_state.subs = parse_srt(text)

if not st.session_state.subs:
    st.info("Please upload your SRT file.")
    st.stop()

# --- Translation Action ---
if st.button("🚀 Translate Entire File (Batches)"):
    api_keys = list(dict.fromkeys([st.secrets.get(k) for k in ["gemini_api_key", "gemini_api_2", "gemini_api_3", "gemini_api_4"] if st.secrets.get(k)]))
    pending_idxs = [i for i, s in enumerate(st.session_state.subs) if not s["arabic"].strip()]
    
    if pending_idxs:
        batch_size = 12
        progress_bar = st.progress(0)
        status_area = st.empty()
        preview_area = st.empty()
        
        key_idx = 0
        current_progress = 0
        
        while current_progress < len(pending_idxs):
            client = genai.Client(api_key=api_keys[key_idx])
            batch_indices = pending_idxs[current_progress : current_progress + batch_size]
            texts = ["\n".join(st.session_state.subs[i]["english_lines"]) for i in batch_indices]
            
            try:
                status_area.info(f"Translating: Blocks {st.session_state.subs[batch_indices[0]]['index']} - {st.session_state.subs[batch_indices[-1]]['index']} (Key #{key_idx+1})")
                translations = translate_batch(client, DEFAULT_MODEL, texts)
                
                applied = 0
                for i, idx in enumerate(batch_indices):
                    if i < len(translations):
                        st.session_state.subs[idx]["arabic"] = translations[i]
                        applied += 1
                
                if translations:
                    preview_area.markdown(f"<div class='live-preview'><b>Live Preview:</b><br>{translations[-1][:100]}...</div>", unsafe_allow_html=True)
                
                current_progress += applied
                progress_bar.progress(current_progress / len(pending_idxs))
                time.sleep(10)
                
            except Exception as e:
                if any(x in str(e).lower() for x in ["quota", "429", "resource"]):
                    st.warning(f"Quota hit on key #{key_idx+1}. Rotating to next key...")
                    time.sleep(2) # Short pause for UI visibility
                    key_idx = (key_idx + 1) % len(api_keys)
                    # If we've looped back to the first key, then we might want a longer wait, 
                    # but per user request, we are removing the 65s between rotations.
                else:
                    st.error(f"Error: {e}")
                    break
        
        status_area.success("Translation complete!")
        time.sleep(1)
        st.rerun()

# --- Build & Download (BEFORE Displaying Blocks) ---
if st.button("📦 Build & Download SRT"):
    out = []
    for s in st.session_state.subs:
        combined = f"{s['arabic']}\n" if s['arabic'] else ""
        combined += "\n".join(s["english_lines"])
        out.append(f"{s['index']}\n{s['start']} --> {s['end']}\n{combined}")
    final_srt = "\n\n".join(out)
    st.text_area("Final Preview", final_srt, height=200)
    st.download_button("Click to Download", final_srt, file_name="translated.srt")

# --- Display and Edit Blocks ---
st.write("### Review Blocks")
for i, s in enumerate(st.session_state.subs):
    with st.container():
        st.markdown(f"<div class='subtitle-block'><div class='block-heading'>Block {s['index']}</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.subs[i]["start"] = st.text_input("Start", s["start"], key=f"start_{i}")
            st.session_state.subs[i]["end"] = st.text_input("End", s["end"], key=f"end_{i}")
        with c2:
            st.session_state.subs[i]["arabic"] = st.text_area("Arabic", s["arabic"], key=f"ar_{i}", height=80)
            st.session_state.subs[i]["english_lines"] = st.text_area("English", "\n".join(s["english_lines"]), key=f"en_{i}", height=80).splitlines()
        st.markdown("</div>", unsafe_allow_html=True)
