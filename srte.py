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

def translate_batch(client, model_id, block_texts):
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a professional subtitle translator. Translate English to natural Arabic. "
            "Maintain line breaks. Output format: Return each translation preceded by its index in brackets, like [1], [2], etc."
        ),
        temperature=0.3,
    )
    prompt = "Translate these blocks:\n\n"
    for i, txt in enumerate(block_texts, 1):
        prompt += f"[{i}]\n{txt}\n\n"
    
    response = client.models.generate_content(model=model_id, contents=prompt, config=config)
    raw_text = response.text.strip()
    
    results = []
    parts = re.split(r'\[\d+\]', raw_text)
    for p in parts:
        if p.strip():
            results.append(p.strip())
    return results

# --- UI: File Upload ---
uploader = st.file_uploader("Upload SRT file", type="srt")
if not uploader:
    st.info("Please upload your SRT file.")
    st.stop()

if "subs" not in st.session_state:
    text = autodetect_decode(uploader)
    st.session_state.subs = parse_srt(text)

# --- Translation: Entire File in Batches ---
if st.button("Translate Entire File (in batches)"):
    raw_keys = [st.secrets.get(k) for k in ["gemini_api_key", "gemini_api_2", "gemini_api_3", "gemini_api_4"]]
    api_keys = list(dict.fromkeys([k for k in raw_keys if k]))
    
    pending_idxs = [i for i, s in enumerate(st.session_state.subs) if not s["arabic"].strip()]
    total_pending = len(pending_idxs)
    batch_size = 12
    progress_bar = st.progress(0)
    status_text = st.empty()

    key_idx = 0
    current_idx = 0
    
    while current_idx < total_pending:
        key = api_keys[key_idx]
        client = genai.Client(api_key=key)
        
        batch_indices = pending_idxs[current_idx : current_idx + batch_size]
        texts = ["\n".join(st.session_state.subs[i]["english_lines"]) for i in batch_indices]
        
        try:
            status_text.info(f"Translating blocks: {st.session_state.subs[batch_indices[0]]['index']} to {st.session_state.subs[batch_indices[-1]]['index']} (Key #{key_idx + 1})")
            
            translations = translate_batch(client, DEFAULT_MODEL, texts)

            applied = 0
            for i, idx in enumerate(batch_indices):
                if i < len(translations):
                    st.session_state.subs[idx]["arabic"] = translations[i]
                    applied += 1
            
            current_idx += applied
            progress_bar.progress(current_idx / total_pending)
            time.sleep(10)

        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["quota", "429", "resourceexhausted"]):
                st.warning(f"Key #{key_idx + 1} limited. Waiting 65s for reset...")
                time.sleep(65)
                key_idx = (key_idx + 1) % len(api_keys)
            else:
                st.error(f"Error: {e}. Retrying in 10s...")
                time.sleep(10)

    if current_idx >= total_pending:
        progress_bar.progress(1.0)
        status_text.success("Translation complete!")

# --- Display Subtitle Blocks for Review ---
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
        "index": s["index"],
        "start": start_time,
        "end": end_time,
        "arabic": arabic_text,
        "english_lines": english_text.splitlines()
    })
    st.markdown("</div>", unsafe_allow_html=True)
st.session_state.subs = edited

# --- Build & Download Section ---
if st.button("Build & Download SRT"):
    final = build_srt(st.session_state.subs)
    st.session_state["final_srt"] = final

if "final_srt" in st.session_state:
    st.write("### Final Updated SRT Content")
    final_content = st.text_area("SRT Content", st.session_state["final_srt"], height=300)
    st.session_state["final_srt"] = final_content
    st.download_button("Download .srt", final_content, "translated.srt", "text/plain")
