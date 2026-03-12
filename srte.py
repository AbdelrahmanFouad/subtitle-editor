import os
import re
import time
import chardet
import streamlit as st
import google.generai as genai
from google.api_core import exceptions

# --- Configure Gemini API ---
genai.configure(api_key=st.secrets["gemini_api_key"])
model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite")

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
    """
    Improved parser:
    - Splits the file into blocks.
    - For each block, splits the lines.
    - Each line from the body is inspected:
          * If it contains any Arabic character (Unicode range: U+0600 to U+06FF),
            it's added to the Arabic lines list.
          * Otherwise, it's added to the English lines list.
    - The Arabic text is then the concatenation (separated by newlines) of all Arabic lines.
    """
    blocks = re.split(r'
\s*
', text.strip())
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
        arabic = "
".join(arabic_lines)
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
        combined = f"{s['arabic']}
" if s['arabic'] else ""
        combined += "
".join(s["english_lines"])
        out.append(f"{s['index']}
{s['start']} --> {s['end']}
{combined}")
    return "

".join(out)

def translate_batch(block_texts: list[str]) -> list[str]:
    prompt = (
        "You are a professional translator. Translate the following English subtitle blocks into Arabic.
"
        "Keep the same line breaks for each block.
"
        "For each block, output ONLY the Arabic lines, without titles or extra text, just pure translation output.

"
    )
    for i, blk in enumerate(block_texts, 1):
        prompt += f"Block {i}:
{blk}

"
    chat = model.start_chat(history=[])
    resp = chat.send_message(prompt, request_options={"timeout": 120})
    text = resp.text.strip()
    # Return translations as a list (one per block)
    return [b.strip() for b in text.split("

") if b.strip()]

# --- UI: File Upload ---
uploader = st.file_uploader("Upload SRT file (Arabic+English or English-only)", type="srt")
if not uploader:
    st.info("Please upload your SRT file.")
    st.stop()

text = autodetect_decode(uploader)
subs = parse_srt(text)
st.session_state.setdefault("subs", subs)

# --- Translation: Entire File in Batches ---
if st.button("Translate Entire File (in batches)"):
    api_keys = [
        st.secrets["gemini_api_key"],
        st.secrets["gemini_api_2"],
        st.secrets["gemini_api_3"],
        st.secrets["gemini_api_4"],
    ]
    key_index = 0
    # Configure with the first key
    genai.configure(api_key=api_keys[key_index])
    st.info(f"Starting translation with API key #{key_index + 1}")

    pending_idxs = [i for i, s in enumerate(st.session_state.subs) if not s["arabic"].strip()]
    total_pending = len(pending_idxs)
    batch_size = 3
    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_start = 0

    while batch_start < total_pending:
        current_batch_indices = pending_idxs[batch_start : batch_start + batch_size]
        texts_to_translate = ["
".join(st.session_state.subs[i]["english_lines"]) for i in current_batch_indices]
        block_numbers = [st.session_state.subs[i]['index'] for i in current_batch_indices]

        try:
            status_text.info(f"Translating blocks: {', '.join(block_numbers)} (using key #{key_index + 1})")
            
            # This call will raise an exception on API failure
            translations = translate_batch(texts_to_translate)

            # --- Success Case ---
            for i, idx in enumerate(current_batch_indices):
                if i < len(translations):
                    st.session_state.subs[idx]["arabic"] = translations[i]
            
            # Move to the next batch
            batch_start += batch_size
            progress = min(1.0, batch_start / total_pending)
            progress_bar.progress(progress)
            
            if batch_start < total_pending:
                time.sleep(7) # Keep the delay between successful calls

        except exceptions.ResourceExhausted as e:
            # --- Quota Failure Case ---
            st.warning(f"Quota error on key #{key_index + 1}. Rotating to next key...")
            time.sleep(1)
            
            key_index += 1
            
            if key_index >= len(api_keys):
                st.error("All API keys have been exhausted. Please check your quota and billing details, then try again later. Aborting translation.")
                break # Exit the while loop
            
            # Configure with new key and retry the same batch
            genai.configure(api_key=api_keys[key_index])
            st.info(f"Retrying with new API key #{key_index + 1}...")
            time.sleep(1.5)
        
        except Exception as e:
            # --- Other Failure Case ---
            st.error(f"An unexpected error occurred: {e}. Aborting translation.")
            break

    if batch_start >= total_pending and total_pending > 0:
        progress_bar.progress(1.0)
        status_text.success("Translation complete!")
    elif total_pending == 0:
        st.info("No untranslated blocks found.")

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
        english_text = st.text_area("English", "
".join(s["english_lines"]), key=f"english_{i}", height=80)
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
