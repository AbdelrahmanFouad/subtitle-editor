import streamlit as st
import re
import chardet

st.set_page_config(layout="wide")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    body {
        scroll-behavior: smooth;
    }
    .subtitle-block {
        padding: 4px;
        margin-bottom: 4px;
        border: 1px solid #ddd;
        border-radius: 4px;
        background-color: #f9f9f9;
        font-size: 0.9rem;
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
    """,
    unsafe_allow_html=True
)

# --- Helper Functions ---
def parse_srt(srt_content: str):
    raw_blocks = re.split(r'\n\s*\n', srt_content.strip())
    subtitles = []
    for block in raw_blocks:
        lines = block.splitlines()
        if len(lines) >= 3:
            index = lines[0].strip()
            times = lines[1].strip()
            if "-->" in times:
                start, end = [t.strip() for t in times.split("-->")]
                text = "\n".join(lines[2:])
                subtitles.append({
                    "index": index,
                    "start": start,
                    "end": end,
                    "text": text,
                    "arabic": "",
                    "english": text  # Default to original as English
                })
    return subtitles

def build_srt(subtitles):
    output = []
    for sub in subtitles:
        combined_text = f"{sub.get('arabic', '').strip()}\n{sub.get('english', '').strip()}"
        block = f"{sub['index']}\n{sub['start']} --> {sub['end']}\n{combined_text}"
        output.append(block)
    return "\n\n".join(output)

def autodetect_decode(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    detected = chardet.detect(raw_bytes)
    encoding = detected.get('encoding', 'utf-8')
    try:
        decoded_text = raw_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        st.error(f"Failed to decode file with encoding {encoding}. Please ensure the file is a valid SRT file.")
        decoded_text = ""
    return decoded_text

# --- Main Application ---
st.title("SRT Editor with Arabic + English and Batch Update")

uploaded_file = st.file_uploader("Upload an SRT file", type="srt")

if uploaded_file:
    srt_text = autodetect_decode(uploaded_file)
    
    if srt_text:
        subtitles = parse_srt(srt_text)
        
        st.write("### Edit Subtitle Blocks")
        edited_subtitles = []
        for i, subtitle in enumerate(subtitles):
            st.markdown("<div class='subtitle-block'>", unsafe_allow_html=True)
            st.markdown(f"<div class='block-heading'>Subtitle Block {subtitle['index']}</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                index_val = st.text_input("Index", value=subtitle['index'], key=f"index_{i}")
                start_val = st.text_input("Start Time", value=subtitle['start'], key=f"start_{i}")
                end_val = st.text_input("End Time", value=subtitle['end'], key=f"end_{i}")
            with col2:
                arabic_val = st.text_area("Arabic Text", value=subtitle.get("arabic", ""), height=100, key=f"arabic_{i}")
                english_val = st.text_area("English Text", value=subtitle.get("english", subtitle['text']), height=100, key=f"english_{i}")

            edited_subtitles.append({
                "index": index_val,
                "start": start_val,
                "end": end_val,
                "arabic": arabic_val,
                "english": english_val
            })

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Apply Changes"):
            final_content = build_srt(edited_subtitles)
            st.session_state.final_content = final_content

        if "final_content" in st.session_state:
            st.write("### Final Updated SRT Content (Editable)")
            final_content = st.text_area("SRT Content", value=st.session_state.final_content, height=300)
            st.session_state.final_content = final_content

            st.download_button(
                label="Download Updated SRT",
                data=st.session_state.final_content,
                file_name="updated_subtitles.srt",
                mime="text/plain"
            )
else:
    st.info("Upload an SRT file to get started.")
