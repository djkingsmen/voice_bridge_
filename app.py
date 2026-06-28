import streamlit as st
import time
from io import BytesIO
from gtts import gTTS

from agent import (
    has_api_key,
    translate_text,
    answer_followup_question,
    explain_text,
    transcribe_audio,
    build_segments_for_audio,
)

st.set_page_config(page_title="VoiceBridge", page_icon="🎙️", layout="wide")

IMAGE_LEFT  = "https://preview.redd.it/two-sides-of-the-same-story-v0-h6a7h75jx99h1.jpeg?auto=webp&s=adb1c0f78a267666b3ab6323bef648e88e3a57cd"
IMAGE_RIGHT = "https://picfiles.alphacoders.com/653/thumb-1920-653981.png"

defaults = {
    "active_mode": None,
    "a_result": None, "a_followup_answer": None,
    "b_result": None, "b_followup_answer": None, "b_explanation": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

LANG_OPTIONS = {
    "Auto Detect": None,
    "Tamil (தமிழ்)": "ta", "Hindi (हिन्दी)": "hi",
    "Telugu (తెలుగు)": "te", "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml", "Bengali (বাংলা)": "bn", "English": "en",
}
LANG_NAMES = {v: k.split(" (")[0] for k, v in LANG_OPTIONS.items() if v}
LANG_NAMES["en"] = "English"

active_mode = st.session_state.active_mode

if active_mode == "A":
    bg_css = f"""
        background-image: linear-gradient(135deg,rgba(91,110,225,0.75),rgba(70,82,156,0.85)), url('{IMAGE_LEFT}');
        background-size: cover, cover;
        background-position: center, center;
        background-repeat: no-repeat, no-repeat;
        background-attachment: fixed, fixed;"""
elif active_mode == "B":
    bg_css = f"""
        background-image: linear-gradient(135deg,rgba(123,63,160,0.75),rgba(83,42,107,0.85)), url('{IMAGE_RIGHT}');
        background-size: cover, cover;
        background-position: center, center;
        background-repeat: no-repeat, no-repeat;
        background-attachment: fixed, fixed;"""
else:
    bg_css = "background: #111 !important;"

st.markdown(f"""
<style>
header[data-testid="stHeader"] {{ display:none!important; }}
#MainMenu {{ display:none!important; }}
footer {{ display:none!important; }}
.stDeployButton {{ display:none!important; }}
[data-testid="stToolbar"] {{ display:none!important; }}

.stApp {{ {bg_css} }}

#vb-home-bg {{
    position:fixed; inset:0; z-index:0; pointer-events:none; display:flex;
}}
#vb-home-bg .hl {{ flex:1; background:url('{IMAGE_LEFT}') center/cover no-repeat; }}
#vb-home-bg .hr {{ flex:1; background:url('{IMAGE_RIGHT}') center/cover no-repeat; }}
#vb-home-bg .ov {{ position:absolute; inset:0; background:rgba(0,0,0,0.48); }}

.vb-title {{
    text-align:center; color:#fff; font-weight:800; font-size:1.6rem;
    letter-spacing:1px; padding-top:0.6rem; text-shadow:0 2px 8px rgba(0,0,0,0.7);
}}
.vb-card {{
    background:rgba(255,255,255,0.96); border-radius:8px;
    padding:1.6rem 1.6rem 1.8rem; box-shadow:0 18px 30px rgba(0,0,0,0.35); margin-bottom:1.4rem;
}}
.vb-result-box {{ background:#f4f4f4; border-radius:6px; padding:1rem 1.2rem; margin-top:0.6rem; margin-bottom:0.8rem; }}
.vb-label {{ text-transform:uppercase; font-size:0.7rem; letter-spacing:1.5px; color:#888; font-weight:700; }}
.vb-original-text  {{ color:#555; font-size:0.95rem; margin:0.2rem 0 0.7rem 0; }}
.vb-translated-text {{ color:#111!important; font-weight:600; font-size:1.1rem; margin-bottom:0.3rem; }}
.vb-roman-text {{ color:#777; font-style:italic; font-size:0.9rem; }}
.vb-tag {{
    display:inline-block; background:#111; color:#fff; border-radius:20px;
    padding:3px 14px; font-size:0.72rem; letter-spacing:1px;
    text-transform:uppercase; margin-right:6px; margin-bottom:0.6rem;
}}
.vb-hero {{ text-align:center; padding:3rem 1rem 2rem; }}
.vb-hero h1 {{ color:#fff; font-size:2.8rem; font-weight:900; text-shadow:0 3px 12px rgba(0,0,0,0.8); margin-bottom:0.4rem; }}
.vb-hero p  {{ color:rgba(255,255,255,0.9); font-size:1.15rem; text-shadow:0 2px 6px rgba(0,0,0,0.7); }}
.vb-divider {{ border:none; border-top:1px solid rgba(255,255,255,0.25); margin:0.5rem 0 1.5rem; }}
div[data-testid="stMetric"] {{ background:#f4f4f4; border-radius:6px; padding:0.5rem 0.3rem; }}
div[data-testid="stMetric"] label {{ color:#555!important; }}
div[data-testid="stMetricValue"] {{ color:#111!important; }}
.stButton button {{
    background:#111!important; color:#fff!important;
    border-radius:5px!important; font-weight:600!important; border:none!important;
}}
.stButton button:hover {{ background:#333!important; }}
</style>
""", unsafe_allow_html=True)

# Inject split background div only on home
if active_mode is None:
    st.markdown("""
    <div id="vb-home-bg">
        <div class="hl"></div>
        <div class="hr"></div>
        <div class="ov"></div>
    </div>
    """, unsafe_allow_html=True)

if not has_api_key():
    st.error("GROQ_API_KEY is not set. Add it to your environment and restart.")
    st.stop()

# ── Nav bar ──────────────────────────────────────────────────────────
st.markdown('<div class="vb-title">🎙️ VoiceBridge</div>', unsafe_allow_html=True)

if active_mode is None:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⌨️ Text Translator", use_container_width=True, key="nav_a"):
            st.session_state.active_mode = "A"; st.rerun()
    with c2:
        if st.button("🎙️ Voice Translator", use_container_width=True, key="nav_b"):
            st.session_state.active_mode = "B"; st.rerun()
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🏠 Home", use_container_width=True, key="nav_home"):
            st.session_state.active_mode = None; st.rerun()
    with c2:
        if st.button("⌨️ Text Translator", use_container_width=True, key="nav_a"):
            st.session_state.active_mode = "A"; st.rerun()
    with c3:
        if st.button("🎙️ Voice Translator", use_container_width=True, key="nav_b"):
            st.session_state.active_mode = "B"; st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────
def render_result_card(result):
    st.markdown('<div class="vb-result-box">', unsafe_allow_html=True)
    st.markdown(f'<span class="vb-tag">{result["source_lang"].upper()} → {result["target_lang"].upper()}</span>', unsafe_allow_html=True)
    st.markdown('<div class="vb-label">Original</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="vb-original-text">{result["original_text"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="vb-label">Converted</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="vb-translated-text">{result["translated_text"]}</div>', unsafe_allow_html=True)
    if result.get("romanized_text"):
        st.markdown(f'<div class="vb-roman-text">{result["romanized_text"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_metrics(result):
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected Source", LANG_NAMES.get(result["source_lang"], result["source_lang"].upper()))
    c2.metric("Converted Into",  LANG_NAMES.get(result["target_lang"], result["target_lang"].upper()))
    c3.metric("Word Count", len(result["translated_text"].split()))

def synthesize_audio(result):
    segments = build_segments_for_audio(result["translated_text"], result["target_lang"])
    buf = BytesIO()
    try:
        with st.spinner("Synthesizing audio..."):
            for seg in segments:
                sl = seg.get("lang", "en")
                if sl not in ["en","ta","hi","te","kn","ml","bn","mr","gu","pa","ur"]: sl = "en"
                try:    gTTS(text=seg["text"], lang=sl,   slow=False).write_to_fp(buf)
                except: gTTS(text=seg["text"], lang="en", slow=False).write_to_fp(buf)
        buf.seek(0); st.audio(buf, format="audio/mp3")
    except Exception as e: st.warning(f"Audio error: {e}")

def render_karaoke(text):
    words = text.split()
    if not words: return
    ph, hl = st.empty(), []
    for w in words:
        hl.append(f"<span style='background:#111;color:#fff;padding:2px 6px;border-radius:3px;margin-right:3px;'>{w}</span>")
        ph.markdown(" ".join(hl) + " " + " ".join(words[len(hl):]), unsafe_allow_html=True)
        time.sleep(0.18)

def render_exports(result, pfx):
    c1, c2 = st.columns(2)
    with c1: st.download_button("📥 Download Transcript", data=result["translated_text"], file_name="transcript.txt", mime="text/plain", use_container_width=True, key=f"{pfx}_dl_txt")
    with c2: st.download_button("📥 Download JSON",       data=str(result),               file_name="payload.json",    mime="application/json", use_container_width=True, key=f"{pfx}_dl_json")

def render_followup(result, pfx, state_key):
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown('<h4 style="color:#111;">Have a question about this?</h4>', unsafe_allow_html=True)
    q = st.text_input("Ask a question:", key=f"{pfx}_fq")
    if st.button("Ask", key=f"{pfx}_ask"):
        if q.strip():
            with st.spinner("Thinking..."):
                try: st.session_state[state_key] = answer_followup_question(result["translated_text"], q, result["target_lang"])
                except Exception as e: st.error(f"Error: {e}")
        else: st.warning("Type a question first.")
    ans = st.session_state.get(state_key)
    if ans:
        st.markdown('<div class="vb-result-box"><div class="vb-label">Answer</div>'
                    f'<div class="vb-translated-text">{ans}</div></div>', unsafe_allow_html=True)

# ── Screens ────────────────────────────────────────────────────────────

# HOME
if active_mode is None:
    st.markdown("""
    <div class="vb-hero">
        <h1>🎙️ VoiceBridge</h1>
        <p>Multilingual Voice &amp; Text Studio — pick a mode below</p>
    </div>
    <hr class="vb-divider">
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("""
        <div style="background:rgba(91,110,225,0.20);border:2px solid rgba(91,110,225,0.6);
                    border-radius:12px;padding:2.5rem 1.5rem;text-align:center;margin-bottom:1rem;">
            <div style="font-size:3rem;">⌨️</div>
            <h2 style="color:#fff;text-shadow:0 2px 8px rgba(0,0,0,0.8);margin:0.4rem 0;">Text Translator</h2>
            <p style="color:rgba(255,255,255,0.88);text-shadow:0 1px 4px rgba(0,0,0,0.7);">
                Type any text and instantly convert it into your chosen language with audio playback.
            </p>
        </div>""", unsafe_allow_html=True)
        if st.button("OPEN TEXT MODE →", use_container_width=True, key="land_a"):
            st.session_state.active_mode = "A"; st.rerun()

    with right:
        st.markdown("""
        <div style="background:rgba(123,63,160,0.20);border:2px solid rgba(123,63,160,0.6);
                    border-radius:12px;padding:2.5rem 1.5rem;text-align:center;margin-bottom:1rem;">
            <div style="font-size:3rem;">🎙️</div>
            <h2 style="color:#fff;text-shadow:0 2px 8px rgba(0,0,0,0.8);margin:0.4rem 0;">Voice Translator</h2>
            <p style="color:rgba(255,255,255,0.88);text-shadow:0 1px 4px rgba(0,0,0,0.7);">
                Upload or record your voice and get a live transcription with instant translation.
            </p>
        </div>""", unsafe_allow_html=True)
        if st.button("OPEN VOICE MODE →", use_container_width=True, key="land_b"):
            st.session_state.active_mode = "B"; st.rerun()

# MODE A
elif active_mode == "A":
    st.markdown('<div class="vb-card">', unsafe_allow_html=True)
    st.markdown("### ⌨️ Text Translator")
    tl = st.selectbox("Convert into:", list(LANG_OPTIONS.keys()), key="a_target")
    target_lang = LANG_OPTIONS[tl]
    txt = st.text_area("Type your text:", height=110, key="a_input")
    karaoke = st.checkbox("Show word-by-word read-along", value=False, key="a_kar")
    if st.button("Convert", key="a_convert"):
        if not txt.strip(): st.warning("Please type something first.")
        else:
            with st.spinner("Converting..."):
                try:
                    st.session_state.a_result = translate_text(txt, target_lang)
                    st.session_state.a_followup_answer = None
                except Exception as e: st.error(f"Failed: {e}")
    r = st.session_state.a_result
    if r:
        render_metrics(r); render_result_card(r); synthesize_audio(r)
        if karaoke: render_karaoke(r["translated_text"])
        render_exports(r, "a"); render_followup(r, "a", "a_followup_answer")
    st.markdown('</div>', unsafe_allow_html=True)

# MODE B
elif active_mode == "B":
    st.markdown('<div class="vb-card">', unsafe_allow_html=True)
    st.markdown("### 🎙️ Voice Translator")
    tl = st.selectbox("Convert into:", list(LANG_OPTIONS.keys()), key="b_target")
    target_lang = LANG_OPTIONS[tl]
    audio_bytes = None
    up_tab, rec_tab = st.tabs(["Upload audio", "Record audio"])
    with up_tab:
        up = st.file_uploader("Upload a voice clip:", type=["wav","mp3","m4a","ogg"])
        if up: audio_bytes = up.read(); st.audio(up)
    with rec_tab:
        if hasattr(st, "audio_input"):
            rec = st.audio_input("Record your voice:")
            if rec: audio_bytes = rec.read()
        else: st.info("Recording not supported — please upload a file.")
    karaoke = st.checkbox("Show word-by-word read-along", value=False, key="b_kar")
    if st.button("Transcribe & Convert", key="b_convert"):
        if not audio_bytes: st.warning("Upload or record audio first.")
        else:
            with st.spinner("Transcribing and converting..."):
                try:
                    st.session_state.b_result = translate_text(transcribe_audio(audio_bytes), target_lang)
                    st.session_state.b_followup_answer = None
                    st.session_state.b_explanation = None
                except Exception as e: st.error(f"Failed: {e}")
    r = st.session_state.b_result
    if r:
        render_metrics(r); render_result_card(r); synthesize_audio(r)
        if karaoke: render_karaoke(r["translated_text"])
        if st.button("💡 Explain", key="b_explain", use_container_width=True):
            with st.spinner("Explaining..."):
                try: st.session_state.b_explanation = explain_text(r["translated_text"], r["target_lang"])
                except Exception as e: st.error(f"Failed: {e}")
        exp = st.session_state.get("b_explanation")
        if exp:
            st.markdown(f'<div class="vb-result-box"><div class="vb-label">Explanation</div>'
                        f'<div class="vb-translated-text">{exp}</div></div>', unsafe_allow_html=True)
        render_exports(r, "b"); render_followup(r, "b", "b_followup_answer")
    st.markdown('</div>', unsafe_allow_html=True)