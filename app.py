import os
import tempfile
import streamlit as st
import whisper

from graph.graph_builder  import build_graph
from multimodal.ocr       import run_ocr
from multimodal.audio     import run_asr
from memory.memory_store  import save_memory


# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="JEE Math Mentor",
    page_icon="📐",
    layout="centered"
)


# ─────────────────────────────────────────────────────────────
# Cache heavy resources — load once, reuse across reruns
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_graph():
    return build_graph()

@st.cache_resource
def get_whisper_model():
    return whisper.load_model("base")


# ─────────────────────────────────────────────────────────────
# Build initial state for the pipeline
# ─────────────────────────────────────────────────────────────

def make_initial_state(input_text: str, input_mode: str, raw_input: str) -> dict:
    return {
        "input_mode"        : input_mode,
        "raw_input"         : raw_input,
        "input_text"        : input_text,
        "parsed_problem"    : {},
        "topic"             : "",
        "retrieved_docs"    : [],
        "sources"           : [],
        "solution"          : "",
        "is_correct"        : False,
        "confidence"        : 0.0,
        "explanation"       : "",
        "needs_human_review": False,
        "hitl_reason"       : "",
        "agent_trace"       : [],
        "feedback"          : "",
        "feedback_comment"  : ""
    }


# ─────────────────────────────────────────────────────────────
# UI — Header
# ─────────────────────────────────────────────────────────────

st.title("📐 JEE Math Mentor")
st.caption("Powered by multi-agent AI — Parser → Router → Solver → Verifier → Explainer")
st.divider()


# ─────────────────────────────────────────────────────────────
# UI — Input mode selector
# ─────────────────────────────────────────────────────────────

input_mode = st.radio(
    label="Select input mode",
    options=["✏️ Text", "🖼️ Image", "🎙️ Audio"],
    horizontal=True
)

# Clear previous result when input mode changes
if "last_input_mode" not in st.session_state:
    st.session_state["last_input_mode"] = input_mode

if st.session_state["last_input_mode"] != input_mode:
    st.session_state.pop("result", None)
    st.session_state["last_input_mode"] = input_mode

st.divider()

problem_text      = ""
raw_input         = ""
hitl_from_input   = False
hitl_reason_input = ""


# ─────────────────────────────────────────────────────────────
# UI — Text Input
# ─────────────────────────────────────────────────────────────

if input_mode == "✏️ Text":

    problem_text = st.text_area(
        label="Type your JEE problem",
        placeholder="e.g. Solve x**2 - 5*x + 6 = 0",
        height=100,
        label_visibility="collapsed"
    )
    raw_input = problem_text


# ─────────────────────────────────────────────────────────────
# UI — Image Input
# ─────────────────────────────────────────────────────────────

elif input_mode == "🖼️ Image":

    uploaded_image = st.file_uploader(
        label="Upload an image of your math problem",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_image is not None:

        st.image(uploaded_image, caption="Uploaded image", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_image.read())
            tmp_path = tmp.name

        with st.spinner("Extracting text from image..."):
            ocr_result = run_ocr(tmp_path)

        os.unlink(tmp_path)

        st.caption(f"OCR confidence: {round(ocr_result['confidence'] * 100)}%")

        if ocr_result["needs_hitl"]:
            st.warning("⚠️ OCR confidence is low. Please review and correct the extracted text below.")
            hitl_from_input   = True
            hitl_reason_input = f"OCR confidence low ({round(ocr_result['confidence'] * 100)}%). Manual review required."

        problem_text = st.text_area(
            label="Extracted text (edit if needed)",
            value=ocr_result["text"],
            height=100
        )
        raw_input = ocr_result["text"]


# ─────────────────────────────────────────────────────────────
# UI — Audio Input
# ─────────────────────────────────────────────────────────────

elif input_mode == "🎙️ Audio":

    uploaded_audio = st.file_uploader(
        label="Upload an audio file of your math question",
        type=["mp3", "wav", "m4a", "ogg", "webm"],
        label_visibility="collapsed"
    )

    if uploaded_audio is not None:

        st.audio(uploaded_audio)

        suffix = "." + uploaded_audio.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_audio.read())
            tmp_path = tmp.name

        with st.spinner("Transcribing audio..."):
            whisper_model = get_whisper_model()
            asr_result    = run_asr(tmp_path, whisper_model)

        os.unlink(tmp_path)

        st.caption(f"Detected language: {asr_result['language']} | Confidence: {round(asr_result['confidence'] * 100)}%")

        if asr_result["raw_transcript"] != asr_result["transcript"]:
            with st.expander("🔍 Raw transcript (before math phrase cleaning)"):
                st.text(asr_result["raw_transcript"])

        if asr_result["needs_hitl"]:
            st.warning(
                "⚠️ Audio transcription confidence is low. Please review and correct the transcript below.",
                icon="⚠️"
            )
            hitl_from_input   = True
            hitl_reason_input = f"ASR confidence low ({round(asr_result['confidence'] * 100)}%). Manual review required."

        problem_text = st.text_area(
            label="Transcript (edit if needed)",
            value=asr_result["transcript"],
            height=100
        )
        raw_input = asr_result["raw_transcript"]


# ─────────────────────────────────────────────────────────────
# UI — Solve button
# ─────────────────────────────────────────────────────────────

st.divider()
solve_clicked = st.button("🧠 Solve", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# UI — Run pipeline on button click
# ─────────────────────────────────────────────────────────────

st.session_state.pop("result", None)
st.session_state["memory_saved"] = False
if solve_clicked:

    if not problem_text.strip():
        st.warning("Please provide a math problem first.")

    else:
        with st.spinner("Running multi-agent pipeline..."):

            graph = get_graph()
            mode  = input_mode.split()[-1].lower()
            state = make_initial_state(problem_text.strip(), mode, raw_input)

            if hitl_from_input:
                state["needs_human_review"] = True
                state["hitl_reason"]        = hitl_reason_input

            result = graph.invoke(state)

        # ── Save to memory immediately after pipeline finishes ──
        # Saved without feedback first — feedback updates it later
        save_memory(result)

        st.session_state["result"]       = result
        st.session_state["memory_saved"] = False  # tracks if feedback update saved


# ─────────────────────────────────────────────────────────────
# UI — Show results
# ─────────────────────────────────────────────────────────────

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    # ── HITL Warning ──────────────────────────────────────────
    if result["needs_human_review"]:
        st.warning(f"⚠️ **Human Review Suggested**\n\n{result['hitl_reason']}")

    # ── Topic + Confidence ────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Topic Detected",
            value=result["topic"].replace("_", " ").title() if result["topic"] else "Unknown"
        )

    with col2:
        confidence_pct = round(result["confidence"] * 100)
        st.metric(
            label="Verifier Confidence",
            value=f"{confidence_pct}%",
            delta="✅ Verified" if result["is_correct"] else "❌ Not verified"
        )

    st.divider()

    # ── Explanation ───────────────────────────────────────────
    st.subheader("📖 Explanation")
    st.markdown(result["explanation"])

    # ── Raw solution (collapsed) ──────────────────────────────
    with st.expander("🔍 View raw solution"):
        st.markdown(result["solution"])

    st.divider()

    # ── Retrieved Sources ─────────────────────────────────────
    st.subheader("📚 Sources from Knowledge Base")
    if result["sources"]:
        for source in result["sources"]:
            st.markdown(f"- `{source}`")
    else:
        st.caption("No sources retrieved from knowledge base.")

    st.divider()

    # ── Feedback ──────────────────────────────────────────────
    st.subheader("Was this helpful?")

    col3, col4 = st.columns(2)

    with col3:
        if st.button("✅ Correct", use_container_width=True):
            st.session_state["result"]["feedback"] = "correct"

            # Re-save to memory with feedback attached
            if not st.session_state.get("memory_saved"):
                save_memory(st.session_state["result"])
                st.session_state["memory_saved"] = True

            st.success("Thanks for your feedback!")

    with col4:
        if st.button("❌ Incorrect", use_container_width=True):
            st.session_state["result"]["feedback"] = "incorrect"

    # Comment box appears only after incorrect clicked
    if st.session_state["result"].get("feedback") == "incorrect":
        comment = st.text_input("What was wrong? (optional)")
        if comment:
            st.session_state["result"]["feedback_comment"] = comment

            # Re-save to memory with feedback + comment attached
            if not st.session_state.get("memory_saved"):
                save_memory(st.session_state["result"])
                st.session_state["memory_saved"] = True

            st.info("Feedback saved. This will help improve future answers.")

    st.divider()

    # ── Agent Trace ───────────────────────────────────────────
    with st.expander("🔎 Agent Trace"):
        for step in result["agent_trace"]:
            st.markdown(f"→ `{step}`")