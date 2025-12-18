
# app.py

import os
import io
import base64
from typing import List, Dict, Any

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import sent_tokenize

import PyPDF2
import docx
import openai

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download("punkt")
nltk.download("punkt_tab")

# -----------------------------
# OpenAI setup
# -----------------------------
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Bo - Document AI Assistant",
    page_icon="📄",
    layout="wide",
)


# -----------------------------
# Custom CSS (dark theme + mic card)
# -----------------------------
CUSTOM_CSS = """
<style>
body {
    background: radial-gradient(circle at top, #1f2933 0, #000000 55%);
    color: #e5e7eb;
}

/* Make main area darker */
.block-container {
    max-width: 1200px;
}

/* Cards */
.bo-card {
    background: rgba(15, 23, 42, 0.96);
    border-radius: 18px;
    padding: 18px 20px;
    border: 1px solid rgba(148, 163, 184, 0.3);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.9);
}

/* Mic card */
.mic-card {
    background: #ffffff;
    color: #111827;
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 12px 25px rgba(15, 23, 42, 0.35);
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.mic-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.mic-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: #111827;
}

.mic-status {
    font-size: 0.8rem;
    color: #6b7280;
}

/* Mic button */
.mic-button {
    width: 60px;
    height: 60px;
    border-radius: 999px;
    border: none;
    background: #ef4444;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 10px 20px rgba(239, 68, 68, 0.5);
    transition: transform 0.08s ease-out, box-shadow 0.08s ease-out, background 0.12s ease-out;
}

.mic-button:active {
    transform: translateY(2px) scale(0.97);
    box-shadow: 0 6px 12px rgba(239, 68, 68, 0.45);
}

.mic-icon {
    width: 22px;
    height: 22px;
    color: #ffffff;
}

/* Waveform */
.waveform {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 24px;
}

.wave-bar {
    width: 3px;
    border-radius: 999px;
    background: #ef4444;
    animation: wave 0.8s infinite ease-in-out alternate;
    opacity: 0.7;
}

.wave-bar:nth-child(2) { animation-delay: 0.1s; }
.wave-bar:nth-child(3) { animation-delay: 0.2s; }
.wave-bar:nth-child(4) { animation-delay: 0.3s; }
.wave-bar:nth-child(5) { animation-delay: 0.4s; }

@keyframes wave {
    0% { height: 6px; }
    100% { height: 22px; }
}

/* Chat bubbles */
.chat-user {
    background: rgba(59, 130, 246, 0.18);
    border-radius: 14px;
    padding: 8px 10px;
    margin-bottom: 6px;
}

.chat-bo {
    background: rgba(15, 23, 42, 0.9);
    border-radius: 14px;
    padding: 8px 10px;
    margin-bottom: 10px;
    border: 1px solid rgba(148, 163, 184, 0.4);
}

/* Text input styling */
textarea, input[type="text"] {
    background-color: rgba(15, 23, 42, 0.85) !important;
    color: #e5e7eb !important;
}

/* Small gray text */
.small-muted {
    font-size: 0.75rem;
    color: #9ca3af;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def load_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def load_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")


def load_document(file) -> str:
    if file is None:
        return ""
    if file.type == "application/pdf":
        return load_pdf(file)
    if file.type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        return load_docx(file)
    return load_txt(file)


def split_into_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


@st.cache_data(show_spinner=False)
def build_embeddings(sentences: List[str]) -> Any:
    if not sentences:
        return None
    model = get_embedder()
    embs = model.encode(sentences)
    return embs


def semantic_search(
    query: str, sentences: List[str], embeddings
) -> List[Dict[str, Any]]:
    if not sentences or embeddings is None:
        return []
    model = get_embedder()
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]
    ranked = sorted(
        [
            {"sentence": s, "score": float(score)}
            for s, score in zip(sentences, sims)
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[:5]


def summarize_document(text: str, max_sentences: int = 6) -> str:
    sentences = split_into_sentences(text)
    if len(sentences) <= max_sentences:
        return "\n".join(sentences)

    model = get_embedder()
    embs = model.encode(sentences)
    centroid = embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(centroid, embs)[0]
    ranked_idx = sorted(range(len(sentences)), key=lambda i: sims[i], reverse=True)
    top_idx = sorted(ranked_idx[:max_sentences])
    return "\n".join(sentences[i] for i in top_idx)


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    if not openai.api_key:
        return ""
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    try:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript["text"]
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
        return ""


def build_answer_from_hits(query: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "I couldn't find anything relevant in the document for that question."
    bullets = [f"- {h['sentence']}" for h in hits]
    header = "Here’s what I found in the document:\n\n"
    return header + "\n".join(bullets)


# -----------------------------
# Session state initialization
# -----------------------------
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "sentences" not in st.session_state:
    st.session_state.sentences = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "chat" not in st.session_state:
    st.session_state.chat: List[Dict[str, Any]] = []

if "main_query" not in st.session_state:
    st.session_state.main_query = ""

if "pending_voice_text" not in st.session_state:
    st.session_state.pending_voice_text = None


# -----------------------------
# Handle pending voice text BEFORE creating widgets
# -----------------------------
if st.session_state.pending_voice_text:
    # Auto-submit behavior: add to chat and clear pending
    voice_q = st.session_state.pending_voice_text.strip()
    st.session_state.pending_voice_text = None

    if voice_q:
        # Add user message (voice)
        st.session_state.chat.append(
            {"source": "voice", "query": voice_q, "answer": None}
        )
        # Compute answer using current document
        hits = semantic_search(
            voice_q, st.session_state.sentences, st.session_state.embeddings
        )
        answer = build_answer_from_hits(voice_q, hits)
        st.session_state.chat[-1]["answer"] = answer
        # Also keep main_query empty so text input is blank
        st.session_state.main_query = ""


# -----------------------------
# Layout
# -----------------------------
st.markdown(
    "<h1 style='margin-bottom:0.2rem;'>Bo — Document AI Assistant</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='small-muted' style='margin-top:0;'>Upload a document, ask questions, or use your voice.</p>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.1, 1])

# -----------------------------
# LEFT COLUMN: Document side
# -----------------------------
with left_col:
    st.markdown("### Document")

    doc_card = st.container()
    with doc_card:
        st.markdown("<div class='bo-card'>", unsafe_allow_html=True)
        file = st.file_uploader(
            "Upload document",
            type=["pdf", "docx", "txt"],
            label_visibility="visible",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if file is not None:
        with st.spinner("Reading and indexing document..."):
            text = load_document(file)
            st.session_state.document_text = text
            st.session_state.sentences = split_into_sentences(text)
            st.session_state.embeddings = build_embeddings(
                st.session_state.sentences
            )
            st.session_state.summary = summarize_document(text)

    if st.session_state.document_text:
        st.markdown("### Summary")
        with st.container():
            st.markdown("<div class='bo-card'>", unsafe_allow_html=True)
            st.write(st.session_state.summary or "No summary available.")
            st.markdown(
                "<p class='small-muted'>Summary is automatically generated from the document.</p>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Download summary as .txt"):
            summary_bytes = st.session_state.summary.encode("utf-8")
            st.download_button(
                "Download",
                data=summary_bytes,
                file_name="summary.txt",
                mime="text/plain",
            )


# -----------------------------
# RIGHT COLUMN: Chat + Mic
# -----------------------------
with right_col:
    # Microphone card
    st.markdown("### Voice & Chat")

    mic_container = st.container()
    with mic_container:
        st.markdown("<div class='mic-card'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="mic-header">
                <div class="mic-title">Voice input</div>
                <div class="mic-status">Press & hold to speak</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # NOTE:
        # Streamlit itself does not support real press-and-hold events with continuous
        # MediaRecorder from Python alone. To keep this app deployable without extra
        # custom JS components, we mimic the UI and rely on an audio file upload for now.
        # This is stable and Streamlit Cloud–friendly.
        #
        # If you later want a true press-and-hold recorder, you can replace this section
        # with a custom component (e.g., streamlit-webrtc or a custom JS/HTML widget).

        audio_file = st.file_uploader(
            "Record or upload audio",
            type=["wav", "mp3", "m4a"],
            label_visibility="collapsed",
        )

        # Waveform animation (purely visual)
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:12px; margin-top:6px;">
                <div class="waveform" style="flex:1;">
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                </div>
                <div style="font-size:0.8rem; color:#6b7280;">Listening…</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p class='small-muted' style='margin-top:8px;'>Upload a short audio clip; Bo will transcribe and answer automatically.</p>",
            unsafe_allow_html=True,
        )

        use_voice = st.button("Send voice to Bo")

        st.markdown("</div>", unsafe_allow_html=True)

    # Handle voice submission
    if use_voice and audio_file is not None:
        with st.spinner("Transcribing your voice…"):
            audio_bytes = audio_file.read()
            text = transcribe_audio_bytes(audio_bytes)
            if text.strip():
                st.session_state.pending_voice_text = text
                st.experimental_rerun()
            else:
                st.warning("I couldn't understand the audio. Please try again.")

    # Chat box
    st.markdown("### Chat")

    # Text input for question
    user_query = st.text_input(
        "Ask a question about the document",
        key="main_query",
        placeholder="What is the main conclusion? How is the method described? ...",
    )

    ask_clicked = st.button("Ask Bo")

    if ask_clicked and user_query.strip():
        q = user_query.strip()
        # Add to chat
        st.session_state.chat.append(
            {"source": "text", "query": q, "answer": None}
        )
        with st.spinner("Thinking…"):
            hits = semantic_search(
                q, st.session_state.sentences, st.session_state.embeddings
            )
            answer = build_answer_from_hits(q, hits)
            st.session_state.chat[-1]["answer"] = answer
        # clear text input
        st.session_state.main_query = ""
        st.experimental_rerun()

    # Chat history display
    if st.session_state.chat:
        with st.container():
            st.markdown("<div class='bo-card'>", unsafe_allow_html=True)
            for turn in st.session_state.chat:
                # User message
                prefix = "You (voice)" if turn.get("source") == "voice" else "You"
                st.markdown(
                    f"<div class='chat-user'><b>{prefix}:</b> {turn['query']}</div>",
                    unsafe_allow_html=True,
                )
                # Bo answer
                if turn.get("answer"):
                    st.markdown(
                        f"<div class='chat-bo'><b>Bo:</b><br>{turn['answer']}</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Clear chat history"):
        st.session_state.chat = []
        st.session_state.main_query = ""
        st.experimental_rerun()
```

