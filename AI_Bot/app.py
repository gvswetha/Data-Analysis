# =============================
# AI Document Assistant — Bo (Full App)
# =============================

import os
import hashlib
import tempfile

import streamlit as st
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize

import PyPDF2
import docx

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# from audiorecorder import audiorecorder
# import openai


# ============================
# PAGE CONFIG & THEME
# ============================
st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #E6D9FF, #000000);
        color: white;
    }
    .hero {
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .upload-box {
        border: 2px dashed #C7B8FF;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        background: rgba(0, 0, 0, 0.25);
    }
    .answer-box {
        background: rgba(0,0,0,0.4);
        padding: 18px;
        border-radius: 12px;
        margin-top: 12px;
    }
    .section-title {
        margin-top: 10px;
        margin-bottom: 8px;
    }
    pre {
        white-space: pre-wrap;
        word-wrap: break-word;
        font-size: 0.9rem;
    }
    mark {
        background-color: #ffe66d;
        padding: 0 2px;
        border-radius: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# HERO
# ============================
st.markdown(
    "<div class='hero'><h1>🤖 Hi, I’m Bo</h1><h4>Your AI document assistant</h4></div>",
    unsafe_allow_html=True
)

# ============================
# HELPER FUNCTIONS
# ============================

def extract_text(file):
    """Extract raw text from PDF, DOCX, or TXT."""
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs])
    else:
        return file.read().decode("utf-8", errors="ignore")


def preprocess_sentences(text):
    """Split into sentences and filter out very short ones."""
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 40]


def summarize_text(sentences, ratio=0.15):
    """Naive extractive summary: first N sentences."""
    if not sentences:
        return "No content to summarize."
    count = max(3, int(len(sentences) * ratio))
    return " ".join(sentences[:count])


def get_context_window(text, target_sentence, window=200):
    """Return a snippet of text around a target sentence, with the sentence highlighted."""
    idx = text.find(target_sentence)
    if idx == -1:
        # Fallback: just highlight the sentence itself
        return f"<mark>{target_sentence}</mark>"

    start = max(0, idx - window)
    end = min(len(text), idx + len(target_sentence) + window)
    snippet = text[start:end]

    # Highlight the exact sentence
    snippet = snippet.replace(target_sentence, f"<mark>{target_sentence}</mark>")
    return snippet


@st.cache_resource
def load_sbert_model():
    """Load SBERT model once."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_doc_hash(text: str) -> str:
    """Stable hash for caching embeddings by document content."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@st.cache_data
def encode_sentences_cached(doc_hash, sentences):
    """Encode sentences for a specific document hash."""
    model = load_sbert_model()
    embeddings = model.encode(sentences, show_progress_bar=False)
    return embeddings


def answer_question(query, sentences, embeddings, full_text, top_k=5):
    """Return structured answer: combined text, bullets, and per-sentence context."""
    model = load_sbert_model()
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for i in top_idx:
        sent = sentences[i]
        score = sims[i]
        context = get_context_window(full_text, sent)
        results.append(
            {
                "sentence": sent,
                "score": float(score),
                "context": context
            }
        )

    combined = " ".join([r["sentence"] for r in results])
    bullets = "\n".join([f"- {r['sentence']}" for r in results])

    return {
        "combined": combined,
        "bullets": bullets,
        "results": results,
    }


# ============================
# SESSION STATE
# ============================
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts: {"query":..., "answer":...}

# ============================
# MAIN LAYOUT (TWO COLUMNS)
# ============================
col_main, col_analysis = st.columns([2, 1])

# -----------
# LEFT COLUMN
# -----------
with col_main:
    # Upload box
    st.markdown(
        "<div class='upload-box'><h4>📄 Upload your document</h4>"
        "<p>Drag & drop or upload a file to get a concise summary and smart answers.</p></div>",
        unsafe_allow_html=True,
    )
    file = st.file_uploader("", type=["pdf", "docx", "txt"], label_visibility="collapsed")

    # Chat input (text)
    user_query = st.text_input(
        "",
        placeholder="🔍 Ask me anything about the document… (or use voice from the sidebar)",
        key="main_query",
    )

    text = None
    sentences = None
    embeddings = None

    if file:
        with st.spinner("🤖 Bo is reading your document…"):
            text = extract_text(file)
            sentences = preprocess_sentences(text)

        if not sentences:
            st.error("No valid sentences found in the document.")
            st.stop()

        # Cached embeddings
        doc_hash = get_doc_hash(text)
        embeddings = encode_sentences_cached(doc_hash, sentences)

        # Summary
        summary = summarize_text(sentences, ratio=0.15)
        st.markdown(
            "<div class='answer-box'><h3 class='section-title'>📘 Document Summary</h3></div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<pre>{summary}</pre>", unsafe_allow_html=True)

        # Download summary
        st.download_button(
            "⬇️ Download Summary",
            summary,
            file_name="document_summary.txt",
            mime="text/plain",
        )

        # Suggested Questions
        st.markdown(
            "<div class='answer-box'><h3 class='section-title'>📌 Suggested Questions</h3></div>",
            unsafe_allow_html=True,
        )
        sample_questions = [
            "What is this document about?",
            "What is the objective of this document?",
            "Explain the key concepts.",
            "Describe the methodology used.",
            "What are the main conclusions?",
        ]

        sq_cols = st.columns(len(sample_questions))
        for idx, q in enumerate(sample_questions):
            with sq_cols[idx]:
                if st.button(q, key=f"sample_{idx}"):
                    st.session_state.main_query = q
                    user_query = q

        # Answer
        if user_query:
            with st.spinner("🤖 Bo is thinking…"):
                result = answer_question(user_query, sentences, embeddings, text, top_k=5)

            st.session_state.chat.append(
                {
                    "query": user_query,
                    "answer": result["combined"],
                }
            )

            st.markdown(
                "<div class='answer-box'><h3 class='section-title'>💡 Answer</h3></div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<p>{result['combined']}</p>", unsafe_allow_html=True)

            st.markdown("<h4 class='section-title'>🔍 Key Points</h4>", unsafe_allow_html=True)
            st.markdown(f"<pre>{result['bullets']}</pre>", unsafe_allow_html=True)

            # Pass result to analysis column via session_state
            st.session_state.last_result = result
            st.session_state.last_text = text
    else:
        # No document uploaded
        if user_query:
            general_response = (
                "Hi, I’m Bo 🤖. Right now I work best with uploaded documents. "
                "Upload a PDF, DOCX, or TXT and I’ll summarize it and answer questions based on its content."
            )
            st.session_state.chat.append(
                {
                    "query": user_query,
                    "answer": general_response,
                }
            )
            st.markdown(
                "<div class='answer-box'><h3 class='section-title'>🤖 Bo</h3></div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<p>{general_response}</p>", unsafe_allow_html=True)


# -----------
# RIGHT COLUMN (ANALYSIS)
# -----------
with col_analysis:
    st.markdown("<div class='answer-box'><h3 class='section-title'>✨ Document Insights</h3></div>", unsafe_allow_html=True)

    last_result = st.session_state.get("last_result", None)
    last_text = st.session_state.get("last_text", None)

    if last_result and last_text:
        st.markdown("**Highlighted context from the document:**")
        for idx, r in enumerate(last_result["results"]):
            label = f"{idx+1}. {r['sentence'][:70]}..."
            with st.expander(label):
                st.markdown(r["context"], unsafe_allow_html=True)
                st.markdown(f"<small>Relevance score: {r['score']:.3f}</small>", unsafe_allow_html=True)
    else:
        st.caption("Ask a question about a document to see highlighted context here.")


# ============================
# SIDEBAR: HISTORY + VOICE
# ============================
# --- FIX OLD CHAT FORMAT ---
cleaned_chat = []
for turn in st.session_state.chat:
    if isinstance(turn, tuple):
        # old format: (question, answer)
        cleaned_chat.append({"query": turn[0], "answer": turn[1]})
    else:
        cleaned_chat.append(turn)
st.session_state.chat = cleaned_chat

# --- Chat history ---
with st.sidebar:
    st.markdown("## 🗂️ History")

    if st.session_state.chat:
        with st.expander("Show Conversation History"):
            for turn in reversed(st.session_state.chat):
                q = turn["query"]
                a = turn["answer"]
                st.markdown(
                    f"""
                    <div style='padding:10px; background:#222; border-radius:8px; margin-bottom:8px;'>
                        <strong style='color:#fff;'>🧑 You:</strong><br>
                        <span style='color:#ccc;'>{q}</span><br><br>
                        <strong style='color:#fff;'>🤖 Bo:</strong><br>
                        <span style='color:#aaa;'>{a}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No history yet. Start by asking something.")

# # --- Voice input (Whisper API) ---
# with st.sidebar:
#     st.markdown("## 🎙️ Voice Input")
#     st.caption("Upload or record an audio file and Bo will transcribe it using Whisper.")

#     audio_file = st.file_uploader(
#         "Upload audio (WAV/MP3/M4A)", 
#         type=["wav", "mp3", "m4a"],
#         label_visibility="collapsed"
#     )

#     if audio_file is not None:
#         st.audio(audio_file, format="audio/wav")

#         try:
#             openai_api_key = os.getenv("OPENAI_API_KEY")
#             if not openai_api_key:
#                 st.error("OPENAI_API_KEY is not set in your environment.")
#             else:
#                 openai.api_key = openai_api_key

#                 # Whisper API call
#                 transcript = openai.audio.transcriptions.create(
#                     model="whisper-1",
#                     file=audio_file
#                 )

#                 spoken_text = transcript.text
#                 st.success(f"Recognized: {spoken_text}")

#                 # Inject into main text input
#                 st.session_state.main_query = spoken_text

#         except Exception as e:
#             st.error(f"Whisper transcription failed: {e}")

