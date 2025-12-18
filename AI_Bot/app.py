# =============================
# AI Document Assistant — Bo (FINAL • CLOUD-SAFE • STREAMLIT-CORRECT)
# =============================

import os
import hashlib
import streamlit as st

import nltk
from nltk.tokenize import sent_tokenize

import PyPDF2
import docx

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# SAFE NLTK SETUP (Cloud-safe)
# ============================
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="AI Document Assistant — Bo", layout="wide")

# ============================
# THEME
# ============================
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(to bottom, #E6D9FF, #000000); color: white; }
    .hero { text-align: center; margin: 20px 0; }
    .upload-box { border: 2px dashed #C7B8FF; padding: 20px; border-radius: 12px; background: rgba(0,0,0,0.25); }
    .answer-box { background: rgba(0,0,0,0.45); padding: 18px; border-radius: 12px; margin-top: 12px; }
    pre { white-space: pre-wrap; word-wrap: break-word; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# HERO
# ============================
st.markdown(
    "<div class='hero'><h1>🤖 Hi, I’m Bo</h1><h4>Your AI document assistant</h4></div>",
    unsafe_allow_html=True,
)

# ============================
# MODEL LOADER (CACHED)
# ============================
@st.cache_resource(show_spinner=True)
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ============================
# HELPERS
# ============================

def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join(p.extract_text() or "" for p in reader.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return " ".join(p.text for p in doc.paragraphs)
    else:
        return file.read().decode("utf-8", errors="ignore")


def preprocess_sentences(text):
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 40]


def summarize_text(sentences, ratio=0.15):
    if not sentences:
        return "No content to summarize."
    n = max(3, int(len(sentences) * ratio))
    return " ".join(sentences[:n])


def get_doc_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def encode_sentences_cached(doc_hash, sentences):
    model = load_sbert_model()
    return model.encode(list(sentences), show_progress_bar=False)


def answer_question(query, sentences, embeddings, top_k=5):
    model = load_sbert_model()
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    bullets = [f"- {sentences[i]}" for i in top_idx]
    combined = " ".join(sentences[i] for i in top_idx)
    return combined, "\n".join(bullets)

# ============================
# SESSION STATE
# ============================
if "chat" not in st.session_state:
    st.session_state.chat = []

# ============================
# LAYOUT
# ============================
left, right = st.columns([2, 1])

# ----------------------------
# LEFT: DOCUMENT + CHAT
# ----------------------------
with left:
    st.markdown("<div class='upload-box'><h4>📄 Upload your document</h4></div>", unsafe_allow_html=True)
    file = st.file_uploader("", type=["pdf", "docx", "txt"], label_visibility="collapsed")

    user_query = st.text_input("", placeholder="Ask me about the document…")

    sentences = None
    embeddings = None
    selected_question = None

    if file:
        with st.spinner("🤖 Bo is reading your document…"):
            text = extract_text(file)
            sentences = preprocess_sentences(text)

        if sentences:
            doc_hash = get_doc_hash(text)
            embeddings = encode_sentences_cached(doc_hash, tuple(sentences))

            summary = summarize_text(sentences)
            st.markdown("<div class='answer-box'><h3>📘 Summary</h3></div>", unsafe_allow_html=True)
            st.markdown(f"<pre>{summary}</pre>", unsafe_allow_html=True)
            st.download_button("⬇️ Download Summary", summary, file_name="summary.txt")

            st.markdown("<div class='answer-box'><h3>📌 Suggested Questions</h3></div>", unsafe_allow_html=True)
            questions = [
                "What is this document about?",
                "What is the objective of this document?",
                "Explain the key concepts",
                "Describe the methodology",
                "What are the conclusions?",
            ]

            cols = st.columns(len(questions))
            for i, q in enumerate(questions):
                with cols[i]:
                    if st.button(q, key=f"sq_{i}"):
                        selected_question = q

            query_to_answer = selected_question or user_query

            if query_to_answer:
                with st.spinner("🤖 Bo is thinking…"):
                    answer, bullets = answer_question(query_to_answer, sentences, embeddings)

                st.session_state.chat.append({"q": query_to_answer, "a": answer})

                st.markdown("<div class='answer-box'><h3>💡 Answer</h3></div>", unsafe_allow_html=True)
                st.markdown(f"<pre>{answer}</pre>", unsafe_allow_html=True)
                st.markdown(f"<pre>{bullets}</pre>", unsafe_allow_html=True)
        else:
            st.warning("No meaningful text found in this document.")
    else:
        if user_query:
            st.markdown(
                "<div class='answer-box'><pre>Hi, I’m Bo 🤖. Upload a document and I’ll summarize it and answer questions based on its content.</pre></div>",
                unsafe_allow_html=True,
            )

# ----------------------------
# RIGHT: CHAT HISTORY
# ----------------------------
with right:
    st.markdown("<div class='answer-box'><h3>💬 Conversation</h3></div>", unsafe_allow_html=True)
    for turn in reversed(st.session_state.chat):
        st.markdown(f"<pre><b>You:</b> {turn['q']}\n<b>Bo:</b> {turn['a']}</pre>", unsafe_allow_html=True)

    if st.button("Clear chat"):
        st.session_state.chat = []
