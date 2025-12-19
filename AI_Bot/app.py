# =============================
# AI Document Assistant — Bo
# =============================

import hashlib
import streamlit as st

# ----------------------------
# NLTK (safe for cloud)
# ----------------------------
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# ----------------------------
# File parsing
# ----------------------------
import PyPDF2
import docx

# ----------------------------
# ML
# ----------------------------
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(to bottom, #E6D9FF, #000000); color: white; }
    .hero { text-align:center; margin:20px 0; }
    .upload-box { border:2px dashed #C7B8FF; padding:20px; border-radius:12px; background:rgba(0,0,0,.25); }
    .answer-box { background:rgba(0,0,0,.4); padding:18px; border-radius:12px; margin-top:12px; }
    pre { white-space: pre-wrap; }
    b { color:#ffe66d; } /* Bold highlight color */
    ul.custom-bullets { list-style-type: none; padding-left: 1em; }
    ul.custom-bullets li::before { content: "• "; color:#C7B8FF; font-weight:bold; }
    h2 { color:#ffe66d; }
    h3 { color:#C7B8FF; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<div class='hero'><h1>🤖 Hi, I’m Bo</h1><h4>Your AI document assistant</h4></div>",
    unsafe_allow_html=True,
)

# ============================
# HELPERS (Updated)
# ============================

def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join(p.extract_text() for p in reader.pages if p.extract_text())
    elif "wordprocessingml" in file.type:
        doc = docx.Document(file)
        return " ".join(p.text for p in doc.paragraphs)
    return file.read().decode("utf-8", errors="ignore")


def preprocess_sentences(text):
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 40]


def summarize_text(sentences, ratio=0.15):
    if not sentences:
        return "No content to summarize."
    count = max(3, int(len(sentences) * ratio))
    summary_sentences = sentences[:count]
    # Add bullet points and ensure sentences end with a period
    formatted = "".join([f"<li>{s.strip().rstrip('.') + '.'}</li>" for s in summary_sentences])
    return f"<ul class='custom-bullets'>{formatted}</ul>"


def get_context(text, sentence, window=200):
    idx = text.find(sentence)
    if idx == -1:
        return f"<b>{sentence}</b>"
    start = max(0, idx - window)
    end = min(len(text), idx + len(sentence) + window)
    snippet = text[start:end]
    return snippet.replace(sentence, f"<b>{sentence}</b>")

# ============================
# EXAMPLE USAGE
# ============================

# Example text (replace with uploaded file content)
sample_text = "Artificial Intelligence helps automate tasks. It improves efficiency and reduces errors. AI can also assist in decision making."

sentences = preprocess_sentences(sample_text)

st.markdown("## 📑 Summary", unsafe_allow_html=True)
st.markdown("### Key Points", unsafe_allow_html=True)
st.markdown(summarize_text(sentences), unsafe_allow_html=True)

st.markdown("## 🔍 Context", unsafe_allow_html=True)
st.markdown(get_context(sample_text, "automate tasks"), unsafe_allow_html=True)
