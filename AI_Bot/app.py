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
    mark { background:#ffe66d; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<div class='hero'><h1>🤖 Hi, I’m Bo</h1><h4>Your AI document assistant</h4></div>",
    unsafe_allow_html=True,
)

# ============================
# HELPERS
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
    # Return Markdown bullet points with proper sentence endings
    return "\n".join([f"- {s.strip().rstrip('.') + '.'}" for s in summary_sentences])



def get_context(text, sentence, window=200):
    idx = text.find(sentence)
    if idx == -1:
        return f"<b>{sentence}</b>"
    start = max(0, idx - window)
    end = min(len(text), idx + len(sentence) + window)
    snippet = text[start:end]
    return snippet.replace(sentence, f"<b>{sentence}</b>")

st.markdown("## 📑 Summary")
st.markdown("### Key Points")
st.markdown(summarize_text(sentences))

st.markdown("## 🔍 Context")
st.markdown(get_context(sample_text, "automate tasks"), unsafe_allow_html=True)




@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def doc_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


@st.cache_data
def embed_sentences(hash_key, sentences_tuple):
    model = load_model()
    return model.encode(list(sentences_tuple), show_progress_bar=False)


def answer_question(query, sentences, embeddings, text, k=5):
    model = load_model()
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top = sims.argsort()[-k:][::-1]

    results = []
    for i in top:
        results.append({
            "sentence": sentences[i],
            "score": float(sims[i]),
            "context": get_context(text, sentences[i])
        })

    return {
        "combined": " ".join(r["sentence"] for r in results),
        "bullets": "\n".join(f"- {r['sentence']}" for r in results),
        "results": results,
    }

# ============================
# SESSION STATE
# ============================
if "chat" not in st.session_state:
    st.session_state.chat = []

# ============================
# LAYOUT
# ============================
col_main, col_side = st.columns([2, 1])

# ----------------------------
# MAIN COLUMN
# ----------------------------
with col_main:
    st.markdown("<div class='upload-box'><h4>📄 Upload document</h4></div>", unsafe_allow_html=True)
    file = st.file_uploader("", type=["pdf", "docx", "txt"], label_visibility="collapsed")

    user_query = st.text_input(
        "Ask a question",
        placeholder="Ask me about the document…",
        key="main_query"
    )

    selected_question = None

    if file:
        text = extract_text(file)
        sentences = preprocess_sentences(text)

        if not sentences:
            st.error("No valid sentences found.")
            st.stop()

        embeddings = embed_sentences(doc_hash(text), tuple(sentences))

        st.markdown("<div class='answer-box'><h3>📘 Summary</h3></div>", unsafe_allow_html=True)
        summary = summarize_text(sentences)
        st.markdown(f"<pre>{summary}</pre>", unsafe_allow_html=True)

        st.download_button("⬇️ Download Summary", summary, "summary.txt")

        st.markdown("<div class='answer-box'><h3>📌 Suggested Questions</h3></div>", unsafe_allow_html=True)
        samples = [
            "What is this document about?",
            "What is the objective?",
            "Explain key concepts",
            "Describe the methodology",
            "What are the conclusions?"
        ]

        cols = st.columns(len(samples))
        for i, q in enumerate(samples):
            with cols[i]:
                if st.button(q, key=f"s{i}"):
                    selected_question = q

        query = selected_question or user_query

        if query:
            result = answer_question(query, sentences, embeddings, text)
            st.session_state.chat.append({"query": query, "answer": result["combined"]})

            st.markdown("<div class='answer-box'><h3>💡 Answer</h3></div>", unsafe_allow_html=True)
            st.markdown(result["combined"])
            st.markdown("<pre>" + result["bullets"] + "</pre>")

            st.session_state.last_result = result
            st.session_state.last_text = text

# ----------------------------
# SIDE COLUMN
# ----------------------------
with col_side:
    st.markdown("<div class='answer-box'><h3>✨ Insights</h3></div>", unsafe_allow_html=True)

    res = st.session_state.get("last_result")
    txt = st.session_state.get("last_text")

    if res and txt:
        for i, r in enumerate(res["results"]):
            with st.expander(f"{i+1}. {r['sentence'][:60]}..."):
                st.markdown(r["context"], unsafe_allow_html=True)
                st.caption(f"Score: {r['score']:.3f}")
    else:
        st.caption("Ask a question to see insights.")

# ----------------------------
# SIDEBAR HISTORY
# ----------------------------
with st.sidebar:
    st.markdown("## 🗂️ History")
    for turn in reversed(st.session_state.chat):
        st.markdown(f"**You:** {turn['query']}")
        st.markdown(f"**Bo:** {turn['answer']}")
        st.divider()

