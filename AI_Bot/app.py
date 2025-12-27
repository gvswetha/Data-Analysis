# =============================
# AI Document Assistant — FREE (OpenRouter Mistral)
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

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Bo — AI Document Assistant", layout="wide")

# =============================
# SESSION STATE (HISTORY)
# =============================
if "history" not in st.session_state:
    st.session_state.history = []

# =============================
# LOADERS
# =============================
@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("punkt_tab")


@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

load_nltk()

# =============================
# OPENROUTER CLIENT (SAFE)
# =============================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = None
if OPENROUTER_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    except ModuleNotFoundError:
        client = None

# =============================
# HELPERS
# =============================
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join(filter(None, [p.extract_text() for p in reader.pages]))
    elif file.type.endswith("wordprocessingml.document"):
        doc = docx.Document(file)
        return " ".join(p.text for p in doc.paragraphs)
    else:
        return file.read().decode("utf-8", errors="ignore")


def preprocess_sentences(text):
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 40]


def get_doc_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


@st.cache_data
def embed_sentences(doc_hash, sentences):
    model = load_sbert()
    return model.encode(sentences, show_progress_bar=False)


def retrieve_context(query, sentences, embeddings, top_k=5):
    model = load_sbert()
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, embeddings)[0]

    top_k = min(top_k, len(sentences))
    idx = sims.argsort()[-top_k:][::-1]
    return [sentences[i] for i in idx]


def structured_answer(query, context_sentences):
    context = "\n".join(context_sentences)

    prompt = f"""
Answer ONLY using the document context below.
Do not use outside knowledge.

Format exactly like this:

## Heading

### Direct Answer
(2–3 sentences)

### Key Points
- Bullet points

### Key Takeaways
- Short takeaways

Document Context:
{context}

Question:
{query}
"""

    # If no API key → graceful message
    if not client:
        return "⚠️ LLM not enabled. Add OPENROUTER_API_KEY in Streamlit secrets."

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct:free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content

# =============================
# UI
# =============================
st.title("🤖 Bo — AI Document Assistant")

st.sidebar.caption("Mode: LLM Enabled" if client else "Mode: Free (No API Key)")

file = st.file_uploader("📄 Upload a document", type=["pdf", "docx", "txt"])
query = st.text_input("🔍 Ask a question about the document")

if file:
    text = extract_text(file)
    sentences = preprocess_sentences(text)

    if not sentences:
        st.error("No valid content found.")
        st.stop()

    embeddings = embed_sentences(get_doc_hash(text), sentences)

    if query:
        with st.spinner("Bo is thinking..."):
            context = retrieve_context(query, sentences, embeddings)
            answer = structured_answer(query, context)

        # Show answer
        st.markdown("### 💡 Answer")
        st.markdown(answer)

        # Save history (LAST 5)
        st.session_state.history.insert(0, {
            "question": query,
            "answer": answer
        })
        st.session_state.history = st.session_state.history[:5]

else:
    st.info("Upload a document to begin.")

# =============================
# SIDEBAR HISTORY
# =============================
with st.sidebar:
    st.subheader("🕘 Last 5 Searches")

    if st.session_state.history:
        for i, item in enumerate(st.session_state.history, 1):
            with st.expander(f"{i}. {item['question']}"):
                st.markdown(item["answer"])
    else:
        st.caption("No searches yet.")

