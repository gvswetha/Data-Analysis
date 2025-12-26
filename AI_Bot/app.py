# =============================
# AI Document Assistant — FREE LLM (Mistral via OpenRouter)
# =============================

import hashlib
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
import PyPDF2
import docx

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from openai import OpenAI

# =============================
# 🔐 PASTE YOUR OPENROUTER KEY
# =============================
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = None

if OPENROUTER_API_KEY:
    from openai import OpenAI
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )


# =============================
# BASIC SETUP
# =============================
# =============================
# SEARCH HISTORY (LAST 5)
# =============================

if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="AI Document Assistant (Free LLM)", layout="wide")

@st.cache_resource
def load_nltk():
    nltk.download("punkt")

load_nltk()

@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

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
        with st.spinner("Bo is thinking (free model)..."):
            context = retrieve_context(query, sentences, embeddings)
            result = structured_answer(query, context)  # ✅ DEFINE RESULT

        # --- UI ---
        st.markdown("### 💡 Answer")
        st.markdown(f"**{result['heading']}**")
        st.markdown(result["answer"])

        st.markdown("#### 🔑 Key Points")
        for b in result["bullets"]:
            st.markdown(f"- {b}")

        # --- SAVE HISTORY (ONLY AFTER RESULT EXISTS) ---
        st.session_state.history.insert(0, {
            "question": query,
            "answer": result["answer"],
            "bullets": result["bullets"],
            "heading": result["heading"]
        })

        st.session_state.history = st.session_state.history[:5]

else:
    st.info("Upload a document to begin.")


