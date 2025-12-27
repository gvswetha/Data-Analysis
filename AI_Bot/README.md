# 🤖 Bo — AI Document Assistant (Free LLM)

Bo is a **Document Question Answering (QA) web app** built with **Streamlit**, **Sentence Transformers**, and a **free LLM (Mistral via OpenRouter)**.  
It allows users to upload documents and ask questions, receiving **structured, well-organized answers** based strictly on the document content.

---

## ✨ Features

- 📄 Upload documents (`PDF`, `DOCX`, `TXT`)
- 🔍 Ask natural-language questions about the document
- 🧠 Semantic search using **Sentence-BERT**
- 🤖 Free LLM support using **Mistral-7B (OpenRouter)**
- 📝 Structured answers with:
  - Headings
  - Direct answers
  - Bullet points
  - Key takeaways
- 🕘 Stores **last 5 questions & answers**
- ☁️ Fully compatible with **Streamlit Cloud**
- 🔐 Runs even **without an API key** (graceful fallback)


## 🛠️ Tech Stack

- **Frontend / App**: Streamlit  
- **LLM**: `mistralai/mistral-7b-instruct:free` (via OpenRouter)  
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)  
- **Similarity Search**: Scikit-learn (cosine similarity)  
- **NLP**: NLTK  
- **Document Parsing**: PyPDF2, python-docx  

---

## 📂 Project Structure


.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation

**APP**

https://data-analysis-abrjb5lvgvw2xfvwwtjo9o.streamlit.app/


