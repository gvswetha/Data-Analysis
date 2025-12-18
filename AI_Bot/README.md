AI Document Assistant — Bo
==========================

Bo is an AI-powered document assistant built with Streamlit. It allows users to:

• Upload PDF, DOCX, or TXT files  
• Automatically summarize the document  
• Ask questions about the content  
• View highlighted context from the original document  
• Use voice input (via Whisper API)  
• Maintain chat history  
• Enjoy a clean two-column professional UI with a dark gradient theme  

---------------------------------------
FEATURES
---------------------------------------

1. Document Upload
   - Supports PDF, DOCX, and TXT
   - Extracts text automatically

2. Smart Summary
   - Generates a concise extractive summary
   - Downloadable as a text file

3. Semantic Search (SBERT)
   - Uses all-MiniLM-L6-v2 for sentence embeddings
   - Cached embeddings for fast repeated queries

4. Question Answering
   - Retrieves the most relevant sentences
   - Combines them into a natural answer
   - Shows bullet points and relevance scores

5. Highlighted Context
   - Shows where each answer came from inside the document
   - Highlights the exact sentence in context

6. Voice Input (Whisper API)
   - Upload audio (wav/mp3/m4a)
   - Whisper transcribes it into text
   - Automatically inserts into the chat

7. Chat History
   - Stored in Streamlit session state
   - Expandable sidebar history

8. Modern UI
   - Two-column layout
   - Dark gradient theme
   - Clean cards and typography

---------------------------------------
REQUIREMENTS
---------------------------------------

Install dependencies:

    pip install -r requirements.txt

Set your OpenAI API key:

    export OPENAI_API_KEY="your_key_here"
    (Windows PowerShell)
    setx OPENAI_API_KEY "your_key_here"

Run the app:

    streamlit run app.py

---------------------------------------
FILE STRUCTURE
---------------------------------------

app.py
README.txt
requirements.txt

---------------------------------------
NOTES
---------------------------------------

• Whisper API requires an OpenAI API key  
• SBERT embeddings are cached per document for speed  
• Voice input uses Streamlit’s built-in audio uploader  
• The app runs entirely locally except for Whisper API calls  

---------------------------------------
AUTHOR
---------------------------------------
swetha 
Bo — AI Document Assistant
Built with Streamlit + SBERT + Whisper

