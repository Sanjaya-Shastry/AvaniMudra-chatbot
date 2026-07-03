# AvaniMudra Chatbot — Local RAG Chatbot using Ollama + LangChain + ChromaDB

This project is a local Retrieval-Augmented Generation (RAG) chatbot built using:
- Ollama (to run open-source LLMs like `llama3`)
- LangChain (to build the conversational agent)
- ChromaDB (as the local vector database)
- Gradio (for a lightweight chat UI)

It is specifically designed to answer questions only about the company AvaniMudra (AM) using scraped company documentation.

---

## Features

- Natural-language question answering
- Uses your own knowledge base (local `.txt` file)
- Remembers context within a conversation
- Fully local — no API keys required
- Runs entirely offline with `llama3` via Ollama

---

## Project Structure

```
.
├── AM_scraped.txt      # Scraped AvaniMudra company text (knowledge base)
├── config.py           # All constants: paths, chunk size/overlap, model names, top-k
├── ingest.py           # Loads AM_scraped.txt, chunks it, builds & persists the Chroma vector store
├── chatbot.py          # Loads the persisted vector store and builds the LangChain
│                       # conversational retrieval chain (retriever + Ollama LLM + memory)
├── app.py               # Gradio ChatInterface entrypoint
├── requirements.txt
└── .gitignore
```

---

## Setup

1. **Python**: 3.10+ recommended.

2. **Install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Install and run Ollama**, then pull the model used by this project:
   ```bash
   ollama pull llama3.2
   ollama serve
   ```
   (The model name is set in `config.py` as `OLLAMA_MODEL`; adjust it there if you use a different local model.)

---

## Usage

1. **Build the vector store** (run once, and again whenever `AM_scraped.txt` changes):
   ```bash
   python ingest.py
   ```
   This reads `AM_scraped.txt`, splits it into overlapping chunks, embeds them with a
   `sentence-transformers/all-MiniLM-L6-v2` model, and persists the result to a local
   Chroma directory (`am_chatbot_db/` by default).

2. **Launch the chatbot UI**:
   ```bash
   python app.py
   ```
   This starts a Gradio chat interface backed by a `ConversationalRetrievalChain`
   (top-3 retrieval + conversation memory) running against your local Ollama model.
   Pass `--share` to also create a temporary public Gradio link.
