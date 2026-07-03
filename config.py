"""
Central configuration for the AvaniMudra chatbot.

All constants that were previously hardcoded / scattered across notebook
cells live here so they can be tuned in one place.
"""

import os

# Base directory of the project (so paths work regardless of cwd).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Data -------------------------------------------------------------
# Scraped company text used as the knowledge base for retrieval.
DATA_FILE = os.path.join(BASE_DIR, "AM_scraped.txt")

# --- Text splitting -----------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Embeddings ---------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Vector store (ChromaDB) ---------------------------------------------
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "am_chatbot_db")

# --- Retrieval ------------------------------------------------------------
RETRIEVER_TOP_K = 3

# --- LLM (Ollama) ---------------------------------------------------------
OLLAMA_MODEL = "llama3.2:latest"

# --- Gradio UI ------------------------------------------------------------
CHAT_TITLE = "AM Chatbot (Local Ollama)"
