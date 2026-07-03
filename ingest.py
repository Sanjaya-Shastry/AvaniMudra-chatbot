"""
Ingestion pipeline for the AvaniMudra chatbot.

Loads the scraped company text file, splits it into chunks, embeds those
chunks with a HuggingFace sentence-transformer model, and persists the
resulting vectors to a local ChromaDB directory.

Run this once (and again whenever AM_scraped.txt changes) before starting
the chat app:

    python ingest.py
"""

import argparse

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import config


def load_text(data_file: str = config.DATA_FILE) -> str:
    """Read the raw scraped text file."""
    with open(data_file, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Split the raw text into overlapping chunks and drop any empty ones."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = [Document(page_content=text)]
    chunks = splitter.split_documents(docs)
    return [doc for doc in chunks if doc.page_content and doc.page_content.strip()]


def build_vectorstore(
    chunks: list[Document],
    persist_directory: str = config.PERSIST_DIRECTORY,
    embedding_model: str = config.EMBEDDING_MODEL,
) -> Chroma:
    """Embed the chunks and persist them to a local Chroma vector store."""
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory,
    )


def run(data_file: str = config.DATA_FILE, persist_directory: str = config.PERSIST_DIRECTORY) -> None:
    text = load_text(data_file)
    chunks = split_into_chunks(text)
    vectorstore = build_vectorstore(chunks, persist_directory=persist_directory)
    vectorstore.persist()
    print(f"Ingested {len(chunks)} chunks from '{data_file}' into '{persist_directory}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/refresh the AvaniMudra vector store.")
    parser.add_argument(
        "--data-file",
        default=config.DATA_FILE,
        help="Path to the scraped text file to ingest.",
    )
    parser.add_argument(
        "--persist-directory",
        default=config.PERSIST_DIRECTORY,
        help="Directory where the Chroma vector store is persisted.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(data_file=args.data_file, persist_directory=args.persist_directory)
