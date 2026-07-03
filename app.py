"""
Gradio entrypoint for the AvaniMudra chatbot.

Requires that ingest.py has already been run at least once so that a
Chroma vector store exists at config.PERSIST_DIRECTORY, and that Ollama is
running locally with the model in config.OLLAMA_MODEL pulled.

Usage:
    python ingest.py      # one-time (or whenever AM_scraped.txt changes)
    python app.py          # launch the chat UI
"""

import argparse

import gradio as gr

import config
from chatbot import build_conversation_chain, make_chat_fn


def build_interface() -> gr.ChatInterface:
    conversation_chain = build_conversation_chain()
    chat = make_chat_fn(conversation_chain)
    return gr.ChatInterface(fn=chat, title=config.CHAT_TITLE, type="messages")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the AvaniMudra chatbot UI.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public, shareable Gradio link (off by default).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_interface()
    demo.launch(share=args.share)
