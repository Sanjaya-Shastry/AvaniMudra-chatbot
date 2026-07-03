"""
Conversational retrieval chain for the AvaniMudra chatbot.

Loads the vector store persisted by ingest.py, wires it up to a local
Ollama LLM via LangChain's ConversationalRetrievalChain, and exposes a
simple `chat()` function for use by the UI in app.py.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import config


def build_retriever(
    persist_directory: str = config.PERSIST_DIRECTORY,
    embedding_model: str = config.EMBEDDING_MODEL,
    top_k: int = config.RETRIEVER_TOP_K,
):
    """Load the persisted Chroma vector store and wrap it as a retriever."""
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
    )
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def build_conversation_chain(
    model: str = config.OLLAMA_MODEL,
    persist_directory: str = config.PERSIST_DIRECTORY,
) -> ConversationalRetrievalChain:
    """Assemble the LLM + retriever + memory into a conversational chain."""
    retriever = build_retriever(persist_directory=persist_directory)
    llm = ChatOllama(model=model)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )


def make_chat_fn(conversation_chain: ConversationalRetrievalChain):
    """Return a `chat(message, history)` function bound to a chain instance,
    matching the signature expected by gradio.ChatInterface."""

    def chat(message: str, history) -> str:
        result = conversation_chain.invoke({"question": message})
        return result["answer"]

    return chat
