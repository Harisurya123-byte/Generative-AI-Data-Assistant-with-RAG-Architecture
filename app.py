# app.py
import os
from pathlib import Path

import streamlit as st

from rag_utils import (
    build_or_load_vectorstore,
    build_rag_chain,
)

st.set_page_config(
    page_title="Simple RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

st.title("📚 Simple RAG Chatbot")
st.write(
    "Ask questions over your own documents using a minimal RAG pipeline "
    "(LangChain + Ollama + Chroma + Streamlit)."
)

# --- Sidebar: document upload ---

st.sidebar.header("📂 Documents")
st.sidebar.write("Upload PDFs or text files. They will be indexed locally.")

uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

if uploaded_files:
    for file in uploaded_files:
        file_path = data_dir / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("Files saved. Click 'Rebuild index' if needed.")

rebuild = st.sidebar.button("🔁 Rebuild index")

# --- Build or load vector store ---

if "vectorstore" not in st.session_state or rebuild:
    with st.spinner("Building / loading vector store..."):
        vs = build_or_load_vectorstore(str(data_dir))
        st.session_state.vectorstore = vs

if st.session_state.get("vectorstore") is None:
    st.warning("No documents found. Upload at least one PDF or text file in the sidebar.")
    st.stop()

# --- Build RAG chain once per session ---

if "rag_chain" not in st.session_state or rebuild:
    st.session_state.rag_chain = build_rag_chain(
        st.session_state.vectorstore,
    )

# --- Chat interface ---

st.subheader("💬 Chat with your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_question = st.chat_input("Ask a question about your documents...")
if user_question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Assistant response via RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.rag_chain.invoke(user_question)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
