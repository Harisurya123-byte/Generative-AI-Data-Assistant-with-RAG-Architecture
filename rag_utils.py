# rag_utils.py
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load .env if present (harmless even if empty)
load_dotenv()

PERSIST_DIR = "chroma_db"


def load_documents(data_dir: str = "data"):
    """Load PDFs and TXT files from the data directory."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )

    docs = []
    for loader in [pdf_loader, txt_loader]:
        try:
            docs.extend(loader.load())
        except Exception:
            pass

    return docs


def _clean_text(text: str) -> str:
    """Remove invalid surrogate characters to avoid Unicode errors."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def build_or_load_vectorstore(data_dir: str = "data"):
    """Create or load a Chroma vector store using local Ollama embeddings."""
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",   # must match `ollama list`
    )
    

    # If persisted DB exists, load it
    if Path(PERSIST_DIR).exists():
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )
        return vectorstore

    # Otherwise build from documents
    docs = load_documents(data_dir)
    print(f"Loaded {len(docs)} documents")

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")


    # Clean all text chunks
    for doc in chunks:
        if isinstance(doc.page_content, str):
            doc.page_content = _clean_text(doc.page_content)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectorstore


def build_rag_chain(vectorstore: Chroma):
    """Build a RAG chain using local ChatOllama + Chroma retriever."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOllama(
        model="llama3.2:latest",          # must match `ollama list`
        temperature=0.1,
    )
    


    system_prompt = (
        "You are a helpful assistant that answers questions using the provided context. "
        "If the answer is not in the context, say you are not sure and avoid hallucinating.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
