# Simple RAG Chatbot (Local, Ollama + LangChain + Streamlit)

A minimal Retrieval-Augmented Generation (RAG) app that lets you **chat with your own PDFs and text files** using a fully local stack:

- Local LLM and embeddings via **Ollama**
- **LangChain** for RAG pipeline (chunking, retrieval, prompting)
- **Chroma** as the local vector store
- **Streamlit** for the web UI

No external API keys or cloud models are required.

---

## 1. Project Overview

### Architecture

1. **Document ingestion**
   - User uploads PDFs / TXT via Streamlit.
   - Files are stored in the `data/` folder.

2. **Preprocessing and indexing**
   - Documents are loaded with `PyPDFLoader` / `TextLoader`.
   - Text is split into overlapping chunks (`RecursiveCharacterTextSplitter`).
   - Each chunk is embedded with a local Ollama embedding model (`nomic-embed-text:latest`).
   - Embeddings + metadata are stored in a persistent **Chroma** DB (`chroma_db/`).

3. **Retrieval-augmented generation**
   - On each question, the top‑k relevant chunks are retrieved from Chroma.
   - A prompt is built with the retrieved context.
   - A local chat model (`llama3.2:latest`) generates an answer grounded in that context.

4. **UI**
   - Streamlit chat interface with history.
   - Sidebar to upload documents and rebuild the index.

---

## 2. Prerequisites

- macOS with **Python 3.10+** (3.11 recommended)
- [Ollama](https://ollama.com) installed and running
- Git and a terminal

### 2.1. Install and prepare Ollama

1. Install Ollama from the official website and open the app.
2. Pull the models used in this project:

ollama pull llama3.2
ollama pull nomic-embed-text


Check they are available:

ollama list

should show llama3.2:latest and nomic-embed-text:latest


---

## 3. Local Setup

### 3.1. Clone the repo

git clone <your-repo-url>.git
cd RAG_agent # or your repo folder name



### 3.2. Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate



### 3.3. Install dependencies

`requirements.txt`:

Install:

pip install -r requirements.txt

---

## 4. Run the App Locally

1. Make sure **Ollama app is running**.
2. From the project folder:

source .venv/bin/activate
rm -rf chroma_db # optional: clear old index when changing models
streamlit run app.py

3. In the browser (default `http://localhost:8501`):

   - Use the sidebar to upload one or more PDFs / TXT files.
   - Click **Rebuild index** once to create the vector store.
   - Ask questions like:
     - “What is this document about?”
     - “How does it define machine learning?”
     - “Summarize chapter 1.”

If you change documents significantly, click **Rebuild index** again to rebuild embeddings.

---

## 5. Deploying to Streamlit Community Cloud

> Note: Ollama and the models run on **your local machine**. For a pure Streamlit Cloud deployment, you would need a cloud‑accessible model (e.g., OpenAI, Gemini). The current configuration is ideal for local use and demonstrations.

Still, you can deploy the UI and replace models later if you choose a cloud LLM.

### 5.1. Prepare the repo

1. Ensure your repo has at least:

   - `app.py`
   - `rag_utils.py`
   - `requirements.txt`
   - `data/` (can be empty in Git)
   - `.gitignore`

2. Example `.gitignore`:

pycache/
.venv/
chroma_db/
.pyc
.env
data/

### 5.2. Push to GitHub

git init
git add .
git commit -m "Initial RAG app"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main

### 5.3. Create the Streamlit app

1. Go to Streamlit Community Cloud and sign in with GitHub.
2. Click **New app**.
3. Select:
   - Repo: your GitHub repo
   - Branch: `main`
   - Main file: `app.py`
4. Click **Deploy**.

Deployment will install dependencies from `requirements.txt` and launch the app.

> Important: Streamlit Cloud cannot access your local Ollama server.  
> To use this app on Streamlit Cloud, replace `ChatOllama` and `OllamaEmbeddings` with a cloud provider (OpenAI, Gemini, etc.), and add the relevant API key in **Secrets**. The RAG logic and UI remain the same.

---

## 6. Files Overview

- `app.py` – Streamlit UI and chat flow.
- `rag_utils.py` – RAG pipeline (loading docs, chunking, embeddings, retrieval, prompting).
- `requirements.txt` – Python dependencies.
- `data/` – Uploaded documents (ignored in Git).
- `chroma_db/` – Local vector store (auto‑created, ignored in Git).

---

## 7. Troubleshooting

- **Ollama errors (`runner process not running`, `EOF`, 500)**  
  - Ensure Ollama is running and models are pulled.  
  - Try smaller PDFs and reduce `chunk_size` in `rag_utils.py`.

- **Streamlit loads but answers ignore documents**  
  - Check that you clicked **Rebuild index** after uploading files.  
  - Remove `chroma_db/` and restart to force a fresh index.


