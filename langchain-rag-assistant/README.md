# LangChain RAG Assistant

A local, private, document-based question-answering system using Retrieval-Augmented Generation (RAG).

## 🚀 Features
- **PDF Ingestion & Processing**: Automatically loads PDF files, splits them into manageable chunks recursively, and stores them in a Vector Database.
- **Local LLMs**: Powered by `Ollama` running local models (`llama3`, `nomic-embed-text`) ensuring privacy without relying on external APIs.
- **Smart Query Routing**: The system determines if a user's prompt is a general chat greeting or a technical RAG question, routing it intelligently to the right process.
- **Interactive UI**: Upload PDFs on the fly and chat directly with them via the Streamlit interface.

## 🛠️ Tech Stack
- **Framework**: LangChain
- **Vector Database**: ChromaDB
- **LLM Engine**: Ollama (Llama3, Nomic-Embed)
- **Web Application**: Streamlit
- **Document Processing**: PyPDF

## 📦 Installation & Setup

### Prerequisites
You need to install [Ollama](https://ollama.com) on your system and run following models:
```bash
ollama run llama3
ollama pull nomic-embed-text
```

### Steps
1. **Clone or Download the repository.**

2. **Setup virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
