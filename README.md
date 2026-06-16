# Agentic RAG PDF Assistant

An enterprise-style Agentic Retrieval-Augmented Generation (RAG) system for multi-document question answering using Docling, ChromaDB, LangGraph, and Gemini.

## Features

* Multi-document PDF ingestion
* OCR-aware document parsing with Docling
* Hierarchical chunking
* ChromaDB vector database
* Dense retrieval using BGE embeddings
* BM25 lexical retrieval
* Hybrid retrieval with Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* Query rewriting
* Retrieval grading and retry loops
* Conversation memory
* LangGraph-based agent workflow
* Source-grounded answers with citations
* Streamlit user interface

## Architecture

```text
                ┌─────────────────┐
                │   Upload PDFs   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     Docling     │
                │ OCR + Parsing   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Chunking     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ BGE Embeddings  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    ChromaDB     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ User Question   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Query Rewrite   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Hybrid Search   │
                │ Dense + BM25    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   RRF Fusion    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Cross Encoder   │
                │   Reranking     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   LangGraph     │
                │ Agent Workflow  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Gemini Response │
                │ + Citations     │
                └─────────────────┘
```

## Tech Stack

* Python
* Docling
* ChromaDB
* LangChain
* LangGraph
* Streamlit
* Google Gemini
* Sentence Transformers
* BM25
* Cross-Encoder Reranking

## Setup

### Clone Repository

```bash
git clone https://github.com/akk-026/agentic-rag-pdf.git
cd agentic-rag-pdf
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create .env File

```env
GOOGLE_API_KEY=your_api_key_here
```

## Running the Project

### Launch Streamlit App

```bash
streamlit run app.py
```

### Run Tests

```bash
python -m tests.test_ingest
python -m tests.test_vector_store
python -m tests.test_hybrid_retrieval
python -m tests.test_agent
```

## How It Works

1. Upload one or more PDFs.
2. Documents are parsed using Docling.
3. Chunks are created and embedded.
4. Embeddings are stored in ChromaDB.
5. User questions are rewritten when necessary.
6. Hybrid retrieval combines Dense Search and BM25.
7. Results are fused using RRF and reranked using a Cross Encoder.
8. LangGraph orchestrates the retrieval and generation workflow.
9. Gemini generates grounded answers with citations.

## Example Use Cases

* Financial report analysis
* Healthcare compliance search
* Enterprise document Q&A
* Multi-document reasoning
* Conversational document assistants

## Future Improvements

* DOCX and PPTX support
* RAG evaluation framework
* Multi-agent workflows
* Cloud deployment

## License

For educational and research purposes.
