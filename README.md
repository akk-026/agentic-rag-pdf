# Agentic Multi-Document RAG System

An enterprise-style Agentic Retrieval-Augmented Generation (RAG) system capable of:

- Multi-PDF document ingestion
- OCR-aware document parsing
- Semantic retrieval
- Metadata-aware search
- ChromaDB vector storage
- Agent-based document reasoning

## Tech Stack

- Python
- LangChain
- Docling
- ChromaDB
- Sentence Transformers
- Streamlit

## Features

- Supports multiple PDFs
- OCR for scanned documents
- Semantic chunking
- Metadata-preserving retrieval
- Enterprise document intelligence

## Project Architecture

```text
src/
├── docling_loader.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── retriever.py
└── agent.py
```

## Status

Currently building:
- Document ingestion pipeline ✅
- OCR support ✅
- Semantic chunking 🚧
- Vector retrieval 🚧
- Agentic workflows 🚧