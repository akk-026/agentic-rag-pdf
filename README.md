# Enterprise Agentic RAG for Multi-Document Intelligence

An enterprise-grade Agentic Retrieval-Augmented Generation (RAG) system designed for understanding, retrieving, reasoning over, and synthesizing information from large collections of documents.

Built using Docling, ChromaDB, BGE Embeddings, Hybrid Retrieval, Reranking, and Gemini LLMs, the system supports multi-document question answering, source-grounded responses, conversational memory, query rewriting, and agentic retrieval workflows.

---

## Key Features

### Intelligent Document Processing

* Multi-document ingestion pipeline
* OCR support for scanned PDFs
* Structured parsing using Docling
* Hierarchical semantic chunking
* Page-level provenance extraction
* Metadata preservation

### Advanced Retrieval Pipeline

* Dense semantic retrieval
* BM25 keyword retrieval
* Hybrid Search
* Reciprocal Rank Fusion (RRF)
* BGE Cross-Encoder Reranking
* Multi-document retrieval
* Context-aware ranking

### Agentic Retrieval Workflows

* Query planning
* Query rewriting and expansion
* Iterative retrieval loops
* Dynamic context refinement
* Multi-step reasoning
* Retrieval validation

### Conversational Intelligence

* Session memory
* Multi-turn conversations
* Context retention
* Follow-up question understanding
* Conversation-aware retrieval

### Multi-Document Reasoning

* Cross-document synthesis
* Contradiction detection
* Comparative analysis
* Evidence aggregation
* Knowledge consolidation

### Source Attribution

* Deterministic citations
* Page-level references
* Document provenance tracking
* Explainable retrieval

### Enterprise UI

* Streamlit-based interface
* Multi-file upload
* Interactive document querying
* Citation display
* Retrieval transparency
* Real-time responses

---

## System Architecture

```text
Documents
     │
     ▼
Docling Ingestion
     │
     ▼
OCR + Parsing
     │
     ▼
Hierarchical Chunking
     │
     ▼
BGE Embeddings
     │
     ▼
ChromaDB Vector Store
     │
     ▼
Hybrid Retrieval
 ├── Dense Search
 ├── BM25 Search
 └── RRF Fusion
     │
     ▼
BGE Reranker
     │
     ▼
Agent Layer
 ├── Query Rewriting
 ├── Retrieval Planning
 ├── Memory Management
 └── Iterative Retrieval
     │
     ▼
Gemini LLM
     │
     ▼
Answer + Citations
```

---

## Technology Stack

### Document Processing

* Docling

### Embeddings

* BAAI/bge-base-en-v1.5

### Vector Database

* ChromaDB

### Retrieval

* BM25
* Reciprocal Rank Fusion (RRF)
* Hybrid Search

### Reranking

* BGE Cross Encoder Reranker

### LLM

* Google Gemini

### Frontend

* Streamlit

### Language

* Python

---

## Example Use Cases

### Regulatory Compliance

**Question**

```text
What are the three main HIPAA rules?
```

**Answer**

```text
Privacy Rule
Security Rule
Breach Notification Rule
```

with source citations.

---

### Financial Analysis

**Question**

```text
What was 3M's worldwide net sales in Q2 2023?
```

**Answer**

```text
$8.325 billion
```

with page-level references.

---

### Cross-Document Reasoning

**Question**

```text
Compare privacy requirements discussed across multiple regulatory documents.
```

**System Workflow**

```text
Retrieve Relevant Evidence
       ↓
Aggregate Findings
       ↓
Reason Across Sources
       ↓
Generate Synthesized Answer
       ↓
Provide Citations
```

---

## Project Structure

```text
src/
│
├── agent.py
├── config.py
├── docling_loader.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── hybrid_retriever.py
├── chunking.py
│
├── llm/
│   ├── base.py
│   ├── factory.py
│   └── gemini_provider.py
│
└── memory/
    └── conversation_memory.py

tests/
│
├── test_ingest.py
├── test_vector_store.py
├── test_agent.py
└── retrieval_benchmarks.py
```

---

## Highlights

* Enterprise-style architecture
* Agentic RAG workflow
* Hybrid retrieval + reranking
* Multi-document intelligence
* Source-grounded generation
* Explainable responses
* Scalable design
* Modular component architecture

---

## Future Enhancements

* Knowledge graph integration
* Multi-modal document understanding
* Long-term memory
* Autonomous report generation
* Retrieval evaluation dashboards
* Human feedback loops
* Multi-agent orchestration

---

## Author

**Akkshit Gupta**

