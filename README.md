# 📄 Agentic RAG PDF Assistant

An enterprise-grade **Agentic Retrieval-Augmented Generation (RAG)** system for conversational question answering over multiple PDFs using **LangGraph ReAct Agents**, **Hybrid Retrieval**, **ChromaDB**, and **persistent chat memory**.

The assistant intelligently decides when to:
- Search uploaded documents
- Search the web
- Perform calculations

while maintaining conversation history and providing grounded answers with source references.

---

## ✨ Features

- 📄 Multi-document PDF ingestion
- 🔍 OCR-aware parsing with Docling
- ✂️ Hierarchical chunking
- 🧠 BGE Embeddings
- 🗄️ ChromaDB Vector Store
- 🔎 BM25 Lexical Retrieval
- ⚡ Hybrid Retrieval (Dense + BM25)
- 🎯 Reciprocal Rank Fusion (RRF)
- 🏆 Cross-Encoder Reranking
- 🤖 LangGraph ReAct Agent
- 🛠️ Tool Calling
  - Document Search
  - Web Search
  - Calculator
- 💬 Persistent SQLite Chat Memory
- 📚 Source-grounded Responses
- 🌐 Multiple LLM Providers
  - Gemini
  - Groq
  - OpenRouter
- 🎨 Streamlit UI

---

# 🏗️ Architecture

```
                    User Question
                           │
                           ▼
                 LangGraph ReAct Agent
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
 Search Documents      Web Search       Calculator
        │
        ▼
 Hybrid Retriever
(Dense + BM25)
        │
        ▼
 Reciprocal Rank Fusion
        │
        ▼
 Cross Encoder Reranker
        │
        ▼
 Relevant Context
        │
        ▼
      LLM Response
        │
        ▼
 Grounded Answer + Sources
```

---

# 🔍 Retrieval Pipeline

```
User Query
     │
     ▼
Dense Retrieval (Embeddings)
     │
     ├─────────────┐
     ▼             ▼
 BM25         Dense Search
     │             │
     └──────┬──────┘
            ▼
 Reciprocal Rank Fusion
            ▼
 Cross Encoder Reranker
            ▼
 Final Retrieved Chunks
```

---

# 🤖 Agent Workflow

```
User
 │
 ▼
LangGraph ReAct Agent
 │
 ├── search_documents
 ├── web_search
 └── calculator
 │
 ▼
LLM
 │
 ▼
Final Answer
```

---

# 🛠 Tech Stack

- Python
- LangChain
- LangGraph
- Streamlit
- ChromaDB
- Docling
- Sentence Transformers
- BM25 Retriever
- Cross Encoder Reranker
- SQLite
- Gemini
- Groq
- OpenRouter

---

# 📂 Project Structure

```
src/
│
├── agent.py
├── db.py
├── hybrid_retriever.py
├── retriever.py
├── vector_store.py
├── embeddings.py
├── tools.py
│
├── llm/
│   ├── factory.py
│   ├── gemini_provider.py
│   └── groq_provider.py
│
└── ...
```

---

# 🚀 Installation

## Clone the repository

```bash
git clone https://github.com/akk-026/agentic-rag-pdf.git
cd agentic-rag-pdf
```

## Create a virtual environment

```bash
python -m venv venv
```

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configure environment variables

Create a `.env` file.

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

---

# ▶️ Running the Application

```bash
streamlit run app.py
```

---

# 🧪 Running Tests

```bash
python -m tests.test_ingest
python -m tests.test_vector_store
python -m tests.test_hybrid_retrieval
python -m tests.test_agent
```

---

# 💡 Example

### User

```
Where is Curiflow located?
```

### Assistant

```
Curiflow is located in Hiranandani Gardens,
Powai, Mumbai.

Sources:
• About Us and Our Work.pdf (Page 4)
```

---

# 🔮 Future Improvements

- True token streaming
- Query rewriting
- Retrieval grading
- RAG evaluation (RAGAS / DeepEval)
- FastAPI backend
- Docker support
- Authentication
- Cloud deployment

---

# 📄 License

This project is intended for educational and research purposes.

---

# ⭐ If you found this project useful, consider giving it a star!
