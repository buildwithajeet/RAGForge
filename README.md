# 🔥 RAGForge

> Production-grade Retrieval-Augmented Generation (RAG) application built with LangChain, ChromaDB, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What is RAGForge?

RAGForge is a production-ready RAG system that answers questions from your documents with high accuracy and zero hallucination. It combines hybrid retrieval, cross-encoder reranking, and streaming generation into a clean chat interface.

Built as a portfolio project to demonstrate advanced RAG engineering skills — not just basic vector search.

---

## Features

- **Hybrid Retrieval** — BM25 + ChromaDB dense search via EnsembleRetriever
- **Metadata Filtering** — filter by file type at query time
- **Parent-Child Chunking** — small chunks for retrieval, large chunks for LLM context
- **Conversational Memory** — multi-turn chat with history passed to LLM
- **Cross-Encoder Reranking** — ms-marco-MiniLM re-scores top 20, passes best 5 to LLM
- **Streaming Responses** — tokens stream word by word in real time
- **Hallucination Control** — strict prompt keeps answers grounded in context
- **Source Attribution** — every answer shows which document it came from
- **PDF + Wikipedia Support** — upload PDFs or load Wikipedia articles
- **Config-based LLM** — switch between Gemini and Ollama via environment variable
- **RAGAS Evaluated** — pipeline measured with objective quality metrics
<!-- - **RAGAS Evaluated** — pipeline measured with objective quality metrics -->

---

## RAG Pipeline

```
User Query
    │
    ▼
Chat History (last 5 turns) + Active Metadata Filter
    │
    ▼
Hybrid Retriever (BM25 + ChromaDB, top 20 candidates)
    │
    ▼
Cross-Encoder Reranker (ms-marco-MiniLM, top 5 selected)
    │
    ▼
Parent Chunk Fetch (rich context for LLM)
    │
    ▼
Strict Context Prompt + Chat History
    │
    ▼
LLM (Gemini / Ollama) — streaming
    │
    ▼
Grounded Answer + Source Attribution
```
```

---

<!-- ## RAGAS Evaluation Scores

Evaluated on 10 question-answer pairs using RAG, LLM, and Vector DB knowledge base.

| Metric | Score | Status |
|---|---|---|
| Context Precision | 0.969 | Excellent ✅ |
| Context Recall | 0.819 | Good ✅ |
| Answer Relevancy | 0.726 | Decent ⚠️ |
| Faithfulness | 0.378 | Improving 🔧 | -->

> Faithfulness is being improved via stricter prompt engineering and CRAG pattern (in progress).

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Gemini API / Ollama (llama3.2) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector store | ChromaDB |
| Sparse retrieval | BM25 (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Orchestration | LangChain 1.2 |
| Evaluation | RAGAS |
| Config | python-dotenv |
| PDF parsing | pypdf |
<!-- | Evaluation | RAGAS | -->

---

## Project Structure

```
RAGForge/
├── app.py                  ← Streamlit UI + session + KB registry
├── config.py               ← all settings in one place
├── pipeline/
│   ├── __init__.py
│   ├── loader.py           ← PDF + Wikipedia loading + parent-child chunking
│   ├── retriever.py        ← hybrid retrieval + reranking + metadata filtering
│   └── generator.py        ← LLM + prompt + streaming + chat history
├── .streamlit/
│   └── config.toml         ← suppress warnings
├── requirements.txt
├── crag_pipeline.py        ← Corrective Rag Pipeline
└── README.md
```
```

---

## Setup

### Prerequisites
- Python 3.12+
- Ollama installed (for local LLM) — [ollama.ai](https://ollama.ai)
- Gemini API key (for cloud LLM) — [aistudio.google.com](https://aistudio.google.com)

### Installation
```bash
# clone the repo
git clone https://github.com/yourusername/RAGForge.git
cd RAGForge

# create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root:
```env
LLM_PROVIDER=gemini          # gemini | ollama
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
OLLAMA_MODEL=llama3.2:latest
RETRIEVER_K=20
RERANKER_TOP_N=5
```

### Run
```bash
# pull Ollama models (if using Ollama)
ollama pull llama3.2
ollama pull nomic-embed-text

# start the app
streamlit run app.py
if warning is showing in the terminal then run this 
streamlit run app.py --server.fileWatcherType none
```

Open `http://localhost:8501` in your browser.

---

## How to Use

1. Open the app in your browser
2. In the sidebar — choose **Wikipedia** or **Upload file**
3. Enter topics or upload a PDF
4. Click **Build Knowledge Base**
5. Wait for indexing to complete
6. Ask any question in the chat input
7. Get streaming answers with source attribution

---

## Key Design Decisions

**Why hybrid retrieval?**
Dense-only retrieval misses exact keyword matches (product codes, names, IDs). BM25-only misses semantic intent. Hybrid with equal weights [0.5, 0.5] gets the best of both.

**Why reranking?**
Bi-encoders embed query and document separately — fast but imprecise. Cross-encoders read them together — slower but much more accurate. Retrieving 20 and reranking to 5 gives accuracy without sacrificing speed.

**Why temperature=0?**
RAG needs factual, consistent answers — not creativity. Temperature=0 makes the LLM deterministic and reduces hallucination risk.

**Why config.py?**
Single source of truth for all settings. Change LLM provider, chunk size, or retrieval parameters in one place without touching pipeline code.

---

## Roadmap

- [x] Hybrid Retrieval (BM25 + Dense)
- [x] Cross-encoder reranking
- [x] Streaming responses
- [x] Conversational memory (multi-turn context)
- [x] Parent-child chunking
- [x] Metadata filtering
- [x] RAGAS evaluation
- [x] LangGraph CRAG pattern (self-correcting retrieval)
- [x] HuggingFace Spaces deployment
- [ ] DOCX support
- [ ] Persistent memory (MySQL)
- [ ] Docker deployment

---

## What I Learned Building This

**RAG Pipeline Design**
- Hybrid search (BM25 + Dense) consistently outperforms dense-only retrieval,
  especially for domain-specific queries with exact keywords
- Retrieving top 20 and reranking to top 5 gives better results than
  retrieving top 5 directly — the reranker needs candidates to work with
- chunk_size and chunk_overlap have more impact on answer quality
  than embedding model choice
- Parent-child chunking solves the precision vs context tradeoff —
  small chunks for retrieval, large chunks for LLM generation

**Evaluation**
- RAGAS evaluation revealed faithfulness as the critical gap — not retrieval quality
- Without evaluation you are guessing — RAGAS gives you numbers to optimize against
- Building a 10-20 question test dataset before optimizing saves significant time

**LangGraph + Agentic RAG**
- Linear RAG pipelines fail silently — LangGraph adds self-correction
- CRAG pattern (retrieve → grade → rewrite → retry) significantly reduces
  irrelevant answers
- Nodes should return partial state dicts — LangGraph handles merging

**Production Lessons**
- Metadata filtering must happen at query time not build time
- Conversational memory must be explicitly passed to LLM every call —
  LLMs are stateless by design
- Empty ChromaDB filter dict crashes — always pass None instead of {}
- Switching embedding models requires deleting old ChromaDB collections
- Temperature=0 is non-negotiable for RAG — creativity causes hallucination

**Debugging Insights**
- Most RAG failures are retrieval failures not LLM failures
- Always check chunk content before blaming the embedding model
- PDF raw bytes vs extracted text is the most common loader bug

---

**Why parent-child chunking?**
Small chunks (200 chars) give precise similarity matching during retrieval.
Large parent chunks (1000 chars) give the LLM rich context to generate 
complete answers. One chunk size can't satisfy both needs.

**Why metadata filtering at query time?**
Filters applied at build time are fixed forever. Query-time filtering 
lets users dynamically narrow search scope per question — 
search only PDFs, only specific sources, or all documents.


## License

MIT License — free to use, modify, and distribute.

---

## Author

Built by **Ajeet** — Senior Backend Engineer learning AI in public.

- 🔗 LinkedIn: [Ajeet Yadav](https://www.linkedin.com/in/ajeet-yadav-55a0861a0/)
- 🐙 GitHub: [buildwithajeet](https://github.com/buildwithajeet)
- 📸 Instagram: [buildwithajeet](https://instagram.com/buildwithajeet)
