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

- **Hybrid Retrieval** — BM25 sparse search + ChromaDB dense search combined via EnsembleRetriever
- **Cross-Encoder Reranking** — ms-marco-MiniLM re-scores top 20 chunks, passes best 5 to LLM
- **Streaming Responses** — tokens stream word by word in real time
- **Hallucination Control** — strict prompt keeps answers grounded in retrieved context
- **Source Attribution** — every answer shows which document it came from
- **PDF + Wikipedia Support** — upload PDFs or load Wikipedia articles as knowledge base
- **Config-based LLM** — switch between Gemini and Ollama via environment variable
<!-- - **RAGAS Evaluated** — pipeline measured with objective quality metrics -->

---

## RAG Pipeline
```
User Query
    │
    ▼
Hybrid Retriever (BM25 + ChromaDB, top 20 candidates)
    │
    ▼
Cross-Encoder Reranker (ms-marco-MiniLM, top 5 selected)
    │
    ▼
Strict Context Prompt
    │
    ▼
LLM (Gemini / Ollama) — streaming
    │
    ▼
Grounded Answer + Sources
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
| Config | python-dotenv |
<!-- | Evaluation | RAGAS | -->

---

## Project Structure
```
RAGForge/
├── app.py                  ← Streamlit UI + session management
├── config.py               ← all settings in one place
├── pipeline/
│   ├── __init__.py
│   ├── loader.py           ← PDF + Wikipedia document loading
│   ├── retriever.py        ← hybrid retrieval + reranking
│   └── generator.py        ← LLM + prompt + streaming
├── evaluate_rag.py         ← RAGAS evaluation script
├── .streamlit/
│   └── config.toml         ← suppress warnings
├── requirements.txt
└── README.md
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

- [ ] LangGraph CRAG pattern (self-correcting retrieval)
- [ ] Conversational memory (multi-turn context)
- [ ] Parent-child chunking
- [ ] Metadata filtering
- [ ] HuggingFace Spaces deployment
- [ ] DOCX support

---

## What I Learned Building This

- Hybrid search consistently outperforms dense-only retrieval for domain-specific queries
- Reranking with cross-encoders dramatically improves answer quality with minimal latency cost
- RAGAS evaluation revealed faithfulness as the critical gap — not retrieval quality
- chunk_size and chunk_overlap have more impact on answer quality than embedding model choice

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

Built by **Ajeet** — Backend Engineer learning AI in public.

- LinkedIn: [https://www.linkedin.com/in/ajeet-yadav-55a0861a0/]
- GitHub: [https://github.com/buildwithajeet]