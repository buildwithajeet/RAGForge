import os
from dotenv import load_dotenv
load_dotenv()

# ── Embedding ──────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace, free, no API key

# ── LLM ───────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "ollama")  # gemini | ollama
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

# ── Retrieval ──────────────────────────────────────────
RETRIEVER_K     = int(os.getenv("RETRIEVER_K", "20"))
RERANKER_TOP_N  = int(os.getenv("RERANKER_TOP_N", "5"))
BM25_WEIGHT     = float(os.getenv("BM25_WEIGHT", "0.5"))
DENSE_WEIGHT    = float(os.getenv("DENSE_WEIGHT", "0.5"))

# ── ChromaDB ───────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
