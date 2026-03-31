import hashlib
import streamlit as st
from pipeline.loader import load_wikipedia_topics, load_uploaded_file, chunk_docs
from pipeline.retriever import build_hybrid_retriever, retrieve
from pipeline.generator import generate_stream

st.set_page_config(page_title="RAGForge", page_icon="🤖", layout="wide")

st.title("RAGForge")
st.caption("Production RAG — Hybrid Retrieval · Reranking · Streaming")

# ── session state ──────────────────────────────────────
if "kb_registry" not in st.session_state:
    st.session_state.kb_registry = {}

if "kb_name" not in st.session_state:
    st.session_state.kb_name = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

# ── cache retriever ─────────────────────────────────────
@st.cache_resource
def get_retriever(chunks, persist_dir):
    return build_hybrid_retriever(chunks, persist_dir)

# ── sidebar ────────────────────────────────────────────
with st.sidebar:
    st.header("Knowledge Base")

    source = st.radio("Source", ["Wikipedia", "Upload files"])

    if source == "Wikipedia":
        wiki_topics = st.text_area(
            "Topics (one per line)",
            placeholder="Retrieval-augmented generation\nLarge language model\nVector database"
        )
    else:
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["txt", "pdf"],
            accept_multiple_files=True
        )

    build_btn = st.button("Build Knowledge Base", type="primary", use_container_width=True)

    # ── existing KB selector ───────────────────────────
    if st.session_state.kb_registry:
        st.divider()
        st.subheader("Your Knowledge Bases")

        selected_kb = st.selectbox(
            "Select KB",
            list(st.session_state.kb_registry.keys()),
            index=0
        )

        if selected_kb:
            st.session_state.kb_name = selected_kb
            kb = st.session_state.kb_registry[selected_kb]
            st.session_state.retriever = kb["retriever"]
            st.session_state.kb_ready = True

        if st.button("Delete KB", use_container_width=True):
            del st.session_state.kb_registry[selected_kb]
            st.session_state.kb_name = None
            st.session_state.kb_ready = False
            st.rerun()

    st.divider()
    st.subheader("Pipeline")
    st.markdown("""
    - Hybrid retrieval (BM25 + Dense)
    - Cross-encoder reranking
    - Streaming generation
    """)

# ── build knowledge base ───────────────────────────────
if build_btn:
    docs = []

    if source == "Wikipedia":
        if not wiki_topics.strip():
            st.sidebar.error("Please enter at least one topic.")
            st.stop()

        topics = [t.strip() for t in wiki_topics.split("\n") if t.strip()]

        with st.spinner("Loading Wikipedia..."):
            docs = load_wikipedia_topics(topics)

        kb_name = ", ".join(topics)

    else:
        if not uploaded_files:
            st.sidebar.error("Please upload at least one file.")
            st.stop()

        with st.spinner("Reading files..."):
            for file in uploaded_files:
                docs.extend(load_uploaded_file(file))

        kb_name = " + ".join([f.name for f in uploaded_files])

    if not docs:
        st.sidebar.error("No content loaded.")
        st.stop()

    with st.spinner("Chunking..."):
        chunks = chunk_docs(docs)

    kb_hash = hashlib.md5(kb_name.encode()).hexdigest()[:8]
    persist_dir = f"./chroma_db_{kb_hash}"

    with st.spinner("Building retriever..."):
        retriever = get_retriever(chunks, persist_dir)

    # ── save KB ────────────────────────────────────────
    st.session_state.kb_registry[kb_name] = {
        "retriever": retriever,
        "chat_history": []
    }

    st.session_state.kb_name = kb_name
    st.session_state.retriever = retriever
    st.session_state.kb_ready = True

    st.sidebar.success(f"KB Ready: {kb_name}")
    st.rerun()

# ── stop if no KB ──────────────────────────────────────
if not st.session_state.kb_ready:
    st.info("Build or select a knowledge base to start.")
    st.stop()

# ── current KB ─────────────────────────────────────────
kb = st.session_state.kb_registry[st.session_state.kb_name]
chat_history = kb["chat_history"]

# ── display chat ───────────────────────────────────────
for msg in chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# ── input ──────────────────────────────────────────────
question = st.chat_input("Ask anything...")

if question:
    # user message
    chat_history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            docs = retrieve(question, st.session_state.retriever, top_n=5)
            sources = list(set([d.metadata["source"] for d in docs]))

        answer = st.write_stream(generate_stream(question, docs))

        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"- {s}")

    chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })