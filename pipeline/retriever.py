from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from config import RETRIEVER_K, RERANKER_TOP_N, BM25_WEIGHT, DENSE_WEIGHT, EMBEDDING_MODEL


print(f"RETRIEVER_K: {RETRIEVER_K}, RERANKER_TOP_N: {RERANKER_TOP_N}, BM25_WEIGHT: {BM25_WEIGHT}, DENSE_WEIGHT: {DENSE_WEIGHT}, EMBEDDING_MODEL: {EMBEDDING_MODEL}")
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
@lru_cache(maxsize=1)
def get_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def build_hybrid_retriever(chunks: list[Document], persist_dir:str = './chroma_db') -> EnsembleRetriever:
     # debug — add these 3 lines
    # print(f"Total chunks received: {len(chunks)}")
    # print(f"First chunk preview: {chunks[0].page_content[:100] if chunks else 'EMPTY'}")
    
    chunks = [c for c in chunks if c.page_content.strip()]
    # print(f"Chunks after filtering empty: {len(chunks)}")
    embedding_model = get_embeddings()
    
    # test embedding on one chunk first
    # test_embed = embedding_model.embed_query("test sentence")
    # print(f"Test embedding length: {len(test_embed)}")
    # print(f"Test embedding sample: {test_embed[:3]}")
    vector_store = Chroma.from_documents(
        documents= chunks,
        embedding=embedding_model,
        persist_directory= persist_dir
    )
    
    dense_retriver = vector_store.as_retriever(search_kwargs={"k" : RETRIEVER_K}) #query
    # dense_retriver = vector_store.as_retriever(search_kwargs={"k" : RETRIEVER_K}) #query
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = RETRIEVER_K
    
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriver],
        weights=[BM25_WEIGHT, DENSE_WEIGHT]
    )
    
    return hybrid_retriever


def rerank(query: str, docs: list[Document], top_n: int = RERANKER_TOP_N) -> list[Document]:
    model  = get_cross_encoder()
    pairs  = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


""" def retrieve(query: str, retriever, top_n: int = RERANKER_TOP_N) -> list[Document]:
    retrieved = retriever.invoke(query)
    reranked  = rerank(query, retrieved, top_n=top_n)
    return reranked """
    
def retrieve(query: str, retriever, top_n: int = RERANKER_TOP_N, filters: dict = {}) -> list[Document]:
    # build filter
    where_clauses = {}
    if filters:
        for key, value in filters.items():
            if value:
                where_clauses[key] = value

    # apply filter to dense retriever (index 1 in EnsembleRetriever)
    dense = retriever.retrievers[1]
    if where_clauses and where_clauses != {}:
        dense.search_kwargs["filter"] = where_clauses
    else:
        dense.search_kwargs.pop("filter", None)

    retrieved = retriever.invoke(query)
    reranked  = rerank(query, retrieved, top_n=top_n)
    return reranked
