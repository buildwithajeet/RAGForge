import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.documents import Document
import wikipedia
from config import OLLAMA_MODEL
# state graph definition
class RagState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    retry_count: int
    
# set up components
print("Loading docs...")
page = wikipedia.page("Natural language processing", auto_suggest=False)
raw_doc = [Document(page_content=page.content, metadata={"source": "wikipedia"})]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75, separators=["\n\n", "\n", ".", " ", ""])
print("Splitting docs...")

chunks = splitter.split_documents(raw_doc)
print("chunck len", len(chunks))
print("embeddings...")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory= "./chroma_langgraph"
)

dense_retriever = vector_store.as_retriever(search_kwargs={"k" : 10})

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10
hybrid_retrienver = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]
)

print("llm model", OLLAMA_MODEL)
llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)


# Nodes

def retrieve_node(state: RagState):
    docs = hybrid_retrienver.invoke(state["question"])
    print(f"[retrieve] got {len(docs)} chunks")
    return {"documents": docs}

def grade_node(state: RagState):
    print(f"[grade] grading {len(state['documents'])} docs...")
    grade_prompt = """Is this document relevant to the question?
    Question: {question}
    Document: {document}
    Answer with ONLY 'relevant' or 'irrelevant':"""

    relevant = []
    for doc in state["documents"]:
        prompt = grade_prompt.format(
            question=state["question"],
            document=doc.page_content[:300]
        )
        result = llm.invoke(prompt).strip().lower()
        print(f"  → {result[:20]} | {doc.page_content[:50]}...")
        if "relevant" in result.split():
            relevant.append(doc)

    print(f"[grade] kept {len(relevant)}/{len(state['documents'])}")
    return {"documents": relevant}

def generate_node(state: RagState):
    print(f"[generate] building answer...")
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    print(f"[generate] context len: {len(context)}")
    print(f"[generate] context preview: {context[:200]}")
    prompt = """
    You are helpful assistant. Use the following Context to answer the question.'.
    
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = prompt.format(context=context, question=state["question"])
    # prompt = PromptTemplate(input_variables=["context", "question"], template=prompt)
    print(f"[generate] full prompt:\n{prompt}")
    answer = llm.invoke(prompt)
    print(f"[generate] raw answer: {answer}")
    return {"answer": answer}

def rewrite_node(state: RagState):
    print(f"[rewrite] rewriting query...")
    prompt = """
    You are query rewriter.Rewrite the question below to be more specific and better for document retrieval.Return ONLY the rewritten question, nothing else.
    Original question: {question}
    Rewritten Question:"""
    prompt = prompt.format(question=state["question"])
    state["retry_count"] += 1
    print(f"[rewrite] full prompt:\n{prompt}")
    rewrite_query = llm.invoke(prompt)
    print(f"[rewrite] raw rewrite_query: {rewrite_query}")
    state["question"] = rewrite_query
    return {"question": rewrite_query, "retry_count": state["retry_count"] + 1}

def decide_function(state: RagState):
    docs = state["documents"]
    retry_count = state["retry_count"]
    if len(docs) > 0:
        return "generate"
    elif retry_count > 2:
        return "generate"
    else:
        return "rewrite"
# build graph

graph = StateGraph(RagState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("grade",    grade_node)
graph.add_node("rewrite",  rewrite_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges(
    "grade",
    decide_function,
    {
        "generate" : "generate",
        "rewrite"  : "rewrite"
    }
)
graph.add_edge("rewrite",  "retrieve")
graph.add_edge("generate", END)


pipeline = graph.compile()

print("Graph compiled!")

question = "What is natural language processing and how does it work?"
question = "Who is the President of the United States?"
result = pipeline.invoke({
    "question"  : question,
    "documents" : [],
    "answer"    : "",
    "retry_count": 0
})

print(f"\n{'='*50}")
print(f"Answer: {result['answer']}")
