from functools import lru_cache

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAI
from config import GEMINI_API_KEY, LLM_PROVIDER, GEMINI_MODEL, OLLAMA_MODEL

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a strict context-based answer bot.

RULES:
- Answer ONLY using the CONTEXT below
- NEVER use your own knowledge
- If answer is not in CONTEXT say: "I don't know based on the provided context."
- Be concise and direct

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)
@lru_cache(maxsize=1)
def get_llm():
    if LLM_PROVIDER == "gemini":
        return GoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, google_api_key=GEMINI_API_KEY)
    else:
        return OllamaLLM(model=OLLAMA_MODEL, temperature=0)
def build_context(docs: list[Document]) -> str:
    return "\n\n".join([
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
        for doc in docs
    ])


def generate(question: str, docs: list[Document]) -> str:
    llm = get_llm()
    context = build_context(docs)
    prompt  = PROMPT.format(context=context, question=question)
    return llm.invoke(prompt)


def generate_stream(question: str, docs: list[Document]):
    llm = get_llm()
    context = build_context(docs)
    prompt  = PROMPT.format(context=context, question=question)
    for chunk in llm.stream(prompt):
        yield chunk