import wikipedia
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import datetime
def load_wikipedia_topics(topics: list[str]) -> list[Document]:
    docs = []
    for topic in topics:
        try:
            page = wikipedia.page(topic, auto_suggest=False)
        except Exception:
            results = wikipedia.search(topic)
            page = wikipedia.page(results[0], auto_suggest=False)
        docs.append(Document(
            page_content=page.content,
            metadata={"source": topic, "type": "wikipedia"}
        ))
    return docs


def load_uploaded_file(file) -> list[Document]:
    filename = file.name.lower()

    # ── PDF ───────────────────────────────────────────
    if filename.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(file)
        docs   = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():  # skip blank pages
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file.name, "type": "pdf", "page": i + 1}
                ))
        return docs

    # ── TXT ───────────────────────────────────────────
    elif filename.endswith(".txt"):
        content = file.read().decode("utf-8", errors="ignore")
        return [Document(
            page_content=content,
            metadata={"source": file.name, "type": "txt"}
        )]

    # ── unsupported ───────────────────────────────────
    else:
        return []


def chunk_docs(documents: list[Document], file_type: str="pdf") -> tuple[list[Document], dict[str, Document]]:
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500,
    #     chunk_overlap=75,
    #     separators=["\n\n", "\n", ".", " ", ""]
    # )
    # return splitter.split_documents(documents)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    parent_chunks = parent_splitter.split_documents(documents)
    all_chunk_docs = []
    parent_docs = {}
    year = datetime.datetime.now().year
    for parent in parent_chunks:
        child_chunks = child_splitter.split_documents([parent])
        parent_id = str(uuid.uuid4())
        parent_docs[parent_id] = parent
        for child in child_chunks:
            child.metadata["parent_id"] = parent_id
            child.metadata['year'] = year
            child.metadata['file_type'] = file_type
            all_chunk_docs.append(child)
    return all_chunk_docs, parent_docs