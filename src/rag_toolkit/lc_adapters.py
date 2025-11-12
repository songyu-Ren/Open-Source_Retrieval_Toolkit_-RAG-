from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document

from .embedder import Embedder
from .index_store import IndexStore
from .retrieval import Retriever


def chunk_to_document(chunk: Dict) -> Document:
    return Document(page_content=chunk["text"], metadata={
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": chunk.get("doc_id"),
        "start": chunk.get("start"),
        "end": chunk.get("end"),
        "score": chunk.get("score"),
    })


class FAISSRetrieverAdapter:
    def __init__(self, store: IndexStore, embedder: Embedder, k: int) -> None:
        self.store = store
        self.embedder = embedder
        self.k = k
        self.retriever = Retriever(store, embedder)

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.retriever.search(query, self.k)
        return [chunk_to_document(r) for r in results]

    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


def citations_from_documents(docs: List[Document]) -> List[Dict]:
    cits: List[Dict] = []
    for d in docs:
        m = d.metadata
        cits.append({
            "doc_id": m.get("doc_id"),
            "start": m.get("start"),
            "end": m.get("end"),
        })
    return cits
