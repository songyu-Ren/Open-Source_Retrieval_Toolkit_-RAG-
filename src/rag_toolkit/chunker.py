from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .logging import get_logger
from .loaders import Document

logger = get_logger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str


def chunk_text(doc: Document, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    text = doc.text or ""
    chunks: List[Chunk] = []
    i = 0
    idx = 0
    while i < len(text):
        start = i
        end = min(i + chunk_size, len(text))
        chunk_text = text[start:end]
        chunk_id = f"{doc.doc_id}:{idx}"
        chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, start=start, end=end, text=chunk_text))
        idx += 1
        if end == len(text):
            break
        i = end - chunk_overlap
        i = max(i, i + 1)  # ensure progress
    return chunks


def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> pd.DataFrame:
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc, chunk_size, chunk_overlap))
    logger.info(f"Created {len(all_chunks)} chunks from {len(docs)} documents")
    df = pd.DataFrame([c.__dict__ for c in all_chunks])
    return df


def persist_chunks(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)
    logger.info(f"Saved chunks to {path}")