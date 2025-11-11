from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .embedder import Embedder
from .index_store import IndexStore
from .logging import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(self, store: IndexStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, query: str, k: int) -> List[Dict]:
        assert self.store.index is not None, "Index not loaded"
        qv = self.embedder.embed_texts([query])
        scores, idxs = self.store.index.search(qv, k)
        idxs = idxs[0]
        scores = scores[0]
        results: List[Dict] = []
        for i, score in zip(idxs, scores):
            if i < 0 or i >= len(self.store.meta):
                continue
            m = self.store.meta[i]
            results.append({
                "chunk_id": m["chunk_id"],
                "doc_id": m["doc_id"],
                "start": m["start"],
                "end": m["end"],
                "text": m["text"],
                "score": float(score),
            })
        return results

    def rerank(self, results: List[Dict]) -> List[Dict]:
        # Stub for reranking extension point
        return results