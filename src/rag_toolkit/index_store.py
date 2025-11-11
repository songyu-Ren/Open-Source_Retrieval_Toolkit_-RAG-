from __future__ import annotations

import json
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)


class IndexStore:
    def __init__(self, index_path: str, meta_path: str) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.meta: List[Dict] = []

    def build(self, embeddings: np.ndarray, chunks_df: pd.DataFrame) -> None:
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        logger.info(f"Built FAISS IndexFlatIP with {embeddings.shape[0]} vectors, dim={d}")
        # Persist meta as JSONL
        self.meta = []
        for _, row in chunks_df.iterrows():
            self.meta.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "text": row["text"],
                }
            )

    def save(self) -> None:
        assert self.index is not None
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m) + "\n")
        logger.info(f"Saved index to {self.index_path} and meta to {self.meta_path}")

    def load(self) -> None:
        self.index = faiss.read_index(self.index_path)
        self.meta = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))
        logger.info(f"Loaded index from {self.index_path} with {len(self.meta)} meta entries")