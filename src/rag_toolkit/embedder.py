from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import EmbeddingCfg
from .logging import get_logger

logger = get_logger(__name__)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


@dataclass
class DummyEmbedder:
    dim: int = 384

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            rs = np.random.RandomState(int.from_bytes(h[:4], "little"))
            v = rs.rand(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            vecs.append(v)
        return np.vstack(vecs)


class Embedder:
    def __init__(self, cfg: EmbeddingCfg, seed: int = 42) -> None:
        self.cfg = cfg
        set_seeds(seed)
        self.model = None
        self.dummy = None
        if cfg.use_dummy:
            self.dummy = DummyEmbedder()
        else:
            from sentence_transformers import SentenceTransformer

            device = cfg.device or "cpu"
            self.model = SentenceTransformer(cfg.model_name, device=device)
            logger.info(f"Loaded sentence-transformers model {cfg.model_name} on {device}")

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        batch_size = batch_size or self.cfg.batch_size
        if self.dummy is not None:
            arr = self.dummy.encode(texts)
        else:
            arr = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False)
        if self.cfg.normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms
        return arr.astype(np.float32)