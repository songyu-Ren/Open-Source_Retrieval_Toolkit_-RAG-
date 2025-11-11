import os
import json
import numpy as np
import pandas as pd

from rag_toolkit.config import load_settings
from rag_toolkit.loaders import load_documents
from rag_toolkit.chunker import chunk_documents, persist_chunks
from rag_toolkit.embedder import Embedder
from rag_toolkit.index_store import IndexStore


def test_embed_and_index(monkeypatch):
    monkeypatch.setenv("RAG_SETTINGS", "config/test_settings.yaml")
    s = load_settings()

    # Prepare chunks
    docs = load_documents(s.paths.raw_data_dir)
    df = chunk_documents(docs, s.chunking.chunk_size, s.chunking.chunk_overlap)
    persist_chunks(df, s.paths.chunks_path)

    # Embed with dummy embedder for deterministic outputs
    emb = Embedder(s.embedding, seed=s.seed)
    vectors = emb.embed_texts(df["text"].tolist(), batch_size=s.embedding.batch_size)
    np.save(s.paths.embeddings_path, vectors)

    # Build and persist index
    store = IndexStore(s.paths.index_path, s.paths.index_meta_path)
    store.build(vectors, df)
    store.save()

    assert os.path.exists(s.paths.index_path)
    assert os.path.exists(s.paths.index_meta_path)
    assert os.path.exists(s.paths.embeddings_path)

    df2 = pd.read_parquet(s.paths.chunks_path)
    with open(s.paths.index_meta_path, "r", encoding="utf-8") as f:
        meta_lines = list(f)
    assert len(meta_lines) == len(df2)