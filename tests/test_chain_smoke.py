import os
import numpy as np
import pandas as pd

from rag_toolkit.config import load_settings
from rag_toolkit.loaders import load_documents
from rag_toolkit.chunker import chunk_documents, persist_chunks
from rag_toolkit.embedder import Embedder
from rag_toolkit.index_store import IndexStore
from rag_toolkit.chains import build_chain


def test_chain_smoke(monkeypatch):
    monkeypatch.setenv("RAG_SETTINGS", "config/test_settings.yaml")
    s = load_settings()

    os.makedirs(s.paths.artifacts_dir, exist_ok=True)
    docs = load_documents(s.paths.raw_data_dir)
    df = chunk_documents(docs, s.chunking.chunk_size, s.chunking.chunk_overlap)
    persist_chunks(df, s.paths.chunks_path)

    emb = Embedder(s.embedding, seed=s.seed)
    vectors = emb.embed_texts(df["text"].tolist(), batch_size=s.embedding.batch_size)
    np.save(s.paths.embeddings_path, vectors)

    store = IndexStore(s.paths.index_path, s.paths.index_meta_path)
    store.build(vectors, df)
    store.save()

    chain = build_chain()
    res = chain.invoke({"query": "what is in these docs?", "k": 3, "stream": False})
    assert isinstance(res, dict)
    assert res.get("engine") == "langchain"
    assert isinstance(res.get("citations"), list) and len(res.get("citations")) > 0
