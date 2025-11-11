import os
from rag_toolkit.loaders import load_documents
from rag_toolkit.chunker import chunk_documents
from rag_toolkit.config import load_settings


def test_chunker_creates_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_SETTINGS", "config/test_settings.yaml")
    s = load_settings()
    docs = load_documents(s.paths.raw_data_dir)
    df = chunk_documents(docs, s.chunking.chunk_size, s.chunking.chunk_overlap)
    assert len(df) > 0
    assert {"chunk_id", "doc_id", "start", "end", "text"}.issubset(df.columns)