import os
from rag_toolkit.config import load_settings
from rag_toolkit.embedder import Embedder
from rag_toolkit.index_store import IndexStore
from rag_toolkit.retrieval import Retriever
from rag_toolkit.eval import evaluate


def test_retrieval_and_metrics(monkeypatch):
    monkeypatch.setenv("RAG_SETTINGS", "config/test_settings.yaml")
    s = load_settings()

    # Load index and run a query
    store = IndexStore(s.paths.index_path, s.paths.index_meta_path)
    store.load()
    emb = Embedder(s.embedding, seed=s.seed)
    retr = Retriever(store, emb)

    results = retr.search("RAG pipeline", k=5)
    assert isinstance(results, list) and len(results) > 0
    assert 0.0 <= results[0]["score"] <= 1.0

    # Prepare evaluation inputs
    qrels = {}
    with open("data/qrels.tsv", "r", encoding="utf-8") as f:
        for line in f:
            qid, doc_id, rel = line.strip().split("\t")
            qrels.setdefault(qid, {})[doc_id] = float(rel)
    queries = []
    with open("data/queries.tsv", "r", encoding="utf-8") as f:
        for line in f:
            qid, text = line.strip().split("\t")
            queries.append({"qid": qid, "text": text})

    metrics = evaluate(queries, retr.search, qrels, k=5)
    assert 0.0 <= metrics["nDCG@k"] <= 1.0
    assert 0.0 <= metrics["MRR"] <= 1.0