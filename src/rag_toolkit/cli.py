from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import typer

from .config import load_settings
from .embedder import Embedder
from .index_store import IndexStore
from .loaders import load_documents
from .chunker import chunk_documents, persist_chunks
from .retrieval import Retriever
from .eval import evaluate, save_eval, log_mlflow
from .llm import get_llm_client
from .logging import get_logger

app = typer.Typer(help="RAG Toolkit CLI")
logger = get_logger(__name__)


@app.command()
def index(data: str = typer.Option("data/raw", help="Path to raw documents")) -> None:
    """Load files, chunk, embed, build FAISS, persist index and metadata."""
    s = load_settings()
    os.makedirs(s.paths.artifacts_dir, exist_ok=True)

    docs = load_documents(data)
    df = chunk_documents(docs, s.chunking.chunk_size, s.chunking.chunk_overlap)
    persist_chunks(df, s.paths.chunks_path)

    emb = Embedder(s.embedding, seed=s.seed)
    vectors = emb.embed_texts(df["text"].tolist(), batch_size=s.embedding.batch_size)
    np.save(s.paths.embeddings_path, vectors)

    store = IndexStore(s.paths.index_path, s.paths.index_meta_path)
    store.build(vectors, df)
    store.save()
    typer.echo(json.dumps({"chunks": len(df), "vectors": int(vectors.shape[0])}))


@app.command()
def query(q: str = typer.Option(..., "--q", help="Query text"), k: int = typer.Option(5, "--k"), llm: bool = typer.Option(False, help="Use LLM to answer")) -> None:
    """Load index, retrieve top-k, show texts/metadata, optional LLM."""
    s = load_settings()
    emb = Embedder(s.embedding, seed=s.seed)
    store = IndexStore(s.paths.index_path, s.paths.index_meta_path)
    store.load()
    retr = Retriever(store, emb)
    t0 = time.time()
    results = retr.search(q, k)
    latency = time.time() - t0
    typer.echo(json.dumps({"latency": latency, "results": results}, indent=2))

    if llm:
        client = get_llm_client(s.llm.enabled, s.llm.api_base, s.llm.model)
        contexts = [r["text"] for r in results]
        answer = client.answer(q, contexts)
        typer.echo(json.dumps({"answer": answer}))


@app.command()
def eval(qrels: str = typer.Option("data/qrels.tsv", help="Path to qrels.tsv"), queries: str = typer.Option("data/queries.tsv", help="Path to queries.tsv"), k: int = typer.Option(10, "--k")) -> None:
    """Compute nDCG@k and MRR; log to MLflow and save JSON."""
    s = load_settings()
    emb = Embedder(s.embedding, seed=s.seed)
    store = IndexStore(s.paths.index_path, s.paths.index_meta_path)
    store.load()
    retr = Retriever(store, emb)

    # Load evaluation data
    qrels_map = {}
    with open(qrels, "r", encoding="utf-8") as f:
        for line in f:
            qid, doc_id, rel = line.strip().split("\t")
            qrels_map.setdefault(qid, {})[doc_id] = float(rel)
    queries_list = []
    with open(queries, "r", encoding="utf-8") as f:
        for line in f:
            qid, text = line.strip().split("\t")
            queries_list.append({"qid": qid, "text": text})

    metrics = evaluate(queries_list, retr.search, qrels_map, k)
    save_eval(s.paths.eval_path, metrics)
    log_mlflow({"tracking_uri": s.mlflow.tracking_uri, "k": k}, metrics)
    typer.echo(json.dumps(metrics, indent=2))


# Hidden commands for DVC stages
@app.command(hidden=True)
def chunk(data: str = typer.Option("data/raw")) -> None:
    s = load_settings()
    os.makedirs(s.paths.artifacts_dir, exist_ok=True)
    docs = load_documents(data)
    df = chunk_documents(docs, s.chunking.chunk_size, s.chunking.chunk_overlap)
    persist_chunks(df, s.paths.chunks_path)


@app.command(hidden=True)
def embed() -> None:
    s = load_settings()
    df = pd.read_parquet(s.paths.chunks_path)
    emb = Embedder(s.embedding, seed=s.seed)
    vectors = emb.embed_texts(df["text"].tolist(), batch_size=s.embedding.batch_size)
    np.save(s.paths.embeddings_path, vectors)


@app.command(hidden=True)
def index_build() -> None:
    s = load_settings()
    df = pd.read_parquet(s.paths.chunks_path)
    vectors = np.load(s.paths.embeddings_path)
    store = IndexStore(s.paths.index_path, s.paths.index_meta_path)
    store.build(vectors, df)
    store.save()


def main():
    app()


if __name__ == "__main__":
    main()