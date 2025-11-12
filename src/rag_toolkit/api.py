from __future__ import annotations

import time
from typing import Dict, List

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from .config import load_settings
from .embedder import Embedder
from .index_store import IndexStore
from .retrieval import Retriever
from .llm import get_llm_client
from .metrics import observe_request, rag_query_score, metrics_response
from .logging import get_logger
from .api_chain import router as chain_router

logger = get_logger(__name__)

app = FastAPI(title="Open-Source RAG Toolkit", version="0.1.0")
app.include_router(chain_router)

_settings = load_settings()
_embedder = Embedder(_settings.embedding, seed=_settings.seed)
_store = IndexStore(_settings.paths.index_path, _settings.paths.index_meta_path)
try:
    _store.load()
    _retriever = Retriever(_store, _embedder)
except Exception:
    _retriever = None


@app.get("/health")
def health() -> Dict:
    status = {
        "index_loaded": _retriever is not None,
        "config": {
            "embedding_model": _settings.embedding.model_name,
            "chunk_size": _settings.chunking.chunk_size,
            "chunk_overlap": _settings.chunking.chunk_overlap,
            "server_port": _settings.server.port,
        },
    }
    observe_request("/health", "GET", "200", 0.0)
    return status


@app.post("/query")
def post_query(payload: Dict) -> JSONResponse:
    t0 = time.time()
    query = payload.get("query", "")
    k = int(payload.get("k", _settings.retrieval.get("k_default", 5)))
    use_llm = bool(payload.get("llm", False))
    if _retriever is None:
        observe_request("/query", "POST", "503", time.time() - t0)
        return JSONResponse(status_code=503, content={"error": "index not loaded"})
    results = _retriever.search(query, k)
    for r in results:
        try:
            rag_query_score.observe(max(0.0, min(1.0, r.get("score", 0.0))))
        except Exception:
            pass
    answer = None
    if use_llm:
        client = get_llm_client(_settings.llm.enabled, _settings.llm.api_base, _settings.llm.model)
        contexts = [r["text"] for r in results]
        answer = client.answer(query, contexts)
    latency = time.time() - t0
    observe_request("/query", "POST", "200", latency)
    return JSONResponse(content={"latency": latency, "results": results, "answer": answer})


@app.get("/metrics")
def get_metrics() -> Response:
    content, status, headers = metrics_response()
    return Response(content=content, status_code=status, media_type=headers["Content-Type"])
