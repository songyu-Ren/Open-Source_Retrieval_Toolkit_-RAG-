from __future__ import annotations

import time
from typing import Callable, Dict

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

rag_requests_total = Counter(
    "rag_requests_total",
    "Total RAG requests",
    labelnames=("endpoint", "method", "status"),
)

rag_request_latency_seconds = Histogram(
    "rag_request_latency_seconds",
    "Latency of RAG requests in seconds",
    labelnames=("endpoint",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

rag_query_score = Histogram(
    "rag_query_score",
    "Distribution of retrieval scores (0..1)",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

rag_chain_requests_total = Counter(
    "rag_chain_requests_total",
    "Total chain/graph requests",
    labelnames=("engine", "status"),
)

rag_chain_latency_seconds = Histogram(
    "rag_chain_latency_seconds",
    "Latency of chain/graph requests in seconds",
    labelnames=("engine",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

rag_chain_tokens_total = Counter(
    "rag_chain_tokens_total",
    "Total tokens processed/emitted",
    labelnames=("engine", "role"),
)

rag_query_rewritten_total = Counter(
    "rag_query_rewritten_total",
    "Total count of query rewrites",
)


def observe_request(endpoint: str, method: str, status: str, latency: float) -> None:
    rag_requests_total.labels(endpoint=endpoint, method=method, status=status).inc()
    rag_request_latency_seconds.labels(endpoint=endpoint).observe(latency)


def observe_chain(engine: str, status: str, latency: float, tokens: Dict[str, int] | None = None) -> None:
    rag_chain_requests_total.labels(engine=engine, status=status).inc()
    rag_chain_latency_seconds.labels(engine=engine).observe(latency)
    if tokens:
        for role, count in tokens.items():
            rag_chain_tokens_total.labels(engine=engine, role=role).inc(count)


def metrics_response() -> tuple:
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}
