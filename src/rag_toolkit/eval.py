from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import mlflow

from .logging import get_logger

logger = get_logger(__name__)


def dcg(rels: List[float]) -> float:
    import math

    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(rels))


def ndcg_at_k(retrieved_doc_ids: List[str], qrels: Dict[str, float], k: int) -> float:
    rels = [qrels.get(doc_id, 0.0) for doc_id in retrieved_doc_ids[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    denom = dcg(ideal) or 1e-12
    return dcg(rels) / denom


def mrr(retrieved_doc_ids: List[str], qrels: Dict[str, float]) -> float:
    for i, doc_id in enumerate(retrieved_doc_ids):
        if qrels.get(doc_id, 0.0) > 0.0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(queries: List[Dict], retrieve_fn, qrels: Dict[str, Dict[str, float]], k: int) -> Dict:
    metrics = {"nDCG@k": [], "MRR": []}
    for q in queries:
        qid = q["qid"]
        qtext = q["text"]
        results = retrieve_fn(qtext, k)
        # convert chunk-level to doc-level ranking
        doc_order: List[str] = []
        seen = set()
        for r in results:
            doc_id = r["doc_id"]
            if doc_id not in seen:
                doc_order.append(doc_id)
                seen.add(doc_id)
        metrics["nDCG@k"].append(ndcg_at_k(doc_order, qrels.get(qid, {}), k))
        metrics["MRR"].append(mrr(doc_order, qrels.get(qid, {})))

    summary = {"nDCG@k": float(sum(metrics["nDCG@k"]) / max(len(metrics["nDCG@k"]), 1)),
               "MRR": float(sum(metrics["MRR"]) / max(len(metrics["MRR"]), 1))}
    return summary


def save_eval(path: str, result: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved evaluation metrics to {path}")


def log_mlflow(params: Dict, metrics: Dict) -> None:
    mlflow.set_tracking_uri(params.get("tracking_uri", "./mlruns"))
    with mlflow.start_run(run_name="rag-eval"):
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))