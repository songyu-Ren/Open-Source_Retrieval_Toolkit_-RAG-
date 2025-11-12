from __future__ import annotations

import json
import time
from typing import Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from .config import load_settings
from .chains import build_chain
from .graphs import build_graph
from .metrics import observe_chain
from .logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _select_engine(engine: str):
    if engine == "langgraph":
        return build_graph()
    return build_chain()


@router.post("/chain_query")
def chain_query(payload: Dict) -> JSONResponse:
    s = load_settings()
    engine = payload.get("engine", s.orchestration.get("engine", "langchain"))
    stream = bool(payload.get("stream", s.orchestration.get("stream", False)))
    q = payload.get("query", "")
    k = int(payload.get("k", s.retrieval.get("k", 5)))
    t0 = time.time()
    chain = _select_engine(engine)
    result = chain.invoke({"query": q, "k": k, "stream": False})
    latency = time.time() - t0
    observe_chain(engine, "200", latency, None)
    return JSONResponse(content=result)


@router.post("/chain_stream")
def chain_stream(payload: Dict):
    s = load_settings()
    engine = payload.get("engine", s.orchestration.get("engine", "langchain"))
    q = payload.get("query", "")
    k = int(payload.get("k", s.retrieval.get("k", 5)))
    chain = _select_engine(engine)

    def _gen():
        t0 = time.time()
        summary = {}
        gen = chain.stream({"query": q, "k": k, "stream": True})
        while True:
            try:
                tok = next(gen)
                yield tok
            except StopIteration as e:
                summary = e.value or {}
                break
        latency = time.time() - t0
        observe_chain(engine, "200", latency, None)
        yield "\n" + json.dumps(summary)

    return StreamingResponse(_gen(), media_type="text/plain")
