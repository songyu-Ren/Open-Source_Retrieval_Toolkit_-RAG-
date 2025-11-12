from __future__ import annotations

import time
from typing import Dict, Generator, List

from jinja2 import Environment, FileSystemLoader
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .config import load_settings
from .embedder import Embedder
from .index_store import IndexStore
from .lc_adapters import FAISSRetrieverAdapter, citations_from_documents
from .providers import build_llm, make_messages
from .metrics import observe_chain, rag_query_rewritten_total
from .logging import get_logger

logger = get_logger(__name__)


def _should_rewrite(q: str) -> bool:
    return len(q.strip().split()) > 6


def _rewrite(q: str) -> str:
    return q.strip()


class LCChain:
    def __init__(self) -> None:
        s = load_settings()
        self.settings = s
        self.embedder = Embedder(s.embedding, seed=s.seed)
        store = IndexStore(s.paths.index_path, s.paths.index_meta_path)
        store.load()
        self.retriever = FAISSRetrieverAdapter(store, self.embedder, k=int(s.retrieval.get("k", 5)))
        env = Environment(loader=FileSystemLoader("src/rag_toolkit"))
        self.template = env.get_template(s.prompt.get("template_path", "prompts/qa.j2"))

    def invoke(self, payload: Dict) -> Dict:
        t0 = time.time()
        q = payload.get("query", "")
        k = int(payload.get("k", self.settings.retrieval.get("k", 5)))
        engine = "langchain"
        stream = bool(payload.get("stream", False))
        rewritten = q
        if _should_rewrite(q):
            rewritten = _rewrite(q)
            rag_query_rewritten_total.inc()
        docs = FAISSRetrieverAdapter(self.retriever.store, self.embedder, k=k).invoke(rewritten)
        contexts = [d.page_content for d in docs]
        rendered = self.template.render(system=self.settings.prompt.get("system", ""), question=rewritten, contexts=contexts)
        provider = self.settings.llm.provider or "null"
        llm = build_llm(provider, self.settings.llm.model, float(self.settings.llm.temperature), int(self.settings.llm.max_tokens), stream)
        ans = llm.invoke(make_messages(self.settings.prompt.get("system", ""), rendered))
        cits = citations_from_documents(docs)
        latency = time.time() - t0
        observe_chain(engine, "200", latency, None)
        return {"answer": ans, "citations": cits, "used_k": len(docs), "engine": engine}

    def stream(self, payload: Dict) -> Generator[str, None, Dict]:
        t0 = time.time()
        q = payload.get("query", "")
        k = int(payload.get("k", self.settings.retrieval.get("k", 5)))
        engine = "langchain"
        rewritten = q
        if _should_rewrite(q):
            rewritten = _rewrite(q)
            rag_query_rewritten_total.inc()
        docs = FAISSRetrieverAdapter(self.retriever.store, self.embedder, k=k).invoke(rewritten)
        contexts = [d.page_content for d in docs]
        rendered = self.template.render(system=self.settings.prompt.get("system", ""), question=rewritten, contexts=contexts)
        provider = self.settings.llm.provider or "null"
        llm = build_llm(provider, self.settings.llm.model, float(self.settings.llm.temperature), int(self.settings.llm.max_tokens), True)
        usage = yield from llm.stream(make_messages(self.settings.prompt.get("system", ""), rendered))
        cits = citations_from_documents(docs)
        latency = time.time() - t0
        observe_chain(engine, "200", latency, usage)
        return {"answer": "", "citations": cits, "used_k": len(docs), "engine": engine}


def build_chain() -> LCChain:
    return LCChain()
