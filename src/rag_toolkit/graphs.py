from __future__ import annotations

import time
from typing import Dict, Generator, List

from jinja2 import Environment, FileSystemLoader
from langgraph.graph import StateGraph

from .config import load_settings
from .embedder import Embedder
from .index_store import IndexStore
from .lc_adapters import FAISSRetrieverAdapter, citations_from_documents
from .providers import build_llm, make_messages
from .metrics import observe_chain, rag_query_rewritten_total
from .logging import get_logger

logger = get_logger(__name__)


class LGGraph:
    def __init__(self) -> None:
        s = load_settings()
        self.settings = s
        self.embedder = Embedder(s.embedding, seed=s.seed)
        store = IndexStore(s.paths.index_path, s.paths.index_meta_path)
        store.load()
        self.store = store
        env = Environment(loader=FileSystemLoader("src/rag_toolkit"))
        self.template = env.get_template(s.prompt.get("template_path", "prompts/qa.j2"))

    def _rewrite(self, state: Dict) -> Dict:
        q = state.get("query", "")
        if len(q.strip().split()) > 6:
            rag_query_rewritten_total.inc()
            state["query"] = q.strip()
        return state

    def _retrieve(self, state: Dict) -> Dict:
        k = int(state.get("k", self.settings.retrieval.get("k", 5)))
        retr = FAISSRetrieverAdapter(self.store, self.embedder, k=k)
        docs = retr.invoke(state.get("query", ""))
        state["docs"] = docs
        return state

    def _generate(self, state: Dict) -> Dict:
        contexts = [d.page_content for d in state.get("docs", [])]
        rendered = self.template.render(system=self.settings.prompt.get("system", ""), question=state.get("query", ""), contexts=contexts)
        provider = self.settings.llm.provider or "null"
        llm = build_llm(provider, self.settings.llm.model, float(self.settings.llm.temperature), int(self.settings.llm.max_tokens), bool(state.get("stream", False)))
        state["llm"] = llm
        state["rendered"] = rendered
        return state

    def _postprocess(self, state: Dict) -> Dict:
        docs = state.get("docs", [])
        cits = citations_from_documents(docs)
        state["citations"] = cits
        state["used_k"] = len(docs)
        return state

    def invoke(self, payload: Dict) -> Dict:
        t0 = time.time()
        engine = "langgraph"
        g = StateGraph(dict)
        g.add_node("rewrite", self._rewrite)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate)
        g.add_node("postprocess", self._postprocess)
        g.add_edge("rewrite", "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "postprocess")
        g.set_entry_point("rewrite")
        graph = g.compile()
        state = graph.invoke({"query": payload.get("query", ""), "k": payload.get("k", self.settings.retrieval.get("k", 5)), "stream": False})
        llm = state.get("llm")
        answer = llm.invoke(make_messages(self.settings.prompt.get("system", ""), state.get("rendered", "")))
        latency = time.time() - t0
        observe_chain(engine, "200", latency, None)
        return {"answer": answer, "citations": state.get("citations", []), "used_k": state.get("used_k", 0), "engine": engine}

    def stream(self, payload: Dict) -> Generator[str, None, Dict]:
        t0 = time.time()
        engine = "langgraph"
        g = StateGraph(dict)
        g.add_node("rewrite", self._rewrite)
        g.add_node("retrieve", self._retrieve)
        g.add_node("generate", self._generate)
        g.add_node("postprocess", self._postprocess)
        g.add_edge("rewrite", "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "postprocess")
        g.set_entry_point("rewrite")
        graph = g.compile()
        state = graph.invoke({"query": payload.get("query", ""), "k": payload.get("k", self.settings.retrieval.get("k", 5)), "stream": True})
        llm = state.get("llm")
        usage = yield from llm.stream(make_messages(self.settings.prompt.get("system", ""), state.get("rendered", "")))
        latency = time.time() - t0
        observe_chain(engine, "200", latency, usage)
        return {"answer": "", "citations": state.get("citations", []), "used_k": state.get("used_k", 0), "engine": engine}


def build_graph() -> LGGraph:
    return LGGraph()
