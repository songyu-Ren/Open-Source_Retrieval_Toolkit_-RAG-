from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            a[k] = _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


@dataclass
class Paths:
    raw_data_dir: str
    artifacts_dir: str
    chunks_path: str
    embeddings_path: str
    index_path: str
    index_meta_path: str
    eval_path: str


@dataclass
class EmbeddingCfg:
    model_name: str
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True
    use_dummy: bool = False


@dataclass
class ChunkingCfg:
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class IndexCfg:
    type: str = "IndexFlatIP"
    metric: str = "ip"


@dataclass
class LLMcfg:
    enabled: bool = False
    max_context_tokens: int = 2000
    prompt_template: str = (
        "You are a helpful assistant. Answer the question using only the provided contexts.\n"
        "Question: {query}\nContexts:\n{contexts}"
    )
    model: str = "gpt-3.5-turbo"
    api_base: str = "https://api.openai.com/v1"


@dataclass
class EvalCfg:
    k_default: int = 10


@dataclass
class ServerCfg:
    port: int = 8002


@dataclass
class MLflowCfg:
    tracking_uri: str = "./mlruns"


@dataclass
class Settings:
    seed: int
    paths: Paths
    embedding: EmbeddingCfg
    chunking: ChunkingCfg
    index: IndexCfg
    retrieval: Dict[str, Any] = field(default_factory=dict)
    llm: LLMcfg = field(default_factory=LLMcfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    mlflow: MLflowCfg = field(default_factory=MLflowCfg)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings() -> Settings:
    base = load_yaml("config/settings.yaml")

    override = os.getenv("RAG_SETTINGS")
    if override:
        # If points to a file, load and merge; otherwise attempt to parse YAML string
        if os.path.exists(override):
            ov = load_yaml(override)
            base = _deep_merge(base, ov)
        else:
            try:
                ov = yaml.safe_load(override) or {}
                base = _deep_merge(base, ov)
            except Exception:
                pass

    # Convert dict to dataclasses
    paths = Paths(**base["paths"])
    emb = EmbeddingCfg(**base["embedding"])
    chk = ChunkingCfg(**base["chunking"])
    idx = IndexCfg(**base["index"])
    llm = LLMcfg(**base.get("llm", {}))
    ev = EvalCfg(**base.get("eval", {}))
    srv = ServerCfg(**base.get("server", {}))
    mf = MLflowCfg(**base.get("mlflow", {}))
    return Settings(
        seed=base.get("seed", 42),
        paths=paths,
        embedding=emb,
        chunking=chk,
        index=idx,
        retrieval=base.get("retrieval", {}),
        llm=llm,
        eval=ev,
        server=srv,
        mlflow=mf,
    )