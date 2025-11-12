from __future__ import annotations

import os
from typing import Dict, Generator, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    ChatOllama = None


class NoLLM:
    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    def invoke(self, messages: List) -> str:
        ctx = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                ctx = m.content
        return ctx[: self.max_tokens]

    def stream(self, messages: List) -> Generator[str, None, Dict[str, int]]:
        full = self.invoke(messages)
        for ch in full:
            yield ch
        return {"prompt": len(full), "completion": len(full)}


class OpenAIClient:
    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: Optional[str], stream: bool) -> None:
        self.llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
        self.stream_enabled = stream

    def invoke(self, messages: List) -> str:
        res = self.llm.invoke(messages)
        return res.content or ""

    def stream(self, messages: List) -> Generator[str, None, Dict[str, int]]:
        usage = {"prompt": 0, "completion": 0}
        for chunk in self.llm.stream(messages):
            txt = chunk.content or ""
            usage["completion"] += len(txt)
            yield txt
        return usage


class AzureOpenAIClient(OpenAIClient):
    def __init__(self, model: str, temperature: float, max_tokens: int, stream: bool) -> None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key, stream=stream)


class OllamaClient:
    def __init__(self, model: str, temperature: float, max_tokens: int, stream: bool) -> None:
        if ChatOllama is None:
            self.llm = None
        else:
            self.llm = ChatOllama(model=model, temperature=temperature)
        self.stream_enabled = stream
        self.max_tokens = max_tokens

    def invoke(self, messages: List) -> str:
        if self.llm is None:
            return ""
        res = self.llm.invoke(messages)
        return res.content or ""

    def stream(self, messages: List) -> Generator[str, None, Dict[str, int]]:
        if self.llm is None:
            return (ch for ch in [])
        usage = {"prompt": 0, "completion": 0}
        for chunk in self.llm.stream(messages):
            txt = chunk.content or ""
            usage["completion"] += len(txt)
            yield txt
        return usage


def build_llm(provider: str, model: str, temperature: float, max_tokens: int, stream: bool):
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAIClient(model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key, stream=stream)
    if provider == "azure":
        return AzureOpenAIClient(model=model, temperature=temperature, max_tokens=max_tokens, stream=stream)
    if provider == "ollama":
        return OllamaClient(model=model, temperature=temperature, max_tokens=max_tokens, stream=stream)
    return NoLLM(max_tokens=max_tokens)


def make_messages(system: str, prompt: str):
    return [SystemMessage(content=system), HumanMessage(content=prompt)]
