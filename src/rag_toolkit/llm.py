from __future__ import annotations

import os
from typing import Dict, List

import httpx

from .logging import get_logger

logger = get_logger(__name__)


class NoLLM:
    def answer(self, query: str, contexts: List[str]) -> str:
        return "\n".join(contexts)


class OpenAICompatibleLLM:
    def __init__(self, api_key: str, api_base: str, model: str) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model

    def answer(self, query: str, contexts: List[str]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question: {query}\nContexts:\n" + "\n".join(contexts)},
            ],
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.api_base}/chat/completions"
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""


def get_llm_client(enabled: bool, api_base: str, model: str) -> object:
    api_key = os.getenv("OPENAI_API_KEY")
    if enabled and api_key:
        return OpenAICompatibleLLM(api_key=api_key, api_base=api_base, model=model)
    return NoLLM()