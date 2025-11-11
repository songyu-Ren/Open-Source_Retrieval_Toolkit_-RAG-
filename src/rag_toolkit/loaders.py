from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

from pdfminer.high_level import extract_text

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    doc_id: str
    path: str
    text: str
    meta: Dict[str, str]


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_pdf(path: str) -> str:
    return extract_text(path) or ""


def load_documents(root: str) -> List[Document]:
    exts = {".txt", ".md", ".pdf"}
    docs: List[Document] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue
            full = os.path.join(dirpath, name)
            try:
                if ext == ".txt":
                    text = _load_txt(full)
                elif ext == ".md":
                    text = _load_md(full)
                else:
                    text = _load_pdf(full)
                doc_id = name  # use filename as id
                meta = {"relpath": os.path.relpath(full, root), "ext": ext}
                docs.append(Document(doc_id=doc_id, path=full, text=text, meta=meta))
            except Exception as e:
                logger.error(f"Failed to load {full}: {e}")
    logger.info(f"Loaded {len(docs)} documents from {root}")
    return docs