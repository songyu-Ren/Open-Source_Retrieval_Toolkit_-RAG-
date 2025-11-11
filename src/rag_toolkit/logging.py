import logging
import os
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    lvl = level or os.getenv("RAG_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format=(
            "{"
            "\"time\":\"%(asctime)s\","
            "\"level\":\"%(levelname)s\","
            "\"name\":\"%(name)s\","
            "\"message\":\"%(message)s\""
            "}"
        ),
    )


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)