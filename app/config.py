from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    top_k: int = int(os.getenv("TOP_K", "6"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.25"))


def get_settings() -> Settings:
    return Settings()
