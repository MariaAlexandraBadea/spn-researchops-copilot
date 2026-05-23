from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_path: str
    source_name: str
    source_type: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metadata"] = self.metadata or {}
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DocumentChunk":
        return DocumentChunk(**data)


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None

    def citation(self) -> str:
        c = self.chunk
        loc = ""
        if c.page is not None:
            loc = f", page {c.page}"
        elif c.line_start is not None:
            loc = f", lines {c.line_start}-{c.line_end or c.line_start}"
        return f"{c.source_name}{loc}"
