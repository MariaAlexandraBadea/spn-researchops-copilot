from __future__ import annotations

from app.schema import DocumentChunk


def chunk_text(
    text: str,
    source_path: str,
    source_name: str,
    source_type: str,
    page: int | None = None,
    max_chars: int = 1400,
    overlap: int = 220,
) -> list[DocumentChunk]:
    """Simple robust chunking for technical documents and code."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            # Prefer paragraph or line boundary.
            para = text.rfind("\n\n", start, end)
            line = text.rfind("\n", start, end)
            boundary = para if para > start + max_chars * 0.55 else line
            if boundary > start + 200:
                end = boundary

        chunk_text_value = text[start:end].strip()
        if chunk_text_value:
            line_start = text[:start].count("\n") + 1 if page is None else None
            line_end = text[:end].count("\n") + 1 if page is None else None
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{source_name}::{page or 0}::{idx}",
                    text=chunk_text_value,
                    source_path=source_path,
                    source_name=source_name,
                    source_type=source_type,
                    page=page,
                    line_start=line_start,
                    line_end=line_end,
                    metadata={"chunk_index": idx},
                )
            )
            idx += 1

        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks
