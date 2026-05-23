from __future__ import annotations

import os
from typing import Iterable

from app.schema import RetrievedChunk


def build_context(results: list[RetrievedChunk], max_chars: int = 6500) -> str:
    parts = []
    used = 0
    for i, r in enumerate(results, start=1):
        c = r.chunk
        header = f"[{i}] Source: {r.citation()} | type={c.source_type} | score={r.score:.3f}\n"
        body = c.text.strip()
        block = header + body
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def extractive_answer(question: str, results: list[RetrievedChunk]) -> str:
    """Fallback when no LLM key is available."""
    if not results:
        return "I could not find enough indexed context to answer this."
    lines = []
    lines.append("I found the most relevant grounded context below. No LLM API key is configured, so this is an extractive answer.")
    for i, r in enumerate(results[:4], start=1):
        snippet = " ".join(r.chunk.text.split())[:700]
        lines.append(f"\n[{i}] {r.citation()} - {snippet}")
    return "\n".join(lines)


def openai_answer(question: str, results: list[RetrievedChunk], system_hint: str | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return extractive_answer(question, results)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        context = build_context(results)
        system = system_hint or (
            "You are an Applied AI research assistant. Answer only from the provided context. "
            "Use concise technical language. Cite sources using bracket numbers like [1], [2]. "
            "If the context is insufficient, say so."
        )
        user = f"Question:\n{question}\n\nContext:\n{context}"
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.15,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        return extractive_answer(question, results) + f"\n\n[LLM fallback reason: {exc}]"
