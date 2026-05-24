from __future__ import annotations

import os
import re
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


def _clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _content_results(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Keep demo files from being used as main evidence.
    demo_questions.md is useful for testing, but not ideal as an answer source.
    """
    filtered = [
        r for r in results
        if "demo_questions" not in r.chunk.source_name.lower()
    ]
    return filtered or results


def _split_evidence(text: str) -> list[str]:
    """
    Extract readable evidence units from Markdown/code comments.
    Keeps bullets and short explanatory sentences.
    """
    text = _clean_text(text)
    items: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(("#", "```", "|")):
            continue

        if line.startswith(("-", "*", "•")):
            item = re.sub(r"^[-*•]\s*", "", line).strip()
            if len(item) > 8:
                items.append(item)
            continue

        # Split longer prose into sentences.
        for sent in re.split(r"(?<=[.!?])\s+", line):
            sent = sent.strip()
            if 25 <= len(sent) <= 260:
                items.append(sent)

    return items


def _dedupe(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []

    for item in items:
        item = _clean_text(item)
        key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()

        if not key or key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


def _keyword_evidence(results: list[RetrievedChunk], keywords: list[str], max_items: int = 8) -> list[str]:
    keywords_l = [k.lower() for k in keywords]
    candidates: list[str] = []

    for r in _content_results(results):
        for item in _split_evidence(r.chunk.text):
            low = item.lower()

            if any(k in low for k in keywords_l):
                candidates.append(item)

    return _dedupe(candidates)[:max_items]


def _sources(results: list[RetrievedChunk], max_sources: int = 4) -> str:
    clean = _content_results(results)
    lines = []

    for i, r in enumerate(clean[:max_sources], start=1):
        lines.append(f"[{i}] {r.citation()}")

    return "\n".join(lines)


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item.rstrip('.')}" for item in items)


def _generic_context_answer(question: str, results: list[RetrievedChunk]) -> str:
    clean = _content_results(results)

    if not clean:
        return "I could not find enough indexed context to answer this question."

    snippets = []

    for r in clean[:4]:
        text = " ".join(r.chunk.text.split())
        snippets.append(text[:420].strip())

    answer = (
        "I found relevant grounded context, but no LLM generation is available. "
        "Here is a concise extractive summary based on the retrieved sources:\n\n"
    )

    answer += _format_bullets(_dedupe(snippets)[:4])
    answer += "\n\nSources:\n" + _sources(results)

    return answer


def extractive_answer(question: str, results: list[RetrievedChunk]) -> str:
    """
    Polished fallback when no LLM key is available or the LLM call fails.
    It creates a structured, source-grounded answer instead of dumping raw chunks.
    """
    if not results:
        return "I could not find enough indexed context to answer this question."

    q = question.lower()

    if any(term in q for term in ["leakage", "preprocessing", "preprocess", "imputation", "climatology"]):
        evidence = _keyword_evidence(
            results,
            [
                "monthly aggregation",
                "climatology",
                "training years",
                "input gaps",
                "pchip",
                "target",
                "never imputed",
                "discarded",
                "disjoint",
                "temporal split",
                "scaling",
                "training windows",
            ],
            max_items=9,
        )

        if not evidence:
            return _generic_context_answer(question, results)

        return (
            "The preprocessing pipeline implements the following leakage controls:\n\n"
            f"{_format_bullets(evidence)}\n\n"
            "In short, the project avoids using future information by computing climatology only from training years, "
            "imputing only inside the historical input window, never imputing targets, and keeping train/validation/test "
            "cities disjoint.\n\n"
            "Sources:\n"
            f"{_sources(results)}"
        )

    if any(term in q for term in ["metric", "rmse", "mae", "crps", "nll", "coverage", "pit", "energy score", "evaluation"]):
        evidence = _keyword_evidence(
            results,
            [
                "rmse",
                "mae",
                "nll",
                "negative log",
                "crps",
                "coverage",
                "pit",
                "calibration",
                "energy score",
                "bootstrap",
                "diebold",
                "newey",
                "held-out",
            ],
            max_items=10,
        )

        if not evidence:
            return _generic_context_answer(question, results)

        return (
            "The model is evaluated with point, probabilistic and multivariate forecasting metrics:\n\n"
            f"{_format_bullets(evidence)}\n\n"
            "This means the evaluation does not focus only on point accuracy. It also checks uncertainty quality, "
            "calibration and the joint multi-horizon predictive distribution.\n\n"
            "Sources:\n"
            f"{_sources(results)}"
        )

    if any(term in q for term in ["architecture", "cnn", "gru", "spn", "gamma", "decoder", "encoder"]):
        evidence = _keyword_evidence(
            results,
            [
                "cnn",
                "cnn1d",
                "skip",
                "gru",
                "encoder",
                "decoder",
                "spn",
                "gamma",
                "root mixture",
                "latent",
                "bernoulli",
                "sign",
                "magnitude",
                "product",
                "horizon",
                "predictive distribution",
            ],
            max_items=9,
        )

        if not evidence:
            return _generic_context_answer(question, results)

        return (
            "The CNN-GRU-SPN-Gamma model is structured as a probabilistic multi-horizon forecasting pipeline:\n\n"
            f"{_format_bullets(evidence)}\n\n"
            "In short, the CNN1D and GRU form the temporal encoder, while the conditional SPN-Gamma head models "
            "a full predictive distribution over future anomaly horizons rather than producing only deterministic values.\n\n"
            "Sources:\n"
            f"{_sources(results)}"
        )

    if any(term in q for term in ["mdn", "mixture density", "standard mixture", "more than"]):
        evidence = _keyword_evidence(
            results,
            [
                "mixture density",
                "mdn",
                "probabilistic circuit",
                "root mixture",
                "product",
                "shared latent",
                "cross-horizon",
                "exact",
                "posterior",
                "masked",
                "marginal",
                "sign-magnitude",
                "joint",
            ],
            max_items=9,
        )

        if not evidence:
            return _generic_context_answer(question, results)

        return (
            "The model has similarities with a Mixture Density Network, but the indexed documentation describes it as a "
            "structured probabilistic circuit:\n\n"
            f"{_format_bullets(evidence)}\n\n"
            "The key distinction is that the shared latent component and product structure over horizons allow tractable "
            "multi-horizon probabilistic inference, posterior responsibilities, masked marginalization and joint sampling.\n\n"
            "Sources:\n"
            f"{_sources(results)}"
        )

    if "model card" in q:
        evidence = _keyword_evidence(
            results,
            [
                "project purpose",
                "dataset",
                "architecture",
                "leakage",
                "evaluation",
                "metrics",
                "limitations",
                "intended",
                "uncertainty",
            ],
            max_items=10,
        )

        if not evidence:
            return _generic_context_answer(question, results)

        return (
            "Model card summary:\n\n"
            "Model: CNN-GRU-SPN-Gamma probabilistic temperature anomaly forecaster.\n\n"
            "Main grounded details:\n"
            f"{_format_bullets(evidence)}\n\n"
            "Sources:\n"
            f"{_sources(results)}"
        )

    return _generic_context_answer(question, results)


def openai_answer(question: str, results: list[RetrievedChunk], system_hint: str | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return extractive_answer(question, results)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        context = build_context(_content_results(results))

        system = system_hint or (
            "You are an Applied AI research assistant. Answer only from the provided context. "
            "Use concise technical language. Cite sources using bracket numbers like [1], [2]. "
            "If the context is insufficient, say so."
        )

        user = f"Question:\n{question}\n\nContext:\n{context}"

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.15,
        )

        return resp.choices[0].message.content or ""

    except Exception as exc:
        return (
            extractive_answer(question, results)
            + f"\n\nLLM generation was unavailable, so the answer above used the improved extractive fallback.\n"
            + f"Fallback reason: {exc}"
        )
