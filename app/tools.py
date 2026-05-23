from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from app.llm import openai_answer
from app.metrics import query_best_metric, summarize_metrics
from app.retrieval import HybridRetriever


def ensure_out_dir(out_dir: str | Path = "outputs") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def answer_question(retriever: HybridRetriever, question: str, top_k: int = 6, source_type: str | None = None) -> dict:
    results = retriever.search(question, top_k=top_k, source_type=source_type)
    confidence = float(sum(r.score for r in results[:3]) / max(1, len(results[:3]))) if results else 0.0
    answer = openai_answer(question, results)
    return {
        "answer": answer,
        "confidence": round(confidence, 3),
        "sources": [
            {
                "citation": r.citation(),
                "score": round(r.score, 3),
                "source_type": r.chunk.source_type,
                "text_preview": " ".join(r.chunk.text.split())[:240],
            }
            for r in results
        ],
    }


def generate_model_card(retriever: HybridRetriever) -> dict:
    prompt = (
        "Generate a concise model card for the CNN-GRU-SPN-Gamma temperature anomaly forecasting model. "
        "Include model purpose, data, architecture, leakage controls, metrics, limitations and intended use."
    )
    return answer_question(retriever, prompt, top_k=8)


def draft_reviewer_response(retriever: HybridRetriever, concern: str) -> dict:
    prompt = (
        "Draft a technical reviewer response for this concern, grounded in the project sources. "
        "Use a respectful academic tone and mention exact model/evaluation details where supported.\n\n"
        f"Concern: {concern}"
    )
    return answer_question(retriever, prompt, top_k=8)


def create_github_issue(title: str, body: str, labels: str = "enhancement", out_dir: str | Path = "outputs") -> str:
    out = ensure_out_dir(out_dir)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in title.lower()).strip("-")[:80]
    path = out / f"github_issue_{safe or 'issue'}.md"
    text = f"# {title}\n\nLabels: {labels}\n\n## Description\n{body}\n\n## Acceptance criteria\n- [ ] Implementation is documented.\n- [ ] Basic test or demo is included.\n- [ ] README is updated if needed.\n"
    path.write_text(text, encoding="utf-8")
    return str(path)


def create_research_plan(topic: str, out_dir: str | Path = "outputs") -> str:
    out = ensure_out_dir(out_dir)
    date = dt.date.today().isoformat()
    path = out / f"research_plan_{date}.md"
    text = f"""# Research Plan - {topic}

## Objective
Clarify the technical objective and expected applied AI outcome.

## Milestones
1. Data/source ingestion and baseline retrieval.
2. Hybrid retrieval and reranking.
3. Grounded generation with citation checks.
4. Evaluation benchmark with expected answers.
5. UI/API demo and README documentation.

## Risks
- Weak grounding if chunks are too large or too broad.
- Hallucination if the model answers outside retrieved context.
- Poor retrieval for code-specific questions without metadata filtering.

## Demo questions
- Explain the model architecture and cite source files.
- Summarize leakage controls.
- Generate a model card.
- Compare evaluation metrics.
"""
    path.write_text(text, encoding="utf-8")
    return str(path)


def summarize_experiments(metrics_root: str | Path) -> str:
    return summarize_metrics(metrics_root)


def best_metric(metrics_root: str | Path, metric: str, mode: str = "min") -> str:
    return query_best_metric(metrics_root, metric, mode=mode)
