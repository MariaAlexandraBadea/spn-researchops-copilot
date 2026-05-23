from __future__ import annotations

import re
from pathlib import Path

from app.retrieval import HybridRetriever
from app.tools import (
    answer_question,
    best_metric,
    create_github_issue,
    create_research_plan,
    draft_reviewer_response,
    generate_model_card,
    summarize_experiments,
)


def route_message(
    message: str,
    retriever: HybridRetriever,
    metrics_root: str | Path | None = None,
    out_dir: str | Path = "outputs",
) -> dict:
    """Small deterministic agent router for Colab demos."""
    msg = message.strip()
    low = msg.lower()

    if "model card" in low:
        return {"tool": "generate_model_card", **generate_model_card(retriever)}

    if "reviewer" in low or "response to reviewer" in low:
        return {"tool": "draft_reviewer_response", **draft_reviewer_response(retriever, msg)}

    if "github issue" in low or "create issue" in low:
        title = "ResearchOps Copilot task"
        body = msg
        path = create_github_issue(title=title, body=body, out_dir=out_dir)
        return {"tool": "create_github_issue", "answer": f"Created GitHub issue draft: {path}", "confidence": 1.0, "sources": []}

    if "research plan" in low or "plan" in low and "research" in low:
        path = create_research_plan(topic=msg, out_dir=out_dir)
        return {"tool": "create_research_plan", "answer": f"Created research plan: {path}", "confidence": 1.0, "sources": []}

    if ("best" in low or "lowest" in low or "highest" in low) and metrics_root:
        metric_match = re.search(r"\b(rmse|mae|crps|nll|coverage|energy|energy_score|f1|mcc|accuracy)\b", low)
        if metric_match:
            metric = metric_match.group(1)
            mode = "max" if metric in {"coverage", "f1", "mcc", "accuracy"} or "highest" in low else "min"
            return {"tool": "best_metric", "answer": best_metric(metrics_root, metric, mode), "confidence": 0.9, "sources": []}

    if ("summarize" in low or "summary" in low) and ("metric" in low or "experiment" in low) and metrics_root:
        return {"tool": "summarize_experiments", "answer": summarize_experiments(metrics_root), "confidence": 0.9, "sources": []}

    source_type = None
    if "code" in low or "implementation" in low or "function" in low:
        source_type = "code"
    elif "latex" in low or "manuscript" in low:
        source_type = "latex"
    elif "paper" in low:
        source_type = "paper"
    elif "csv" in low or "table" in low or "metric" in low:
        source_type = "table"

    return {"tool": "rag_answer", **answer_question(retriever, msg, top_k=7, source_type=source_type)}
