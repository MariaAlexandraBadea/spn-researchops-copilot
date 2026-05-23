from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from app.agent import route_message
from app.retrieval import HybridRetriever


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--benchmark", default="data/benchmark_questions.json")
    parser.add_argument("--metrics-root", default="data/source_repo")
    args = parser.parse_args()

    retriever = HybridRetriever.load(args.index)
    questions = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))

    rows = []
    for item in questions:
        q = item["question"]
        expected_keywords = [k.lower() for k in item.get("expected_keywords", [])]
        result = route_message(q, retriever, metrics_root=args.metrics_root)
        answer = result.get("answer", "").lower()
        hit = sum(1 for k in expected_keywords if k in answer)
        rows.append(
            {
                "question": q,
                "tool": result.get("tool"),
                "confidence": result.get("confidence"),
                "keyword_hits": hit,
                "expected_keywords": expected_keywords,
            }
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
