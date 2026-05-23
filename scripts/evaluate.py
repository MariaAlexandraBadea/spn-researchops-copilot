from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from app.agent import route_message
from app.pretty import header, kv, mini_table, section, success, warn, write_json_preview
from app.retrieval import HybridRetriever


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run a small grounded-QA benchmark.")
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--benchmark", default="data/benchmark_questions.json")
    parser.add_argument("--metrics-root", default="data/source_repo")
    parser.add_argument("--out", default="outputs/evaluation_report.json")
    args = parser.parse_args()

    header("SPN ResearchOps Copilot - Benchmark")

    kv("Index", args.index)
    kv("Benchmark", args.benchmark)
    kv("Metrics root", args.metrics_root)

    retriever = HybridRetriever.load(args.index)
    questions = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))

    rows = []

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        expected_keywords = [k.lower() for k in item.get("expected_keywords", [])]

        result = route_message(question, retriever, metrics_root=args.metrics_root)

        answer = result.get("answer", "").lower()
        confidence = round(float(result.get("confidence", 0.0)), 3)
        tool = result.get("tool", "unknown")

        hits = sum(1 for keyword in expected_keywords if keyword in answer)
        total = len(expected_keywords)

        rows.append(
            {
                "#": i,
                "Tool": tool,
                "Confidence": confidence,
                "Keyword hits": f"{hits}/{total}",
                "Question": question,
            }
        )

    section("Benchmark results")
    mini_table(rows, ["#", "Tool", "Confidence", "Keyword hits", "Question"])

    avg_confidence = sum(float(row["Confidence"]) for row in rows) / max(1, len(rows))

    total_hits = sum(int(str(row["Keyword hits"]).split("/")[0]) for row in rows)
    total_expected = sum(int(str(row["Keyword hits"]).split("/")[1]) for row in rows)

    keyword_coverage = total_hits / max(1, total_expected)

    section("Evaluation summary")
    kv("Average confidence", round(avg_confidence, 3))
    kv("Keyword coverage", f"{total_hits}/{total_expected} ({round(keyword_coverage * 100, 1)}%)")

    if avg_confidence >= 0.45 and keyword_coverage >= 0.70:
        success("Status: PASS - retrieval quality is acceptable for a technical demo.")
    elif avg_confidence >= 0.30:
        warn("Status: PARTIAL - retrieval works, but more documentation or better chunking may improve answers.")
    else:
        warn("Status: LOW CONFIDENCE - add clearer source documents or check the index.")

    write_json_preview(args.out, rows)


if __name__ == "__main__":
    main()
