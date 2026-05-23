from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

import gradio as gr

from app.agent import route_message
from app.retrieval import HybridRetriever


def format_response(result: dict) -> tuple[str, str]:
    answer = result.get("answer", "")
    meta = {
        "tool": result.get("tool"),
        "confidence": result.get("confidence"),
        "sources": result.get("sources", []),
    }
    return answer, json.dumps(meta, indent=2, ensure_ascii=False)


def build_ui(index_dir: str, metrics_root: str):
    retriever = HybridRetriever.load(index_dir)

    def ask(message: str):
        if not message.strip():
            return "Please enter a question.", "{}"
        result = route_message(message, retriever, metrics_root=metrics_root)
        return format_response(result)

    examples = [
        "Explain the CNN-GRU-SPN-Gamma architecture and cite the implementation sources.",
        "What leakage controls are implemented in the preprocessing pipeline?",
        "How does the conditional SPN-Gamma decoder differ from a standard MDN?",
        "What metrics are used for probabilistic forecast evaluation?",
        "Generate a model card for this forecasting model.",
        "Draft a reviewer response about uncertainty quantification.",
        "Create a GitHub issue for adding calibration plots.",
    ]

    with gr.Blocks(title="SPN ResearchOps Copilot") as demo:
        gr.Markdown("# SPN ResearchOps Copilot")
        gr.Markdown(
            "Agentic RAG assistant for probabilistic ML research. "
            "Ask questions over code, papers, LaTeX and experiment outputs."
        )
        inp = gr.Textbox(label="Question or task", lines=3, placeholder="Ask about the SPN model, code, metrics, reviewer response...")
        btn = gr.Button("Run")
        out = gr.Textbox(label="Answer", lines=18)
        meta = gr.Code(label="Tool / confidence / sources", language="json")
        gr.Examples(examples=examples, inputs=inp)
        btn.click(ask, inputs=inp, outputs=[out, meta])
        inp.submit(ask, inputs=inp, outputs=[out, meta])
    return demo


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--metrics-root", default="data/source_repo")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui(args.index, args.metrics_root)
    demo.launch(share=args.share, debug=True)


if __name__ == "__main__":
    main()
