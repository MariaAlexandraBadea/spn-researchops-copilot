from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv

from app.config import get_settings
from app.loaders import load_folder
from app.retrieval import HybridRetriever


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input file/folder")
    parser.add_argument("--index", default="data/index", help="Index output folder")
    parser.add_argument("--reset", action="store_true", help="Reset index folder")
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    index_dir = Path(args.index)
    if args.reset and index_dir.exists():
        shutil.rmtree(index_dir)

    print(f"[INFO] Loading documents from {args.input}")
    chunks = load_folder(args.input)
    print(f"[INFO] Loaded {len(chunks)} chunks")
    if not chunks:
        raise SystemExit("No chunks found. Add supported files and try again.")

    settings = get_settings()
    model_name = args.embedding_model or settings.embedding_model
    retriever = HybridRetriever(embedding_model=model_name)
    retriever.build(chunks)
    retriever.save(index_dir)
    print(f"[OK] Index saved to {index_dir}")


if __name__ == "__main__":
    main()
