from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from pypdf import PdfReader

from app.chunking import chunk_text
from app.schema import DocumentChunk


SUPPORTED_SUFFIXES = {".pdf", ".py", ".tex", ".md", ".txt", ".csv", ".json", ".yml", ".yaml"}


def infer_source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "paper"
    if suffix == ".py":
        return "code"
    if suffix == ".tex":
        return "latex"
    if suffix == ".csv":
        return "table"
    if suffix in {".json", ".yml", ".yaml"}:
        return "structured"
    return "note"


def iter_supported_files(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in SUPPORTED_SUFFIXES:
        yield root
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            # Ignore common hidden / dependency dirs.
            parts = {p.lower() for p in path.parts}
            if ".git" in parts or "__pycache__" in parts or ".venv" in parts:
                continue
            yield path


def load_pdf(path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    reader = PdfReader(str(path))
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.extend(
            chunk_text(
                text=text,
                source_path=str(path),
                source_name=path.name,
                source_type="paper",
                page=page_idx,
            )
        )
    return chunks


def load_csv(path: Path) -> list[DocumentChunk]:
    try:
        df = pd.read_csv(path)
        preview = df.head(30).to_markdown(index=False)
        stats = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats.append(
                    {
                        "column": col,
                        "mean": float(df[col].mean()) if df[col].notna().any() else None,
                        "min": float(df[col].min()) if df[col].notna().any() else None,
                        "max": float(df[col].max()) if df[col].notna().any() else None,
                    }
                )
        text = (
            f"CSV file: {path.name}\n"
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
            f"Columns: {', '.join(map(str, df.columns))}\n\n"
            f"Preview:\n{preview}\n\n"
            f"Numeric summary JSON:\n{json.dumps(stats, indent=2)}"
        )
    except Exception as exc:
        text = f"CSV file: {path.name}\nCould not parse with pandas: {exc}"
    return chunk_text(str(text), str(path), path.name, "table")


def load_textlike(path: Path) -> list[DocumentChunk]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = path.read_text(errors="ignore")
    return chunk_text(text, str(path), path.name, infer_source_type(path))


def load_file(path: str | Path) -> list[DocumentChunk]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".csv":
        return load_csv(path)
    return load_textlike(path)


def load_folder(root: str | Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in iter_supported_files(root):
        try:
            chunks.extend(load_file(path))
        except Exception as exc:
            print(f"[WARN] Could not load {path}: {exc}")
    return chunks
