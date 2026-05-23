from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def header(title: str) -> None:
    line = "=" * min(max(len(title) + 8, 44), 88)
    print(_c(f"\n{line}\n  {title}\n{line}", "1;36"))


def section(title: str) -> None:
    print(_c(f"\n▶ {title}", "1;34"))


def info(message: str) -> None:
    print(_c(f"ℹ {message}", "36"))


def success(message: str) -> None:
    print(_c(f"✓ {message}", "32"))


def warn(message: str) -> None:
    print(_c(f"⚠ {message}", "33"))


def fail(message: str) -> None:
    print(_c(f"✗ {message}", "31"))


def kv(key: str, value: object) -> None:
    print(f"  {_c(str(key) + ':', '1')} {value}")


def mini_table(rows: list[dict], columns: list[str] | None = None, max_width: int = 52) -> None:
    if not rows:
        return

    columns = columns or list(rows[0].keys())

    def fmt(value: object) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\n", " ")
        return text[: max_width - 1] + "…" if len(text) > max_width else text

    widths = {
        col: min(max(len(col), max(len(fmt(row.get(col, ""))) for row in rows)), max_width)
        for col in columns
    }

    print("  " + "  ".join(_c(col.ljust(widths[col]), "1") for col in columns))
    print("  " + "  ".join("-" * widths[col] for col in columns))

    for row in rows:
        print("  " + "  ".join(fmt(row.get(col, "")).ljust(widths[col]) for col in columns))


def chunk_summary(chunks: Iterable) -> None:
    chunks = list(chunks)

    by_type = Counter(getattr(chunk, "source_type", "unknown") for chunk in chunks)
    by_source = Counter(getattr(chunk, "source_name", "unknown") for chunk in chunks)

    section("Index summary")
    kv("Total chunks", len(chunks))

    if by_type:
        mini_table(
            [{"Source type": key, "Chunks": value} for key, value in by_type.most_common()],
            ["Source type", "Chunks"],
        )

    if by_source:
        section("Top indexed files")
        mini_table(
            [{"File": key, "Chunks": value} for key, value in by_source.most_common(10)],
            ["File", "Chunks"],
        )


def write_json_preview(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    success(f"Saved JSON report: {path}")
