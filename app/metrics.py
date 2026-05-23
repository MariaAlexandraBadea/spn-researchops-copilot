from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def find_metric_files(root: str | Path) -> list[Path]:
    root = Path(root)
    files = []
    if root.is_file():
        return [root]
    for suffix in ("*.csv", "*.json"):
        files.extend(root.rglob(suffix))
    return files


def summarize_metrics(root: str | Path, max_rows: int = 10) -> str:
    files = find_metric_files(root)
    if not files:
        return "No CSV/JSON metric files were found."

    parts = []
    for path in files[:20]:
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                df = pd.json_normalize(data)
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            parts.append(f"File: {path.name} | shape={df.shape} | numeric_cols={numeric_cols}")
            if numeric_cols:
                parts.append(df[numeric_cols].describe().round(4).to_markdown())
            parts.append("Preview:")
            parts.append(df.head(max_rows).to_markdown(index=False))
        except Exception as exc:
            parts.append(f"Could not parse {path}: {exc}")
    return "\n\n".join(parts)


def query_best_metric(root: str | Path, metric: str, mode: str = "min") -> str:
    files = find_metric_files(root)
    rows = []
    for path in files:
        try:
            df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.json_normalize(json.loads(path.read_text()))
            candidates = [c for c in df.columns if c.lower() == metric.lower()]
            if not candidates:
                candidates = [c for c in df.columns if metric.lower() in c.lower()]
            if not candidates:
                continue
            col = candidates[0]
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            idx = df[col].idxmin() if mode == "min" else df[col].idxmax()
            row = df.loc[idx].to_dict()
            rows.append((path.name, col, float(df.loc[idx, col]), row))
        except Exception:
            continue

    if not rows:
        return f"No numeric metric matching '{metric}' was found."

    rows.sort(key=lambda x: x[2], reverse=(mode == "max"))
    best = rows[0]
    return (
        f"Best match for metric '{metric}' using mode='{mode}':\n"
        f"File: {best[0]}\nColumn: {best[1]}\nValue: {best[2]}\nRow:\n{json.dumps(best[3], indent=2, default=str)}"
    )
