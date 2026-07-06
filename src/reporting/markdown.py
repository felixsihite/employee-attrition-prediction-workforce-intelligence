from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable

import pandas as pd


def clean_markdown(text: str) -> str:
    """Normalize generated Markdown so portfolio reports render consistently."""
    lines = textwrap.dedent(text).strip().splitlines()
    cleaned: list[str] = []
    for line in lines:
        if line.startswith("            "):
            line = line[12:]
        elif line.startswith("        "):
            line = line[8:]
        elif line.startswith("    "):
            line = line[4:]
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).strip() + "\n"


def markdown_table(frame: pd.DataFrame, *, float_digits: int = 4) -> str:
    """Render a compact GitHub-flavored Markdown table from a DataFrame."""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.{float_digits}f}")
        else:
            display[column] = display[column].astype(str)

    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, separator, *rows])


def write_markdown(path: Path, text: str) -> Path:
    """Write normalized Markdown and return the created artifact path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(clean_markdown(text), encoding="utf-8")
    return path


def artifact_inventory(root: Path, patterns: Iterable[str] = ("*.md", "*.csv", "*.png", "*.joblib")) -> pd.DataFrame:
    """Create a lightweight inventory of portfolio artifacts for audit reporting."""
    rows: list[dict[str, object]] = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                rows.append(
                    {
                        "artifact": str(path.relative_to(root)).replace("\\", "/"),
                        "type": path.suffix.lower().lstrip("."),
                        "size_kb": round(path.stat().st_size / 1024, 1),
                    }
                )
    return pd.DataFrame(rows).sort_values(["type", "artifact"]).reset_index(drop=True)