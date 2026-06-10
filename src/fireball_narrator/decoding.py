from __future__ import annotations

from pathlib import Path
from typing import Any


def read_forbidden_terms(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def build_bad_words_ids(tokenizer: Any, terms: list[str]) -> list[list[int]]:
    sequences: set[tuple[int, ...]] = set()
    for term in terms:
        for variant in (term, f" {term}", term.capitalize(), f" {term.capitalize()}"):
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if token_ids:
                sequences.add(tuple(token_ids))
    return [list(sequence) for sequence in sorted(sequences)]

