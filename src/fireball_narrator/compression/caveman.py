from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)?|[.,;:!?]")

# Negations, temporal markers, quantities and spatial relations are deliberately
# absent because they often carry facts needed for narrative continuity.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "our",
    "ours",
    "she",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
    "yours",
}


def compress_text(text: str) -> str:
    tokens = TOKEN_PATTERN.findall(re.sub(r"\s+", " ", text).strip())
    kept: list[str] = []
    previous = ""
    for token in tokens:
        lower = token.casefold()
        if token in ".,;:!?":
            if kept and kept[-1] not in ".,;:!?":
                kept.append(token)
            continue
        if lower in STOPWORDS:
            continue
        if lower == previous:
            continue
        kept.append(token)
        previous = lower

    compressed = " ".join(kept)
    compressed = re.sub(r"\s+([.,;:!?])", r"\1", compressed)
    compressed = re.sub(r"([.,;:!?]){2,}", r"\1", compressed)
    return compressed.strip(" ,;:")


def compress_history(turns: list[str]) -> list[str]:
    return [compressed for turn in turns if (compressed := compress_text(turn))]


def token_compression_ratio(
    original: str,
    compressed: str,
    token_counter: Callable[[str], int] | None = None,
) -> float:
    counter = token_counter or (lambda value: len(value.split()))
    original_count = counter(original)
    compressed_count = counter(compressed)
    if original_count == 0:
        return 0.0
    return 1.0 - (compressed_count / original_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress JSONL history fields with deterministic syntax minimization."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-field", default="history")
    parser.add_argument(
        "--tokenizer",
        help="Optional Hugging Face tokenizer for exact token compression ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token_counter = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        token_counter = lambda value: len(tokenizer.encode(value, add_special_tokens=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.input.open("r", encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as target,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            history = row.get(args.history_field)
            if not isinstance(history, list):
                raise ValueError(
                    f"{args.input}:{line_number} field {args.history_field!r} "
                    "must be a list of strings"
                )
            compressed_history = compress_history([str(turn) for turn in history])
            original_text = "\n".join(map(str, history))
            compressed_text = "\n".join(compressed_history)
            row["compressed_history"] = compressed_history
            row["token_compression_ratio"] = token_compression_ratio(
                original_text,
                compressed_text,
                token_counter,
            )
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

