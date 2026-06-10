from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(sum(items) / len(items)) if items else 0.0


def thematic_adherence_rate(adherent: Iterable[bool]) -> float:
    return _mean(1.0 if value else 0.0 for value in adherent)


def token_compression_ratio(original_tokens: int, compressed_tokens: int) -> float:
    if original_tokens <= 0:
        return 0.0
    return 1.0 - (compressed_tokens / original_tokens)


def _normalize_fact(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def semantic_preservation_score(
    original_facts: Iterable[str],
    preserved_facts: Iterable[str],
) -> float:
    original = {_normalize_fact(fact) for fact in original_facts if fact.strip()}
    preserved = {_normalize_fact(fact) for fact in preserved_facts if fact.strip()}
    if not original:
        return 0.0
    return len(original & preserved) / len(original)


def information_retention_accuracy(correct: Iterable[bool]) -> float:
    return _mean(1.0 if value else 0.0 for value in correct)


def sycophancy_mitigation_index(
    disruptive_prompts: list[str],
    responses: list[str],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    if len(disruptive_prompts) != len(responses):
        raise ValueError("Prompt and response counts must match")
    if not disruptive_prompts:
        return 0.0

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model)
    prompt_embeddings = model.encode(
        disruptive_prompts,
        normalize_embeddings=True,
    )
    response_embeddings = model.encode(
        responses,
        normalize_embeddings=True,
    )
    cosine = np.sum(prompt_embeddings * response_embeddings, axis=1)
    normalized_distance = (1.0 - cosine) / 2.0
    return float(np.mean(normalized_distance))

