"""Retrieval metrics and evaluation helpers."""

from animal_re_id.evaluation.retrieval import (
    RetrievalError,
    RetrievalResult,
    cosine_similarity,
    rank_retrieval,
)

__all__ = ["RetrievalError", "RetrievalResult", "cosine_similarity", "rank_retrieval"]
