"""Closed-set query/gallery retrieval metrics.

The metrics follow the image-level protocol of the PolarBearVidID paper: each
query is compared against the whole gallery, the query image itself is removed
from its own ranking (self-match), and every remaining image of the same
identity counts as a positive. Rank-1, Rank-3 and mAP are then averaged over
queries that have at least one positive.

Ties are broken by gallery order, which is fixed by the split, so results are
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class RetrievalError(ValueError):
    """Raised when the query/gallery task cannot be scored."""


@dataclass
class RetrievalResult:
    rank1: float
    rank3: float
    mean_average_precision: float
    n_queries: int
    n_scored_queries: int
    n_gallery: int
    n_identities: int
    per_query: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def n_queries_without_positives(self) -> int:
        return self.n_queries - self.n_scored_queries

    def to_dict(self) -> dict:
        return {
            "rank1": self.rank1,
            "rank3": self.rank3,
            "mAP": self.mean_average_precision,
            "n_queries": self.n_queries,
            "n_scored_queries": self.n_scored_queries,
            "n_queries_without_positives": self.n_queries_without_positives,
            "n_gallery": self.n_gallery,
            "n_identities": self.n_identities,
        }


def cosine_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Return the cosine similarity matrix of two embedding matrices."""
    query = np.asarray(query, dtype="float64")
    gallery = np.asarray(gallery, dtype="float64")
    if query.ndim != 2 or gallery.ndim != 2:
        raise RetrievalError("Features must be 2D matrices")
    if query.shape[1] != gallery.shape[1]:
        raise RetrievalError(
            f"Feature dimensions differ: query {query.shape[1]}, gallery {gallery.shape[1]}"
        )

    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    gallery_norm = np.linalg.norm(gallery, axis=1, keepdims=True)
    if np.any(query_norm == 0) or np.any(gallery_norm == 0):
        raise RetrievalError("Cannot compute cosine similarity for zero-norm embeddings")
    return (query / query_norm) @ (gallery / gallery_norm).T


def rank_retrieval(
    similarity: np.ndarray,
    query: pd.DataFrame,
    gallery: pd.DataFrame,
) -> RetrievalResult:
    """Score a closed-set retrieval task.

    Args:
        similarity: ``(n_query, n_gallery)`` cosine similarity matrix.
        query: Query metadata, must contain ``identity``, ``sequence``, ``path``.
        gallery: Gallery metadata with the same columns.
    """
    similarity = np.asarray(similarity, dtype="float64")
    if similarity.ndim != 2:
        raise RetrievalError("Similarity matrix must be 2D")
    if similarity.shape != (len(query), len(gallery)):
        raise RetrievalError(
            f"Similarity matrix has shape {similarity.shape}, expected "
            f"({len(query)}, {len(gallery)})"
        )
    if len(gallery) < 2:
        raise RetrievalError("Gallery needs at least two images for a leave-one-out task")

    query_paths = query["path"].to_numpy()
    gallery_paths = gallery["path"].to_numpy()
    query_labels = query["identity"].to_numpy()
    gallery_labels = gallery["identity"].to_numpy()

    # A query image is also in the gallery; it must not be its own positive.
    self_match = query_paths[:, None] == gallery_paths[None, :]
    order = np.argsort(-np.where(self_match, -np.inf, similarity), axis=1, kind="stable")
    eligible = ~self_match

    rank1_hits = np.zeros(len(query), dtype=bool)
    rank3_hits = np.zeros(len(query), dtype=bool)
    average_precisions = np.full(len(query), np.nan)
    first_positive = np.full(len(query), np.nan)
    scored = np.zeros(len(query), dtype=bool)

    for row in range(len(query)):
        positive = (gallery_labels == query_labels[row]) & eligible[row]
        n_positive = int(positive.sum())
        if n_positive == 0:
            continue

        ranked_positive = positive[order[row]]
        ranks = np.flatnonzero(ranked_positive) + 1
        positions = np.arange(1, len(ranks) + 1)
        average_precisions[row] = float(np.mean(positions / ranks))
        first_positive[row] = float(ranks[0])
        scored[row] = True

        ranked_labels = gallery_labels[order[row]]
        rank1_hits[row] = ranked_labels[0] == query_labels[row]
        rank3_hits[row] = bool(np.any(ranked_labels[:3] == query_labels[row]))

    if not scored.any():
        raise RetrievalError("No query has a positive gallery match")

    per_query = pd.DataFrame(
        {
            "sequence": query["sequence"].to_numpy(),
            "identity": query_labels,
            "first_positive_rank": first_positive,
            "average_precision": average_precisions,
            "scored": scored,
        }
    )

    return RetrievalResult(
        rank1=float(rank1_hits[scored].mean()),
        rank3=float(rank3_hits[scored].mean()),
        mean_average_precision=float(np.nanmean(average_precisions)),
        n_queries=int(len(query)),
        n_scored_queries=int(scored.sum()),
        n_gallery=int(len(gallery)),
        n_identities=int(len(set(gallery_labels))),
        per_query=per_query,
    )
