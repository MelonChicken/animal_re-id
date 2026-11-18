"""Retrieval metrics against hand-computed expectations.

The tests use tiny hand-built similarity matrices so that Rank-1, Rank-3 and
mAP are checked against numbers computed by hand, not against the
implementation itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from animal_re_id.evaluation.retrieval import (
    RetrievalError,
    cosine_similarity,
    rank_retrieval,
)


def make_frame(identities: list[str], paths: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "identity": identities,
            "path": paths,
            "sequence": [f"{identity}|{index}" for index, identity in enumerate(identities)],
        }
    )


def test_cosine_similarity_is_bounded_and_symmetric_for_identical_input() -> None:
    features = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    similarity = cosine_similarity(features, features)
    assert np.allclose(np.diag(similarity), 1.0)
    assert np.allclose(similarity, similarity.T)
    assert similarity.min() >= -1.0 - 1e-12 and similarity.max() <= 1.0 + 1e-12


def test_cosine_similarity_rejects_zero_norm() -> None:
    with pytest.raises(RetrievalError, match="zero-norm"):
        cosine_similarity(np.zeros((1, 3)), np.ones((1, 3)))


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(RetrievalError, match="Feature dimensions differ"):
        cosine_similarity(np.ones((1, 3)), np.ones((1, 4)))


def test_perfect_ranking_gives_rank1_rank3_and_map_of_one() -> None:
    query = make_frame(["A", "B"], ["a1", "b1"])
    gallery = make_frame(["A", "A", "B", "B"], ["a1", "a2", "b1", "b2"])
    similarity = np.array(
        [
            [1.0, 0.9, 0.1, 0.0],  # a1: a2 first
            [0.0, 0.1, 1.0, 0.9],  # b1: b2 first
        ]
    )

    result = rank_retrieval(similarity, query, gallery)
    assert result.rank1 == pytest.approx(1.0)
    assert result.rank3 == pytest.approx(1.0)
    assert result.mean_average_precision == pytest.approx(1.0)
    assert result.n_queries == 2 and result.n_scored_queries == 2
    assert result.n_gallery == 4 and result.n_identities == 2


def test_self_match_is_excluded_from_the_ranking() -> None:
    # a1 has a perfect self-similarity of 1.0; it must not count as its own hit.
    query = make_frame(["A"], ["a1"])
    gallery = make_frame(["A", "A", "B"], ["a1", "a2", "b1"])
    similarity = np.array([[1.0, 0.5, 0.6]])

    result = rank_retrieval(similarity, query, gallery)
    # Without self-match masking, a1 would rank itself first and rank1 would be 1.
    assert result.per_query["first_positive_rank"].iloc[0] == 2.0
    assert result.rank1 == 0.0
    assert result.mean_average_precision == pytest.approx(0.5)


def test_hand_computed_map_with_interleaved_negatives() -> None:
    # One query, four gallery images (self excluded): positives at rank 1 and 3.
    query = make_frame(["A"], ["a1"])
    gallery = make_frame(["A", "A", "B", "A"], ["a1", "a2", "b1", "a3"])
    # Ranking after masking self (a1): a3 (0.9), b1 (0.8), a2 (0.7)
    similarity = np.array([[1.0, 0.7, 0.8, 0.9]])

    result = rank_retrieval(similarity, query, gallery)
    # positives land at ranks 1 and 3 -> AP = mean(1/1, 2/3) = 0.8333...
    assert result.per_query["first_positive_rank"].iloc[0] == 1.0
    assert result.rank1 == 1.0
    assert result.mean_average_precision == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_rank3_differs_from_rank1() -> None:
    query = make_frame(["A"], ["a1"])
    gallery = make_frame(["A", "B", "C", "A"], ["a1", "b1", "c1", "a2"])
    # a2 sits at rank 3 after removing the self-match.
    similarity = np.array([[1.0, 0.9, 0.8, 0.7]])

    result = rank_retrieval(similarity, query, gallery)
    assert result.rank1 == 0.0
    assert result.rank3 == 1.0
    assert result.per_query["first_positive_rank"].iloc[0] == 3.0
    assert result.mean_average_precision == pytest.approx(1.0 / 3.0)


def test_query_without_positives_is_reported_but_not_scored() -> None:
    # The single "C" query has no other C image, so it cannot be scored.
    query = make_frame(["A", "C"], ["a1", "c1"])
    gallery = make_frame(["A", "A", "C"], ["a1", "a2", "c1"])
    similarity = np.array([[1.0, 0.9, 0.0], [0.0, 0.1, 1.0]])

    result = rank_retrieval(similarity, query, gallery)
    assert result.n_queries == 2
    assert result.n_scored_queries == 1
    assert result.n_queries_without_positives == 1
    assert result.rank1 == 1.0  # only the A query is averaged
    assert bool(result.per_query.loc[result.per_query["identity"] == "C", "scored"].iloc[0]) is False


def test_all_queries_without_positives_raises() -> None:
    query = make_frame(["A", "B"], ["a1", "b1"])
    gallery = make_frame(["A", "B"], ["a1", "b1"])
    similarity = np.array([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(RetrievalError, match="No query has a positive gallery match"):
        rank_retrieval(similarity, query, gallery)


def test_similarity_shape_is_validated() -> None:
    query = make_frame(["A"], ["a1"])
    gallery = make_frame(["A", "A"], ["a1", "a2"])
    with pytest.raises(RetrievalError, match="Similarity matrix has shape"):
        rank_retrieval(np.ones((1, 3)), query, gallery)


def test_gallery_needs_two_images() -> None:
    query = make_frame(["A"], ["a1"])
    gallery = make_frame(["A"], ["a1"])
    with pytest.raises(RetrievalError, match="at least two images"):
        rank_retrieval(np.ones((1, 1)), query, gallery)


def test_ties_are_broken_by_gallery_order() -> None:
    query = make_frame(["A"], ["a1"])
    gallery = make_frame(["A", "B", "A"], ["a1", "b1", "a2"])
    similarity = np.array([[1.0, 0.5, 0.5]])

    first = rank_retrieval(similarity, query, gallery)
    second = rank_retrieval(similarity.copy(), query, gallery)
    assert first.rank1 == second.rank1 == 0.0  # b1 comes before a2 in gallery order
    assert first.mean_average_precision == second.mean_average_precision == pytest.approx(0.5)
