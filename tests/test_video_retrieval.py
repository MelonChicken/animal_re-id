"""Video retrieval: official pair protocol and clip-id mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from animal_re_id.evaluation.video_retrieval import (
    VideoRetrievalError,
    clip_id_from_image_id,
    euclidean_distance,
    load_pair_task,
    rank_pair_task,
    resolve_pair_task,
    sequence_clip_ids,
    validate_pair_task,
)


# --------------------------------------------------------------------------- #
# Distance
# --------------------------------------------------------------------------- #
def test_euclidean_distance_matches_hand_computation():
    query = np.array([[0.0, 0.0], [3.0, 4.0]])
    gallery = np.array([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]])
    distances = euclidean_distance(query, gallery)
    assert distances.shape == (2, 3)
    assert np.allclose(distances[0], [5.0, 0.0, 10.0])
    assert np.allclose(distances[1], [0.0, 5.0, 5.0])


def test_euclidean_distance_rejects_bad_input():
    with pytest.raises(VideoRetrievalError):
        euclidean_distance(np.zeros((2, 3)), np.zeros(3))
    with pytest.raises(VideoRetrievalError):
        euclidean_distance(np.zeros((1, 3)), np.zeros((1, 4)))
    with pytest.raises(VideoRetrievalError):
        euclidean_distance(np.array([[np.nan, 0.0]]), np.zeros((1, 2)))


# --------------------------------------------------------------------------- #
# Pair file
# --------------------------------------------------------------------------- #
def write_pair_file(path, rows):
    width = max(len(row) for row in rows)
    columns = ["Query", "q0"] + [f"g{index}" for index in range(1, width - 1)]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_load_pair_task_keeps_every_column_as_a_candidate(tmp_path):
    path = write_pair_file(
        tmp_path / "pairs.csv", [["000_001", "000_002", "001_003", "002_004"]]
    )
    task = load_pair_task(path)
    assert task.shape[0] == 1
    assert task.loc[0, "candidates"] == ("000_001", "000_002", "001_003", "002_004")
    assert task.loc[0, "positive_a"] == "000_001"
    assert task.loc[0, "positive_b"] == "000_002"


def test_load_pair_task_rejects_missing_file(tmp_path):
    with pytest.raises(VideoRetrievalError):
        load_pair_task(tmp_path / "absent.csv")


def test_load_pair_task_rejects_repeated_clip(tmp_path):
    path = write_pair_file(
        tmp_path / "pairs.csv", [["000_001", "000_002", "001_003", "001_003"]]
    )
    with pytest.raises(VideoRetrievalError):
        load_pair_task(path)


def test_load_pair_task_rejects_too_few_columns(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("a,b\n000_001,000_002\n", encoding="utf-8")
    with pytest.raises(VideoRetrievalError):
        load_pair_task(path)


# --------------------------------------------------------------------------- #
# Clip ids
# --------------------------------------------------------------------------- #
@pytest.fixture
def prepared(catalogue):
    from animal_re_id.data.splits import add_sequence_columns

    return add_sequence_columns(catalogue)


def test_clip_id_from_image_id_drops_camera_and_frame():
    assert clip_id_from_image_id("009C01T109F000") == "009_109"
    assert clip_id_from_image_id("000C03T000F099") == "000_000"


def test_clip_id_from_image_id_rejects_unknown_layout():
    with pytest.raises(VideoRetrievalError):
        clip_id_from_image_id("Charlotte|0")


def test_sequence_clip_ids_is_one_row_per_sequence(prepared):
    mapping = sequence_clip_ids(prepared)
    assert len(mapping) == prepared["sequence"].nunique()
    assert mapping["clip_id"].is_unique
    assert mapping["sequence"].is_unique


def test_sequence_clip_ids_rejects_ambiguous_ids(prepared):
    ambiguous = prepared.copy()
    ambiguous["image_id"] = "000C01T000F000"
    with pytest.raises(VideoRetrievalError):
        sequence_clip_ids(ambiguous)


def test_resolve_pair_task_translates_ids(prepared, tmp_path):
    mapping = sequence_clip_ids(prepared)
    row = mapping.iloc[0]
    other = mapping.iloc[1]
    third = mapping.iloc[2]
    path = write_pair_file(
        tmp_path / "pairs.csv",
        [[row.clip_id, other.clip_id, third.clip_id]],
    )
    resolved = resolve_pair_task(load_pair_task(path), mapping)
    assert resolved.loc[0, "positive_a"] == row.sequence
    assert resolved.loc[0, "positive_b"] == other.sequence
    assert resolved.loc[0, "candidates"][0] == row.sequence
    assert resolved.loc[0, "clip_ids"][0] == row.clip_id


def test_resolve_pair_task_rejects_unmapped_clip(prepared, tmp_path):
    path = write_pair_file(
        tmp_path / "pairs.csv", [["999_999", "000_000", "001_001"]]
    )
    with pytest.raises(VideoRetrievalError):
        resolve_pair_task(load_pair_task(path), sequence_clip_ids(prepared))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def make_task(rows):
    return pd.DataFrame(
        [
            {
                "pair_id": index,
                "positive_a": row[0],
                "positive_b": row[1],
                "candidates": tuple(row),
            }
            for index, row in enumerate(rows)
        ]
    )


def test_validate_pair_task_accepts_a_resolvable_task():
    task = make_task([["000_001", "000_002", "001_003"]])
    validate_pair_task(task, {"000_001", "000_002", "001_003"})


def test_validate_pair_task_allows_clips_shared_across_rows():
    """The official file reuses a clip as first member and as second member."""
    task = make_task(
        [
            ["000_001", "000_002", "001_003"],
            ["000_002", "000_004", "001_005"],
        ]
    )
    available = {"000_001", "000_002", "000_004", "001_003", "001_005"}
    validate_pair_task(task, available)


def test_validate_pair_task_rejects_a_pair_of_different_identities():
    task = make_task([["000_001", "000_002", "000_003"]])
    task.loc[0, "positive_b"] = "001_002"
    with pytest.raises(VideoRetrievalError):
        validate_pair_task(task, {"000_001", "001_002", "000_003"})


def test_validate_pair_task_rejects_a_self_pair():
    task = make_task([["000_001", "000_001", "001_003"]])
    with pytest.raises(VideoRetrievalError):
        validate_pair_task(task, {"000_001", "001_003"})


def test_clip_identity_understands_both_naming_schemes():
    from animal_re_id.evaluation.video_retrieval import clip_identity

    assert clip_identity("009_109") == "009"
    assert clip_identity("Charlotte|91") == "Charlotte"
    with pytest.raises(VideoRetrievalError):
        clip_identity("Charlotte")


def test_validate_pair_task_reports_missing_embeddings():
    task = make_task([["000_001", "000_002", "001_003"]])
    with pytest.raises(VideoRetrievalError):
        validate_pair_task(task, {"000_001", "000_002"})


def test_validate_pair_task_rejects_single_identity():
    task = make_task([["000_001", "000_002", "000_003"]])
    with pytest.raises(VideoRetrievalError):
        validate_pair_task(task, {"000_001", "000_002", "000_003"})


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
@pytest.fixture
def pair_fixture():
    """Three pairs of six candidates; the reference loop needs >=5 to reach 3."""
    task = make_task(
        [
            ["000_001", "000_002", "001_003", "001_004", "001_005", "001_006"],
            ["002_001", "002_002", "003_003", "003_004", "003_005", "003_006"],
            ["004_001", "004_002", "005_003", "005_004", "005_005", "005_006"],
        ]
    )
    embeddings = {
        # Pair 1: partner is the nearest candidate in both directions.
        "000_001": np.array([0.0, 0.0]),
        "000_002": np.array([0.1, 0.0]),
        "001_003": np.array([5.0, 0.0]),
        "001_004": np.array([6.0, 0.0]),
        "001_005": np.array([7.0, 0.0]),
        "001_006": np.array([8.0, 0.0]),
        # Pair 2: partner is fourth in both directions (outside top-3).
        "002_001": np.array([10.0, 10.0]),
        "002_002": np.array([0.0, 50.0]),
        "003_003": np.array([20.0, 10.5]),
        "003_004": np.array([5.0, 40.0]),
        "003_005": np.array([15.0, 45.0]),
        "003_006": np.array([30.0, 60.0]),
        # Pair 3: partner is second, so top-3 yes but top-1 no.
        "004_001": np.array([0.0, 0.0]),
        "004_002": np.array([0.0, 1.0]),
        "005_003": np.array([0.0, 0.5]),
        "005_004": np.array([0.0, 2.5]),
        "005_005": np.array([0.0, 3.0]),
        "005_006": np.array([0.0, 4.0]),
    }
    return task, embeddings


def test_rank_pair_task_scores_both_directions(pair_fixture):
    task, embeddings = pair_fixture
    result = rank_pair_task(embeddings, task)

    assert result.n_pairs == 3
    assert result.n_trials == 6
    assert result.n_gallery == 6
    assert result.per_trial["rank_of_positive"].tolist() == [1, 1, 4, 4, 2, 2]
    assert result.top1 == pytest.approx(2 / 6)
    assert result.top3 == pytest.approx(4 / 6)


def test_rank_pair_task_distinguishes_top1_from_top3(pair_fixture):
    task, embeddings = pair_fixture
    result = rank_pair_task(embeddings, task)
    third_pair = result.per_trial[result.per_trial["pair_id"] == 2]
    assert third_pair["top1"].tolist() == [False, False]
    assert third_pair["top3"].tolist() == [True, True]


def test_rank_pair_task_never_ranks_the_query_first(pair_fixture):
    task, embeddings = pair_fixture
    result = rank_pair_task(embeddings, task)
    assert (result.per_trial["rank_of_positive"] >= 1).all()
    assert not (result.per_trial["query"] == result.per_trial["positive"]).any()


def test_rank_pair_task_reports_no_map(pair_fixture):
    task, embeddings = pair_fixture
    payload = rank_pair_task(embeddings, task).to_dict()
    assert payload["mAP_defined"] is False
    assert payload["mean_average_precision"] is None


def test_official_ranking_drops_a_tied_third_place():
    from animal_re_id.evaluation.video_retrieval import official_ranking

    distances = np.array([0.0, 0.5, 0.7, 0.7, 0.9])
    assert official_ranking(distances, banned_index=0) == [1, 3]

    distances = np.array([0.0, 0.5, 0.6, 0.7, 0.9])
    assert official_ranking(distances, banned_index=0) == [1, 2, 3]


def test_official_ranking_returns_nothing_when_all_candidates_tie():
    from animal_re_id.evaluation.video_retrieval import official_ranking

    distances = np.array([0.0, 0.5, 0.5])
    assert official_ranking(distances, banned_index=0) == []


def test_rank_pair_task_uses_the_official_ranking_rule():
    task = make_task([["000_001", "000_002", "001_003"]])
    embeddings = {
        "000_001": np.array([0.0, 0.0]),
        "000_002": np.array([1.0, 0.0]),
        "001_003": np.array([1.0, 0.0]),
    }
    result = rank_pair_task(embeddings, task)
    # Both candidates are exactly equidistant, so the reference rule yields no
    # ranked candidate and the trial cannot be a hit.
    assert result.per_trial["top1"].tolist() == [False, False]
    assert result.per_trial["top3"].tolist() == [False, False]
    assert result.per_trial["official_top3"].tolist() == [(), (2,)]


def test_rank_pair_task_accepts_an_embedding_frame(pair_fixture):
    task, embeddings = pair_fixture
    frame = pd.DataFrame(
        {
            "sequence": list(embeddings),
            "embedding": [value for value in embeddings.values()],
        }
    )
    assert rank_pair_task(frame, task).top1 == pytest.approx(2 / 6)


def test_rank_pair_task_rejects_missing_embedding(pair_fixture):
    task, embeddings = pair_fixture
    embeddings = dict(embeddings)
    embeddings.pop("003_004")
    with pytest.raises(VideoRetrievalError):
        rank_pair_task(embeddings, task)
