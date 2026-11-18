"""Frame sampling and temporal pooling.

The sampling rules are copied from the RoVF reference implementation, so these
tests pin the exact index sequences rather than only their shape. If a test
here fails, published numbers stop being comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from animal_re_id.data.splits import add_sequence_columns
from animal_re_id.data.video import (
    FRAMES_PER_SEQUENCE,
    FrameSamplingError,
    nested_frame_indices,
    pool_sequence,
    sample_frame_indices,
    sampled_frames,
    sequence_embeddings,
    temporal_mean,
)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_sample_matches_official_linspace():
    for n_frames in (1, 2, 5, 20, 21, 100, 143):
        expected = np.linspace(0, n_frames - 1, FRAMES_PER_SEQUENCE, dtype=int)
        assert np.array_equal(sample_frame_indices(n_frames), expected)


def test_sample_includes_first_and_last_frame():
    positions = sample_frame_indices(101)
    assert positions[0] == 0
    assert positions[-1] == 100
    assert len(positions) == FRAMES_PER_SEQUENCE


def test_sample_is_deterministic():
    first = sample_frame_indices(77)
    second = sample_frame_indices(77)
    assert np.array_equal(first, second)


def test_short_sequences_repeat_frames_instead_of_failing():
    """Reference behaviour: ``linspace`` pads short clips by reusing frames."""
    positions = sample_frame_indices(4)
    assert len(positions) == FRAMES_PER_SEQUENCE
    assert set(positions) <= {0, 1, 2, 3}
    assert positions[0] == 0 and positions[-1] == 3


def test_sampling_rejects_degenerate_inputs():
    with pytest.raises(FrameSamplingError):
        sample_frame_indices(0)
    with pytest.raises(FrameSamplingError):
        sample_frame_indices(10, count=0)


def test_nested_indices_composes_both_steps():
    positions = nested_frame_indices(101, stored=20, selected=5)
    clip = np.linspace(0, 100, 20, dtype=int)
    keep = np.linspace(0, 19, 5, dtype=int)
    assert np.array_equal(positions, clip[keep])
    assert len(positions) == 5
    assert positions[0] == 0 and positions[-1] == 100


def test_nested_indices_rejects_more_selected_than_stored():
    with pytest.raises(FrameSamplingError):
        nested_frame_indices(101, stored=20, selected=21)


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #
def test_temporal_mean_is_mean_over_frames_and_keeps_dtype():
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32")
    result = temporal_mean(features)
    assert result.dtype == np.float32
    assert np.allclose(result, [3.0, 4.0])


def test_temporal_mean_rejects_bad_shapes_and_non_finite():
    with pytest.raises(FrameSamplingError):
        temporal_mean(np.zeros((3,)))
    with pytest.raises(FrameSamplingError):
        temporal_mean(np.zeros((0, 4)))
    with pytest.raises(FrameSamplingError):
        temporal_mean(np.array([[1.0, np.nan]]))


def test_pool_sequence_selects_then_averages():
    rng = np.random.default_rng(0)
    stored = rng.normal(size=(20, 8)).astype("float32")
    result = pool_sequence(stored, n_source_frames=101, stored=20, selected=5)
    keep = np.linspace(0, 19, 5, dtype=int)
    assert np.allclose(result, stored[keep].mean(axis=0))


def test_pool_sequence_requires_a_full_clip():
    with pytest.raises(FrameSamplingError):
        pool_sequence(np.zeros((19, 4)), n_source_frames=101, stored=20, selected=5)


# --------------------------------------------------------------------------- #
# Catalogue sampling
# --------------------------------------------------------------------------- #
@pytest.fixture
def prepared(catalogue: pd.DataFrame) -> pd.DataFrame:
    return add_sequence_columns(catalogue)


def test_sampled_frames_keeps_sequence_and_order(prepared: pd.DataFrame):
    sampled = sampled_frames(prepared, count=FRAMES_PER_SEQUENCE)
    assert len(sampled) == prepared["sequence"].nunique() * FRAMES_PER_SEQUENCE

    for sequence, group in sampled.groupby("sequence"):
        ordered = group.sort_values("sample_order")
        assert ordered["sample_order"].tolist() == list(range(FRAMES_PER_SEQUENCE))
        assert ordered["position"].is_monotonic_increasing
        # Frames must be read in tracklet order, not alphabetically or shuffled.
        assert ordered["frame"].is_monotonic_increasing


def test_sampled_frames_reports_source_positions(prepared: pd.DataFrame):
    sampled = sampled_frames(prepared)
    group = sampled[sampled["sequence"] == prepared["sequence"].iloc[0]]
    n_frames = int((prepared["sequence"] == group["sequence"].iloc[0]).sum())
    assert group["position"].tolist() == sample_frame_indices(n_frames).tolist()


def test_sampled_frames_is_deterministic(prepared: pd.DataFrame):
    first = sampled_frames(prepared)
    second = sampled_frames(prepared)
    pd.testing.assert_frame_equal(first, second)


def test_sampled_frames_requires_prepared_catalogue(catalogue: pd.DataFrame):
    with pytest.raises(FrameSamplingError):
        sampled_frames(catalogue)


def test_sequence_embeddings_align_with_sequences(prepared: pd.DataFrame):
    sampled = sampled_frames(prepared, count=4)
    rng = np.random.default_rng(1)
    features = rng.normal(size=(len(sampled), 6)).astype("float32")
    n_source = prepared.groupby("sequence").size().to_dict()

    metadata, matrix = sequence_embeddings(
        features, sampled, n_source, stored=4, selected=2
    )

    assert matrix.shape == (prepared["sequence"].nunique(), 6)
    assert metadata["sequence"].is_monotonic_increasing
    assert metadata["sequence"].tolist() == sorted(prepared["sequence"].unique())

    # Recompute one sequence by hand from the sampled rows.
    sequence = metadata["sequence"].iloc[0]
    rows = sampled[sampled["sequence"] == sequence].sort_values("sample_order").index
    expected = features[rows][np.linspace(0, 3, 2, dtype=int)].mean(axis=0)
    assert np.allclose(matrix[0], expected)


def test_sequence_embeddings_rejects_misaligned_features(prepared: pd.DataFrame):
    sampled = sampled_frames(prepared, count=4)
    n_source = prepared.groupby("sequence").size().to_dict()
    with pytest.raises(FrameSamplingError):
        sequence_embeddings(
            np.zeros((len(sampled) - 1, 6)), sampled, n_source, stored=4, selected=2
        )
