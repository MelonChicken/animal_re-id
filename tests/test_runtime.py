"""Runtime helpers: feature cache keys and cache files.

The cache is the reason repeated evaluations are cheap, so a wrong key would
silently score the wrong embeddings. These tests pin what the key covers, and
that a damaged cache file is treated as a miss instead of aborting a run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from animal_re_id.utils.runtime import (
    display,
    feature_cache_key,
    load_cached_features,
    resolve,
    save_cached_features,
    split_path_for,
)

MODEL = {"checkpoint": "facebook/dinov2-small", "preprocessing": "imagenet"}


def make_frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence": ["Anna|0", "Anna|0", "Bruno|1"],
            "path": ["anna0.jpg", "anna1.jpg", "bruno0.jpg"],
            "image_id": ["000C01T000F000", "000C01T000F001", "001C01T001F000"],
        }
    )


# --------------------------------------------------------------------------- #
# Cache key
# --------------------------------------------------------------------------- #
def test_cache_key_is_deterministic():
    frames = make_frames()
    first = feature_cache_key(frames, MODEL, "rovf_linspace20")
    second = feature_cache_key(frames.copy(), dict(MODEL), "rovf_linspace20")
    assert first == second
    assert len(first) == 64


def test_cache_key_changes_with_model_and_selection():
    frames = make_frames()
    base = feature_cache_key(frames, MODEL, "rovf_linspace20")
    assert base != feature_cache_key(frames, {**MODEL, "preprocessing": "rovf"}, "rovf_linspace20")
    assert base != feature_cache_key(frames, MODEL, "uniform10")


def test_cache_key_changes_when_frame_order_changes():
    frames = make_frames()
    base = feature_cache_key(frames, MODEL, "rovf_linspace20")
    shuffled = frames.iloc[[0, 2, 1]].reset_index(drop=True)
    assert base != feature_cache_key(shuffled, MODEL, "rovf_linspace20")


def test_cache_key_ignores_unrelated_columns():
    """Split and protocol columns must not invalidate embeddings."""
    frames = make_frames()
    base = feature_cache_key(frames, MODEL, "rovf_linspace20")
    tagged = frames.assign(split=["test"] * 3, protocol="rovf_split", fold=4)
    assert feature_cache_key(tagged, MODEL, "rovf_linspace20") == base


# --------------------------------------------------------------------------- #
# Cache files
# --------------------------------------------------------------------------- #
def test_feature_cache_round_trip(tmp_path):
    frames = make_frames()
    features = np.arange(12, dtype="float32").reshape(3, 4)
    path = tmp_path / "cache" / "features.npz"

    save_cached_features(path, "key-1", features, frames)
    loaded = load_cached_features(path, "key-1", expected_rows=3)

    assert loaded is not None
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, features)


def test_cache_file_records_the_frames_it_came_from(tmp_path):
    frames = make_frames()
    path = tmp_path / "features.npz"
    save_cached_features(path, "key-1", np.zeros((3, 2), dtype="float32"), frames)

    with np.load(path, allow_pickle=False) as data:
        assert str(data["key"]) == "key-1"
        assert list(data["sequences"]) == frames["sequence"].tolist()
        assert list(data["paths"]) == frames["path"].tolist()


def test_cache_miss_on_key_or_row_mismatch(tmp_path):
    frames = make_frames()
    path = tmp_path / "features.npz"
    save_cached_features(path, "key-1", np.zeros((3, 2), dtype="float32"), frames)

    assert load_cached_features(path, "key-2", expected_rows=3) is None
    assert load_cached_features(path, "key-1", expected_rows=4) is None
    assert load_cached_features(tmp_path / "absent.npz", "key-1", expected_rows=3) is None


def test_truncated_cache_is_a_miss_not_a_crash(tmp_path):
    frames = make_frames()
    path = tmp_path / "features.npz"
    save_cached_features(path, "key-1", np.zeros((3, 2), dtype="float32"), frames)

    # Simulate an interrupted write: keep the head of the archive only.
    content = path.read_bytes()
    path.write_bytes(content[: len(content) // 2])

    assert load_cached_features(path, "key-1", expected_rows=3) is None


def test_save_does_not_leave_a_temporary_file_behind(tmp_path):
    frames = make_frames()
    path = tmp_path / "features.npz"
    save_cached_features(path, "key-1", np.zeros((3, 2), dtype="float32"), frames)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["features.npz"]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def test_split_path_is_stable_and_protocol_specific(tmp_path):
    first = split_path_for(tmp_path, "rovf_split", 0)
    assert first == split_path_for(tmp_path, "rovf_split", 0)
    assert first.name == "polarbear_rovf_split_seed0.json"
    assert first != split_path_for(tmp_path, "rovf_split", 1)
    assert first != split_path_for(tmp_path, "five_fold", 0)


def test_resolve_is_relative_to_the_repository_root():
    assert resolve("experiments/results").is_absolute()
    assert resolve("experiments/results").name == "results"


def test_display_prefers_a_path_relative_to_the_repository():
    shown = display(resolve("experiments/results"))
    assert shown.replace("\\", "/") == "experiments/results"
