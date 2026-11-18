"""The existing loader is the contract the rest of the pipeline builds on.

Note on semantics: ``load_polar_bear_vid_id`` builds its catalogue by scanning
the files that exist on disk, so a deleted frame does not produce a "missing
path" error — it silently disappears from the catalogue instead. That is why the
frame-count cross-check against ``track_info.csv`` exists, and why it is the
thing tested here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from animal_re_id.data.wildlife import _compare_track_manifest, load_polar_bear_vid_id


def test_catalogue_shape_and_identity_count(catalogue: pd.DataFrame) -> None:
    assert len(catalogue) == 3 * 6 * 4
    assert catalogue["identity"].nunique() == 3
    assert catalogue["video"].nunique() == 6  # video ids repeat across identities
    assert catalogue["path"].is_unique
    assert catalogue["path"].map(lambda path: Path(path).is_file()).all()


def test_paths_are_absolute(catalogue: pd.DataFrame) -> None:
    assert catalogue["path"].map(lambda path: Path(path).is_absolute()).all()


def test_video_is_not_a_standalone_sequence_key(catalogue: pd.DataFrame) -> None:
    per_video = catalogue.groupby("video")["identity"].nunique()
    assert (per_video > 1).all(), "fixture must reproduce repeated video ids across identities"

    composite = catalogue[["identity", "video"]].drop_duplicates()
    assert len(composite) == catalogue.groupby(["identity", "video"]).ngroups == 18


def test_missing_metadata_file_is_reported(make_dataset, tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "PolarBearVidID")
    (root / "animal_db.csv").unlink()
    with pytest.raises(FileNotFoundError, match="animal_db.csv"):
        load_polar_bear_vid_id(root)


def test_deleted_frame_is_caught_by_the_track_manifest(make_dataset, tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "PolarBearVidID")
    victim = next((root / "000").glob("*.jpg"))
    victim.unlink()

    catalogue = load_polar_bear_vid_id(root)  # scans the disk, so no error here
    assert len(catalogue) == 3 * 6 * 4 - 1
    stats = _compare_track_manifest(root, catalogue)
    assert stats["track_manifest_checked"] is True
    assert stats["frames_missing_from_declared_tracks"] == 1


def test_track_manifest_disagreement_is_an_error(make_dataset, tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "PolarBearVidID")
    tracks = pd.read_csv(root / "track_info.csv")
    tracks.loc[0, "end"] = tracks.loc[0, "end"] + 2  # claims frames that do not exist
    tracks.to_csv(root / "track_info.csv", index=False)

    catalogue = load_polar_bear_vid_id(root)
    stats = _compare_track_manifest(root, catalogue)
    assert stats["frames_missing_from_declared_tracks"] > 0


def test_consistent_manifest_reports_no_disagreement(synthetic_root: Path) -> None:
    catalogue = load_polar_bear_vid_id(synthetic_root)
    stats = _compare_track_manifest(synthetic_root, catalogue)
    assert stats["track_manifest_checked"] is True
    assert stats["frames_missing_from_declared_tracks"] == 0
    assert stats["frames_beyond_declared_tracks"] == 0
