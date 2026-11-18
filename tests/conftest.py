"""Shared fixtures: a tiny PolarBearVidID-shaped dataset on disk.

The fixture mirrors the real release layout, including the official fold files
``track_fold_info_1.csv`` ... ``track_fold_info_5.csv`` and the zero-based
tracklet and frame numbering, so the production split code paths are exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

IDENTITIES = [
    (0, "Vera", "Nuremberg"),
    (1, "Nanuq", "Nuremberg"),
    (2, "Tonja", "Berlin"),
]


def write_dataset(
    root: Path,
    identities: list[tuple[int, str, str]] = IDENTITIES,
    sequences_per_identity: int = 6,
    frames_per_sequence: int = 4,
    image_size: tuple[int, int] = (32, 16),
    n_folds: int = 5,
    write_track_info: bool = True,
    write_folds: bool = True,
) -> Path:
    """Create the minimum structure the loader and split code validate."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    tracks = []
    rng = np.random.default_rng(0)

    for identifier, name, zoo in identities:
        folder = root / str(identifier).zfill(3)
        folder.mkdir(exist_ok=True)
        rows.append(
            {
                "id": identifier,
                "name": name,
                "type": "PolarBear",
                "sex": "female",
                "birth": 2000 + identifier,
                "zoo": zoo,
                "tracklets": sequences_per_identity,
            }
        )
        for tracklet in range(sequences_per_identity):
            tracks.append(
                {
                    "id": identifier,
                    "cam": 1,
                    "tracklet": tracklet,
                    "start": 0,
                    "end": frames_per_sequence - 1,
                }
            )
            colour = int(rng.integers(0, 255))
            for frame in range(frames_per_sequence):
                file_name = f"{identifier:03d}C01T{tracklet:03d}F{frame:03d}.jpg"
                image = Image.fromarray(
                    np.full(
                        (image_size[1], image_size[0], 3),
                        (colour, identifier * 40 % 255, 90),
                        dtype=np.uint8,
                    )
                )
                image.save(folder / file_name)

    pd.DataFrame(rows).to_csv(root / "animal_db.csv", index=False)
    tracks = pd.DataFrame(tracks)
    if write_track_info:
        tracks.to_csv(root / "track_info.csv", index=False)
    if write_folds:
        _write_fold_files(root, tracks, n_folds=n_folds)
    return root


def _write_fold_files(root: Path, tracks: pd.DataFrame, n_folds: int = 5) -> None:
    """Round-robin tracklets into ``n_folds`` files, per identity."""
    tracks = tracks.copy()
    tracks["fold"] = tracks.groupby("id").cumcount() % n_folds + 1
    for fold in range(1, n_folds + 1):
        tracks.loc[tracks["fold"] == fold].drop(columns="fold").to_csv(
            root / f"track_fold_info_{fold}.csv", index=False
        )


@pytest.fixture
def make_dataset():
    """Factory fixture so tests can build datasets with custom sizes."""
    return write_dataset


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    return write_dataset(tmp_path / "PolarBearVidID")


@pytest.fixture
def catalogue(synthetic_root: Path) -> pd.DataFrame:
    from animal_re_id.data.wildlife import load_polar_bear_vid_id

    return load_polar_bear_vid_id(synthetic_root)
