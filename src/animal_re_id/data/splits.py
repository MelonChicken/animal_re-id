"""Sequence-level dataset splits for PolarBearVidID.

A sequence is identified by the composite key ``(identity, video)``. Video ids
are only unique per identity, so neither column alone is a valid key.

Three protocols are supported:

``five_fold``
    The image-level protocol of the PolarBearVidID paper (Zuerl et al., 2023).
    The archive ships the fold membership used in the paper as
    ``track_fold_info_1.csv`` ... ``track_fold_info_5.csv``, one row per
    tracklet ``(id, cam, tracklet, start, end)``. Those files are the protocol:
    they partition all 1,431 tracklets and reproduce Table A1 exactly. Each fold
    is evaluated as a closed-set leave-one-out retrieval task.

``five_fold_stratified``
    A custom, seeded re-implementation of the same five-fold design. It is kept
    only to measure how much the fold assignment itself moves the numbers,
    because the official files are a single fixed draw. It is *not* the paper's
    protocol and results from it must be labelled as custom.

``rovf_split``
    The sequence split of the official RoVF repository
    (``Dataset/convert_polar_bear_dataset.py``): per identity, iterating
    ``track_info.csv`` in file order, the first 70% of tracklets are train, the
    next 10% validation and the last 20% test. Test tracklets become query and
    gallery.

This module only produces assignments and retrieval tasks. It never reads image
content and never trains anything.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEQUENCE_COLUMNS = ["identity", "video"]
FRAME_PATTERN = re.compile(r"F(\d+)$")
SEQUENCE_KEY_SEPARATOR = "|"
FOLD_FILE_TEMPLATE = "track_fold_info_{fold}.csv"
FOLD_FILE_COLUMNS = ["id", "cam", "tracklet", "start", "end"]
ANIMAL_DB_FILE = "animal_db.csv"
TRACK_INFO_FILE = "track_info.csv"

PROTOCOL_FIVE_FOLD = "five_fold"
PROTOCOL_FIVE_FOLD_STRATIFIED = "five_fold_stratified"
PROTOCOL_ROVF = "rovf_split"
PROTOCOLS = (PROTOCOL_FIVE_FOLD, PROTOCOL_FIVE_FOLD_STRATIFIED, PROTOCOL_ROVF)

GALLERY_SCOPES = ("fold", "all")


class SplitError(ValueError):
    """Raised when a split violates a structural assumption of the protocol."""


# --------------------------------------------------------------------------- #
# Catalogue preparation
# --------------------------------------------------------------------------- #
def parse_frame_number(image_id: str) -> int:
    """Return the frame index encoded in a PolarBearVidID file name.

    File names look like ``000C01T000F049.jpg``; ``image_id`` is the file name
    without extension. Frame numbering is zero-based in this release.
    """
    match = FRAME_PATTERN.search(str(image_id))
    if match is None:
        raise SplitError(f"Cannot parse a frame number from image id {image_id!r}")
    return int(match.group(1))


def add_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the catalogue with ``sequence`` and ``frame`` columns.

    ``sequence`` is ``"<identity>|<video>"``. The result is sorted by
    ``(identity, video, frame)`` so every downstream step is order-independent.
    """
    missing = [column for column in (*SEQUENCE_COLUMNS, "image_id", "path") if column not in df]
    if missing:
        raise SplitError(f"Catalogue is missing required columns: {sorted(missing)}")

    catalogue = df.copy()
    if catalogue["identity"].isna().any() or catalogue["video"].isna().any():
        raise SplitError("Catalogue contains rows without identity or video")
    catalogue["identity"] = catalogue["identity"].astype(str)
    catalogue["video"] = catalogue["video"].astype("int64")
    catalogue["sequence"] = (
        catalogue["identity"] + SEQUENCE_KEY_SEPARATOR + catalogue["video"].astype(str)
    )
    catalogue["frame"] = catalogue["image_id"].map(parse_frame_number)

    duplicated = catalogue["path"].duplicated(keep=False)
    if duplicated.any():
        raise SplitError(f"Catalogue contains {int(duplicated.sum())} duplicated image paths")

    return catalogue.sort_values(["identity", "video", "frame"]).reset_index(drop=True)


def sequence_table(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Return one row per sequence with its frame count and frame range."""
    table = (
        catalogue.groupby(SEQUENCE_COLUMNS, sort=True)
        .agg(
            sequence=("sequence", "first"),
            n_frames=("frame", "size"),
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
        )
        .reset_index()
    )
    return table.sort_values(SEQUENCE_COLUMNS).reset_index(drop=True)


def representative_frames(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Return the frame closest to the middle of each sequence.

    The PolarBearVidID paper states that its image-level evaluation used a
    single image per sequence but does not publish which frame. The middle frame
    is used here because it minimises the distance to every other frame of the
    tracklet, and it is a deterministic, documented choice. For an even number
    of frames the lower of the two middle frames is taken.
    """
    target = catalogue.groupby(SEQUENCE_COLUMNS)["frame"].transform("median")
    distance = (catalogue["frame"] - target).abs()
    ties = distance.groupby([catalogue["identity"], catalogue["video"]]).transform("min")
    candidates = catalogue.loc[distance.eq(ties)]
    return (
        candidates.sort_values([*SEQUENCE_COLUMNS, "frame"])
        .groupby(SEQUENCE_COLUMNS, sort=True)
        .head(1)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Official metadata tables
# --------------------------------------------------------------------------- #
def identity_id_map(dataset_root: str | Path) -> dict[str, int]:
    """Return the mapping ``identity name -> numeric id`` from ``animal_db.csv``."""
    path = Path(dataset_root) / ANIMAL_DB_FILE
    if not path.is_file():
        raise SplitError(f"Missing {path}; cannot resolve identity ids")
    animals = pd.read_csv(path)
    for column in ("id", "name"):
        if column not in animals:
            raise SplitError(f"{path} is missing the {column!r} column")
    if animals["name"].duplicated().any():
        raise SplitError(f"{path} maps one identity name to several ids")
    return {str(name): int(identifier) for identifier, name in zip(animals["id"], animals["name"], strict=True)}


def load_official_folds(dataset_root: str | Path, n_folds: int = 5) -> pd.DataFrame:
    """Load ``track_fold_info_<n>.csv`` into an ``identity, video, fold`` table."""
    root = Path(dataset_root)
    frames = []
    for fold in range(1, n_folds + 1):
        path = root / FOLD_FILE_TEMPLATE.format(fold=fold)
        if not path.is_file():
            raise SplitError(f"Missing official fold file {path}")
        table = pd.read_csv(path)
        missing = [column for column in FOLD_FILE_COLUMNS if column not in table]
        if missing:
            raise SplitError(f"{path} is missing columns {missing}")
        frames.append(table.assign(fold=fold))

    folds = pd.concat(frames, ignore_index=True)
    if folds[["id", "cam", "tracklet"]].duplicated().any():
        raise SplitError("Official fold files assign a tracklet to more than one fold")

    # ``video`` in the WildlifeDatasets catalogue is the 3-digit tracklet id, and
    # (id, tracklet) is unique in this release (one camera per tracklet).
    if folds.duplicated(["id", "tracklet"]).any():
        raise SplitError("Official fold files are ambiguous: (id, tracklet) is not unique")

    animals = pd.read_csv(root / ANIMAL_DB_FILE, usecols=["id", "name"])
    folds = folds.merge(animals, on="id", how="left", validate="many_to_one")
    if folds["name"].isna().any():
        raise SplitError("Official fold files reference identity ids missing from animal_db.csv")

    return (
        folds.rename(columns={"name": "identity", "tracklet": "video"})
        .loc[:, ["identity", "video", "fold"]]
        .assign(identity=lambda frame: frame["identity"].astype(str), video=lambda frame: frame["video"].astype("int64"))
        .sort_values(["identity", "video"])
        .reset_index(drop=True)
    )


def track_info_order(dataset_root: str | Path) -> pd.DataFrame:
    """Return ``track_info.csv`` with ``identity, video`` and its file order."""
    path = Path(dataset_root) / TRACK_INFO_FILE
    if not path.is_file():
        raise SplitError(f"Missing {path}; the RoVF split is defined over its row order")
    tracks = pd.read_csv(path, usecols=["id", "tracklet"])
    animals = pd.read_csv(Path(dataset_root) / ANIMAL_DB_FILE, usecols=["id", "name"])
    tracks = tracks.merge(animals, on="id", how="left", validate="many_to_one")
    if tracks["name"].isna().any():
        raise SplitError(f"{path} references identity ids missing from animal_db.csv")
    tracks = tracks.rename(columns={"name": "identity", "tracklet": "video"})
    tracks["identity"] = tracks["identity"].astype(str)
    tracks["video"] = tracks["video"].astype("int64")
    return tracks.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #
def five_fold_assignment(
    catalogue: pd.DataFrame, dataset_root: str | Path, n_folds: int = 5
) -> pd.DataFrame:
    """Assign every sequence to the fold it has in the official release."""
    sequences = sequence_table(catalogue)
    folds = load_official_folds(dataset_root, n_folds=n_folds)

    assignment = sequences.merge(folds, on=SEQUENCE_COLUMNS, how="left", validate="one_to_one")
    unpublished = assignment.loc[assignment["fold"].isna(), "sequence"]
    if not unpublished.empty:
        raise SplitError(
            f"{len(unpublished)} sequences have no official fold, e.g. {unpublished.iloc[0]!r}"
        )
    assignment["fold"] = assignment["fold"].astype("int64")
    assignment["protocol"] = PROTOCOL_FIVE_FOLD

    published = set(map(tuple, folds[SEQUENCE_COLUMNS].to_numpy()))
    observed = set(map(tuple, sequences[SEQUENCE_COLUMNS].to_numpy()))
    if stray := published - observed:
        raise SplitError(f"Official folds list {len(stray)} tracklets absent from the catalogue")

    _check_every_identity_in_every_fold(assignment, n_folds)
    return assignment.sort_values("sequence").reset_index(drop=True)


def stratified_five_fold_assignment(
    catalogue: pd.DataFrame, seed: int = 0, n_folds: int = 5
) -> pd.DataFrame:
    """Custom seeded re-draw of the same five-fold design.

    Folds are stratified per identity so that each fold holds the same share of
    every animal. This is *not* the paper's split; it exists only to measure how
    sensitive the metrics are to the particular fold assignment.
    """
    if n_folds < 2:
        raise SplitError("n_folds must be at least 2")

    sequences = sequence_table(catalogue)
    rng = np.random.default_rng(seed)
    folds = np.empty(len(sequences), dtype="int64")

    for _, group in sequences.groupby("identity", sort=True):
        shuffled = rng.permutation(group.index.to_numpy())
        for fold_index, chunk in enumerate(np.array_split(shuffled, n_folds), start=1):
            folds[chunk] = fold_index

    assignment = sequences.assign(fold=folds, protocol=PROTOCOL_FIVE_FOLD_STRATIFIED)
    _check_every_identity_in_every_fold(assignment, n_folds)
    return assignment.sort_values("sequence").reset_index(drop=True)


def rovf_assignment(
    catalogue: pd.DataFrame,
    dataset_root: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.8,
) -> pd.DataFrame:
    """Reproduce the RoVF repository's PolarBearVidID tracklet split.

    The reference implementation iterates ``track_info.csv`` in file order,
    increments a counter per identity and sends the first 70% of tracklets to
    train, the next 10% to validation and the remaining 20% to test. The row
    order of the published ``track_info.csv`` is reused verbatim rather than
    re-derived, so the split cannot drift if the file is re-sorted.
    """
    if not 0 < train_ratio < val_ratio < 1:
        raise SplitError("Expected 0 < train_ratio < val_ratio < 1")

    order = track_info_order(dataset_root)
    sequences = sequence_table(catalogue)

    assignment = (
        order.reset_index()
        .rename(columns={"index": "position"})
        .merge(sequences, on=SEQUENCE_COLUMNS, how="outer", validate="one_to_one", indicator=True)
    )
    unmatched = assignment.loc[assignment["_merge"] != "both"]
    if not unmatched.empty:
        raise SplitError(
            f"track_info.csv and the catalogue disagree on {len(unmatched)} sequences"
        )
    assignment = assignment.drop(columns="_merge").sort_values("position").reset_index(drop=True)
    assignment["rank"] = assignment.groupby("identity")["position"].rank(method="first") - 1
    assignment["total"] = assignment.groupby("identity")["position"].transform("size")

    assignment["split"] = np.where(
        assignment["rank"] >= assignment["total"] * val_ratio,
        "test",
        np.where(
            assignment["rank"] >= assignment["total"] * train_ratio, "val", "train"
        ),
    )
    assignment = assignment.drop(columns=["rank", "total"]).assign(protocol=PROTOCOL_ROVF)
    for name in ("train", "val", "test"):
        if assignment.loc[assignment["split"] == name].empty:
            raise SplitError(f"RoVF split produced an empty {name!r} set")
    _check_sequence_disjoint(assignment, "split")
    return (
        assignment.drop(columns="position")
        .sort_values("sequence")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _check_every_identity_in_every_fold(assignment: pd.DataFrame, n_folds: int) -> None:
    expected = set(range(1, n_folds + 1))
    for identity, group in assignment.groupby("identity"):
        missing = expected - set(group["fold"])
        if missing:
            raise SplitError(f"Identity {identity!r} is missing from folds {sorted(missing)}")


def _check_sequence_disjoint(assignment: pd.DataFrame, column: str) -> None:
    """Fail if the same ``(identity, video)`` sequence appears in two groups."""
    duplicated = assignment.duplicated(["sequence", column])
    if duplicated.any():
        raise SplitError(
            f"{int(duplicated.sum())} sequences are assigned to more than one {column}"
        )


def check_assignments(assignment: pd.DataFrame) -> None:
    """Validate the structural invariants shared by every protocol."""
    for column in ("sequence", "identity", "video", "protocol"):
        if column not in assignment:
            raise SplitError(f"Assignment is missing the {column!r} column")
    if assignment["sequence"].duplicated().any():
        raise SplitError("Assignment contains duplicated sequences")
    grouped = assignment.groupby("sequence").agg(
        identities=("identity", "nunique"), videos=("video", "nunique")
    )
    if (grouped["identities"] != 1).any() or (grouped["videos"] != 1).any():
        raise SplitError("A sequence key maps to more than one (identity, video) pair")

    protocols = set(assignment["protocol"].astype(str))
    if len(protocols) != 1:
        raise SplitError(f"Assignment mixes protocols {sorted(protocols)}")

    if "fold" in assignment and assignment["fold"].isna().any():
        raise SplitError("Fold assignment contains missing folds")
    if "split" in assignment and assignment["split"].isna().any():
        raise SplitError("Split assignment contains missing splits")


def build_retrieval_task(
    representative: pd.DataFrame,
    assignment: pd.DataFrame,
    protocol: str,
    fold: int | None = None,
    gallery_scope: str = "fold",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the ``(query, gallery)`` frames of one evaluation unit.

    Both frames contain query images: the protocol is leave-one-out, so the
    query image also appears in the gallery and is masked out as a self-match
    during scoring. Every other sequence of the same identity supplies
    positives; other identities supply negatives.

    Args:
        gallery_scope: ``"fold"`` restricts the gallery to the held-out unit
            (the primary reading of "five-fold closed-set evaluation"), while
            ``"all"`` ranks against every sequence in the dataset.
    """
    if gallery_scope not in GALLERY_SCOPES:
        raise SplitError(f"Unknown gallery scope {gallery_scope!r}; expected {GALLERY_SCOPES}")

    if protocol in (PROTOCOL_FIVE_FOLD, PROTOCOL_FIVE_FOLD_STRATIFIED):
        if fold is None:
            raise SplitError(f"{protocol} retrieval needs an explicit fold")
        if "fold" not in assignment:
            raise SplitError(f"{protocol} assignment has no fold column")
        selected = assignment.loc[assignment["fold"] == fold, "sequence"]
    elif protocol == PROTOCOL_ROVF:
        selected = assignment.loc[assignment["split"] == "test", "sequence"]
    else:
        raise SplitError(f"Unknown protocol {protocol!r}; expected one of {PROTOCOLS}")

    query = representative.loc[representative["sequence"].isin(set(selected))]
    if query.empty:
        raise SplitError(f"Protocol {protocol!r} selected no frames")
    query = query.sort_values("sequence").reset_index(drop=True)

    if gallery_scope == "all":
        gallery = representative.sort_values("sequence").reset_index(drop=True)
    else:
        gallery = query.copy()

    validate_retrieval_task(query, gallery, protocol=protocol)
    return query, gallery


def validate_retrieval_task(
    query: pd.DataFrame, gallery: pd.DataFrame, protocol: str, min_positives: int = 1
) -> None:
    """Check that the task is a well-formed closed-set retrieval problem.

    Raises:
        SplitError: duplicated images, missing positives, or missing identities.
    """
    for name, frame in (("query", query), ("gallery", gallery)):
        if frame.empty:
            raise SplitError(f"{name} set is empty")
        if frame["path"].duplicated().any():
            raise SplitError(f"{name} set contains duplicated image paths")

    query_identities = set(query["identity"])
    gallery_identities = set(gallery["identity"])
    unknown = query_identities - gallery_identities
    if unknown:
        raise SplitError(
            f"Closed-set {protocol} retrieval requires every query identity in the "
            f"gallery; missing {sorted(unknown)}"
        )

    available = gallery.groupby("identity")["path"].apply(list)
    for identity, path, sequence in zip(
        query["identity"], query["path"], query["sequence"], strict=True
    ):
        positives = [candidate for candidate in available[identity] if candidate != path]
        if len(positives) < min_positives:
            raise SplitError(
                f"Sequence {sequence!r} has {len(positives)} positive gallery images "
                f"(at least {min_positives} required)"
            )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def split_signature(assignment: pd.DataFrame, extra: dict | None = None) -> str:
    """Return a stable hash of a split, used to protect the feature cache."""
    payload = {
        "protocol": str(assignment["protocol"].iloc[0]),
        "rows": assignment.sort_values("sequence")[
            ["sequence", "identity", "video"]
            + [column for column in ("fold", "split") if column in assignment]
        ].to_dict(orient="records"),
        "extra": extra or {},
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_split(
    assignment: pd.DataFrame,
    directory: str | Path,
    name: str,
    metadata: dict | None = None,
) -> Path:
    """Write a split to ``directory`` and return the file path."""
    check_assignments(assignment)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "protocol": str(assignment["protocol"].iloc[0]),
        "signature": split_signature(assignment),
        "metadata": metadata or {},
        "counts": _counts(assignment),
        "sequences": assignment.sort_values("sequence").to_dict(orient="records"),
    }
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_split(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load a split written by :func:`save_split`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assignment = pd.DataFrame(payload["sequences"])
    check_assignments(assignment)
    assignment["protocol"] = assignment["protocol"].astype(str)
    return assignment, payload


def _counts(assignment: pd.DataFrame) -> dict:
    counts: dict = {
        "sequences": int(len(assignment)),
        "identities": int(assignment["identity"].nunique()),
        "frames": int(assignment["n_frames"].sum()),
    }
    for column in ("fold", "split"):
        if column in assignment:
            counts[column] = {
                str(key): int(value)
                for key, value in assignment[column].value_counts().sort_index().items()
            }
    return counts
