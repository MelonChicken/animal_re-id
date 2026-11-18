"""Video-level retrieval scoring.

Two protocols are implemented, and they answer different questions.

``official_pair``
    The protocol of the RoVF paper and of
    ``evaluation/get_metrics.py`` + ``Precomputed_test_examples_polarbear.csv``.
    Each row lists eleven clips: the first two are a positive pair (same
    identity, different tracklets) and the remaining nine are negatives drawn
    from nine other identities. Both members of the pair are used as a query in
    turn, so one row yields two trials. The ranking is by Euclidean distance
    and only Top-1 / Top-3 are reported; the paper does not report mAP, so this
    module does not invent one.

    The header of the published CSV is misleading: the column named ``Query``
    is itself the first candidate, because the reference code stacks *all*
    columns as the gallery. This loader keeps every column as a candidate and
    records that fact.

``closed_set``
    A leave-one-out sequence retrieval task, scored with
    :func:`animal_re_id.evaluation.retrieval.rank_retrieval` so Rank-1, Rank-3
    and mAP are defined. It is *not* the RoVF protocol and is reported
    separately.

Distances are Euclidean in both cases, matching the reference implementation
(``compute_distances`` in ``get_metrics.py``). Embeddings are not normalized
by default, again matching the reference; the caller decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# PolarBearVidID frame ids look like ``009C01T000F012``: identity folder,
# camera, tracklet and frame number, each fixed width. The RoVF pair file
# addresses a clip as ``<folder>_<tracklet>``, e.g. ``009_000``.
IMAGE_ID_PATTERN = re.compile(r"^(?P<folder>\d{3})(?P<cam>C\d{2})T(?P<tracklet>\d{3})F\d{3}$")


class VideoRetrievalError(ValueError):
    """Raised when a video retrieval task cannot be scored."""


@dataclass
class PairRetrievalResult:
    top1: float
    top3: float
    n_pairs: int
    n_trials: int
    n_gallery: int
    per_trial: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        return {
            "top1": self.top1,
            "top3": self.top3,
            "n_pairs": self.n_pairs,
            "n_trials": self.n_trials,
            "n_gallery": self.n_gallery,
            "mean_average_precision": None,
            "mAP_defined": False,
        }


def euclidean_distance(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Return the ``(n_query, n_gallery)`` Euclidean distance matrix."""
    query = np.asarray(query, dtype="float64")
    gallery = np.asarray(gallery, dtype="float64")
    if query.ndim != 2 or gallery.ndim != 2:
        raise VideoRetrievalError("Embeddings must be 2D matrices")
    if query.shape[1] != gallery.shape[1]:
        raise VideoRetrievalError(
            f"Embedding dimensions differ: query {query.shape[1]}, gallery {gallery.shape[1]}"
        )
    if not np.isfinite(query).all() or not np.isfinite(gallery).all():
        raise VideoRetrievalError("Embeddings contain non-finite values")
    differences = query[:, np.newaxis, :] - gallery[np.newaxis, :, :]
    return np.sqrt(np.sum(differences**2, axis=2))


def load_pair_task(path: str | Path) -> pd.DataFrame:
    """Load the official pair file into one row of candidate ids per pair.

    Returns a frame with ``pair_id``, ``positive_a``, ``positive_b`` and
    ``candidates`` (a tuple of every column in file order, including the two
    positives). All columns are treated as gallery entries, which is what the
    reference metric code does.
    """
    path = Path(path)
    if not path.is_file():
        raise VideoRetrievalError(
            f"Missing official pair file {path}. Download "
            "Dataset/polarbears_h5files/Precomputed_test_examples_polarbear.csv "
            "from the RoVF repository."
        )
    table = pd.read_csv(path)
    if table.shape[1] < 3:
        raise VideoRetrievalError(
            f"{path} needs at least one positive pair and one negative, got "
            f"{table.shape[1]} columns"
        )

    rows = []
    for index, row in table.iterrows():
        candidates = tuple(str(value) for value in row.tolist())
        if len(set(candidates)) != len(candidates):
            raise VideoRetrievalError(
                f"Row {index} of {path} repeats a clip; the gallery must be unique"
            )
        rows.append(
            {
                "pair_id": index,
                "positive_a": candidates[0],
                "positive_b": candidates[1],
                "candidates": candidates,
            }
        )
    return pd.DataFrame(rows)


def clip_id_from_image_id(image_id: str) -> str:
    """Convert a PolarBearVidID frame id to the clip id used by the pair file.

    ``009C01T109F000`` becomes ``009_109``. The pair file addresses clips by
    identity folder and tracklet only, so the camera is dropped.
    """
    match = IMAGE_ID_PATTERN.match(str(image_id))
    if match is None:
        raise VideoRetrievalError(
            f"Unrecognised PolarBearVidID image id {image_id!r}; expected e.g. 009C01T000F000"
        )
    return f"{match.group('folder')}_{match.group('tracklet')}"


def sequence_clip_ids(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Return ``(sequence, clip_id)`` for every sequence in the catalogue.

    Args:
        catalogue: Frame catalogue from
            :func:`animal_re_id.data.wildlife.load_polar_bear_vid_id`, with an
            ``image_id`` column.

    Returns:
        One row per sequence, sorted by ``sequence``. Raises if a clip id maps
        to more than one sequence, which would make the pair file ambiguous.
    """
    if "image_id" not in catalogue or "sequence" not in catalogue:
        raise VideoRetrievalError("Catalogue needs 'image_id' and 'sequence' columns")
    first = catalogue.groupby("sequence", sort=True)["image_id"].first().reset_index()
    first["clip_id"] = [clip_id_from_image_id(value) for value in first["image_id"]]
    duplicated = first["clip_id"].duplicated()
    if duplicated.any():
        examples = first.loc[duplicated, "clip_id"].head(3).tolist()
        raise VideoRetrievalError(
            f"Clip ids are not unique per sequence, e.g. {examples}"
        )
    return first[["sequence", "clip_id"]].reset_index(drop=True)


def resolve_pair_task(
    task: pd.DataFrame, clip_ids: dict[str, str] | pd.DataFrame
) -> pd.DataFrame:
    """Translate pair-file clip ids into the project's sequence keys.

    The RoVF files identify clips as ``<folder>_<tracklet>`` while this project
    keys sequences as ``<identity>|<video>``. Scoring must happen on the latter
    because that is what the embeddings are indexed by.

    Args:
        task: Output of :func:`load_pair_task`.
        clip_ids: ``{clip_id: sequence}``, or the frame returned by
            :func:`sequence_clip_ids`.

    Returns:
        A copy of ``task`` with ``candidates``, ``positive_a`` and
        ``positive_b`` expressed as sequence keys, plus ``clip_ids`` holding the
        original ids so results can still be traced back to the paper file.
    """
    if isinstance(clip_ids, pd.DataFrame):
        if not {"sequence", "clip_id"}.issubset(clip_ids.columns):
            raise VideoRetrievalError("clip_ids frame needs 'sequence' and 'clip_id'")
        mapping = dict(zip(clip_ids["clip_id"], clip_ids["sequence"]))
    else:
        mapping = dict(clip_ids)

    resolved = task.copy()
    used: list[tuple[str, ...]] = []
    candidates: list[tuple[str, ...]] = []
    for original in task["candidates"]:
        missing = [clip for clip in original if clip not in mapping]
        if missing:
            raise VideoRetrievalError(
                f"{len(missing)} clips have no sequence, e.g. {missing[:3]}"
            )
        used.append(tuple(str(clip) for clip in original))
        candidates.append(tuple(mapping[clip] for clip in original))

    resolved["clip_ids"] = used
    resolved["candidates"] = candidates
    resolved["positive_a"] = [values[0] for values in candidates]
    resolved["positive_b"] = [values[1] for values in candidates]
    return resolved


def clip_identity(clip: str) -> str:
    """Return the identity part of a clip id, in either naming scheme."""
    text = str(clip)
    if "|" in text:  # project sequence key: "Charlotte|91"
        return text.split("|", 1)[0]
    if "_" in text:  # official pair file: "009_109"
        return text.split("_", 1)[0]
    raise VideoRetrievalError(f"Cannot read an identity from clip id {clip!r}")


def validate_pair_task(task: pd.DataFrame, available: set[str]) -> None:
    """Check that the pair file is well formed and fully resolvable.

    A clip may legitimately appear in several rows, and may be the first member
    of one pair and the second member of another: the reference file draws from
    a fixed pool of test clips. What must hold is that the two members of a row
    are different clips of the same identity, and that every referenced clip has
    an embedding.
    """
    if task.empty:
        raise VideoRetrievalError("Pair task is empty")

    self_pairs = task.loc[task["positive_a"] == task["positive_b"], "pair_id"].tolist()
    if self_pairs:
        raise VideoRetrievalError(
            f"Pairs {self_pairs[:3]} use the same clip as both positives"
        )

    mismatched = [
        (int(row.pair_id), row.positive_a, row.positive_b)
        for row in task.itertuples()
        if clip_identity(row.positive_a) != clip_identity(row.positive_b)
    ]
    if mismatched:
        pair_id, first, second = mismatched[0]
        raise VideoRetrievalError(
            f"Pair {pair_id} couples clips of different identities: {first!r} vs {second!r}"
        )

    referenced = {candidate for candidates in task["candidates"] for candidate in candidates}
    missing = sorted(referenced - available)
    if missing:
        raise VideoRetrievalError(
            f"{len(missing)} clips of the pair file have no embedding, e.g. {missing[:3]}"
        )

    identities = {clip_identity(candidate) for candidate in referenced}
    if len(identities) < 2:
        raise VideoRetrievalError("Pair task references fewer than two identities")


def official_ranking(distances: np.ndarray, banned_index: int) -> list[int]:
    """Replicate ``indices_of_smallest`` from the reference ``get_metrics.py``.

    The reference does not rank by index, it collects the three smallest
    *distinct distance values* and drops the third when it is tied with the
    fourth. With continuous embeddings ties are practically impossible, but the
    rule is copied exactly so the metric is the published one rather than a
    lookalike. Callers should treat the returned list as the official top-3.

    Args:
        distances: 1D distances from one query to every candidate.
        banned_index: Index of the query itself, excluded from the ranking.
    """
    distances = np.asarray(distances, dtype="float64")
    if distances.ndim != 1:
        raise VideoRetrievalError("Distance vector must be one-dimensional")
    if not np.isfinite(distances).all():
        raise VideoRetrievalError("Distances contain non-finite values")

    order = np.argsort(distances)
    order = order[order != banned_index]
    ranked = distances[order]
    if len(ranked) < 2:  # the reference loop never runs; keep it scoreable
        return [int(index) for index in order[:3]]

    values: list[float] = []
    indices: list[int] = []
    for position in range(len(ranked) - 1):
        if ranked[position] != ranked[position + 1]:
            values.append(float(ranked[position]))
            indices.append(int(order[position]))
        if len(values) == 3:
            break

    if len(values) == 3 and ranked[position + 1] == ranked[position]:
        last = values[-1]
        return [index for index, value in zip(indices, values) if value != last]
    return indices


def rank_pair_task(
    embeddings: dict[str, np.ndarray] | pd.DataFrame,
    task: pd.DataFrame,
) -> PairRetrievalResult:
    """Score the official pair protocol.

    Args:
        embeddings: Either ``{clip_id: vector}`` or a frame with ``sequence``
            and an ``embedding`` column.
        task: Output of :func:`load_pair_task`.

    Returns:
        Top-1 / Top-3 accuracy over both query directions of every pair.
    """
    lookup = _embedding_lookup(embeddings)
    validate_pair_task(task, set(lookup))

    trials = []
    for _, row in task.iterrows():
        candidates = list(row["candidates"])
        matrix = np.stack([lookup[candidate] for candidate in candidates])
        distances = euclidean_distance(matrix, matrix)

        for query_index in (0, 1):
            partner_index = 1 - query_index
            # Official rules: the nearest candidate must be a pair member, and
            # a pair member must appear in the distinct-distance top three.
            official = official_ranking(distances[query_index], query_index)
            top1 = bool(official) and official[0] in (0, 1)
            top3 = 0 in official[:3] or 1 in official[:3]

            order = np.argsort(distances[query_index], kind="stable")
            diagnostic = [index for index in order if index != query_index]
            trials.append(
                {
                    "pair_id": int(row["pair_id"]),
                    "query": candidates[query_index],
                    "positive": candidates[partner_index],
                    "n_gallery": len(candidates),
                    "top1": top1,
                    "top3": bool(top3),
                    "rank_of_positive": int(diagnostic.index(partner_index) + 1),
                    "official_top3": tuple(official),
                }
            )

    per_trial = pd.DataFrame(trials)
    if per_trial.empty:
        raise VideoRetrievalError("Pair task produced no trials")
    return PairRetrievalResult(
        top1=float(per_trial["top1"].mean()),
        top3=float(per_trial["top3"].mean()),
        n_pairs=int(task.shape[0]),
        n_trials=int(len(per_trial)),
        n_gallery=int(per_trial["n_gallery"].iloc[0]),
        per_trial=per_trial,
    )


def _embedding_lookup(
    embeddings: dict[str, np.ndarray] | pd.DataFrame,
) -> dict[str, np.ndarray]:
    if isinstance(embeddings, pd.DataFrame):
        if not {"sequence", "embedding"}.issubset(embeddings.columns):
            raise VideoRetrievalError(
                "Embedding frame needs 'sequence' and 'embedding' columns"
            )
        return {
            str(row["sequence"]): np.asarray(row["embedding"], dtype="float64")
            for _, row in embeddings.iterrows()
        }
    return {str(key): np.asarray(value, dtype="float64") for key, value in embeddings.items()}
