"""Frame sampling and temporal pooling for video sequences.

PolarBearVidID sequences are tracklets of consecutive frames. Two sampling
steps are involved, and they are easy to confuse:

1. **Clip construction.** The RoVF reference implementation
   (``Dataset/convert_polar_bear_dataset.py``) resizes every frame to 224x224
   and keeps 20 evenly spaced frames per tracklet::

       interval = np.linspace(0, total_frames - 1, 20, dtype=int)

2. **Evaluation subsampling.** The reference dataloader then keeps ``n`` of
   those 20 stored frames for the model grid search::

       interval = np.linspace(0, self.total_frames - 1, self.num_frames, dtype=int)

   with ``total_frames = 20``. This module composes the two steps so a
   5-frame or 10-frame variant is reproducible without re-reading images.

``np.linspace(..., dtype=int)`` truncates towards zero. It also *repeats*
indices when the sequence has fewer frames than requested, so a short tracklet
is padded by reusing frames. That is the reference behaviour and is preserved
here rather than silently "fixed"; :func:`sample_frame_indices` documents it.
Sequences keep their original frame order at every step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FRAMES_PER_SEQUENCE = 20
SEQUENCE_COLUMNS = ("identity", "video")


class FrameSamplingError(ValueError):
    """Raised when frame sampling cannot be defined for the given sequences."""


def sample_frame_indices(n_frames: int, count: int = FRAMES_PER_SEQUENCE) -> np.ndarray:
    """Return ``count`` evenly spaced positions in ``[0, n_frames - 1]``.

    This is exactly ``np.linspace(0, n_frames - 1, count, dtype=int)``, the rule
    used by the official RoVF conversion and dataloader code. When
    ``n_frames < count`` some positions repeat, which pads short sequences by
    reusing frames. The first and last frames are always included when
    ``count > 1``.
    """
    if n_frames < 1:
        raise FrameSamplingError("A sequence must contain at least one frame")
    if count < 1:
        raise FrameSamplingError("count must be at least one")
    return np.linspace(0, n_frames - 1, count, dtype=int)


def nested_frame_indices(
    n_frames: int,
    stored: int = FRAMES_PER_SEQUENCE,
    selected: int = FRAMES_PER_SEQUENCE,
) -> np.ndarray:
    """Compose clip construction and evaluation subsampling.

    Args:
        n_frames: Frames in the original tracklet.
        stored: Frames kept when the clip is built (20 in the release).
        selected: Frames fed to the model (5, 10 or 20 in the paper grid).

    Returns:
        Positions into the *original* tracklet, in increasing order, length
        ``selected``.
    """
    if selected > stored:
        raise FrameSamplingError(
            f"Cannot select {selected} frames from a clip storing only {stored}"
        )
    clip = sample_frame_indices(n_frames, stored)
    keep = sample_frame_indices(stored, selected)
    return clip[keep]


def sampled_frames(
    catalogue: pd.DataFrame, count: int = FRAMES_PER_SEQUENCE
) -> pd.DataFrame:
    """Return the sampled frames of every sequence, in clip order.

    Args:
        catalogue: Output of
            :func:`animal_re_id.data.splits.add_sequence_columns`; it must be
            sorted by ``(identity, video, frame)`` and carry ``sequence``.
        count: Frames stored per sequence.

    Returns:
        One row per sampled position with the source columns plus
        ``position`` (index inside the tracklet), ``sample_order`` (0-based
        position inside the clip) and ``sample_index`` (== ``position``, kept
        for readability in logs and failure reports).
    """
    if "sequence" not in catalogue or "frame" not in catalogue:
        raise FrameSamplingError(
            "Catalogue must be prepared with add_sequence_columns before sampling"
        )
    if count < 1:
        raise FrameSamplingError("count must be at least one")

    pieces = []
    for (identity, video), group in catalogue.groupby(list(SEQUENCE_COLUMNS), sort=True):
        frames = group.sort_values("frame").reset_index(drop=True)
        positions = sample_frame_indices(len(frames), count)
        selected = frames.iloc[positions].copy()
        selected["position"] = positions
        selected["sample_index"] = positions
        selected["sample_order"] = np.arange(len(positions))
        selected["identity"] = identity
        selected["video"] = video
        selected["sequence"] = f"{identity}|{video}"
        pieces.append(selected)

    if not pieces:
        raise FrameSamplingError("Catalogue contains no sequences")

    sampled = pd.concat(pieces, ignore_index=True)
    summary = sampled.groupby("sequence")["sample_order"].agg(["size", "nunique"])
    if (summary["size"] != count).any() or (summary["nunique"] != count).any():
        raise FrameSamplingError(
            f"Every sequence must contribute exactly {count} ordered samples"
        )
    return sampled.sort_values(["identity", "video", "sample_order"]).reset_index(drop=True)


def temporal_mean(frame_features: np.ndarray) -> np.ndarray:
    """Average frame embeddings in their given order.

    Pooling is order independent by definition; the order is preserved upstream
    so that the input to this function can be inspected and logged.

    Args:
        frame_features: ``(n_frames, dim)`` array in clip order.

    Returns:
        ``(dim,)`` float32 sequence embedding.
    """
    features = np.asarray(frame_features, dtype="float64")
    if features.ndim != 2:
        raise FrameSamplingError(
            f"Expected a (frames, dim) matrix, got shape {features.shape}"
        )
    if features.shape[0] == 0:
        raise FrameSamplingError("Cannot pool zero frames")
    if not np.isfinite(features).all():
        raise FrameSamplingError("Frame features contain non-finite values")
    return features.mean(axis=0).astype("float32")


def pool_sequence(
    frame_features: np.ndarray,
    n_source_frames: int,
    stored: int = FRAMES_PER_SEQUENCE,
    selected: int = FRAMES_PER_SEQUENCE,
) -> np.ndarray:
    """Select ``selected`` of the stored frames and average them.

    ``frame_features`` must be ``(stored, dim)`` in clip order, i.e. the sample
    order produced by :func:`sampled_frames`. Selection reuses
    :func:`nested_frame_indices`, so the result matches the official code for a
    tracklet of ``n_source_frames`` frames.
    """
    features = np.asarray(frame_features)
    if features.shape[0] != stored:
        raise FrameSamplingError(
            f"Expected {stored} stored frames, got {features.shape[0]}"
        )
    positions = nested_frame_indices(n_source_frames, stored=stored, selected=selected)
    clip_positions = sample_frame_indices(n_source_frames, stored)
    lookup = {int(position): index for index, position in enumerate(clip_positions)}
    try:
        indices = [lookup[int(position)] for position in positions]
    except KeyError as error:  # pragma: no cover - construction guarantees membership
        raise FrameSamplingError(f"Selected frame {error} is not in the clip") from error
    return temporal_mean(features[indices])


def sequence_embeddings(
    frame_features: np.ndarray,
    sampled: pd.DataFrame,
    n_source_frames: dict[str, int],
    stored: int = FRAMES_PER_SEQUENCE,
    selected: int = FRAMES_PER_SEQUENCE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Collapse per-frame features into one embedding per sequence.

    Args:
        frame_features: ``(len(sampled), dim)`` row-aligned with ``sampled``.
        sampled: Output of :func:`sampled_frames`.
        n_source_frames: ``sequence -> frames in the original tracklet``.
        stored: Frames per clip (20).
        selected: Frames averaged per sequence.

    Returns:
        ``(sequence metadata, embeddings)`` with rows in sorted sequence order.
        The metadata keeps ``identity``, ``video``, ``sequence`` and
        ``n_frames`` so it can be fed to the closed-set evaluator unchanged.
    """
    frame_features = np.asarray(frame_features)
    if len(frame_features) != len(sampled):
        raise FrameSamplingError(
            f"{len(frame_features)} feature rows for {len(sampled)} sampled frames"
        )

    sampled = sampled.reset_index(drop=True)
    rows = []
    embeddings = []
    for sequence, group in sampled.groupby("sequence", sort=True):
        order = group.sort_values("sample_order")
        positions = order.index.to_numpy()
        features = frame_features[positions]
        source = int(n_source_frames[sequence])
        embeddings.append(
            pool_sequence(features, source, stored=stored, selected=selected)
        )
        rows.append(
            {
                "identity": str(order["identity"].iloc[0]),
                "video": int(order["video"].iloc[0]),
                "sequence": str(sequence),
                "n_frames": source,
                "stored_frames": int(len(order)),
            }
        )

    metadata = pd.DataFrame(rows).sort_values("sequence").reset_index(drop=True)
    matrix = np.stack(embeddings).astype("float32")
    if matrix.shape[0] != len(metadata):
        raise FrameSamplingError("Sequence metadata and embeddings disagree on row count")
    return metadata, matrix
