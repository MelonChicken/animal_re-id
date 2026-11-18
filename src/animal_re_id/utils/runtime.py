"""Reproducibility helpers shared by the evaluation scripts.

Nothing in this module is model specific. It centralises the pieces that were
previously private to ``scripts/evaluate.py`` (environment record, dataset
signature, feature cache, split cache) so that a second baseline can reuse them
instead of copy-pasting them.

The on-disk formats are unchanged: a feature cache is still a compressed
``.npz`` holding ``key``, ``features``, ``sequences`` and ``paths``, and a split
is still the JSON written by :func:`animal_re_id.data.splits.save_split`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from animal_re_id.data.splits import (
    PROTOCOL_FIVE_FOLD,
    PROTOCOL_FIVE_FOLD_STRATIFIED,
    PROTOCOL_ROVF,
    SplitError,
    check_assignments,
    five_fold_assignment,
    load_split,
    rovf_assignment,
    save_split,
    stratified_five_fold_assignment,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FOLD_PROTOCOLS = (PROTOCOL_FIVE_FOLD, PROTOCOL_FIVE_FOLD_STRATIFIED)
DEFAULT_ENVIRONMENT_PACKAGES = (
    "torch",
    "torchvision",
    "timm",
    "transformers",
    "wildlife-datasets",
    "wildlife-tools",
    "numpy",
    "pandas",
)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def resolve(path: str | Path) -> Path:
    """Resolve a configuration path relative to the repository root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def display(path: Path) -> str:
    """Render a path relative to the repository root when possible."""
    path = Path(path)
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


# --------------------------------------------------------------------------- #
# Environment record
# --------------------------------------------------------------------------- #
def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record(
    device: str, packages: tuple[str, ...] = DEFAULT_ENVIRONMENT_PACKAGES
) -> dict:
    """Describe the interpreter, the checkout and the accelerator in use."""
    record = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "packages": {name: package_version(name) for name in packages},
        "device": device,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
        record["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    if torch.backends.cudnn.version() is not None:
        record["cudnn"] = torch.backends.cudnn.version()
    return record


# --------------------------------------------------------------------------- #
# Dataset and cache keys
# --------------------------------------------------------------------------- #
def dataset_signature(catalogue: pd.DataFrame, dataset_root: Path) -> str:
    """Hash the frame catalogue so a cache cannot be reused across datasets."""
    payload = {
        "frames": int(len(catalogue)),
        "sequences": int(catalogue["sequence"].nunique()),
        "identities": sorted(catalogue["identity"].unique().tolist()),
        "paths_hash": hashlib.sha256(
            "\n".join(sorted(catalogue["path"])).encode("utf-8")
        ).hexdigest(),
        "root": Path(dataset_root).name,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def feature_cache_key(frames: pd.DataFrame, model: dict, frame_selection: str) -> str:
    """Hash everything that determines the embedding matrix.

    The split is deliberately *not* part of this key: the embeddings are a pure
    function of the model, the preprocessing and the list of images. Changing the
    split must not force a recomputation, and reusing embeddings cannot leak a
    split, because metrics are always recomputed from the current assignment and
    the split signature is recorded in the result file.
    """
    payload = {
        "model": model,
        "frame_selection": frame_selection,
        "sequences": frames["sequence"].tolist(),
        "paths": frames["path"].tolist(),
        "image_ids": frames["image_id"].tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_cached_features(path: Path, key: str, expected_rows: int) -> np.ndarray | None:
    """Return cached features if they match ``key`` and the expected row count.

    A cache file that cannot be read (truncated by an interrupted save, wrong
    format) is treated as a miss rather than an error: recomputing features is
    slow but always correct, and a poisoned cache should not block future runs.
    """
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) != key:
                return None
            features = np.asarray(data["features"], dtype="float32")
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None
    return features if len(features) == expected_rows else None


def save_cached_features(
    path: Path, key: str, features: np.ndarray, frames: pd.DataFrame
) -> None:
    """Write the cache atomically so an interrupted run leaves no half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            key=np.array(key),
            features=features,
            # Plain unicode arrays, not pickled object arrays: the file stays
            # readable with ``allow_pickle=False``.
            sequences=np.asarray(frames["sequence"], dtype=str),
            paths=np.asarray(frames["path"], dtype=str),
        )
    os.replace(temporary, path)


# --------------------------------------------------------------------------- #
# Split construction
# --------------------------------------------------------------------------- #
def split_path_for(split_dir: Path, protocol: str, seed: int) -> Path:
    return Path(split_dir) / f"polarbear_{protocol}_seed{seed}.json"


def build_or_load_split(
    catalogue: pd.DataFrame,
    config: dict,
    split_path: Path,
    dataset_root: Path,
) -> tuple[pd.DataFrame, dict]:
    """Return the split described by ``config``, reusing an on-disk copy if valid."""
    protocol = config["protocol"]
    name = protocol["name"]
    seed = int(protocol.get("seed", 0))
    n_folds = int(protocol.get("n_folds", 5))
    fingerprint = dataset_signature(catalogue, dataset_root)

    if split_path.exists():
        assignment, payload = load_split(split_path)
        metadata = payload.get("metadata", {})
        mismatches = [
            label
            for label, existing, wanted in (
                ("protocol", payload.get("protocol"), name),
                ("dataset", metadata.get("dataset_signature"), fingerprint),
                ("seed", metadata.get("seed"), seed),
                ("n_folds", metadata.get("n_folds"), n_folds),
            )
            if existing != wanted
        ]
        if mismatches:
            raise ValueError(
                f"Existing split {split_path} does not match the configuration "
                f"({', '.join(mismatches)}). Delete it to rebuild."
            )
        return assignment, payload

    if name == PROTOCOL_FIVE_FOLD:
        assignment = five_fold_assignment(catalogue, dataset_root, n_folds=n_folds)
    elif name == PROTOCOL_FIVE_FOLD_STRATIFIED:
        assignment = stratified_five_fold_assignment(catalogue, seed=seed, n_folds=n_folds)
    elif name == PROTOCOL_ROVF:
        assignment = rovf_assignment(
            catalogue,
            dataset_root,
            train_ratio=float(protocol.get("train_ratio", 0.7)),
            val_ratio=float(protocol.get("val_ratio", 0.8)),
        )
    else:  # pragma: no cover - guarded by the configuration loader
        raise SplitError(f"Unsupported protocol {name!r}")

    check_assignments(assignment)
    save_split(
        assignment,
        split_path.parent,
        split_path.stem,
        metadata={
            "dataset": config["dataset"]["name"],
            "dataset_signature": fingerprint,
            "seed": seed,
            "n_folds": n_folds,
            "frame_selection": config["dataset"].get("frame_selection"),
            "source": "official release" if name == PROTOCOL_FIVE_FOLD else "derived",
        },
    )
    _, payload = load_split(split_path)
    return assignment, payload


def evaluation_units(assignment: pd.DataFrame, protocol: str) -> list[int | None]:
    """Return the held-out units of a protocol: folds, or a single RoVF test set."""
    if protocol in FOLD_PROTOCOLS:
        return sorted(int(fold) for fold in assignment["fold"].unique())
    return [None]
