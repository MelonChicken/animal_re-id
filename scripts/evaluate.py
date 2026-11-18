"""Run the frozen MegaDescriptor baseline on PolarBearVidID.

    uv run python scripts/evaluate.py --config config/polarbear.yaml

The script is a straight line: load configuration, load the dataset catalogue,
build (or load) the sequence split, embed one frame per sequence with a frozen
MegaDescriptor, score a closed-set query/gallery retrieval task, and write the
result together with everything needed to reproduce it.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from animal_re_id.data.splits import (
    GALLERY_SCOPES,
    PROTOCOLS,
    add_sequence_columns,
    build_retrieval_task,
    representative_frames,
    sequence_table,
    split_signature,
    validate_retrieval_task,
)
from animal_re_id.data.wildlife import load_polar_bear_vid_id
from animal_re_id.evaluation.retrieval import cosine_similarity, rank_retrieval
from animal_re_id.models.megadescriptor import (
    DEFAULT_MODEL_NAME,
    build_transform,
    describe_model,
    extract_features,
    load_model,
)
from animal_re_id.utils.runtime import (
    build_or_load_split,
    display as _display,
    environment_record,
    evaluation_units,
    feature_cache_key,
    load_cached_features,
    resolve as _resolve,
    save_cached_features,
    split_path_for,
)


# --------------------------------------------------------------------------- #
# Configuration and environment
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("dataset", "protocol", "model", "paths"):
        if key not in config:
            raise ValueError(f"Configuration {path} is missing the {key!r} section")
    if config["protocol"]["name"] not in PROTOCOLS:
        raise ValueError(
            f"Unknown protocol {config['protocol']['name']!r}; expected one of {PROTOCOLS}"
        )
    scope = config.get("evaluation", {}).get("gallery_scope", "fold")
    if scope not in GALLERY_SCOPES:
        raise ValueError(f"Unknown gallery_scope {scope!r}; expected one of {GALLERY_SCOPES}")
    return config


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def embed_all_frames(config: dict, frames: pd.DataFrame) -> tuple[np.ndarray, dict, dict]:
    """Load MegaDescriptor, reuse or compute the embedding matrix, return it."""
    frame_selection = config["dataset"].get("frame_selection", "middle")
    device = str(config["model"].get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configuration asks for CUDA but torch.cuda.is_available() is False")

    model_name = str(config["model"].get("name", DEFAULT_MODEL_NAME))
    model = load_model(model_name, device=device)
    transform = build_transform(model)
    description = describe_model(model_name, model)

    key = feature_cache_key(frames, description, frame_selection)
    cache_path = _resolve(config["paths"]["cache_dir"]) / f"features_{key[:16]}.npz"
    features = load_cached_features(cache_path, key, len(frames))
    extraction = {"cached": features is not None, "seconds": 0.0}
    if features is None:
        started = time.time()
        features = extract_features(
            model,
            transform,
            frames,
            batch_size=int(config["model"].get("batch_size", 32)),
            device=device,
            num_workers=int(config["model"].get("num_workers", 0)),
        )
        extraction["seconds"] = round(time.time() - started, 2)
        save_cached_features(cache_path, key, features, frames)

    record = {
        "name": model_name,
        "description": description,
        "transform": transform,
        "device": device,
        "dimension": int(features.shape[1]),
        "n_images": int(features.shape[0]),
        "cache_path": _display(cache_path),
        "cache_key": key,
        **extraction,
    }
    return features, record, description


def run(config: dict) -> dict:
    protocol = config["protocol"]
    protocol_name = protocol["name"]
    seed = int(protocol.get("seed", 0))
    n_folds = int(protocol.get("n_folds", 5))
    gallery_scope = config.get("evaluation", {}).get("gallery_scope", "fold")
    dataset_root = _resolve(config["dataset"]["root"])
    paths = config["paths"]
    split_dir = _resolve(paths["split_dir"])
    result_dir = _resolve(paths["result_dir"])
    experiment = str(config.get("experiment", f"megadescriptor_{protocol_name}"))

    started = time.time()
    catalogue = add_sequence_columns(load_polar_bear_vid_id(dataset_root))
    frames = representative_frames(catalogue)

    split_path = split_path_for(split_dir, protocol_name, seed)
    assignment, split_payload = build_or_load_split(catalogue, config, split_path, dataset_root)

    features, feature_record, model_description = embed_all_frames(config, frames)

    position = {sequence: index for index, sequence in enumerate(frames["sequence"])}
    unit_metrics = []
    per_query_frames = []
    for unit in evaluation_units(assignment, protocol_name):
        query, gallery = build_retrieval_task(
            frames, assignment, protocol_name, fold=unit, gallery_scope=gallery_scope
        )
        query_index = [position[sequence] for sequence in query["sequence"]]
        gallery_index = [position[sequence] for sequence in gallery["sequence"]]
        similarity = cosine_similarity(features[query_index], features[gallery_index])
        result = rank_retrieval(similarity, query, gallery)

        metrics = result.to_dict()
        metrics["unit"] = unit
        metrics["n_queries"] = len(query)
        metrics["n_gallery"] = len(gallery)
        unit_metrics.append(metrics)

        per_query = result.per_query.copy()
        per_query["unit"] = unit
        per_query["gallery_scope"] = gallery_scope
        per_query_frames.append(per_query)

    aggregate = {}
    for metric in ("rank1", "rank3", "mAP"):
        values = np.array([unit[metric] for unit in unit_metrics], dtype="float64")
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }

    result = {
        "experiment": experiment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "environment": environment_record(str(config["model"].get("device", "cpu"))),
        "dataset": {
            "name": config["dataset"]["name"],
            "root": _display(dataset_root),
            "frames": int(len(catalogue)),
            "sequences": int(sequence_table(catalogue).shape[0]),
            "identities": int(catalogue["identity"].nunique()),
            "frame_selection": config["dataset"].get("frame_selection", "middle"),
            "images_embedded": int(len(frames)),
        },
        "split": {
            "protocol": protocol_name,
            "seed": seed,
            "n_folds": n_folds,
            "path": _display(split_path),
            "signature": split_signature(assignment),
            "counts": split_payload.get("counts"),
            "metadata": split_payload.get("metadata"),
        },
        "model": feature_record,
        "evaluation": {
            "unit": "image",
            "level": "sequence-aware image retrieval (one middle frame per sequence)",
            "query": "one representative frame per held-out sequence",
            "gallery": (
                "all sequences, self-match excluded"
                if gallery_scope == "all"
                else "held-out sequences only, self-match excluded"
            ),
            "gallery_scope": gallery_scope,
            "distance": "cosine similarity",
            "metrics": ["rank1", "rank3", "mAP"],
            "closed_set": True,
        },
        "results": {"per_unit": unit_metrics, "aggregate": aggregate},
        "runtime_seconds": round(time.time() - started, 2),
    }

    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{experiment}_{gallery_scope}gallery"
    result_path = result_dir / f"{stem}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.concat(per_query_frames, ignore_index=True).to_csv(
        result_dir / f"{stem}_per_query.csv", index=False
    )
    result["result_path"] = _display(result_path)
    return result


def split_only(config: dict) -> dict:
    dataset_root = _resolve(config["dataset"]["root"])
    catalogue = add_sequence_columns(load_polar_bear_vid_id(dataset_root))
    frames = representative_frames(catalogue)
    protocol_name = config["protocol"]["name"]
    seed = int(config["protocol"].get("seed", 0))
    split_path = split_path_for(_resolve(config["paths"]["split_dir"]), protocol_name, seed)
    assignment, payload = build_or_load_split(catalogue, config, split_path, dataset_root)

    for unit in evaluation_units(assignment, protocol_name):
        query, gallery = build_retrieval_task(frames, assignment, protocol_name, fold=unit)
        validate_retrieval_task(query, gallery, protocol=protocol_name)
        print(
            f"unit={unit}: {len(query)} queries, {len(gallery)} gallery, "
            f"{query['identity'].nunique()} identities"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/polarbear.yaml", type=Path)
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Build and validate the split without loading a model.",
    )
    parser.add_argument(
        "--gallery-scope",
        choices=GALLERY_SCOPES,
        default=None,
        help="Override evaluation.gallery_scope from the config.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Override the experiment name used for the result file.",
    )
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default=None,
        help="Override protocol.name from the config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override protocol.seed (only used by five_fold_stratified).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.gallery_scope:
        config.setdefault("evaluation", {})["gallery_scope"] = args.gallery_scope
    if args.experiment:
        config["experiment"] = args.experiment
    if args.protocol:
        config["protocol"]["name"] = args.protocol
    if args.seed is not None:
        config["protocol"]["seed"] = args.seed
    if args.split_only:
        payload = split_only(config)
        print(json.dumps(payload["counts"], indent=2))
        return 0

    result = run(config)
    summary = {
        metric: {"mean": values["mean"], "std": values["std"]}
        for metric, values in result["results"]["aggregate"].items()
    }
    print(json.dumps({"results": summary, "per_unit": result["results"]["per_unit"]}, indent=2))
    print(f"Wrote {result['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
