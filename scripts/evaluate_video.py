"""Run the frozen DINOv2 video baseline on PolarBearVidID.

    uv run python scripts/evaluate_video.py --config config/dinov2_video.yaml

Pipeline: load the catalogue, sample 20 frames per sequence with the official
RoVF rule, embed every sampled frame once with a frozen DINOv2-S (patch-token
spatial mean, plus the alternative reductions used by the reference code),
average the selected frames per sequence, then score two video protocols:

``official_pair``
    The RoVF paper protocol: 100 rows of one positive pair and nine negatives,
    both pair members used as a query in turn (200 trials), Euclidean distance,
    Top-1 / Top-3.
``closed_set``
    Sequence-level leave-one-out retrieval on the configured split, which also
    defines mAP.

Every run writes ``experiments/polarbear/<experiment>/<run_id>/`` with
``config.yaml``, ``metadata.json``, ``metrics.json``, ``run.log`` and
``failures.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from animal_re_id.data.splits import (
    GALLERY_SCOPES,
    PROTOCOLS,
    SplitError,
    add_sequence_columns,
    build_retrieval_task,
    sequence_table,
    split_signature,
)
from animal_re_id.data.video import (
    FRAMES_PER_SEQUENCE,
    sampled_frames,
    sequence_embeddings,
)
from animal_re_id.data.wildlife import load_polar_bear_vid_id
from animal_re_id.evaluation.retrieval import rank_retrieval
from animal_re_id.evaluation.video_retrieval import (
    VideoRetrievalError,
    euclidean_distance,
    load_pair_task,
    rank_pair_task,
    resolve_pair_task,
    sequence_clip_ids,
    validate_pair_task,
)
from animal_re_id.models.dinov2 import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PREPROCESSING,
    DEFAULT_REDUCTION,
    PREPROCESSING_MODES,
    SPATIAL_REDUCTIONS,
    build_transform,
    describe_model,
    extract_frame_features,
    load_model,
)
from animal_re_id.utils.logging import ExperimentRun
from animal_re_id.utils.runtime import (
    build_or_load_split,
    display,
    environment_record,
    evaluation_units,
    feature_cache_key,
    load_cached_features,
    resolve,
    save_cached_features,
    split_path_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = "experiments/polarbear"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("dataset", "protocol", "model", "paths", "evaluation"):
        if key not in config:
            raise ValueError(f"Configuration {path} is missing the {key!r} section")
    if config["protocol"]["name"] not in PROTOCOLS:
        raise ValueError(
            f"Unknown protocol {config['protocol']['name']!r}; expected one of {PROTOCOLS}"
        )
    scope = config["evaluation"].get("gallery_scope", "fold")
    if scope not in GALLERY_SCOPES:
        raise ValueError(f"Unknown gallery_scope {scope!r}; expected one of {GALLERY_SCOPES}")

    model = config["model"]
    reductions = tuple(model.get("spatial_reductions", [DEFAULT_REDUCTION]))
    unknown = [reduction for reduction in reductions if reduction not in SPATIAL_REDUCTIONS]
    if unknown or not reductions:
        raise ValueError(
            f"Unknown spatial_reductions {unknown}; expected a subset of {SPATIAL_REDUCTIONS}"
        )
    preprocessing = str(model.get("preprocessing", DEFAULT_PREPROCESSING))
    if preprocessing not in PREPROCESSING_MODES:
        raise ValueError(
            f"Unknown preprocessing {preprocessing!r}; expected one of {PREPROCESSING_MODES}"
        )
    return config


def primary_configuration(config: dict) -> dict:
    primary = config["evaluation"].get("primary", {})
    reduction = str(primary.get("spatial_reduction", DEFAULT_REDUCTION))
    frames = int(primary.get("frame_count", config["dataset"].get("frames_per_sequence", FRAMES_PER_SEQUENCE)))
    if reduction not in tuple(config["model"].get("spatial_reductions", (reduction,))):
        raise ValueError(f"primary.spatial_reduction {reduction!r} is not in model.spatial_reductions")
    if frames not in [int(value) for value in config["evaluation"].get("frame_counts", [frames])]:
        raise ValueError(f"primary.frame_count {frames} is not in evaluation.frame_counts")
    return {"spatial_reduction": reduction, "frame_count": frames}


# --------------------------------------------------------------------------- #
# Feature extraction and cache
# --------------------------------------------------------------------------- #
def embed_frames(config: dict, run: ExperimentRun, sampled: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Embed every sampled frame once, reusing the cache when the key matches."""
    model_config = config["model"]
    device = str(model_config.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configuration asks for CUDA but torch.cuda.is_available() is False")

    reductions = tuple(model_config.get("spatial_reductions", [DEFAULT_REDUCTION]))
    preprocessing = str(model_config.get("preprocessing", DEFAULT_PREPROCESSING))
    image_size = int(model_config.get("image_size", DEFAULT_IMAGE_SIZE))
    model_name = str(model_config.get("name", DEFAULT_MODEL_NAME))

    with run.stage("model_init"):
        model = load_model(model_name, device=device)
        description = describe_model(model_name, model)
        transform = build_transform(image_size=image_size, preprocessing=preprocessing)
    run.update_metadata(
        model={
            "name": model_name,
            "description": description,
            "preprocessing": preprocessing,
            "image_size": image_size,
            "spatial_reductions": list(reductions),
            "transform": repr(transform),
            "device": device,
        }
    )

    cache_payload = {
        **description,
        "preprocessing": preprocessing,
        "image_size": image_size,
        "reductions": list(reductions),
    }
    key = feature_cache_key(sampled, cache_payload, "rovf_linspace20")
    cache_dir = resolve(config["paths"]["cache_dir"])
    cache_path = cache_dir / f"dinov2_frames_{key[:16]}.npz"

    features = load_cached_features(cache_path, key, len(sampled))
    cache_hit = features is not None
    if cache_hit:
        run.info(f"Feature cache hit: {display(cache_path)}")
    else:
        run.info(
            f"Feature cache miss: embedding {len(sampled)} frames "
            f"({sampled['sequence'].nunique()} sequences) with {model_name}"
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        batch_size = int(model_config.get("batch_size", 32))
        chunk_size = int(model_config.get("progress_chunk", 5000))
        started = time.time()
        blocks = []
        with run.stage("feature_extraction"):
            for start in range(0, len(sampled), chunk_size):
                block = sampled.iloc[start : start + chunk_size]
                blocks.append(
                    extract_frame_features(
                        model,
                        transform,
                        block,
                        reductions=reductions,
                        batch_size=batch_size,
                        device=device,
                        num_workers=int(model_config.get("num_workers", 0)),
                    )
                )
                done = min(start + chunk_size, len(sampled))
                run.info(
                    f"Extraction progress: {done}/{len(sampled)} frames "
                    f"({done / len(sampled):.0%})"
                )
        features = np.concatenate(blocks, axis=0)
        extraction_seconds = round(time.time() - started, 2)
        run.info(f"Extracted {features.shape[0]} frames in {extraction_seconds}s")
        save_cached_features(cache_path, key, features, sampled)
        run.info(f"Feature cache written: {display(cache_path)}")
        cache_hit = False

    record = {
        "checkpoint": model_name,
        "description": description,
        "preprocessing": preprocessing,
        "image_size": image_size,
        "spatial_reductions": list(reductions),
        "dimension": int(features.shape[2]),
        "n_frames": int(features.shape[0]),
        "cache_key": key,
        "cache_path": display(cache_path),
        "cache_hit": cache_hit,
        "cache_bytes": int(cache_path.stat().st_size) if cache_path.exists() else None,
    }
    if torch.cuda.is_available():
        record["gpu_peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return features, record


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def per_identity_metrics(per_query: pd.DataFrame) -> list[dict]:
    rows = []
    for identity, group in per_query[per_query["scored"]].groupby("identity", sort=True):
        rows.append(
            {
                "identity": str(identity),
                "n_queries": int(len(group)),
                "rank1": float((group["first_positive_rank"] <= 1).mean()),
                "rank3": float((group["first_positive_rank"] <= 3).mean()),
                "mAP": float(group["average_precision"].mean()),
            }
        )
    return rows


def closed_set_task(
    embeddings: pd.DataFrame,
    assignment: pd.DataFrame,
    protocol: str,
    gallery_scope: str,
    unit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the sequence-level query/gallery task from the assignment.

    ``path`` is set to the sequence key so the existing leave-one-out scorer
    can identify the self-match without a second code path.
    """
    unit_frames = embeddings.assign(path=embeddings["sequence"])
    return build_retrieval_task(
        unit_frames, assignment, protocol, fold=unit, gallery_scope=gallery_scope
    )


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run(config: dict) -> dict:
    protocol = config["protocol"]
    protocol_name = protocol["name"]
    seed = int(protocol.get("seed", 0))
    n_folds = int(protocol.get("n_folds", 5))
    gallery_scope = config["evaluation"].get("gallery_scope", "fold")
    dataset_root = resolve(config["dataset"]["root"])
    stored = int(config["dataset"].get("frames_per_sequence", FRAMES_PER_SEQUENCE))
    frame_counts = [int(value) for value in config["evaluation"].get("frame_counts", [stored])]
    experiment = str(config["experiment"])
    primary = primary_configuration(config)

    started = time.time()
    run = ExperimentRun(
        experiment=experiment,
        config=config,
        root=resolve(EXPERIMENTS_ROOT),
        stream=bool(config["evaluation"].get("stream_log", True)),
    )
    with run:
        run.update_metadata(environment=environment_record(str(config["model"].get("device", "cpu"))))

        with run.stage("dataset_load"):
            catalogue = add_sequence_columns(load_polar_bear_vid_id(dataset_root))
        run.update_metadata(
            dataset={
                "name": config["dataset"]["name"],
                "root": display(dataset_root),
                "frames": int(len(catalogue)),
                "sequences": int(sequence_table(catalogue).shape[0]),
                "identities": int(catalogue["identity"].nunique()),
                "frames_per_sequence": stored,
                "frame_sampling": "np.linspace(0, n_frames - 1, 20, dtype=int)",
            }
        )
        run.info(
            f"Dataset loaded: {len(catalogue)} frames, "
            f"{catalogue['sequence'].nunique()} sequences, "
            f"{catalogue['identity'].nunique()} identities"
        )

        split_path = split_path_for(resolve(config["paths"]["split_dir"]), protocol_name, seed)
        with run.stage("split_build"):
            assignment, split_payload = build_or_load_split(
                catalogue, config, split_path, dataset_root
            )
        signature = split_signature(assignment)
        run.update_metadata(
            split={
                "protocol": protocol_name,
                "seed": seed,
                "n_folds": n_folds,
                "path": display(split_path),
                "signature": signature,
                "counts": split_payload.get("counts"),
                "metadata": split_payload.get("metadata"),
            }
        )
        run.info(f"Split {protocol_name} signature {signature[:16]}")

        with run.stage("frame_sampling"):
            sampled = sampled_frames(catalogue, count=stored)
        n_source = catalogue.groupby("sequence").size().to_dict()
        run.update_metadata(
            sampling={
                "stored_frames": stored,
                "frame_counts": frame_counts,
                "sampled_rows": int(len(sampled)),
                "short_sequences": int(
                    (pd.Series(n_source).reindex(sampled["sequence"].unique()) < stored).sum()
                ),
            }
        )

        features, feature_record = embed_frames(config, run, sampled)
        run.update_metadata(feature_cache=feature_record)

        pair_path = config["evaluation"].get("pair_file")
        pair_task = None
        if pair_path:
            try:
                clip_index = sequence_clip_ids(catalogue)
                pair_task = resolve_pair_task(
                    load_pair_task(resolve(pair_path)), clip_index
                )
                validate_pair_task(pair_task, set(sampled["sequence"].unique()))
                run.info(
                    f"Official pair task loaded: {len(pair_task)} pairs "
                    f"({2 * len(pair_task)} trials) from {pair_path}"
                )
                claimed = {clip for values in pair_task["clip_ids"] for clip in values}
                run.update_metadata(
                    pair_task={
                        "file": display(resolve(pair_path)),
                        "pairs": int(len(pair_task)),
                        "trials": int(2 * len(pair_task)),
                        "gallery_size": int(len(pair_task["candidates"].iloc[0])),
                        "clips": len(claimed),
                        "id_mapping": "image_id[:3] + '_' + tracklet -> sequence",
                    }
                )
            except (VideoRetrievalError, OSError) as error:
                run.warning(f"Official pair protocol unavailable: {error}")
                run.record_failure(cause=str(error), unit="official_pair", status="skipped")
                run.update_metadata(
                    protocol_comparison={
                        "official_pair": {"available": False, "reason": str(error)}
                    }
                )
                pair_task = None

        configurations = []
        per_query_frames = []
        run.info(
            f"evaluation: started ({len(feature_record['spatial_reductions'])} reductions "
            f"x {len(frame_counts)} frame counts)"
        )
        for reduction_index, reduction in enumerate(feature_record["spatial_reductions"]):
            for frame_count in frame_counts:
                with run.stage(f"pool_{reduction}_{frame_count}f"):
                    embeddings, matrix = sequence_embeddings(
                        features[:, reduction_index, :],
                        sampled,
                        n_source,
                        stored=stored,
                        selected=frame_count,
                    )
                embeddings = embeddings.assign(
                    embedding=list(map(np.asarray, matrix)),
                    spatial_reduction=reduction,
                    frame_count=frame_count,
                )
                run.info(
                    f"Sequence embeddings: reduction={reduction} frames={frame_count} "
                    f"shape={matrix.shape}"
                )

                if pair_task is not None:
                    try:
                        pair_result = rank_pair_task(
                            {
                                str(row.sequence): row.embedding
                                for row in embeddings.itertuples()
                            },
                            pair_task,
                        )
                        configurations.append(
                            {
                                "protocol": "official_pair",
                                "spatial_reduction": reduction,
                                "frame_count": frame_count,
                                **pair_result.to_dict(),
                            }
                        )
                        per_trial = pair_result.per_trial.merge(
                            pair_task[["pair_id", "clip_ids"]], on="pair_id", how="left"
                        ).assign(spatial_reduction=reduction, frame_count=frame_count)
                        per_query_frames.append(per_trial)
                        run.info(
                            f"official_pair {reduction} {frame_count}f: "
                            f"Top-1 {pair_result.top1:.3f} Top-3 {pair_result.top3:.3f} "
                            f"({pair_result.n_trials} trials)"
                        )
                    except (VideoRetrievalError, ValueError) as error:
                        run.record_failure(
                            cause=str(error),
                            unit=f"official_pair/{reduction}/{frame_count}f",
                        )

                try:
                    for unit in evaluation_units(assignment, protocol_name):
                        query, gallery = closed_set_task(
                            embeddings, assignment, protocol_name, gallery_scope, unit
                        )
                        distances = euclidean_distance(
                            np.stack(query["embedding"].to_numpy()),
                            np.stack(gallery["embedding"].to_numpy()),
                        )
                        result = rank_retrieval(-distances, query, gallery)
                        per_query = result.per_query.copy()
                        per_query["unit"] = unit
                        per_query["spatial_reduction"] = reduction
                        per_query["frame_count"] = frame_count
                        per_query["gallery_scope"] = gallery_scope
                        per_query_frames.append(per_query)
                        configurations.append(
                            {
                                "protocol": f"closed_set:{gallery_scope}",
                                "spatial_reduction": reduction,
                                "frame_count": frame_count,
                                "unit": unit,
                                "rank1": result.rank1,
                                "rank3": result.rank3,
                                "mean_average_precision": result.mean_average_precision,
                                "n_queries": result.n_queries,
                                "n_scored_queries": result.n_scored_queries,
                                "n_gallery": result.n_gallery,
                                "n_identities": result.n_identities,
                                "per_identity": per_identity_metrics(result.per_query),
                            }
                        )
                        run.info(
                            f"closed_set {gallery_scope} {reduction} {frame_count}f "
                            f"unit={unit}: Rank-1 {result.rank1:.3f} "
                            f"Rank-3 {result.rank3:.3f} "
                            f"mAP {result.mean_average_precision:.3f} "
                            f"({result.n_scored_queries}/{result.n_queries} queries)"
                        )
                except (SplitError, ValueError, KeyError) as error:
                    run.record_failure(
                        cause=str(error),
                        unit=f"closed_set/{reduction}/{frame_count}f",
                    )

        run.info("evaluation: finished")

        primary_rows = [
            row
            for row in configurations
            if row["spatial_reduction"] == primary["spatial_reduction"]
            and row["frame_count"] == primary["frame_count"]
        ]
        aggregate = _aggregate(configurations)
        metrics = {
            "evaluation_unit": "video",
            "protocol": {
                "split": protocol_name,
                "gallery_scope": gallery_scope,
                "distance": "euclidean",
                "official_pair_file": pair_path,
                "official_pair_available": pair_task is not None,
                "protocols": sorted({row["protocol"] for row in configurations}),
            },
            "primary": {**primary, "rows": primary_rows},
            "aggregate": aggregate,
            "configurations": configurations,
            "feature_cache": feature_record,
            "runtime_seconds": None,
        }

        result = {
            "experiment": experiment,
            "run_id": run.run_id,
            "created_at": run.metadata["started_at"],
            "config": config,
            "environment": environment_record(str(config["model"].get("device", "cpu"))),
            "dataset": run.metadata["dataset"],
            "split": run.metadata["split"],
            "model": feature_record,
            "sampling": run.metadata["sampling"],
            "evaluation": metrics["protocol"],
            "results": {
                "configurations": configurations,
                "aggregate": aggregate,
                "primary": metrics["primary"],
            },
            "runtime_seconds": round(time.time() - started, 2),
            "run_directory": str(run.paths.root),
        }

        result_dir = resolve(config["paths"]["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{experiment}_{run.run_id}.json"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        if per_query_frames:
            pd.concat(per_query_frames, ignore_index=True).to_csv(
                result_dir / f"{experiment}_{run.run_id}_per_query.csv", index=False
            )
        summary_path = result_dir / f"{experiment}_{run.run_id}_summary.csv"
        _write_summary(summary_path, configurations)
        result["result_path"] = display(result_path)
        result["summary_path"] = display(summary_path)

        metrics["runtime_seconds"] = round(time.time() - started, 2)
        metrics["result_path"] = result["result_path"]
        run.finish(metrics=metrics, status="completed")
    return result


def _aggregate(configurations: list[dict]) -> dict:
    """Summarise closed-set rows across folds, per protocol and configuration."""
    groups: dict[tuple, list[dict]] = {}
    for row in configurations:
        if not row["protocol"].startswith("closed_set"):
            continue
        key = (row["protocol"], row["spatial_reduction"], row["frame_count"])
        groups.setdefault(key, []).append(row)

    aggregate = {}
    for (protocol, reduction, frame_count), rows in sorted(groups.items()):
        values = {}
        for metric in ("rank1", "rank3", "mean_average_precision"):
            samples = np.array([row[metric] for row in rows], dtype="float64")
            values[metric] = {
                "mean": float(samples.mean()),
                "std": float(samples.std(ddof=1)) if len(samples) > 1 else 0.0,
                "values": samples.tolist(),
            }
        aggregate[f"{protocol}|{reduction}|{frame_count}f"] = values
    return aggregate


def _write_summary(path: Path, configurations: list[dict]) -> None:
    columns = [
        "protocol",
        "spatial_reduction",
        "frame_count",
        "unit",
        "top1",
        "top3",
        "rank1",
        "rank3",
        "mean_average_precision",
        "n_queries",
        "n_scored_queries",
        "n_gallery",
        "n_pairs",
        "n_trials",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in configurations:
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.experiment:
        config["experiment"] = args.experiment
    if args.protocol:
        config["protocol"]["name"] = args.protocol
    if args.device:
        config["model"]["device"] = args.device
    if args.batch_size:
        config["model"]["batch_size"] = args.batch_size
    if args.preprocessing:
        config["model"]["preprocessing"] = args.preprocessing
    if args.reductions:
        config["model"]["spatial_reductions"] = args.reductions
        config["evaluation"].setdefault("primary", {})["spatial_reduction"] = args.reductions[0]
    if args.frame_counts:
        config["evaluation"]["frame_counts"] = args.frame_counts
        config["evaluation"].setdefault("primary", {})["frame_count"] = args.frame_counts[0]
    if args.gallery_scope:
        config["evaluation"]["gallery_scope"] = args.gallery_scope
    if args.no_stream:
        config["evaluation"]["stream_log"] = False
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/dinov2_video.yaml", type=Path)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--protocol", choices=PROTOCOLS, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--preprocessing", choices=PREPROCESSING_MODES, default=None)
    parser.add_argument(
        "--reductions",
        nargs="+",
        choices=SPATIAL_REDUCTIONS,
        default=None,
        help="Spatial pooling variants to evaluate.",
    )
    parser.add_argument(
        "--frame-counts",
        nargs="+",
        type=int,
        default=None,
        help="Frames averaged per sequence (e.g. 5 10 20).",
    )
    parser.add_argument("--gallery-scope", choices=GALLERY_SCOPES, default=None)
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Write run.log only, without mirroring to stderr.",
    )
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(args.config), args)
    result = run(config)
    print(
        json.dumps(
            {
                "run_directory": result["run_directory"],
                "primary": result["results"]["primary"],
                "aggregate": result["results"]["aggregate"],
            },
            indent=2,
        )
    )
    print(f"Wrote {result['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
