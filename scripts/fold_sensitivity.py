"""Measure how much the *fold assignment* moves the metrics.

    uv run python scripts/fold_sensitivity.py --trials 5

The released ``track_fold_info_*.csv`` files are one fixed draw of a five-fold
split. This script re-scores the same cached embeddings after permuting the
official fold labels within each identity, which preserves every per-identity
fold size exactly but destroys the grouping the authors used.

The gap it reports is why this project evaluates the *official* folds rather
than a regenerated stratified split: the two are not interchangeable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import (  # noqa: E402
    REPO_ROOT,
    _display,
    _resolve,
    embed_all_frames,
    load_config,
)

from animal_re_id.data.splits import (  # noqa: E402
    PROTOCOL_FIVE_FOLD,
    add_sequence_columns,
    build_retrieval_task,
    five_fold_assignment,
    representative_frames,
)
from animal_re_id.data.wildlife import load_polar_bear_vid_id  # noqa: E402
from animal_re_id.evaluation.retrieval import cosine_similarity, rank_retrieval  # noqa: E402


def score(assignment: pd.DataFrame, frames: pd.DataFrame, features: np.ndarray) -> dict:
    position = {sequence: index for index, sequence in enumerate(frames["sequence"])}
    rank1, rank3, average_precision = [], [], []
    for fold in sorted(assignment["fold"].unique()):
        query, gallery = build_retrieval_task(
            frames, assignment, PROTOCOL_FIVE_FOLD, fold=int(fold)
        )
        similarity = cosine_similarity(
            features[[position[s] for s in query["sequence"]]],
            features[[position[s] for s in gallery["sequence"]]],
        )
        result = rank_retrieval(similarity, query, gallery)
        rank1.append(result.rank1)
        rank3.append(result.rank3)
        average_precision.append(result.mean_average_precision)
    return {
        "rank1": float(np.mean(rank1)),
        "rank1_std": float(np.std(rank1, ddof=1)),
        "rank3": float(np.mean(rank3)),
        "mAP": float(np.mean(average_precision)),
        "mAP_std": float(np.std(average_precision, ddof=1)),
        "per_fold_rank1": rank1,
    }


def permute_within_identity(assignment: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle fold labels inside each identity, keeping that identity's fold sizes."""
    permuted = assignment.copy()
    values = pd.Series(index=permuted.index, dtype="int64")
    for _, group in permuted.groupby("identity"):
        if group["fold"].nunique() == 1:
            values.loc[group.index] = group["fold"].to_numpy()
            continue
        values.loc[group.index] = rng.permutation(group["fold"].to_numpy())
    permuted["fold"] = values.to_numpy()
    return permuted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/polarbear.yaml", type=Path)
    parser.add_argument("--trials", type=int, default=5, help="Number of permuted redraws.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    dataset_root = _resolve(config["dataset"]["root"])
    catalogue = add_sequence_columns(load_polar_bear_vid_id(dataset_root))
    frames = representative_frames(catalogue)
    features, model_record, _ = embed_all_frames(config, frames)

    official = five_fold_assignment(catalogue, dataset_root)
    reference = score(official, frames, features)
    print(
        f"official folds            rank1={reference['rank1']:.4f}+-{reference['rank1_std']:.4f}"
        f"  rank3={reference['rank3']:.4f}  mAP={reference['mAP']:.4f}"
    )

    rng = np.random.default_rng(args.seed)
    redraws = []
    for trial in range(args.trials):
        redrawn = score(permute_within_identity(official, rng), frames, features)
        redraws.append(redrawn)
        print(
            f"permuted #{trial:<3d}            rank1={redrawn['rank1']:.4f}+-{redrawn['rank1_std']:.4f}"
            f"  rank3={redrawn['rank3']:.4f}  mAP={redrawn['mAP']:.4f}"
        )

    values = np.array([row["rank1"] for row in redraws])
    summary = {
        "official": reference,
        "permuted": redraws,
        "permuted_rank1_mean": float(values.mean()),
        "permuted_rank1_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "gap_rank1": float(reference["rank1"] - values.mean()),
        "trials": args.trials,
        "seed": args.seed,
        "model": model_record["name"],
        "config": config,
    }

    output = REPO_ROOT / "experiments" / "results" / "fold_sensitivity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(
        f"\npermuted redraws: rank1={summary['permuted_rank1_mean']:.4f}"
        f"+-{summary['permuted_rank1_std']:.4f}  gap={summary['gap_rank1']:+.4f}"
    )
    print(f"Wrote {_display(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
