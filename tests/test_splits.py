"""Split invariants: composite sequence keys, official folds, disjointness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from animal_re_id.data.splits import (
    PROTOCOL_FIVE_FOLD,
    PROTOCOL_ROVF,
    SplitError,
    add_sequence_columns,
    build_retrieval_task,
    check_assignments,
    five_fold_assignment,
    identity_id_map,
    load_official_folds,
    load_split,
    representative_frames,
    rovf_assignment,
    save_split,
    sequence_table,
    split_signature,
    stratified_five_fold_assignment,
    validate_retrieval_task,
)


@pytest.fixture
def prepared(catalogue: pd.DataFrame) -> pd.DataFrame:
    return add_sequence_columns(catalogue)


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
def test_sequence_column_uses_composite_key(prepared: pd.DataFrame) -> None:
    sequences = prepared["sequence"].unique()
    assert len(sequences) == prepared.groupby(["identity", "video"]).ngroups
    # A bare video id maps to several identities, the composite key does not.
    assert prepared.groupby("video")["sequence"].nunique().max() > 1
    assert prepared.groupby("sequence")["identity"].nunique().max() == 1


def test_duplicate_paths_are_rejected(prepared: pd.DataFrame) -> None:
    broken = pd.concat([prepared, prepared.iloc[[0]]], ignore_index=True)
    with pytest.raises(SplitError, match="duplicated image paths"):
        add_sequence_columns(broken)


def test_sequence_table_counts_frames(prepared: pd.DataFrame) -> None:
    table = sequence_table(prepared)
    assert table["n_frames"].eq(4).all()
    assert table["first_frame"].eq(0).all() and table["last_frame"].eq(3).all()
    assert np.issubdtype(table["video"].dtype, np.integer)


def test_representative_frame_is_the_middle_one(prepared: pd.DataFrame) -> None:
    frames = representative_frames(prepared)
    assert len(frames) == prepared["sequence"].nunique()
    assert frames["sequence"].is_unique
    # frames 0..3 -> two middle candidates, the lower one wins the tie-break
    assert set(frames["frame"]) == {1}


# --------------------------------------------------------------------------- #
# Official five-fold protocol
# --------------------------------------------------------------------------- #
def test_identity_id_map_matches_animal_db(synthetic_root: Path) -> None:
    assert identity_id_map(synthetic_root) == {"Vera": 0, "Nanuq": 1, "Tonja": 2}


def test_official_folds_partition_every_sequence(prepared: pd.DataFrame, synthetic_root: Path) -> None:
    assignment = five_fold_assignment(prepared, synthetic_root)
    assert len(assignment) == prepared["sequence"].nunique()
    assert assignment["sequence"].is_unique
    assert set(assignment["protocol"]) == {PROTOCOL_FIVE_FOLD}
    assert set(assignment["fold"]) == {1, 2, 3, 4, 5}


def test_official_folds_are_stratified_per_identity(
    prepared: pd.DataFrame, synthetic_root: Path
) -> None:
    assignment = five_fold_assignment(prepared, synthetic_root)
    counts = assignment.pivot_table(
        index="identity", columns="fold", values="sequence", aggfunc="count"
    )
    assert counts.notna().all().all(), "every identity must appear in every fold"
    assert counts.max(axis=1).sub(counts.min(axis=1)).le(1).all()


def test_official_folds_are_independent_of_catalogue_order(
    prepared: pd.DataFrame, synthetic_root: Path
) -> None:
    shuffled = prepared.sample(frac=1.0, random_state=11).reset_index(drop=True)
    first = five_fold_assignment(prepared, synthetic_root)
    second = five_fold_assignment(shuffled, synthetic_root)
    merged = first.merge(second, on="sequence", suffixes=("_a", "_b"))
    assert len(merged) == len(first)
    assert (merged["fold_a"] == merged["fold_b"]).all()


def test_each_fold_is_disjoint_from_the_rest(
    prepared: pd.DataFrame, synthetic_root: Path
) -> None:
    assignment = five_fold_assignment(prepared, synthetic_root)
    for fold in assignment["fold"].unique():
        held_out = set(assignment.loc[assignment["fold"] == fold, "sequence"])
        rest = set(assignment.loc[assignment["fold"] != fold, "sequence"])
        assert held_out.isdisjoint(rest)


def test_missing_fold_file_is_reported(prepared: pd.DataFrame, synthetic_root: Path) -> None:
    (synthetic_root / "track_fold_info_4.csv").unlink()
    with pytest.raises(SplitError, match="Missing official fold file"):
        five_fold_assignment(prepared, synthetic_root)


def test_a_sequence_without_a_published_fold_is_reported(
    prepared: pd.DataFrame, synthetic_root: Path
) -> None:
    victim = synthetic_root / "track_fold_info_2.csv"
    table = pd.read_csv(victim)
    table.iloc[:-1].to_csv(victim, index=False)
    with pytest.raises(SplitError, match="no official fold"):
        five_fold_assignment(prepared, synthetic_root)


def test_ambiguous_fold_files_are_rejected(synthetic_root: Path) -> None:
    first = pd.read_csv(synthetic_root / "track_fold_info_1.csv")
    duplicated = pd.concat([first, first.iloc[[0]]], ignore_index=True)
    duplicated.to_csv(synthetic_root / "track_fold_info_2.csv", index=False)
    with pytest.raises(SplitError, match="more than one fold|not unique"):
        load_official_folds(synthetic_root)


# --------------------------------------------------------------------------- #
# Custom seeded five-fold ablation
# --------------------------------------------------------------------------- #
def test_stratified_fold_redraw_is_deterministic(prepared: pd.DataFrame) -> None:
    first = stratified_five_fold_assignment(prepared, seed=0)
    second = stratified_five_fold_assignment(prepared, seed=0)
    pd.testing.assert_frame_equal(first, second)

    counts = first.pivot_table(
        index="identity", columns="fold", values="sequence", aggfunc="count"
    )
    assert counts.notna().all().all()
    assert counts.max(axis=1).sub(counts.min(axis=1)).le(1).all()


def test_stratified_fold_redraw_changes_with_the_seed(prepared: pd.DataFrame) -> None:
    first = stratified_five_fold_assignment(prepared, seed=0)
    other = stratified_five_fold_assignment(prepared, seed=1)
    assert not first["fold"].equals(other["fold"])


# --------------------------------------------------------------------------- #
# RoVF protocol
# --------------------------------------------------------------------------- #
@pytest.fixture
def rovf_dataset(make_dataset, tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Ten tracklets per identity, the smallest size where 70/10/20 yields a val set."""
    from animal_re_id.data.wildlife import load_polar_bear_vid_id

    root = make_dataset(tmp_path / "PolarBearVidID", sequences_per_identity=10)
    return root, add_sequence_columns(load_polar_bear_vid_id(root))


def test_rovf_split_matches_reference_ratios(rovf_dataset) -> None:
    root, prepared = rovf_dataset
    assignment = rovf_assignment(prepared, root)
    counts = assignment.pivot_table(
        index="identity", columns="split", values="sequence", aggfunc="count"
    )
    assert set(counts.columns) == {"train", "val", "test"}
    for identity, row in counts.iterrows():
        assert row.sum() == 10
        # 10 tracklets, zero-based ranks: 0-6 train (<70%), 7 val (>=70%, <80%),
        # 8-9 test (>=80%). Note 10*0.8 == 8.0, so rank 8 is already test.
        assert row["train"] == 7 and row["val"] == 1 and row["test"] == 2, identity


def test_rovf_split_is_order_independent(rovf_dataset) -> None:
    root, prepared = rovf_dataset
    shuffled = prepared.sample(frac=1.0, random_state=7).reset_index(drop=True)
    first = rovf_assignment(prepared, root)
    second = rovf_assignment(shuffled, root)
    merged = first.merge(second, on="sequence", suffixes=("_a", "_b"))
    assert len(merged) == len(first)
    assert (merged["split_a"] == merged["split_b"]).all()


def test_rovf_split_follows_track_info_row_order(rovf_dataset) -> None:
    root, prepared = rovf_dataset
    assignment = rovf_assignment(prepared, root).set_index("sequence")
    tracks = pd.read_csv(root / "track_info.csv")
    animals = pd.read_csv(root / "animal_db.csv").set_index("id")["name"]
    for position, row in enumerate(tracks.itertuples()):
        identity = animals[row.id]
        per_identity = tracks[tracks["id"] == row.id].reset_index(drop=True)
        rank = per_identity.index[per_identity["tracklet"] == row.tracklet][0]
        total = len(per_identity)
        expected = "test" if rank >= total * 0.8 else "val" if rank >= total * 0.7 else "train"
        assert assignment.loc[f"{identity}|{row.tracklet}", "split"] == expected


def test_rovf_rejects_impossible_ratios(prepared: pd.DataFrame, synthetic_root: Path) -> None:
    with pytest.raises(SplitError, match="train_ratio"):
        rovf_assignment(prepared, synthetic_root, train_ratio=0.8, val_ratio=0.7)


def test_rovf_requires_track_info(prepared: pd.DataFrame, synthetic_root: Path) -> None:
    (synthetic_root / "track_info.csv").unlink()
    with pytest.raises(SplitError, match="track_info.csv"):
        rovf_assignment(prepared, synthetic_root)


# --------------------------------------------------------------------------- #
# Retrieval task construction
# --------------------------------------------------------------------------- #
def test_retrieval_task_is_leave_one_out(prepared: pd.DataFrame, synthetic_root: Path) -> None:
    assignment = five_fold_assignment(prepared, synthetic_root)
    frames = representative_frames(prepared)
    query, gallery = build_retrieval_task(frames, assignment, PROTOCOL_FIVE_FOLD, fold=1)

    assert len(query) == len(gallery)
    assert set(query["sequence"]) == set(gallery["sequence"])
    # The query image is in the gallery and is masked later as a self-match.
    assert query["path"].iloc[0] in set(gallery["path"])
    for _, row in query.iterrows():
        positives = gallery[(gallery["identity"] == row["identity"]) & (gallery["path"] != row["path"])]
        assert not positives.empty


def test_full_gallery_scope_ranks_against_everything(
    prepared: pd.DataFrame, synthetic_root: Path
) -> None:
    assignment = five_fold_assignment(prepared, synthetic_root)
    frames = representative_frames(prepared)
    query, gallery = build_retrieval_task(
        frames, assignment, PROTOCOL_FIVE_FOLD, fold=1, gallery_scope="all"
    )
    assert len(gallery) == len(frames) == 18
    assert len(query) < len(gallery)
    assert set(query["sequence"]) < set(gallery["sequence"])


def test_unknown_gallery_scope_is_rejected(prepared: pd.DataFrame, synthetic_root: Path) -> None:
    assignment = five_fold_assignment(prepared, synthetic_root)
    frames = representative_frames(prepared)
    with pytest.raises(SplitError, match="gallery scope"):
        build_retrieval_task(frames, assignment, PROTOCOL_FIVE_FOLD, fold=1, gallery_scope="zoo")


def test_every_identity_is_required_in_the_gallery() -> None:
    query = pd.DataFrame(
        {"identity": ["A", "B"], "sequence": ["A|1", "B|1"], "path": ["a1", "b1"], "frame": [1, 1]}
    )
    gallery = pd.DataFrame(
        {"identity": ["A", "A"], "sequence": ["A|1", "A|2"], "path": ["a1", "a2"], "frame": [1, 1]}
    )
    with pytest.raises(SplitError, match="every query identity in the gallery"):
        validate_retrieval_task(query, gallery, protocol=PROTOCOL_ROVF)


def test_a_query_without_positive_is_rejected() -> None:
    query = pd.DataFrame(
        {"identity": ["A"], "sequence": ["A|1"], "path": ["a1"], "frame": [1]}
    )
    gallery = pd.DataFrame(
        {"identity": ["A", "B"], "sequence": ["A|1", "B|1"], "path": ["a1", "b1"], "frame": [1, 1]}
    )
    with pytest.raises(SplitError, match="positive gallery images"):
        validate_retrieval_task(query, gallery, protocol=PROTOCOL_ROVF)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_split_round_trip_and_signature(prepared: pd.DataFrame, synthetic_root: Path, tmp_path: Path) -> None:
    assignment = five_fold_assignment(prepared, synthetic_root)
    check_assignments(assignment)
    signature = split_signature(assignment)
    path = save_split(assignment, tmp_path, "polarbear_five_fold", metadata={"seed": 3})
    assert path.exists()

    loaded, payload = load_split(path)
    assert payload["signature"] == signature
    assert split_signature(loaded) == signature
    assert payload["counts"]["sequences"] == len(assignment)
    assert payload["counts"]["identities"] == 3
    assert payload["metadata"]["seed"] == 3


def test_bad_assignment_is_rejected(prepared: pd.DataFrame, synthetic_root: Path) -> None:
    assignment = five_fold_assignment(prepared, synthetic_root)
    broken = pd.concat([assignment, assignment.iloc[[0]]], ignore_index=True)
    with pytest.raises(SplitError, match="duplicated sequences"):
        check_assignments(broken)


def test_mixed_protocols_are_rejected(prepared: pd.DataFrame, synthetic_root: Path) -> None:
    assignment = five_fold_assignment(prepared, synthetic_root)
    broken = assignment.copy()
    broken.loc[broken.index[0], "protocol"] = PROTOCOL_ROVF
    with pytest.raises(SplitError, match="mixes protocols"):
        check_assignments(broken)
