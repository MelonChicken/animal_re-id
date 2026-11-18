"""Per-run experiment directories.

A run directory is the only durable record of an experiment, so these tests
check the files themselves: names, status transitions and the fact that a
crashed run still explains what happened.
"""

from __future__ import annotations

import json

import pytest

from animal_re_id.utils.logging import (
    FAILURE_COLUMNS,
    ExperimentRun,
    new_run_id,
    read_metadata,
    read_metrics,
)

CONFIG = {
    "experiment": "unit-test",
    "model": {"name": "facebook/dinov2-small"},
    "evaluation": {"frame_counts": [5, 10]},
}


def make_run(tmp_path, **kwargs) -> ExperimentRun:
    kwargs.setdefault("experiment", "unit-test")
    kwargs.setdefault("config", CONFIG)
    kwargs.setdefault("run_id", "run-001")
    kwargs.setdefault("stream", False)
    return ExperimentRun(root=tmp_path, **kwargs)


def test_run_id_is_sortable_and_unique():
    ids = {new_run_id() for _ in range(20)}
    assert len(ids) == 20
    assert all(len(value.split("-")[0]) == 16 for value in ids)


def test_run_creates_the_expected_layout(tmp_path):
    run = make_run(tmp_path)
    run.finish(metrics={"rank1": 0.5}, status="completed")

    root = tmp_path / "unit-test" / "run-001"
    assert run.paths.root == root
    for name in ("config.yaml", "metadata.json", "metrics.json", "run.log", "failures.csv"):
        assert (root / name).is_file(), name


def test_config_is_written_verbatim(tmp_path):
    import yaml

    run = make_run(tmp_path)
    run.finish(metrics={}, status="completed")
    written = yaml.safe_load(run.paths.config.read_text(encoding="utf-8"))
    assert written == CONFIG


def test_metadata_carries_the_config_too(tmp_path):
    """A run directory stays readable even if config.yaml is moved away."""
    run = make_run(tmp_path)
    run.finish(metrics={}, status="completed")

    metadata = read_metadata(run.paths.metadata)
    assert metadata["config"] == CONFIG


def test_metadata_starts_running_and_finishes_completed(tmp_path):
    run = make_run(tmp_path)
    assert read_metadata(run.paths.metadata)["status"] == "running"

    with run.stage("dataset_load"):
        pass
    run.finish(metrics={"rank1": 0.5}, status="completed")

    metadata = read_metadata(run.paths.metadata)
    assert metadata["status"] == "completed"
    assert metadata["run_id"] == "run-001"
    assert metadata["experiment"] == "unit-test"
    assert metadata["finished_at"] is not None
    assert metadata["runtime_seconds"] >= 0
    assert metadata["timings"]["dataset_load"] >= 0
    assert set(metadata["files"]) == {"config", "metadata", "metrics", "log", "failures"}


def test_metrics_are_serialized(tmp_path):
    run = make_run(tmp_path)
    run.finish(metrics={"rank1": 0.75, "nested": {"fold": [1, 2]}}, status="completed")

    payload = read_metrics(run.paths.metrics)
    assert payload["status"] == "completed"
    assert payload["metrics"]["rank1"] == 0.75
    assert payload["metrics"]["nested"]["fold"] == [1, 2]
    assert payload["run_id"] == "run-001"


def test_failures_are_written_with_cause_and_status(tmp_path):
    run = make_run(tmp_path)
    run.record_failure(cause="missing embedding\nfor sequence", unit=3, sequence="A|1")
    run.record_failure(cause="pair file absent", unit="official_pair", status="skipped")
    run.finish(metrics={}, status="completed")

    header = run.paths.failures.read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split(",")) == FAILURE_COLUMNS

    lines = run.paths.failures.read_text(encoding="utf-8").splitlines()[1:]
    assert len(lines) == 2
    assert "missing embedding for sequence" in lines[0]
    assert ",skipped" in lines[1]
    assert read_metadata(run.paths.metadata)["failures"] == 2


def test_failed_run_preserves_the_traceback(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(RuntimeError):
        with run:
            run.info("about to break")
            raise RuntimeError("checkpoint missing")

    metadata = read_metadata(run.paths.metadata)
    assert metadata["status"] == "failed"
    assert metadata["error"]["type"] == "RuntimeError"
    assert "checkpoint missing" in metadata["error"]["message"]
    assert "RuntimeError: checkpoint missing" in metadata["error"]["traceback"]

    log = run.paths.log.read_text(encoding="utf-8")
    assert "about to break" in log
    assert "Run failed: RuntimeError" in log
    assert "Traceback" in log


def test_log_records_stages_and_finish(tmp_path):
    run = make_run(tmp_path)
    with run.stage("feature_extraction"):
        run.info("embedded 40 frames")
    run.finish(metrics={}, status="completed")

    log = run.paths.log.read_text(encoding="utf-8")
    assert "feature_extraction: started" in log
    assert "embedded 40 frames" in log
    assert "feature_extraction: finished in" in log
    assert "status completed" in log


def test_stage_timing_is_recorded_even_when_the_stage_raises(tmp_path):
    run = make_run(tmp_path)
    with pytest.raises(ValueError):
        with run:
            with run.stage("split_build"):
                raise ValueError("bad split")

    assert read_metadata(run.paths.metadata)["timings"]["split_build"] >= 0


def test_update_metadata_deep_merges(tmp_path):
    run = make_run(tmp_path)
    run.update_metadata(dataset={"name": "PolarBearVidID", "frames": 10})
    run.update_metadata(dataset={"frames": 12})
    run.finish(metrics={}, status="completed")

    dataset = read_metadata(run.paths.metadata)["dataset"]
    assert dataset == {"name": "PolarBearVidID", "frames": 12}


def test_metrics_file_is_absent_when_a_run_never_finished(tmp_path):
    run = make_run(tmp_path)
    run.info("started")
    assert not run.paths.metrics.exists()
    assert json.loads(run.paths.metadata.read_text(encoding="utf-8"))["status"] == "running"
