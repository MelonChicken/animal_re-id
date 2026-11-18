"""Per-run experiment directory, log and result files.

One :class:`ExperimentRun` owns one directory::

    experiments/polarbear/<experiment>/<run_id>/
        config.yaml      the configuration that was executed
        metadata.json    environment, model, dataset, split, cache and status
        metrics.json     evaluation results
        run.log          timestamped runtime log
        failures.csv     one row per failed unit, with cause and status

The module deliberately does not use MLflow, W&B or any external service. It is
a thin layer over :mod:`logging` plus JSON and CSV writers, so a run directory
can be archived, diffed and read without this project.

Files are written eagerly where it matters: ``config.yaml`` at construction and
``metadata.json`` with ``status: running`` before any work starts. If the
process dies mid-run, the directory still explains what was attempted.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

FAILURE_COLUMNS = ("unit", "sequence", "cause", "status")
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class ExperimentRunError(RuntimeError):
    """Raised when a run directory cannot be created or updated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Return a sortable, collision-resistant run id."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _deep_merge(target: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


@dataclass
class RunPaths:
    root: Path
    config: Path
    metadata: Path
    metrics: Path
    log: Path
    failures: Path

    @classmethod
    def under(cls, root: Path) -> "RunPaths":
        return cls(
            root=root,
            config=root / "config.yaml",
            metadata=root / "metadata.json",
            metrics=root / "metrics.json",
            log=root / "run.log",
            failures=root / "failures.csv",
        )


@dataclass
class ExperimentRun:
    """Create and maintain one experiment run directory.

    Args:
        experiment: Experiment name, used as the directory above ``run_id``.
        config: The configuration being executed; written verbatim to YAML.
        root: Directory holding all experiments (``experiments/polarbear``).
        run_id: Override the generated id. Mostly useful in tests.
        stream: Mirror log lines to stderr in addition to ``run.log``.
    """

    experiment: str
    config: dict
    root: Path = Path("experiments/polarbear")
    run_id: str = ""
    stream: bool = True
    paths: RunPaths = field(init=False)
    metadata: dict = field(init=False, default_factory=dict)
    failures: list[dict] = field(init=False, default_factory=list)
    timings: dict = field(init=False, default_factory=dict)
    _logger: logging.Logger = field(init=False, repr=False)
    _handlers: list[logging.Handler] = field(init=False, repr=False, default_factory=list)
    _finished: bool = field(init=False, default=False)
    _started: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.run_id = self.run_id or new_run_id()
        self.paths = RunPaths.under(self.root / self.experiment / self.run_id)
        try:
            self.paths.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ExperimentRunError(
                f"Cannot create experiment directory {self.paths.root}: {error}"
            ) from error

        self._logger = logging.getLogger(f"animal_re_id.run.{self.run_id}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        formatter.converter = time.gmtime

        file_handler = logging.FileHandler(self.paths.log, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        self._attach(file_handler)
        if self.stream:
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setFormatter(formatter)
            self._attach(stream_handler)

        self.paths.config.write_text(
            yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        self._started = time.time()
        self.metadata = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "status": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "runtime_seconds": None,
            "config_file": str(self.paths.config),
            "config": self.config,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "timings": self.timings,
            "failures": 0,
        }
        self.write_metadata()
        self.info(f"Run {self.run_id} started in {self.paths.root}")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _attach(self, handler: logging.Handler) -> None:
        self._logger.addHandler(handler)
        self._handlers.append(handler)

    def log(self, message: str, level: str = "INFO") -> None:
        self._logger.log(getattr(logging, level.upper(), logging.INFO), message)

    def info(self, message: str) -> None:
        self.log(message, "INFO")

    def warning(self, message: str) -> None:
        self.log(message, "WARNING")

    def error(self, message: str) -> None:
        self.log(message, "ERROR")

    @contextmanager
    def stage(self, name: str):
        """Time a stage and log its duration, even when it raises."""
        self.info(f"{name}: started")
        started = time.time()
        try:
            yield
        finally:
            elapsed = round(time.time() - started, 2)
            self.timings[name] = elapsed
            self.info(f"{name}: finished in {elapsed}s")

    # ------------------------------------------------------------------ #
    # Metadata, metrics and failures
    # ------------------------------------------------------------------ #
    def update_metadata(self, **sections: Any) -> None:
        _deep_merge(self.metadata, sections)
        self.write_metadata()

    def write_metadata(self) -> None:
        self.metadata["timings"] = self.timings
        self.paths.metadata.write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def record_failure(
        self,
        cause: str,
        unit: str | int | None = None,
        sequence: str | None = None,
        status: str = "failed",
    ) -> None:
        """Record a unit that could not be evaluated and keep going."""
        entry = {
            "unit": "" if unit is None else str(unit),
            "sequence": "" if sequence is None else str(sequence),
            "cause": str(cause).replace("\n", " ").strip(),
            "status": status,
        }
        self.failures.append(entry)
        self.metadata["failures"] = len(self.failures)
        self.error(f"failure recorded: {entry}")
        self._write_failures()

    def _write_failures(self) -> None:
        with open(self.paths.failures, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS)
            writer.writeheader()
            writer.writerows(self.failures)

    def finish(self, metrics: dict | None = None, status: str = "completed") -> None:
        """Write ``metrics.json`` and ``failures.csv`` and close the log."""
        if self._finished:
            return
        self._finished = True

        if metrics is not None:
            payload = {
                "run_id": self.run_id,
                "experiment": self.experiment,
                "status": status,
                "finished_at": utc_now(),
                "metrics": metrics,
            }
            self.paths.metrics.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        self._write_failures()

        self.metadata["status"] = status
        self.metadata["finished_at"] = utc_now()
        self.metadata["runtime_seconds"] = round(time.time() - self._started, 2)
        self.metadata["failures"] = len(self.failures)
        self.metadata["files"] = {
            "config": str(self.paths.config),
            "metadata": str(self.paths.metadata),
            "metrics": str(self.paths.metrics),
            "log": str(self.paths.log),
            "failures": str(self.paths.failures),
        }
        self.write_metadata()

        self.info(f"Run {self.run_id} finished with status {status}")
        for handler in self._handlers:
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)
        self._handlers.clear()

    def fail(self, error: BaseException) -> None:
        """Mark the run as failed, preserving the traceback in ``run.log``."""
        self.error(f"Run failed: {type(error).__name__}: {error}")
        self._logger.error(
            "Traceback:\n%s", "".join(traceback.format_exception(type(error), error, error.__traceback__))
        )
        self.metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        self.finish(metrics=None, status="failed")

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "ExperimentRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.fail(exc)
            return False
        if not self._finished:
            self.finish(metrics=None, status="completed")
        return False


def read_metadata(path: str | Path) -> dict:
    """Load ``metadata.json``; used by tests and by the comparison report."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_metrics(path: str | Path) -> dict:
    """Load ``metrics.json``; used by tests and by the comparison report."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
