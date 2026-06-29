from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None


@dataclass
class ExperimentContext:
    run_name: str
    tracker_backend: str
    artifact_uri: str


class ExperimentTracker:
    """MLflow-first tracker with a local JSON fallback for dev environments."""

    def __init__(self, tracking_dir: Path) -> None:
        self.tracking_dir = tracking_dir
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def run(self, experiment_name: str, run_name: str) -> Iterator[ExperimentContext]:
        if mlflow is not None:
            mlflow.set_tracking_uri(self.tracking_dir.as_uri())
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name):
                yield ExperimentContext(
                    run_name=run_name,
                    tracker_backend="mlflow",
                    artifact_uri=mlflow.get_artifact_uri(),
                )
            return

        run_dir = self.tracking_dir / experiment_name / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        context = ExperimentContext(
            run_name=run_name,
            tracker_backend="local-json",
            artifact_uri=str(run_dir),
        )
        (run_dir / "meta.json").write_text(
            json.dumps({"run_name": run_name, "started_at": datetime.utcnow().isoformat()})
        )
        yield context

    def log_params(self, params: dict[str, Any]) -> None:
        if mlflow is not None:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        if mlflow is not None:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        if mlflow is not None:
            mlflow.log_artifact(str(path))
