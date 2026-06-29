from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from warehouse_ai.core.schemas import EvaluationMetric


@dataclass
class TrainResult:
    engine_name: str
    model_version: str
    artifact_path: Path
    metrics: list[EvaluationMetric]
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseEngine:
    engine_name: str = "base"

    def save_state(
        self,
        model: torch.nn.Module,
        path: Path,
        extra: Optional[dict[str, Any]] = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state_dict": model.state_dict(), "extra": extra or {}}
        torch.save(payload, path)
        return path

    def load_state(self, model: torch.nn.Module, path: Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
        return payload.get("extra", {})

    def find_latest_artifact(self, artifacts_dir: Path) -> Optional[Path]:
        if not artifacts_dir.exists():
            return None
        artifacts = list(artifacts_dir.glob(f"{self.engine_name}_*.pt"))
        if not artifacts:
            return None
        return max(artifacts, key=lambda p: p.stat().st_mtime)
