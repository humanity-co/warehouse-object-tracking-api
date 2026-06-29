from __future__ import annotations

import os
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared across services and training jobs."""

    project_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("WAREHOUSE_PROJECT_ROOT", "/Users/devsmac/Documents/warehouse")
        )
    )
    artifacts_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "WAREHOUSE_ARTIFACTS_DIR",
                "/Users/devsmac/Documents/warehouse/artifacts",
            )
        )
    )
    runs_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("WAREHOUSE_RUNS_DIR", "/Users/devsmac/Documents/warehouse/runs")
        )
    )
    secret_key: str = field(
        default_factory=lambda: os.getenv(
            "WAREHOUSE_SECRET_KEY", "warehouse-intelligence-secret"
        )
    )
    google_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )
    access_token_ttl_minutes: int = int(
        os.getenv("WAREHOUSE_ACCESS_TTL_MINUTES", "120")
    )
    event_retention: int = int(os.getenv("WAREHOUSE_EVENT_RETENTION", "5000"))
    default_quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)

    def ensure_directories(self) -> None:
        for path in (self.project_root, self.artifacts_dir, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)

