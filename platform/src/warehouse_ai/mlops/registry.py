from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelVersion:
    engine: str
    version: str
    stage: str
    path: Path
    metadata: dict[str, Any]


class LocalModelRegistry:
    """Filesystem model registry with explicit promotion gates."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        engine: str,
        version: str,
        source_path: Path,
        metadata: dict[str, Any],
        stage: str = "staging",
    ) -> ModelVersion:
        target_dir = self.base_dir / engine / version
        target_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = target_dir / "metadata.json"
        metadata_path.write_text(json.dumps({**metadata, "stage": stage}, indent=2))
        if source_path.is_file():
            target_file = target_dir / source_path.name
            target_file.write_bytes(source_path.read_bytes())
        return ModelVersion(
            engine=engine,
            version=version,
            stage=stage,
            path=target_dir,
            metadata=metadata,
        )

    def promote(self, engine: str, version: str, new_stage: str) -> ModelVersion:
        metadata_path = self.base_dir / engine / version / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["stage"] = new_stage
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return ModelVersion(
            engine=engine,
            version=version,
            stage=new_stage,
            path=metadata_path.parent,
            metadata=metadata,
        )

    def load(self, engine: str, stage: str = "production") -> Optional[ModelVersion]:
        engine_dir = self.base_dir / engine
        if not engine_dir.exists():
            return None
        for version_dir in sorted(engine_dir.iterdir(), reverse=True):
            metadata_path = version_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("stage") == stage:
                    return ModelVersion(
                        engine=engine,
                        version=version_dir.name,
                        stage=stage,
                        path=version_dir,
                        metadata=metadata,
                    )
        return None
