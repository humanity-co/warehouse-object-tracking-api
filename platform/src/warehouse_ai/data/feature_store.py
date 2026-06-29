from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class FeatureRecord:
    entity_id: str
    timestamp: datetime
    features: dict[str, Any]


class InMemoryFeatureStore:
    """A Redis-replaceable feature store used for local development and tests."""

    def __init__(self) -> None:
        self._store: dict[str, list[FeatureRecord]] = defaultdict(list)

    def materialize(self, namespace: str, frame: pd.DataFrame, entity_key: str) -> None:
        for row in frame.to_dict(orient="records"):
            timestamp = row.get("timestamp")
            if timestamp is None:
                raise ValueError("feature rows must include a timestamp column")
            entity_id = str(row[entity_key])
            features = {k: v for k, v in row.items() if k not in {entity_key}}
            self._store[f"{namespace}:{entity_id}"].append(
                FeatureRecord(
                    entity_id=entity_id,
                    timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                    features=features,
                )
            )
        for records in self._store.values():
            records.sort(key=lambda item: item.timestamp)

    def get_latest(self, namespace: str, entity_id: str) -> dict[str, Any]:
        records = self._store.get(f"{namespace}:{entity_id}", [])
        if not records:
            raise KeyError(f"no features found for {namespace}:{entity_id}")
        return records[-1].features

    def get_window(
        self,
        namespace: str,
        entity_id: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        records = self._store.get(f"{namespace}:{entity_id}", [])
        return [
            record.features
            for record in records
            if start <= record.timestamp <= end
        ]

