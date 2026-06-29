from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from warehouse_ai.core.events import LineageRecord


@dataclass
class LineageTracker:
    """Trace raw events through feature materialization into model outputs."""

    records_by_output: dict[str, LineageRecord]

    def __init__(self) -> None:
        self.records_by_output = {}
        self.records_by_model = defaultdict(list)

    def record(self, record: LineageRecord) -> None:
        self.records_by_output[record.output_event_id] = record
        self.records_by_model[record.model_name].append(record)

    def by_output(self, output_event_id: str) -> Optional[LineageRecord]:
        return self.records_by_output.get(output_event_id)

    def by_model(self, model_name: str) -> list[LineageRecord]:
        return list(self.records_by_model.get(model_name, []))
