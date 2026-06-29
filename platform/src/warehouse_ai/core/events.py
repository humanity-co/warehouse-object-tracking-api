from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    demand_observed = "demand_observed"
    inventory_updated = "inventory_updated"
    transfer_requested = "transfer_requested"
    route_planned = "route_planned"
    vision_analyzed = "vision_analyzed"
    anomaly_detected = "anomaly_detected"
    maintenance_scored = "maintenance_scored"
    slotting_optimized = "slotting_optimized"
    model_promoted = "model_promoted"
    decision_executed = "decision_executed"
    copilot_interaction = "copilot_interaction"


class WarehouseEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: EventType
    timestamp: datetime
    warehouse_id: str
    source: str
    subject_id: str
    payload: dict[str, Any]
    trace_id: str = Field(default_factory=lambda: uuid4().hex)

    @property
    def topic(self) -> str:
        return f"{self.event_type.value}.{self.warehouse_id}"


class LineageRecord(BaseModel):
    lineage_id: str = Field(default_factory=lambda: uuid4().hex)
    source_event_ids: list[str]
    feature_set_id: str
    model_name: str
    model_version: str
    output_event_id: str
    created_at: datetime
    notes: dict[str, Any] = Field(default_factory=dict)

