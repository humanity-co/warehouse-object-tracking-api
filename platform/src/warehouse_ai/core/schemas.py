from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class ZoneType(str, Enum):
    ambient = "ambient"
    chilled = "chilled"
    hazmat = "hazmat"
    bulky = "bulky"
    fast_pick = "fast_pick"


class EquipmentType(str, Enum):
    conveyor = "conveyor"
    shuttle = "shuttle"
    forklift = "forklift"
    sorter = "sorter"


class Dimensions(BaseModel):
    length_cm: float
    width_cm: float
    height_cm: float
    weight_kg: float

    @property
    def volume_cm3(self) -> float:
        return self.length_cm * self.width_cm * self.height_cm


class SKUProfile(BaseModel):
    sku_id: str
    category: str
    storage_zone: ZoneType
    supplier_id: str
    lead_time_days: int
    unit_cost: float
    shelf_life_days: int
    dimensions: Dimensions
    service_level_target: float = 0.97


class WarehouseZone(BaseModel):
    zone_id: str
    warehouse_id: str
    zone_type: ZoneType
    capacity_units: int
    x: float
    y: float
    congestion_sensitivity: float


class WarehouseProfile(BaseModel):
    warehouse_id: str
    name: str
    latitude: float
    longitude: float
    zones: list[WarehouseZone]


class DemandRecord(BaseModel):
    timestamp: datetime
    sku_id: str
    warehouse_id: str
    units: int
    promotion_flag: int
    external_signal: float
    price_index: float


class InventorySnapshot(BaseModel):
    timestamp: datetime
    warehouse_id: str
    sku_id: str
    on_hand: int
    in_transit: int
    reserved: int
    damaged: int
    reorder_point: int
    unit_cost: float


class EquipmentTelemetry(BaseModel):
    timestamp: datetime
    equipment_id: str
    warehouse_id: str
    equipment_type: EquipmentType
    vibration_rms: float
    motor_temp_c: float
    acoustic_db: float
    cycle_count: int
    load_factor: float


class VisionObservation(BaseModel):
    timestamp: datetime
    camera_id: str
    warehouse_id: str
    zone_id: str
    expected_sku_id: str
    predicted_sku_id: Optional[str] = None
    observed_count: int
    barcode_text: Optional[str] = None
    damage_score: float = 0.0
    damage_severity: str = "none"


class RouteNode(BaseModel):
    node_id: str
    warehouse_id: str
    zone_id: str
    x: float
    y: float


class OrderLine(BaseModel):
    order_id: str
    sku_id: str
    warehouse_id: str
    quantity: int
    requested_at: datetime
    promised_by: datetime
    priority: int = 1


class CopilotDocument(BaseModel):
    doc_id: str
    source: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


class ModelExplanation(BaseModel):
    engine: str
    summary: str
    confidence: float
    feature_contributions: dict[str, float]
    evidence: list[str] = Field(default_factory=list)


class DecisionAction(BaseModel):
    action_type: str
    entity_id: str
    recommended_value: Union[float, int, str]
    confidence: float
    explanation: ModelExplanation
    created_at: datetime


class EvaluationMetric(BaseModel):
    name: str
    value: float
    split: str = "validation"
