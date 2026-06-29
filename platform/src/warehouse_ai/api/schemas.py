from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class BootstrapRequest(BaseModel):
    seed: int = 7
    sku_count: int = 16
    warehouse_count: int = 3
    days: int = 160
    train: bool = True


class ForecastRequest(BaseModel):
    sku_id: str
    warehouse_id: str


class VisionScanRequest(BaseModel):
    samples: int = 12


class RoutingRequest(BaseModel):
    order_limit: int = 24
    blocked_nodes: List[str] = Field(default_factory=list)
    skus: Optional[List[str]] = Field(default_factory=list)


class CopilotQueryRequest(BaseModel):
    question: str


class APIResponse(BaseModel):
    status: str = "ok"
    data: Any
    error: Optional[str] = None

