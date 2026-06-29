from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from warehouse_ai.api.auth import issue_token, verify_token
from warehouse_ai.api.control_plane import WarehouseControlPlane
from warehouse_ai.api.schemas import (
    APIResponse,
    AuthRequest,
    AuthResponse,
    BootstrapRequest,
    CopilotQueryRequest,
    ForecastRequest,
    RoutingRequest,
    VisionScanRequest,
)
from warehouse_ai.core.events import WarehouseEvent
from warehouse_ai.config.settings import Settings


def create_app() -> FastAPI:
    settings = Settings()
    control_plane = WarehouseControlPlane(settings)
    app = FastAPI(
        title="Warehouse Intelligence Platform API",
        version="1.0.0",
        description="AI-first warehouse intelligence control plane",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    def require_auth(authorization: str = Header(...)) -> Dict:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.replace("Bearer ", "", 1)
        return verify_token(token, settings.secret_key)

    @app.post("/api/v1/auth/token", response_model=AuthResponse, tags=["auth"])
    async def login(payload: AuthRequest) -> AuthResponse:
        print(f"DEBUG: Login request received for user: {payload.username}")
        if payload.username != "admin" or payload.password != "warehouse":
            raise HTTPException(status_code=401, detail="invalid credentials")
        return AuthResponse(
            access_token=issue_token(payload.username, settings.secret_key, settings.access_token_ttl_minutes)
        )

    @app.get("/api/v1/health", response_model=APIResponse, tags=["system"])
    async def health() -> APIResponse:
        return APIResponse(data={"status": "healthy"})

    @app.post("/api/v1/simulation/bootstrap", response_model=APIResponse, tags=["simulation"])
    async def bootstrap(payload: BootstrapRequest, _: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.bootstrap(**payload.dict()))

    @app.post("/api/v1/simulation/push", response_model=APIResponse, tags=["simulation"])
    async def push_event(event: WarehouseEvent, _: Dict = Depends(require_auth)) -> APIResponse:
        control_plane.push_event(event)
        return APIResponse(data={"status": "event_pushed", "timestamp": event.timestamp})

    @app.get("/api/v1/simulation/summary", response_model=APIResponse, tags=["simulation"])
    async def summary(_: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.summary())

    @app.post("/api/v1/forecast", response_model=APIResponse, tags=["engines"])
    async def forecast(payload: ForecastRequest, _: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.forecast(payload.sku_id, payload.warehouse_id))

    @app.post("/api/v1/inventory/{sku_id}/optimize", response_model=APIResponse, tags=["engines"])
    async def inventory(sku_id: str, _: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.inventory_plan(sku_id))

    @app.post("/api/v1/vision/scan", response_model=APIResponse, tags=["engines"])
    async def vision(payload: VisionScanRequest, _: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.vision_scan(payload.samples))

    @app.post("/api/v1/slotting/optimize", response_model=APIResponse, tags=["engines"])
    async def slotting(_: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.slotting_plan())

    @app.post("/api/v1/routing/plan", response_model=APIResponse, tags=["engines"])
    async def routing(payload: RoutingRequest, _: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.route_plan(payload.order_limit, blocked_nodes=payload.blocked_nodes, skus=payload.skus))

    @app.post("/api/v1/anomaly/detect", response_model=APIResponse, tags=["engines"])
    async def anomaly(_: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.anomaly_scan())

    @app.post("/api/v1/maintenance/predict", response_model=APIResponse, tags=["engines"])
    async def maintenance(_: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.maintenance_scan())

    @app.post("/api/v1/copilot/query", response_model=APIResponse, tags=["copilot"])
    async def copilot(payload: CopilotQueryRequest, _: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.copilot_query(payload.question))

    @app.get("/api/v1/pick-path/layout", response_model=APIResponse, tags=["routing"])
    async def get_pick_layout(_: Dict = Depends(require_auth)) -> APIResponse:
        layout_path = Path("/Users/devsmac/Documents/warehouse/artifacts/route_nodes.csv")
        if not layout_path.exists():
            return APIResponse(status="error", error="Layout not found")
        df = pd.read_csv(layout_path)
        return APIResponse(data=df.to_dict(orient="records"))

    @app.get("/api/v1/pick-path/inventory", response_model=APIResponse, tags=["routing"])
    async def get_pick_inventory(_: Dict = Depends(require_auth)) -> APIResponse:
        inv_path = Path("/Users/devsmac/Documents/warehouse/artifacts/inventory_mapping.csv")
        if not inv_path.exists():
            return APIResponse(status="error", error="Inventory mapping not found")
        df = pd.read_csv(inv_path)
        return APIResponse(data=df.to_dict(orient="records"))

    @app.get("/api/v1/routing/safety-topology", response_model=APIResponse, tags=["routing"])
    async def get_safety_topology(_: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.get_safety_topology())

    @app.get("/api/v1/demand/heatmap", response_model=APIResponse, tags=["demand"])
    async def get_demand_heatmap(_: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.get_demand_heatmap())

    @app.get("/api/v1/inventory/heatmap", response_model=APIResponse, tags=["inventory"])
    async def get_inventory_heatmap(_: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.get_inventory_heatmap())

    @app.get("/api/v1/events", response_model=APIResponse, tags=["events"])
    async def events(_: Dict = Depends(require_auth)) -> APIResponse:
        return APIResponse(data=control_plane.event_snapshot())

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                snapshot = control_plane.event_snapshot(limit=25)
                # Use default=str to handle datetime objects
                await websocket.send_text(json.dumps(snapshot, default=str))
                await asyncio.sleep(1.5)
        except WebSocketDisconnect:
            return

    return app

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

