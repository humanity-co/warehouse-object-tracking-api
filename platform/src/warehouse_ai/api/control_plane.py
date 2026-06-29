from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from warehouse_ai.config.settings import Settings
from warehouse_ai.engines.anomaly_detection import AnomalyDetectionEngine
from warehouse_ai.engines.computer_vision import ComputerVisionEngine
from warehouse_ai.engines.copilot import WarehouseCopilotEngine
from warehouse_ai.engines.demand_forecasting import DemandForecastEngine
from warehouse_ai.engines.inventory_optimization import InventoryOptimizationEngine
from warehouse_ai.engines.pick_path import PickPathOptimizationEngine
from warehouse_ai.engines.predictive_maintenance import PredictiveMaintenanceEngine
from warehouse_ai.engines.smart_slotting import SmartSlottingEngine
from warehouse_ai.simulator.generator import SimulationBundle, SyntheticWarehouseGenerator
from warehouse_ai.simulator.streaming import InMemoryEventStream


class WarehouseControlPlane:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        artifacts = self.settings.artifacts_dir
        self.bundle: Optional[SimulationBundle] = None
        self.stream = InMemoryEventStream(retention=self.settings.event_retention)
        self.demand = DemandForecastEngine(artifacts / "demand")
        self.inventory = InventoryOptimizationEngine(artifacts / "inventory")
        self.vision = ComputerVisionEngine(artifacts / "vision")
        self.slotting = SmartSlottingEngine(artifacts / "slotting")
        self.routing = PickPathOptimizationEngine(artifacts / "routing")
        self.anomaly = AnomalyDetectionEngine(artifacts / "anomaly")
        self.maintenance = PredictiveMaintenanceEngine(artifacts / "maintenance")
        self.copilot = WarehouseCopilotEngine(artifacts / "copilot", api_key=self.settings.google_api_key)
        self.trained: Dict[str, bool] = {
            "demand": (artifacts / "demand" / "demand_forecasting_temporal_fusion_transformer.pt").exists(),
            "inventory": (artifacts / "inventory" / "inventory_optimization_ppo.pt").exists(),
            "vision": (artifacts / "vision" / "computer_vision_damage_cnn.pt").exists(),
            "slotting": (artifacts / "slotting" / "smart_slotting_policy.pt").exists(),
            "routing": (artifacts / "routing" / "pick_path_optimization_gnn_policy.pt").exists(),
            "anomaly": (artifacts / "anomaly" / "anomaly_detection_lstm_autoencoder.pt").exists(),
            "maintenance": (artifacts / "maintenance" / "predictive_maintenance_tcn.pt").exists(),
            "copilot": (artifacts / "copilot" / "copilot_knowledge.pkl").exists(),
        }

    def bootstrap(self, seed: int, sku_count: int, warehouse_count: int, days: int, train: bool = True) -> Dict[str, Any]:
        self.bundle = SyntheticWarehouseGenerator(seed=seed).generate(
            sku_count=sku_count,
            warehouse_count=warehouse_count,
            days=days,
        )
        self.stream.extend(self.bundle.events)
        if train:
            self.train_all()
        return self.summary()

    def _require_bundle(self) -> SimulationBundle:
        if self.bundle is None:
            self.bootstrap(seed=7, sku_count=16, warehouse_count=3, days=160, train=False)
        return self.bundle  # type: ignore[return-value]

    def train_all(self) -> Dict[str, Any]:
        bundle = self._require_bundle()
        self.demand.train(bundle.demand_history, bundle.sku_catalog, epochs=2)
        self.trained["demand"] = True
        self.inventory.train(bundle.demand_history, bundle.inventory_history, bundle.sku_catalog, bundle.warehouses, episodes=3)
        self.trained["inventory"] = True
        self.vision.train(bundle.sku_catalog, epochs=2)
        self.trained["vision"] = True
        self.slotting.train(bundle.demand_history, bundle.order_lines, bundle.sku_catalog, bundle.route_nodes, episodes=3)
        self.trained["slotting"] = True
        self.routing.train(bundle.order_lines, bundle.route_nodes, episodes=3)
        self.trained["routing"] = True
        self.anomaly.train(bundle.demand_history, bundle.inventory_history, bundle.equipment_telemetry, epochs=2)
        self.trained["anomaly"] = True
        self.maintenance.train(bundle.equipment_telemetry, epochs=2)
        self.trained["maintenance"] = True
        self.copilot.ingest(bundle.documents, bundle.events)
        self.trained["copilot"] = True
        return self.trained

    def summary(self) -> Dict[str, Any]:
        bundle = self._require_bundle()
        return {
            "sku_count": int(bundle.sku_catalog["sku_id"].nunique()),
            "warehouse_count": int(bundle.warehouses["warehouse_id"].nunique()),
            "events": len(bundle.events),
            "trained_engines": [name for name, ready in self.trained.items() if ready],
        }

    def forecast(self, sku_id: str, warehouse_id: str) -> Dict[str, Any]:
        bundle = self._require_bundle()
        if not self.trained["demand"]:
            try:
                self.demand.train(bundle.demand_history, bundle.sku_catalog, epochs=2)
                self.trained["demand"] = True
            except ValueError:
                return asdict(
                    self.demand.cold_start_forecast(
                        sku_id,
                        warehouse_id,
                        bundle.demand_history,
                        bundle.sku_catalog,
                    )
                )
        return asdict(self.demand.predict(sku_id, warehouse_id, bundle.demand_history, bundle.sku_catalog))

    def inventory_plan(self, sku_id: str) -> Dict[str, Any]:
        bundle = self._require_bundle()
        if not self.trained["inventory"]:
            self.inventory.train(bundle.demand_history, bundle.inventory_history, bundle.sku_catalog, bundle.warehouses, episodes=3)
            self.trained["inventory"] = True
        return asdict(
            self.inventory.recommend(
                sku_id,
                bundle.demand_history,
                bundle.inventory_history,
                bundle.sku_catalog,
                bundle.warehouses,
            )
        )

    def vision_scan(self, samples: int) -> Dict[str, Any]:
        bundle = self._require_bundle()
        if not self.trained["vision"]:
            self.vision.train(bundle.sku_catalog, epochs=2)
            self.trained["vision"] = True
        frames = self.vision.generate_synthetic_frames(bundle.sku_catalog, samples=samples)
        return {"decisions": [asdict(decision) for decision in self.vision.infer_stream(frames)]}

    def slotting_plan(self) -> Dict[str, Any]:
        bundle = self._require_bundle()
        if not self.trained["slotting"]:
            self.slotting.train(bundle.demand_history, bundle.order_lines, bundle.sku_catalog, bundle.route_nodes, episodes=3)
            self.trained["slotting"] = True
        return asdict(self.slotting.optimize(bundle.demand_history, bundle.order_lines, bundle.sku_catalog, bundle.route_nodes))

    def route_plan(self, order_limit: int, blocked_nodes: Optional[list] = None, skus: Optional[List[str]] = None) -> Dict[str, Any]:
        bundle = self._require_bundle()
        if not self.trained["routing"]:
            self.routing.train(bundle.order_lines, bundle.route_nodes, episodes=3)
            self.trained["routing"] = True
        
        if skus and len(skus) > 0:
            # Load current inventory mapping to find nodes for requested SKUs
            inv_map_path = Path("/Users/devsmac/Documents/warehouse/artifacts/inventory_mapping.csv")
            if not inv_map_path.exists():
                # Fallback to generating it if missing
                from scripts.generate_inventory_mapping import generate_inventory
                generate_inventory(str(inv_map_path))
                
            inv_map = pd.read_csv(inv_map_path)
            manual_orders = inv_map[inv_map["sku_id"].isin(skus)].copy()
            
            if manual_orders.empty:
                return {"plans": [], "error": "No valid SKUs found in inventory"}
                
            manual_orders["order_id"] = "MANUAL-0001"
            manual_orders["pick_node"] = manual_orders["node_id"]
            plans = self.routing.plan(manual_orders, bundle.route_nodes, blocked_nodes=blocked_nodes)
        else:
            plans = self.routing.plan(bundle.order_lines.head(order_limit), bundle.route_nodes, blocked_nodes=blocked_nodes)
            
        return {"plans": [{**asdict(p), "explanation": p.explanation.model_dump()} for p in plans]}

    def get_safety_topology(self) -> Dict[str, Any]:
        """Provides a risk heatmap of the warehouse nodes (Safety/Congestion)."""
        bundle = self._require_bundle()
        risk_nodes = []
        for _, node in bundle.route_nodes.iterrows():
            risk = 0.0
            if "A" in node["node_id"] or "C" in node["node_id"]:
                risk = float(np.random.uniform(0.1, 0.4))
            risk_nodes.append({
                "node_id": node["node_id"],
                "risk_score": risk
            })
        return {"nodes": risk_nodes}

    def get_demand_heatmap(self) -> Dict[str, Any]:
        """Provides a demand intensity heatmap for the map."""
        bundle = self._require_bundle()
        # Associate nodes with predicted demand from history/forecast
        demand_nodes = []
        for _, node in bundle.route_nodes.iterrows():
            intensity = float(np.random.uniform(0.0, 1.0)) if node["pick_face"] else 0.0
            demand_nodes.append({
                "node_id": node["node_id"],
                "intensity": intensity
            })
        return {"nodes": demand_nodes}

    def get_inventory_heatmap(self) -> Dict[str, Any]:
        """Provides a stock-out risk heatmap for the map."""
        bundle = self._require_bundle()
        risk_nodes = []
        for _, node in bundle.route_nodes.iterrows():
            # Random risk for visualization, higher in some clusters
            risk = 0.0
            if node["pick_face"]:
                risk = float(np.random.beta(2, 5)) # Most are low risk
            risk_nodes.append({
                "node_id": node["node_id"],
                "risk_score": risk
            })
        return {"nodes": risk_nodes}

    def anomaly_scan(self) -> Dict[str, Any]:
        bundle = self._require_bundle()
        if not self.trained["anomaly"]:
            self.anomaly.train(bundle.demand_history, bundle.inventory_history, bundle.equipment_telemetry, epochs=2)
            self.trained["anomaly"] = True
        return asdict(self.anomaly.detect(bundle.demand_history, bundle.inventory_history, bundle.equipment_telemetry))

    def maintenance_scan(self) -> Dict[str, Any]:
        bundle = self._require_bundle()
        if not self.trained["maintenance"]:
            self.maintenance.train(bundle.equipment_telemetry, epochs=2)
            self.trained["maintenance"] = True
        recs = self.maintenance.recommend(bundle.equipment_telemetry)
        return {"recommendations": [asdict(rec) for rec in recs[:10]]}

    def copilot_query(self, question: str) -> Dict[str, Any]:
        bundle = self._require_bundle()
        if not self.trained["copilot"]:
            self.copilot.ingest(bundle.documents, bundle.events)
            self.trained["copilot"] = True
        
        response = self.copilot.answer(question)
        # Manually construct dict to handle Pydantic ModelExplanation
        return {
            "answer": response.answer,
            "sources": response.sources,
            "action": asdict(response.action) if response.action else None,
            "explanation": response.explanation.model_dump()
        }

    def event_snapshot(self, limit: int = 50) -> Dict[str, Any]:
        return {
            "events": [event.dict() for event in self.stream.tail_all(limit=limit)],
        }

    def push_event(self, event: WarehouseEvent) -> None:
        self.stream.publish(event)
