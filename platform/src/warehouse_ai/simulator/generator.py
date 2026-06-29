from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import pi
from random import Random

import numpy as np
import pandas as pd

from warehouse_ai.core.events import EventType, WarehouseEvent
from warehouse_ai.core.schemas import EquipmentType, ZoneType


@dataclass
class SimulationBundle:
    sku_catalog: pd.DataFrame
    warehouses: pd.DataFrame
    demand_history: pd.DataFrame
    inventory_history: pd.DataFrame
    equipment_telemetry: pd.DataFrame
    route_nodes: pd.DataFrame
    order_lines: pd.DataFrame
    documents: pd.DataFrame
    events: list[WarehouseEvent]


class SyntheticWarehouseGenerator:
    """Generates coherent, multi-engine training data from a single latent world."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.random = Random(seed)

    def generate(self, sku_count: int = 32, warehouse_count: int = 3, days: int = 180) -> SimulationBundle:
        sku_catalog = self._generate_sku_catalog(sku_count)
        warehouses = self._generate_warehouses(warehouse_count)
        demand_history = self._generate_demand_history(sku_catalog, warehouses, days)
        inventory_history = self._generate_inventory_history(sku_catalog, demand_history)
        equipment = self._generate_equipment_telemetry(warehouses, days * 8)
        route_nodes = self._generate_route_nodes(warehouses)
        order_lines = self._generate_orders(demand_history)
        documents = self._generate_documents(demand_history, inventory_history, equipment)
        events = self._generate_events(demand_history, inventory_history, equipment, order_lines)
        return SimulationBundle(
            sku_catalog=sku_catalog,
            warehouses=warehouses,
            demand_history=demand_history,
            inventory_history=inventory_history,
            equipment_telemetry=equipment,
            route_nodes=route_nodes,
            order_lines=order_lines,
            documents=documents,
            events=events,
        )

    def _generate_sku_catalog(self, sku_count: int) -> pd.DataFrame:
        categories = ["beverage", "snack", "health", "electronics", "household"]
        zones = [ZoneType.ambient.value, ZoneType.chilled.value, ZoneType.fast_pick.value]
        records = []
        for idx in range(sku_count):
            base_demand = int(self.rng.integers(8, 80))
            shelf_life = int(self.rng.integers(30, 365))
            records.append(
                {
                    "sku_id": f"SKU-{idx:04d}",
                    "category": self.random.choice(categories),
                    "storage_zone": self.random.choice(zones),
                    "supplier_id": f"SUP-{idx % 7:03d}",
                    "lead_time_days": int(self.rng.integers(2, 21)),
                    "unit_cost": float(np.round(self.rng.uniform(4, 120), 2)),
                    "shelf_life_days": shelf_life,
                    "length_cm": float(np.round(self.rng.uniform(8, 60), 2)),
                    "width_cm": float(np.round(self.rng.uniform(8, 40), 2)),
                    "height_cm": float(np.round(self.rng.uniform(5, 45), 2)),
                    "weight_kg": float(np.round(self.rng.uniform(0.2, 22), 2)),
                    "service_level_target": float(np.round(self.rng.uniform(0.92, 0.99), 3)),
                    "base_demand": base_demand,
                }
            )
        return pd.DataFrame.from_records(records)

    def _generate_warehouses(self, warehouse_count: int) -> pd.DataFrame:
        records = []
        for idx in range(warehouse_count):
            records.append(
                {
                    "warehouse_id": f"WH-{idx+1:02d}",
                    "name": f"Regional Hub {idx+1}",
                    "latitude": 18.0 + idx,
                    "longitude": 72.0 + idx,
                    "capacity_units": int(self.rng.integers(25_000, 80_000)),
                }
            )
        return pd.DataFrame.from_records(records)

    def _generate_demand_history(
        self,
        sku_catalog: pd.DataFrame,
        warehouses: pd.DataFrame,
        days: int,
    ) -> pd.DataFrame:
        start = datetime.utcnow() - timedelta(days=days)
        rows = []
        for _, sku in sku_catalog.iterrows():
            for _, warehouse in warehouses.iterrows():
                warehouse_name = str(warehouse["name"])
                wh_scale = 0.85 + warehouse_name.count("2") * 0.18 + warehouse_name.count("3") * 0.3
                promo_days = set(self.rng.choice(np.arange(days), size=max(days // 12, 1), replace=False))
                for day in range(days):
                    ts = start + timedelta(days=int(day))
                    weekly = 1.0 + 0.22 * np.sin(2 * pi * day / 7.0)
                    monthly = 1.0 + 0.15 * np.sin(2 * pi * day / 30.0 + float(sku["base_demand"]) / 12)
                    trend = 1.0 + (day / max(days, 1)) * self.rng.uniform(-0.08, 0.18)
                    promotion = 1 if day in promo_days else 0
                    external = 0.8 + 0.4 * np.sin(2 * pi * day / 45.0 + len(str(sku["sku_id"])))
                    mean_demand = float(sku["base_demand"]) * wh_scale * weekly * monthly * trend
                    if promotion:
                        mean_demand *= 1.35
                    demand = int(max(0, self.rng.poisson(max(mean_demand, 1.0))))
                    rows.append(
                        {
                            "timestamp": ts,
                            "sku_id": sku["sku_id"],
                            "warehouse_id": warehouse["warehouse_id"],
                            "units": demand,
                            "promotion_flag": promotion,
                            "external_signal": float(np.round(external, 4)),
                            "price_index": float(np.round(self.rng.uniform(0.9, 1.1), 4)),
                        }
                    )
        return pd.DataFrame.from_records(rows)

    def _generate_inventory_history(self, sku_catalog: pd.DataFrame, demand_history: pd.DataFrame) -> pd.DataFrame:
        sku_index = sku_catalog.set_index("sku_id")
        rows = []
        grouped = demand_history.sort_values("timestamp").groupby(["warehouse_id", "sku_id"], sort=False)
        for (warehouse_id, sku_id), group in grouped:
            sku = sku_index.loc[sku_id]
            lead_time = int(sku["lead_time_days"])
            service_level = float(sku["service_level_target"])
            on_hand = int(group["units"].iloc[:14].sum() + self.rng.integers(40, 200))
            rolling = group["units"].rolling(14, min_periods=1).mean()
            volatility = group["units"].rolling(14, min_periods=2).std().fillna(0)
            for (_, row), mean_units, std_units in zip(group.iterrows(), rolling, volatility):
                safety_stock = int((mean_units * lead_time) * (1.0 - (1.0 - service_level)) + 1.65 * std_units)
                reorder_point = int(mean_units * lead_time + safety_stock)
                inbound = int(max(0, reorder_point - on_hand) if on_hand < reorder_point else 0)
                sales = int(row["units"])
                damaged = int(self.rng.binomial(n=max(sales, 1), p=0.01))
                on_hand = max(0, on_hand + inbound - sales - damaged)
                rows.append(
                    {
                        "timestamp": row["timestamp"],
                        "warehouse_id": warehouse_id,
                        "sku_id": sku_id,
                        "on_hand": on_hand,
                        "in_transit": inbound,
                        "reserved": int(self.rng.integers(0, max(sales // 2, 1))),
                        "damaged": damaged,
                        "reorder_point": reorder_point,
                        "unit_cost": float(sku["unit_cost"]),
                    }
                )
        return pd.DataFrame.from_records(rows)

    def _generate_equipment_telemetry(self, warehouses: pd.DataFrame, periods: int) -> pd.DataFrame:
        start = datetime.utcnow() - timedelta(hours=periods)
        equipment_types = list(EquipmentType)
        rows = []
        for _, warehouse in warehouses.iterrows():
            for eq_idx in range(6):
                equipment_id = f"{warehouse['warehouse_id']}-EQ-{eq_idx+1:02d}"
                equipment_type = equipment_types[eq_idx % len(equipment_types)]
                degradation = self.rng.uniform(0.0, 0.3)
                for step in range(periods):
                    ts = start + timedelta(hours=int(step))
                    age_factor = degradation + step / max(periods, 1) * self.rng.uniform(0.1, 0.8)
                    rows.append(
                        {
                            "timestamp": ts,
                            "equipment_id": equipment_id,
                            "warehouse_id": warehouse["warehouse_id"],
                            "equipment_type": equipment_type.value,
                            "vibration_rms": float(np.round(0.5 + age_factor * 3 + self.rng.normal(0, 0.1), 4)),
                            "motor_temp_c": float(np.round(45 + age_factor * 35 + self.rng.normal(0, 1.2), 4)),
                            "acoustic_db": float(np.round(55 + age_factor * 18 + self.rng.normal(0, 1.8), 4)),
                            "cycle_count": int(step * self.rng.integers(8, 16)),
                            "load_factor": float(np.clip(np.round(0.45 + age_factor + self.rng.normal(0, 0.05), 4), 0.1, 1.3)),
                        }
                    )
        return pd.DataFrame.from_records(rows)

    def _generate_route_nodes(self, warehouses: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, warehouse in warehouses.iterrows():
            for aisle in range(1, 8):
                for slot in range(1, 10):
                    rows.append(
                        {
                            "node_id": f"{warehouse['warehouse_id']}-A{aisle:02d}-S{slot:02d}",
                            "warehouse_id": warehouse["warehouse_id"],
                            "zone_id": f"{warehouse['warehouse_id']}-Z{(aisle % 4)+1}",
                            "x": aisle * 10,
                            "y": slot * 5,
                            "pick_face": slot <= 4,
                        }
                    )
        return pd.DataFrame.from_records(rows)

    def _generate_orders(self, demand_history: pd.DataFrame) -> pd.DataFrame:
        order_rows = []
        grouped = demand_history[demand_history["units"] > 0].sample(
            frac=min(0.22, max(500 / max(len(demand_history), 1), 0.02)),
            random_state=self.seed,
        )
        for idx, row in grouped.reset_index(drop=True).iterrows():
            order_rows.append(
                {
                    "order_id": f"ORD-{idx:06d}",
                    "sku_id": row["sku_id"],
                    "warehouse_id": row["warehouse_id"],
                    "quantity": int(max(1, row["units"] // self.rng.integers(2, 6))),
                    "requested_at": row["timestamp"],
                    "promised_by": row["timestamp"] + timedelta(hours=int(self.rng.integers(12, 72))),
                    "priority": int(self.rng.integers(1, 4)),
                }
            )
        return pd.DataFrame.from_records(order_rows)

    def _generate_documents(
        self,
        demand_history: pd.DataFrame,
        inventory_history: pd.DataFrame,
        equipment: pd.DataFrame,
    ) -> pd.DataFrame:
        docs = []
        top_skus = (
            demand_history.groupby("sku_id")["units"].sum().sort_values(ascending=False).head(10)
        )
        for sku_id, units in top_skus.items():
            docs.append(
                {
                    "doc_id": f"doc-{sku_id.lower()}",
                    "source": "ops_log",
                    "title": f"Demand review for {sku_id}",
                    "content": (
                        f"{sku_id} moved {int(units)} units in the last synthetic planning horizon. "
                        "Monitor promotional uplift and maintain fast-pick capacity."
                    ),
                    "timestamp": datetime.utcnow(),
                    "metadata": {"sku_id": sku_id},
                }
            )
        risk_equipment = (
            equipment.groupby("equipment_id")["motor_temp_c"].mean().sort_values(ascending=False).head(6)
        )
        for equipment_id, temp in risk_equipment.items():
            docs.append(
                {
                    "doc_id": f"doc-{equipment_id.lower()}",
                    "source": "maintenance_note",
                    "title": f"Maintenance summary for {equipment_id}",
                    "content": (
                        f"{equipment_id} average motor temperature reached {temp:.2f}C. "
                        "Reroute picks when degradation score exceeds threshold."
                    ),
                    "timestamp": datetime.utcnow(),
                    "metadata": {"equipment_id": equipment_id},
                }
            )
        low_stock = inventory_history.sort_values("timestamp").groupby("sku_id").tail(1).nsmallest(10, "on_hand")
        for _, row in low_stock.iterrows():
            docs.append(
                {
                    "doc_id": f"doc-inv-{row['sku_id'].lower()}-{row['warehouse_id'].lower()}",
                    "source": "inventory_log",
                    "title": f"Inventory pressure for {row['sku_id']} in {row['warehouse_id']}",
                    "content": (
                        f"{row['sku_id']} at {row['warehouse_id']} is down to {int(row['on_hand'])} units "
                        f"with reorder point {int(row['reorder_point'])}."
                    ),
                    "timestamp": datetime.utcnow(),
                    "metadata": {"sku_id": row["sku_id"], "warehouse_id": row["warehouse_id"]},
                }
            )
        return pd.DataFrame.from_records(docs)

    def _generate_events(
        self,
        demand_history: pd.DataFrame,
        inventory_history: pd.DataFrame,
        equipment: pd.DataFrame,
        orders: pd.DataFrame,
    ) -> list[WarehouseEvent]:
        events: list[WarehouseEvent] = []
        for _, row in demand_history.tail(400).iterrows():
            events.append(
                WarehouseEvent(
                    event_type=EventType.demand_observed,
                    timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                    warehouse_id=row["warehouse_id"],
                    source="simulator.demand",
                    subject_id=row["sku_id"],
                    payload={"units": int(row["units"]), "promotion_flag": int(row["promotion_flag"])},
                )
            )
        for _, row in inventory_history.tail(300).iterrows():
            events.append(
                WarehouseEvent(
                    event_type=EventType.inventory_updated,
                    timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                    warehouse_id=row["warehouse_id"],
                    source="simulator.inventory",
                    subject_id=row["sku_id"],
                    payload={"on_hand": int(row["on_hand"]), "in_transit": int(row["in_transit"])},
                )
            )
        for _, row in equipment.tail(240).iterrows():
            events.append(
                WarehouseEvent(
                    event_type=EventType.maintenance_scored,
                    timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                    warehouse_id=row["warehouse_id"],
                    source="simulator.equipment",
                    subject_id=row["equipment_id"],
                    payload={
                        "vibration_rms": float(row["vibration_rms"]),
                        "motor_temp_c": float(row["motor_temp_c"]),
                    },
                )
            )
        for _, row in orders.tail(150).iterrows():
            events.append(
                WarehouseEvent(
                    event_type=EventType.route_planned,
                    timestamp=pd.Timestamp(row["requested_at"]).to_pydatetime(),
                    warehouse_id=row["warehouse_id"],
                    source="simulator.orders",
                    subject_id=row["order_id"],
                    payload={"sku_id": row["sku_id"], "quantity": int(row["quantity"])},
                )
            )
        return sorted(events, key=lambda event: event.timestamp)
