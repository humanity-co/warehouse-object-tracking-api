import time
import requests
import random
import json
from datetime import datetime
from uuid import uuid4

API_BASE = "http://127.0.0.1:8000"
USERNAME = "admin"
PASSWORD = "warehouse"

import pandas as pd
from pathlib import Path

class WarehouseSimulator:
    def __init__(self):
        self.token = self._get_token()
        self.skus = [f"SKU-{i:04d}" for i in range(12)]
        self.warehouses = ["WH-01", "WH-02", "WH-03"]
        self.equipment = ["CNV-01", "CNV-02", "CNV-03", "RBT-01", "RBT-02"]
        
        # Load Layout
        layout_path = Path("/Users/devsmac/Documents/warehouse/artifacts/route_nodes.csv")
        if layout_path.exists():
            self.route_nodes = pd.read_csv(layout_path)
            self.node_ids = self.route_nodes["node_id"].tolist()
        else:
            self.node_ids = [f"NODE-{i}" for i in range(20)]

    def _get_token(self):
        max_retries = 10
        for i in range(max_retries):
            try:
                resp = requests.post(f"{API_BASE}/api/v1/auth/token", json={"username": USERNAME, "password": PASSWORD})
                resp.raise_for_status()
                return resp.json()["access_token"]
            except Exception as e:
                if i < max_retries - 1:
                    print(f"⌛ Waiting for backend... (Attempt {i+1}/{max_retries})")
                    time.sleep(3)
                else:
                    raise e

    def push(self, event_type, subject_id, warehouse_id, payload):
        event = {
            "event_id": uuid4().hex,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "warehouse_id": warehouse_id,
            "source": "digital_twin_sim",
            "subject_id": subject_id,
            "payload": payload,
            "trace_id": uuid4().hex
        }
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            requests.post(f"{API_BASE}/api/v1/simulation/push", json=event, headers=headers)
        except Exception as e:
            print(f"Failed to push event: {e}")

    def run(self):
        print("🚀 Starting Live Warehouse Digital Twin Simulator...")
        print("Feeding data to all 8 engines at http://localhost:8000")
        
        try:
            while True:
                # 1. Demand & Inventory Events
                sku = random.choice(self.skus)
                wh = random.choice(self.warehouses)
                units = random.randint(1, 10)
                self.push("demand_observed", sku, wh, {"units": units, "price": 19.99})
                self.push("inventory_updated", sku, wh, {"on_hand_delta": -units, "reason": "sale"})
                
                # 2. Vision Events
                self.push("vision_analyzed", sku, wh, {
                    "camera_id": f"CAM-{wh}-01",
                    "bbox": [random.randint(0, 50), random.randint(0, 50), random.randint(60, 100), random.randint(60, 100)],
                    "confidence": 0.98,
                    "status": "Healthy" if random.random() > 0.05 else "Damaged"
                })
                
                # 3. Telemetry / Anomaly Events
                eq = random.choice(self.equipment)
                vibration = 0.3 + (random.random() * 0.1)
                if random.random() > 0.98: # Spike!
                    vibration += 1.2
                self.push("anomaly_detected" if vibration > 1.0 else "maintenance_scored", eq, wh, {
                    "vibration_rms": vibration,
                    "motor_temp": 45 + (random.random() * 5),
                    "load_factor": 0.7
                })

                # 4. Slotting & Routing Events
                if random.random() > 0.9:
                    self.push("slotting_optimized", "ZONE-A", wh, {"relocations": 1, "sku_id": sku})
                    
                    # Generate a realistic picking path
                    num_picks = random.randint(3, 6)
                    picks = random.sample(self.node_ids, num_picks)
                    self.push("route_planned", f"ORDER-{random.randint(1000, 9999)}", wh, {
                        "pick_nodes": picks,
                        "path_length": random.randint(50, 200),
                        "estimated_time_mins": random.randint(5, 15)
                    })

                # 5. Copilot Signal
                if random.random() > 0.95:
                    self.push("copilot_interaction", "USER-42", wh, {"query": "Status update", "intent": "general_info"})

                time.sleep(2) # Pulse every 2 seconds
        except KeyboardInterrupt:
            print("\n🛑 Simulator stopped.")

if __name__ == "__main__":
    sim = WarehouseSimulator()
    sim.run()
