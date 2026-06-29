import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_dummy_data():
    base_path = "platform/data/dummy"
    os.makedirs(base_path, exist_ok=True)
    
    # 1. Demand Forecasting (1st)
    print("Generating Demand Data...")
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    skus = [f"SKU-{i:04d}" for i in range(5)]
    demand_data = []
    for date in dates:
        for sku in skus:
            demand_data.append({
                "timestamp": date,
                "sku_id": sku,
                "warehouse_id": "WH-01",
                "units": np.random.randint(10, 100),
                "promotion_flag": 0,
                "price_index": 1.0
            })
    pd.DataFrame(demand_data).to_csv(f"{base_path}/demand_history.csv", index=False)
    
    # 2. Inventory Optimization (2nd)
    print("Generating Inventory Data...")
    inventory_data = []
    for date in dates:
        for sku in skus:
            inventory_data.append({
                "timestamp": date,
                "warehouse_id": "WH-01",
                "sku_id": sku,
                "on_hand": np.random.randint(100, 500),
                "in_transit": np.random.randint(0, 50),
                "reserved": np.random.randint(0, 20),
                "reorder_point": 150
            })
    pd.DataFrame(inventory_data).to_csv(f"{base_path}/inventory_history.csv", index=False)
    
    # 4. Smart Slotting (4th)
    print("Generating Slotting Data...")
    slotting_data = []
    for i in range(20):
        slotting_data.append({
            "bin_id": f"BIN-{i:03d}",
            "warehouse_id": "WH-01",
            "zone_id": "ZONE-A",
            "x": i % 5,
            "y": i // 5,
            "velocity_rank": np.random.randint(1, 11),
            "sku_id": f"SKU-{i%5:04d}"
        })
    pd.DataFrame(slotting_data).to_csv(f"{base_path}/slotting_data.csv", index=False)
    
    # 6 & 7. Anomaly & Maintenance (6th & 7th)
    print("Generating Telemetry Data...")
    telemetry_data = []
    for i in range(200):
        ts = datetime.now() - timedelta(minutes=i*10)
        telemetry_data.append({
            "timestamp": ts,
            "equipment_id": "CONV-001",
            "vibration_rms": 0.5 + np.random.normal(0, 0.1),
            "motor_temp_c": 45 + np.random.normal(0, 2),
            "acoustic_db": 65 + np.random.normal(0, 5),
            "load_factor": 0.7
        })
    pd.DataFrame(telemetry_data).to_csv(f"{base_path}/telemetry_data.csv", index=False)
    
    # 8. AI Copilot (8th)
    print("Generating Copilot SOP...")
    sop_content = """
    WAREHOUSE STANDARD OPERATING PROCEDURES (SOP)
    
    1. SAFETY FIRST: All operators must wear high-visibility vests and steel-toed boots.
    2. INVENTORY: Stock counts must be performed every Friday at 4 PM.
    3. REORDERING: When SKU-0001 falls below 150 units, trigger a reorder alert.
    4. ANOMALIES: If conveyor vibration exceeds 0.8 RMS, shut down the line immediately.
    5. MAINTENANCE: All shuttle robots require a battery check every 24 hours.
    6. SLOTTING: High-velocity items must be stored in ZONE-A for faster picking.
    """
    with open(f"{base_path}/warehouse_sop.txt", "w") as f:
        f.write(sop_content)
    
    print(f"✅ All dummy datasets created in {base_path}")

if __name__ == "__main__":
    generate_dummy_data()
