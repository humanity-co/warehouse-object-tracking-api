import pandas as pd
import numpy as np
import torch
from pathlib import Path
from warehouse_ai.engines.pick_path.engine import PickPathOptimizationEngine

def train_routing():
    artifacts_dir = Path("/Users/devsmac/Documents/warehouse/artifacts")
    layout_path = artifacts_dir / "route_nodes.csv"
    
    if not layout_path.exists():
        print("❌ Layout file not found. Run generate_pick_layout.py first.")
        return

    # Load Layout
    route_nodes = pd.read_csv(layout_path)
    
    # Generate Synthetic Orders
    num_orders = 100
    order_data = []
    sku_ids = [f"SKU-{i:04d}" for i in range(100)]
    
    for i in range(num_orders):
        order_id = f"ORD-{i:06d}"
        # Each order has 3-8 lines
        num_lines = np.random.randint(3, 8)
        selected_skus = np.random.choice(sku_ids, num_lines, replace=False)
        for sku in selected_skus:
            order_data.append({
                "order_id": order_id,
                "sku_id": sku,
                "warehouse_id": "WH-01",
                "priority": np.random.randint(1, 4),
                "quantity": np.random.randint(1, 5)
            })
            
    order_lines = pd.DataFrame(order_data)
    
    # Initialize Engine and Train
    print("🚀 Initializing Pick Path Optimization Engine...")
    engine = PickPathOptimizationEngine(artifacts_dir)
    
    print(f"🧠 Training with {len(order_lines)} lines over 50 episodes...")
    result = engine.train(order_lines, route_nodes, episodes=50)
    
    print(f"✅ Training Complete!")
    print(f"Model saved at: {result.artifact_path}")
    print(f"Metrics: {result.metrics}")

if __name__ == "__main__":
    train_routing()
