import pandas as pd
import numpy as np
from pathlib import Path

def generate_inventory_mapping():
    artifacts_dir = Path("/Users/devsmac/Documents/warehouse/artifacts")
    layout_path = artifacts_dir / "route_nodes.csv"
    
    if not layout_path.exists():
        print("❌ Layout file not found. Run generate_pick_layout.py first.")
        return

    # Load Layout
    route_nodes = pd.read_csv(layout_path)
    pickable_nodes = route_nodes[route_nodes["pick_face"] == True]["node_id"].tolist()
    
    # Generate 50 SKUs
    num_skus = 50
    inventory = []
    
    for i in range(num_skus):
        sku_id = f"SKU-{i:04d}"
        # Assign to a random pick face
        node_id = pickable_nodes[i % len(pickable_nodes)]
        inventory.append({
            "sku_id": sku_id,
            "node_id": node_id,
            "quantity": np.random.randint(5, 50),
            "warehouse_id": "WH-01"
        })
        
    df = pd.DataFrame(inventory)
    out_path = artifacts_dir / "inventory_mapping.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Generated inventory mapping for {len(df)} SKUs at {out_path}")

if __name__ == "__main__":
    generate_inventory_mapping()
