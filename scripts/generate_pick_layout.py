import pandas as pd
import numpy as np
from pathlib import Path

def generate_layout(output_path: str):
    nodes = []
    
    # Depot (Start/End point)
    nodes.append({
        "warehouse_id": "WH-01",
        "node_id": "NODE-DD-00-Z",
        "x": 0.0,
        "y": 0.0,
        "zone_id": "ZONE-A",
        "pick_face": False
    })

    # Aisles and Racks
    num_aisles = 5
    racks_per_aisle = 10
    aisle_spacing = 10.0
    rack_spacing = 5.0
    
    for a in range(num_aisles):
        x = (a + 1) * aisle_spacing
        for r in range(racks_per_aisle):
            y = (r + 1) * rack_spacing
            # Left side of aisle
            nodes.append({
                "warehouse_id": "WH-01",
                "node_id": f"NODE-{a:02d}-{r:02d}-L",
                "x": x - 2.0,
                "y": y,
                "zone_id": f"ZONE-{chr(65+a)}",
                "pick_face": True
            })
            # Right side of aisle
            nodes.append({
                "warehouse_id": "WH-01",
                "node_id": f"NODE-{a:02d}-{r:02d}-R",
                "x": x + 2.0,
                "y": y,
                "zone_id": f"ZONE-{chr(65+a)}",
                "pick_face": True
            })

    df = pd.DataFrame(nodes)
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} nodes for warehouse layout at {output_path}")

if __name__ == "__main__":
    out = Path("/Users/devsmac/Documents/warehouse/artifacts/route_nodes.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_layout(str(out))
