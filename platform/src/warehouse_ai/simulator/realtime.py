import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import httpx
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from warehouse_ai.core.events import EventType, WarehouseEvent
from warehouse_ai.core.schemas import EquipmentType, ZoneType

# --- Config ---
class Config:
    API_BASE_URL = "http://127.0.0.1:8001/api/v1"
    REAL_TIME_FACTOR = 3600  # 1s real = 1h sim
    INTERVAL_SECONDS = 1.0
    AUTH_CREDENTIALS = {"username": "admin", "password": "warehouse"}
    LOG_BUFFER_FILE = "buffer.txt"

# --- State ---
class SimulatorState:
    def __init__(self):
        self.skus = [f"SKU-{i:04d}" for i in range(1, 21)]  # 20 SKUs
        self.warehouses = ["WH-01", "WH-02", "WH-03"]
        self.zones = ["ZONE-A", "ZONE-B", "ZONE-C"]
        self.cameras = [f"CAM-{i:02d}" for i in range(1, 6)]
        self.equipment = [
            {"id": "CONV-01", "type": EquipmentType.conveyor, "vibration": 0.3, "temp": 45.0, "load": 0.5, "failure_prob": 0.01},
            {"id": "FORK-01", "type": EquipmentType.forklift, "vibration": 0.5, "temp": 50.0, "load": 0.3, "failure_prob": 0.02},
            {"id": "ROBOT-01", "type": EquipmentType.shuttle, "vibration": 0.2, "temp": 40.0, "load": 0.7, "failure_prob": 0.005},
        ]
        self.inventory = {sku: random.randint(50, 200) for sku in self.skus}
        self.velocity_ranks: Dict[str, str] = {sku: "medium" for sku in self.skus}
        self.orders = []
        self.sim_time = datetime.now()
        self.events_pushed = 0
        self.last_sync_real_time = time.time()

    def update_sim_time(self):
        now = time.time()
        delta = now - self.last_sync_real_time
        self.sim_time += timedelta(seconds=delta * Config.REAL_TIME_FACTOR)
        self.last_sync_real_time = now

# --- AI Engine Logics ---

class SimulationEngine:
    def __init__(self, state: SimulatorState, console: Console):
        self.state = state
        self.console = console
        self.client = httpx.AsyncClient(base_url=Config.API_BASE_URL, timeout=10.0)
        self.token = None

    async def login(self):
        try:
            resp = await self.client.post("/auth/token", json=Config.AUTH_CREDENTIALS)
            resp.raise_for_status()
            self.token = resp.json()["access_token"]
            self.client.headers["Authorization"] = f"Bearer {self.token}"
        except Exception as e:
            self.console.print(f"[bold red]Failed to login to API: {e}[/bold red]")

    async def push_event(self, event_type: EventType, warehouse_id: str, subject_id: str, payload: Dict):
        event = WarehouseEvent(
            event_type=event_type,
            timestamp=self.state.sim_time,
            warehouse_id=warehouse_id,
            source="realtime_simulator",
            subject_id=subject_id,
            payload=payload
        )
        try:
            # We use a dict for the payload to ensure it's JSON serializable
            event_dict = event.dict()
            event_dict["timestamp"] = event_dict["timestamp"].isoformat()
            resp = await self.client.post("/simulation/push", json=event_dict)
            if resp.status_code == 200:
                self.state.events_pushed += 1
        except Exception as e:
            pass # Silent failure for simulation speed

    def log_to_buffer(self, message: str):
        with open(Config.LOG_BUFFER_FILE, "a") as f:
            f.write(f"[{self.state.sim_time.isoformat()}] {message}\n")

    # 1. Demand Forecasting
    async def simulate_demand(self):
        hour = self.state.sim_time.hour
        multiplier = 5.0 if 12 <= hour <= 15 else 1.0
        
        for sku in self.state.skus:
            # Poisson distribution: lambda is base_demand * multiplier
            lambda_val = (0.2 if "COLD_DRINK" in sku else 0.1) * multiplier
            quantity = random.choices([0, 1, 2, 3], weights=[1-lambda_val, lambda_val*0.7, lambda_val*0.2, lambda_val*0.1])[0]
            
            if quantity > 0:
                wh = random.choice(self.state.warehouses)
                await self.push_event(EventType.demand_observed, wh, sku, {"units": quantity, "is_seasonal": multiplier > 1})
                self.state.inventory[sku] = max(0, self.state.inventory[sku] - quantity)
                self.state.orders.append({"sku": sku, "wh": wh, "qty": quantity, "time": self.state.sim_time})

    # 2. Inventory Optimization
    async def simulate_inventory(self):
        for sku, stock in self.state.inventory.items():
            if stock < 20: # 20% threshold (assuming 100 is base)
                wh = random.choice(self.state.warehouses)
                lead_time = random.randint(1, 5)
                await self.push_event(EventType.inventory_updated, wh, sku, {"on_hand": stock, "status": "REORDER_TRIGGERED", "lead_time_days": lead_time})
                # Simulate restock
                self.state.inventory[sku] += 50 
                self.log_to_buffer(f"Restock arrived for {sku}")

    # 3. Computer Vision
    async def simulate_vision(self):
        cam = random.choice(self.state.cameras)
        zone = random.choice(self.state.zones)
        sku = random.choice(self.state.skus)
        damage_score = random.uniform(0.8, 1.0) if random.random() < 0.05 else random.uniform(0, 0.2)
        bbox = [random.randint(0, 100), random.randint(0, 100), random.randint(100, 200), random.randint(100, 200)]
        
        await self.push_event(EventType.vision_analyzed, "WH-01", cam, {
            "zone_id": zone,
            "sku_id": sku,
            "damage_score": damage_score,
            "bbox": bbox
        })

    # 4. Smart Slotting
    async def simulate_slotting(self):
        # Every 30s sim time (roughly)
        sku = random.choice(self.state.skus)
        new_rank = random.choice(["high", "medium", "low"])
        if new_rank == "high" and self.state.velocity_ranks[sku] != "high":
            await self.push_event(EventType.slotting_optimized, "WH-01", sku, {
                "old_velocity": self.state.velocity_ranks[sku],
                "new_velocity": new_rank,
                "recommendation": "Move to Front Zone"
            })
        self.state.velocity_ranks[sku] = new_rank

    # 5. Pick Path Optimization
    async def simulate_routing(self):
        if len(self.state.orders) >= 3:
            bundle = self.state.orders[:random.randint(3, 5)]
            self.state.orders = self.state.orders[len(bundle):]
            sku_ids = [o["sku"] for o in bundle]
            bins = [f"BIN-{random.randint(1, 100)}" for _ in bundle]
            await self.push_event(EventType.route_planned, "WH-01", "PICKLIST-001", {
                "skus": sku_ids,
                "bins": bins,
                "grid_size": [10, 10]
            })

    # 6. Anomaly Detection & 7. Predictive Maintenance
    async def simulate_telemetry(self, chaos: bool = False):
        for eq in self.state.equipment:
            eq["failure_prob"] += 0.001
            eq["temp"] += eq["load"] * 0.5
            
            vibration = eq["vibration"]
            if chaos and eq["id"] == "CONV-01":
                vibration = 1.5 # Spike
                self.log_to_buffer(f"Conveyor {eq['id']} sudden spike in vibration!")
            else:
                vibration += random.uniform(-0.05, 0.05)
            
            await self.push_event(EventType.anomaly_detected, "WH-01", eq["id"], {
                "vibration_rms": vibration,
                "motor_temp_c": eq["temp"],
                "failure_probability": eq["failure_prob"]
            })

    # 8. AI Copilot
    async def simulate_copilot_feed(self):
        events = [
            "Shift Change: Operator A logged in",
            f"Restock arrived for {random.choice(self.state.skus)}",
            "Conveyor 4 rebooted",
            "Zone B congestion cleared"
        ]
        msg = random.choice(events)
        self.log_to_buffer(msg)

# --- UI ---

def create_dashboard(state: SimulatorState):
    grid = Table.grid(expand=True)
    
    # Header
    header = Panel(
        Text.from_markup(f"[bold cyan]WAREHOUSE INTELLIGENCE PLATFORM SIMULATOR[/bold cyan]\n[dim]Sim Time: {state.sim_time.strftime('%Y-%m-%d %H:%M:%S')} | Factor: {Config.REAL_TIME_FACTOR}x[/dim]"),
        style="blue"
    )
    grid.add_row(header)
    
    # Stats Table
    stats = Table(show_header=True, header_style="bold magenta", expand=True)
    stats.add_column("Entity", style="dim")
    stats.add_column("Status", justify="right")
    stats.add_row("Events Pushed", str(state.events_pushed))
    stats.add_row("Active Warehouses", str(len(state.warehouses)))
    stats.add_row("SKUs Managed", str(len(state.skus)))
    stats.add_row("Pending Orders", str(len(state.orders)))
    
    grid.add_row(Panel(stats, title="[bold]Real-Time Metrics[/bold]"))
    
    # Progress
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    progress.add_task("[green]Demand Engine", total=100, completed=random.randint(40, 90))
    progress.add_task("[yellow]Vision Engine", total=100, completed=random.randint(20, 70))
    
    grid.add_row(Panel(progress, title="Engine Pulsing"))
    
    return grid

async def main():
    state = SimulatorState()
    console = Console()
    engine = SimulationEngine(state, console)
    
    # Create empty buffer
    with open(Config.LOG_BUFFER_FILE, "w") as f:
        f.write("--- Simulator Log Started ---\n")
    
    await engine.login()
    
    last_vision_time = 0
    last_slotting_time = 0
    last_chaos_time = time.time()
    chaos_active = False
    
    with Live(create_dashboard(state), refresh_per_second=4) as live:
        while True:
            state.update_sim_time()
            now = time.time()
            
            # 1. Demand
            await engine.simulate_demand()
            
            # 2. Inventory
            await engine.simulate_inventory()
            
            # 3. Vision (every 5s real)
            if now - last_vision_time > 5:
                await engine.simulate_vision()
                last_vision_time = now
            
            # 4. Slotting (every 30s sim)
            # Roughly every few seconds real time if factor is 3600
            if now - last_slotting_time > 1:
                await engine.simulate_slotting()
                last_slotting_time = now
                
            # 5. Routing
            await engine.simulate_routing()
            
            # 6 & 7 Telemetry & Maintenance
            # Chaos trigger every 2 mins
            if now - last_chaos_time > 120:
                chaos_active = True
                if now - last_chaos_time > 125: # 5s chaos
                    chaos_active = False
                    last_chaos_time = now
            
            await engine.simulate_telemetry(chaos=chaos_active)
            
            # 8. Copilot
            if random.random() < 0.1:
                await engine.simulate_copilot_feed()
            
            live.update(create_dashboard(state))
            await asyncio.sleep(Config.INTERVAL_SECONDS)

if __name__ == "__main__":
    asyncio.run(main())
