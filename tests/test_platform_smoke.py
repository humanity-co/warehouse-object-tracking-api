import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "platform" / "src"))

from warehouse_ai.api.control_plane import WarehouseControlPlane
from warehouse_ai.config.settings import Settings
from warehouse_ai.simulator.generator import SyntheticWarehouseGenerator


def test_simulator_generates_coherent_bundle():
    bundle = SyntheticWarehouseGenerator(seed=21).generate(sku_count=4, warehouse_count=2, days=40)
    assert bundle.sku_catalog["sku_id"].nunique() == 4
    assert bundle.warehouses["warehouse_id"].nunique() == 2
    assert len(bundle.events) > 0


def test_control_plane_bootstrap_and_cold_start_forecast():
    settings = Settings(
        project_root=Path("/Users/devsmac/Documents/warehouse"),
        artifacts_dir=Path("/Users/devsmac/Documents/warehouse/artifacts/test"),
        runs_dir=Path("/Users/devsmac/Documents/warehouse/runs/test"),
    )
    control_plane = WarehouseControlPlane(settings)
    summary = control_plane.bootstrap(seed=3, sku_count=4, warehouse_count=2, days=80, train=False)
    assert summary["sku_count"] == 4
    forecast = control_plane.forecast("SKU-0000", "WH-01")
    assert "horizons" in forecast
    assert forecast["model_name"] in {"similarity_transfer", "attention_bilstm", "temporal_fusion_transformer"}
