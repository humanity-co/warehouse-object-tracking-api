# End-to-End Data Flow

## Raw signal origins

- Demand events: generated from synthetic SKU-by-warehouse daily order curves.
- Inventory events: on-hand, in-transit, reserved, damaged, and reorder-point snapshots.
- IoT telemetry: vibration, temperature, acoustics, cycles, and load factor per equipment unit.
- Route topology: warehouse nodes and connectivity graph.
- Copilot corpus: generated operational summaries plus serialized event history.

## Processing path

1. `SyntheticWarehouseGenerator.generate()` builds a coherent world state and event ledger.
2. `DataQualityGate` validates incoming frames with Pydantic and optional Great Expectations checks.
3. `InMemoryFeatureStore.materialize()` makes entity-centric feature windows available to engines.
4. Training jobs transform raw slices into engine-specific tensors or tabular matrices.
5. Trained engines persist artifacts to `artifacts/` and register promotion metadata.
6. The control plane invokes engines on demand and serializes outputs into API responses.
7. The event stream exposes recent events to the frontend and any downstream automation consumers.

## Example: reorder explanation

1. Demand history for `SKU-0001` is featurized with lags, rolling stats, Fourier seasonality, and future-known inputs.
2. The forecasting engine produces `P10/P50/P90` demand estimates and a drift signal.
3. The inventory engine combines current stock, forecast volatility, capacity, and supplier reliability into an action.
4. The action explanation and the source evidence are available through the copilot retrieval layer.
5. The frontend renders the resulting action and evidence in the dashboard and copilot panel.

## Example: equipment risk loop

1. Telemetry sequences enter the maintenance engine.
2. The TCN produces RUL and 72h failure probability.
3. The anomaly engine scores the same telemetry in parallel for operational deviation.
4. The routing engine can use the degraded-zone output to reroute future pick waves.

