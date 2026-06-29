# Architecture Overview

## Service topology

- `warehouse-api`: FastAPI control plane with JWT auth, OpenAPI, engine orchestration, and WebSocket event feed.
- `warehouse-web`: Next.js operator console consuming authenticated APIs plus the event socket.
- `postgres + timescaledb`: operational system of record and time-series store target.
- `redis`: hot-path cache and stream-compatible event buffer target.
- `kafka + schema registry`: event backbone and service contract boundary.
- `platform/src/warehouse_ai`: shared domain, simulator, MLOps, and engine packages.

## Execution model

1. Simulator produces synthetic operational state for local development and evaluation.
2. Each engine trains against coherent slices of that state rather than isolated toy datasets.
3. The API layer initializes engines lazily and exposes each capability through a dedicated route.
4. Events are appended to the in-memory stream abstraction, which mirrors how Redis Streams or Kafka topics would be consumed in production.
5. The frontend treats the API as a control tower, not a CRUD surface: high-density KPIs, live event flow, and copilot interactions dominate the UX.

## Engine boundaries

- Demand forecasting provides forecast distributions and drift flags.
- Inventory optimization consumes forecast shape plus inventory state to emit reorder and transfer recommendations.
- Vision emits validation, misplacement, and damage decisions from frame streams.
- Slotting and routing optimize warehouse geometry.
- Anomaly detection and predictive maintenance protect operating integrity.
- Copilot unifies logs, documents, and decisions into a natural-language interface.

## Production adaptation points

- Replace the in-memory feature store with Redis or Feast online serving.
- Replace the in-memory event stream with Kafka + schema registry and add consumer groups per engine.
- Replace the simulator stream with real WMS, PLC, vision, and supplier feeds.
- Promote engine artifacts through MLflow + a production registry instead of local files.

