# ADR 001: Use a Simulator-First Development Substrate

## Status

Accepted

## Context

The repository started empty, but the product mandate required end-to-end AI services, not isolated notebooks. A coherent synthetic environment was needed so every engine could train, infer, and explain decisions against the same world model.

## Decision

Build a `SyntheticWarehouseGenerator` that emits demand, inventory, telemetry, routing topology, documents, and event history from shared latent assumptions.

## Consequences

- Local development can exercise all engines without external dependencies.
- Cross-engine reasoning becomes possible because the data slices are internally consistent.
- Production replacements are explicit integration points rather than hidden TODOs.

