# ADR 003: Decompose Intelligence into Independent Engines with Explicit Fallbacks

## Status

Accepted

## Context

The product brief demanded eight independent intelligence engines, each with real model code and graceful fallback behavior under sparse data or missing production dependencies.

## Decision

Implement each engine in its own package under `warehouse_ai.engines`, pair primary models with explicit fallbacks, and keep orchestration in the control plane rather than inside the models.

## Consequences

- Each engine can be trained, tested, and eventually deployed independently.
- Fallbacks remain visible architectural choices instead of hidden exception handlers.
- The control plane can evolve into true microservices without a rewrite of the modeling code.

