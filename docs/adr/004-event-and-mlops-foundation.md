# ADR 004: Prefer Event Streams and Lightweight Local MLOps Adapters

## Status

Accepted

## Context

The platform needed Kafka/Redis/MLflow-style architecture, but the local implementation also had to run in a blank workspace without standing up the full ecosystem first.

## Decision

Provide local abstractions for event streaming, experiment tracking, registry promotion, and drift detection while keeping the production integration shape visible in code and deployment assets.

## Consequences

- Developers get a working platform immediately.
- The codebase preserves operational seams for Redis Streams, Kafka, and MLflow.
- Some local components are intentionally lighter than their production counterparts, but they are functional rather than placeholder-only.

