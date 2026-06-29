# ADR 006: Build a Data-Dense Operator Console Instead of a CRUD Dashboard

## Status

Accepted

## Context

The product target was a real warehouse control tower, not an admin panel. Operators need live signal density, decision transparency, and a natural-language way to interrogate the system.

## Decision

Use a single-screen Next.js control tower with KPI tiles, live event feed, watchlists, and a copilot panel fed by the FastAPI control plane.

## Consequences

- The UI reflects the platform’s AI-centric operating model.
- Real-time events and explanations are first-class UX elements.
- The frontend remains thin because orchestration and reasoning stay server-side.
