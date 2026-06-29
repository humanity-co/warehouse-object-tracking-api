# ADR 002: Adopt FastAPI as the Control Plane

## Status

Accepted

## Context

The platform needed typed APIs, async support, OpenAPI docs, and a clean way to expose many AI services through a single authenticated edge.

## Decision

Use FastAPI with manual HMAC JWT issuance/verification, versioned `/api/v1/*` routes, and a WebSocket event endpoint.

## Consequences

- Typed request/response contracts are available immediately.
- The service layer remains lightweight and easy to deploy in containers.
- The auth layer is self-contained for local development and replaceable with enterprise SSO later.

