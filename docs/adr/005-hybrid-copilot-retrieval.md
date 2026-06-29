# ADR 005: Use Hybrid Dense + Sparse Retrieval for the Copilot

## Status

Accepted

## Context

The copilot needed to answer both semantic and literal operational questions, including SKU-specific and event-specific lookups.

## Decision

Use TF-IDF + SVD dense embeddings, BM25-style sparse scoring, optional FAISS acceleration, and intent parsing over the combined knowledge base.

## Consequences

- Semantic retrieval works locally without depending on a hosted embedding service.
- Exact identifiers such as SKU IDs and event types remain highly retrievable.
- The architecture can upgrade to FAISS or Pinecone-backed embedding stores without changing the copilot contract.

