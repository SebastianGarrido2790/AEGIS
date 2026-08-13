# System Design — Architectural Decision Record — AEGIS _(working title)_

**Actuarial Elasticity & Governance Intelligence System**
Author: Sebastián Garrido Arévalo | Date: August 13, 2026
**Status: Phase 0 — Planned architecture, not yet implemented.** This document reflects the target design at the close of Phase 0 and will be updated at the close of every subsequent phase to reflect the actual implemented state.

---

## 1. Architecture Overview

```
                    ┌──────────────────────────────┐
                    │   FastAPI Serving Layer       │
                    │ (Underwriting/Pricing Client)│
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │  LangGraph Orchestrator        │
                    │  (Tier 2 — Agentic, Brain)     │
                    │                                │
                    │  Pricing Strategy Agent         │
                    │        │                       │
                    │        ▼                       │
                    │  Regulatory Compliance Agent    │◄──── Redis Stack
                    │        │        (RAG, RedisVL)  │      (RegVL/HNSW index)
                    │        ▼                       │
                    │  Revenue/Loss-Ratio Impact Agent│
                    └───────────────┬────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                ┌──────────────────┐  ┌──────────────────────┐
                │   LLM Gateway      │  │  Tier 1 — Deterministic│
                │   (LiteLLM)        │  │  ML (Brawn)            │
                │                    │  │  GLM → Causal Elasticity│
                │  Fallback / Router │  │  → Contextual Bandit    │
                │  Cache (Redis      │  └───────────┬────────────┘
                │  Stack, shared)    │              │
                │  Guardrails        │              ▼
                │  Cost/Trace Export │      MLflow + DVC (+ GX gates)
                │  Cost/Trace Export │
                └────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │  Tier 3 — Governance            │
                    │  HITL Gate │ Deterministic       │
                    │  Fallback │ Structured Audit Log │
                    └──────────────────────────────┘
```

## 2. Components (Planned)

| Component                      | Responsibility                                                                                                              | Key technology                                      |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Serving layer                  | Exposes the recommendation flow to an underwriting/pricing client                                                           | FastAPI, Docker                                     |
| Tier 1 — Deterministic ML      | GLM baseline, causal elasticity model, contextual bandit                                                                    | scikit-learn, EconML/DoWhy, MLflow, DVC             |
| Data contracts                 | CI-blocking validation of both data intake paths                                                                            | Great Expectations                                  |
| LLM Gateway                    | Single enforcement boundary for every provider call: abstraction, fallback, routing, caching, guardrails, cost/trace export | LiteLLM (in-process)                                |
| Regulatory RAG index           | Backing store for the Compliance Agent's retrieval and the Gateway's semantic cache                                         | Redis Stack (RedisVL/HNSW)                          |
| Tier 2 — Agentic orchestration | Pricing Strategy, Regulatory Compliance, Revenue/Loss-Ratio Impact agents on shared state                                   | LangGraph, Pydantic                                 |
| Tier 3 — Governance            | HITL escalation, deterministic fallback, structured audit log                                                               | Custom, backed by a persistence layer (TBD Phase 7) |
| Observability                  | Tracing, cost, drift, and evaluation                                                                                        | OTel, LLM-as-judge regression suite                 |
| CI/CD                          | Reproducibility, coverage, and data-contract gating                                                                         | GitHub Actions                                      |

## 3. Data Flow (Planned)

1. A segment/policy is submitted to the FastAPI layer.
2. Tier 1 returns a causal elasticity estimate and a bandit-bounded proposed adjustment.
3. The LangGraph orchestrator invokes the Pricing Strategy Agent, which formats the proposal.
4. The Regulatory Compliance Agent retrieves grounding evidence from the Redis Stack index (via the Gateway) and returns a pass/flag verdict.
5. The Revenue/Loss-Ratio Impact Agent projects the financial effect.
6. The Governance layer evaluates the combined output: within-bounds-and-compliant proposals are marked ready for routine underwriter sign-off; anything else is escalated with full state attached.
7. A structured audit record is persisted regardless of outcome.

## 4. Architectural Decision Records

### ADR-001: LLM Gateway — LiteLLM (in-process) over a standalone proxy

**Context:** All production agent/RAG LLM calls route through a Gateway enforcing six capabilities (unified access, resiliency, routing, caching, guardrails, cost observability), and permits either a standalone proxy (e.g., MLflow AI Gateway) or an in-process library (e.g., LiteLLM) as the implementation topology.

**Decision:** Use LiteLLM as an in-process library.

**Rationale:** No additional service to deploy or operate, which matters directly for a solo-practitioner scope (Charter §9 risk register). LiteLLM is already a stack dependency for the `@observe` tracing integration, so this extends an existing dependency rather than introducing a new one.

**Consequences:** The Gateway's resiliency and guardrail logic runs in-process with the orchestrator rather than as an independently scalable service; acceptable for the project's non-production scope (PRD §13 — no live deployment in v1).

### ADR-002: Vector store — Redis Stack (RedisVL) over ChromaDB

> **Core Reality:** Enterprise RAG users ask variants of the same core questions repeatedly. Without an intermediate cache layer, every variant executes a full pipeline: cross-network document retrieval, embedding generation, reranking, prompt construction, and an LLM API call. Redis eliminates the majority of these redundant operations by serving as both the primary vector database and an in-memory semantic cache within the same memory space.

**Context:** The Compliance Agent requires a RAG index over the regulatory corpus. ChromaDB was the initial default (embedded, no server to run); however, ADR-001's Gateway mandates a two-tier Redis-backed prompt cache.

**Decision:** Use Redis Stack (RedisVL/HNSW) as the regulatory RAG index, sharing the same instance that backs the Gateway's prompt cache, rather than introducing ChromaDB as a second, separate vector store.

**Rationale:** One platform serving both roles directly reduces the infrastructure footprint and the scope risk named in Charter §9, and mirrors a pattern already proven in prior work (Medical AI Assistant's Redis Stack HNSW implementation) rather than introducing an unproven combination.

**Consequences:** The regulatory corpus, being small and infrequently updated, is somewhat over-served by Redis Stack's capacity relative to its actual scale — an accepted tradeoff in exchange for infrastructure consolidation.

### ADR-003: Data contracts — Great Expectations paired with DVC, CI-blocking

**Context:** DVC versions data and pipeline artifacts but does not validate them; without an independent validation gate, invalid data can be faithfully version-controlled ("garbage in, garbage out").

**Decision:** Two Great Expectations suites — one for elasticity training data (schema, value ranges, post-treatment leakage check), one for regulatory corpus ingestion (non-empty chunks, required metadata, duplicate detection) — both enforced as CI-blocking gates ahead of DVC versioning, not as advisory warnings.

**Rationale:** The regulatory corpus suite in particular is a direct upstream mitigation for the system's highest-severity failure mode (RAG-grounded compliance hallucination, Charter §9) — malformed or duplicated chunks are rejected before indexing, not caught after the fact.

**Consequences:** Adds a validation step ahead of every data pipeline run; accepted as necessary given the severity of the failure mode it mitigates.

## 5. Open Implementation Notes

- The specific persistence layer for the Tier 3 audit log is deferred to Phase 7 and not yet decided.
- The minimal HITL review interface (Phase 7) is scoped as an endpoint, not a full dashboard, consistent with PRD §12 (out of scope: full production UI).
- This document will be updated at the close of each phase in the Technical Roadmap to reflect the actual implemented state.

## 6. Update Protocol

At the close of each phase in `../references/technical_roadmap.md`:

1. Update "Current Implementation Status" to reflect what was actually built.
2. Mark each relevant ADR's status as **Validated** (implementation matched the decision) or **Amended** (implementation diverged, the amendment must be logged as a new dated entry under the original ADR, not a silent edit).
3. Add new ADRs for any architecturally significant decision made during implementation that wasn't anticipated in Phase 0.
