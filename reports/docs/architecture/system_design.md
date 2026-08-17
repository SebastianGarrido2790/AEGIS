# System Design — Architectural Decision Record — AEGIS _(working title)_

**Actuarial Elasticity & Governance Intelligence System**
Author: Sebastián Garrido Arévalo | Date: August 13, 2026
**Status: Phase 1 Implemented.** Repository scaffold, namespaced package layout, Pydantic configuration schema, Great Expectations data contracts (regulatory and elasticity suites), DVC parallel DAGs, INV-8 module size enforcement, and unified CI workflow are fully implemented and validated.

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

### ADR-004: Package layout & build toolchain — Namespaced `src/aegis/` with Hatchling & Python 3.12

**Status:** Validated (Phase 1)

**Context:** Package layout and packaging configuration determine module import paths across the codebase (`from aegis.gateway import ...`) and how dependencies and builds are managed. Generic top-level package names risk collisions with third-party libraries.

**Decision:** Adopt a namespaced package layout rooted at `src/aegis/`, specify `hatchling` as the build backend in `pyproject.toml`, and set the Python requirement to `>=3.12,<3.13` with `3.12` pinned in `.python-version` (P1-D1).

**Rationale:** Namespacing under `src/aegis/` is the standard `uv`-recommended packaging pattern and prevents shadowing third-party modules. `hatchling` provides mature tooling, wide community documentation, and seamless integration with `uv`. Minor-version flooring (`>=3.12,<3.13`) combined with `uv.lock` provides both flexibility across patch releases and exact reproducibility.

**Consequences:** All internal code imports are prefixed with `aegis.` (e.g. `aegis.gateway`, `aegis.agents`).

### ADR-005: Configuration management — Domain-nested `params.yaml` with Pydantic validation

**Status:** Validated (Phase 1)

**Context:** The system requires a single source of truth for all tunable parameters across multi-tier ML, agentic orchestration, and governance components, while maintaining a strict zero-secrets policy and fail-loudly error handling.

**Decision:** Structure `params.yaml` into domain-nested sections mirroring the system architecture tiers (`gateway:`, `tier1_ml:`, `tier2_agents:`, `governance:`, `data_contracts:`, `dvc:`), validate all settings at load time against a strict Pydantic model (`aegis.config`), and enforce zero committed secrets (P1-D2).

**Rationale:** Domain nesting prevents key collisions between tiers and keeps configuration scannable. Pydantic validation guarantees that invalid or missing parameters fail fast at application startup rather than deep in runtime execution, bringing configuration under CI type checking and contract validation.

**Consequences:** Any addition or modification to `params.yaml` requires corresponding updates to the Pydantic configuration schemas in `src/aegis/config/`.

### ADR-006: Data contract architecture — File-based GX Core JSON suites with hand-crafted fixtures

**Status:** Validated (Phase 1)

**Context:** Per INV-3 and ADR-003, data contracts gate both elasticity training data and regulatory corpus ingestion. The contract format, execution mode, and CI verification strategy must be defined without introducing unnecessary external cloud dependencies.

**Decision:** Use open-source GX Core file-based expectation suites serialized as native JSON files in `data_contracts/`, verified during CI using minimal, hand-crafted valid and malformed fixtures in `data_contracts/fixtures/` (P1-D3).

**Rationale:** Avoids external cloud service dependencies (GX Cloud), aligning with the local-first zero-cost development posture (Charter §6, INV-10). Native JSON files provide declarative, cleanly version-controlled definitions. Hand-crafted fixtures verify specific failure modes (e.g., negative exposure, post-treatment leakage, missing chunk metadata) in milliseconds without pulling real dataset dependencies into Phase 1.

**Consequences:** Expectation suites are versioned directly in git; custom expectation functions (e.g., elasticity leakage checks) are integrated through GX Core compatible definitions.

### ADR-007: DVC pipeline architecture — Local filesystem remote with fine-grained DAG stages

**Status:** Validated (Phase 1)

**Context:** DVC pipeline orchestration must enforce Great Expectations gates before versioning datasets or artifacts, following the local-first zero-cost development model.

**Decision:** Configure DVC with a local filesystem cache/remote and define fine-grained pipeline stages (`ingest`, `validate_gx`, `version`) in `dvc.yaml`, making Great Expectations validation an explicit node in the DVC DAG (P1-D4).

**Rationale:** Local filesystem remotes eliminate cloud configuration complexity during local development while adhering to INV-10. Fine-grained stages allow DVC to cache intermediate steps and only rerun validation when raw inputs change, making data contract enforcement explicit and inspectable in the DAG.

**Consequences:** Pipeline reproduction (`dvc repro`) explicitly verifies GX validation stages before producing versioned outputs.

### ADR-008: CI/CD scaffold & invariant enforcement — Unified GitHub Actions workflow with day-one gates

**Status:** Validated (Phase 1)

**Context:** Continuous integration must enforce project standards, code quality, test coverage, and architectural invariants (notably INV-8 module line ceiling and INV-3 data contracts) from the very first commit.

**Decision:** Implement a single unified GitHub Actions workflow (`.github/workflows/ci.yml`) executing sequential gates: linting (`ruff`), type checking (`pyright`), module size enforcement (`scripts/check_module_size.py` for INV-8), GX fixture validation gates (INV-3), and automated test execution (`pytest` minimal scaffold) (P1-D5).

**Rationale:** A single workflow provides a clear pass/fail status and minimizes CI maintenance overhead. Enforcing the 1,000-line limit (INV-8) and running a minimal pytest scaffold from Phase 1 guarantees that no unverified or non-compliant code enters the repository at any stage.

**Consequences:** All PRs and commits must pass all quality, invariant, and contract gates before merging.

## 5. Open Implementation Notes

- The specific persistence layer for the Tier 3 audit log is deferred to Phase 7 and not yet decided.
- The minimal HITL review interface (Phase 7) is scoped as an endpoint, not a full dashboard, consistent with PRD §12 (out of scope: full production UI).
- This document will be updated at the close of each phase in the Technical Roadmap to reflect the actual implemented state.

## 6. Update Protocol

At the close of each phase in `../references/technical_roadmap.md`:

1. Update "Current Implementation Status" to reflect what was actually built.
2. Mark each relevant ADR's status as **Validated** (implementation matched the decision) or **Amended** (implementation diverged, the amendment must be logged as a new dated entry under the original ADR, not a silent edit).
3. Add new ADRs for any architecturally significant decision made during implementation that wasn't anticipated in Phase 0.

