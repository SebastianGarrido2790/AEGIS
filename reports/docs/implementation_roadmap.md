# Implementation Roadmap — AEGIS

Non-authoritative, high-level delivery map. Its only job: major capabilities in dependency order, and the evidence unlocking each next stage. No technical specifications, no task lists, no decisions — those live in feature specs, phase plans, and `system_design.md`.

Durations are best-effort and flexible; no phase begins before the prior phase's exit criteria are demonstrated, not merely coded.

---

## Phase 0 — Planning & Design _(complete)_

Goal:
Establish the strategic and product foundation before any implementation.

Capabilities:
- scoping document, PRD, user story, this roadmap, system design ADRs, challenges runbook
- LiteLLM Gateway, Redis Stack, GX+DVC data-contract decisions confirmed across every document

Exit criteria:
- all Phase 0 artifacts complete and internally consistent

Dependencies:
None.

---

## Phase 1 — Scaffolding & Data Contracts _(complete)_

Goal:
Stand up the repo skeleton and CI-blocking data contracts before any modeling — no data enters the pipeline unvalidated.

Capabilities:
- repository structure, `uv` environment, `params.yaml` schema
- Great Expectations suites for elasticity training data and regulatory corpus ingestion
- DVC pipeline skeleton wired to both GX gates; CI skeleton

Exit criteria:
- CI green on the scaffold
- each GX suite demonstrably fails a deliberately malformed fixture and passes a valid one

Dependencies:
Phase 0.

---

## Phase 2 — Tier 1: Deterministic ML Baseline _(reset — pending EDA-first redo)_

Goal:
Establish the GLM actuarial baseline and the causal elasticity model, with a first showcase slice — preceded this time by exploratory data analysis of the raw data before any training-pipeline code is written.

Capabilities:
- exposure/frequency/severity feature engineering
- GLM baseline; `CausalForestDML` elasticity estimator with confounding sensitivity analysis
- MLflow tracking and registry; showcase slice exposing elasticity output for preset scenarios

Exit criteria:
- causal estimator recovers the known synthetic treatment effect with defensible per-segment and rank-based confidence intervals (`scoping_doc.md` §0, §2)
- showcase slice renders correctly, clearly labeled demo rather than production view

Dependencies:
Phase 1 (data contracts).

---

## Phase 3 — Contextual Bandit Exploration Engine

Goal:
Wrap the elasticity output in a bounded, live-exploration mechanism.

Capabilities:
- `params.yaml` exploration-corridor bounds
- Thompson Sampling over the bounded corridor
- synthetic policy/segment stream simulator; regret backtest vs. static-price baseline

Exit criteria:
- lower cumulative regret than the static-price baseline on the synthetic stream

Dependencies:
Phase 2.

---

## Phase 4 — LLM Gateway Integration

Goal:
Stand up the in-process LiteLLM Gateway and Redis Stack before any agent is built, so every agent develops against the Gateway abstraction from day one.

Capabilities:
- unified model-string abstraction, provider fallback chain, load-balancing strategy in `params.yaml`
- Redis Stack service shared by the Gateway cache and the later RAG index
- pre-flight guardrail callbacks; cost/trace OTel emission

Exit criteria:
- a simulated provider failure triggers the fallback chain
- a known injection pattern is blocked at the Gateway boundary in a unit test

Dependencies:
Phase 1. Independent of Phases 2–3; sequenced for solo-practitioner bandwidth.

---

## Phase 5 — Regulatory RAG Foundation

Goal:
Prove retrieval quality independently before wrapping it in agent logic.

Capabilities:
- GX-gated corpus ingestion, chunking, RedisVL/HNSW indexing
- retrieval-quality harness: the four-metric diagnostic matrix (ADR-010) on curated compliant/violating scenarios

Exit criteria:
- harness meets the defined four-metric thresholds on the curated set — highest-priority gate in this roadmap, since RAG grounding is the system's highest-severity failure mode (`scoping_doc.md` §11)

Dependencies:
Phase 1 (corpus gate), Phase 4 (Gateway/Redis).

---

## Phase 6 — Agentic Orchestration Layer

Goal:
Assemble Pricing Strategy → [Compliance ‖ Revenue Impact] → Governance into one LangGraph orchestrator over shared typed state.

Capabilities:
- `AgentState` schema; agent nodes; application-level circuit breaking alongside Gateway fallback
- structured-output validation per node boundary; showcase trace-panel slice with an escalation scenario

Exit criteria:
- Primary Scenario (`groundedness/prd.md` §10) runs end-to-end on both auto-approved and escalated branches, each with a complete audit record
- trace panel visibly distinguishes the two branches

Dependencies:
Phases 2, 3 (Tier 1 outputs), 4 (Gateway), 5 (RAG).

---

## Phase 7 — Governance Layer

Goal:
Make every recommendation traceable and safely bounded.

Capabilities:
- HITL escalation gate; deterministic fallback to last compliance-approved rate table
- audit log schema and persistence; minimal review endpoint; showcase audit/HITL slice with a fallback scenario

Exit criteria:
- 100% audit-record completeness across a batch spanning both branches
- fallback triggers correctly under simulated agent failure, demonstrated in the showcase

Dependencies:
Phase 6.

---

## Phase 8 — Evaluation & Observability Hardening

Goal:
Make the system's own reliability measurable, not assumed.

Capabilities:
- CI-gated LLM-as-judge regression suite (absolute + baseline-relative blocking)
- continuous-score judge calibration; labeled golden dataset with coverage queries
- Stage 3 production monitoring with HITL-gated harvest loop; drift monitoring; OTel spans and cost/trace dashboard

Exit criteria:
- regression suite blocks on a deliberately regressed version and on an above-floor baseline-relative regression
- mis-calibrated judge fails calibration; drift monitor flags an injected shift; sub-threshold traces route to harvest without auto-promotion

Dependencies:
Phases 6, 7.

---

## Phase 9 — Containerization, CI/CD & Documentation Close-Out

Goal:
Production-shape packaging; documentation reflecting the as-built state.

Capabilities:
- multi-stage Docker; Compose wiring FastAPI, Redis Stack, Gateway
- full GitHub Actions pipeline; as-built `system_design.md`; runbook populated with actually-encountered challenges

Exit criteria:
- `scoping_doc.md` §3 definition of done fully satisfied
- system runs end-to-end via `docker-compose up`

Dependencies:
All prior phases.
