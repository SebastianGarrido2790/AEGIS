# Technical Roadmap — AEGIS _(working title)_

**Actuarial Elasticity & Governance Intelligence System**
Author: Sebastián Garrido Arévalo | Date: August 13, 2026 | Phase 0 — Planning

Durations are best-effort and flexible to accommodate new findings; no phase begins before the prior phase's exit criteria are satisfied (phase-gated build order).

---

## Phase 0 — Planning & Design _(complete)_

- **Goal:** Establish the strategic and product foundation before any implementation.
- **Deliverables:** `canvas.md`, `project_charter.md`, `prd.md`, `user_story.md`, this roadmap, `system_design.md`, `challenges_and_solutions_guide.md`.
- **Exit criteria:** all seven Phase 0 artifacts complete and internally consistent; the three architectural decisions (LiteLLM Gateway, Redis Stack, GX+DVC data contracts) confirmed and reflected across every document.
- **Dependencies:** none.

## Phase 1 — Project Scaffolding & Data Contracts

- **Goal:** Stand up the repo skeleton, dependency management, configuration schema, and CI-blocking data contracts before any modeling begins — no data enters the pipeline unvalidated.
- **Key tasks:**
  - Repository structure, `uv` environment, `params.yaml` schema definition.
  - Great Expectations suite for elasticity training data (schema conformance, value ranges, post-treatment leakage check).
  - Great Expectations suite for regulatory corpus ingestion (non-empty chunks, required metadata, duplicate detection).
  - DVC pipeline skeleton wired to both GX gates.
  - CI skeleton: lint, type-check (Pyright/Ruff), and GX gate execution.
- **Deliverables:** working repo scaffold; both GX suites executing and blocking on injected bad-data test cases; DVC-tracked raw datasets.
- **Exit criteria:** CI is green on the scaffold; each GX suite demonstrably fails a deliberately malformed test fixture and passes a valid one.
- **Dependencies:** none.
- **Estimated duration:** 1–2 weeks.

## Phase 2 — Tier 1: Deterministic ML Baseline

- **Goal:** Establish the GLM actuarial baseline, then the causal elasticity/uplift model that supersedes it.
- **Key tasks:**
  - Feature engineering on exposure/frequency/severity data.
  - GLM baseline model (actuarial-standard reference point).
  - Causal elasticity model (`CausalForestDML` or equivalent double-ML estimator), including a confounding sensitivity analysis.
  - MLflow experiment tracking and model registry integration.
- **Deliverables:** baseline and causal model artifacts; MLflow experiment logs; evaluation report on calibration and treatment-effect confidence intervals.
- **Exit criteria:** causal model outperforms the GLM baseline with defensible confidence intervals (PRD §11).
- **Dependencies:** Phase 1 (data contracts).
- **Estimated duration:** 2–3 weeks.

## Phase 3 — Contextual Bandit Exploration Engine

- **Goal:** Wrap the elasticity model's output in a bounded, live-exploration mechanism.
- **Key tasks:**
  - Define the exploration corridor bounds in `params.yaml`.
  - Implement Thompson Sampling over the bounded corridor.
  - Build a synthetic production-analog policy/segment stream simulator.
  - Backtest cumulative regret against a static-price baseline.
- **Deliverables:** bandit module; regret evaluation report.
- **Exit criteria:** bandit demonstrates lower cumulative regret than the static-price baseline on the synthetic stream (PRD §11).
- **Dependencies:** Phase 2.
- **Estimated duration:** 1–2 weeks.

## Phase 4 — LLM Gateway Integration

- **Goal:** Stand up the LiteLLM in-process Gateway and its backing Redis Stack instance _before_ any agent is built, so every agent is developed against the Gateway abstraction from day one — no agent ever holds a raw provider client.
- **Key tasks:**
  - Configure LiteLLM: unified model-string abstraction, provider fallback chain, load-balancing strategy — all declared in `params.yaml`.
  - Deploy Redis Stack container (shared by the Gateway's two-tier cache and, later, the regulatory RAG index).
  - Implement pre-flight guardrail callbacks: PII redaction and prompt-injection blocking.
  - Wire cost/trace OTel emission (`llm.tokens.input/output`, `llm.fallback.count`).
- **Deliverables:** working Gateway; Redis Stack service in `docker-compose.yaml`; guardrail unit test suite.
- **Exit criteria:** a simulated provider failure correctly triggers the fallback chain; a known injection pattern is blocked at the Gateway boundary in a unit test.
- **Dependencies:** Phase 1 (infra scaffolding). Independent of Phases 2–3; sequenced here rather than parallelized, given solo-practitioner bandwidth.
- **Estimated duration:** 1–2 weeks.

## Phase 5 — Regulatory RAG Foundation

- **Goal:** Build and validate the Compliance Agent's retrieval substrate before wrapping it in agent logic — retrieval quality must be proven independently first.
- **Key tasks:**
  - Ingest and chunk the regulatory corpus (post-GX gate).
  - Embed and index in Redis Stack (RedisVL/HNSW).
  - Build the retrieval-quality evaluation harness (groundedness, evidence coverage) against a curated set of known-compliant and known-violating scenarios.
- **Deliverables:** indexed regulatory corpus; retrieval evaluation report with baseline groundedness/evidence-coverage scores.
- **Exit criteria:** the retrieval harness meets the defined groundedness/evidence-coverage threshold (PRD §11) on the curated test set — this gate is treated as the highest-priority exit criterion in the roadmap, given RAG grounding is the system's highest-severity failure mode (Charter §9).
- **Dependencies:** Phase 1 (corpus GX gate), Phase 4 (Gateway/Redis Stack).
- **Estimated duration:** 2 weeks.

## Phase 6 — Agentic Orchestration Layer

- **Goal:** Assemble the three cooperating agents into a single LangGraph orchestrator over shared, typed state.
- **Key tasks:**
  - Define the `AgentState` Pydantic schema.
  - Implement agent nodes in build order: Pricing Strategy → Regulatory Compliance → Revenue/Loss-Ratio Impact.
  - Wire the LangGraph-level circuit breaker (application-level failures) to run alongside the Gateway's provider-level fallback, both active simultaneously.
  - Structured-output validation at each node boundary.
- **Deliverables:** working orchestrator graph; per-agent unit tests; a full end-to-end trace of the PRD §10 Primary Scenario.
- **Exit criteria:** the Primary Scenario runs end-to-end on both the auto-approved and escalated branches, each producing a complete audit record.
- **Dependencies:** Phase 2, 3 (Tier 1 outputs), Phase 4 (Gateway), Phase 5 (RAG foundation).
- **Estimated duration:** 2–3 weeks.

## Phase 7 — Governance Layer

- **Goal:** Implement the HITL escalation gate, deterministic fallback, and audit logging that make every recommendation traceable and safely bounded.
- **Key tasks:**
  - Escalation trigger logic (bounds violation or Compliance Agent flag).
  - Deterministic fallback to the last compliance-approved rate table.
  - Audit log schema and persistence layer.
  - Minimal HITL review endpoint for human sign-off.
- **Deliverables:** governance module; audit log store; review endpoint.
- **Exit criteria:** 100% audit-record completeness across a test batch spanning both escalated and auto-approved scenarios (PRD §11); fallback triggers correctly under a simulated agent failure.
- **Dependencies:** Phase 6.
- **Estimated duration:** 1–2 weeks.

## Phase 8 — Evaluation & Observability Hardening

- **Goal:** Make the system's own reliability measurable, not assumed — LLM-as-judge regression, drift detection, and full tracing.
- **Key tasks:**
  - LLM-as-judge regression suite, CI-gated on every prompt/model version change.
  - Drift monitoring on incoming segment feature distributions.
  - OTel spans across both the Gateway and the LangGraph orchestrator.
  - Cost/trace dashboard.
- **Deliverables:** CI-gated regression suite; drift monitor; observability dashboard.
- **Exit criteria:** the regression suite blocks CI on a deliberately regressed model/prompt version; the drift monitor correctly flags an injected distribution shift in a test.
- **Dependencies:** Phase 6, 7.
- **Estimated duration:** 2 weeks.

## Phase 9 — Containerization, CI/CD & Documentation Close-Out

- **Goal:** Full production-shape packaging and finalize documentation to reflect the actual implemented state.
- **Key tasks:**
  - Multi-stage Dockerfiles; `docker-compose.yaml` wiring FastAPI, Redis Stack, and the Gateway.
  - GitHub Actions pipeline: lint, type-check, tests, GX gates, reproducibility check, coverage gate.
  - Update `system_design.md` to reflect the as-built architecture.
  - Populate `challenges_and_solutions_guide.md` with challenges actually encountered during Phases 1–8.
- **Deliverables:** fully containerized system; green end-to-end CI pipeline; finalized documentation set.
- **Exit criteria:** Charter §5 Definition of Done fully satisfied; the system runs end-to-end via `docker-compose up`.
- **Dependencies:** all prior phases.
- **Estimated duration:** 1–2 weeks.

---

**Total estimated duration:** ~14–19 weeks, phase-gated and best-effort. Phases 2–3 (Tier 1) and Phase 4 (Gateway) have no mutual dependency and could be parallelized with additional contributor capacity, but are sequenced here to match solo-practitioner bandwidth (Charter §9).
