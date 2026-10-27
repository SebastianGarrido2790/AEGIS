# Implementation Roadmap — AEGIS

Non-authoritative, high-level delivery map. Its only job: major capabilities in dependency order, and the evidence unlocking each next stage. No technical specifications, no task lists, no decisions — those live in feature specs, phase plans, and `architecture/system_design.md`.

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

## Phase 2 — Tier 1: Deterministic ML Baseline _(reset — decisions approved; EDA-first redo pending)_

Goal:
Establish the GLM actuarial baseline and the causal elasticity model, with a first showcase slice — preceded this time by exploratory data analysis of the raw data before any training-pipeline code is written.

Capabilities:
- exposure and capped-loss feature engineering
- capped-pure-premium Tweedie GLM loss-cost baseline; `CausalForestDML` elasticity estimator with confounding sensitivity analysis
- MLflow tracking and registry; four-preset validation showcase reading the `approved` segment artifact (no bandit output or proposed adjustment)

Exit criteria:
- causal estimator recovers the known synthetic treatment effect with defensible per-segment and rank-based confidence intervals (`grounding/scoping_doc.md` §0, §2)
- showcase slice renders correctly, clearly labeled demo rather than production view (carries an explicit developer-diagnostic exemption from INV-5 — $\epsilon$/recovery metrics only, never a proposed $\Delta$ — which strictly binds pricing recommendations)

Carryover into the redo — the confirmed Phase 2 findings stand and constrain the new design:
- heavy-tailed feature/loss distribution (re-confirmed on raw data by the EDA run, `../../artifacts/eda_summary.json`: max claim €2.04M ≈ 5.91% of portfolio losses) — loss capping and robust validation are mandatory, not optional
- aggregate correlation-to-ground-truth is sample-size- and outlier-sensitive — no single aggregate Pearson gate; per-segment and Spearman metrics only

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
- corridor interaction documented: exploration allows ±15% while auto-approval covers ±5%, so bandit proposals in (5%, 15%] escalate by design

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
- per-session/request cost ceiling set from measured OTel cost data (covers Compliance and future Briefing calls)

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
- harness meets the defined four-metric thresholds on the curated set — highest-priority gate in this roadmap, since RAG grounding is the system's highest-severity failure mode (`grounding/scoping_doc.md` §11)

Dependencies:
Phase 1 (corpus gate), Phase 4 (Gateway/Redis).

---

## Phase 6 — Agentic Orchestration Layer

Goal:
Assemble the Option C decision loop (ADR-011): deterministic Pricing Strategy → [Compliance Agent (LLM) ‖ Revenue Impact (deterministic)] → Governance disposition node (`READY_FOR_REVIEW` vs. `ESCALATED`), with a deterministic Jinja2 templated underwriter briefing.

Entry criteria (must hold before Tier 2 ships):
- off-switch/kill-switch ADR approved: who can halt agent invocations, how fast, with what state preservation (per gates Part 1.5)

Capabilities:
- `AgentState` schema; deterministic Pricing and Impact nodes (first-order elasticity volume approximation); Compliance reasoning agent (RAG over Redis Stack)
- deterministic Governance disposition node (`READY_FOR_REVIEW` vs. `ESCALATED`)
- deterministic Jinja2 underwriter briefing template satisfying Story 2.1 without adding LLM complexity to Phase 6
- application-level circuit breaking alongside Gateway fallback; structured-output validation per node boundary; showcase trace-panel slice (ADR-009)
- interim (no persistence/fallback engine yet): grounding failure yields `ESCALATED` with no proposal presented as a recommendation

Exit criteria:
- Primary Scenario (`grounding/prd.md` §10) runs end-to-end on both auto-approved and escalated branches, each emitting a complete in-memory record flagged non-persisted (persistence lands in Phase 7)
- trace panel visibly distinguishes the branches

Dependencies:
Phases 2, 3 (Tier 1 outputs), 4 (Gateway), 5 (RAG).

---

## Phase 7 — Governance Layer & Audit Persistence

Goal:
Make every recommendation traceable, fail-safe, and safely bounded.

Capabilities:
- full immutable audit log persistence (schema validation, replayability, INV-7 blocking response gate)
- deterministic operational fallback engine (INV-6) safely reverting to the last compliance-approved rate table ($Rate_{approved}$) with `fallback_active = True`
- formal HITL review endpoint for human sign-off
- showcase audit/HITL slice with a fallback scenario demonstrating fail-safe reversion

Exit criteria:
- 100% audit-record completeness across a batch spanning both branches (INV-7)
- fallback triggers correctly under simulated agent failure, demonstrated in the showcase

Dependencies:
Phase 6.

---

## Phase 8 — Evaluation, Observability & Narrative Hardening

Goal:
Make the system's own reliability measurable, and introduce the post-governance LLM Briefing Generator only if a richer template cannot satisfy Story 2.1.

Capabilities (sequenced — evaluation core first, briefing last):
1. CI-gated LLM-as-judge regression suite (absolute + baseline-relative blocking)
2. continuous-score judge calibration; labeled golden dataset with coverage queries
3. Stage 3 production monitoring with HITL-gated harvest loop; drift monitoring; OTel spans and cost/trace dashboard
4. adversarial red-team suite covering the Compliance Agent and (if built) the Briefing Generator
5. template-sufficiency spike: test whether an enriched Jinja2 template satisfies Story 2.1 before building the LLM briefing; build it only on demonstrated failure, and explicitly revisit the prompt-isolation contract first if the briefing needs the compliance rationale to be useful
6. post-governance LLM Underwriter Briefing generator with prompt isolation (reads only structured fields) and multi-layer deterministic validator (numeric match, citation ID lookup, disposition consistency, immediate template fallback) — only if step 5 fails
7. briefing validation metrics: tracking validator-pass rate and template-fallback rate (`briefing.validation.pass_rate`, `briefing.fallback.count`)

Exit criteria:
- regression suite blocks on a deliberately regressed version and on an above-floor baseline-relative regression
- mis-calibrated judge fails calibration; drift monitor flags an injected shift; sub-threshold traces route to harvest without auto-promotion
- red-team suite green on Compliance (+ Briefing if built)
- template-sufficiency verdict recorded; briefing validator telemetry confirms pass and fallback tracking operational (if built)

Dependencies:
Phases 6, 7.

---

## Phase 9 — Containerization, CI/CD & Documentation Close-Out

Goal:
Production-shape packaging; documentation reflecting the as-built state.

Capabilities:
- multi-stage Docker; Compose wiring FastAPI, Redis Stack, Gateway
- full GitHub Actions pipeline; as-built `architecture/system_design.md`; runbook populated with actually-encountered challenges

Exit criteria:
- `grounding/scoping_doc.md` §3 definition of done fully satisfied
- system runs end-to-end via `docker-compose up`

Dependencies:
All prior phases.
