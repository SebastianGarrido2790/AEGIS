# Machine Learning Canvas

| Product                                                                         | Authors                   | Date            | Version                  |
| ------------------------------------------------------------------------------- | ------------------------- | --------------- | ------------------------ |
| AEGIS — Actuarial Elasticity & Governance Intelligence System _(working title)_ | Sebastián Garrido Arévalo | August 12, 2026 | 0.1 (Phase 0 — Planning) |

---

## 1. Background

Personal auto (P&C) insurers set premiums through periodic, manually filed rate tables that are actuarially sound at filing time but decay in accuracy between filing cycles (often 6–18 months) as loss experience, competitive positioning, and customer risk profiles drift. Pricing and actuarial teams face a structural tension: they need to adjust rates continuously to reflect true risk and elasticity, but every adjustment must survive regulatory scrutiny — unfair-discrimination rules, rate-filing requirements, actuarial justification — before it can be deployed.

Today this reconciliation is manual. Actuaries build elasticity models in isolation; legal/compliance reviews proposed changes in a separate, slower cycle; the two functions rarely share a single, auditable decision trail. The result is rate tables that are simultaneously slower to adapt than the business needs and harder to audit than regulators require.

## 2. Value Proposition

AEGIS is a production-grade agentic AI platform that closes the gap between causal pricing science and regulatory governance inside a single, auditable decision loop. It does not merely predict elasticity — it produces premium-adjustment recommendations that are pre-validated against regulatory constraints, quantified for revenue and loss-ratio impact, and escalated to a human underwriter whenever they fall outside pre-approved bounds.

The value is not "a better pricing model." It is a _governed decision system_: every recommendation carries its causal justification, its compliance-grounding citation, and its projected financial impact — compressing a multi-week actuarial-legal review cycle into a same-day, traceable decision.

## 3. Objectives

1. Estimate causal price elasticity of retention per risk segment, replacing correlational demand proxies with a genuine treatment-effect estimate.
2. Constrain any live rate exploration to a bounded, pre-approved corridor via contextual bandit exploration — never unconstrained optimization.
3. Ground every proposed rate change against real regulatory and actuarial-fairness source material before it can advance, with retrievable evidence citations.
4. Quantify the revenue and loss-ratio impact of each proposed change prior to human review.
5. Escalate any recommendation outside defined guardrails to a human underwriter, with full state and rationale attached.
6. Provide an evaluation and observability layer (retrieval-quality metrics, LLM-as-judge regression testing, tracing, drift detection) so the system's own reliability is measurable over time, not assumed.

## 4. Solution

**Core features:**

- **Tier 1 (deterministic):** a causal elasticity/retention model (double machine learning) plus a contextual bandit exploration engine, both governed by `params.yaml` and versioned via MLflow/DVC.
- **Tier 2 (agentic, LangGraph):** three cooperating agents — _Pricing Strategy_, _Regulatory Compliance_ (RAG-grounded over a Redis Stack/RedisVL index), and _Revenue/Loss-Ratio Impact_.
- **Tier 3 (governance):** a HITL escalation gate, a deterministic fallback to the last compliance-approved rate table, and a structured audit log per decision.
- **Middleware (LLM Gateway):** every agent and RAG call routes through an in-process LiteLLM gateway — no agent holds a raw provider SDK client or API key. The gateway enforces provider fallback/resiliency, pre-flight input guardrails (PII redaction, prompt-injection blocking) ahead of the Compliance Agent's highest-severity failure surface, and emits the token/cost/trace data the evaluation layer (§8) consumes.
- **Showcase interface (ADR-009):** a minimal, explicitly-labeled glass-box demo built with FastAPI + Jinja2, distinct from and not a substitute for the production UI excluded below. Built incrementally alongside Phases 2, 6, and 7, it exposes a curated set of preset scenarios — including at least one escalation case and one fallback case — so the governance mechanics themselves are directly observable, not just the happy path.

**Integration:** exposed as a FastAPI service consumed by an underwriting/pricing dashboard; agents invoked through a LangGraph orchestrator operating on a Pydantic-validated shared state object, with all provider calls mediated by the LLM Gateway. The showcase interface reuses this same FastAPI serving layer rather than introducing a separate demo-app stack.

**Constraints:** no live production rate deployment — this is a decision-support and recommendation system; a human underwriter always holds final authority to publish a rate.

**Out of scope (v1):** multi-state simultaneous filing logic, integration with a live carrier core system (API contracts modeled but not connected), claims severity modeling (frequency/elasticity only in v1).

## 5. Feasibility

Technically feasible using established open tooling: EconML/DoWhy for causal estimation, LangGraph for orchestration, Redis Stack (RedisVL) for regulatory RAG, MLflow/DVC for tracking, Docker/FastAPI for serving. Redis Stack was chosen over a standalone vector store because it is also the mandatory backing store for the LLM Gateway's two-tier prompt cache (§4) — one platform serves both roles instead of introducing a third distinct data technology, which directly reduces the solo-practitioner scope risk noted below.

Data feasibility is strong: public actuarial pricing datasets (e.g., the French Motor Third-Party Liability claims dataset) provide a realistic frequency/severity/exposure base for elasticity modeling, and NAIC model laws and state Unfair Trade Practices Act text are publicly available and suitable for the compliance RAG corpus.

The primary feasibility risk is solo-practitioner scope: three agents, two ML sub-layers, and an evaluation/observability layer represent a large build surface for one contributor. This is mitigated by strict phase gating (Rule 9) and a hard `src/` line-count ceiling per module.

## 6. Data

- **Training data:** a public P&C pricing dataset (e.g., French Motor Third-Party Liability / freMTPL2, or an equivalent actuarial dataset) providing policy exposure, claim frequency, claim severity, and risk-factor features (driver age, vehicle characteristics, region, bonus-malus).
- **Regulatory corpus (RAG):** public regulatory and actuarial-fairness source material — NAIC model laws/regulations on unfair discrimination in rating, publicly available state Unfair Trade Practices Act excerpts, and relevant Actuarial Standards of Practice (ASOPs) — chunked, embedded, and indexed in Redis Stack (RedisVL/HNSW), the same store backing the LLM Gateway's semantic cache.
- **Production-analog data:** a synthetic policy/segment stream generated to simulate a live book of business for bandit exploration and end-to-end testing, since no live carrier integration exists in v1.
- **Labeling:** no manual labeling required for the ML layer (supervised on historical claims); the RAG corpus requires a one-time chunking/indexing pass, not labeling.
- **Data contracts:** Great Expectations suites gate both intake paths as CI-blocking checks, not warnings — one suite validates the elasticity training data (schema conformance, value ranges, and a post-treatment leakage check specific to the causal model), and a second validates regulatory corpus ingestion (non-empty chunks, required metadata such as jurisdiction/section/effective date, and duplicate/near-duplicate detection), since a malformed regulatory chunk is a direct contributor to the RAG hallucination risk named in the charter's risk register. DVC versions the data and pipeline artifacts that pass these gates.

## 7. Metrics

- **ML metrics:** elasticity model — calibration and treatment-effect confidence intervals (not predictive AUC/RMSE alone, since the target is causal, not correlational); bandit — cumulative regret vs. an oracle static-price baseline.
- **Business metrics:** projected retention-adjusted revenue delta per segment; loss-ratio stability under proposed rate changes.
- **Agentic/system metrics:** Compliance Agent groundedness score and evidence coverage; LLM-as-judge agreement rate on Compliance Agent rulings; HITL escalation rate (automation efficiency vs. human review load); end-to-end decision latency.

## 8. Evaluation

**Offline:** backtesting the elasticity model on held-out historical policy cohorts; retrieval-quality evaluation (groundedness, evidence coverage) against a curated set of known-compliant and known-violating rate-change scenarios; an LLM-as-judge regression suite run on every prompt or model version change.

**Online (simulated):** bandit performance tracked via cumulative regret against the static-price baseline on the synthetic production-analog stream; drift monitoring on incoming segment feature distributions to flag when the elasticity model requires retraining.

## 9. Modeling

Iterative build order, phase-gated — no phase begins before the prior phase's exit criteria are met:

1. Baseline frequency/severity model (GLM) as an actuarial-standard reference point.
2. Causal elasticity/uplift model (`CausalForestDML` or equivalent double-ML estimator), benchmarked against the GLM baseline.
3. Contextual bandit layer (Thompson Sampling) wrapping the elasticity model's output within a bounded exploration corridor.
4. Agentic layer: Pricing Strategy Agent first (consumes Tier 1 output), then the Compliance Agent (RAG-grounded), then the Revenue/Loss-Ratio Impact Agent.
5. Governance layer: HITL gate, deterministic fallback, and structured audit logging — added last, wrapping the fully assembled pipeline.

## 10. Inference

Batch inference for elasticity model retraining (scheduled, off the critical path). Real-time (online) inference for the agentic recommendation loop — a pricing/underwriting user submits a segment or policy for review and receives a synthesized, compliance-checked recommendation within the same session.

## 11. Feedback

Underwriter HITL decisions (approve/reject/modify) feed back as labeled outcomes to refine the bandit's exploration corridor over time. Compliance Agent rulings flagged as incorrect by a human reviewer feed the LLM-as-judge regression suite as new adversarial test cases, closing the evaluation loop.

## 12. Project

- **Team:** solo practitioner (Sebastián Garrido Arévalo), acting as architect, ML engineer, and agentic systems engineer.
- **Deliverables:** the full Phase 0 planning suite (this canvas, the project charter, PRD, user story, technical roadmap, system design ARD, and challenges/solutions runbook), followed by a phase-gated implementation delivering the three-tier system described above, containerized and CI-gated.
- **Projected timeline:** Phase 0 planning completed before implementation begins; implementation phases sequenced across the remainder of 2026, following the same plan-first, phase-gated discipline used across this portfolio.
