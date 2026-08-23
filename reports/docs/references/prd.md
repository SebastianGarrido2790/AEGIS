# Product Requirements Document — AEGIS _(working title)_

**Actuarial Elasticity & Governance Intelligence System**
Author: Sebastián Garrido Arévalo | Date: August 13, 2026 | Phase 0 — Planning

---

## 1. Executive Summary

AEGIS is a three-tier agentic AI platform that recommends personal auto insurance premium adjustments by combining a causal elasticity model with a RAG-grounded regulatory compliance check, a financial impact projection, and a human-in-the-loop (HITL) escalation gate. It replaces the current disconnected, multi-week actuarial-legal review cycle with a single, auditable decision loop: every recommendation carries its causal justification, its compliance-evidence citation, and its projected revenue/loss-ratio impact, and never reaches deployment without either falling inside pre-approved bounds or being explicitly reviewed by a human underwriter.

## 2. Project Analogy

Imagine a pricing team at an insurer as three separate departments that rarely talk to each other in real time: the actuaries who know the numbers, the compliance lawyers who know the rules, and the finance team who knows the P&L impact. Today, a single rate change has to travel physically between these three desks, in sequence, each one waiting on the last — a process that takes weeks.

AEGIS is like putting all three specialists in one room with a shared whiteboard, where each one does their part of the analysis simultaneously and the room's supervisor (a human underwriter) only needs to step in when the three specialists don't agree, or when the proposed change is unusually large. The specialists never make the final call alone — they prepare a fully documented recommendation, and a person always signs off before anything goes live.

## 3. Problem Statement (Stakeholder Perspective)

From the perspective of a pricing/actuarial leader: "I have the data to know our rates are stale within months of filing, but by the time my team's elasticity analysis clears legal review, the market has already moved again. I can't tell whether the bottleneck is analytical or procedural, and I have no single record showing why a given rate change was approved or rejected — which makes both regulatory audits and my own retrospectives slower than they should be."

The stakeholder problem is not "we lack a pricing model." It is the absence of a shared, auditable process that lets causal analysis, compliance validation, and financial sign-off happen as one traceable decision rather than three sequential, loosely coupled ones.

## 4. Goals & Non-Goals

**Goals:**

- Produce premium-adjustment recommendations grounded in causal (not correlational) elasticity estimates.
- Pre-validate every recommendation against regulatory/actuarial-fairness source material, with retrievable evidence.
- Quantify projected revenue and loss-ratio impact before any human review.
- Escalate to a human underwriter whenever a recommendation exceeds pre-approved bounds, with full context attached.
- Produce a complete, structured audit trail for every recommendation, approved or rejected.
- Make the system's own reliability measurable: retrieval-quality metrics, LLM-as-judge regression testing, and drift detection are first-class outputs, not incidental.

**Non-Goals:**

- AEGIS does not autonomously publish a rate change. A human underwriter always holds final authority.
- AEGIS does not model claims severity in v1 — frequency/elasticity only.
- AEGIS is not a multi-jurisdiction compliance product; the regulatory corpus models a single illustrative jurisdiction.
- AEGIS does not integrate with a live carrier core system in v1; integration points are modeled, not connected.

## 5. Personas (Summary)

| Persona                                   | Role                                               | Primary need from AEGIS                                                                                          |
| ----------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Elena — Senior Pricing Actuary            | Builds and defends elasticity/rate models          | Faster iteration on rate proposals without losing actuarial rigor                                                |
| Marco — Underwriting & Compliance Manager | Reviews and approves/rejects proposed rate changes | Confidence that every proposal has already been checked against regulatory constraints before it reaches him     |
| Renata — Chief Pricing Officer            | Owns the P&L and competitive pricing strategy      | Visibility into financial impact and audit-readiness across every rate decision, not just the ones that go wrong |

(Full stories for each persona are developed in `user_story.md`.)

## 6. Functional Requirements

1. The system SHALL ingest historical policy exposure, claim frequency, and risk-factor data, and produce a causal elasticity/retention estimate per segment.
2. The system SHALL constrain any live rate exploration to a bounded corridor via contextual bandit exploration; it SHALL NOT recommend adjustments outside that corridor without explicit escalation.
3. The system SHALL retrieve and cite relevant regulatory/actuarial-fairness source material for every proposed rate change via the Compliance Agent, backed by the Redis Stack (RedisVL) index.
4. The system SHALL project revenue and loss-ratio impact for every proposed change via the Revenue/Loss-Ratio Impact Agent.
5. The system SHALL escalate to a human underwriter (HITL gate) whenever a recommendation falls outside pre-approved bounds or the Compliance Agent flags a potential violation.
6. The system SHALL fall back deterministically to the last compliance-approved rate table if any agent or retrieval component degrades or fails.
7. The system SHALL log a complete, structured audit record (inputs, agent outputs, evidence citations, financial projection, HITL decision) for every recommendation.
8. All LLM calls SHALL route through the LLM Gateway (LiteLLM) — no agent may hold a raw provider client or key.
9. All data entering the training pipeline or the regulatory corpus SHALL pass its respective Great Expectations suite as a CI-blocking gate before being versioned by DVC.

## 7. Non-Functional Requirements

- **Auditability:** every decision record must be reconstructable end-to-end from stored state, without relying on live re-querying of any external system.
- **Reliability:** provider-level failures must trigger the Gateway's fallback chain; application-level failures (e.g., structured-output validation) must trigger the LangGraph-level circuit breaker; both must be independently observable via OTel metrics.
- **Latency:** end-to-end recommendation latency for a single segment/policy review should remain within an interactive range suitable for same-session use by an underwriter (target: low single-digit seconds for the agentic path, excluding scheduled batch retraining).
- **Cost governance:** token/cost per recommendation must be tracked and attributable per agent via the Gateway's cost/trace export.
- **Reproducibility:** the full pipeline (data → model → agent outputs) must be reproducible bit-for-bit under a fixed seed and CI-gated.
- **Modularity:** no module in `src/` exceeds 1,000 lines; each of the three agents, the Gateway integration, and the governance layer are independently testable.

## 8. System Architecture

**Tier 1 — Deterministic ML (Brawn):** GLM baseline → causal elasticity/uplift model (double machine learning) → contextual bandit exploration engine, all governed by `params.yaml`, tracked via MLflow, versioned via DVC, and gated on ingestion by Great Expectations.

**Tier 2 — Agentic Orchestration (Brain), LangGraph:**

- _Pricing Strategy Agent_ — consumes Tier 1 output, proposes a segment-level premium adjustment.
- _Regulatory Compliance Agent_ — RAG-grounded over the Redis Stack (RedisVL/HNSW) regulatory index; validates the proposal and returns evidence citations.
- _Revenue/Loss-Ratio Impact Agent_ — projects the financial effect of the proposal.

**Middleware — LLM Gateway (LiteLLM):** unified provider abstraction, fallback/resiliency, smart routing, exact/semantic caching (shared Redis Stack instance), pre-flight input guardrails, and cost/trace export — sits between every agent and the external LLM providers.

**Tier 3 — Governance:** HITL escalation gate; deterministic fallback to the last compliance-approved rate table; structured audit logging.

**Serving:** FastAPI service, containerized via Docker/Docker Compose, consumed by an underwriting/pricing dashboard. GitHub Actions CI enforces reproducibility and coverage gates; OTel provides tracing and observability across both the Gateway and the LangGraph orchestrator.

## 9. Data Sources

- Public P&C pricing dataset (e.g., French Motor Third-Party Liability / freMTPL2) for elasticity/retention training.
- Public regulatory and actuarial-fairness source material (NAIC model laws, state Unfair Trade Practices Act excerpts, relevant ASOPs) for the Compliance Agent's RAG index.
- Synthetic policy/segment stream for bandit exploration and end-to-end testing, standing in for a live production feed.

## 10. Primary Scenario (Acceptance Criteria / Test)

**Scenario:** Elena submits a mid-sized urban driver segment showing declining retention for Compliance and Financial review.

1. Tier 1 returns a causal elasticity estimate and a bandit-proposed adjustment within the bounded corridor.
2. The Pricing Strategy Agent formats a proposed premium adjustment.
3. The Compliance Agent retrieves and cites relevant regulatory material, returning a pass/flag verdict with evidence.
4. The Revenue/Loss-Ratio Impact Agent projects the financial effect.
5. **If** the proposal is within bounds and Compliance passes: the system logs the full record and marks it auto-approved-for-review, ready for Marco's routine sign-off.
   **If** the proposal exceeds bounds or Compliance flags a concern: the system escalates directly to Marco with full state attached and withholds any "approved" status.
6. **Acceptance criteria:** the audit record for either path is complete (causal justification, evidence citation, financial projection, and final disposition), reconstructable without live re-querying, and the Compliance Agent's citation is independently verifiable against the source corpus.

## 11. Success Metrics & KPIs

- Elasticity model: calibration and treatment-effect confidence interval width, vs. GLM baseline.
- Bandit: cumulative regret vs. static-price baseline (simulated).
- Compliance Agent: groundedness score and evidence coverage on the adversarial evaluation suite; LLM-as-judge agreement rate.
- Governance: HITL escalation rate (should reflect genuine ambiguity, not over- or under-triggering).
- System: end-to-end decision latency; audit-record completeness rate (target: 100%).

## 12. Out of Scope (Future Iterations)

- Claims severity modeling (frequency/elasticity only in v1).
- Multi-jurisdiction, multi-state simultaneous filing logic.
- Live carrier core system integration (API contracts modeled, not connected).
- Autonomous rate publication without human sign-off.
- **A full production underwriting UI** — a real, daily-use console for Marco's persona, with full input validation, arbitrary-policy lookup, and multi-user access control, remains out of scope. This is distinct from the showcase interface (ADR-009, Canvas §4): a minimal, explicitly-labeled glass-box demo built for the PRD §5 secondary audience (evaluators, not underwriters), exposing curated preset scenarios rather than production-grade arbitrary input. The showcase interface's existence does not reopen this exclusion.

## 13. Dependencies & Constraints

- Depends on public dataset and regulatory corpus availability and quality; no proprietary carrier data is used.
- Depends on LLM provider availability, mediated through the Gateway's fallback chain.
- Solo-practitioner build capacity constrains scope; phase gating (Rule 9) and the `src/` line-count ceiling are the primary controls.
- No live production deployment is in scope; all "production-grade" claims refer to architecture and engineering discipline, not live operational status.

## 14. Open Questions / Risks (To Be Revisited After Implementation)

- How sensitive is the causal elasticity estimate to unmeasured confounding in the public dataset, and does the sensitivity analysis hold up under adversarial review?
- Is the chosen groundedness/evidence-coverage threshold for the Compliance Agent strict enough to catch subtle regulatory misapplications, not just obviously ungrounded citations?
- Does the HITL escalation rate, once measured against the synthetic stream, suggest the bounded corridor is calibrated correctly, or does it need tightening/loosening?
- Will the HITL escalation rate metric remain meaningful once real (not synthetic) production data is introduced, given the synthetic stream's known generalization limits (Charter §9)?
