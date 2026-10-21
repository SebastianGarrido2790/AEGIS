# Product Requirements Document — AEGIS

**Actuarial Elasticity & Governance Intelligence System**
Author: Sebastián Garrido Arévalo | Date: August 13, 2026

---

## 1. Executive Summary

AEGIS is a governed-by-construction agentic platform built to production engineering and actuarial standards that recommends personal auto insurance premium adjustments by combining causal elasticity estimation with a RAG-grounded regulatory compliance check, a financial impact projection, and a human-in-the-loop (HITL) escalation gate. Under statutory rate-filing regimes, personal auto rate revisions occur on cycles often lasting many months, leaving pricing teams with stale rates or manual, disconnected review cycles. AEGIS replaces the current disconnected actuarial-legal review cycle with a single, auditable decision loop: every recommendation carries its causal justification, its compliance-evidence citation, and its projected revenue/loss-ratio impact, and never reaches deployment without either falling inside pre-approved bounds or being explicitly reviewed by a human underwriter.

## 2. Project Analogy

Imagine a pricing team at an insurer as three separate departments that rarely talk to each other in real time: the actuaries who know the numbers, the compliance lawyers who know the rules, and the finance team who knows the P&L impact. Today, a single rate change has to travel physically between these three desks, in sequence, each one waiting on the last — a process that takes weeks.

AEGIS is like putting all three specialists in one room with a shared whiteboard, where each one does their part of the analysis simultaneously and the room's supervisor (a human underwriter) only needs to step in when the proposal exceeds pre-approved bounds or compliance flags it, or when the proposed change is unusually large. The specialists never make the final call alone — they prepare a fully documented recommendation, and a person always signs off before anything goes live.

## 3. Problem Statement (Stakeholder Perspective)

From the perspective of a pricing/actuarial leader: "I have the data to know our rates decay in accuracy over filing cycles often lasting many months, but by the time my team's elasticity analysis clears legal review, the market has already moved again. I can't tell whether the bottleneck is analytical or procedural, and I have no single record showing why a given rate change was approved or rejected — which makes both regulatory audits and my own retrospectives slower than they should be."

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
- AEGIS does not build a separate claims-severity model or use severity-driven pricing in v1. The Phase 2 baseline is a capped-pure-premium Tweedie GLM (compound Poisson-Gamma), whose loss-cost output is used as the baseline reference; large-loss excess is handled under the approved cap-and-load policy.
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

1. The system SHALL ingest empirical policy exposure, claim frequency, losses, and rating factors from the original French MTPL (`freMTPLfreq`/`freMTPLsev`) benchmark. Its seven rating factors are Power, Brand, Gas, Region, DriverAge, CarAge, and Density; BonusMalus and Area are not present. The system SHALL evaluate causal elasticity estimation against a semi-synthetic price-treatment experiment with known ground truth per segment.
2. The system SHALL constrain any live rate exploration to a bounded corridor via contextual bandit exploration; it SHALL NOT recommend adjustments outside that corridor without explicit escalation.
3. The system SHALL retrieve and cite relevant regulatory/actuarial-fairness source material for every proposed rate change via the Compliance Agent, backed by the Redis Stack (RedisVL) index.
4. The system SHALL project revenue and loss-ratio impact for every proposed change via the Revenue/Loss-Ratio Impact deterministic tool node (INV-5), using a first-order demand elasticity volume approximation rather than static premium multiplication, assuming homogeneous retention risk within segment.
5. The system SHALL escalate to a human underwriter (HITL gate) whenever a recommendation falls outside pre-approved bounds or the Compliance Agent flags a potential violation.
6. The system SHALL fall back deterministically to the last compliance-approved rate table if compliance retrieval groundedness drops below threshold or a component degrades (INV-6), emitting an escalated disposition with operational fallback flagged (`fallback_active = True`).
7. The system SHALL log a complete, structured audit record (inputs, agent outputs, evidence citations, financial projection, HITL decision) for every recommendation before emitting an API response (INV-7).
8. All LLM calls SHALL route through the LLM Gateway (LiteLLM) — no agent may hold a raw provider client or key.
9. All data entering the training pipeline or the regulatory corpus SHALL pass its respective Great Expectations suite as a CI-blocking gate before being versioned by DVC.

## 7. Non-Functional Requirements

- **Auditability:** every decision record must be reconstructable end-to-end from stored state, without relying on live re-querying of any external system.
- **Reliability:** provider-level failures must trigger the Gateway's fallback chain; application-level failures (e.g., structured-output validation) must trigger the LangGraph-level circuit breaker; both must be independently observable via OTel metrics.
- **Latency:** end-to-end recommendation latency for a single segment/policy review should remain within an interactive range suitable for same-session use by an underwriter (target: low single-digit seconds for the agentic path, excluding scheduled batch retraining). Aspirational — no SLA is set (see `scoping_doc.md` §8); the Gateway/tool timeouts in `params.yaml` are failure bounds, not latency budgets.
- **Cost governance:** token/cost per recommendation must be tracked and attributable per agent via the Gateway's cost/trace export.
- **Phase 2 reproducibility:** under the same lockfile and operating system, Phase 2 data products (cleaned data, DGP table, truth sidecar, and manifest) are bit-identical; model outputs reproduce within stated numeric tolerances. Cross-operating-system model identity is not claimed. CI runs a committed small-fixture smoke tier; the full-data DVC run and multi-seed validation campaign run locally/on demand, with the validation summary committed. Later phases specify their own reproducibility gates.
- **Modularity:** no module in `src/` exceeds 1,000 lines; the Compliance Agent, deterministic tool nodes, Gateway integration, and governance layer are independently testable.

## 8. System Architecture

**Tier 1 — Deterministic ML (Brawn):** GLM baseline → causal elasticity/uplift model (double machine learning over semi-synthetic benchmark) → contextual bandit exploration engine, all governed by `params.yaml`, tracked via MLflow, versioned via DVC, and gated on ingestion by Great Expectations.

**Tier 2 — Governed Agentic Workflow, LangGraph (ADR-011):**

- _Pricing Strategy (Deterministic Node)_ — consumes Tier 1 output, validates inputs, and constructs the typed segment-level rate proposal.
- _Regulatory Compliance Agent (LLM)_ — RAG-grounded over the Redis Stack (RedisVL/HNSW) regulatory index; evaluates statutory compliance and returns evidence citations.
- _Revenue/Loss-Ratio Impact (Deterministic Node)_ — computes projected portfolio financial effects ($LossCost$, $\Delta Rev$, $LossRatio$) accounting for demand elasticity ($Q_{new} = Q_0 \times (1 + \epsilon \cdot \Delta)$, $\Delta Rev \approx Rev_0 \cdot [(1+\Delta)(1+\epsilon \Delta) - 1]$) via first-order actuarial approximations assuming homogeneous retention risk within segment (no adverse selection in lapse behavior), without LLM involvement (INV-5).
- _Underwriter Briefing & Deterministic Validator_ — in Phase 6, generates an executive narrative via a deterministic Jinja2 template. When enhanced with an LLM Briefing Generator post-Phase 7 (Phase 8), it executes synchronously under a bounded timeout ($\le 2\text{s}$), ingesting strictly structured non-free-text fields (never raw corpus text or un-sanitized LLM rationales, per AGENTS.md §2 prompt isolation) with statutory quotes inserted deterministically by citation ID lookup. Verified by a deterministic validator (numeric match, citation ID match, and disposition consistency rule) with template fallback on any discrepancy, ensuring 100% complete audit records before API response per INV-7.

**Middleware — LLM Gateway (LiteLLM):** unified provider abstraction, fallback/resiliency, smart routing, exact/semantic caching (shared Redis Stack instance), pre-flight input guardrails, and cost/trace export — sits between every agent and the external LLM providers.

**Tier 3 — Governance & Audit:** HITL escalation gate; deterministic fallback to the last compliance-approved rate table; structured audit logging.

**Serving:** FastAPI service, containerized via Docker/Docker Compose, consumed by an underwriting/pricing dashboard. GitHub Actions CI enforces reproducibility and coverage gates; OTel provides tracing and observability across both the Gateway and the LangGraph orchestrator.

## 9. Data Sources

- Public P&C benchmark dataset: the original French Motor Third-Party Liability frequency/severity files (`freMTPLfreq`/`freMTPLsev`), providing 413,169 policy records, 16,181 claim records, and seven rating factors (Power, Brand, Gas, Region, DriverAge, CarAge, Density); BonusMalus and Area are absent. Input files shall be pinned by checksum in a DVC-tracked manifest. Because these observational files lack randomized price variation and renewal records, price variation and price elasticity are modeled via a semi-synthetic data-generating process (DGP) over the empirical covariates to benchmark causal recovery against known ground truth. The Phase 2 baseline uses a capped-pure-premium Tweedie GLM with excess load; it is not a separate severity model or severity-driven pricing system.
- Public regulatory and actuarial-fairness source material (NAIC model laws, state Unfair Trade Practices Act excerpts, relevant ASOPs) for the Compliance Agent's RAG index.
- Synthetic policy/segment stream for bandit exploration and end-to-end testing, standing in for a live production feed.

## 10. Primary Scenario (Acceptance Criteria / Test)

**Scenario:** Elena submits a mid-sized urban driver segment showing declining retention for Compliance and Financial review.

1. Tier 1 returns a causal elasticity estimate $\epsilon(X)$ and a bandit-proposed adjustment within the bounded corridor.
2. The Pricing Strategy deterministic node validates inputs and constructs the typed rate adjustment proposal.
3. In parallel:
   - The Compliance Agent retrieves and cites relevant regulatory material, returning a pass/flag verdict with evidence.
   - The Revenue/Loss-Ratio Impact deterministic node computes the financial projection accounting for volume elasticity (INV-5).
4. The Governance deterministic node evaluates corridor bounds and compliance status:
   - **If** the proposal is within bounds and Compliance passes: marked `READY_FOR_REVIEW`, ready for Marco's routine sign-off.
   - **If** the proposal exceeds bounds or Compliance flags a concern (or if grounding drops below threshold, triggering the last compliance-approved rate table): marked `ESCALATED` with full state attached, withholding any approved status and flagging operational fallback (`fallback_active = True`) if triggered.
5. Synchronous Underwriter Briefing: generated via deterministic template (Phase 6) or bounded-timeout LLM with multi-layer validator (post-Phase 7), ensuring a complete narrative before response (INV-7).
6. **Acceptance criteria:** the audit record for either path is complete (causal justification, evidence citation, financial projection, final disposition, briefing narrative, validation outcome), reconstructable without live re-querying, and the Compliance Agent's citation is independently verifiable against the source corpus.

## 11. Success Metrics & KPIs

- Elasticity model: recovery of the known segment-varying synthetic treatment effect under the approved held-out, multi-seed protocol, with per-segment and Spearman metrics, finite/self-consistent 95% intervals, sign agreement, and negative/positive controls; numeric thresholds are frozen after Q3 calibration. GLM baseline calibration: holdout mean Tweedie deviance and actual-versus-expected by predicted-loss decile and segment; MAE/RMSE are secondary metrics.
- Bandit: cumulative regret vs. static-price baseline (simulated).
- Compliance Agent: four-metric diagnostic matrix (Faithfulness, Answer Relevancy, Context Recall, Context Precision), read jointly per ADR-010 — no single metric gates alone; LLM-as-judge agreement rate.
- Governance: HITL escalation rate (should reflect genuine ambiguity, not over- or under-triggering).
- Underwriting result: absolute $Rev - Losses$ as the elasticity-sensitive headline (the loss ratio is $\epsilon$-invariant under the homogeneity assumption).
- Briefing generation: validator-pass rate and template-fallback rate (tracked in evaluation and OTel telemetry).
- System: end-to-end decision latency; audit-record completeness rate (target: 100%).

## 12. Out of Scope (Future Iterations)

- A separate claims-severity model or severity-driven pricing in v1; capped loss cost from the baseline Tweedie GLM is in scope as the reference loss-cost output.
- Multi-jurisdiction, multi-state simultaneous filing logic.
- Live carrier core system integration (API contracts modeled, not connected).
- Autonomous rate publication without human sign-off.
- **A full production underwriting UI** — a real, daily-use console for Marco's persona, with full input validation, arbitrary-policy lookup, and multi-user access control, remains out of scope. This is distinct from the showcase interface (ADR-009): a minimal, explicitly-labeled glass-box demo built for the PRD §5 secondary audience (evaluators, not underwriters), exposing curated preset scenarios rather than production-grade arbitrary input. Under INV-5, the standalone Phase 2 developer showcase / model diagnostic UI is explicitly exempted from requiring Compliance and Impact context, provided the UI is labeled as a raw model validation demo and not a pricing recommendation. The diagnostic view may show $\epsilon$ estimates and recovery metrics, but never a proposed rate adjustment ($\Delta$). INV-5 strictly binds any rate recommendation or decision output.

## 13. Dependencies & Constraints

- Depends on public dataset and regulatory corpus availability and quality; no proprietary carrier data is used.
- Depends on LLM provider availability, mediated through the Gateway's fallback chain.
- Solo-practitioner build capacity constrains scope; phase gating (`scoping_doc.md` §2) and the `src/` line-count ceiling are the primary controls.
- No live production deployment is in scope; all "production-grade" claims refer to architecture and engineering discipline, not live operational status.

## 14. Open Questions / Risks (To Be Revisited After Implementation)

- How sensitive is the causal elasticity estimate to unmeasured confounding in the public dataset, and does the sensitivity analysis hold up under adversarial review?
- Are the four diagnostic-matrix thresholds (Faithfulness, Answer Relevancy, Context Recall, Context Precision) for the Compliance Agent strict enough, individually and jointly, to catch subtle regulatory misapplications — not just obviously ungrounded citations?
- Does the HITL escalation rate, once measured against the synthetic stream, suggest the bounded corridor is calibrated correctly, or does it need tightening/loosening?
- Will the HITL escalation rate metric remain meaningful once real (not synthetic) production data is introduced, given the synthetic stream's known generalization limits (see `scoping_doc.md` §11)?
