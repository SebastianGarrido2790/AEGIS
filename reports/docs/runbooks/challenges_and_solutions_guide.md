# Challenges & Solutions Guide — AEGIS *(working title)*

**Actuarial Elasticity & Governance Intelligence System**
Author: Sebastián Garrido Arévalo | Date: August 13, 2026

This is the master engineering procedures and troubleshooting manual for AEGIS. Each entry records a challenge domain, the problem/symptom observed, its root cause, the implemented solution, and the governing authority (rule, ADR, or PRD/Charter section) that justified the resolution.

**Status note:** as of Phase 0, no implementation challenges have been encountered yet — this document is seeded below with entries **anticipated** from the Charter §9 and PRD §14 risk registers, so the structure is ready to receive real entries from Phase 1 onward. Anticipated entries are marked `[ANTICIPATED]` and must be replaced or confirmed with an actual resolution once encountered during implementation; they are not to be treated as already resolved.

---

## Entry Template

| Field | Description |
|---|---|
| Challenge Domain | The subsystem or concern area (e.g., Causal Inference, RAG Retrieval, Governance) |
| Problem / Symptom | What was observed |
| Root Cause | Why it happened |
| Implemented Solution | What was actually done to resolve it |
| Governing Authority | The rule, ADR, or planning-doc section that justifies the resolution |

---

## `[ANTICIPATED]` Entry 1 — Causal Inference

| Field | Detail |
|---|---|
| Challenge Domain | Causal Inference / Elasticity Modeling |
| Problem / Symptom | Elasticity estimates that look plausible but may reflect unmeasured confounding rather than a genuine treatment effect. |
| Root Cause | The public dataset is observational, not experimental; confounders affecting both price and retention may not be fully captured in available features. |
| Implemented Solution | *(To be confirmed in Phase 2.)* Planned mitigation: a formal sensitivity analysis on the causal estimate, and treating the model's output strictly as a recommendation input to the agentic layer — never as ground truth passed through unchallenged. |
| Governing Authority | Charter §9 (risk register); PRD §14 (open question) |

## `[ANTICIPATED]` Entry 2 — RAG Grounding

| Field | Detail |
|---|---|
| Challenge Domain | Regulatory Compliance Agent / RAG Retrieval |
| Problem / Symptom | The Compliance Agent could generate a plausible-sounding but ungrounded or misattributed regulatory citation. |
| Root Cause | Standard RAG failure mode — retrieval returning topically related but not sufficiently authoritative chunks, or the generation step drifting from the retrieved evidence. |
| Implemented Solution | *(To be confirmed in Phase 5.)* Planned mitigation: a dedicated retrieval-quality evaluation harness (groundedness, evidence coverage) gating the Compliance Agent's release, an upstream Great Expectations suite rejecting malformed/duplicate corpus chunks before indexing (ADR-003), and a deterministic fallback to the last compliance-approved rate table if the Compliance Agent's confidence falls below threshold. |
| Governing Authority | ADR-003 (`system_design.md`); Charter §9; PRD §6 (Functional Requirement 6) |

## `[ANTICIPATED]` Entry 3 — Governance Calibration

| Field | Detail |
|---|---|
| Challenge Domain | HITL Escalation Gate |
| Problem / Symptom | The escalation rate could be miscalibrated — too high, creating a review bottleneck that defeats the system's purpose; too low, letting risky proposals through with insufficient scrutiny. |
| Root Cause | The bounded exploration corridor and escalation thresholds are initially set from design assumptions, not observed production behavior. |
| Implemented Solution | *(To be confirmed in Phase 7.)* Planned mitigation: measure the HITL escalation rate against the synthetic production-analog stream as an explicit exit-criterion check, and treat threshold tuning as an expected post-launch iteration, not a one-time setting. |
| Governing Authority | Charter §6 (Large-Scale Costs — human review capacity); PRD §14 (open question) |

## `[ANTICIPATED]` Entry 4 — Scope Management

| Field | Detail |
|---|---|
| Challenge Domain | Solo-Practitioner Build Scope |
| Problem / Symptom | Three agents, two ML sub-layers, and a full evaluation/observability layer represent a large surface for a single contributor; risk of an unfinished or over-scoped system. |
| Root Cause | The system's governance and reliability requirements (Gateway, RAG evaluation, drift detection, audit logging) are non-negotiable given the regulated domain, leaving few areas to trim without compromising the project's core thesis. |
| Implemented Solution | Strict phase gating (no phase begins before the prior phase's exit criteria are met) and a hard `src/` line-count ceiling per module, both enforced from Phase 1 onward. |
| Governing Authority | Charter §9; Technical Roadmap (phase-gated structure) |

---

*New entries should be appended below as they are actually encountered during implementation, following the template above. Anticipated entries above should be updated in place — not duplicated — once their real resolution is known.*
