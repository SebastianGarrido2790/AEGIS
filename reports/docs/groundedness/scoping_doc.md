# Scoping Document — AEGIS

**Actuarial Elasticity & Governance Intelligence System**

*This document is what the project is, why it exists, and how to use it: the single strategic source defining the problem space, goals, constraints, and success metrics for AEGIS. Read this before `prd.md`; it is the source material `prd.md`, `user_story.md`, and `../architecture/system_design.md` were built from and are checked against.*

Author: Sebastián Garrido Arévalo | Date: September 28, 2026

---

## 0. Project Summary

- **Problem:** Surface problem — "our pricing team can't adjust rates fast enough to keep up with elasticity and competitive shifts." Real engineering problem — there is no shared, auditable decision substrate connecting causal pricing science, regulatory compliance validation, and financial impact assessment; each function operates in an isolated tool or spreadsheet, so speed and auditability trade off against each other by design.
- **Why it matters / impact:** demonstrates — and is intended to be judged as evidence of — the ability to design a governed agentic platform where deterministic ML, RAG-grounded compliance reasoning, and human oversight are integrated by construction, not reconciled after the fact. This is the core positioning this entire portfolio is built around.
- **Proposed approach:** a three-tier hybrid system. Tier 1 (deterministic): a causal elasticity model plus a GLM actuarial baseline. Tier 2 (agentic): a LangGraph workflow coordinating a deterministic Pricing Strategy node, a RAG-grounded Regulatory Compliance agent (the sole decision LLM), and a deterministic Revenue/Loss-Ratio Impact node, followed by deterministic Governance and a bounded post-governance executive Briefing narrative (ADR-011). Tier 3 (governance): a HITL escalation gate, a deterministic fallback, and a structured audit log.
- **Primary success metric:** *(amended, per the Phase 2 causal-validation finding recorded below)* — not "the causal model outperforms the GLM baseline," but that the causal estimator demonstrably recovers a known, segment-varying synthetic treatment effect, evidenced by per-segment and rank-based (Spearman) correlation — metrics robust to the heavy-tailed leverage points Phase 2 found in the real feature distribution — rather than a single aggregate Pearson correlation on one seed.
- **Key risks / unknowns:** causal identification rests on a semi-synthetic benchmark with constructed ground truth over empirical freMTPL2 covariates, since no real price variation or renewal experiment exists in the public dataset; correlation-based estimator validation was found empirically to be sample-size- and outlier-sensitive during Phase 2's remediation, not a settled, stable metric; RAG grounding (Tier 2, not yet built) remains the system's designated highest-severity failure mode; solo-practitioner scope is a standing, now well-evidenced constraint — the Phase 2 remediation cycle alone ran considerably longer than originally scoped.

## 1. Business / User Problem

**Primary users:** actuarial/pricing analysts and underwriting/compliance reviewers (in-universe personas Elena, Marco, Renata — `user_story.md`).
**Secondary users:** hiring managers, technical evaluators, and academic reviewers assessing this as portfolio evidence (`prd.md` §5; ADR-009's showcase interface exists specifically for this audience).

**Pain point:** a rate change justified by the data still has to travel physically between an actuarial desk, a compliance/legal desk, and a finance desk, in sequence, each waiting on the last — a process that takes weeks, and produces no single record of why a given change was approved or rejected.

**Why an agentic approach, not traditional software or classical ML alone:** the task requires synthesizing three distinct domains of judgment — actuarial, regulatory, financial — into one governed decision before a human acts on it, where the regulatory piece specifically requires reasoning over free-text source material a fixed rule engine cannot generalize over. **This question is now resolved:** the companion Pre-Build Validation Gates (`../evaluations/capability_profile_gates.md`, amended by ADR-011) confirm that probabilistic reasoning is required strictly at the Compliance node for free-text statutory interpretation, while Pricing Strategy and Financial Impact are deterministic code nodes (Option C), completed by a post-governance underwriter briefing narrative.

## 2. Goal

1. **Tier 1:** the causal elasticity estimator recovers a known synthetic treatment effect with defensible per-segment correlation and Spearman rank-correlation metrics, and self-consistent, finite confidence intervals (point estimate contained in its own reported interval).
2. **Tier 2 (Phase 6):** the LangGraph orchestration produces a complete, auditable recommendation trace for the Primary Scenario (`prd.md` §10) on both the auto-approved and escalated branches.
3. **Tier 3 (Phase 7):** 100% audit-record completeness across a test batch spanning both escalated and auto-approved scenarios.
4. **Showcase (ADR-009, incremental across Phases 2/6/7):** a glass-box demo interface that visibly distinguishes a governed outcome — an approval, an escalation, a fallback — for the secondary audience, never presented as a production interface.

Tied to the phase-gated roadmap (`../implementation_roadmap.md`, Phases 1–9); no phase begins before the prior phase's exit criteria are demonstrated, not merely coded.

## 3. End State

AEGIS is a governed, three-tier agentic platform that recommends personal auto insurance premium adjustments by combining a causal elasticity model, a RAG-grounded regulatory compliance agent, and a deterministic financial impact tool — escalating to a human underwriter whenever a recommendation falls outside pre-approved bounds. Every recommendation carries its causal justification, its compliance-evidence citation, its projected financial impact, and a validated briefing narrative in one structured, auditable decision record.

**Definition of done:** the causal estimator recovers the known synthetic effect with defensible confidence intervals *(amended wording — see §0)*; the Compliance Agent meets its four-metric diagnostic-matrix threshold (ADR-010); the bandit beats a static-price baseline; 100% audit-record completeness; the full system reproducible end-to-end via CI and `docker-compose up`.

## 4. Stakeholders

- **Sebastián Garrido Arévalo** — sole practitioner, acting as architect, ML engineer, and agentic systems engineer. No PM, design, data, or legal function exists on this project, by design — this is itself a standing scope constraint (§12), not an oversight.
- **Academic stakeholder:** the certificate instructor evaluating this project as coursework-adjacent portfolio evidence.
- **Portfolio/hiring stakeholder:** the secondary audience named in §1 — technical evaluators and recruiters, served directly by the ADR-009 showcase interface.

## 5. The ROI Situation

Brutally honest: as a solo portfolio project, this system generates no direct revenue. Its ROI is demonstrative — a reusable architectural template proving that causal inference, RAG-grounded compliance, and HITL governance can be integrated into one auditable pipeline. If deployed inside a real insurer, the ROI case would rest on cycle-time reduction and reduced compliance exposure — a claim this project cannot substantiate without real loss-ratio data it does not have.

## 6. Prior Work

- **Domain precedent:** frequency-severity GLM decomposition on the freMTPL2 benchmark is a well-established actuarial case-study convention; the Tier 1 baseline for the redo follows this precedent.
- **Portfolio precedent:** ACRAS (Agentic Credit Risk Assessment System), a sibling project in this same portfolio, applies the same deterministic-ML-plus-agentic-orchestration thesis to a different regulated domain (credit risk) via a relay-agent pattern. AEGIS extends the thesis with a heavier emphasis on RAG-grounded regulatory compliance and a parallel fan-out (not relay) orchestration topology.
- **No existing system** addresses this specific problem — a governed, auditable link between causal elasticity, regulatory compliance, and financial impact — directly; today's heuristic solution is the disconnected manual review chain described in §1.

## 7. Input and Output

- **Input:** a policy/segment submitted for pricing review (driver age, vehicle age/power, bonus-malus, exposure, region, and related rating factors); regulatory corpus documents (jurisdiction, section, effective date, chunk text) feeding the Compliance Agent's retrieval index (Phase 5, not yet built).
- **Output:** a structured recommendation record — proposed premium adjustment, causal justification with confidence interval, compliance verdict with evidence citation, financial impact projection, post-governance briefing narrative, and final disposition (auto-approved-for-review or escalated). The system never outputs a published rate directly (INV-4).
- **Storage:** DVC-versioned training and regulatory data; MLflow-tracked and registered models; a SQLite-backed structured audit log (Tier 3, Phase 7); a Redis Stack-backed regulatory vector index (Phase 5).

## 8. Constraints

| Constraint | Target | Notes |
|---|---|---|
| Latency (p50 / p99) | **Not yet set** | `prd.md` §7 states a qualitative target only ("low single-digit seconds for the agentic path"); no hard p50/p99 SLA has been formally set. Open item for Phase 4+ NFR refinement — not invented here to fill the table. |
| Cost per request | **Not yet set** | Zero marginal cost today by construction — Tier 1 makes no LLM calls. Once the Gateway exists (Phase 4), cost is tracked via its OTel cost export, but no target $/request ceiling has been decided. |
| Quality bar | Tier 1: calibrated per-segment/Spearman recovery threshold (in progress). Tier 2 (future): four-metric diagnostic-matrix threshold, Faithfulness/Answer Relevancy/Context Recall/Context Precision (ADR-010) | Threshold values themselves are Phase 5/6 work, not yet set. |
| Privacy / compliance | Zero committed secrets (INV-1); single-jurisdiction disclaimer on every Compliance verdict (INV-9); no live carrier-system integration (INV-10) | All three already enforced as standing invariants, not aspirational. |
| Availability / uptime | **Not applicable** | No live production deployment is in scope (`prd.md` §13) — this is a portfolio system, not a service with an uptime SLA. |

## 9. Success Metrics

- **User-facing:** not applicable in the strict sense — no live users. The closest analog is the showcase interface's ability to let a technical evaluator directly observe governance mechanics (an escalation firing, a fallback engaging) within a single demo session.
- **Technical:** *(amended)* causal estimator's per-segment and Spearman recovery of the known synthetic effect; GLM baseline calibration (MAE/RMSE, finite and self-consistent confidence intervals); the future four-metric Compliance Agent diagnostic matrix; judge calibration (Spearman correlation, MAE, bias) for any continuous-score judge feeding a gate or a dashboard.
- **System:** 100% audit-record completeness (Tier 3); full CI and DVC reproducibility; the module-size ceiling (INV-8) held everywhere.

## 10. Checklist

- [x] Evidence the problem is worth solving — a real, recognized inefficiency in the actuarial rate-review cycle, and a recognized regulatory-fairness sensitivity in insurance pricing generally.
- [x] Evidence an agentic approach is the right tool — closed by the companion Pre-Build Validation Gates (`../evaluations/capability_profile_gates.md`): Option C governed workflow validated (single Compliance LLM agent + deterministic specialist nodes + post-governance briefing narrative; ADR-011), open gaps recorded there.
- [x] Goals and metrics align with stakeholder priorities — aligned with the sole practitioner's stated career-positioning goal (governed agentic systems engineering in regulated domains).
- [x] Scope shared with relevant stakeholders — limited in practice to the sole practitioner and the academic stakeholder; noted honestly rather than skipped.
- [x] Decisions and trade-offs recorded in the decision log — `../architecture/system_design.md`'s ADR-001 through ADR-011 (and counting), plus the `decisions/` and `phases/` pillars.

## 11. What Could Go Wrong Along the Way

Three items below were promoted from hypothetical to **confirmed, evidenced** based on what Phase 2 actually surfaced:

- **Confounding in causal estimation on observational data — CONFIRMED, not merely anticipated.** Phase 2's real feature data turned out to be heavily heavy-tailed (`risk_index` kurtosis 222, top 0.1% of rows carrying 19% of its variance), and the causal estimator's correlation-to-ground-truth metric proved unstable and sample-size-sensitive across seeds as a direct consequence. This is no longer a theoretical risk-register line; it happened.
- **RAG grounding failure** (Tier 2, not yet built) remains the system's designated highest-severity risk (`../architecture/system_design.md`, ADR-010).
- **Solo-practitioner scope risk — CONFIRMED at real cost.** The Phase 2 remediation cycle ran substantially longer, and through substantially more review rounds, than originally scoped — direct evidence of this constraint, not a hedge.
- **Synthetic-data generalization limits — CONFIRMED by design, not a residual risk.** The causal validation exercise cannot and does not claim real-world elasticity discovery; this is the entire reason for the Option B amendment to `prd.md` §11 recorded alongside this document.
- **Governance threshold miscalibration** (Tier 3, not yet built) remains an open, anticipated risk.
