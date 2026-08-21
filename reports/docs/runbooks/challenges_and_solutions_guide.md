# Challenges & Solutions Guide — AEGIS *(working title)*

**Actuarial Elasticity & Governance Intelligence System**
Author: Sebastián Garrido Arévalo | Date: August 21, 2026

This is the master engineering procedures and troubleshooting manual for AEGIS. Each entry records a challenge domain, the problem/symptom observed, its root cause, the implemented solution, and the governing authority (rule, ADR, or PRD/Charter section) that justified the resolution.

**Status note:** Phase 1 implementation challenges and post-implementation review remediations are documented below in Section 2. Section 1 retains remaining entries **anticipated** from the Charter §9 and PRD §14 risk registers for Phases 2–7. Anticipated entries are marked `[ANTICIPATED]` and must be replaced or confirmed with an actual resolution once encountered during implementation; they are not to be treated as already resolved.

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

## 1. `[ANTICIPATED]` Entries (Phases 2–7)

### `[ANTICIPATED]` Entry 1 — Causal Inference

| Field | Detail |
|---|---|
| Challenge Domain | Causal Inference / Elasticity Modeling |
| Problem / Symptom | Elasticity estimates that look plausible but may reflect unmeasured confounding rather than a genuine treatment effect. |
| Root Cause | The public dataset is observational, not experimental; confounders affecting both price and retention may not be fully captured in available features. |
| Implemented Solution | *(To be confirmed in Phase 2.)* Planned mitigation: a formal sensitivity analysis on the causal estimate, and treating the model's output strictly as a recommendation input to the agentic layer — never as ground truth passed through unchallenged. |
| Governing Authority | Charter §9 (risk register); PRD §14 (open question) |

### `[ANTICIPATED]` Entry 2 — RAG Grounding

| Field | Detail |
|---|---|
| Challenge Domain | Regulatory Compliance Agent / RAG Retrieval |
| Problem / Symptom | The Compliance Agent could generate a plausible-sounding but ungrounded or misattributed regulatory citation. |
| Root Cause | Standard RAG failure mode — retrieval returning topically related but not sufficiently authoritative chunks, or the generation step drifting from the retrieved evidence. |
| Implemented Solution | *(To be confirmed in Phase 5.)* Planned mitigation: a dedicated retrieval-quality evaluation harness (groundedness, evidence coverage) gating the Compliance Agent's release, an upstream Great Expectations suite rejecting malformed/duplicate corpus chunks before indexing (ADR-003), and a deterministic fallback to the last compliance-approved rate table if the Compliance Agent's confidence falls below threshold. |
| Governing Authority | ADR-003 (`system_design.md`); Charter §9; PRD §6 (Functional Requirement 6) |

### `[ANTICIPATED]` Entry 3 — Governance Calibration

| Field | Detail |
|---|---|
| Challenge Domain | HITL Escalation Gate |
| Problem / Symptom | The escalation rate could be miscalibrated — too high, creating a review bottleneck that defeats the system's purpose; too low, letting risky proposals through with insufficient scrutiny. |
| Root Cause | The bounded exploration corridor and escalation thresholds are initially set from design assumptions, not observed production behavior. |
| Implemented Solution | *(To be confirmed in Phase 7.)* Planned mitigation: measure the HITL escalation rate against the synthetic production-analog stream as an explicit exit-criterion check, and treat threshold tuning as an expected post-launch iteration, not a one-time setting. |
| Governing Authority | Charter §6 (Large-Scale Costs — human review capacity); PRD §14 (open question) |

### `[ANTICIPATED]` Entry 4 — Scope Management

| Field | Detail |
|---|---|
| Challenge Domain | Solo-Practitioner Build Scope |
| Problem / Symptom | Three agents, two ML sub-layers, and a full evaluation/observability layer represent a large surface for a single contributor; risk of an unfinished or over-scoped system. |
| Root Cause | The system's governance and reliability requirements (Gateway, RAG evaluation, drift detection, audit logging) are non-negotiable given the regulated domain, leaving few areas to trim without compromising the project's core thesis. |
| Implemented Solution | Strict phase gating (no phase begins before the prior phase's exit criteria are met) and a hard `src/` line-count ceiling per module, both enforced from Phase 1 onward. |
| Governing Authority | Charter §9; Technical Roadmap (phase-gated structure) |

---

## 2. Phase 1 — Repository & Data Contract Setup (Resolved)

### Entry 1.1 — Data Contract Rule Isolation (R-1)

| Field | Detail |
|---|---|
| Challenge Domain | Data Contracts & Verification Testing (Gate 3) |
| Problem / Symptom | `regulatory_invalid_duplicate.json` simultaneously tripped two expectations (`expect_column_values_to_be_unique` on `chunk_id` and on `chunk_text`), violating Gate 3's requirement that each malformed fixture trip exactly one named rule. |
| Root Cause | The test fixture duplicated both the metadata ID and the entire statutory clause text across rows. |
| Implemented Solution | Differentiated `chunk_text` across duplicate ID rows, ensuring solely the `chunk_id` uniqueness check fires. Updated unit test assertions to enforce `len(failed_expectations) == 1` and verify that the failed column is strictly `chunk_id`. |
| Governing Authority | ADR-006 (`system_design.md`); Phase 1 Execution Workflow (Gate 3); PRD §11 |

### Entry 1.2 — Configuration Schema Forward-Specification (R-2)

| Field | Detail |
|---|---|
| Challenge Domain | Configuration Management (Stage 2) |
| Problem / Symptom | Forward-declaring sections for unbuilt tiers (`gateway:`, `tier1_ml:`, `tier2_agents:`, `governance:`) with required fields forced premature vendor and model selections in `params.yaml` three phases ahead of their implementation. |
| Root Cause | Pydantic schema declared required fields without default factories for unbuilt stages. |
| Implemented Solution | Configured `default_factory` on root `AEGISConfig` and explicit defaults on tier models so unbuilt tiers remain optional at startup, while keeping `DataContractsConfig` strictly required. Explicitly recorded this rationale in `phase_1_implementation_plan.md` (Part E). |
| Governing Authority | ADR-005 (`system_design.md`); Phase 1 Implementation Plan (Part E) |

### Entry 1.3 — Decorative DVC Configuration Cleanup (R-3)

| Field | Detail |
|---|---|
| Challenge Domain | Pipeline Orchestration (Stage 5) |
| Problem / Symptom | `DVCConfig` in `schema.py` and `params.yaml` carried stage name literals (`stage_elasticity_ingest`, etc.) that were never referenced in pipeline code, creating an illusion of synchronization with `dvc.yaml`. |
| Root Cause | Premature configuration modeling of DVC stage names before DAG definition. |
| Implemented Solution | Removed all decorative stage name fields from `DVCConfig` and `params.yaml`, retaining only the actual configuration keys: `remote_name` and `remote_url`. |
| Governing Authority | ADR-007 (`system_design.md`); PRD §12 (Simplicity & Local-first Posture) |

### Entry 1.4 — CLI Parameter Drift Mitigation (R-4)

| Field | Detail |
|---|---|
| Challenge Domain | Pipeline CLI Tooling (Stage 5) |
| Problem / Symptom | `src/aegis/pipelines/cli.py` hardcoded default fixture and suite paths in `build_parser()`, introducing a risk of silent configuration drift from `params.yaml`. |
| Root Cause | Hardcoded string literals in argparse argument defaults rather than resolving from the configuration loader. |
| Implemented Solution | Refactored `cli.py` so that path arguments default to `None` and are dynamically resolved via `get_config()` within execution functions, maintaining `params.yaml` as the sole source of truth. |
| Governing Authority | ADR-005 (`system_design.md`); INV-8 Design Principles |

---

*New entries should be appended below as they are actually encountered during implementation, following the template above. Anticipated entries above should be updated in place — not duplicated — once their real resolution is known.*
