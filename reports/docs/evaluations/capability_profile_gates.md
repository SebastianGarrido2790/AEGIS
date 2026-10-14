# Capability Profile Gates — AEGIS Tier 2 (Agentic Orchestration Layer)

**What this document is:** the Pre-Build Validation Gates and Complexity Diagnostic for AEGIS's Tier 2 agentic layer (the LangGraph three-agent orchestration) — the record of whether an agent is justified here at all, and at what complexity level.
**Why it exists:** these questions are meant to be answered before planning documents specify how to build the agent — to decide whether an agent is justified at all and at what complexity level.
**How to use it:** as the record of that decision. Where this exercise finds AEGIS already correctly scoped, it says so and moves on. Where it finds a gap, the gap is stated plainly and left open — not quietly fixed by this document.

**Honest sequencing note, stated rather than glossed over:** this gate exercise is being completed *after* AEGIS's Tier 2 architecture already exists in `system_design.md` and `prd.md` — not before, as the gate sequencing intends. That's out of order. The value of doing it now is that it can check the *existing* design against the gate's questions honestly, including finding real gaps the original build didn't surface — which is what happened here (see Part 1, Gate 5, and Part 3, Alignment). It is not, and should not be read as, a claim that this was done in the right order the first time.

---

## Part 1 — Pre-Build Validation Gates

### 1. Measurable Success

The Primary Scenario (`prd.md` §10) runs end-to-end on both branches — auto-approved-for-review and escalated — each producing a complete audit record (INV-7, target: 100% completeness). Supporting metrics: HITL escalation rate reflecting genuine ambiguity rather than over- or under-triggering; the Compliance Agent's four-metric diagnostic matrix (ADR-010) meeting its threshold once Phase 5/6 ships.

### 2. Blast Radius / Worst-Case Failure

Worst case: the system recommends a regulatorily non-compliant or financially unsound premium adjustment, and that recommendation reaches a published state without human review. This is structurally bounded two ways already on record: INV-4 (the governance output schema has no "published" state at all — publication is a manual, out-of-system action) and INV-6 (a grounding failure below threshold triggers fallback and escalation, never a guess). The single highest-severity variant of this failure — a hallucinated compliance citation producing a false "pass" — is already named as the system's worst failure mode in the risk register (see `scoping_doc.md` §11) and ADR-010.

### 3. Session Economics & Cost Ceiling

**Not yet set.** This gate exercise doesn't resolve it — it confirms the gap already flagged in `scoping_doc.md` §8's constraints table ("Cost per request: Not yet set"). Recommendation, not a decision: a per-session cost ceiling should be set once the Gateway's OTel cost export exists (Phase 4), using real token-cost data rather than an estimate made before the Gateway is built. Tracked as an open item, not resolved here.

### 4. Evaluation Harness

Specified, not yet built. The golden dataset schema (`FailureMode` taxonomy, `GoldenCase` dataclass) is already architecturally committed — see Part 5 below and the planned `src/evaluation/golden_schema.py` — but Phase 5/8 is where it's actually implemented. This gate is satisfied at the specification level; implementation status is tracked honestly as pending, not implied complete.

### 5. Graceful Off-Switch

**Not addressed anywhere in AEGIS's current documentation.** This is the first genuine, previously-undocumented gap this exercise surfaces, not a restatement of something already known. Neither `system_design.md` nor any other current project document specifies who can halt Tier 2 agent invocations, how fast, with what in-flight-state preservation, or what re-enabling requires. INV-4 and the HITL gate bound what the system can *decide*; neither says anything about stopping the system from running at all. Recommendation: this needs a dated ADR before Phase 6 ships Tier 2 for real, not a silent fix folded into this document.

---

## Part 2 — The Four-Level Complexity Diagnostic

**Classification: Level 3 — Bounded Agent.**

Walking the criteria directly, against the actual design already committed in `system_design.md`:

- **Level 1 (Static Prompt):** doesn't fit — three agents, multi-step synthesis; no single call produces the output.
- **Level 2 (Deterministic Workflow):** close, but doesn't fully fit — the *inter-node* topology is fixed (Pricing Strategy → [Compliance ‖ Revenue Impact] → Governance), but *within* the Compliance node, the model itself decides which regulatory chunks to retrieve. That's model-driven tool selection, which Level 2 explicitly excludes ("the model produces intermediate text but has no say in routing or tool choice").
- **Level 3 (Bounded Agent):** fits on every criterion. The right next step (auto-approve vs. escalate) depends strictly on what Compliance and Impact returned. The toolset is small and constrained (a retrieval tool, a financial-calculation tool — nowhere near 15). The step budget is small and fixed (three to four nodes, not an open-ended loop). Wrong actions are low-stakes *by construction* — INV-4 means a wrong recommendation can never reach a live rate table; it can only ever reach HITL review.
- **Level 4 (Full Autonomous Agent):** doesn't fit and was never claimed — no 30+ step horizons, no persistent cross-session memory (each review is explicitly single-shot by design), no large or dynamic toolset.

**What this retroactively validates:** the three-agent, parallel-branch design already built is the correct complexity level for this problem — not an over-engineered multi-agent system reaching for sophistication it doesn't need. That's a genuine finding, not a foregone conclusion; a project this deep into a regulated-domain, governance-heavy build was a real candidate for `Level 4` over-scoping, and it didn't happen.

---

## Part 3 — The Capability Profile Matrix

Filled in against AEGIS's actual Tier 2 design. Blank rows are stated as deliberate non-selections with a reason, not silently skipped.

| Capability | Selected Pattern(s) | AEGIS-Specific Notes |
|---|---|---|
| **1. Perception** | *(none selected)* | Policy/segment input is already typed (Pydantic) at the tool boundary; regulatory-document structuring (jurisdiction, section, effective date) is a deterministic, GX-gated ingestion step (INV-3), not an LLM-driven perception agent. Consistent with the Brain/Brawn separation — this capability doesn't need an agent here. |
| **2. Reasoning** | **CoT Auditor** *(adapted)* | AEGIS doesn't implement a literal chain-of-thought trace audit — it implements the four-metric diagnostic matrix (Faithfulness, Answer Relevancy, Context Recall, Context Precision; ADR-010) as its auditable-reasoning mechanism for the Compliance Agent. Named here as the closest-fit pattern, not a literal off-the-shelf match. |
| **3. Planning** | **Plan-Then-Execute** | The topology is fixed at design time, not discovered at runtime — no ReAct loop, no adaptive replanning. |
| **4. Memory** | *(none selected)* | Deliberate, not an oversight: each review is single-shot (no session continuity needed), and the persistent audit log (INV-7) is a write-once compliance record, not memory fed back into future reasoning. |
| **5. Tool Use** | **Tool Selector**, **Side-Effect Auditor** | The Compliance Agent selects which regulatory chunks to retrieve. Every recommendation producing a complete, structured audit record (INV-7) is functionally what a Side-Effect Auditor does — wrapping the decision with pre/post state capture. |
| **6. Coordination** | **Pipeline Orchestrator**, **HITL Liaison** | The LangGraph coordinator is a fixed pipeline with a parallel fan-out/fan-in, not a dynamic Supervisor-Worker delegation. The Governance node's escalation gate is a direct HITL liaison. |
| **7. Learning** | **Feedback Loop** *(planned, not built)* | Matches the Phase 8 production harvest loop that routes sub-threshold traces to a human-reviewed golden-dataset candidate queue. Selected here as the target architecture; not yet implemented, and not claimed to be. |
| **8. Alignment** | **Constitution-Bound**, **Provenance Tracker**, **Explainer Agent**, **Drift-Detector** *(planned)* | Constitution-Bound maps directly to the Non-Negotiable Invariants (see `system_design.md` ADR log) acting as a hard, un-negotiable constraint set. Provenance Tracker maps to INV-7 and the Compliance Agent's evidence citations. Explainer Agent maps to INV-5 — the causal justification, compliance evidence, and financial impact bundle attached to every recommendation. Drift-Detector matches the planned Phase 8 feature-distribution monitoring. |

**Two deliberate non-selections in Alignment, surfaced as gaps rather than silently omitted:**

- **Off-Switch-Compatible — not selected, and this is the same gap Part 1 Gate 5 already found.** Two independent parts of this exercise landing on the same missing piece is a stronger signal than either alone — it's a real, load-bearing absence.
- **Red-Team Auditor — not selected.** No adversarial test suite exists for AEGIS's Compliance Agent yet (the kind of 20–50-attack corpus a production safety review calls for) — recorded here as a tracked open item, not fixed here.
- **Privacy-Preserving — not selected, and deliberately so rather than a gap.** AEGIS's current scope operates on rating-factor and segment data, not named-individual PII — this capability isn't yet a modeled concern. Worth revisiting explicitly if the project ever moves beyond synthetic/benchmark data toward anything resembling real customer records.

---

## Part 4 — Key Structural Interfaces

### 4.1 The Validator Sandwich

Already implemented, not re-specified here: the Top Bun is `src/utils/sanitization.py` (prompt-injection defense); the Bottom Bun is two-layer validation — Pydantic schema first, then a semantic/business-rule guardrail — applied to every agent output before it reaches Governance or a caller.

### 4.2 Reversibility and Rollback

Two distinct mappings, not one:

- **The Compliance Agent's "action" (a verdict):** the pre-action state snapshot is the last compliance-approved rate table itself; the inverse operation is INV-6's deterministic fallback to that same snapshot.
- **The elasticity model's promotion (a Tier 1 artifact, not a Tier 2 agent action):** the MLflow registry's staged versioning is the state capture; rollback is a registry stage-tag flip, gated by required human review of the calibration report before promotion.

---

## Part 5 — Reference Benchmark & Evaluation Schema

The golden dataset schema, stated here in full so this document stands alone:

- **`FailureMode` taxonomy (closed set):** hallucination, retrieval miss, context ignored, multi-hop failure, safety violation, tool misuse, reasoning incoherence, format failure, latency failure.
- **`GoldenCase` fields:** stable `id`; `query`; `expected_context` (documents that should be retrieved — required for the Context Recall check); `ideal_answer` and/or `answer_criteria`; `must_not_include` negative constraints; the `failure_modes` it targets; and a non-empty `added_reason` recording why the case exists.

The authoritative implementation lives in the planned `src/evaluation/golden_schema.py` (Phase 5/8); this section is the specification that implementation must satisfy.

---

## Part 6 — System Blueprint Comparison

AEGIS's safety layering nests, outermost to innermost: Off-Switch-Compatible → Constitution-Bound → Deterministic Harness. AEGIS's actual design has the inner two layers solidly in place — Constitution-Bound (the Non-Negotiable Invariants, see `system_design.md`) wrapping a Deterministic Harness (the compiled LangGraph). **The outermost layer, Off-Switch-Compatible, is missing entirely** — the same finding as Part 1 Gate 5 and Part 3's Alignment row, surfaced a third time from a third angle. Three independent parts of this exercise converging on the same gap is the clearest single conclusion this document reaches.

---

## Summary

**What this exercise confirms was already right:** the three-agent, parallel-branch Tier 2 design is correctly scoped at Level 3 (Bounded Agent) — not multi-agent theater (merging any two agents would lose something real: retrieval vs. calculation specialization, distinct Compliance/Governance authority, independently-retrieved evidence, parallel execution, and isolated grounding-failure fallback), and not over-built toward Level 4 capabilities the problem doesn't need.

**What this exercise found that wasn't previously documented:**
1. No off-switch/kill-switch mechanism exists for Tier 2 (found three separate ways — Part 1 Gate 5, Part 3, Part 6).
2. No per-session or per-request cost ceiling has been set (Part 1 Gate 3) — consistent with, not new beyond, the gap already on record in `scoping_doc.md` §8.
3. No adversarial/red-team test suite exists for the Compliance Agent (Part 3) — tracked here as an open item.

None of these three are resolved by this document. They're recorded here as findings, to be picked up as their own dated decisions — consistent with how every other gap this project has surfaced has been handled.
