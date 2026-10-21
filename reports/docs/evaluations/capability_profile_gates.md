# Capability Profile Gates — AEGIS Tier 2 (Agentic Orchestration Layer)

**What this document is:** the Pre-Build Validation Gates and Complexity Diagnostic for AEGIS's Tier 2 agentic layer — the record of whether an agent is justified here at all, and at what complexity level.
**Why it exists:** these questions are meant to be answered before planning documents specify how to build the agent — to decide whether an agent is justified at all and at what complexity level.
**How to use it:** as the record of that decision. Where this exercise finds AEGIS already correctly scoped, it says so and moves on. Where it finds a gap, the gap is stated plainly and left open — not quietly fixed by this document.

**Honest sequencing note, stated rather than glossed over:** this gate exercise is being completed *after* AEGIS's Tier 2 architecture already exists in `../architecture/system_design.md` and `../groundedness/prd.md` — not before, as the gate sequencing intends. That's out of order. The value of doing it now is that it can check the *existing* design against the gate's questions honestly, including finding real gaps the original build didn't surface — which is what happened here (see Part 1, Gate 5, and Part 3, Alignment). It is not, and should not be read as, a claim that this was done in the right order the first time.

---

## Amendment (October 2026) — Node-Level LLM Necessity & Topology Realignment (ADR-011)

**Context of this amendment:**
The original Part 2 and Part 3 evaluations applied the "Merging Test" across three candidate agents (Pricing Strategy, Compliance, Revenue Impact) and concluded that Level 3 (Bounded Agent) was justified because merging them would lose separation of authority, specialization, and risk isolation.

**The Retrospective Finding:**
The Merging Test answered the wrong question first. It asked *whether three agents should be merged*, but never asked *whether Pricing Strategy and Revenue Impact required an LLM at all*.

Evaluating each node against the Brain/Brawn separation and INV-5 surfaces the following corrections:
1. **Pricing Strategy is Deterministic:** It consumes Tier 1 outputs and validates them into a Pydantic schema. Wrapping this in an LLM call adds token spend, latency, and parsing fragility for zero reasoning benefit.
2. **Revenue/Loss-Ratio Impact is Deterministic:** INV-5 strictly prohibits LLM financial arithmetic. Financial projections must account for the causal demand elasticity response ($Q_{new} = Q_0 \times \max(0, 1 + \epsilon \cdot \Delta)$, $\Delta Rev \approx Rev_0 \cdot [(1+\Delta)(1+\epsilon \Delta) - 1]$, $LossRatio_{new}$) computed by first-order deterministic actuarial approximations assuming homogeneous retention risk. Having an LLM generate or narrate these numbers during the decision path creates a dangerous hallucination surface.
3. **Governance is Deterministic & Dispositions are Clarified:** Review routing evaluates corridor bounds and the compliance pass/flag verdict to assign `READY_FOR_REVIEW` or `ESCALATED`. Operational Fallback (INV-6) is an emergency fail-safe route rather than a third review destination: upon grounding failure or technical breakdown, the system reverts the rate proposal to the baseline approved table ($Rate_{approved}$) and emits disposition `ESCALATED` with `fallback_active = True`.
4. **Compliance is the Sole Decision Agent:** Retrieving and evaluating unstructured regulatory statutes against pricing proposals is the only node in the decision path that genuinely requires natural-language reasoning.
5. **Briefing Narrative & Phase Sequencing (INV-7 Compliant):** To keep Phase 6 bounded and protect solo build capacity, Phase 6 generates the underwriter briefing narrative via a deterministic Jinja2 template. When the LLM Briefing Generator is introduced post-Phase 7 (Phase 8), it executes synchronously under a bounded timeout ($\le 2\text{s}$), ensuring the audit record is 100% complete before API response per INV-7 (no unverified async state).
6. **Robust Multi-Layer Narrative Validator & Prompt Isolation:**
   - *Prompt isolation (AGENTS.md §2):* The briefing LLM receives strictly non-free-text structured fields (proposal parameters, numerical deltas, citation IDs, disposition enum). It never ingests raw corpus text or un-sanitized LLM rationales (preventing direct and second-order prompt injection). Statutory quotes are inserted deterministically via citation ID lookup.
   - *Mechanical checks:* Numbers, currency values, percentages, and citation IDs are mechanically verified against the structured record.
   - *Semantic consistency:* The validator enforces that the narrative contains zero disposition-contradicting language (no approving phrasing on an `ESCALATED` record with or without `fallback_active = True`; no violation phrasing on an `READY_FOR_REVIEW` record).
   - *Fallback & telemetry:* On any validation mismatch, parsing failure, or timeout, the system reverts immediately to a deterministic template. The briefing's validator-pass rate and template-fallback rate are tracked in Phase 8 metrics (`briefing.validation.pass_rate`, `briefing.fallback.count`).
7. **Accurate Cost & Latency Profile:**
   - Total pipeline LLM calls: **1 call** in Phase 6 (Compliance only); **2 calls** when the post-Phase 7 LLM briefing is enabled (Compliance + Briefing), down from 3 in the initial framing (net reduction of **one call** overall).
   - Synchronous critical decision path LLM calls: strictly **1 call** (Compliance only), eliminating 2 LLM calls from the decision loop and removing the risk of LLM-hallucinated financial figures.

**Compliance Complexity Level (Level 3 vs. Level 2 Open Evaluation):**
Part 2 classified Compliance as Level 3 (Bounded Agent) under the premise that the LLM performs model-driven tool selection over regulatory chunks.
- *The Caveat:* If a deterministic query generator (built programmatically from proposal segment, rating factors, and rate delta) matches or outperforms model-driven retrieval on the Phase 5 evaluation harness, the Compliance node simplifies to a **Level 2 Deterministic Workflow with an LLM Evaluation Step** (deterministic retrieval → LLM judgment).
- *Phase 5 Evaluation Gate:* Whether Compliance remains a Level 3 Bounded Agent or is refactored to a Level 2 Workflow is an empirical evaluation question for Phase 5, not a settled architectural premise.

---

## Part 1 — Pre-Build Validation Gates

### 1. Measurable Success

The Primary Scenario (`../groundedness/prd.md` §10) runs end-to-end on both branches — auto-approved-for-review and escalated — each emitting a complete in-memory record flagged non-persisted at Phase 6 (full INV-7 persistence lands in Phase 7), with 100% completeness the standing target. Supporting metrics: HITL escalation rate reflecting genuine ambiguity rather than over- or under-triggering; the Compliance Agent's four-metric diagnostic matrix (ADR-010) meeting its threshold once Phase 5/6 ships.

### 2. Blast Radius / Worst-Case Failure

Worst case: the system recommends a regulatorily non-compliant or financially unsound premium adjustment, and that recommendation reaches a published state without human review. This is structurally bounded two ways already on record: INV-4 (the governance output schema has no "published" state at all — publication is a manual, out-of-system action) and INV-6 (a grounding failure below threshold triggers fallback and escalation, never a guess). The single highest-severity variant of this failure — a hallucinated compliance citation producing a false "pass" — is already named as the system's worst failure mode in the risk register (see `../groundedness/scoping_doc.md` §11) and ADR-010.

### 3. Session Economics & Cost Ceiling

**Not yet set.** This gate exercise doesn't resolve it — it confirms the gap already flagged in `../groundedness/scoping_doc.md` §8's constraints table ("Cost per request: Not yet set"). Recommendation, not a decision: a per-session cost ceiling should be set once the Gateway's OTel cost export exists (Phase 4), covering both the Compliance Agent and (when enabled post-Phase 7) the Briefing Generator, using real token-cost data rather than an estimate made before the Gateway is built. Tracked as an open item, not resolved here.

### 4. Evaluation Harness

Specified, not yet built. The golden dataset schema (`FailureMode` taxonomy, `GoldenCase` dataclass) is already architecturally committed — see Part 5 below and the planned `src/aegis/evaluation/golden_schema.py` — but Phase 5/8 is where it's actually implemented. This gate is satisfied at the specification level; implementation status is tracked honestly as pending, not implied complete.

### 5. Graceful Off-Switch

**Not addressed anywhere in AEGIS's current documentation.** This is the first genuine, previously-undocumented gap this exercise surfaces, not a restatement of something already known. Neither `../architecture/system_design.md` nor any other current project document specifies who can halt Tier 2 agent invocations, how fast, with what in-flight-state preservation, or what re-enabling requires. INV-4 and the HITL gate bound what the system can *decide*; neither says anything about stopping the system from running at all. Recommendation: this needs a dated ADR before Phase 6 ships Tier 2 for real, not a silent fix folded into this document.

---

## Part 2 — The Four-Level Complexity Diagnostic

**Classification: Level 3 — Bounded Agent (Provisional — pending Phase 5 evaluation).**

Walking the criteria directly, against the actual design already committed in `../architecture/system_design.md`:

- **Level 1 (Static Prompt):** doesn't fit — multi-step synthesis and statutory reasoning; no single prompt produces the recommendation.
- **Level 2 (Deterministic Workflow):** the overall topology is a deterministic pipeline with parallel fan-out (Pricing [Det.] → [Compliance ‖ Impact [Det.]] → Governance [Det.] → Briefing [LLM]). Whether the Compliance node itself qualifies as Level 3 (Bounded Agent) depends on whether regulatory chunk retrieval requires model-driven tool selection or can be executed as a deterministic structured query followed by LLM statutory judgment. This is an empirical Phase 5 evaluation question (see Amendment above).
- **Level 3 (Bounded Agent):** current baseline classification for the Compliance Agent under model-driven retrieval. The toolset is small and constrained (retrieval tool), the step budget is small and fixed, and wrong actions are low-stakes *by construction* — INV-4 prevents any recommendation from reaching a live rate table without human review.
- **Level 4 (Full Autonomous Agent):** doesn't fit and was never claimed — no 30+ step horizons, no persistent cross-session memory (each review is explicitly single-shot by design), no large or dynamic toolset.

**What this retroactively validates (as amended by ADR-011):** the architecture avoids multi-agent theater by restricting LLM calls strictly to where free-text reasoning is required (statutory interpretation and post-governance briefing narrative), keeping calculation and routing deterministic.

---

## Part 3 — The Capability Profile Matrix

Filled in against AEGIS's actual Tier 2 design. Blank rows are stated as deliberate non-selections with a reason, not silently skipped.

| Capability | Selected Pattern(s) | AEGIS-Specific Notes |
|---|---|---|
| **1. Perception** | *(none selected)* | Policy/segment input is already typed (Pydantic) at the tool boundary; regulatory-document structuring (jurisdiction, section, effective date) is a deterministic, GX-gated ingestion step (INV-3), not an LLM-driven perception agent. Consistent with the Brain/Brawn separation — this capability doesn't need an agent here. |
| **2. Reasoning** | **CoT Auditor** *(adapted)* | AEGIS doesn't implement a literal chain-of-thought trace audit — it implements the four-metric diagnostic matrix (Faithfulness, Answer Relevancy, Context Recall, Context Precision; ADR-010) as its auditable-reasoning mechanism for the Compliance Agent. Named here as the closest-fit pattern, not a literal off-the-shelf match. |
| **3. Planning** | **Plan-Then-Execute** | The topology is fixed at design time, not discovered at runtime — no ReAct loop, no adaptive replanning. |
| **4. Memory** | *(none selected)* | Deliberate, not an oversight: each review is single-shot (no session continuity needed), and the persistent audit log (INV-7) is a write-once compliance record, not memory fed back into future reasoning. |
| **5. Tool Use** | **Tool Selector**, **Side-Effect Auditor** | The Compliance Agent selects which regulatory chunks to retrieve (Tool Selector — provisionally assumes model-driven retrieval tool use, subject to Phase 5 empirical comparison against deterministic query construction). Every recommendation producing a complete, structured audit record (INV-7) is functionally what a Side-Effect Auditor does — wrapping the decision with pre/post state capture. |
| **6. Coordination** | **Pipeline Orchestrator**, **HITL Liaison** | The LangGraph coordinator is a fixed pipeline with a parallel fan-out/fan-in, not a dynamic Supervisor-Worker delegation. The Governance node's escalation gate is a direct HITL liaison. |
| **7. Learning** | **Feedback Loop** *(planned, not built)* | Matches the Phase 8 production harvest loop that routes sub-threshold traces to a human-reviewed golden-dataset candidate queue. Selected here as the target architecture; not yet implemented, and not claimed to be. |
| **8. Alignment** | **Constitution-Bound**, **Provenance Tracker**, **Explainer Agent**, **Drift-Detector** *(planned)* | Constitution-Bound maps directly to the Non-Negotiable Invariants (see `../architecture/system_design.md` ADR log) acting as a hard, un-negotiable constraint set. Provenance Tracker maps to INV-7 and the Compliance Agent's evidence citations. Explainer Agent maps to INV-5 — the causal justification, compliance evidence, and financial impact bundle attached to every recommendation. Drift-Detector matches the planned Phase 8 feature-distribution monitoring. |

**Three unselected Alignment patterns — two gaps and one deliberate non-selection — surfaced explicitly rather than silently omitted:**

- **Off-Switch-Compatible — not selected (gap — the same gap Part 1 Gate 5 already found).** Two independent parts of this exercise landing on the same missing piece is a stronger signal than either alone — it's a real, load-bearing absence.
- **Red-Team Auditor — not selected (gap).** No adversarial test suite exists for AEGIS's Compliance Agent yet (the kind of 20–50-attack corpus a production safety review calls for). When the post-Phase 7 LLM Briefing Generator is introduced, it represents an additional prompt-injection/hallucination surface requiring adversarial testing alongside Compliance — recorded here as a tracked open item, not fixed here.
- **Privacy-Preserving — not selected (deliberate non-selection, not a gap).** AEGIS's current scope operates on rating-factor and segment data, not named-individual PII — this capability isn't yet a modeled concern. Worth revisiting explicitly if the project ever moves beyond synthetic/benchmark data toward anything resembling real customer records.

---

## Part 4 — Key Structural Interfaces

### 4.1 The Validator Sandwich

Already implemented, not re-specified here: the Top Bun is the planned sanitization module at `src/aegis/utils/sanitization.py` (prompt-injection defense, Phase 4/5 — not yet built); the Bottom Bun is two-layer validation — Pydantic schema first, then a semantic/business-rule guardrail — applied to every agent output before it reaches Governance or a caller.

### 4.2 Reversibility and Rollback

Two distinct mappings, not one:

- **The Compliance Agent's "action" (a verdict):** the pre-action state snapshot is the last compliance-approved rate table itself; the inverse operation is INV-6's deterministic fallback to that same snapshot.
- **The elasticity model's promotion (a Tier 1 artifact, not a Tier 2 agent action):** the MLflow registry's `approved` alias is the state capture; rollback re-points that alias to the previous version. Promotion requires a human review record tied to the calibration-report hash and complete lineage. The Phase 2 decision uses aliases (P2-D11a; MLflow 3.15.2 verified October 21, 2026).

---

## Part 5 — Reference Benchmark & Evaluation Schema

The golden dataset schema, stated here in full so this document stands alone:

- **`FailureMode` taxonomy (closed set):** hallucination, retrieval miss, context ignored, multi-hop failure, safety violation, tool misuse, reasoning incoherence, format failure, latency failure.
- **`GoldenCase` fields:** stable `id`; `query`; `expected_context` (documents that should be retrieved — required for the Context Recall check); `ideal_answer` and/or `answer_criteria`; `must_not_include` negative constraints; the `failure_modes` it targets; and a non-empty `added_reason` recording why the case exists.

The authoritative implementation lives in the planned `src/aegis/evaluation/golden_schema.py` (Phase 5/8); this section is the specification that implementation must satisfy.

---

## Part 6 — System Blueprint Comparison

AEGIS's safety layering nests, outermost to innermost: Off-Switch-Compatible → Constitution-Bound → Deterministic Harness. AEGIS's actual design has the inner two layers solidly in place — Constitution-Bound (the Non-Negotiable Invariants, see `../architecture/system_design.md`) wrapping a Deterministic Harness (the compiled LangGraph). **The outermost layer, Off-Switch-Compatible, is missing entirely** — the same finding as Part 1 Gate 5 and Part 3's Alignment row, surfaced a third time from a third angle. Three independent parts of this exercise converging on the same gap is the clearest single conclusion this document reaches.

---

## Summary

**What this exercise confirms (as amended by ADR-011):** Tier 2 is correctly scoped as a governed workflow with a single probabilistic compliance reasoning agent (provisionally Level 3 Bounded Agent, pending Phase 5 evaluation), deterministic specialist nodes, and a bounded post-governance briefing narrative (Option C) — not an over-engineered multi-agent system calling LLMs where deterministic code belongs, nor a Level 4 autonomous system.

**What this exercise found that wasn't previously documented:**
1. No off-switch/kill-switch mechanism exists for Tier 2 (found three separate ways — Part 1 Gate 5, Part 3, Part 6).
2. No per-session or per-request cost ceiling has been set (Part 1 Gate 3) — consistent with, not new beyond, the gap already on record in `../groundedness/scoping_doc.md` §8.
3. No adversarial/red-team test suite exists for the Compliance Agent (Part 3) — tracked here as an open item.

None of these three are resolved by this document. They're recorded here as findings, to be picked up as their own dated decisions — consistent with how every other gap this project has surfaced has been handled.
