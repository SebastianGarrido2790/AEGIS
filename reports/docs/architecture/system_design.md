# System Design — Architectural Decision Record — AEGIS

**Actuarial Elasticity & Governance Intelligence System**
Author: Sebastián Garrido Arévalo | Date: August 13, 2026
**Status: Phase 1 Validated; Phase 2 reset for EDA-first redo.** Phase 1 foundations remain validated. Historical Phase 2 attempt-1 implementation (including archived former ADRs 011–019) is preserved in branch `archive/phase2-attempt-1`. ADR-011 on `main` establishes a governed decision topology (single Compliance agent + deterministic specialist nodes + post-governance briefing narrative). Tier 2 and Tier 3 remain future implementation phases.

---

## 1. Architecture Overview

```text
                     ┌──────────────────────────────┐
                     │   FastAPI Serving Layer       │
                     │ (Underwriting/Pricing Client)│
                     └───────────────┬──────────────┘
                                     │
                                     ▼
                     ┌──────────────────────────────┐
                     │   LangGraph Orchestrator     │
                     │  (Tier 2 — Governed Workflow)│
                     │                              │
                     │  [Pricing Strategy] (Det.)   │
                     │         │                    │
                     │    ┌────┴────────────────┐   │
                     │    ▼                     ▼   │
                     │[Compliance Agent]  [Revenue  │◄──── Redis Stack
                     │   (RAG, LLM)        Impact]  │      (RegVL/HNSW index)
                     │    │                (Det.)   │
                     │    └────┬────────────────┘   │
                     │         ▼                    │
                     │  [Governance Node] (Det.)    │
                     │  (Corridor & Compliance Gate)│
                     │         │                    │
                     │         ▼                    │
                     │  [Underwriter Briefing]* (LLM)│
                     │  (Phase 8; template in Ph.6) │
                     │         │                    │
                     │         ▼                    │
                     │  [Narrative Validator]* (Det)│
                     │  (Phase 8; template in Ph.6) │
                     └───────────────┬──────────────┘
(* Phase 8 nodes; Phase 6 uses a deterministic Jinja2 briefing template)
                                     │
                           ┌─────────┴─────────┐
                           ▼                   ▼
                 ┌──────────────────┐  ┌──────────────────────┐
                 │   LLM Gateway      │  │  Tier 1 — Deterministic│
                 │   (LiteLLM)        │  │  ML (Brawn)            │
                 │                    │  │  GLM → Causal Elasticity│
                 │  Fallback / Router │  │  → Contextual Bandit    │
                 │  Cache (Redis      │  └───────────┬────────────┘
                 │  Stack, shared)    │              │
                 │  Guardrails        │              ▼
                 │  Cost/Trace Export │      MLflow + DVC (+ GX gates)
                 └────────────────────┘
                                     │
                                     ▼
                     ┌──────────────────────────────┐
                     │  Tier 3 — Governance & Audit │
                     │  HITL Review Endpoint        │
                     │  Deterministic Rate Fallback │
                     │  Structured Audit Log (DB)   │
                     └──────────────────────────────┘
```

## 2. Components (Planned)

| Component                      | Responsibility                                                                                                              | Key technology                                      |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Serving layer                  | Exposes recommendation flow and glass-box showcase demo interface to underwriting/pricing clients                            | FastAPI, Jinja2, Chart.js, Docker                   |
| Tier 1 — Deterministic ML      | GLM baseline, causal elasticity estimator (semi-synthetic benchmark), contextual bandit exploration corridor                | scikit-learn, statsmodels, EconML/DoWhy, MLflow, DVC |
| Data contracts                 | CI-blocking validation of both data intake paths                                                                            | Great Expectations                                  |
| LLM Gateway                    | Single enforcement boundary for every provider call: abstraction, fallback, routing, caching, guardrails, cost/trace export | LiteLLM (in-process)                                |
| Regulatory RAG index           | Backing store for Compliance Agent retrieval and Gateway semantic prompt cache                                              | Redis Stack (RedisVL/HNSW)                          |
| Tier 2 — Governed orchestration| Compliance reasoning agent, deterministic Pricing & Impact nodes, post-governance Briefing narrative & deterministic validator | LangGraph, Pydantic, LiteLLM                        |
| Tier 3 — Governance & Audit    | HITL escalation gate, deterministic fallback rate table, structured immutable audit log                                     | Custom, backed by SQLite (dev) / DB (Phase 7)       |
| Observability                  | Tracing, cost, drift, and evaluation                                                                                        | OTel, LLM-as-judge regression suite                 |
| CI/CD                          | Reproducibility, coverage, and data-contract gating                                                                         | GitHub Actions                                      |

## 3. Data Flow (Target Architecture)

1. A segment/policy is submitted to the FastAPI layer.
2. Tier 1 returns a causal elasticity estimate $\epsilon(X)$ and a bandit-bounded proposed adjustment $\Delta$.
3. The **Pricing Strategy deterministic node** validates inputs and constructs the typed rate adjustment proposal.
4. Parallel execution fan-out:
   - The **Regulatory Compliance Agent (LLM)** retrieves grounding evidence from Redis Stack (via the Gateway) and evaluates statutory compliance, returning a pass/flag verdict with citations.
   - The **Revenue/Loss-Ratio Impact deterministic node** calculates projected financial effects accounting for the causal demand elasticity volume response ($Q_{new} = Q_0 \times (1 + \epsilon \cdot \Delta)$, $\Delta Rev \approx Rev_0 \cdot [(1+\Delta)(1+\epsilon \Delta) - 1]$, $LossCost$, $LossRatio$) using first-order actuarial approximations without LLM involvement (INV-5).
5. The **Governance deterministic node** evaluates corridor bounds and the compliance verdict, assigning a terminal review disposition: `READY_FOR_REVIEW` (routine underwriter sign-off) or `ESCALATED` (human review required).
   - *Operational Fallback (INV-6):* If compliance grounding falls below threshold or a component degrades, the system safely reverts the rate to the last compliance-approved rate table ($Rate_{approved}$) and emits disposition `ESCALATED` with `fallback_active = True`. In this state, the proposed adjustment and its counterfactual impact remain recorded but are explicitly tagged as unapplied (`projected_unapplied = True`), with in-force financials reflecting the reverted rate ($\Delta = 0$).
6. Synchronous Underwriter Briefing & Deterministic Validation (INV-7 compliant):
   - In Phase 6, the underwriter briefing narrative is generated by a **deterministic Jinja2 template**, guaranteeing sub-millisecond execution and a 100% complete audit record before API response.
   - When the LLM Briefing Generator is introduced (post-Phase 7), it executes synchronously under a bounded timeout ($\le 2\text{s}$) strictly consuming non-free-text structured fields (proposal parameters, numerical deltas, citation IDs, and disposition enum). It never ingests raw corpus text or un-sanitized LLM rationales (prompt isolation, AGENTS.md §2). Official statutory quotes are inserted deterministically via citation ID lookup.
   - The **Deterministic Narrative Validator** checks that numbers and citation IDs match the record, and the narrative contains no disposition-contradicting language. Any mismatch or timeout triggers an immediate fallback to the deterministic Jinja2 template.
7. An immutable structured audit record (proposal, compliance evidence, financial projection, disposition, briefing narrative, validation outcome) is persisted to storage and validated before the API response is returned (INV-7).

## 4. Architectural Decision Records

### ADR-001: LLM Gateway — LiteLLM (in-process) over a standalone proxy

**Context:** All production agent/RAG LLM calls route through a Gateway enforcing six capabilities (unified access, resiliency, routing, caching, guardrails, cost observability), and permits either a standalone proxy (e.g., MLflow AI Gateway) or an in-process library (e.g., LiteLLM) as the implementation topology.

**Decision:** Use LiteLLM as an in-process library.

**Rationale:** No additional service to deploy or operate, which matters directly for a solo-practitioner scope (`../groundedness/scoping_doc.md` §11 risk register). LiteLLM is already a stack dependency for the `@observe` tracing integration, so this extends an existing dependency rather than introducing a new one.

**Consequences:** The Gateway's resiliency and guardrail logic runs in-process with the orchestrator rather than as an independently scalable service; acceptable for the project's non-production scope (PRD §13 — no live deployment in v1).

### ADR-002: Vector store — Redis Stack (RedisVL) over ChromaDB

> **Core Reality:** Enterprise RAG users ask variants of the same core questions repeatedly. Without an intermediate cache layer, every variant executes a full pipeline: cross-network document retrieval, embedding generation, reranking, prompt construction, and an LLM API call. Redis eliminates the majority of these redundant operations by serving as both the primary vector database and an in-memory semantic cache within the same memory space.

**Context:** The Compliance Agent requires a RAG index over the regulatory corpus. ChromaDB was the initial default (embedded, no server to run); however, ADR-001's Gateway mandates a two-tier Redis-backed prompt cache.

**Decision:** Use Redis Stack (RedisVL/HNSW) as the regulatory RAG index, sharing the same instance that backs the Gateway's prompt cache, rather than introducing ChromaDB as a second, separate vector store.

**Rationale:** One platform serving both roles directly reduces the infrastructure footprint and the scope risk named in `../groundedness/scoping_doc.md` §11, and mirrors a pattern already proven in prior work (Medical AI Assistant's Redis Stack HNSW implementation) rather than introducing an unproven combination.

**Consequences:** The regulatory corpus, being small and infrequently updated, is somewhat over-served by Redis Stack's capacity relative to its actual scale — an accepted tradeoff in exchange for infrastructure consolidation.

### ADR-003: Data contracts — Great Expectations paired with DVC, CI-blocking

**Context:** DVC versions data and pipeline artifacts but does not validate them; without an independent validation gate, invalid data can be faithfully version-controlled ("garbage in, garbage out").

**Decision:** Two Great Expectations suites — one for elasticity training data (schema, value ranges, post-treatment leakage check), one for regulatory corpus ingestion (non-empty chunks, required metadata, duplicate detection) — both enforced as CI-blocking gates ahead of DVC versioning, not as advisory warnings.

**Rationale:** The regulatory corpus suite in particular is a direct upstream mitigation for the system's highest-severity failure mode (RAG-grounded compliance hallucination, `../groundedness/scoping_doc.md` §11) — malformed or duplicated chunks are rejected before indexing, not caught after the fact.

**Consequences:** Adds a validation step ahead of every data pipeline run; accepted as necessary given the severity of the failure mode it mitigates.

### ADR-004: Package layout & build toolchain — Namespaced `src/aegis/` with Hatchling & Python 3.12

**Status:** Validated (Phase 1)

**Context:** Package layout and packaging configuration determine module import paths across the codebase (`from aegis.gateway import ...`) and how dependencies and builds are managed. Generic top-level package names risk collisions with third-party libraries.

**Decision:** Adopt a namespaced package layout rooted at `src/aegis/`, specify `hatchling` as the build backend in `pyproject.toml`, and set the Python requirement to `>=3.12,<3.13` with `3.12` pinned in `.python-version` (P1-D1).

**Rationale:** Namespacing under `src/aegis/` is the standard `uv`-recommended packaging pattern and prevents shadowing third-party modules. `hatchling` provides mature tooling, wide community documentation, and seamless integration with `uv`. Minor-version flooring (`>=3.12,<3.13`) combined with `uv.lock` provides both flexibility across patch releases and exact reproducibility.

**Consequences:** All internal code imports are prefixed with `aegis.` (e.g. `aegis.gateway`, `aegis.agents`).

### ADR-005: Configuration management — Domain-nested `params.yaml` with Pydantic validation

**Status:** Validated (Phase 1)

**Context:** The system requires a single source of truth for all tunable parameters across multi-tier ML, agentic orchestration, and governance components, while maintaining a strict zero-secrets policy and fail-loudly error handling.

**Decision:** Structure `params.yaml` into domain-nested sections mirroring the system architecture tiers (`gateway:`, `tier1_ml:`, `tier2_agents:`, `governance:`, `data_contracts:`, `dvc:`), validate all settings at load time against a strict Pydantic model (`aegis.config`), and enforce zero committed secrets (P1-D2).

**Rationale:** Domain nesting prevents key collisions between tiers and keeps configuration scannable. Pydantic validation guarantees that invalid or missing parameters fail fast at application startup rather than deep in runtime execution, bringing configuration under CI type checking and contract validation.

**Consequences:** Any addition or modification to `params.yaml` requires corresponding updates to the Pydantic configuration schemas in `src/aegis/config/`.

### ADR-006: Data contract architecture — File-based GX Core JSON suites with hand-crafted fixtures

**Status:** Validated (Phase 1)

**Context:** Per INV-3 and ADR-003, data contracts gate both elasticity training data and regulatory corpus ingestion. The contract format, execution mode, and CI verification strategy must be defined without introducing unnecessary external cloud dependencies.

**Decision:** Use open-source GX Core file-based expectation suites serialized as native JSON files in `data_contracts/`, verified during CI using minimal, hand-crafted valid and malformed fixtures in `data_contracts/fixtures/` (P1-D3).

**Rationale:** Avoids external cloud service dependencies (GX Cloud), aligning with the local-first zero-cost development posture (INV-10; PRD §13 — no live deployment in v1). Native JSON files provide declarative, cleanly version-controlled definitions. Hand-crafted fixtures verify specific failure modes (e.g., negative exposure, post-treatment leakage, missing chunk metadata) in milliseconds without pulling real dataset dependencies into Phase 1.

**Consequences:** Expectation suites are versioned directly in git; custom expectation functions (e.g., elasticity leakage checks) are integrated through GX Core compatible definitions.

### ADR-007: DVC pipeline architecture — Local filesystem remote with fine-grained DAG stages

**Status:** Validated (Phase 1)

**Context:** DVC pipeline orchestration must enforce Great Expectations gates before versioning datasets or artifacts, following the local-first zero-cost development model.

**Decision:** Configure DVC with a local filesystem cache/remote and define fine-grained pipeline stages (`ingest`, `validate_gx`, `version`) in `dvc.yaml`, making Great Expectations validation an explicit node in the DVC DAG (P1-D4).

**Rationale:** Local filesystem remotes eliminate cloud configuration complexity during local development while adhering to INV-10. Fine-grained stages allow DVC to cache intermediate steps and only rerun validation when raw inputs change, making data contract enforcement explicit and inspectable in the DAG.

**Consequences:** Pipeline reproduction (`dvc repro`) explicitly verifies GX validation stages before producing versioned outputs.

### ADR-008: CI/CD scaffold & invariant enforcement — Unified GitHub Actions workflow with day-one gates

**Status:** Validated (Phase 1)

**Context:** Continuous integration must enforce project standards, code quality, test coverage, and architectural invariants (notably INV-8 module line ceiling and INV-3 data contracts) from the very first commit.

**Decision:** Implement a single unified GitHub Actions workflow (`.github/workflows/ci.yml`) executing sequential gates: linting (`ruff`), type checking (`pyright`), module size enforcement (`scripts/check_module_size.py` for INV-8), GX fixture validation gates (INV-3), and automated test execution (`pytest` minimal scaffold) (P1-D5).

**Rationale:** A single workflow provides a clear pass/fail status and minimizes CI maintenance overhead. Enforcing the 1,000-line limit (INV-8) and running a minimal pytest scaffold from Phase 1 guarantees that no unverified or non-compliant code enters the repository at any stage.

**Consequences:** All PRs and commits must pass all quality, invariant, and contract gates before merging.

### ADR-009: Showcase interface — Incremental glass-box demo, curated scenarios, FastAPI + Jinja2

**Status:** Approved (pre-Phase 2)

**Context:** The PRD names a secondary audience (hiring managers, technical evaluators, academic reviewers) distinct from the in-universe personas. No artifact currently lets that audience observe the governed multi-agent decision loop running without reading code or ADRs. This is a separate question from PRD §12's exclusion of a full production UI — that exclusion concerns a real underwriting console for daily operational use, not a demo interface for portfolio evaluation. Full deliberation recorded in `../decisions/showcase_ui_assessment.md`.

**Decision:** Build a minimal, explicitly-labeled "glass-box" demo interface — not a production dashboard — using FastAPI with server-rendered Jinja2 templates (HTML-over-the-wire), consistent with the SSR dashboard pattern already established elsewhere in this portfolio. It is built incrementally, with a slice added at the close of each phase that produces something worth showing (Phase 2: elasticity/bandit output view; Phase 6: multi-agent trace panel; Phase 7: audit record and HITL view) rather than as a dedicated standalone phase. It exposes a curated set of 3–5 preset scenarios via dropdown selection — not free-form input — deliberately including at least one scenario engineered to trigger escalation and one to trigger the deterministic fallback, not only successful-approval paths.

**Rationale:** A code listing cannot communicate that the system is _governed_, not merely agentic — an evaluator needs to see an escalation trigger or a fallback engage to register that distinction. FastAPI + Jinja2 was chosen over Streamlit/Gradio to remain consistent with established portfolio precedent and to avoid the "quick model demo" association those frameworks carry, which risks undercutting the production-engineering positioning this project is built around. Incremental delivery avoids concentrating UI risk at the end of the roadmap, when schedule pressure is highest and scope quietly erodes. Curated scenarios over free-form input were chosen because INV-10 means there is no live data source for genuinely free-form input to query against — and because showing only successful approvals would hide the actual differentiator (the governance layer), which is the entire point of exposing this to the secondary audience in the first place.

**Consequences:** `../groundedness/scoping_doc.md`, `../groundedness/prd.md` §12, and `../implementation_roadmap.md` are updated to reflect this component and its phase-by-phase delivery slices. The interface must at every stage remain visually and textually distinguishable from a production system — mislabeling risks contradicting PRD §12's non-goal rather than complementing it.

### ADR-010: Compliance Agent evaluation refined to a four-metric diagnostic matrix (INV-6 amendment)

**Status:** Approved (pre-Phase 2)

**Context:** INV-6 originally gated the Compliance Agent's grounding check on an ad hoc "groundedness/evidence-coverage score" without specifying which underlying metrics constituted that score. Retrieval and generation are independently-failing surfaces in any RAG pipeline: a generation-layer check like Faithfulness or Answer Relevancy can look healthy even while retrieval itself has silently collapsed, because a fluent, well-supported-sounding answer can still be built on the wrong — or missing — retrieved context. Gating on either of those two metrics alone risks exactly the silent failure mode INV-6 exists to prevent.

**Decision:** INV-6 is amended to require a joint, four-metric read — Faithfulness, Answer Relevancy, Context Recall, and Context Precision — reported together, with no single metric permitted to gate a CI/CD or production decision alone. The two Phase-1-era placeholder fields in `params.yaml` (`groundedness_threshold`, `evidence_coverage_threshold`) are retained as informal shorthand for now but must be replaced with the full four-metric threshold set before the Compliance Agent ships in Phase 5/6.

**Rationale:** A retrieval-recall collapse is invisible to a Faithfulness-only or Answer-Relevancy-only check, since the model can still generate a fluent, internally-consistent answer from whatever — possibly wrong — context it did retrieve. Reading all four metrics jointly is the only way to distinguish "the answer is wrong because retrieval failed" from "the answer is wrong because generation ignored good context" — two failure modes with different fixes, indistinguishable from a single aggregate score.

**Consequences:** INV-6 and its new §9 (Evaluation, Calibration & Monitoring) are updated to reflect this. Phase 5/6 must expand `Tier2AgentsConfig` beyond its current two placeholder threshold fields before the Compliance Agent can ship — tracked as known, not silent, scope.

### ADR-011: Governed Decision Topology — Single Compliance Agent with Deterministic Specialist Nodes and Post-Governance Briefing Narrative

**Status:** Approved (October 2026; amends PRD §8 and `system_design.md` §1/§3; aligns the ADR-009 Phase 6 trace panel with Option C)

**Context:**
The initial architecture assumed a three-agent LangGraph topology (`Pricing Strategy Agent → [Regulatory Compliance Agent ‖ Revenue/Loss-Ratio Impact Agent] → Governance`). A critical review of this design against INV-5, cost/latency realities, and `capability_profile_gates.md` surfaced several structural defects:

1. **Category Error on Node Selection:** The original "merging test" evaluated whether Pricing, Compliance, and Impact could be collapsed into one node, concluding they should not because their responsibilities differed. It never asked the prior, load-bearing question: *Does each node need an LLM at all?*
2. **Deterministic Violations & Hallucination Surface:** 
   - Pricing Strategy merely extracts Tier 1 numerical outputs and packages them into a typed Pydantic contract. An LLM call here is pure waste (latency, token spend, parsing-failure risk) for zero reasoning value.
   - Revenue/Loss-Ratio Impact computes portfolio financial effects ($LossCost$, $\Delta Rev$, $LossRatio$). INV-5 strictly prohibits LLM financial arithmetic. Having an LLM generate or wrap numbers it did not compute creates a dangerous failure mode where the model misstates or hallucinates deterministic figures.
   - Governance evaluates corridor bounds and the compliance pass/flag verdict to assign a review disposition. This is a deterministic conditional rule, not an LLM judgment.
3. **Incomplete Financial Formula & Elasticity Definition:** The naive static formulation ($Exposure \times Premium \times \Delta Rate$) ignores the retention/lapse response, which is the entire reason the causal elasticity model exists. A realistic projection must model the causal volume response via elasticity, drop the "exact" claim in favor of an analytic approximation, precisely define $\epsilon(X)$, and explicitly declare the homogeneous retention risk assumption (no adverse selection in lapse behavior).
4. **Topology and Data Flow Contradiction:** `system_design.md` §1 and §3 previously rendered a sequential pipeline (Pricing → Compliance → Impact), directly contradicting the parallel fan-out specified in `AGENTS.md` and `capability_profile_gates.md`.
5. **Disposition Taxonomy Ambiguity:** Previous documentation left ambiguous whether `FALLBACK` was a third disposition or an operational fail-safe on the escalated path.
6. **Prompt Isolation vs. Quote Verification Conflict:** Checking quotes against raw corpus text implies the briefing LLM ingests untrusted text, creating prompt injection risks.
7. **Async Briefing vs. INV-7 Conflict:** An asynchronous briefing cannot be part of an immutable audit record that must be 100% complete before API response per INV-7.

**Decision:**
Adopt **Single Decision Agent + Deterministic Specialist Nodes + Bounded Post-Governance Briefing (Option C)**:

1. **Pricing Strategy Node (Deterministic):** Pure Python/Pydantic graph node that constructs the typed rate adjustment proposal from Tier 1 causal and bandit outputs.
2. **Parallel Decision Fan-Out:**
   - **Compliance Agent (Probabilistic LLM Agent):** The *only* agent in the decision path. It retrieves regulatory evidence from Redis Stack and reasons over statutory text to return a pass/flag verdict with citations. (Whether retrieval is model-driven or deterministic query-built is an open Phase 5 evaluation question).
   - **Revenue/Loss-Ratio Impact Node (Deterministic):** Pure Python/Pydantic tool executing first-order financial impact approximations accounting for the causal demand elasticity response ($Q_{new} = Q_0 \times \max(0, 1 + \epsilon \cdot \Delta)$, $\Delta Rev \approx Rev_0 \cdot [(1 + \Delta)(1 + \epsilon \cdot \Delta) - 1]$, $LossRatio_{new} = \frac{Losses_{new}}{Rev_{new}}$). Here $\epsilon(X)$ is the dimensionless price elasticity of retention ($\% \Delta Q / \% \Delta P$), converted from CausalForestDML's estimated treatment effect $\tau(X)$ on retention probability. This projection explicitly assumes homogeneous lapse risk within segment (no adverse selection in lapse behavior). Under this assumption the volume terms cancel in the loss ratio ($LossRatio_{new} = LossRatio_0 / (1 + \Delta)$), so $\epsilon$ moves revenue and the absolute underwriting result ($Rev - Losses$, the elasticity-sensitive headline metric) but not the loss ratio; relaxing homogeneity requires an explicit adverse-selection parameter. The volume factor is floored at zero (no negative policy counts). Full closed-form formulas, the $\tau \to \epsilon$ conversion, and bound/floor rules must be formally specified in the Phase 6 feature specification before node implementation. Zero LLM involvement.
3. **Governance Node (Deterministic) & Disposition Taxonomy:**
   - Evaluates corridor bounds and the compliance verdict, assigning a terminal review disposition: `READY_FOR_REVIEW` (routine underwriter sign-off) or `ESCALATED` (human review required).
   - **Operational Fallback (INV-6):** An emergency fail-safe route rather than a third review destination. When compliance retrieval groundedness drops below threshold or a technical failure occurs, the system safely reverts the rate proposal to the last compliance-approved rate table ($Rate_{approved}$) and emits disposition `ESCALATED` with `fallback_active = True`. In this state, the proposed adjustment and its counterfactual projected impact remain recorded but are explicitly tagged as unapplied (`projected_unapplied = True`), with in-force financials reflecting the reverted rate ($\Delta = 0$). Publication to a live rate table is never permitted (INV-4).
4. **Underwriter Briefing Generator & Phase Sequencing (INV-7 Compliant):** 
   - **Phase 6 Scope:** To keep Phase 6 bounded and protect solo build capacity, the underwriter briefing narrative is generated by a **deterministic Jinja2 template** satisfying Story 2.1 with sub-millisecond execution.
   - **Post-Phase 7 Enhancement:** When the LLM Briefing Generator is introduced (Phase 8), it executes **synchronously under a bounded timeout ($\le 2\text{s}$)**, ensuring that the audit record is 100% complete before API response per INV-7 (no unverified async state). Crucially, this node has **zero decision authority**: it cannot modify the proposal, verdict, financials, or disposition.
   - **Prompt Isolation (Untrusted Data Defense, AGENTS.md §2):** The briefing model ingests *strictly non-free-text structured fields*: proposal parameters, citation IDs, numerical deltas, and the disposition enum. It is strictly prohibited from ingesting raw external corpus text or un-sanitized LLM rationales (preventing direct and second-order prompt injection). Statutory quotes are inserted deterministically via citation ID lookup.
5. **Multi-Layer Deterministic Narrative Validator & Fallback:**
   - *Mechanical verification:* Numbers, currency values, percentages, and citation IDs are verified against the structured decision record.
   - *Disposition consistency rule:* The validator enforces that the narrative contains zero disposition-contradicting language (no approving phrasing on an `ESCALATED` record with or without `fallback_active = True`; no violation phrasing on an `READY_FOR_REVIEW` record).
   - *Template fallback:* On any validation mismatch, parsing failure, or timeout, the system falls back immediately to the deterministic Jinja2 template.
   - *Audit trail:* The audit record stores both the final narrative, validation outcome, and fallback status before the API response is emitted (INV-7).
6. **Cost & Latency Realities:**
   - Total LLM calls across the complete pipeline: **1 call** in Phase 6 (Compliance only); **2 calls** when the post-Phase 7 LLM briefing is enabled (Compliance + Briefing), down from 3 in the initial framing (a net reduction of **one call** overall).
   - Critical decision path LLM calls: **1 call** (Compliance only), eliminating 2 LLM calls from the critical path and removing the risk of LLM-hallucinated financial figures.
7. **INV-5 Enforcement Relocation & Phase 2 Exemption:**
   - The schema requirement enforcing non-null Compliance and Impact fields is moved from Pricing Strategy's output schema (which runs first and cannot know them) to the **Governance output schema / recommendation record boundary** leaving Tier 2.
   - **Phase 2 Showcase Exemption:** The standalone Phase 2 Tier 1 developer showcase / model diagnostic UI (ADR-009) is explicitly exempted from requiring Compliance and Impact context, provided the UI is labeled as a raw model validation demo and not a pricing recommendation. The diagnostic view may show $\epsilon$ estimates and recovery metrics, but never a proposed rate adjustment ($\Delta$). INV-5 strictly binds any rate recommendation or decision output.

**Consequences:**
- **ADR-009 Alignment:** ADR-009 remains valid for showcase interface architecture and incremental delivery, but its Phase 6 trace-panel deliverable must now reflect this execution sequence: deterministic Pricing Strategy → parallel [Compliance Agent ‖ Revenue Impact] → deterministic Governance → templated briefing (Phase 6; LLM Briefing + Validator display lands with Phase 8).
- **Phase 8 Observability:** The Briefing Generator's validator-pass rate and template-fallback rate (`briefing.validation.pass_rate`, `briefing.fallback.count`) are added to the Phase 8 evaluation suite and OTel dashboard.
- The system is positioned as a **governed-by-construction agentic workflow with a single compliance reasoning agent**, not "three agents."
- `capability_profile_gates.md` is amended to reflect the node-level LLM test, provisional Level 3 classification, and Phase 5 retrieval-mode evaluation.
- `PRD.md`, `scoping_doc.md`, `implementation_roadmap.md`, `system_design.md`, and `README.md` are reconciled in lockstep.

## 5. Open Implementation Notes

- The specific persistence layer for the Tier 3 audit log is deferred to Phase 7 and not yet decided.
- The minimal HITL review interface (Phase 7) is scoped as an endpoint, not a full dashboard, consistent with PRD §12 (out of scope: full production UI).
- This document will be updated at the close of each phase in the Implementation Roadmap to reflect the actual implemented state.

## 6. Engineering Practices & Quality Standards (Remediation Lessons)

The Phase 2 remediation established explicit engineering practices that govern all subsequent phases:

- **Falsifiable Testing Discipline:** Unit tests must be constructed so that defective implementations (e.g., zero noise variance, out-of-bounds estimates, or silent fallback markers) genuinely fail. Tautological assertions (e.g., asserting `val == pytest.approx(val)` or `mae >= 0.0`) are strictly prohibited.
- **Module Size & Decomposition Ceiling (INV-8):** Every source file under `src/` must remain strictly under 1,000 lines. Logic must be decomposed by responsibility before extending.
- **Zero Silent Fallbacks:** Bare `except` handlers that catch and convert pipeline errors into successful return payloads are prohibited. Refuters and estimators must execute live or fail loudly with typed exceptions.
- **Scientific Reporting Conventions:** P-values resulting from finite simulation or permutation tests that yield zero empirical rejections are reported as $p < 0.001$ rather than absolute zero. Diagnostic variance fractions ($1 - R^2$) and raw noise variances are logged together for full data provenance.

## 7. Update Protocol

At the close of each phase in `../implementation_roadmap.md`:

1. Update "Current Implementation Status" to reflect what was actually built.
2. Mark each relevant ADR's status as **Validated** (implementation matched the decision) or **Amended** (implementation diverged, the amendment must be logged as a new dated entry under the original ADR, not a silent edit).
3. Add new ADRs for any architecturally significant decision made during implementation that wasn't anticipated in Phase 0.

