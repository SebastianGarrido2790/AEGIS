# System Design — Architectural Decision Record — AEGIS _(working title)_

**Actuarial Elasticity & Governance Intelligence System**
Author: Sebastián Garrido Arévalo | Date: August 13, 2026
**Status: Phase 1 Implemented.** Repository scaffold, namespaced package layout, Pydantic configuration schema, Great Expectations data contracts (regulatory and elasticity suites), DVC parallel DAGs, INV-8 module size enforcement, and unified CI workflow are fully implemented and validated.

---

## 1. Architecture Overview

```
                    ┌──────────────────────────────┐
                    │   FastAPI Serving Layer       │
                    │ (Underwriting/Pricing Client)│
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │  LangGraph Orchestrator        │
                    │  (Tier 2 — Agentic, Brain)     │
                    │                                │
                    │  Pricing Strategy Agent         │
                    │        │                       │
                    │        ▼                       │
                    │  Regulatory Compliance Agent    │◄──── Redis Stack
                    │        │        (RAG, RedisVL)  │      (RegVL/HNSW index)
                    │        ▼                       │
                    │  Revenue/Loss-Ratio Impact Agent│
                    └───────────────┬────────────────┘
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
                │  Cost/Trace Export │
                └────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │  Tier 3 — Governance            │
                    │  HITL Gate │ Deterministic       │
                    │  Fallback │ Structured Audit Log │
                    └──────────────────────────────┘
```

## 2. Components (Planned)

| Component                      | Responsibility                                                                                                              | Key technology                                      |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Serving layer                  | Exposes the recommendation flow and glass-box showcase interface to underwriting/pricing clients | FastAPI, Jinja2, Chart.js, Docker                   |
| Tier 1 — Deterministic ML      | GLM baseline, causal elasticity model, contextual bandit                                                                    | scikit-learn, statsmodels, EconML/DoWhy, MLflow, DVC |
| Data contracts                 | CI-blocking validation of both data intake paths                                                                            | Great Expectations                                  |
| LLM Gateway                    | Single enforcement boundary for every provider call: abstraction, fallback, routing, caching, guardrails, cost/trace export | LiteLLM (in-process)                                |
| Regulatory RAG index           | Backing store for the Compliance Agent's retrieval and the Gateway's semantic cache                                         | Redis Stack (RedisVL/HNSW)                          |
| Tier 2 — Agentic orchestration | Pricing Strategy, Regulatory Compliance, Revenue/Loss-Ratio Impact agents on shared state                                   | LangGraph, Pydantic                                 |
| Tier 3 — Governance            | HITL escalation, deterministic fallback, structured audit log                                                               | Custom, backed by a persistence layer (TBD Phase 7) |
| Observability                  | Tracing, cost, drift, and evaluation                                                                                        | OTel, LLM-as-judge regression suite                 |
| CI/CD                          | Reproducibility, coverage, and data-contract gating                                                                         | GitHub Actions                                      |

## 3. Data Flow (Planned)

1. A segment/policy is submitted to the FastAPI layer.
2. Tier 1 returns a causal elasticity estimate and a bandit-bounded proposed adjustment.
3. The LangGraph orchestrator invokes the Pricing Strategy Agent, which formats the proposal.
4. The Regulatory Compliance Agent retrieves grounding evidence from the Redis Stack index (via the Gateway) and returns a pass/flag verdict.
5. The Revenue/Loss-Ratio Impact Agent projects the financial effect.
6. The Governance layer evaluates the combined output: within-bounds-and-compliant proposals are marked ready for routine underwriter sign-off; anything else is escalated with full state attached.
7. A structured audit record is persisted regardless of outcome.

## 4. Architectural Decision Records

### ADR-001: LLM Gateway — LiteLLM (in-process) over a standalone proxy

**Context:** All production agent/RAG LLM calls route through a Gateway enforcing six capabilities (unified access, resiliency, routing, caching, guardrails, cost observability), and permits either a standalone proxy (e.g., MLflow AI Gateway) or an in-process library (e.g., LiteLLM) as the implementation topology.

**Decision:** Use LiteLLM as an in-process library.

**Rationale:** No additional service to deploy or operate, which matters directly for a solo-practitioner scope (Charter §9 risk register). LiteLLM is already a stack dependency for the `@observe` tracing integration, so this extends an existing dependency rather than introducing a new one.

**Consequences:** The Gateway's resiliency and guardrail logic runs in-process with the orchestrator rather than as an independently scalable service; acceptable for the project's non-production scope (PRD §13 — no live deployment in v1).

### ADR-002: Vector store — Redis Stack (RedisVL) over ChromaDB

> **Core Reality:** Enterprise RAG users ask variants of the same core questions repeatedly. Without an intermediate cache layer, every variant executes a full pipeline: cross-network document retrieval, embedding generation, reranking, prompt construction, and an LLM API call. Redis eliminates the majority of these redundant operations by serving as both the primary vector database and an in-memory semantic cache within the same memory space.

**Context:** The Compliance Agent requires a RAG index over the regulatory corpus. ChromaDB was the initial default (embedded, no server to run); however, ADR-001's Gateway mandates a two-tier Redis-backed prompt cache.

**Decision:** Use Redis Stack (RedisVL/HNSW) as the regulatory RAG index, sharing the same instance that backs the Gateway's prompt cache, rather than introducing ChromaDB as a second, separate vector store.

**Rationale:** One platform serving both roles directly reduces the infrastructure footprint and the scope risk named in Charter §9, and mirrors a pattern already proven in prior work (Medical AI Assistant's Redis Stack HNSW implementation) rather than introducing an unproven combination.

**Consequences:** The regulatory corpus, being small and infrequently updated, is somewhat over-served by Redis Stack's capacity relative to its actual scale — an accepted tradeoff in exchange for infrastructure consolidation.

### ADR-003: Data contracts — Great Expectations paired with DVC, CI-blocking

**Context:** DVC versions data and pipeline artifacts but does not validate them; without an independent validation gate, invalid data can be faithfully version-controlled ("garbage in, garbage out").

**Decision:** Two Great Expectations suites — one for elasticity training data (schema, value ranges, post-treatment leakage check), one for regulatory corpus ingestion (non-empty chunks, required metadata, duplicate detection) — both enforced as CI-blocking gates ahead of DVC versioning, not as advisory warnings.

**Rationale:** The regulatory corpus suite in particular is a direct upstream mitigation for the system's highest-severity failure mode (RAG-grounded compliance hallucination, Charter §9) — malformed or duplicated chunks are rejected before indexing, not caught after the fact.

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

**Rationale:** Avoids external cloud service dependencies (GX Cloud), aligning with the local-first zero-cost development posture (Charter §6, INV-10). Native JSON files provide declarative, cleanly version-controlled definitions. Hand-crafted fixtures verify specific failure modes (e.g., negative exposure, post-treatment leakage, missing chunk metadata) in milliseconds without pulling real dataset dependencies into Phase 1.

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

**Consequences:** `canvas.md` §4, `prd.md` §12, and `technical_roadmap.md` are updated to reflect this component and its phase-by-phase delivery slices. The interface must at every stage remain visually and textually distinguishable from a production system — mislabeling risks contradicting PRD §12's non-goal rather than complementing it.

### ADR-010: Compliance Agent evaluation refined to a four-metric diagnostic matrix (INV-6 amendment)

**Status:** Approved (pre-Phase 2)

**Context:** INV-6 originally gated the Compliance Agent's grounding check on an ad hoc "groundedness/evidence-coverage score" without specifying which underlying metrics constituted that score. Retrieval and generation are independently-failing surfaces in any RAG pipeline: a generation-layer check like Faithfulness or Answer Relevancy can look healthy even while retrieval itself has silently collapsed, because a fluent, well-supported-sounding answer can still be built on the wrong — or missing — retrieved context. Gating on either of those two metrics alone risks exactly the silent failure mode INV-6 exists to prevent.

**Decision:** INV-6 is amended to require a joint, four-metric read — Faithfulness, Answer Relevancy, Context Recall, and Context Precision — reported together, with no single metric permitted to gate a CI/CD or production decision alone. The two Phase-1-era placeholder fields in `params.yaml` (`groundedness_threshold`, `evidence_coverage_threshold`) are retained as informal shorthand for now but must be replaced with the full four-metric threshold set before the Compliance Agent ships in Phase 5/6.

**Rationale:** A retrieval-recall collapse is invisible to a Faithfulness-only or Answer-Relevancy-only check, since the model can still generate a fluent, internally-consistent answer from whatever — possibly wrong — context it did retrieve. Reading all four metrics jointly is the only way to distinguish "the answer is wrong because retrieval failed" from "the answer is wrong because generation ignored good context" — two failure modes with different fixes, indistinguishable from a single aggregate score.

**Consequences:** INV-6 and its new §9 (Evaluation, Calibration & Monitoring) are updated to reflect this. Phase 5/6 must expand `Tier2AgentsConfig` beyond its current two placeholder threshold fields before the Compliance Agent can ship — tracked as known, not silent, scope.

### ADR-011: Elasticity dataset source & ingestion — OpenML/Kaggle freMTPL2 with local DVC tracking

**Status:** Approved (Phase 2)

**Context:** Phase 1 established data contracts and deferred raw dataset acquisition to Phase 2 (P1-D4c). Phase 2 modeling requires motor third-party liability exposure, claim frequency, and claim amount data.

**Decision:** Acquire freMTPL2 (`freMTPL2freq` and `freMTPL2sev`) via `sklearn.datasets.fetch_openml` (with Kaggle ODbL public dataset mirror as verified source, `https://www.kaggle.com/datasets/karansarpal/fremtpl2-french-motor-tpl-insurance-claims`), persist the joined table into `data/raw/`, and DVC-track the resulting file.

**Rationale:** `fetch_openml` eliminates extra download tools, while local DVC tracking ensures all subsequent DVC reproductions (`dvc repro`) and CI runs remain 100% offline, reproducible, and compliant with INV-10 (no live external network dependency).

**Consequences:** `data/raw/elasticity_fremtpl2.csv` is versioned via DVC and validated against `elasticity_training_suite.json` (INV-3).

### ADR-012: Feature engineering architecture — Decomposed transformers, grouped policy split & shared transformation function

**Status:** Approved (Phase 2)

**Context:** Raw insurance claims data requires transformations across multiple domains (exposure offset, driver risk profile, vehicle specs, geographic/bonus-malus features). Training and inference require strict parity without heavyweight feature-store infrastructure.

**Decision:** Decompose feature transformers by domain under `src/aegis/pipelines/feature/`, implement a grouped train/test split by `policy_id`, and expose a single, deterministic, versioned transformation function imported identically by offline training pipelines and online FastAPI inference.

**Rationale:** Modular transformers adhere to the INV-8 design ceiling and isolate unit-testable components. Grouped train/test splitting prevents policyholder leakage across evaluation folds. A single transformation function guarantees training-serving parity without external feature store overhead.

**Consequences:** Feature logic is centralized and tested in `src/aegis/pipelines/feature/` and shared directly with `src/aegis/api/`.

### ADR-013: Actuarial GLM baseline — Single Tweedie GLM on pure premium with statsmodels confidence intervals

**Status:** Approved (Phase 2)

**Context:** Phase 2 exit criteria mandate that the causal model outperforms an actuarial-standard GLM baseline with defensible confidence intervals.

**Decision:** Fit a single Tweedie compound Poisson-Gamma GLM ($1 < p < 2$) on pure premium (loss cost = claim amount / exposure) with log link and exposure log-offset using `statsmodels.api.GLM`.

**Rationale:** The Tweedie compound Poisson-Gamma distribution models zero-inflated continuous claim costs in a single unified loss-cost formulation matching the causal elasticity target. `statsmodels` provides closed-form asymptotic standard errors and parameter confidence intervals without computationally intensive bootstrapping.

**Consequences:** `statsmodels` is added as a declared dependency in `pyproject.toml`.

### ADR-014: Causal elasticity model — CausalForestDML with synthetic treatment validation & DoWhy refutation suite

**Status:** Approved (Phase 2)

**Context:** Observational auto insurance data lacks randomized price experiments. Estimator recovery must be scientifically verifiable, and confounding robustness must be provable.

**Decision:** Train EconML's `CausalForestDML` on an engineered synthetic rate-change treatment with a known segment-varying elasticity ground truth function. Perform confounding sensitivity analysis using DoWhy's refutation suite (placebo treatment, random common cause, data subset refuters) and log refutation outputs to MLflow.

**Rationale:** Synthetic treatment injection provides an exact ground truth to rigorously validate treatment effect recovery, parameter bias, and confidence interval coverage. DoWhy refutation tests provide standardized, reproducible confounding sensitivity checks.

**Consequences:** Causal models and sensitivity results are logged to MLflow; the evaluation report explicitly documents this as estimator recovery validation.

### ADR-015: MLflow tracking & model registry — Local SQLite backend with structured refutation artifacts

**Status:** Approved (Phase 2)

**Context:** The Roadmap requires MLflow experiment tracking and Model Registry integration. MLflow's default local file store (`./mlruns`) does not support the Model Registry.

**Decision:** Configure MLflow with a local SQLite backend URI (`sqlite:///mlflow.db`) and local artifact directory (`./artifacts/mlflow/`). Register distinct models (`aegis-glm-baseline` and `aegis-causal-elasticity`) and log DoWhy refutation summaries as structured JSON run artifacts.

**Rationale:** SQLite enables full MLflow Model Registry capabilities (model versioning, staging, aliases) without requiring a persistent standalone server process, adhering to the local-first zero-cost development posture.

**Consequences:** `mlflow.db` is local and gitignored; MLflow UI is launched locally via `uv run mlflow ui`.

### ADR-016: Evaluation documentation — Versioned Markdown report with MLflow run mirroring

**Status:** Approved (Phase 2)

**Context:** The Technical Roadmap requires an evaluation report on calibration, treatment-effect confidence intervals, and confounding analysis.

**Decision:** Author a versioned Markdown evaluation report in `reports/docs/evaluations/phase_2_evaluation_report.md` (populating the "Evaluations" documentation pillar) and attach an identical copy as an MLflow run artifact.

**Rationale:** Markdown reports provide durable, diffable documentation in git, while MLflow run artifacts maintain provenance coupling with specific trained model versions.

**Consequences:** `reports/docs/evaluations/` receives its first comprehensive evaluation report.

### ADR-017: Serving layer & showcase interface — Dedicated `src/aegis/api/` package with segment-varying preset scenarios

**Status:** Approved (Phase 2)

**Context:** ADR-009 mandates a glass-box showcase interface slice for Phase 2. Serving infrastructure must be separated from internal agent tools, and the Roadmap's Phase 2 deliverable line must accurately reflect Phase 2 scope.

**Decision:** Establish `src/aegis/api/` for the FastAPI serving application and Jinja2 showcase templates. Correct `technical_roadmap.md` to specify elasticity output (omitting premature bandit references). Configure preset risk scenarios ("Young Urban Commuter", "Experienced Rural Driver", "High-Mileage Commercial Driver") to demonstrate segment-varying elasticity.

**Rationale:** Separating `src/aegis/api/` from `src/aegis/tools/` maintains clean architectural boundaries. Preset scenarios showcase the core value proposition of Double-ML heterogeneous treatment effect estimation.

**Consequences:** `src/aegis/api/` is introduced into the codebase; `technical_roadmap.md` is updated.

### ADR-018: Visualization stack — Matplotlib for static reports & Chart.js CDN for interactive showcase UI

**Status:** Approved (Phase 2)

**Context:** Visualizations are required for both static evaluation reports and the interactive showcase demo interface.

**Decision:** Use `matplotlib` to render static figures for the Markdown evaluation report, and use Chart.js (via CDN) for dynamic, interactive elasticity curves and confidence interval bands in the Jinja2 showcase UI.

**Rationale:** Split tooling provides self-contained static figures for git-based documentation and rich, responsive client-side interactivity for the demo interface without server-side rendering overhead.

**Consequences:** `matplotlib` is added to Python dependencies; Chart.js is integrated into frontend HTML/Jinja2 templates.

## 5. Open Implementation Notes

- The specific persistence layer for the Tier 3 audit log is deferred to Phase 7 and not yet decided.
- The minimal HITL review interface (Phase 7) is scoped as an endpoint, not a full dashboard, consistent with PRD §12 (out of scope: full production UI).
- This document will be updated at the close of each phase in the Technical Roadmap to reflect the actual implemented state.

## 6. Update Protocol

At the close of each phase in `../references/technical_roadmap.md`:

1. Update "Current Implementation Status" to reflect what was actually built.
2. Mark each relevant ADR's status as **Validated** (implementation matched the decision) or **Amended** (implementation diverged, the amendment must be logged as a new dated entry under the original ADR, not a silent edit).
3. Add new ADRs for any architecturally significant decision made during implementation that wasn't anticipated in Phase 0.

