# AEGIS — Actuarial Elasticity & Governance Intelligence System

A governed-by-construction agentic platform built to production engineering and actuarial standards that recommends personal auto insurance premium adjustments by combining causal elasticity estimation, RAG-grounded regulatory compliance validation, and financial impact projection — escalating to a human underwriter whenever a recommendation falls outside pre-approved bounds. Consumed by actuarial/pricing analysts and underwriting/compliance reviewers.

Designed for actuarial and pricing analysts, underwriting teams, and compliance reviewers as a governed, auditable decision loop — never as an autonomous rate publisher.

> **Differentiator:** LLMs are used only where free-text reasoning is required — interpreting regulatory text (Compliance Agent) and drafting the post-governance underwriter briefing. Everything else is deterministic, schema-validated code: causal estimation, financial projections (accounting for volume elasticity), governance routing, and multi-layer narrative validation (ADR-011).

---

## 🚦 Project Status

| Phase                                              | Description                                                                                                     | Status         |
| :------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- | :------------- |
| **Phase 0 — Planning & Design**                    | Scoping, PRD, User Story, Implementation Roadmap, System Design ADRs                              | ✅ Complete    |
| **Phase 1 — Scaffolding & Data Contracts**         | Repository structure, `uv` toolchain, `params.yaml`, Great Expectations suites (elasticity & regulatory corpus) | ✅ Complete    |
| **Phase 2 — Tier 1: Deterministic ML Baseline**    | Original French MTPL ingestion, capped-loss Tweedie GLM, semi-synthetic causal validation, MLflow registry, diagnostic showcase | 🔄 Reset — EDA-first redo pending; decisions approved, qualification tests and implementation pending |
| **Phase 3 — Contextual Bandit Exploration Engine** | Bounded Thompson Sampling exploration corridor, synthetic production stream simulator, regret evaluation        | ⬜ Scheduled   |
| **Phase 4 — LLM Gateway Integration**              | In-process LiteLLM Gateway, Redis Stack cache, pre-flight guardrails (PII/injection), OTel cost/trace emission  | ⬜ Scheduled   |
| **Phase 5 — Regulatory RAG Foundation**            | RedisVL/HNSW vector index, regulatory corpus ingestion, retrieval-quality evaluation harness                    | ⬜ Scheduled   |
| **Phase 6 — Agentic Orchestration Layer**          | Option C LangGraph decision loop: deterministic Pricing/Impact nodes, Compliance Agent, templated briefing (ADR-011) | ⬜ Scheduled   |
| **Phase 7 — Governance & Audit Layer**             | Full immutable audit log persistence, formal HITL review endpoint, deterministic rate fallback                  | ⬜ Scheduled   |
| **Phase 8 — Evaluation & Narrative Hardening**     | Post-governance LLM Briefing & validator, CI-blocking LLM-as-judge regression suite, drift monitoring, OTel tracing | ⬜ Scheduled   |
| **Phase 9 — Containerization & CI/CD Close-Out**   | Multi-stage Docker packaging, Docker Compose orchestration, full GitHub Actions pipeline                        | ⬜ Scheduled   |

---

## 🔑 Non-Negotiable System Invariants

AEGIS is built upon strict architectural guardrails and engineering invariants:

| ID | Invariant | Origin | Enforcement |
|---|---|---|---|
| **INV-1** | **Gateway exclusivity.** No module may import or instantiate a raw LLM provider SDK client. Every LLM call routes through the LiteLLM Gateway. | ADR-001 | CI import-boundary check: build fails if a provider SDK is imported outside `src/aegis/gateway/`. |
| **INV-2** | **Single vector store.** Redis Stack (RedisVL) is the only vector store. No second vector database without a superseding ADR. | ADR-002 | CI dependency audit flags any new vector-store package in `pyproject.toml`. |
| **INV-3** | **Data contracts are blocking.** No dataset or corpus document is DVC-versioned or indexed without first passing its Great Expectations suite. A failed suite halts the pipeline. | ADR-003 | GX validation is a hard upstream dependency of every downstream DVC stage. |
| **INV-4** | **No autonomous publication.** The system recommends a premium adjustment; it never publishes one. Every recommendation terminates at auto-approved-for-review or HITL escalation. | PRD §4; `scoping_doc.md` §§3,7 | Governance output schema has no "published" state; publication is manual, out-of-system. |
| **INV-5** | **Causal output is advisory, not authoritative.** Elasticity/bandit output is never surfaced without accompanying Compliance and Revenue/Loss-Ratio Impact context. | PRD §6; `scoping_doc.md` §11 | Governance output schema and recommendation record require non-null Compliance and Impact fields. (Phase 2 model diagnostic showcase carries an explicit developer-only exemption — $\epsilon$/recovery metrics only, never a proposed adjustment; ADR-011). |
| **INV-6** | **Grounding failure triggers fallback, not guessing.** Grounding is a joint read across Faithfulness, Answer Relevancy, Context Recall, Context Precision — never a single score. Below threshold: fall back to the last compliance-approved rate table and escalate. | ADR-003 rationale; `scoping_doc.md` §11; refined by ADR-010 | Threshold check raises a typed exception in the Compliance node, caught by the governance fallback handler. |
| **INV-7** | **Complete audit record, no exceptions.** Every recommendation produces one complete, structured, independently reconstructable audit record. | PRD FR-7, PRD §11 | Audit schema validation is the last gate before any API response; failure raises. |
| **INV-8** | **Module size ceiling.** No file under `src/` exceeds 1,000 lines. Decompose, don't extend. | Project router (Phase 1) | `uv run python scripts/check_module_size.py`; over-limit fails the build. |
| **INV-9** | **Single-jurisdiction scope is explicit, never implied.** Every compliance verdict states the corpus models one illustrative jurisdiction. | PRD §12, User Story §7 | Fixed disclaimer field in every Compliance output schema. |
| **INV-10** | **No live core-system integration in v1.** Carrier core-system contracts are modeled/mocked, never connected live. | PRD §12 | No production carrier credential slot exists in configuration. |

**Amendment record:** INV-6 was refined from an ad hoc single grounding score to the four-metric joint read above, recorded as ADR-010 — not a silent edit. The current `tier2_agents.grounding_threshold` / `evidence_coverage_threshold` fields in `params.yaml` are Phase 1-era placeholders (ADR-005) and MUST be replaced with the full four-metric threshold set before the Compliance Agent ships (Phase 5/6).
---

## 🏗️ Architecture Overview

```text
┌────────────────────────┐
│  Pricing / Actuary UI  │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ FastAPI Showcase (demo)│
└───────────┬────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Registered MLflow metadata   │
│ `aegis-causal-elasticity`    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Phase 2: Tier 1 ML          │
│  - DVC + GX data contracts   │
│  - Shared deterministic      │
│    feature pipeline          │
│  - Tweedie GLM baseline      │
│  - CausalForestDML validation│
└──────────────────────────────┘

Future phases add the LangGraph coordinator, LLM Gateway, RAG, and governance
tiers downstream of this registered deterministic output.
```

---

## 🛠️ Stack & Technologies

- **Language:** Python 3.12 (Strict typing with `pyright`)
- **Dependency Management:** `uv`
- **Agent Orchestration:** LangGraph (Option C governed workflow: single Compliance reasoning agent, deterministic specialist nodes, post-governance Briefing narrative & validator; ADR-011)
- **Deterministic & Causal ML:** statsmodels Tweedie GLM, EconML/DoWhy (`CausalForestDML`); contextual bandit is Phase 3
- **LLM Gateway & Caching:** LiteLLM (In-Process Gateway), Redis Stack (`RedisVL`/HNSW)
- **Data Versioning & Contracts:** DVC, Great Expectations (GX)
- **Experiment Tracking:** MLflow
- **API & Serving:** FastAPI, Docker, Docker Compose
- **Quality & Evaluation:** Ruff, Pyright, Pytest, Retrieval-Quality Grounding Harness, LLM-as-Judge Suite

---

## 📂 Repository Structure

```text
.
├── src/aegis/
│   ├── api/             # FastAPI showcase and registered-model service
│   ├── gateway/         # LiteLLM Gateway boundary (future phase)
│   ├── agents/          # LangGraph agents (future phase)
│   ├── governance/      # HITL, fallback, and audit components (future phase)
│   ├── pipelines/       # DVC, contracts, features, and training
│   ├── bandit/          # Contextual bandit exploration (Phase 3)
│   ├── tools/           # Deterministic tool services
│   ├── schemas/         # Pydantic I/O contracts
│   ├── utils/           # Typed exception hierarchy and utilities
│   └── config/           # Centralized configuration and params loader
├── data_contracts/      # Great Expectations suites (elasticity & regulatory corpus)
├── tests/
│   ├── unit/            # Unit tests for tools, models, and schemas
│   ├── integration/     # Integration tests
│   └── evals/           # Evaluation suites for later agentic phases
├── reports/
│   └── docs/            # Architecture ADRs, PRD, Roadmap, Scoping, Runbooks
├── scripts/
│   └── check_module_size.py # Enforces the 1,000-line module ceiling (INV-8)
├── params.yaml          # Centralized hyperparameters & governance thresholds
├── docker-compose.yaml  # Container orchestration (FastAPI, Redis Stack, Gateway)
└── pyproject.toml       # Python dependencies managed via uv
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- `uv` package manager (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker & Docker Compose (for Redis Stack and local services)

### Installation

```bash
# Clone the repository
git clone https://github.com/SebastianGarrido2790/AEGIS.git
cd AEGIS

# Sync and install virtual environment dependencies
uv sync
```

### Essential Commands

| Action                                  | Command                                                 |
| :-------------------------------------- | :------------------------------------------------------ |
| **Start Phase 2 Showcase**              | `uv run uvicorn aegis.api.app:app --reload`             |
| **Run Test Suite**                      | `uv run pytest -q`                                      |
| **Lint & Type Check**                   | `uv run ruff check . && uv run pyright`                 |
| **Enforce Module Line Ceiling (INV-8)** | `uv run python scripts/check_module_size.py`            |
| **Run Data Contract Suite**             | `uv run great_expectations checkpoint run <suite_name>` |
| **Reproduce DVC Pipeline**              | `uv run dvc repro`                                      |
| **Launch MLflow UI**                    | `uv run mlflow ui`                                      |
| **Run Compliance Retrieval Eval**       | `uv run python scripts/eval_compliance_agent.py`        |
| **Launch Full Stack via Docker**        | `docker compose up --build`                             |

---

## 📜 License

Licensed under the MIT License — see the [LICENSE.txt](LICENSE.txt) file for details.
