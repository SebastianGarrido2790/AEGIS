# AEGIS — Actuarial Elasticity & Governance Intelligence System

A governed agentic platform that recommends personal auto insurance premium adjustments by combining causal elasticity estimation, RAG-grounded regulatory compliance validation, and financial impact projection — escalating to a human underwriter whenever a recommendation falls outside pre-approved bounds.

Designed for actuarial and pricing analysts, underwriting teams, and compliance reviewers as a governed, auditable decision loop — never as an autonomous rate publisher.

---

## 🚦 Project Status

| Phase                                              | Description                                                                                                     | Status         |
| :------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- | :------------- |
| **Phase 0 — Planning & Design**                    | ML Canvas, Project Charter, PRD, User Story, Technical Roadmap, System Design ADRs                              | ✅ Complete    |
| **Phase 1 — Scaffolding & Data Contracts**         | Repository structure, `uv` toolchain, `params.yaml`, Great Expectations suites (elasticity & regulatory corpus) | ✅ Complete    |
| **Phase 2 — Tier 1: Deterministic ML Baseline**    | freMTPL2 ingestion, deterministic features, Tweedie GLM, synthetic causal validation, MLflow registry, showcase | ✅ Complete    |
| **Phase 3 — Contextual Bandit Exploration Engine** | Bounded Thompson Sampling exploration corridor, synthetic production stream simulator, regret evaluation        | ⬜ Scheduled   |
| **Phase 4 — LLM Gateway Integration**              | In-process LiteLLM Gateway, Redis Stack cache, pre-flight guardrails (PII/injection), OTel cost/trace emission  | ⬜ Scheduled   |
| **Phase 5 — Regulatory RAG Foundation**            | RedisVL/HNSW vector index, regulatory corpus ingestion, retrieval-quality evaluation harness                    | ⬜ Scheduled   |
| **Phase 6 — Agentic Orchestration Layer**          | LangGraph multi-agent workflow: Pricing Strategy → [Compliance ‖ Revenue Impact] → Governance synthesis         | ⬜ Scheduled   |
| **Phase 7 — Governance Layer**                     | Human-in-the-loop (HITL) escalation gate, deterministic fallback to approved rate tables, structured audit log  | ⬜ Scheduled   |
| **Phase 8 — Evaluation & Observability Hardening** | CI-blocking LLM-as-judge regression suite, drift monitoring on segment features, OpenTelemetry tracing          | ⬜ Scheduled   |
| **Phase 9 — Containerization & CI/CD Close-Out**   | Multi-stage Docker packaging, Docker Compose orchestration, full GitHub Actions pipeline                        | ⬜ Scheduled   |

---

## 🔑 Non-Negotiable System Invariants

AEGIS is built upon strict architectural guardrails and engineering invariants:

- **INV-1: Gateway Exclusivity.** No agent or module imports a raw provider SDK. Every LLM call routes through an in-process LiteLLM Gateway enforcing fallback chains, caching, and guardrails.
- **INV-2: Single Vector Store.** Redis Stack (`RedisVL`/HNSW) serves as the unified store for both the regulatory RAG index and the Gateway's semantic prompt cache.
- **INV-3: Data Contracts are Blocking.** Great Expectations suites validate elasticity training data and regulatory text prior to DVC versioning or indexing; contract failures halt the pipeline.
- **INV-4: No Autonomous Publication.** AEGIS produces recommendations terminating at auto-approved routine sign-off or underwriter escalation — never publishing directly to a live rate table.
- **INV-5: Advisory Causal Output.** Causal elasticity estimates and bandit suggestions are never surfaced to a user without accompanying compliance citations and revenue/loss-ratio financial context.
- **INV-6: Grounding Failure Triggers Fallback.** When compliance retrieval groundedness drops below threshold, the system safely falls back to the last approved rate table and flags an escalation rather than guessing.
- **INV-7: 100% Complete Audit Record.** Every decision produces a structured, immutable, independently reconstructable audit record.
- **INV-8: 1,000-Line Module Ceiling.** Source modules under `src/` are capped at 1,000 lines, enforcing tight modularity and isolated testability.
- **INV-9: Explicit Single-Jurisdiction Scope.** Compliance verdicts explicitly denote evaluation against an illustrative jurisdiction corpus, never claiming multi-state legal coverage.
- **INV-10: Modeled Core-System Contracts.** Carrier core-system API contracts are modeled and mocked in v1 without live external network connections.

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
- **Agent Orchestration:** LangGraph (Coordinator with parallel Compliance & Impact branches)
- **Deterministic & Causal ML:** statsmodels Tweedie GLM, EconML/DoWhy (`CausalForestDML`); contextual bandit is Phase 3
- **LLM Gateway & Caching:** LiteLLM (In-Process Gateway), Redis Stack (`RedisVL`/HNSW)
- **Data Versioning & Contracts:** DVC, Great Expectations (GX)
- **Experiment Tracking:** MLflow
- **API & Serving:** FastAPI, Docker, Docker Compose
- **Quality & Evaluation:** Ruff, Pyright, Pytest, Retrieval-Quality Groundedness Harness, LLM-as-Judge Suite

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
│   └── docs/            # Architecture ADRs, PRD, Roadmap, Charter, Runbooks
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
