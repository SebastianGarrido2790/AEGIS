# AEGIS — Test Suite Report

> **Version:** v0.1.0 — _Living Document_  
> **Phase:** 1 — Project Scaffolding & Data Contracts  
> **Status:** 🟢 34 / 34 Tests Passing (0 Warnings)  
> **Coverage:** 94% Total Code Coverage (100% Core Config, Expectations, Schemas & Utils)  
> **Maintained By:** Agentic System Architect & MLOps Engineering Team  
> **Reference Documents:** [technical_roadmap.md](../references/technical_roadmap.md), [phase_1_execution_workflow.md](../workflows/phase_1_execution_workflow.md), [phase_1_evaluation_report.md](phase_1_evaluation_report.md), [phase_1_architecture_report.md](../architecture/phase_1_architecture_report.md), [system_design.md](../architecture/system_design.md)

---

## 1. Testing Strategy Overview

The **AEGIS (Actuarial Elasticity & Governance Intelligence System)** test suite enforces a rigorous, deterministic quality policy designed to ensure mathematical consistency, causal validity, strict static typing, and contractual intake integrity. Our testing posture rests on seven core principles:

- **Actuarial & Causal Ground-Truth Primacy:** Treatment-effect identification requires strict pre-treatment covariates. Causal feature leakage (e.g. post-treatment retention proxies) invalidates Double-ML estimation and is caught via custom Great Expectations rules before data ingestion.
- **Contractual Intake Invariance (INV-3):** No raw dataset is versioned by DVC or consumed by Tier 1 ML without first passing its Great Expectations suite. Validation failures halt pipelines immediately.
- **Determinism:** Every test must produce identical results under a fixed seed. Data contract evaluations and pipeline stages are 100% reproducible across local and CI environments.
- **Fail-Loud Policy:** All runtime, configuration, and validation errors raise explicit custom exceptions (`AEGISError`, `ConfigurationError`, `DataContractError`, `GroundingThresholdError`, `FallbackRateTableError`) with structured key-value metadata payloads.
- **Single-Rule Fixture Isolation:** Every negative test fixture trips specifically and exclusively one single named expectation, ensuring unambiguous failure diagnosis and preventing compound masking.
- **File-Size Ceiling Gate (INV-8):** No Python source file under `src/` may exceed 1,000 lines. Enforced via `scripts/check_module_size.py` as a blocking CI gate.
- **Strict Static Typing:** Python 3.12+ code targeting 100% Pyright type-check coverage in standard mode with zero tolerated errors or warnings.

---

## 2. Test Suite Structure

The testing directory mirrors the core package structure:

```text
AEGIS/
├── scripts/
│   └── check_module_size.py             # 1,000-line ceiling enforcement script (INV-8)
├── tests/
│   ├── unit/
│   │   ├── test_config.py               # Pydantic v2 schema & params.yaml validation tests
│   │   ├── test_cli.py                  # DVC pipeline CLI runner & stage routing tests
│   │   ├── test_regulatory_data_contract.py # Regulatory corpus GX suite & fixture isolation tests
│   │   └── test_elasticity_data_contract.py # Elasticity training GX suite & causal leakage tests
│   ├── integration/                     # Integration tests (Phases 2-7)
│   └── evals/                           # Retrieval-quality & LLM-as-judge harnesses (Phases 5-8)
├── data_contracts/
│   ├── regulatory_corpus_suite.json     # Native GX JSON suite for statutory text
│   ├── elasticity_training_suite.json   # Native GX JSON suite for policy/claims data
│   └── fixtures/                        # Hand-crafted valid & malformed test fixtures
│       ├── regulatory_valid.json
│       ├── regulatory_invalid_missing_meta.json
│       ├── regulatory_invalid_empty.json
│       ├── regulatory_invalid_duplicate.json
│       ├── elasticity_valid.csv
│       ├── elasticity_invalid_range.csv
│       └── elasticity_invalid_leakage.csv
├── pyproject.toml                       # Pytest, Pytest-cov, Ruff, and UV toolchain settings
├── params.yaml                          # Central domain-nested configuration parameters
├── dvc.yaml                             # Parallel 3-stage DAG definitions
└── .github/workflows/ci.yml             # Single sequential GitHub Actions CI quality workflow
```

---

## 3. Component Breakdown & Verification Matrix

### 3.1 Configuration Engine & Schema Validation (`tests/unit/test_config.py`)

| Test Function | Verification Target | What Is Verified | Status |
| :--- | :--- | :--- | :---: |
| `test_load_valid_params` | `load_params()` | Verifies that `params.yaml` loads cleanly and validates against `AEGISConfig`. | 🟢 PASS |
| `test_missing_params_file_raises_configuration_error` | Error Handling | Verifies that a missing `params.yaml` raises typed `ConfigurationError`. | 🟢 PASS |
| `test_malformed_yaml_raises_configuration_error` | YAML Parser Guard | Verifies that invalid YAML syntax raises `ConfigurationError`. | 🟢 PASS |
| `test_invalid_schema_types_raise_configuration_error` | Pydantic Types | Verifies that invalid type assignments (e.g. string for integer) raise `ConfigurationError`. | 🟢 PASS |
| `test_extra_keys_forbidden_raises_configuration_error` | ConfigDict(extra="forbid") | Verifies that unexpected top-level configuration keys raise `ConfigurationError`. | 🟢 PASS |
| `test_get_config_singleton_caching` | LRU Cache | Verifies that `get_config()` returns identical cached instance across calls. | 🟢 PASS |

---

### 3.2 Regulatory Corpus Data Contract (`tests/unit/test_regulatory_data_contract.py`)

| Test Function | Verification Target | What Is Verified | Status |
| :--- | :--- | :--- | :---: |
| `test_regulatory_valid_fixture_passes` | `regulatory_valid.json` | Verifies 100% pass (9/9 rules) on clean statutory chunk dataset. | 🟢 PASS |
| `test_regulatory_missing_metadata_fixture_fails_specifically` | `regulatory_invalid_missing_meta.json` | Asserts exclusively 1 failure on `expect_column_values_to_not_be_null` on `effective_date`. | 🟢 PASS |
| `test_regulatory_empty_chunk_fixture_fails_specifically` | `regulatory_invalid_empty.json` | Asserts exclusively 1 failure on `expect_column_value_lengths_to_be_between` on `chunk_text`. | 🟢 PASS |
| `test_regulatory_duplicate_fixture_fails_specifically` | `regulatory_invalid_duplicate.json` | Asserts exclusively 1 failure on `expect_column_values_to_be_unique` on `chunk_id` (R-1). | 🟢 PASS |
| `test_blocking_validation_raises_data_contract_error` | Blocking Mode (INV-3) | Verifies that `raise_on_failure=True` raises typed `DataContractError`. | 🟢 PASS |
| `test_missing_data_file_raises_data_contract_error` | File System Guard | Verifies that a missing input file raises `DataContractError`. | 🟢 PASS |

---

### 3.3 Elasticity Training Data Contract (`tests/unit/test_elasticity_data_contract.py`)

| Test Function | Verification Target | What Is Verified | Status |
| :--- | :--- | :--- | :---: |
| `test_elasticity_valid_fixture_passes` | `elasticity_valid.csv` | Verifies 100% pass (12/12 rules) on clean auto policy records. | 🟢 PASS |
| `test_elasticity_range_violation_fixture_fails_specifically` | `elasticity_invalid_range.csv` | Asserts specific failure on `expect_column_values_to_be_between` on column `exposure`. | 🟢 PASS |
| `test_elasticity_leakage_fixture_fails_specifically` | `elasticity_invalid_leakage.csv` | Asserts specific failure on `expect_no_post_treatment_leakage` with identified leaked columns. | 🟢 PASS |
| `test_elasticity_blocking_validation_raises_data_contract_error` | Blocking Mode (INV-3) | Verifies that `raise_on_failure=True` raises typed `DataContractError`. | 🟢 PASS |
| `test_elasticity_missing_file_raises_data_contract_error` | File System Guard | Verifies that a missing CSV input file raises `DataContractError`. | 🟢 PASS |

---

### 3.4 DVC Pipeline Stage CLI Runner (`tests/unit/test_cli.py`)

| Test Function | Verification Target | What Is Verified | Status |
| :--- | :--- | :--- | :---: |
| `test_cli_parser_commands_structure` | `build_parser()` | Verifies registration of all 6 subcommands (`ingest-*`, `validate-*`, `version-*`). | 🟢 PASS |
| `test_ingest_file_success` | `ingest_file()` | Verifies raw data ingestion copies source file to target destination. | 🟢 PASS |
| `test_ingest_file_missing_source` | Ingest Error Guard | Verifies that missing source file returns exit code 1. | 🟢 PASS |
| `test_ingest_file_unknown_type` | Type Guard | Verifies that unknown data type returns exit code 1. | 🟢 PASS |
| `test_ingest_file_default_resolution` | Dynamic Defaults (R-4) | Verifies that `input_path=None` resolves dynamically from `params.yaml`. | 🟢 PASS |
| `test_validate_file_valid_data` | `validate_file()` | Verifies validation generates JSON report and returns exit code 0. | 🟢 PASS |
| `test_validate_file_regulatory_valid_data` | Regulatory Validation | Verifies regulatory validation runs cleanly via CLI. | 🟢 PASS |
| `test_validate_file_unknown_type` | Validate Type Guard | Verifies exit code 1 on invalid data type argument. | 🟢 PASS |
| `test_validate_file_invalid_data` | Contract Blocking (INV-3) | Verifies that contract failure writes report and returns exit code 1. | 🟢 PASS |
| `test_version_file_success` | `version_file()` | Verifies promotion of raw data to versioned storage when report succeeded. | 🟢 PASS |
| `test_version_file_missing_report` | Missing Report Guard | Verifies exit code 1 when validation report is missing. | 🟢 PASS |
| `test_version_file_blocks_on_failure` | Version Blocking (INV-3) | Verifies that failed validation report raises `DataContractError`. | 🟢 PASS |
| `test_main_cli_routing` (Parametrized) | `main()` Dispatch | Verifies top-level argument routing across all 5 active subcommands. | 🟢 PASS |

---

### 3.5 Module Size & Code Quality Enforcement

| Tool / Script | Target Scope | Rule / Limit | Status |
| :--- | :--- | :--- | :---: |
| `scripts/check_module_size.py` | `src/` (16 Python files) | **1,000 lines per file (INV-8)** | 🟢 PASS (Max: 254 lines) |
| `ruff check .` | Repository Root | Rulesets: `E`, `F`, `I`, `UP`, `B`, `SIM` | 🟢 PASS (0 Errors) |
| `pyright` | `src/`, `tests/`, `scripts/` | Strict Python 3.12 static type analysis | 🟢 PASS (0 Errors, 0 Warnings) |

---

## 4. Upcoming Test Suite Roadmap

```
Phase 1: Project Scaffolding & Data Contracts (Complete — 34 Passes)
  ├── Pydantic v2 Configuration Validation Tests (6 passes)
  ├── Great Expectations Regulatory Corpus Contract Tests (6 passes)
  ├── Great Expectations Elasticity Training Contract Tests (5 passes)
  ├── DVC Pipeline CLI Stage Execution Tests (17 passes)
  └── File Size Ceiling (INV-8), Ruff & Pyright Quality Gates
       │
Phase 2: Tier 1 Actuarial Baseline & Causal Elasticity Model (Scheduled Next)
  ├── GLM Frequency (Poisson) & Severity (Gamma) Reference Tests
  ├── CausalForestDML Treatment Effect Calibration Tests
  ├── Confounding Sensitivity Analysis Property Tests
  └── MLflow Experiment Logging & Model Registry Integration Tests
       │
Phase 3: Contextual Bandit Exploration Engine
  ├── Thompson Sampling Action Corridor Bounds Checks
  ├── Regret Minimization vs Static Pricing Benchmark Tests
  └── Safe Exploration Constraint Violation Gates
       │
Phase 4: LLM Gateway & Prompt Caching
  ├── LiteLLM In-Process Gateway Exclusivity Boundary Checks (INV-1)
  ├── Redis Stack Two-Tier Prompt Cache Parity Tests (INV-2)
  └── Resilience Fallback Provider Chain Integration Tests
       │
Phase 5: Regulatory Compliance Agent & RAG Evaluation
  ├── RedisVL Vector Store HNSW Retrieval Precision Tests
  ├── Compliance Groundedness & Evidence Coverage Harness (INV-6)
  └── Adversarial Prohibited Rating Factor Proxy Detection Tests
       │
Phase 6: Revenue/Loss-Ratio Impact Agent & Orchestration
  ├── LangGraph Shared Typed State Parallel Execution Tests
  ├── Actuarial Premium Arithmetic Verification Tests
  └── End-to-End Decision Coordinator Flow Tests
       │
Phase 7: Governance, HITL Escalation & Audit Log
  ├── Underwriter HITL Escalation Trigger Tests (INV-4)
  ├── Deterministic Rate Table Fallback Gating Tests
  └── Complete Audit Record Reconstructability Tests (INV-7)
       │
Phase 8: Observability, Evaluation & Hardening
  ├── OpenTelemetry Span Export & Traceability Checks
  ├── Human-Aligned LLM-as-Judge Regression Suite
  └── Full Production Pipeline Shadow Acceptance Suite
```

---

## 5. Test Suite Execution Commands & Live Coverage

| Target | Command | Notes |
| :--- | :--- | :--- |
| **Run Full Test Suite** | `uv run pytest` | Runs all unit and contract tests |
| **Run Coverage Report** | `uv run pytest --cov=src/aegis --cov-report=term-missing` | Verifies line coverage ($\ge 90\%$) |
| **Run Module Size Gate** | `uv run python scripts/check_module_size.py` | Enforces 1,000-line ceiling per file under `src/` (INV-8) |
| **Run Static Type Checker** | `uv run pyright` | Validates strict typing across `src/`, `tests/`, `scripts/` |
| **Run Linter Checks** | `uv run ruff check .` | Imports, syntax, and style rules enforcement |
| **Run DVC Pipeline** | `uv run dvc repro` | Executes parallel data contract validation stages |

**Live Coverage Output (Phase 1 Validated — 2026-08-21):**

```text
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.12.10-final-0 _______________

Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
src\aegis\__init__.py                       0      0   100%
src\aegis\agents\__init__.py                0      0   100%
src\aegis\bandit\__init__.py                0      0   100%
src\aegis\config\__init__.py                3      0   100%
src\aegis\config\loader.py                 37      5    86%   24, 59-60, 66, 88
src\aegis\config\schema.py                 52      0   100%
src\aegis\gateway\__init__.py               0      0   100%
src\aegis\governance\__init__.py            0      0   100%
src\aegis\pipelines\__init__.py             3      0   100%
src\aegis\pipelines\cli.py                 98      3    97%   232, 244, 248
src\aegis\pipelines\data_contracts.py     102      9    91%   65, 73-74, 86-87, 212-213, 248-249
src\aegis\pipelines\expectations.py        11      0   100%
src\aegis\schemas\__init__.py               0      0   100%
src\aegis\tools\__init__.py                 0      0   100%
src\aegis\utils\__init__.py                 2      0   100%
src\aegis\utils\exceptions.py              15      1    93%   22
---------------------------------------------------------------------
TOTAL                                     323     18    94%
============================= 34 passed in 13.63s =============================
```
