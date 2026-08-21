# Phase 1 Architecture Report — AEGIS

> **System:** Actuarial Elasticity & Governance Intelligence System (AEGIS)  
> **Phase:** Phase 1 — Project Scaffolding & Data Contracts  
> **Status:** 🟢 Validated & Complete  
> **Author:** Sebastián Garrido Arévalo | Date: August 21, 2026  
> **Related Documents:** [system_design.md](system_design.md), [phase_1_evaluation_report.md](../evaluations/phase_1_evaluation_report.md), [test_suite_report.md](../evaluations/test_suite_report.md), [phase_1_implementation_plan.md](../decisions/phase_1_implementation_plan.md)

---

## 1. Executive Summary & Architectural Scope

Phase 1 establishes the foundational infrastructure, dependency isolation, typed configuration engine, and data contract gating required to prevent corrupted, unvalidated, or causally contaminated data from entering AEGIS.

The core architectural imperative of Phase 1 is **Governance Over Raw Execution**: establishing day-one enforcement of non-negotiable invariants (**INV-1 through INV-10**), ensuring every dataset is contractually validated before versioning (**INV-3**), enforcing a 1,000-line modularity ceiling (**INV-8**), and providing a fully reproducible, zero-secrets continuous integration pipeline.

```
+---------------------------------------------------------------------------------------+
|                                    AEGIS REPOSITORY                                   |
|                                                                                       |
|  [ pyproject.toml / uv.lock ] ──────> [ params.yaml ] ──────> [ Pydantic v2 Schema ]  |
|         (Hatchling Backend)                 (Zero Secrets)               (Type Safety)|
|                                                                                       |
|  [ Data Contracts (GX Core) ] ──────> [ Custom Expectation ] ──> [ DVC Parallel DAG ] |
|     (JSON Declarative Suites)            (Causal Leakage Check)       (Fine-Grained)  |
|                                                                                       |
|  [ Scripts / Tooling ] ─────────────> [ Pytest Unit Matrix ] ──> [ GitHub Actions CI ]|
|       (INV-8 Line Check)                 (34 Tests, 94% Cov)          (Blocking Gates)|
+---------------------------------------------------------------------------------------+
```

---

## 2. Granular System Diagrams

### 2.1 Package Namespacing & Subsystem Topology

Per **ADR-004**, all source code is namespaced under `src/aegis/` to prevent top-level module collisions and enable clean package distribution via Hatchling.

```mermaid
graph TD
    subgraph RepoRoot["Repository Root"]
        PYP["pyproject.toml (Hatchling, uv.lock)"]
        PARAM["params.yaml (Domain-Nested Config)"]
        DVCY["dvc.yaml (Parallel DAG Definition)"]
        GHA[".github/workflows/ci.yml (Day-1 CI)"]
    end

    subgraph SourceTree["src/aegis/ (Namespaced Core)"]
        CONFIG["config/ (schema.py, loader.py)"]
        PIPELINES["pipelines/ (data_contracts.py, expectations.py, cli.py)"]
        UTILS["utils/ (exceptions.py)"]
        GATEWAY["gateway/ (LiteLLM Invariant Boundary - INV-1)"]
        AGENTS["agents/ (LangGraph Orchestration - Tier 2)"]
        GOV["governance/ (HITL, Fallback, Audit - Tier 3)"]
        BANDIT["bandit/ (Contextual Exploration)"]
        SCHEMAS["schemas/ (Pydantic I/O Models)"]
        TOOLS["tools/ (Deterministic Microservices)"]
    end

    subgraph DataContracts["data_contracts/ (Version-Controlled Contracts)"]
        ELAST_SUITE["elasticity_training_suite.json"]
        REG_SUITE["regulatory_corpus_suite.json"]
        FIXTURES["fixtures/ (*.csv, *.json)"]
    end

    RepoRoot --> SourceTree
    RepoRoot --> DataContracts
    CONFIG --> UTILS
    PIPELINES --> CONFIG
    PIPELINES --> UTILS
```

---

### 2.2 Domain-Nested Configuration Architecture (ADR-005)

Configuration is consolidated in `params.yaml` with strict, fail-loud Pydantic validation on load.

```mermaid
sequenceDiagram
    autonumber
    participant App as Application / CLI
    participant Loader as aegis.config.loader
    participant YAML as params.yaml
    participant Schema as aegis.config.schema.AEGISConfig
    participant Exc as ConfigurationError

    App->>Loader: get_config() / load_params()
    Loader->>Loader: find_project_root()
    Loader->>YAML: Read YAML text (utf-8)
    alt YAML missing or corrupted
        YAML-->>Loader: FileNotFoundError / ScannerError
        Loader->>Exc: Raise ConfigurationError("Failed to parse params.yaml")
    else YAML parsed successfully
        YAML-->>Loader: raw_dict
        Loader->>Schema: AEGISConfig.model_validate(raw_dict)
        alt Pydantic ValidationError (Missing required / Invalid type / Extra keys)
            Schema-->>Loader: ValidationError
            Loader->>Exc: Raise ConfigurationError("Configuration schema validation failed")
        else Validation Succeeded
            Schema-->>Loader: validated AEGISConfig instance
            Loader-->>App: AEGISConfig (Cached singleton)
        end
    end
```

---

### 2.3 Great Expectations Core 1.x Validation Flow (ADR-006)

Data contracts use ephemeral in-memory sessions without external cloud dependencies.

```mermaid
flowchart TD
    subgraph Intake["Data Intake Layer"]
        RAW["Raw Dataset (CSV / JSON)"]
        SUITE_JSON["Native Expectation Suite (*.json)"]
    end

    subgraph GXEngine["Great Expectations Core 1.x In-Memory Execution"]
        CTX["Ephemeral DataContext (gx.get_context)"]
        LOAD_SUITE["Load & Register ExpectationSuite"]
        DS["Pandas Datasource (ds_name)"]
        ASSET["Dataframe Asset (asset_name)"]
        BATCH_DEF["Batch Definition (Whole Dataframe)"]
        BATCH["Data Batch (In-Memory DataFrame)"]
        VALIDATE["batch.validate(suite)"]
    end

    subgraph Evaluation["Contract Evaluation & Invariant Gate (INV-3)"]
        RES["ExpectationSuiteValidationResult"]
        WRAPPER["ContractValidationResult (Structured Report)"]
        GATE{"Validation Succeeded?"}
        PASS["Return Passed Report (Exit 0)"]
        FAIL["Raise DataContractError / Halt DAG (Exit 1)"]
    end

    RAW --> DS
    SUITE_JSON --> LOAD_SUITE
    LOAD_SUITE --> CTX
    DS --> ASSET --> BATCH_DEF --> BATCH
    CTX --> VALIDATE
    BATCH --> VALIDATE
    VALIDATE --> RES
    RES --> WRAPPER
    WRAPPER --> GATE
    GATE -- Yes --> PASS
    GATE -- No (INV-3) --> FAIL
```

---

### 2.4 Custom Causal Leakage Expectation Metamodel

To enforce econometric validity in Tier 1 modeling, `ExpectNoPostTreatmentLeakage` intercepts post-treatment proxy variables before causal elasticity estimation.

```mermaid
classDiagram
    class BatchExpectation {
        +metric_dependencies: tuple
        +_validate(metrics, runtime_configuration, execution_engine)
    }

    class ExpectNoPostTreatmentLeakage {
        +metric_dependencies: tuple = ("table.columns",)
        +prohibited_columns: list[str]
        +_validate(metrics, runtime_configuration, execution_engine) dict
    }

    class RegulatoryCorpusSuite {
        +expect_table_columns_to_match_set
        +expect_column_values_to_not_be_null
        +expect_column_value_lengths_to_be_between
        +expect_column_values_to_be_unique
    }

    class ElasticityTrainingSuite {
        +expect_table_columns_to_match_set
        +expect_column_values_to_not_be_null
        +expect_column_values_to_be_between
        +expect_no_post_treatment_leakage
    }

    BatchExpectation <|-- ExpectNoPostTreatmentLeakage
    ElasticityTrainingSuite *-- ExpectNoPostTreatmentLeakage
```

---

### 2.5 DVC Parallel DAG Topology (ADR-007)

Two independent, parallel 3-stage workflows enforce fine-grained caching and contract execution.

```mermaid
flowchart LR
    subgraph ElasticityPipeline["1. Elasticity Training Data DAG"]
        IE["ingest_elasticity"]
        VE["validate_elasticity_gx"]
        VSE["version_elasticity"]

        IE -->|data/raw/elasticity_raw.csv| VE
        IE -->|data/raw/elasticity_raw.csv| VSE
        VE -->|data/validated/elasticity_validation_report.json| VSE
        VSE -->|Output| OUT_E["data/versioned/elasticity_training_data.csv"]
    end

    subgraph RegulatoryPipeline["2. Regulatory Corpus Data DAG"]
        IR["ingest_regulatory"]
        VR["validate_regulatory_gx"]
        VSR["version_regulatory"]

        IR -->|data/raw/regulatory_raw.json| VR
        IR -->|data/raw/regulatory_raw.json| VSR
        VR -->|data/validated/regulatory_validation_report.json| VSR
        VSR -->|Output| OUT_R["data/versioned/regulatory_corpus.json"]
    end
```

---

### 2.6 GitHub Actions Continuous Integration Flow (ADR-008)

```mermaid
graph TD
    PUSH["Git Push / Pull Request (main)"] --> G0["Ubuntu Latest Runner / Python 3.12 / uv sync"]
    G0 --> G1["Gate 1: Linting Check (Ruff)"]
    G1 --> G2["Gate 2: Static Type Analysis (Pyright)"]
    G2 --> G3["Gate 3: Module Size Limit (scripts/check_module_size.py - INV-8)"]
    G3 --> G4["Gate 4: Data Contract Pipeline Reproduction (dvc repro - INV-3)"]
    G4 --> G5["Gate 5: Automated Test Matrix (pytest --cov)"]
    G5 --> SUCCESS["🟢 CI Status: PASS (Merge Permitted)"]

    G1 -- Exit != 0 --> FAIL["🔴 CI Status: BLOCKED"]
    G2 -- Exit != 0 --> FAIL
    G3 -- Exit != 0 --> FAIL
    G4 -- Exit != 0 --> FAIL
    G5 -- Exit != 0 --> FAIL
```

---

## 3. Design Patterns & Architectural Decisions

| Pattern                                  | Implementation                                                             | Architectural Rationale                                                                                                                                   |
| ---------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Domain-Nested Typed Configuration**    | [src/aegis/config/schema.py](src/aegis/config/schema.py)                   | Eliminates flat key namespace collision; validates all configuration parameters at startup with typed Pydantic models.                                    |
| **Declarative File-Based Contracts**     | [data_contracts/\*.json](data_contracts/)                                  | Decouples data validation rules from imperative application code; expectation suites are versioned natively in Git.                                       |
| **Custom Batch Expectation AST**         | [src/aegis/pipelines/expectations.py](src/aegis/pipelines/expectations.py) | Allows domain-specific econometric validation (prohibiting post-treatment feature leakage) while maintaining 100% JSON suite serialization compatibility. |
| **Dynamic CLI Configuration Resolution** | [src/aegis/pipelines/cli.py](src/aegis/pipelines/cli.py)                   | CLI argument defaults resolve lazily from `params.yaml` via `get_config()`, eliminating path duplication and preventing silent drift.                     |
| **Isolated Rule Failure Design**         | [data_contracts/fixtures/](data_contracts/fixtures/)                       | Every malformed test fixture trips specifically and exclusively one single named expectation, ensuring unambiguous failure diagnosis.                     |
| **Fail-Loud Exception Hierarchy**        | [src/aegis/utils/exceptions.py](src/aegis/utils/exceptions.py)             | All system exceptions derive from `AEGISError`, formatting detailed key-value metadata payloads for deterministic stack traces.                           |

---

## 4. Invariant Enforcement Matrix

| Invariant ID | Requirement                                                 | Enforcement Mechanism                                                  | Phase 1 Verification                                                         |
| :----------- | :---------------------------------------------------------- | :--------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **INV-1**    | Gateway exclusivity (No raw LLM imports)                    | Directory isolation & upcoming CI import boundary check                | Gateway stub initialized; zero external provider SDKs imported.              |
| **INV-2**    | Single vector store (Redis Stack / RedisVL only)            | CI dependency check                                                    | `pyproject.toml` contains zero secondary vector databases.                   |
| **INV-3**    | Data contracts are blocking (GX gates DVC versioning)       | `cli.py`, `dvc.yaml`, `validate_dataframe()` raise `DataContractError` | Ingesting bad data halts `dvc repro` and prevents promotion to `versioned/`. |
| **INV-8**    | Module size ceiling (Max 1,000 lines per file under `src/`) | `scripts/check_module_size.py` in CI Gate 3                            | All 16 source files scanned: max size is 254 lines (0 violations).           |
| **INV-10**   | Local-first zero cloud credentials in v1                    | Local filesystem remote (`.dvc/local_remote`), zero secrets in CI      | DVC operates entirely on local disk; CI pipeline requires zero secrets.      |

---

## 5. Technical Implementation Stages Summary

```
Stage 1: Repository Bootstrap (pyproject.toml, uv.lock, Python 3.12, src/aegis/)
   │
Stage 2: Configuration Schema & Module Size Checker (params.yaml, schema.py, check_module_size.py)
   │
Stage 3: Regulatory Corpus Data Contract (regulatory_corpus_suite.json, fixtures, data_contracts.py)
   │
Stage 4: Elasticity Training Suite & Leakage Check (expectations.py, elasticity_training_suite.json)
   │
Stage 5: DVC Pipeline Skeleton (dvc.yaml, dvc.lock, cli.py, local remote)
   │
Stage 6: CI Quality Gate & Invariant Falsification Pass (.github/workflows/ci.yml, 5/5 gates proven)
```

Phase 1 provides a fully verified, type-safe foundation ready for **Phase 2 — Tier 1: Deterministic ML Baseline**.
