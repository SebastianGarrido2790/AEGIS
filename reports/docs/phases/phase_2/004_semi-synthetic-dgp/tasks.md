# Tasks: P2-F04 Semi-Synthetic DGP & Training Contract
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F04` (`004_semi-synthetic-dgp`)  
**Spec:** APPROVED October 26, 2026 (OD-1…OD-4)  
**Plan:** `plan.md` (consistent with the approved spec)  
**Status:** COMPLETED — Gate 6 PASSED (2026-10-26)  
**Date:** October 26, 2026  

---

## Prerequisite

**P0 — Clean baseline.** Stage 5 committed and working tree clean (`04903e6`).

---

## Task List

### T1 — Configuration surface for DGP
- **Objective:** Add `synthetic_dgp` parameter block and `dgp_paths` to `params.yaml`; validate via Pydantic schema in `src/aegis/config/schema.py`.
- **FR / criteria:** FR-2, FR-5, FR-7…FR-11, AC-2.
- **Scope:** Additive configuration; no generation code.
- **Dependencies:** P0.
- **Affected files:** `params.yaml`, `src/aegis/config/schema.py`, `tests/unit/test_config.py`.
- **Done when:** Config loads with approved OD parameters; bounds and threshold constraints validated; config unit tests green.
- **Validation:** `uv run pytest tests/unit/test_config.py -q`; `uv run ruff check .`; `uv run pyright`.

### T2 — Typed models and exception classes
- **Objective:** Implement `SelfTestResult`, `DGPManifest`, `TruthSidecarRow`, `TruthSidecarTable` schemas and `DGPVerificationError`, `TruthIsolationError` exceptions.
- **FR / criteria:** FR-12, FR-14, AC-3, AC-7.
- **Scope:** Schemas and exceptions.
- **Dependencies:** T1.
- **Affected files:** `src/aegis/utils/exceptions.py`, `src/aegis/schemas/dgp.py`, `tests/unit/test_dgp_schemas.py`.
- **Done when:** Pydantic schemas validate valid payloads and reject invalid types/missing fields; exceptions subclass `AEGISError`.
- **Validation:** `uv run pytest tests/unit/test_dgp_schemas.py -q`.

### T3 — Synthetic DGP core generator & 5 self-tests
- **Objective:** Implement `generate_semi_synthetic_dataset` and the 5 automated self-test functions in `src/aegis/pipelines/synthetic_dgp.py`.
- **FR / criteria:** FR-1…FR-11, AC-1, AC-2.
- **Scope:** Generator math, clipping, Bernoulli sampling, self-tests.
- **Dependencies:** T2.
- **Affected files:** `src/aegis/pipelines/synthetic_dgp.py`, `tests/unit/test_synthetic_dgp.py`.
- **Done when:** Treatment is bounded in $[-0.15, +0.15]$; renewal is binary in $\{0, 1\}$; true $q_0, \tau, \epsilon$ match mathematical formulas; all 5 self-tests pass; unit tests green.
- **Validation:** `uv run pytest tests/unit/test_synthetic_dgp.py -q`.

### T4 — Truth loader module
- **Objective:** Implement `load_truth_sidecar` in `src/aegis/pipelines/truth_loader.py` for isolated validation harness use.
- **FR / criteria:** FR-12, AC-7.
- **Scope:** Loader function with schema validation.
- **Dependencies:** T2, T3.
- **Affected files:** `src/aegis/pipelines/truth_loader.py`, `tests/unit/test_truth_loader.py`.
- **Done when:** Loader returns validated DataFrame; raises `TruthIsolationError` on missing/invalid files or unapproved caller context.
- **Validation:** `uv run pytest tests/unit/test_truth_loader.py -q`.

### T5 — Training table GX suite, fixtures & runner
- **Objective:** Re-base `data_contracts/elasticity_training_suite.json` to post-DGP schema; create valid/invalid fixtures; implement `validate_elasticity_training` runner in `data_contracts.py`.
- **FR / criteria:** FR-13, AC-4, AC-5 (Falsification 1).
- **Scope:** Data contract definition, fixtures, validation runner.
- **Dependencies:** T3.
- **Affected files:** `data_contracts/elasticity_training_suite.json`, `data_contracts/fixtures/`, `src/aegis/pipelines/data_contracts.py`, `tests/unit/test_data_contracts.py`.
- **Done when:** Valid fixture passes; fixture with truth column (`tau_ground_truth`) fails loudly; fixture with out-of-bounds treatment fails.
- **Validation:** `uv run pytest tests/unit/test_data_contracts.py -k "elasticity" -q`.

### T6 — CLI commands & DVC stages
- **Objective:** Add `generate-dgp` and `validate-training-data` CLI subcommands; add `generate_dgp` and `validate_training_data` stages to `dvc.yaml`.
- **FR / criteria:** FR-12, FR-13, AC-3, AC-4.
- **Scope:** Orchestration, CLI, DVC.
- **Dependencies:** T3, T5.
- **Affected files:** `src/aegis/pipelines/cli.py`, `dvc.yaml`, `tests/unit/test_cli.py`.
- **Done when:** CLI subcommands execute cleanly and return 0; DVC stages reproduce; three artifacts written.
- **Validation:** `uv run pytest tests/unit/test_cli.py -k "dgp" -q`.

### T7 — Falsification tests & import boundary test
- **Objective:** Add structural AST import-boundary test and dynamic falsification tests in `tests/unit/test_truth_isolation.py`.
- **FR / criteria:** FR-14, AC-5, AC-6.
- **Scope:** Tests only.
- **Dependencies:** T4, T5, T6.
- **Affected files:** `tests/unit/test_truth_isolation.py`.
- **Done when:** Falsification 1 (planted truth column fails GX suite) and Falsification 2 (import of truth loader by causal estimator fails boundary test) pass.
- **Validation:** `uv run pytest tests/unit/test_truth_isolation.py -q`.

### T8 — Full quality gate, Stage 6 execution, and evidence recording
- **Objective:** Run full pipeline, verify 413,169 rows, check artifact hashes and determinism, record evidence in `stage_evidence.md`, update `execution_plan.md`.
- **FR / criteria:** Gate 6.
- **Scope:** Execution and evidence documentation.
- **Dependencies:** T1…T7.
- **Affected files:** `reports/docs/phases/phase_2/stage_evidence.md`, `execution_plan.md`, `tasks.md`, `dvc.lock`.
- **Done when:** `uv run pytest -q`, `uv run ruff check .`, `uv run pyright`, `check_module_size.py`, and `uv run dvc repro` pass; Gate 6 marked PASSED.
- **Validation:** Verification commands in `AGENTS.md` §6.

---

## Traceability

| FR | Tasks |
|---|---|
| FR-1 | T3 |
| FR-2 | T1, T3 |
| FR-3, FR-4, FR-5, FR-6 | T3 |
| FR-7, FR-8, FR-9, FR-10, FR-11 | T1, T3 |
| FR-12 | T2, T3, T6 |
| FR-13 | T5, T6 |
| FR-14 | T4, T7 |
