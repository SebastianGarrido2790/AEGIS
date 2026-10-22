# Tasks: P2-F02 Deterministic Feature Pipeline
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F02` (`002_feature-pipeline`)  
**Spec:** APPROVED October 21, 2026  
**Plan:** `plan.md` (consistent with the approved spec)  
**Status:** COMPLETED — Gate 2 PASSED (2026-10-21)  
**Date:** October 21, 2026

---

## Prerequisite

**P0 — Clean baseline.** Stage 1 must be committed and the working tree clean before T1. The plan and tasks below assume no uncommitted Stage 1 work; if the tree is dirty, stop and resolve it first.

---

## Task List

### T1 — Configuration surface for feature rules
- **Objective:** Declare exposure cap, age bounds, band edges/labels, fixed factor level lists, and output paths in `params.yaml`; validate them in Pydantic.
- **FR / criteria:** FR-4, FR-8, AC-5, AC-8.
- **Scope:** Additive config only; no transform logic.
- **Dependencies:** P0.
- **Affected files:** `params.yaml`, `src/aegis/config/schema.py`, `tests/unit/test_config.py`.
- **Done when:** Config loads; bad edge order, mismatched label counts, or non-positive bounds raise `ConfigurationError`; existing config tests remain green.
- **Validation:** `uv run pytest tests/unit/test_config.py -q`; `uv run ruff check .`; `uv run pyright`.

### T2 — Typed errors and metadata contracts
- **Objective:** Add `FeaturePipelineError` and Pydantic models for clip counts, view contracts, distribution summary, and metadata.
- **FR / criteria:** FR-11, FR-13, AC-8, AC-9.
- **Scope:** Error type and serialization models; no transformation.
- **Dependencies:** T1.
- **Affected files:** `src/aegis/utils/exceptions.py`, `src/aegis/schemas/features.py`, `tests/unit/test_feature_pipeline.py`.
- **Done when:** Metadata with missing/extra fields is rejected; error details carry column, rule, count, and sample keys.
- **Validation:** `uv run pytest tests/unit/test_feature_pipeline.py -q`; `ruff`; `pyright`.

### T3 — Input contract validation
- **Objective:** Validate exact columns, dtypes, duplicate `PolicyID`, nulls, `Exposure > 0`, `Density > 0`, non-empty input, and category membership before any transform.
- **FR / criteria:** FR-2, FR-13, AC-9.
- **Scope:** Validation helpers and typed failures; no derived columns.
- **Dependencies:** T2.
- **Affected files:** `src/aegis/pipelines/feature_pipeline.py`, `tests/unit/test_feature_pipeline.py`.
- **Done when:** Each malformed case raises `FeaturePipelineError` naming the column/rule; no partial output is written.
- **Validation:** `uv run pytest tests/unit/test_feature_pipeline.py -k "invalid or contract" -q`.

### T4 — Core derivations: exposure, density, bands, tail clips
- **Objective:** Clip exposure to 1.0; clip DriverAge/CarAge to approved bounds; derive `log_density` and the three bands; return exact clip counts.
- **FR / criteria:** FR-1, FR-3, FR-4, FR-8, AC-2, AC-5, AC-6, AC-7.
- **Scope:** Derivation only; views and persistence follow.
- **Dependencies:** T3.
- **Affected files:** `src/aegis/pipelines/feature_pipeline.py`, `tests/unit/test_feature_pipeline.py`.
- **Done when:** Boundary values map to the intended band; 421 exposure clips are counted on the empirical input; clip counts match independent recomputation.
- **Validation:** `uv run pytest tests/unit/test_feature_pipeline.py -k "clip or band or density" -q`.

### T5 — GLM view
- **Objective:** Emit the fixed-order GLM view with identity fields, capped exposure, `ClaimAmount`, raw ages, `log_density`, bands, and categorical factors.
- **FR / criteria:** FR-5, AC-8.
- **Scope:** View construction only.
- **Dependencies:** T4.
- **Affected files:** `src/aegis/pipelines/feature_pipeline.py`, `tests/unit/test_feature_pipeline.py`.
- **Done when:** Column list equals the approved contract exactly; row count equals input; no claims field other than the carried `ClaimAmount` appears.
- **Validation:** `uv run pytest tests/unit/test_feature_pipeline.py -k "glm" -q`.

### T6 — Forest view and fixed encoding
- **Objective:** Emit the numeric/ordinal forest view with ordinal Power and band codes and fixed one-hot columns for Brand, Gas, Region.
- **FR / criteria:** FR-6, FR-7, AC-3, AC-8.
- **Scope:** View construction and encoding metadata.
- **Dependencies:** T4.
- **Affected files:** `src/aegis/pipelines/feature_pipeline.py`, `tests/unit/test_feature_pipeline.py`.
- **Done when:** No claims field or Exposure covariate appears in `forest_covariates`; one-hot columns are identical when a level is absent; slug collisions fail loudly.
- **Validation:** `uv run pytest tests/unit/test_feature_pipeline.py -k "forest or encoding" -q`.

### T7 — Artifact writer, metadata, and distribution summary
- **Objective:** Write both views deterministically, hash the bytes, build the metadata (counts, hashes, edges, mappings, feature lists, summary), validate it, and write JSON.
- **FR / criteria:** FR-9, FR-11, AC-1, AC-4, AC-8.
- **Scope:** Persistence and metadata; no new transforms.
- **Dependencies:** T5, T6.
- **Affected files:** `src/aegis/pipelines/feature_pipeline.py`, `tests/unit/test_feature_pipeline.py`.
- **Done when:** Two consecutive writes produce equal hashes and metadata; metadata round-trips through the Pydantic model; whitelist test finds no composite column.
- **Validation:** `uv run pytest tests/unit/test_feature_pipeline.py -k "metadata or determinism or composite" -q`.

### T8 — CLI subcommand and DVC stage
- **Objective:** Add `build-features` to the CLI and wire `build_features` after `version_raw` in `dvc.yaml`.
- **FR / criteria:** FR-12, AC-10.
- **Scope:** Entrypoint and DAG wiring only.
- **Dependencies:** T7.
- **Affected files:** `src/aegis/pipelines/cli.py`, `dvc.yaml`, `tests/unit/test_cli.py`.
- **Done when:** CLI runs the full pipeline; `dvc repro` executes `build_features` after `version_raw` and produces the three artifacts; the DAG does not bypass the raw GX gate.
- **Validation:** `uv run pytest tests/unit/test_cli.py -q`; `uv run dvc repro`.

### T9 — Determinism and empirical integration tests
- **Objective:** Prove AC-1/AC-2/AC-6 on the empirical input and add a repeat-run hash test.
- **FR / criteria:** FR-10, AC-1, AC-2, AC-6, AC-11.
- **Scope:** Integration tests only.
- **Dependencies:** T8.
- **Affected files:** `tests/unit/test_feature_pipeline.py`.
- **Done when:** Full empirical transform yields exactly 421 exposure clips; two runs yield equal SHA-256 for both views; independent `ln(Density)` check passes.
- **Validation:** `uv run pytest tests/unit/test_feature_pipeline.py -q`.

### T10 — Full quality gate and evidence
- **Objective:** Run the complete validation suite, record Gate 2 evidence, and update the Stage 2 statuses.
- **FR / criteria:** AC-10, AC-11, Gate 2.
- **Scope:** Verification and documentation only; no behavior changes.
- **Dependencies:** T9.
- **Affected files:** `reports/docs/phases/phase_2/002_feature-pipeline/tasks.md`, `reports/docs/phases/phase_2/stage_evidence.md`, `reports/docs/phases/phase_2/execution_plan.md`.
- **Done when:** `pytest`, `ruff`, `pyright`, module-size check, and `dvc repro` are green; Stage 2 evidence records exact counts, hashes, and outcomes; Gate 2 is marked PASSED.
- **Validation:** `uv run pytest -q`; `uv run ruff check .`; `uv run pyright`; `uv run python scripts/check_module_size.py`; `uv run dvc repro`.

---

## Traceability

| Requirement | Tasks |
|---|---|
| FR-1, FR-2 | T3, T4, T9 |
| FR-3, FR-8 | T4, T9 |
| FR-4 | T1, T4 |
| FR-5 | T5 |
| FR-6, FR-7 | T6 |
| FR-9 | T7 |
| FR-10, FR-11 | T7, T9 |
| FR-12 | T8 |
| FR-13 | T2, T3 |
| Gate 2 | T10 |

## Completion

All tasks completed and verified on 2026-10-21. Gate 2 passed; executed evidence is recorded in `stage_evidence.md` §Stage 2.
