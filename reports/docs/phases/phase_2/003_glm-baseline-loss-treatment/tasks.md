# Tasks: P2-F03 GLM Loss-Cost Baseline & Calibration Reporting
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F03` (`003_glm-baseline-loss-treatment`)
**Spec:** APPROVED October 21, 2026 (OD-1…OD-5)
**Plan:** `plan.md` (consistent with the approved spec)
**Status:** COMPLETED — Gate 3 PASSED (2026-10-21)
**Date:** October 21, 2026

---

## Prerequisite

**P0 — Clean baseline.** Stage 2 must be committed and the working tree clean before T1. The plan and tasks below assume no uncommitted Stage 2 work; if the tree is dirty, stop and resolve it first.

---

## Task List

### T1 — Configuration surface for GLM baseline
- **Objective:** Declare the cap default, variance-power grid, fallback, decile count, and output paths in `params.yaml`; validate them in Pydantic.
- **FR / criteria:** FR-4, FR-13, AC-8.
- **Scope:** Additive config only; no fit logic.
- **Dependencies:** P0.
- **Affected files:** `params.yaml`, `src/aegis/config/schema.py`, `tests/unit/test_config.py`.
- **Done when:** Config loads; `p_min >= p_max`, non-positive steps, fallback outside `(1, 2)`, `link != "log"`, or `decile_count < 2` raise `ConfigurationError`; existing config tests remain green.
- **Validation:** `uv run pytest tests/unit/test_config.py -q`; `uv run ruff check .`; `uv run pyright`.

### T2 — Typed errors and GLM metadata contracts
- **Objective:** Add `GLMBaselineError` and Pydantic models for the profile point, model metadata, A/E row, secondary diagnostics, and calibration report.
- **FR / criteria:** FR-14, AC-9.
- **Scope:** Error type and serialization models; no fitting.
- **Dependencies:** T1.
- **Affected files:** `src/aegis/utils/exceptions.py`, `src/aegis/schemas/glm.py`, `tests/unit/test_glm_baseline.py`.
- **Done when:** Metadata/report with missing/extra fields are rejected; error details carry rule, column, count, and sample keys.
- **Validation:** `uv run pytest tests/unit/test_glm_baseline.py -q`; `ruff`; `pyright`.

### T3 — Fit core: design matrix, target, split, profile deviance, fit
- **Objective:** Implement `build_design_matrix`, `build_target`, `split_indices`, `select_tweedie_p`, and `fit_glm`.
- **FR / criteria:** FR-1, FR-2, FR-3, FR-5, FR-6, FR-7, FR-13, AC-1.
- **Scope:** Fitting and parameter selection; no persistence or calibration.
- **Dependencies:** T2.
- **Affected files:** `src/aegis/pipelines/glm_baseline.py`, `tests/unit/test_glm_baseline.py`.
- **Done when:** Target matches `min(ClaimAmount, cap)/exposure_capped`; design-matrix columns equal the frozen contract; split is deterministic 80/20; `select_tweedie_p` returns an interior `p` on synthetic data and honors fallback on degenerate data; a non-converged final fit raises.
- **Validation:** `uv run pytest tests/unit/test_glm_baseline.py -k "design or target or split or p or converge" -q`.

### T4 — Artifact writer, metadata, and orchestration
- **Objective:** Compute per-policy fitted loss cost, write the fitted CSV and model metadata JSON, validate them, and implement `run_glm_baseline`.
- **FR / criteria:** FR-4, FR-8, FR-14, AC-5, AC-8, AC-9.
- **Scope:** Persistence and orchestration; no new transforms.
- **Dependencies:** T3.
- **Affected files:** `src/aegis/pipelines/glm_baseline.py`, `src/aegis/schemas/glm.py`, `tests/unit/test_glm_baseline.py`.
- **Done when:** Two consecutive runs produce equal metadata/fitted-CSV hashes; metadata round-trips through the Pydantic model; cap is a parameter.
- **Validation:** `uv run pytest tests/unit/test_glm_baseline.py -k "metadata or determinism or cap" -q`.

### T5 — Calibration evaluation module
- **Objective:** Implement holdout mean deviance, decile A/E, segment-band A/E, and secondary MAE/RMSE, and assemble the typed report.
- **FR / criteria:** FR-8, FR-9, FR-10, FR-11, AC-2, AC-4, AC-7.
- **Scope:** Evaluation only; consumes fitted results.
- **Dependencies:** T4.
- **Affected files:** `src/aegis/pipelines/glm_calibration.py`, `tests/unit/test_glm_calibration.py`.
- **Done when:** Decile A/E is ratio-of-sums and bins partition the test set; segment A/E covers the three band dimensions plus the 36-cell cross; MAE/RMSE are computed but not used by any gate.
- **Validation:** `uv run pytest tests/unit/test_glm_calibration.py -q`.

### T6 — CLI subcommand and DVC stage
- **Objective:** Add `build-glm` to the CLI and wire `glm_baseline` after `build_features` in `dvc.yaml`.
- **FR / criteria:** AC-10.
- **Scope:** Entrypoint and DAG wiring only.
- **Dependencies:** T5.
- **Affected files:** `src/aegis/pipelines/cli.py`, `dvc.yaml`, `tests/unit/test_cli.py`.
- **Done when:** CLI runs the full baseline; `dvc repro` executes `glm_baseline` after `build_features` and produces the three artifacts; the DAG does not bypass the feature pipeline.
- **Validation:** `uv run pytest tests/unit/test_cli.py -q`; `uv run dvc repro`.

### T7 — Determinism, single-model, and empirical integration tests
- **Objective:** Prove AC-5/AC-6 on the empirical input and add a repeat-run hash test and a structural single-model assertion.
- **FR / criteria:** FR-12, AC-5, AC-6.
- **Scope:** Integration and structural tests only.
- **Dependencies:** T6.
- **Affected files:** `tests/unit/test_glm_baseline.py`.
- **Done when:** Empirical run converges and produces deviance + decile/segment A/E; two runs yield equal artifact SHA-256; no frequency-only/severity-only estimator is present.
- **Validation:** `uv run pytest tests/unit/test_glm_baseline.py -q`.

### T8 — Full quality gate and evidence
- **Objective:** Run the complete validation suite, record Gate 3 evidence, and update Stage 3 statuses.
- **FR / criteria:** AC-10, Gate 3.
- **Scope:** Verification and documentation only; no behavior changes.
- **Dependencies:** T7.
- **Affected files:** `reports/docs/phases/phase_2/003_glm-baseline-loss-treatment/tasks.md`, `reports/docs/phases/phase_2/stage_evidence.md`, `reports/docs/phases/phase_2/execution_plan.md`.
- **Done when:** `pytest`, `ruff`, `pyright`, module-size check, and `dvc repro` are green; Stage 3 evidence records the selected `p`, convergence, holdout deviance, and A/E outputs; Gate 3 is marked PASSED.
- **Validation:** `uv run pytest -q`; `uv run ruff check .`; `uv run pyright`; `uv run python scripts/check_module_size.py`; `uv run dvc repro`.

---

## Traceability

| Requirement | Tasks |
|---|---|
| FR-1, FR-2, FR-3 | T3 |
| FR-4 | T1, T3 |
| FR-5 | T3 |
| FR-6 | T3 |
| FR-7 | T3 |
| FR-8 | T4, T5 |
| FR-9, FR-10 | T5 |
| FR-11 | T5 |
| FR-12 | T7 |
| FR-13 | T1, T3 |
| FR-14 | T2, T4 |
| Gate 3 | T8 |

## Completion

All tasks completed and verified on 2026-10-21. Gate 3 passed (`p=1.84`, converged, holdout deviance 16.9016, decile/segment A/E generated); executed evidence is recorded in `stage_evidence.md` §Stage 3.
