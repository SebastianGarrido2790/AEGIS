# Tasks: P2-F05 Segment Definition & Tier 1 Output Contract
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F05` (`005_segments-tier1-contract`)
**Spec:** APPROVED October 21, 2026 (OD-1…OD-5)
**Plan:** `plan.md` (consistent with the approved spec)
**Status:** COMPLETED — Gate 4 PASSED (2026-10-21)
**Date:** October 21, 2026

---

## Prerequisite

**P0 — Clean baseline.** Stage 3 must be committed and the working tree clean before T1 (it is: `2e40180`).

---

## Task List

### T1 — Configuration surface for segments
- **Objective:** Declare `min_policy_count`, `premium_loading`, explicit `cell_merges`, and output paths in `params.yaml`; validate in Pydantic, including cross-checks against the frozen band labels.
- **FR / criteria:** FR-2, FR-3, FR-6, AC-3, AC-8.
- **Scope:** Additive config only; no assignment logic.
- **Dependencies:** P0.
- **Affected files:** `params.yaml`, `src/aegis/config/schema.py`, `tests/unit/test_config.py`.
- **Done when:** Config loads with the approved values; unknown band labels in a merge, duplicate/empty `merged_density_band`, `min_policy_count < 1`, or negative `premium_loading` raise `ConfigurationError`; existing config tests remain green.
- **Validation:** `uv run pytest tests/unit/test_config.py -q`; `uv run ruff check .`; `uv run pyright`.

### T2 — Typed error and Tier 1 contract
- **Objective:** Add `SegmentError` and the `Tier1SegmentOutput` / `Tier1SegmentTable` / `SegmentDefinition` / `CellMergeRecord` / `SegmentMetadata` models, with dual defense against proposed adjustments.
- **FR / criteria:** FR-7, FR-8, FR-10, AC-6, AC-9.
- **Scope:** Schemas and error type; no pipeline I/O.
- **Dependencies:** T1.
- **Affected files:** `src/aegis/utils/exceptions.py`, `src/aegis/schemas/segments.py`, `tests/unit/test_segments.py`.
- **Done when:** Valid 13-field payload validates; `proposed_adjustment`, `delta`, `proposed_premium`, `new_premium`, `price_adjustment` are rejected; inverted or non-covering intervals rejected; `policy_count < 1000` rejected; duplicate `segment_id` in a table rejected.
- **Validation:** `uv run pytest tests/unit/test_segments.py -k "tier1 or contract" -q`; `ruff`; `pyright`.

### T3 — Segment assignment and merge rule
- **Objective:** Implement base 36-cell assignment, canonical `segment_id`, and the configured merge rule (OD-1).
- **FR / criteria:** FR-1, FR-2, FR-3, FR-4, AC-1, AC-2.
- **Scope:** Assignment/merge; no aggregation.
- **Dependencies:** T2.
- **Affected files:** `src/aegis/pipelines/segments.py`, `tests/unit/test_segments.py`.
- **Done when:** All 413,169 policies map to a valid segment; empirical run yields 35 segments with the merged `D:18-24_C:5-9_Dens:>500` at 2,234 policies; every realized segment ≥1,000; unknown band label raises.
- **Validation:** `uv run pytest tests/unit/test_segments.py -k "assign or merge or segment_id" -q`.

### T4 — Segment loss-cost and base-premium aggregation
- **Objective:** Implement exposure-weighted loss-cost aggregation and loaded base premium, and assemble the 35-row table.
- **FR / criteria:** FR-5, FR-6, AC-4.
- **Scope:** Aggregation; no file writes.
- **Dependencies:** T3.
- **Affected files:** `src/aegis/pipelines/segments.py`, `tests/unit/test_segments.py`.
- **Done when:** Loss cost equals `Σ(μ·w)/Σw` on hand-calculated fixtures; base premium equals `L(1+loading)`; totals reconcile to portfolio exposure.
- **Validation:** `uv run pytest tests/unit/test_segments.py -k "loss_cost or premium" -q`.

### T5 — Ratio-of-sums elasticity aggregation
- **Objective:** Implement `aggregate_ratio_of_sums(tau, q0)` with input validation.
- **FR / criteria:** FR-9, AC-4.
- **Scope:** Pure function.
- **Dependencies:** T2.
- **Affected files:** `src/aegis/pipelines/segments.py`, `tests/unit/test_segments.py`.
- **Done when:** Matches `Στ/Σq0` on known vectors; zero/negative denominator, length mismatch, empty input, and non-finite values raise `SegmentError`.
- **Validation:** `uv run pytest tests/unit/test_segments.py -k "ratio_of_sums" -q`.

### T6 — Orchestration, artifacts, CLI, DVC
- **Objective:** Implement `run_segment_pipeline` (load/validate, assign, aggregate, write three artifacts), the `build-segments` CLI, and the `build_segments` DVC stage.
- **FR / criteria:** FR-11, FR-12, AC-5, AC-7.
- **Scope:** Persistence and orchestration; no new transforms.
- **Dependencies:** T3, T4, T5.
- **Affected files:** `src/aegis/pipelines/segments.py`, `src/aegis/pipelines/cli.py`, `dvc.yaml`, `tests/unit/test_segments.py`, `tests/unit/test_cli.py`.
- **Done when:** Three artifacts written and validate on reload; two consecutive runs are byte-identical; policy mapping row count is 413,169; CLI returns 0 on success and 1 with a typed message on failure.
- **Validation:** `uv run pytest tests/unit/test_segments.py tests/unit/test_cli.py -q`.

### T7 — Falsification and empirical integration
- **Objective:** Add the mandated falsification control and the empirical end-to-end test.
- **FR / criteria:** Falsification, AC-1…AC-9.
- **Scope:** Tests only.
- **Dependencies:** T6.
- **Affected files:** `tests/unit/test_segments.py`.
- **Done when:** `test_tier1_output_rejects_proposed_adjustment` passes; empirical test asserts 35 segments, floor met, totals reconcile, and SHA-256 determinism.
- **Validation:** `uv run pytest tests/unit/test_segments.py -q`.

### T8 — Full quality gate and Stage 4 evidence
- **Objective:** Run the full gate, record Stage 4 evidence, and update stage statuses.
- **FR / criteria:** Gate 4.
- **Scope:** Verification and documentation.
- **Dependencies:** T7.
- **Affected files:** `reports/docs/phases/phase_2/stage_evidence.md`, `execution_plan.md`, `tasks.md`, `dvc.lock`.
- **Done when:** `pytest`, `ruff`, `pyright`, module-size, and `dvc repro` pass; evidence recorded; Gate 4 closed.
- **Validation:** Commands in `AGENTS.md` §6.

---

## Traceability

| FR | Tasks |
|---|---|
| FR-1, FR-2, FR-3, FR-4 | T3 |
| FR-5, FR-6 | T4 |
| FR-7, FR-8, FR-10 | T2 |
| FR-9 | T5 |
| FR-11, FR-12 | T6 |
| Falsification | T7 |
| Gate 4 | T8 |

## Completion

All tasks completed and verified on 2026-10-21. Gate 4 passed (35 realized segments, minimum 1,067 policies, merged cell 2,234, portfolio loss cost €126.53/yr and base premium €151.84/yr, adjustment fields rejected); executed evidence is recorded in `stage_evidence.md` §Stage 4.
