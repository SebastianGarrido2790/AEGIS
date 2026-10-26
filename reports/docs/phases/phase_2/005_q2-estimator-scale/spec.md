# Specification: Q2 Estimator Training Scale & Budget
## Phase 2 — Stage 7

**Feature ID:** `P2-F06` (Q2 qualification portion; estimator code lives in Stage 8)  
**Governing Decisions:** P2-D5 (approved method + budget criteria), ADR-013  
**Status:** APPROVED — October 26, 2026 (OD-1a ratio-of-sums, OD-2a single seed)  
**Date:** October 26, 2026  

---

## 1. Context

P2-D5 approved running `CausalForestDML` across four sample sizes and two nuisance-model families to determine whether the full 413,169-policy dataset fits within a 30-minute / 80%-RAM budget, or whether a subsample is needed. Stage 0 resolved that Q2 sample sizes represent **total slices with the 80/20 held-out split applied inside each** (V8: 12 cores, 47.93 GB RAM; 80% ceiling = 38.34 GB).

This stage produces a **qualification report** and **frozen configuration** — no production estimator code ships here (that is Stage 8).

## 2. Objective

Execute the Q2 grid, select the training scale and nuisance-model family by approved budget criteria, freeze the result in `params.yaml`, and record the evidence.

## 3. Scope

**In scope:**
- A self-contained Q2 experiment script under `scripts/q2_estimator_scale.py`.
- Q2 grid: 4 sample sizes × 2 nuisance families × `cv_folds=5` cross-fitting.
- Metrics per cell: wall-clock time, peak RSS, segment Spearman ρ vs truth, 95% CI coverage.
- Selection logic: if full-data fit ≤ 30 min and ≤ 80% RAM → select full data; else smallest n with Spearman within 0.02 of largest n.
- Report at `reports/docs/phases/phase_2/q2_estimator_scale_report.md`.
- Frozen params in `params.yaml` under a new `tier1_ml.estimator` block (selected n*, nuisance family, cv_folds).
- Stage evidence appended to `stage_evidence.md`.

**Out of scope:**
- Production estimator module (Stage 8).
- Validation harness / multi-seed recovery (Stage 9–10).
- DVC stage for Q2 (one-off experiment, not in the reproducible pipeline).

## 4. Functional Requirements

| ID | Requirement |
|---|---|
| FR-1 | Load Stage 6 training table and truth sidecar; load Stage 4 segment definitions. |
| FR-2 | For each $N \in \{25\text{k}, 100\text{k}, 250\text{k}, 413{,}169\}$, apply stratified 80/20 split (stratified by `segment_id`, seed = `tier1_ml.random_state`). |
| FR-3 | For each split, train `CausalForestDML` with each nuisance family: (a) GBT — `GradientBoostingClassifier` / `GradientBoostingRegressor`; (b) Linear — `LogisticRegressionCV` / `ElasticNetCV`. |
| FR-4 | Record per-cell: wall-clock seconds, peak RSS (MB), out-of-sample segment Spearman ρ against ground truth τ, and fraction of segments whose 95% CI covers the true segment-average τ. |
| FR-5 | Apply approved selection rule: full-data fit passes budget → adopt full data; else adopt smallest n* with Spearman within 0.02 of the largest n tested. Between nuisance families, prefer the one with higher Spearman at n*; if tied within 0.02, prefer GBT for consistency. |
| FR-6 | Write structured report with a table of all grid cells (zero skipped), selection rationale, and selected configuration. |
| FR-7 | Add `tier1_ml.estimator` block to `params.yaml` + Pydantic schema: `training_sample_size`, `nuisance_model` (enum: `"gbt"` / `"linear"`), `cv_folds`, and `runtime_budget_minutes`. |

## 5. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NF-1 | Q2 script is a standalone runnable (`uv run python scripts/q2_estimator_scale.py`), not a pipeline stage. |
| NF-2 | Peak RSS measured via `tracemalloc` (cross-platform, no psutil dependency). |
| NF-3 | Script logs progress to stdout; no MLflow tracking for Q2. |

## 6. Open Decisions

### OD-1 — Segment Spearman computation method

**Options:**
- **(a) Recommended: Ratio-of-sums.** Aggregate policy-level CATE predictions to segment averages using the ratio-of-sums method already implemented in `segments.py`, then compute Spearman between predicted and true segment τ.
- (b) Simple mean aggregation within segments.

**Recommendation:** (a) — consistent with how the production estimator will aggregate; already tested in Stage 4.

### OD-2 — Pilot seed count

**Options:**
- **(a) Recommended: Single seed** (the configured `random_state = 42`). Q2 measures scale feasibility, not seed stability (that's Stages 9–10 / D6).
- (b) 3 seeds with median metric.

**Recommendation:** (a) — keeps Q2 runtime bounded; multi-seed is D6's scope.

## 7. Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-1 | All 8 grid cells (4 sizes × 2 families) run to completion with no skipped cells. |
| AC-2 | Report contains a table with all 4 metrics per cell. |
| AC-3 | Selected n* and nuisance family satisfy the P2-D5 budget criteria. |
| AC-4 | `params.yaml` updated with `tier1_ml.estimator` block; config loads without error. |
| AC-5 | `stage_evidence.md` Stage 7 section records the full grid, selection rationale, and frozen parameters. |
