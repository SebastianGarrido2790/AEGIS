# Specification: P2-F02 Deterministic Feature Pipeline
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F02` (`002_feature-pipeline`)  
**Governing Decisions:** ADR-012 (baseline target, loss treatment, feature preparation), P2-D4, P2-D3/ADR-014 (segment bands), A4 (claims variables excluded from elasticity covariates)  
**SDD Depth:** Full (`spec.md` → approval → `plan.md` → `tasks.md`)  
**Status:** APPROVED — October 21, 2026 (OD-1: DriverAge `[18, 90]`, CarAge `[0, 34]`; OD-2: one-hot unordered factors)  
**Date:** October 21, 2026

---

## 1. Context and Problem

Stages 3 and 6 consume two different model families over one fixed set of pre-treatment rating factors. The GLM needs interpretable categorical/banded factors; the causal forest needs a numeric/ordinal design matrix. Attempt 1's engineered composite index reached kurtosis 222 and concentrated 19% of its variance in the top 0.1% of rows. The pipeline must therefore produce two views from one deterministic cleaning step, with no composite index, no claims history in the elasticity covariates, and every clip or exclusion counted and logged.

The raw policy table passed Stage 1 (20/20 expectations; 413,169 rows) and is DVC-versioned. This feature turns that table into model-ready views.

## 2. Objective and Expected Outcome

Produce two reproducible, model-ready views and their versioned transformation metadata:

- **GLM view:** categorical/banded factors plus exposure and loss components for the Tweedie baseline.
- **Forest view:** numeric/ordinal design matrix of pre-treatment covariates only, for the causal estimator.

Re-running the same transform on the same input must produce bit-identical outputs. No later stage may recompute feature logic independently.

## 3. Scope

### In scope
- Exposure handling: cap at 1.0 with exact affected-record count logged.
- Density transform: natural log of Density.
- DriverAge and CarAge derived bands aligned to the approved segment grid.
- Factor representations for both views.
- Tail-bound clipping for DriverAge and CarAge at the approved bounds (see §10).
- Covariate hygiene: claims variables excluded from the elasticity covariate set.
- Determinism guarantees.
- Persistence and versioning of views, band edges, clip/exclusion log, feature lists, and a baseline feature-distribution summary.

### Out of scope
- Model fitting (GLM, causal forest) or any prediction.
- Target construction beyond passing through the fields needed by Stage 3 (the loss cap is applied in Stage 3 and swept in Stage 5).
- Imputation, outlier removal by model residual, feature selection, target encoding, or any learned transformation.
- Composite risk indices, interaction features, or polynomial expansions.
- Drift monitoring, production scoring, or serving logic.
- Changes to the raw GX contract or Stage 1 join semantics.

## 4. Functional Requirements (EARS)

**FR-1 (exposure).** When a policy record has `Exposure > 1.0`, the pipeline shall set its model exposure to `1.0`, preserve the raw value only in the retyped column, and increment a logged clip count.

**FR-2 (exposure validity).** If `Exposure` is null or `<= 0`, the pipeline shall raise a typed feature-pipeline error and produce no output.

**FR-3 (density).** The pipeline shall represent Density as `ln(Density)` in both views; since the validated raw contract guarantees `Density >= 2`, no offset or epsilon shall be added.

**FR-4 (bands).** The pipeline shall derive DriverAge and CarAge bands from frozen configuration edges and shall not recompute edges from the input data at run time.

**FR-5 (GLM view).** The GLM view shall contain: the capped exposure, `log_density`, DriverAge band, CarAge band, and categorical encodings of Power, Brand, Gas, and Region. It shall also carry `PolicyID`, `ClaimAmount`, and the raw DriverAge/CarAge values for traceability.

**FR-6 (forest view).** The forest view shall contain one numeric/ordinal row per policy with `PolicyID` and only pre-treatment covariates. It shall contain no claim-count, claim-amount, pure-premium, exposure-as-covariate, or other post-treatment field.

**FR-7 (ordered vs unordered factors).** Power and the derived age bands shall be encoded ordinally in the forest view. Unordered factors Brand (7 levels), Gas (2 levels), and Region (10 levels) shall be one-hot encoded in the forest view and remain categorical in the GLM view; they shall not receive an arbitrary single ordinal code.

**FR-8 (tail bounds).** When DriverAge falls outside `[18, 90]` or CarAge outside `[0, 34]`, the pipeline shall clip the value to the bound, log the count by column and side, and record the bounds in the transformation metadata.

**FR-9 (no composite index).** The pipeline shall emit no column derived as a weighted combination, ratio, or aggregate of multiple rating factors.

**FR-10 (determinism).** Two runs over the same input and configuration shall produce byte-identical view files and identical metadata, excluding any wall-clock field.

**FR-11 (metadata).** The pipeline shall persist, as a versioned artifact: band edges, clip/exclusion counts, input and output row counts, view column lists, the ordered factor level mappings, a SHA-256 of each view, and a baseline feature-distribution summary.

**FR-12 (single source of truth).** Stage 3 and Stage 6 shall consume the persisted views; they shall not re-derive bands, clips, or encodings.

**FR-13 (fail loud).** If the input contains an unexpected categorical level, a null in a required field, a duplicate `PolicyID`, or a missing expected column, the pipeline shall raise a typed error identifying the column and offending values; it shall not coerce, drop, or impute silently.

## 5. Data Contracts

### 5.1 Input
`data/versioned/raw_policy_data.csv`, as produced and validated in Stage 1, with 413,169 rows and the 13 columns defined by `raw_policy_suite.json`.

### 5.2 Outputs
- `glm_view`: one row per policy; includes `PolicyID`, `exposure_capped`, `log_density`, `driver_age_band`, `car_age_band`, categorical factors, `ClaimAmount`, and raw age fields.
- `forest_view`: one row per policy; includes `PolicyID` plus only pre-treatment numeric/ordinal covariates.
- `feature_metadata`: JSON artifact covering FR-11.

Views are persisted as CSV and metadata as JSON, consistent with the existing pipeline and requiring no new dependency; exact paths are fixed in `plan.md`. The logical contracts above are binding.

### 5.3 Covariate set
Pre-treatment covariates: Power, Brand, Gas, Region, DriverAge, CarAge, Density. Explicitly excluded from the forest view: `ClaimNb`, `ClaimAmount`, `sev_claim_count`, `is_orphan_severity`, and any post-treatment field.

## 6. Edge Cases

| Case | Required behavior |
|---|---|
| `Exposure > 1.0` (421 records) | Clip to 1.0; log exact count; retain raw value where the view allows. |
| `Exposure == 1.0` | Unchanged; not counted as a clip. |
| `Exposure <= 0` or null | Typed error; halt. |
| `Density` null or `<= 0` | Typed error; halt (log undefined). |
| DriverAge/CarAge outside approved bounds | Clip and log count by column/side. |
| Duplicate `PolicyID` | Typed error; halt. |
| Unknown categorical level | Typed error listing the level; no silent unknown bucket. |
| Missing required column | Typed error; halt. |
| Empty input table | Typed error; no empty artifacts. |
| Repeated run | Byte-identical outputs. |

## 7. Error Handling

A single typed `FeaturePipelineError` (subclass of the project's existing base exception pattern) shall carry: stage, column, rule, offending count, and up to ten offending keys. Errors propagate; the DVC stage exits non-zero. No bare `except` may convert an error into a successful output.

## 8. Non-Functional Constraints

- **INV-8:** every source module stays below 1,000 lines.
- **ADR-005:** all thresholds, band edges, and file paths come from `params.yaml`; no hardcoded values beyond type-level constants.
- **INV-3:** views are produced only from validated, versioned input and are themselves versioned by DVC.
- **Auditability:** every clipping, exclusion, and encoding decision is reconstructable from persisted metadata.
- **Performance:** one pass over 413,169 rows; no quadratic operations; full transform completes comfortably within local interactive use.

## 9. Acceptance Criteria

- **AC-1:** Two consecutive runs produce identical SHA-256 hashes for both views and identical metadata (excluding timestamps).
- **AC-2:** Exactly 421 exposure clips are logged on the empirical input; maximum `exposure_capped` equals 1.0.
- **AC-3:** A schema test proves no claims variable appears in the forest-view column list.
- **AC-4:** A whitelist test proves no composite-index column exists in either view.
- **AC-5:** Band edges are read from configuration; a config change alters bands without code changes, and metadata records the edges used.
- **AC-6:** `log_density` equals `ln(Density)` within floating-point tolerance for all rows.
- **AC-7:** Tail-clip counts per column and side match a direct recomputation on the empirical input.
- **AC-8:** Metadata reloads and validates against a typed schema; it records row counts, column lists, level mappings, clip counts, band edges, and view hashes.
- **AC-9:** Every malformed case in §6 raises `FeaturePipelineError` with the offending column and count.
- **AC-10:** `pytest`, `ruff`, `pyright`, and the module-size check pass; the new feature modules have unit tests for each rule above.
- **AC-11:** The Stage 2 falsification statement holds: determinism is demonstrated by hash equality, and leakage prevention by a schema assertion test.

## 10. Approved Decisions (resolved October 21, 2026)

Both decisions were approved by the operator on October 21, 2026.

**OD-1 — Tail bounds: APPROVED as DriverAge `[18, 90]`, CarAge `[0, 34]`.**  
Stage 0: DriverAge 18–99 (P99 = 81, P99.9 = 90); CarAge 0–100 (P99 = 22, P99.9 = 34). The P99.9 bounds remove roughly 0.1% extreme leverage per factor while retaining actuarially plausible values. Counts are logged by column and side.

**OD-2 — Forest encoding: APPROVED as one-hot for unordered factors.**  
Brand (7), Gas (2), and Region (10) are one-hot encoded; Power (d→o) and the age bands are ordinal. This avoids imposing false order and adds 19 columns, well within the estimator's capacity. Missing levels fail loudly rather than producing an unseen-level bucket.

## 11. Definition of Done

1. This spec is approved by the operator.
2. `plan.md` and `tasks.md` are written and consistent with this spec.
3. All acceptance criteria pass with recorded evidence.
4. Outputs and metadata are DVC-versioned and readable by Stage 3/Stage 6 without re-derivation.
5. Gate 2 evidence is recorded in `stage_evidence.md`.
6. `pytest`, `ruff`, `pyright`, and module-size checks are green.

## 12. Traceability

| Requirement | Source |
|---|---|
| FR-1, FR-2 | ADR-012, P2-D4, Stage 0 exposure finding |
| FR-3, FR-8 | ADR-012, P2-D4, Stage 0 distributions |
| FR-4 | ADR-014 / P2-D3 band grid |
| FR-5, FR-6, FR-7 | ADR-012, P2-D4, A4 |
| FR-9 | ADR-012, attempt-1 composite-index failure |
| FR-10, FR-11 | ADR-015 tiered reproducibility; INV-3 |
| FR-12, FR-13 | INV-3, zero-silent-fallback practice |
