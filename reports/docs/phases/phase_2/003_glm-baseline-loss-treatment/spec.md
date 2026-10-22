# Specification: P2-F03 GLM Loss-Cost Baseline & Calibration Reporting
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F03` (`003_glm-baseline-loss-treatment`)
**Governing Decisions:** ADR-012 (Tweedie baseline, compound Poisson-Gamma, calibration reporting), P2-D1 (target/severity scope, power justification, MAE/RMSE secondary), P2-D1a (capping policy; level via Q1), ADR-014 / P2-D3 (segment bands), ADR-015 (tiered reproducibility), ADR-005 (config-only tunables)
**SDD Depth:** Full (`spec.md` → approval → `plan.md` → `tasks.md`)
**Status:** APPROVED — October 21, 2026 (OD-1…OD-5 as recommended)
**Date:** October 21, 2026

---

## 1. Context and Problem

Phase 6 requires a per-segment baseline loss cost (`Losses₀`) and a premium base consistent with it. The portfolio has 96.28% zero-claim policies and 16,181 claims with a heavy severity tail (largest claim €2,036,833). A Poisson frequency model cannot yield per-segment loss cost without a non-modeled severity constant; a frequency × severity pair doubles the calibration surface for a baseline whose roles are "reference" and "Losses₀ source".

P2-D1 resolves this by mandating a single Tweedie GLM (compound Poisson-Gamma) on capped pure premium as the sanctioned combined frequency-severity model — explicitly *not* a separate severity model. ADR-012 fixes the calibration reporting as holdout mean Tweedie deviance plus actual-versus-expected (A/E) by predicted-loss decile and by segment, with MAE/RMSE strictly secondary. The loss cap level is open until Q1 (Stage 5); Stage 3 must therefore build a **cap-parameterized** baseline and a calibration suite that can be re-run at each candidate cap.

The Stage 2 feature pipeline already produced the categorical/banded GLM view (14 columns, 413,169 rows) and the approved band edges and factor levels. Stage 3 consumes that view and must not re-derive bands, clips, or encodings (FR-12 of P2-F02).

## 2. Objective and Expected Outcome

Fit one deterministic, cap-parameterized Tweedie GLM on the capped pure premium of the 80% training split, and emit a standardized calibration report on the 20% held-out split. The outputs are the fitted per-policy loss cost (for Stage 4 segment aggregation and Stage 5 cap sweeping), the model metadata (coefficients, estimated variance power, convergence status), and the calibration report (holdout deviance, decile A/E, segment-band A/E, secondary MAE/RMSE).

## 3. Scope

### In scope
- A single Tweedie GLM (variance power `p ∈ (1, 2)`, log link) with exposure weights.
- Target and weight construction from the Stage 2 GLM view.
- Cap parameterization (default cap in config; overridable for the Stage 5 sweep).
- Tweedie variance-power selection via deterministic profile deviance on the training split (with a documented fallback).
- Seeded 80/20 train/test split (`tier1_ml.random_state: 42`).
- Calibration evaluation: holdout mean Tweedie deviance, A/E by predicted-loss decile, A/E by segment band, secondary MAE/RMSE.
- Versioned, typed artifacts: model metadata JSON, per-policy fitted loss-cost CSV, calibration report JSON.

### Out of scope
- Excess-load calculation and final cap selection (Stage 5, Q1).
- Segment definition/merge rule and the Tier 1 segment output contract (Stage 4, P2-F05).
- Any frequency model or severity model other than the single Tweedie (FR-12).
- Causal elasticity estimation, bandit exploration, or premium adjustment.
- MLflow registration / `candidate`/`approved` promotion (Stage 8 / Stage 14).
- Drift monitoring, serving, or production scoring.
- Changes to the Stage 2 feature pipeline or the Stage 1 raw contract.

## 4. Functional Requirements (EARS)

**FR-1 (target).** When constructing the modeling target for a policy, the pipeline shall compute `target = min(ClaimAmount, cap) / exposure_capped`, using the Stage 2 capped exposure strictly positive, and shall raise a typed error if any `exposure_capped` is non-positive.

**FR-2 (weights).** The pipeline shall pass `exposure_capped` as the model's frequency weights, so each policy contributes in proportion to its policy-years.

**FR-3 (family and link).** The pipeline shall fit a `statsmodels` Tweedie GLM with variance power `p ∈ (1, 2)` and log link; no other family or link shall be used.

**FR-4 (cap parameterization).** The pipeline shall accept the loss cap as a runtime parameter (default from config `tier1_ml.glm_baseline.default_cap`), applying it only in target construction, so Stage 5 can re-run the same code at each candidate level without modification.

**FR-5 (design matrix).** The GLM design matrix shall consist of the seven rating factors — Power, Brand, Gas, Region (categorical) and DriverAge band, CarAge band, Density band (categorical) — reference-coded (drop-first dummy), with a fixed column order derived from the frozen config level lists and band labels. No composite index, interaction, polynomial, or claim-history term shall enter the design matrix.

**FR-6 (variance power).** The pipeline shall select `p` by a deterministic profile-deviance grid over the training split, restricting to converged fits; if the profile is degenerate or no grid point converges, it shall fall back to a documented fixed value with stated actuarial justification. The full profile table and the selected `p` shall be recorded in the model metadata.

**FR-7 (split).** The pipeline shall split the 413,169 policies into 80% train / 20% test using a fixed seed (`tier1_ml.random_state: 42`) over a deterministic policy ordering, and shall compute all calibration metrics on the test split only.

**FR-8 (holdout deviance).** The pipeline shall compute and record the out-of-sample mean Tweedie deviance on the test split using the selected variance power.

**FR-9 (decile A/E).** The pipeline shall rank test policies by predicted pure premium, form deciles, and report the ratio of sums `A/E = Σ actual / Σ expected` per decile (actual = capped loss, expected = fitted mean × exposure).

**FR-10 (segment-band A/E).** The pipeline shall report the ratio of sums `A/E` for each DriverAge band, CarAge band, Density band, and the 36-cell cross-product on the test split.

**FR-11 (secondary diagnostics).** The pipeline shall compute and record MAE and RMSE on the test split as secondary diagnostics; no primary calibration gate may be derived from them (P2-D1).

**FR-12 (single model).** The pipeline shall create exactly one model; no separate frequency or severity model shall exist. A structural test shall assert the absence of any frequency- or severity-only estimator in the module.

**FR-13 (convergence).** The pipeline shall verify the final fit converged; any non-convergence or numerical error shall raise a typed error and produce no report, rather than emit an unverified baseline.

**FR-14 (artifacts).** The pipeline shall persist, deterministically and with no pickle: (a) model metadata JSON (coefficients, `p`, dispersion, convergence flag, cap, split info, profile table), (b) a per-policy fitted loss-cost CSV, and (c) the calibration report JSON. Artifacts shall validate against typed schemas before writing.

## 5. Data Contracts

### 5.1 Input
`data/processed/features_glm.csv` (Stage 2 GLM view): `PolicyID`, `exposure_capped`, `ClaimAmount`, `DriverAge`, `CarAge`, `Density`, `log_density`, `driver_age_band`, `car_age_band`, `density_band`, `Power`, `Brand`, `Gas`, `Region`. 413,169 rows; sorted by `PolicyID`.

### 5.2 Outputs
- `glm_baseline_model.json` — `GLMBaselineMetadata`: cap, selected `p`, fallback flag, profile table, coefficients (name → value), dispersion, converged, split row counts, `random_state`, config fingerprint.
- `glm_fitted_loss_cost.csv` — per policy: `PolicyID`, `exposure_capped`, `capped_actual_loss`, `fitted_loss_cost_per_year`, `split` (`train`/`test`), `decile` (test rows only).
- `glm_calibration_report.json` — `GLMCalibrationReport`: holdout mean deviance, decile A/E rows, segment-band A/E rows, secondary MAE/RMSE, and the values used (cap, `p`, split counts).

Exact paths are fixed in `plan.md`; the logical contracts above are binding. All artifacts are DVC-versioned.

## 6. Edge Cases

| Case | Required behavior |
|---|---|
| `exposure_capped <= 0` | Typed error; halt (FR-1). |
| No claims above `cap` in some category | Model still fits; fitted values remain positive via log link; no special-casing. |
| Profile-deviance grid point fails to converge | Skip the point; record it as non-converged in the profile table. |
| All grid points fail to converge | Fall back to the fixed `p`; if that final fit also fails, raise typed error (FR-13). |
| Profile is monotonic (no interior minimum) | Fall back to the documented fixed `p`; record `fallback=true`. |
| Repeated run | Byte-identical artifacts (deterministic ordering, fixed seed, no wall-clock fields). |
| Cap below the minimum claim or above the maximum claim | Allowed; target degenerates to raw values in the limit, still valid. |

## 7. Error Handling

A single typed `GLMBaselineError` (subclass of `AEGISError`) shall carry `rule`, `column` (where applicable), `count`, and sample keys. Conditions: non-positive exposure, missing/extra columns, non-convergence, numerical failure, degenerate profile without fallback, or artifact schema validation failure. Errors propagate; the CLI and DVC stage exit non-zero. No bare `except` may convert a failure into a successful output.

## 8. Non-Functional Constraints

- **INV-8:** every source module stays below 1,000 lines; the fit, calibration, and schema logic are separated into `glm_baseline.py`, `glm_calibration.py`, and `schemas/glm.py`.
- **ADR-005:** cap, variance-power grid, fallback value, decile count, and output paths come from `params.yaml`; no hardcoded tunables.
- **ADR-015 / determinism:** fixed seed and deterministic ordering; no wall-clock fields; artifacts reproducible under the locked lockfile.
- **INV-3:** the pipeline reads the versioned Stage 2 view and is itself a DVC stage after `build_features`.
- **INV-5:** the baseline performs deterministic arithmetic via `statsmodels`; no LLM is involved in fitting, deviance, or A/E computation.
- **Performance:** one fit plus a small profile grid over ~330k rows and ~35 columns completes well within the 30-minute budget (expected order of seconds per fit).

## 9. Acceptance Criteria

- **AC-1:** The Tweedie GLM converges with no numerical warnings for the selected variance power.
- **AC-2:** Holdout mean Tweedie deviance and the decile A/E table are generated and persisted.
- **AC-3:** The selected `p` is documented with either the profile-deviance table or the stated fallback justification.
- **AC-4:** MAE and RMSE are present in the report and are referenced by no primary pass/fail gate (asserted in tests).
- **AC-5:** Re-running the pipeline produces identical artifact hashes.
- **AC-6:** A structural test proves no frequency-only or severity-only model exists (FR-12).
- **AC-7:** A/E rows use ratio-of-sums semantics (sum actual / sum expected), not mean of ratios.
- **AC-8:** The cap is a parameter: running with two different caps changes the target and report but shares one code path.
- **AC-9:** Artifacts reload and validate against typed schemas; coefficients, `p`, dispersion, convergence, cap, and split counts are recorded.
- **AC-10:** `pytest`, `ruff`, `pyright`, and the module-size check pass; each rule above has unit coverage.

## 10. Approved Decisions (approved October 21, 2026)

All five decisions were approved by the operator on October 21, 2026, as recommended.

**OD-1 — Variance power `p` (FR-6).** **Recommend: estimate by profile deviance on the training split.** A coarse grid `[1.05, 1.15, …, 1.95]` (step 0.05) is fitted; the point minimizing summed unit deviance among converged fits is refined locally (step 0.01). The selected `p` is recorded and **fixed for Stage 5's cap sweep** so the sweep isolates the cap effect. Fallback: `p = 1.5` — the midpoint of the compound Poisson-Gamma family, a conventional auto-insurance starting point — applied only if the profile is degenerate or no point converges, and always flagged in metadata. *(Alternative: always fix `p = 1.5` — simpler but leaves the baseline's overdispersion unverified, which P2-D1 cautions against.)*

**OD-2 — GLM covariates (FR-5).** **Recommend: all seven rating factors as categorical/banded factors with drop-first reference coding.** Power (12), Brand (7), Gas (2), Region (10), DriverAge band (4), CarAge band (3), Density band (3) → ~34 dummy columns plus intercept. This aligns the baseline directly with the ADR-014 segment grid so Stage 4 aggregation is a sum of fitted values. *(Alternative: continuous ages/`log_density` for finer fit — better discrimination but decouples the model from the segment grid and adds an interpretation burden.)*

**OD-3 — Default cap for Stage 3 (FR-4).** **Recommend: €100,000.** It is a Q1 candidate, a conventional round level, and near the EDA's cited 14.6% loss-removal point; it exists only to exercise the parameterized pipeline. The final cap is selected in Stage 5 (Q1) and does not depend on this default.

**OD-4 — Calibration metric definitions (FR-8/FR-9/FR-10).** **Recommend: ratio-of-sums A/E** (Σ actual / Σ expected) per decile/band, and holdout mean deviance as the exposure-weighted summed unit deviance divided by the number of holdout policies. Deciles are formed by ranking test policies on predicted pure premium (not expected total loss), so exposure does not distort the ranking.

**OD-5 — Artifact format (FR-14).** **Recommend: JSON + CSV only, no pickle**, consistent with Stage 2 OD-2. The fitted model is represented by its coefficients and metadata (reconstructable for prediction); the fitted estimator object is not serialized.

## 11. Definition of Done

1. This spec is approved by the operator.
2. `plan.md` and `tasks.md` are written and consistent with this spec.
3. All acceptance criteria pass with recorded evidence.
4. Gate 3 evidence is recorded in `stage_evidence.md`.
5. `pytest`, `ruff`, `pyright`, and the module-size check are green; `dvc repro` runs the new stage cleanly.

## 12. Traceability

| Requirement | Source |
|---|---|
| FR-1, FR-2, FR-3 | ADR-012, P2-D1 |
| FR-4 | P2-D1a, ADR-012 (cap via Q1) |
| FR-5 | ADR-014 / P2-D3 (Region/Power/Brand/Gas GLM-only; segment bands) |
| FR-6 | P2-D1 (power justified, not a default), ADR-012 |
| FR-7 | ADR-013 (held-out evaluation), ADR-015 |
| FR-8, FR-9, FR-10, FR-11 | ADR-012 (deviance + A/E; MAE/RMSE secondary), P2-D1 |
| FR-12 | P2-D1 (no separate severity model) |
| FR-13, FR-14 | INV-3, ADR-015, ADR-005 |
