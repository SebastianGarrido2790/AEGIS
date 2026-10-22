# Technical Plan: P2-F03 GLM Loss-Cost Baseline & Calibration Reporting
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F03` (`003_glm-baseline-loss-treatment`)
**Spec:** `reports/docs/phases/phase_2/003_glm-baseline-loss-treatment/spec.md` — APPROVED October 21, 2026 (OD-1…OD-5)
**Status:** Ready for task decomposition
**Date:** October 21, 2026

---

## 1. Architecture and Components

The GLM baseline sits between the Stage 2 feature views and the Stage 4 segment aggregation / Stage 5 cap sweep.

```text
data/processed/features_glm.csv   (Stage 2, deterministic)
                │
                ▼
        glm_baseline (fit) ──► glm_calibration (evaluate)
        ├── design matrix (7 factors, drop-first)
        ├── target = min(ClaimAmount, cap) / exposure_capped
        ├── Tweedie GLM (p via profile deviance, log link)
        ├── seeded 80/20 split
        └── outputs
             ├── data/processed/glm_baseline_model.json
             ├── data/processed/glm_fitted_loss_cost.csv
             └── data/processed/glm_calibration_report.json
```

### Components

| Component | Responsibility | Location |
|---|---|---|
| Config schema | Cap default, variance-power grid, fallback, decile count, output paths | `params.yaml`, `src/aegis/config/schema.py` |
| GLM errors | Typed fit/calibration failure | `src/aegis/utils/exceptions.py` |
| Metadata contracts | Pydantic models for model metadata and calibration report | `src/aegis/schemas/glm.py` |
| Fit core | Design matrix, target/weights, profile deviance, fit, predictions | `src/aegis/pipelines/glm_baseline.py` |
| Calibration | Holdout deviance, decile/segment A/E, MAE/RMSE | `src/aegis/pipelines/glm_calibration.py` |
| CLI entrypoint | `build-glm` subcommand | `src/aegis/pipelines/cli.py` |
| DVC stage | `glm_baseline` after `build_features` | `dvc.yaml` |
| Tests | Unit/contract/determinism tests | `tests/unit/test_glm_baseline.py`, `tests/unit/test_glm_calibration.py` |

`schemas/glm.py` is a separate typed module because the metadata and report are persisted, versioned interfaces consumed by Stage 4/5.

**Covers:** FR-14, AC-9.

## 2. Data and Control Flow

1. Read `features_glm.csv` (Stage 2 GLM view) with `PolicyID` retained.
2. Validate input columns and positivity (`exposure_capped > 0`, `ClaimAmount >= 0`).
3. Build the design matrix: dummy-code Power/Brand/Gas/Region and the three band factors (drop-first), in a fixed column order derived from config level lists and band labels.
4. Split indices 80/20 with `random_state=42` over the sorted `PolicyID` order; record train/test row counts.
5. Build `target = min(ClaimAmount, cap) / exposure_capped` (cap from config/parameter).
6. Select `p`: fit a Tweedie GLM on the training split at each grid point `[1.05, …, 1.95]` (step 0.05), record `converged` and `deviance`; refine locally (step 0.01) around the minimizer; fall back to `p_fallback` if degenerate or no point converges.
7. Fit the final model at the selected `p`; require convergence.
8. Compute per-policy fitted pure premium (`mu`) on all rows; write the fitted loss-cost CSV.
9. Compute calibration on the test split: holdout mean deviance, decile A/E, segment-band A/E, MAE/RMSE.
10. Assemble metadata and report; validate against Pydantic; write JSON artifacts.

No randomness beyond the fixed seed; no wall-clock fields enter the artifacts.

**Covers:** FR-1–FR-8, FR-12, FR-13, FR-14, AC-1, AC-2, AC-5, AC-8, AC-9.

## 3. Interfaces and Contracts

### 3.1 Configuration additions

```yaml
tier1_ml:
  glm_baseline:
    default_cap: 100000.0
    tweedie_p_min: 1.05
    tweedie_p_max: 1.95
    tweedie_p_step: 0.05
    tweedie_p_refine_step: 0.01
    tweedie_p_fallback: 1.5
    decile_count: 10
    link: "log"
  glm_paths:
    fitted_loss_cost: "data/processed/glm_fitted_loss_cost.csv"
    model_metadata: "data/processed/glm_baseline_model.json"
    calibration_report: "data/processed/glm_calibration_report.json"
```

The Pydantic schema validates `1 < p_min < p_max < 2`, positive steps, `1 < fallback < 2`, `decile_count > 1`, and `link == "log"`.

### 3.2 Output contracts

**Model metadata (`GLMBaselineMetadata`):** `cap`, `tweedie_p`, `fallback`, `profile` (list of `{p, deviance, converged}`), `coefficients` (`dict[str, float]`), `dispersion`, `converged`, `train_rows`, `test_rows`, `random_state`, `config_fingerprint`.

**Fitted loss-cost CSV (fixed order):** `PolicyID`, `exposure_capped`, `capped_actual_loss`, `fitted_loss_cost_per_year`, `split`, `decile`. `decile` is 1–10 for test rows and empty for train rows.

**Calibration report (`GLMCalibrationReport`):** `cap`, `tweedie_p`, `holdout_mean_deviance`, `decile_ae` (list of `AERow`), `segment_ae` (list of `AERow`), `secondary` (`{mae, rmse}`), `train_rows`, `test_rows`, `random_state`.

`AERow` = `{group, actual, expected, ae_ratio, count, exposure}`.

### 3.3 Function signatures

```python
# glm_baseline.py
def build_design_matrix(view: pd.DataFrame, config: Tier1MLConfig) -> pd.DataFrame: ...
def build_target(view: pd.DataFrame, cap: float) -> pd.Series: ...
def split_indices(n: int, random_state: int, test_size: float) -> tuple[pd.Index, pd.Index]: ...
def select_tweedie_p(X, y, w, cfg: GLMBaselineConfig) -> tuple[float, list[ProfilePoint], bool]: ...
def fit_glm(X, y, w, p: float) -> GLMResults: ...
def run_glm_baseline(
    glm_path: Path | str | None = None,
    cap: float | None = None,
    fitted_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> GLMCalibrationReport: ...
```

`run_glm_baseline` orchestrates fit + calibration + artifact writing and returns the report.

```python
# glm_calibration.py
def mean_tweedie_deviance(family, y, mu, w) -> float: ...
def decile_ae(y, mu, w, actual, decile_count: int) -> list[AERow]: ...
def segment_ae(view, y, mu, w, actual) -> list[AERow]: ...
def secondary_diagnostics(actual, expected) -> SecondaryDiagnostics: ...
def build_calibration_report(...) -> GLMCalibrationReport: ...
```

Public functions and returns are typed; no raw `dict` crosses the module boundary.

**Covers:** FR-8–FR-11, FR-14, AC-2, AC-4, AC-7, AC-9.

## 4. Algorithms and Rules

### 4.1 Design matrix

For each categorical factor, build drop-first dummy columns with a fixed order from config:

- `Power` (12 levels) → 11 columns
- `Brand` (7) → 6
- `Gas` (2) → 1
- `Region` (10) → 9
- `driver_age_band` (4) → 3
- `car_age_band` (3) → 2
- `density_band` (3) → 2

Total 34 dummy columns plus intercept. Column names use `pd.get_dummies(drop_first=True)` over `pd.Categorical` with frozen category order, so the schema cannot drift between runs. `log_density` is deliberately excluded (redundant with `density_band`); raw ages are excluded in favor of the banded factors (OD-2).

### 4.2 Target and weights

```text
capped_loss = min(ClaimAmount, cap)
target      = capped_loss / exposure_capped      # pure premium per policy-year
weights     = exposure_capped                     # frequency weights
```

### 4.3 Profile deviance for `p`

```text
grid = [round(p_min + i*step, 6) for i in 0..round((p_max-p_min)/step)]
for p in grid: fit GLM(Tweedie(var_power=p, link=log), freq_weights=w); record (p, deviance, converged)
best = argmin deviance among converged
if best exists and profile has an interior minimum:
    refined = [round(best + i*refine_step, 6) for i in -1..+1]
    choose refined argmin (converged) → selected_p, fallback=False
else:
    selected_p = p_fallback, fallback=True
```

`deviance` is `results.deviance` (summed unit deviance at the fitted parameters). The final model is fit at `selected_p` and must report `converged=True`.

### 4.4 Calibration metrics

```text
holdout_mean_deviance = family.deviance(y_test, mu_test, freq_weights=w_test, scale=1) / n_test
```

Decile A/E: rank test rows by `mu` (predicted pure premium); split into `decile_count` bins; per bin compute `actual = Σ capped_loss`, `expected = Σ mu*exposure_capped`, `A/E = actual/expected`, plus count and exposure.

Segment-band A/E: same ratio-of-sums per DriverAge band, CarAge band, Density band, and the 36-cell cross-product.

Secondary: `mae = mean(|actual_loss − expected_loss|)`, `rmse = sqrt(mean((actual_loss − expected_loss)^2))` on per-policy total loss (EUR), recorded only.

**Covers:** FR-1–FR-3, FR-8–FR-11, AC-2, AC-4, AC-7.

## 5. Determinism and Evidence

- Fixed `random_state=42`; split over the sorted `PolicyID` ordering.
- Fixed design-matrix column order from frozen config.
- `p` grid generated arithmetically (no floating-point drift: rounded to 6 decimals).
- Artifacts written with `lineterminator="\n"`, JSON with `sort_keys=True`, no timestamps.
- Gate evidence: two consecutive runs produce identical `glm_baseline_model.json`, `glm_fitted_loss_cost.csv`, and `glm_calibration_report.json` hashes; a test recomputes them.

**Covers:** FR-14, AC-5.

## 6. Error Handling

Add `GLMBaselineError(AEGISError)` with `details` carrying `rule`, `column`, `count`, and sample keys. Conditions:

- missing/extra input columns;
- non-positive `exposure_capped`;
- negative `ClaimAmount`;
- final fit non-converged;
- empty input;
- artifact schema validation failure.

No `except` may swallow these into a successful return. The CLI returns exit code 1 and DVC halts.

**Covers:** FR-1, FR-13, AC-1.

## 7. Testing Strategy

| Layer | Test | Covers |
|---|---|---|
| Unit | Target = `min(ClaimAmount, cap)/exposure_capped`; non-positive exposure raises | FR-1, FR-2 |
| Unit | Design matrix columns equal the frozen contract; no claims/log_density/raw-age term | FR-5 |
| Unit | Split is 80/20 and deterministic for a fixed seed | FR-7 |
| Unit | `select_tweedie_p` returns an interior `p` on synthetic data and honors fallback on degenerate data | FR-6 |
| Unit | Convergence flag honored; non-converged final fit raises | FR-13, AC-1 |
| Unit | Decile A/E is ratio-of-sums; bins partition the test set | FR-9, AC-7 |
| Unit | Segment A/E covers DriverAge/CarAge/Density + 36-cell cross | FR-10 |
| Unit | MAE/RMSE computed but not referenced by any gate | FR-11, AC-4 |
| Unit | No frequency-only/severity-only estimator present | FR-12, AC-6 |
| Unit | Cap parameterization: two caps share one code path, different targets | FR-4, AC-8 |
| Integration | Two runs → identical artifact hashes | FR-14, AC-5 |
| Integration | Metadata/report round-trip through Pydantic | FR-14, AC-9 |
| Regression | Empirical run: converges, produces deviance + decile/segment A/E | AC-2 |

Tests use small synthetic frames for rule isolation plus one empirical run for the end-to-end gate. No test asserts a value equals itself; determinism is asserted by hash comparison.

## 8. Security and Authority

- Input is untrusted data; column and numeric checks are enforced before fitting.
- No dynamic code execution, SQL, network access, or secret access.
- Output paths come from config/CLI, never from data content.
- The pipeline fits and reports only; it does not approve, promote, or publish (PRODUCE ≠ APPROVE).
- No personal identifiers beyond the benchmark `PolicyID`; no external enrichment.

**Covers:** INV-3, INV-5, INV-10.

## 9. Observability

- Console summary: cap, selected `p`, `fallback`, `converged`, split counts, holdout deviance, output paths.
- Durable evidence: metadata and report JSON; DVC lock records artifact hashes.

## 10. Dependencies and Risks

| Risk | Mitigation |
|---|---|
| `statsmodels` Tweedie deviance/link API differences | Pin lockfile; verify exact semantics in T8; fall back to documented fixed `p` |
| Non-convergence at grid extremes | Restrict to converged points; fallback + final-convergence requirement |
| Profile deviance ignores the Tweedie normalizing term | Documented heuristic per OD-1; flagged `fallback` path available |
| Category with no claims in a dummy | Log link keeps positive means; no special-casing |
| Full-data memory/time | Two ~413k-row frames; one fit + ~19 grid fits at ~35 columns; within budget |
| Schema drift from upstream | Exact column-set check; extra columns rejected |

No new dependency beyond `statsmodels` (already pinned).

## 11. Impact on Existing Systems

- `dvc.yaml`: new `glm_baseline` stage after `build_features`; existing stages unchanged.
- `params.yaml` / `config/schema.py`: additive `glm_baseline` + `glm_paths`; existing keys unchanged.
- `cli.py`: new `build-glm` subcommand; existing routing unchanged.
- `pipelines/__init__.py`: no required change.
- `schemas/glm.py`: new typed module.
- Tests: new files; existing tests unaffected.

## 12. Deployment

None. Local DVC execution only. CI-smoke integration is deferred to Stage 14.

## 13. Traceability

| Requirement | Component | Validation |
|---|---|---|
| FR-1, FR-2, FR-3 | Fit core | T3 tests |
| FR-4 | Config + target | T1/T3 tests |
| FR-5 | Design matrix | T3 test |
| FR-6 | Profile deviance | T3 test |
| FR-7 | Split | T3 test |
| FR-8, FR-9, FR-10 | Calibration | T5 tests |
| FR-11 | Secondary diagnostics | T5 test |
| FR-12 | Single-model assertion | T7 test |
| FR-13 | Convergence handling | T1/T3 tests |
| FR-14 | Writer + schemas | T2/T4 tests |
