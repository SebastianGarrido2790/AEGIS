# Specification: P2-F06 Causal Estimator & Segment Inference
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F06` (`006_causal-estimator`, Stage 8)  
**Governing Decisions:** ADR-013 (CausalForestDML, held-out evaluation, segment-average interval inference, truth isolation), ADR-014 (Tier 1 segment contract fields), P2-D5 (Q2-frozen scale, nuisance family, folds), P2-D6 (segment-average inference, never average of policy endpoints), P2-D6a (thresholds deferred to Q3), ADR-005 (config-only tunables), ADR-015 (reproducibility), INV-4 / INV-5 (deterministic arithmetic, no LLM, no price adjustments), INV-8 (module ceiling)  
**SDD Depth:** Full (`spec.md` → approval → `plan.md` → `tasks.md`)  
**Status:** APPROVED — October 26, 2026 (OD-1 … OD-5 as recommended)  
**Implementation Status:** ✅ COMPLETE — Gate 8 PASSED October 26, 2026  
**Date:** October 26, 2026  

---

## 1. Context and Problem

Stage 7 (Q2) froze the estimator scale and nuisance configuration: `CausalForestDML` with boosted-tree nuisance models, five-fold cross-fitting, and the full 413,169-policy sample split 80/20 into ~330,535 training and ~82,634 held-out policies. The Q2 harness proved the configuration is budget-feasible and produces a segment Spearman of 0.9014 on one pilot seed. Q2 did **not** deliver a production estimator pipeline, a persisted model artifact, or the segment-level inference table that Phase 3 and the showcase will consume.

Stage 8 must turn the Q2 configuration into a deterministic, truth-isolated production pipeline that:

1. Fits `CausalForestDML` on the training split only.
2. Produces segment-average treatment-effect and elasticity estimates with 95% confidence intervals on the **segment mean**, not an average of policy-level interval endpoints.
3. Proves interval finiteness and self-consistency before any downstream validation or registration.
4. Keeps the fitted estimator out of the serving path; serving consumes plain segment data only.

The central design constraint is that the DGP truth sidecar (`q0`, `tau`, `epsilon`) is validation-only (ADR-013). The production elasticity denominator `q0` must therefore be estimated from estimator-side artifacts alone. This specification fixes that method and the interval construction, because both are load-bearing for Stage 9 coverage measurement and Stage 10 recovery claims.

---

## 2. Objective and Expected Outcome

Deliver a reproducible production estimator pipeline that:

1. Validates the 29-column Stage 6 training contract and rejects any truth column before fitting.
2. Reproduces the frozen Q2 split (stratified 80/20, `random_state: 42`) and fits the frozen GBT-configuration `CausalForestDML` on the training partition only.
3. Estimates a deconfounded baseline retention probability `q0_hat` per held-out policy from the fitted nuisance models, without reading the truth sidecar.
4. Emits 35 segment-level rows containing `segment_id`, held-out policy count, `q0_hat`, `tau_hat`, `epsilon_hat`, and both τ and ε confidence intervals.
5. Asserts that 100% of segment intervals are finite, ordered, and contain their own point estimate; any violation raises a typed error and halts.
6. Persists the fitted estimator as a non-promotable run artifact and proves that serving modules never import or deserialize it.
7. Produces deterministic segment estimates and metadata, with input/config SHA-256 hashes for lineage.

---

## 3. Scope

### In scope

- `src/aegis/pipelines/causal_estimator.py`: contract validation, frozen split, estimator fitting, truth-free q0 estimation, segment aggregation, interval construction, artifact writing.
- `src/aegis/schemas/causal.py`: typed `SegmentElasticityEstimate`, `CausalEstimatorMetadata`, and table-level contracts.
- `CausalEstimatorError` in `src/aegis/utils/exceptions.py`.
- Estimator configuration additions in `params.yaml` and `src/aegis/config/schema.py` (explicit forest hyperparameters, `ci_alpha`, `q0_floor`, artifact paths, `n_jobs`).
- CLI subcommand `train-estimator` in `src/aegis/pipelines/cli.py` with typed failure handling.
- A non-promotable fitted-model run artifact and a run manifest under `artifacts/`.
- Unit, property, and import-boundary tests, including a fast fit on a small synthetic fixture.
- One full-data execution producing the Stage 8 gate evidence.

### Out of scope

- Validation metrics against the truth sidecar, negative/positive controls, and Q3 threshold calibration (Stage 9, P2-F07).
- Multi-seed final validation and threshold immutability (Stage 10).
- Hidden-confounder sensitivity (Stage 11, P2-F08).
- MLflow packaging, candidate/approved aliases, and promotion (Stage 12, P2-F09).
- DVC DAG assembly and full CI tiers (Stage 14, P2-F10).
- Hyperparameter tuning. The Q2-effective forest configuration is frozen, not optimized.

---

## 4. Functional Requirements (EARS)

**FR-1 (estimator input contract).** When the estimator pipeline starts, it shall load `data/processed/elasticity_training.csv`, require the unique 29-column Stage 6 contract (`PolicyID`, 25 pre-treatment features, `treatment_rate_change`, `renewed`, `segment_id`), require a non-null, unique `PolicyID`, and reject any of the truth columns `tau`, `epsilon`, `q0`, `true_effect`, `tau_ground_truth`, `baseline_renewal_q0`, `true_tau`, `true_epsilon`, or `fitted_loss_cost` with `CausalEstimatorError` before fitting.

**FR-2 (frozen configuration).** The pipeline shall read all estimator settings from `params.yaml`: `training_sample_size`, `nuisance_model`, `cv_folds`, `random_state`, forest hyperparameters, `ci_alpha`, `q0_floor`, `n_jobs`, and output paths. No tunable value may be hardcoded in the module.

**FR-3 (frozen split).** The pipeline shall draw a segment-stratified total sample of `training_sample_size` policies (the full table when the configured size equals its row count), then split it with `test_size = tier1_ml.test_size` and `random_state = tier1_ml.random_state`, stratified by `segment_id`. The training partition alone shall be passed to `fit`. The held-out partition shall be used only for inference and diagnostics.

**FR-4 (nuisance and forest configuration).** When `nuisance_model` is `gbt`, the pipeline shall use `GradientBoostingClassifier(random_state=random_state)` for the outcome nuisance and `GradientBoostingRegressor(random_state=random_state)` for the treatment nuisance. It shall construct `CausalForestDML` with `discrete_outcome=True`, `discrete_treatment=False`, `inference=True`, `cv=cv_folds`, `random_state=random_state`, and the frozen `n_estimators`, `min_samples_leaf`, `min_samples_split`, `max_depth`, `max_samples`, and `subforest_size` values from configuration.

**FR-5 (policy-level effect estimation).** After fitting, the pipeline shall predict one treatment effect per held-out policy using `effect(X, T0=0, T1=1)`, normalize the result to a one-dimensional float array, and reject non-finite values.

**FR-6 (truth-free baseline retention estimate).** The pipeline shall estimate baseline retention without the truth sidecar. For each held-out policy `i` it shall compute:

$$\hat{p}_y(i) = \frac{1}{F}\sum_{f=1}^{F} \hat{E}[Y \mid X_i]^{(f)}, \qquad \hat{p}_t(i) = \frac{1}{F}\sum_{f=1}^{F} \hat{E}[T \mid X_i]^{(f)}$$

using the `F = cv_folds` cross-fitted nuisance models stored by the fitted estimator, then

$$\hat{q}_{0i} = \operatorname{clip}\left(\hat{p}_y(i) - \hat{\tau}_i \cdot \hat{p}_t(i),\; q_{0,\text{floor}},\; 1.0\right).$$

The pipeline shall validate that the nuisance model collection has exactly one outcome and `cv_folds` folds; an unexpected structure raises `CausalEstimatorError`. `q0_floor` must be strictly positive.

**FR-7 (segment-average point estimates).** For each realized segment `s` among the held-out policies, the pipeline shall compute:

- Segment-average treatment effect: $\hat{\tau}_s = \text{mean\_point}$ from `ate_inference(X_s, T0=0, T1=1)`; the estimator's own segment-average estimate is authoritative so that its interval encloses it by construction.
- Segment baseline retention: $\hat{q}_{0s} = \frac{1}{N_s}\sum_{i \in s}\hat{q}_{0i}$.
- Segment elasticity: $\hat{\epsilon}_s = \hat{\tau}_s / \hat{q}_{0s} = \sum_{i \in s}\hat{\tau}_i / \sum_{i \in s}\hat{q}_{0i}$ (ratio of sums).

A non-positive `q0_s` raises `CausalEstimatorError`; no silent floor widening is permitted.

**FR-8 (segment-average confidence intervals).** For each segment the pipeline shall obtain the 95% τ interval from `ate_inference(X_s, T0=0, T1=1).conf_int_mean(alpha=ci_alpha)`. It shall convert it to the ε scale by dividing both bounds by the segment's `q0_s` point estimate:

$$[\ell_{\epsilon,s}, u_{\epsilon,s}] = [\ell_{\tau,s}/\hat{q}_{0s},\; u_{\tau,s}/\hat{q}_{0s}].$$

The interval is conditional on the estimated segment `q0`; `q0` estimation uncertainty is not propagated in Stage 8 and is measured empirically by the Stage 9 coverage metric.

**FR-9 (interval self-consistency).** When writing segment estimates, the pipeline shall assert for every segment that all six bounds/points are finite, that `lower <= point <= upper` for both τ and ε, and that every configured segment is present exactly once. Any failure raises `CausalEstimatorError`; intervals are never silently widened, clipped, or dropped.

**FR-10 (truth isolation).** The estimator module, its CLI path, and its tests used for production behavior shall not import `aegis.pipelines.truth_loader`, read the truth sidecar path, or contain truth columns. The existing AST import-boundary test shall continue to cover the new module.

**FR-11 (serving decoupling).** Modules under `src/aegis/api/` shall not import `aegis.pipelines.causal_estimator`, `econml`, or any model-deserialization library for the estimator artifact. The fitted model shall never be loaded on the serving path. A structural test shall scan API module imports and fail on violation.

**FR-12 (deterministic segment artifact).** The segment-estimates CSV shall be sorted by `segment_id`, use a fixed column order, and be byte-identical across repeated runs with the same inputs, configuration, and seed. The deterministic metadata JSON shall contain no wall-clock timings or model-binary hashes.

**FR-13 (run artifacts).** The pipeline shall persist:
1. `data/processed/segment_elasticity_estimates.csv` — deterministic segment table.
2. `data/processed/causal_estimator_metadata.json` — deterministic lineage/configuration metadata.
3. `artifacts/causal_estimator/estimator.joblib` — non-promotable fitted estimator.
4. `artifacts/causal_estimator/run_manifest.json` — run-only diagnostics (model SHA-256, wall time).

The two `artifacts/` files are gitignored run artifacts and are not part of the serving package.

**FR-14 (typed failures).** Contract violations, nuisance-structure errors, missing segments, non-finite predictions, non-positive `q0`, and interval violations shall raise `CausalEstimatorError` with structured `details`. Unexpected library exceptions during fit shall be wrapped with context and re-raised as `CausalEstimatorError`; no bare exception may be converted into a successful return payload. The CLI shall print `[BLOCK]` and return exit code 1.

**FR-15 (CLI entrypoint).** The CLI shall expose `train-estimator` with optional output overrides and shall return 0 only after all artifacts are written and validated.

---

## 5. Data Contracts

### 5.1 Input contracts

| Input | Contract |
|---|---|
| `data/processed/elasticity_training.csv` | 413,169 rows; 29 unique columns; `PolicyID` unique/non-null; 25 pre-treatment features; `treatment_rate_change ∈ [-0.15, 0.15]`; `renewed ∈ {0,1}`; `segment_id` present; zero truth columns. Expected SHA-256 `f08918631e891203f845f7107d9e7c33572d817b16f81d75b7f737424471cdb8`. |
| `data/processed/segment_definitions.json` | Loaded only for `segment_count` (expected 35) and lineage hash; contains no policy-level truth. |
| `params.yaml` | Frozen Q2 estimator configuration plus Stage 8 settings. |

**Prohibited:** `data/processed/truth_sidecar.csv` and any truth column. The production module has no code path that opens the sidecar.

### 5.2 Output contract — `segment_elasticity_estimates.csv`

35 rows, one per realized segment, sorted by `segment_id`, with fixed columns:

| Column | Type | Rule |
|---|---|---|
| `segment_id` | string | Canonical `D:<driver>_C:<car>_Dens:<density>`; unique |
| `inference_policy_count` | int | Held-out policies in the segment; > 0 |
| `baseline_retention_q0` | float | Segment mean `q0_hat`; in `(0, 1]` |
| `tau_hat` | float | Segment-average effect; finite |
| `tau_ci_lower` | float | Finite; `<= tau_hat` |
| `tau_ci_upper` | float | Finite; `>= tau_hat` |
| `epsilon_hat` | float | `tau_hat / baseline_retention_q0`; finite |
| `epsilon_ci_lower` | float | Finite; `<= epsilon_hat` |
| `epsilon_ci_upper` | float | Finite; `>= epsilon_hat` |

No proposed adjustment, rate delta, or premium field may appear (INV-4, INV-5).

### 5.3 Output contract — `causal_estimator_metadata.json`

Validated by `CausalEstimatorMetadata`:

- `estimator_name` (`CausalForestDML`), `nuisance_model`, `cv_folds`, `random_state`
- `n_estimators`, `min_samples_leaf`, `min_samples_split`, `max_depth`, `max_samples`, `subforest_size`
- `training_sample_size`, `train_rows`, `test_rows`, `feature_count`, `segment_count`
- `ci_alpha`, `q0_floor`, `q0_estimation_method` (`deconfounded_nuisance_fold_average`)
- `interval_self_consistency_rate`, `finite_interval_rate` (both must equal 1.0)
- `input_hashes` (training table, segment definitions), `config_fingerprint` (SHA-256)
- `model_version` (deterministic identifier derived from config/data hashes)

No wall-clock time, no truth metrics, no person-level data.

### 5.4 Run-only contract — `artifacts/causal_estimator/run_manifest.json`

- `estimator_artifact_sha256`, `estimator_artifact_bytes`, `fit_wall_seconds`, `pipeline_wall_seconds`, `artifact_path`.
- Not a DVC data product; not consumed by serving.

---

## 6. Edge Cases

| Case | Required behavior |
|---|---|
| Training table contains a truth column | Reject before fit with `CausalEstimatorError` naming the column; no model is constructed. |
| Nuisance model collection shape differs from `(1, cv_folds)` | Raise `CausalEstimatorError`; no silent fallback to a single fold. |
| `q0_hat` clipped to floor for a policy | Permitted; the segment denominator must still be positive or the run fails. |
| Segment absent from held-out split | Raise `CausalEstimatorError` listing missing segment IDs; no partial table is written. |
| Non-finite effect, `q0`, point estimate, or bound | Raise `CausalEstimatorError`; do not drop the segment or replace with zero. |
| Inverted or non-containing interval | Raise `CausalEstimatorError`; do not widen bounds. |
| `training_sample_size` exceeds available rows | Raise `CausalEstimatorError`. |
| Repeated full run | Segment CSV is byte-identical; run manifest timings may differ. |
| Serving attempts to import estimator/econml | Import-boundary test fails. |
| Truth loader import injected into estimator module | Existing AST boundary test fails. |

---

## 7. Error Handling

Add to `src/aegis/utils/exceptions.py`:

- `CausalEstimatorError(AEGISError)` — raised for input-contract violations, fit failures, nuisance-structure violations, missing segments, non-finite values, non-positive baselines, interval violations, and artifact-validation failures. Carries `details` with rule, count, and sample keys where applicable.

Errors propagate. The CLI returns 1 and DVC (Stage 14) will halt. No silent fallback, no partial artifact publication.

---

## 8. Non-Functional Constraints

- **Truth isolation (ADR-013):** no truth sidecar read; no truth columns; AST import boundary preserved.
- **INV-4 / INV-5:** deterministic Tier 1 arithmetic only; zero LLM calls; no proposed price adjustment.
- **INV-8:** every module strictly below 1,000 lines.
- **Performance:** the full-data fit plus 35 segment inferences shall remain within the frozen 30-minute budget (Q2 measured 514.8 s for fit plus inference on one seed).
- **Determinism:** same inputs and seed produce byte-identical segment estimates; forest and nuisance models use fixed seeds.
- **Reproducibility:** metadata records SHA-256 input hashes and a canonical config fingerprint.
- **Security:** no secrets in code, config, artifacts, or logs; the joblib artifact is local-only and never deserialized by serving.

---

## 9. Acceptance Criteria

- **AC-1:** Full run fits on the training partition only and produces estimates for all 35 realized segments from held-out policies.
- **AC-2:** `segment_elasticity_estimates.csv` has 35 rows and the exact nine-column contract; all values are finite.
- **AC-3:** 100% of τ and ε intervals contain their own point estimate; `interval_self_consistency_rate = 1.0` and `finite_interval_rate = 1.0`.
- **AC-4:** For at least one segment, an independent hand calculation confirms `epsilon_hat = sum(tau_i)/sum(q0_i)`.
- **AC-5:** Planting any truth column in a training fixture fails before fit with `CausalEstimatorError`.
- **AC-6:** The estimator module does not import the truth loader; `src/aegis/api/` does not import the estimator, `econml`, or the model artifact.
- **AC-7:** Two consecutive runs on identical inputs produce byte-identical segment CSVs.
- **AC-8:** Metadata validates against `CausalEstimatorMetadata` and records input/config hashes; no timings or truth metrics are present.
- **AC-9:** The fitted estimator is persisted only under `artifacts/` as a non-promotable run artifact and is absent from `data/processed/` and the serving package.
- **AC-10:** `pytest`, Ruff (changed files), Pyright, and the module-size check pass; Gate 8 evidence is recorded in `stage_evidence.md`.

---

## 10. Open Decisions for Operator Approval

### OD-1 — Truth-free baseline retention `q0` estimation (FR-6, FR-7)

- **Context:** The published elasticity denominator cannot come from the truth sidecar. The DGP makes true `q0` constant within each segment, but the production pipeline must estimate it from estimator-side artifacts.
- **Options:**
  - **(A) Recommended — deconfounded nuisance fold average:** average the cross-fitted `E[Y|X]` and `E[T|X]` predictions and compute `clip(E[Y|X] - tau_hat * E[T|X], q0_floor, 1.0)`. At 100k probe rows this produced segment-ε MAE 0.0038 and correlation 0.9997 against the synthetic truth; at full scale it should improve further.
  - **(B) Raw nuisance `E[Y|X]`:** simpler, but biased by the confounded treatment mean; probe segment-ε MAE 0.0064.
  - **(C) Observed segment renewal rate:** simplest, but treatment assignment is risk-correlated; probe segment-ε MAE 0.0074 and it uses post-treatment outcome information without adjustment.
- **Recommendation:** A. It uses only fitted nuisance models, reduces confounding bias, and showed the lowest error and highest rank correlation in the Stage 7-adjacent probe.
- **Consequence:** The Stage 8 interval treats `q0` as a point estimate (OD-3); Stage 9 coverage will expose any residual bias.

### OD-2 — Inference population for the published segment table (FR-3, FR-6, FR-7)

- **Context:** D6 requires evaluation on held-out policies. The training split is ~330,535 policies; the held-out split is ~82,634. Inference could use the held-out split or all policies.
- **Options:**
  - **(A) Recommended — held-out test policies only:** fit on train, estimate every segment from its ~20% held-out share. Honest out-of-sample behavior, consistent with D6, and the same inference path Stage 9 will validate.
  - **(B) Fit on train, infer on all policies:** more rows per segment, but ~80% of inference rows are in-sample and intervals are optimistic.
  - **(C) Fit on all policies and infer in-sample:** uses all data but contradicts the held-out principle for the published artifact.
- **Recommendation:** A.
- **Consequence:** Each segment still clears the 1,000-policy floor overall; held-out shares are ~200+ policies per segment, sufficient for segment-average intervals but wider than in-sample intervals. This is intentional and honest.

### OD-3 — ε confidence-interval construction (FR-8)

- **Context:** The segment interval must be on the segment-average effect, not an average of policy-level endpoints (P2-D6). `ate_inference(...).conf_int_mean` supplies the τ interval.
- **Options:**
  - **(A) Recommended — scale the τ interval by the segment `q0` point estimate:** monotone, guarantees `epsilon_hat` lies inside when the τ interval contains `tau_hat`, and keeps the inference on the segment mean. Uncertainty in `q0_hat` is not propagated.
  - **(B) Delta method:** propagate `q0` variance/covariance; requires an additional variance estimate not produced by the current nuisance setup and adds methodological complexity.
  - **(C) Bootstrap:** re-fit or resample per segment; materially more expensive and not part of the frozen Q2 configuration.
- **Recommendation:** A, with the conditional-on-`q0` limitation documented in the metadata and measured by Stage 9 coverage.
- **Consequence:** Stage 9's `m3` coverage metric is the explicit test of whether the interval is adequate. If coverage fails, Stage 9 returns to estimator/interval design rather than lowering thresholds (P2-D6a).

### OD-4 — Persist the fitted estimator (FR-13)

- **Context:** ADR-014 requires the fitted `CausalForestDML` to remain a non-promotable run artifact. Stage 12 packages lineage and the validation summary; retraining later would create a different model than the one whose interval properties were gated.
- **Options:**
  - **(A) Recommended — persist `artifacts/causal_estimator/estimator.joblib` plus a run manifest:** the exact Stage 8 model is available to Stage 12; serving never loads it; gitignored and local-only.
  - **(B) Do not persist:** lower local disk use, but Stage 12 must retrain or register an unvalidated model.
- **Recommendation:** A. Add `joblib` as an explicit dependency rather than relying on a transitive scikit-learn dependency.
- **Consequence:** The model binary hash lives only in `run_manifest.json`, so deterministic metadata and segment outputs are unaffected.

### OD-5 — Freeze effective forest hyperparameters in `params.yaml` (FR-2, FR-4)

- **Context:** Q2 used `CausalForestDML` library defaults (100 trees, `min_samples_leaf=5`, `min_samples_split=10`, `max_depth=None`, `max_samples=0.45`, `subforest_size=4`). Leaving them implicit conflicts with the config-only tunables rule and makes the production fit less auditable.
- **Options:**
  - **(A) Recommended — record the exact effective Q2 values explicitly in `params.yaml` and the config schema:** no behavior change from Q2; fully auditable; future changes require an approved config change.
  - **(B) Leave library defaults implicit:** fewer config keys, but the effective model is not visible in the repository and could drift with an EconML upgrade.
- **Recommendation:** A. Also add `ci_alpha: 0.05`, `q0_floor: 1e-6`, `n_jobs: -1`, and the four Stage 8 artifact paths.
- **Consequence:** The production fit is pinned to the Q2-effective configuration; no retuning occurs in Stage 8.

---

## 11. Definition of Done

1. This specification is approved by the operator (OD-1 … OD-5 resolved).
2. `plan.md` and `tasks.md` are drafted and approved.
3. `causal_estimator.py`, `schemas/causal.py`, config, exceptions, and CLI are implemented.
4. Unit/property/import-boundary tests pass, including a fast small-fixture fit.
5. Full-data CLI run completes within budget and writes all four artifacts.
6. All 35 segments have finite, self-consistent τ and ε intervals.
7. Gate 8 evidence is recorded in `stage_evidence.md`; the execution-plan feature register and stage index are updated.
8. `pytest`, Ruff (changed files), Pyright, and module-size checks pass.
9. Human Gate 8 sign-off is recorded; Stage 9 is unlocked.

---

## 12. Traceability

| Requirement | Source |
|---|---|
| FR-1 | Stage 6 training contract; ADR-015; INV-3 |
| FR-2, FR-4, OD-5 | ADR-005; Q2 frozen configuration; P2-D5 |
| FR-3 | P2-D5; D6 held-out evaluation; Stage 0 Q2 sample-size resolution |
| FR-5, FR-6, FR-7, OD-1 | ADR-013; P2-D2a; truth isolation; ratio-of-sums (ADR-014, P2-D3) |
| FR-8, OD-3 | P2-D6 segment-average inference; Tier1SegmentOutput interval contract |
| FR-9 | Gate 8 finite/self-consistent interval requirement |
| FR-10 | ADR-013 truth isolation; Stage 6 import boundary |
| FR-11 | ADR-014 plain-data serving; Gate 8 serving-decoupling requirement |
| FR-12, FR-13, OD-4 | ADR-015 tiered reproducibility; ADR-014 non-promotable run artifact |
| FR-14, FR-15 | Typed failure conventions; INV-5; Stage 14 DAG entrypoint |
| AC-1 … AC-10 | Execution Plan Stage 8 gate; Definition of Done |
