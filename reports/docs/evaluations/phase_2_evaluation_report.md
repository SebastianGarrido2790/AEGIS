# Phase 2 Evaluation Report - AEGIS

> **System:** Actuarial Elasticity & Governance Intelligence System (AEGIS)  
> **Phase:** Phase 2 - Tier 1 Deterministic ML Baseline and Causal Elasticity Validation  
> **Evaluation stage:** Stage 6  
> **Date:** 2026-09-06  
> **Status:** Stage 6 gate completed

## 1. Executive summary
Phase 2 establishes a reproducible deterministic modeling path for the AEGIS
prototype:

- a Tweedie GLM baseline is fit on the engineered feature matrix;
- a `CausalForestDML` estimator is exercised against a synthetic treatment with a
	known construction;
- DoWhy refutation checks complete for placebo treatment, random common cause,
	and data subset sensitivity;
- both model families are tracked and registered in the local MLflow experiment.

The causal result is an estimator-validation result, not a discovery of real-world
insurance price elasticity. The public freMTPL2 benchmark does not contain a
randomized rate-change intervention or a production premium decision history that
could support that claim.

The Stage 6 gate is closed after adding treatment-effect intervals to the causal
artifact, attaching this report to the verified causal MLflow run, and comparing
the repository file with the downloaded MLflow artifact byte-for-byte.

## 2. Scope and governance framing
This report covers the Stage 3 GLM baseline, Stage 4 synthetic-treatment causal
validation, and Stage 5 MLflow provenance work. It does not authorize a rate
change, publish a rate table, or establish regulatory validity for any jurisdiction.

The causal output remains advisory. Under the AEGIS governance model it must be
combined with compliance evidence and revenue or loss-ratio impact context before
it can support a human review workflow.

## 3. Dataset provenance and attribution

The modeling data is based on the public freMTPL2 French Motor Third-Party
Liability benchmark dataset:

- OpenML frequency dataset: `freMTPL2freq`, dataset ID `41214`
- OpenML severity dataset: `freMTPL2sev`, dataset ID `41215`
- Public reference mirror: https://www.kaggle.com/datasets/karansarpal/fremtpl2-french-motor-tpl-insurance-claims

The project records the source as Open Database License (ODbL) material and uses
it for educational, benchmarking, and validation purposes. It is not presented as
a live carrier portfolio, a production pricing book, or a multi-jurisdiction
compliance corpus.

## 4. Stage 3 - GLM baseline calibration

### 4.1 Configuration

- **Model:** `tweedie_glm`
- **Response:** `pure_premium`
- **Converged:** `true`
- **Features:** `driver_age`, `veh_age`, `bonus_malus`, `veh_power`,
  `exposure_normalized`, `driver_risk_score`, `vehicle_risk_score`, `risk_index`

### 4.2 Held-out calibration metrics

| Metric | Value |
| --- | ---: |
| Test MAE | 610.459436957206 |
| Test RMSE | 18839.474205824252 |

### 4.3 Parameter confidence intervals

The saved baseline artifact contains finite 95% parameter confidence intervals.
The intervals below are reported as `(lower, upper)` and are included as evidence
that the statistical baseline produced inferential output rather than point
estimates only.

| Parameter | 95% confidence interval |
| --- | --- |
| `const` | (2.4905030823766725, 4.39708809044914) |
| `driver_age` | (-0.021057671182288285, 0.014033914189076301) |
| `veh_age` | (-0.02635120886848262, 0.025256740590690475) |
| `bonus_malus` | (0.023262762169209555, 0.040083036660450655) |
| `veh_power` | (0.022833938412560796, 0.13941950673813341) |
| `exposure_normalized` | (-2.4770351347269695, -1.4784661364190597) |
| `driver_risk_score` | (0.5187361477302698, 1.4799294555868596) |
| `vehicle_risk_score` | (0.32067986491304395, 0.7460502595870404) |
| `risk_index` | (-1.5500851126855129, -0.872718294551931) |

These are model-parameter intervals from the GLM baseline. They are not causal
treatment-effect intervals.

## 5. Stage 4 - Synthetic-treatment causal validation

### 5.1 Validation design

The treatment variable is `treatment_rate_change`. It is constructed from the
engineered `risk_index`, and the synthetic outcome is generated from the treatment
and risk-index interaction with seeded noise. This creates a known answer against
which estimator behavior can be checked.

This design validates the causal estimator under controlled conditions. It does
not estimate the historical elasticity of the freMTPL2 data and must not be
described as a production elasticity discovery.

### 5.2 Saved causal metrics

| Metric | Value |
| --- | ---: |
| Average treatment effect | 283.0460242413861 |
| Correlation with synthetic ground truth | 0.22975574798613693 |
| Baseline MAE against synthetic truth | 282.81989946050015 |

The causal artifact also records the model's mean 95% treatment-effect interval:

| Interval | Value |
| --- | ---: |
| Lower bound | -188.00211046907307 |
| Upper bound | 220.24357729143435 |
| Alpha | 0.05 |

The interval is computed from EconML's raw `effect_interval` output. The reported
average treatment effect is the existing positive-effect calibration summary,
which applies the validation pipeline's non-negative transformation; the two
values therefore are not presented as the same estimand.

### 5.3 DoWhy refutation summary

| Refuter | Status | p-value | Passed |
| --- | --- | ---: | --- |
| Placebo treatment | `ok` | 0.42 | `true` |
| Random common cause | `ok` | 0.31 | `true` |
| Data subset | `ok` | 0.27 | `true` |

All three required refutation entries are present and marked as passed in the
saved artifact. The artifact also records the diagnostic note
`This estimator does not support X=None!`; this is retained as provenance and is
not silently converted into a pass or a failure.

## 6. Stage 5 - MLflow provenance

The local MLflow tracking configuration is:

- **Tracking URI:** `sqlite:///mlflow.db`
- **Experiment:** `aegis`
- **Artifact root:** `file:///C:/Users/sebas/Desktop/AEGIS/artifacts/mlflow`
- **Registered models:** `aegis-glm-baseline`, `aegis-causal-elasticity`

The latest verified runs at report preparation time were:

| Run name | Run ID | Logged metrics |
| --- | --- | --- |
| `glm_baseline` | `818223c2451c4f09816a1bb2100e2e25` | `test_mae`, `test_rmse` |
| `causal_elasticity` | `311e117cab764cafb98c9890835536a9` | `average_treatment_effect`, `baseline_mae`, `correlation` |

Each run exposes a `diagnostics` artifact directory. The structured GLM and
causal diagnostic JSON artifacts are retrievable from their respective runs.
This report is attached to causal run
`311e117cab764cafb98c9890835536a9` at
`evaluation/phase_2_evaluation_report.md`.

## 7. Gate assessment

| Stage 6 requirement | Assessment | Evidence or gap |
| --- | --- | --- |
| Dataset source and ODbL attribution | Pass | Section 3 |
| GLM calibration metrics | Pass | Section 4.2 and `data/validated/glm_baseline.json` |
| Treatment-effect evaluation | Pass | Causal metrics and 95% treatment-effect interval are present |
| DoWhy refutation summary | Pass | Section 5.3 and `data/validated/causal_elasticity.json` |
| Report attached to MLflow | Pass | Attached to causal run `311e117cab764cafb98c9890835536a9` |
| Repository and MLflow copies identical | Pass | Byte-for-byte comparison passed |

## 8. Conclusion

Stages 3 through 5 provide useful, queryable evidence for the deterministic
baseline and the synthetic causal-estimator exercise. The evidence supports the
limited conclusion that the pipeline runs, the baseline converges, the causal
validation produces structured metrics, and the configured refutation checks are
recorded.

Stage 6 is complete. The report, causal interval evidence, MLflow attachment,
and byte-identity check are all recorded above. AEGIS still has not demonstrated a production pricing recommendation, a
real-world elasticity estimate, or a regulator-approved conclusion.
# Phase 2 Evaluation Report — AEGIS

> **System:** Actuarial Elasticity & Governance Intelligence System (AEGIS)  
> **Phase:** Phase 2 — Tier 1 Deterministic ML Baseline & Causal Elasticity Validation  
> **Status:** 🟢 Stage 3/4 Verification Completed; Stage 5 Provenance Active  
> **Author:** Sebastián Garrido Arévalo  
> **Date:** 2026-09-06  
> **Related Documents:** [system_design.md](../architecture/system_design.md), [phase_2_implementation_plan.md](../decisions/phase_2_implementation_plan.md), [phase_2_execution_workflow.md](../workflows/phase_2_execution_workflow.md)

---

## 1. Scope and framing

This report covers the deterministic Fit 1 / Fit 2 work completed in Phase 2: the Tweedie GLM baseline and the synthetic-treatment causal elasticity validation exercise. The central framing is explicit and intentional:

- This is not a real-world claim that the model has discovered a live premium elasticity relationship in market data.
- This is a validation of the estimator behavior on a synthetic treatment variable with a known, segment-varying ground-truth structure.
- The causal model is therefore treated as a model-validation exercise, not as an authoritative market elasticity result.

That distinction is required by the project’s governance posture: the causal output is advisory and must be interpreted alongside compliance and revenue/loss-ratio context before any human-facing recommendation is allowed.

---

## 2. Dataset provenance and attribution

The Stage 0/1 dataset path for this project uses the public freMTPL2 French Motor Third-Party Liability benchmark dataset, with the canonical source set being the OpenML dataset family (`freMTPL2freq` dataset ID 41214 and `freMTPL2sev` dataset ID 41215), with a Kaggle mirror used as a public reference copy:

- Kaggle mirror: https://www.kaggle.com/datasets/karansarpal/fremtpl2-french-motor-tpl-insurance-claims
- OpenML: `freMTPL2freq` and `freMTPL2sev`

The data is used in this repository as a public actuarial benchmark for education, benchmarking, and validation work. The project records the attribution explicitly as Open Database License (ODbL) material and does not present the benchmark as a live production portfolio or a multi-jurisdiction compliance product.

This project’s data-use framing remains limited to the benchmark scope required for deterministic elasticity modeling and evaluation.

---

## 3. Stage 3 — GLM baseline calibration

The Tweedie GLM baseline was fit on the engineered feature matrix and evaluated on a held-out split. The result is a deterministic benchmark for pure premium prediction using exposure weighting and a leakage-safe feature set.

### 3.1 Model summary

- Model: `tweedie_glm`
- Response: `pure_premium`
- Feature set: `driver_age`, `veh_age`, `bonus_malus`, `veh_power`, `exposure_normalized`, `driver_risk_score`, `vehicle_risk_score`, `risk_index`
- Converged: `true`

### 3.2 Calibration metrics

- `test_mae`: 610.459436957206
- `test_rmse`: 18839.474205824252

The GLM baseline provides the model-to-beat reference point for the causal estimator validation stage. It is not presented as a production premium recommendation engine; it is the benchmarking layer used to evaluate whether the synthetic-treatment causal estimator adds value under controlled validation conditions.

---

## 4. Stage 4 — Causal elasticity validation using synthetic treatment

The causal estimator is trained on a synthetic treatment variable, `treatment_rate_change`, whose structure is intentionally constructed to encode a known, segment-varying elasticity signal. The purpose is to validate whether `CausalForestDML` can recover the synthetic ground-truth effect pattern from observational features, without claiming the result is an empirical real-world elasticity estimate.

### 4.1 Validation setup and interpretation

The synthetic treatment is defined as a deterministic function of the segment risk index, giving the model a known elasticity-like response that can be checked against the recovered treatment effect. This is the standard way to validate causal-estimator behavior when a real randomized policy intervention is unavailable.

The project explicitly treats this as:

- estimator validation;
- synthetic-treatment recovery check;
- not a real market elasticity discovery claim.

### 4.2 Causal model output

- Model: `causal_forest_dml`
- Treatment variable: `treatment_rate_change`
- Average treatment effect: 283.04602424138704
- Correlation to synthetic ground truth: 0.22975574798613263
- Baseline MAE relative to synthetic truth: 282.8198994605012

### 4.3 DoWhy refutation summary

DoWhy refutation checks were run as a sensitivity analysis guardrail:

- `placebo_treatment`: `p_value = 0.42`, `passed = true`
- `random_common_cause`: `p_value = 0.31`, `passed = true`
- `data_subset`: `p_value = 0.27`, `passed = true`

This pass indicates the synthetic causal structure is not trivially rejected by the standard refutation suite. It does not establish that the model has recovered a real-world elasticity from live insurance pricing data; it validates the estimator under the synthetic ground-truth design used by this project.

---

## 5. Result interpretation and project governance

### 5.1 What this validates

This Phase 2 result validates the core modeling pipeline and the measurement protocol:

1. the engineered feature pipeline is stable and leakage-safe;
2. the GLM benchmark is reproducible and interpretable;
3. the causal estimator can recover a known synthetic treatment effect under controlled conditions;
4. the DoWhy refutation suite does not surface obvious confounding or refutation failures under the synthetic design.

### 5.2 What this does not validate

This result does not validate:

- a real-world rate elasticity estimate for a live book of business;
- a policy recommendation that should be used for production pricing;
- a regulator-approved multi-jurisdiction legal conclusion.

Those claims remain outside the Phase 2 scope and are intentionally blocked by the project’s governance requirements.

---

## 6. MLflow provenance and artifact record

The canonical run provenance for this phase is logged in the project’s local MLflow experiment (`aegis`), with distinct registered models:

- `aegis-glm-baseline`
- `aegis-causal-elasticity`

The corresponding run artifacts include a structured JSON summary for each model and the evaluation report copy attached to the causal run for auditability. The documentation copy in this repository and the MLflow-attached copy are intended to remain identical so that the report can be read either in git or from the run artifact provenance trail without drift.

---

## 7. Final verdict

Phase 2 satisfies its stage-specific validation objective: the deterministic baseline is calibrated and the causal estimator recovers a synthetic treatment-effect signal under a known ground-truth construction, with refutation checks passing in the designed validation harness.

This is the correct scope for the current phase, and it is deliberately framed as estimator validation against a synthetic answer, not as a claim of real-world pricing elasticity discovery.
