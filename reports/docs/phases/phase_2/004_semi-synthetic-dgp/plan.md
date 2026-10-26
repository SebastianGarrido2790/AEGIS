# Technical Plan: P2-F04 Semi-Synthetic DGP & Training Contract
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F04` (`004_semi-synthetic-dgp`)  
**Spec:** `reports/docs/phases/phase_2/004_semi-synthetic-dgp/spec.md` — APPROVED October 26, 2026 (OD-1…OD-4)  
**Status:** Ready for task decomposition  
**Date:** October 26, 2026  

---

## 1. Architecture and Components

Stage 6 sits between the Stage 4 segment assignments / Stage 3 GLM fitted values and the Stage 7 Q2 compute benchmark / Stage 8 causal estimator.

```text
data/processed/features_forest.csv       (Stage 2: 25 pre-treatment features)
data/processed/policy_segments.csv       (Stage 4: segment_id per policy)
data/processed/glm_fitted_loss_cost.csv  (Stage 3: fitted loss cost for confounding)
                 │
                 ▼
      synthetic_dgp (P2-F04)
      ├── Risk confounding: μ_T(X) = γ · 0.08 · tanh(L̃) + σ_T · η
      ├── Treatment clipping: T ∈ [-0.15, +0.15]
      ├── Logistic renewal: P(renew|X, T) = σ(a(X) + b(X) · T)
      ├── Closed-form truth: q_0 = σ(a(X)), τ = q_0(1 - q_0)b(X), ε = (1 - q_0)b(X)
      ├── Five automated self-tests:
      │    ├── Estimand gap (≤ 0.02)
      │    ├── Overlap variance floor (Var(T|X) ≥ 0.001)
      │    ├── Between-segment dispersion (Std(τ_seg) ≥ 0.010)
      │    ├── Within-segment homogeneity (corr(ε, L) ≤ 0.05)
      │    └── Elasticity bound (|ε| · 0.15 < 1.0)
      └── Outputs (Three physically isolated artifacts)
           ├── data/processed/elasticity_training.csv  (Estimator training table; zero truth)
           ├── data/processed/truth_sidecar.csv        (Isolated truth table)
           └── data/processed/dgp_manifest.json        (Metadata + test metrics + hashes)
                 │
                 ▼
      data_contracts (elasticity_training_suite.json)
      └── Prohibits truth columns, enforces T in [-0.15, 0.15], checks Y in {0, 1}
```

### Components

| Component | Responsibility | Location |
|---|---|---|
| Config schema | `synthetic_dgp` parameters, self-test thresholds, paths | `params.yaml`, `src/aegis/config/schema.py` |
| DGP exceptions | Typed self-test and isolation failures | `src/aegis/utils/exceptions.py` |
| Contract models | `SelfTestResult`, `DGPManifest`, `TruthSidecarRow` | `src/aegis/schemas/dgp.py` |
| Synthetic DGP | Generation of treatment, outcomes, truth, and 5 self-tests | `src/aegis/pipelines/synthetic_dgp.py` |
| Truth loader | Isolated loader and validator for truth sidecar | `src/aegis/pipelines/truth_loader.py` |
| Data contract | Re-based `elasticity_training_suite.json` + runner | `data_contracts/`, `src/aegis/pipelines/data_contracts.py` |
| CLI commands | `generate-dgp` and `validate-training-data` subcommands | `src/aegis/pipelines/cli.py` |
| DVC stages | `generate_dgp` and `validate_training_data` | `dvc.yaml` |
| Tests | Unit, determinism, falsification, import boundary | `tests/unit/test_synthetic_dgp.py`, `tests/unit/test_truth_isolation.py`, `tests/unit/test_truth_loader.py` |

---

## 2. Data and Control Flow

1. Read inputs:
   - `features_forest.csv` (413,169 rows, 27 columns; drop `exposure_capped` per A1/R12).
   - `policy_segments.csv` (413,169 rows, `PolicyID`, `segment_id`).
   - `glm_fitted_loss_cost.csv` (413,169 rows, `PolicyID`, `fitted_loss_cost`).
   Validate row count (413,169) and identical `PolicyID` sets.
2. Treatment Assignment (FR-2, OD-2):
   - Standardize GLM fitted loss cost: $\tilde{L}_i = (\text{LossCost}_i - \text{median}) / \text{IQR}$.
   - Confounded mean: $\mu_T(X_i) = \gamma \cdot 0.08 \cdot \tanh(\tilde{L}_i)$.
   - Add noise: $T_i = \text{clip}(\mu_T(X_i) + \sigma_T \cdot \eta_i, -0.15, +0.15)$ with $\eta_i \sim \mathcal{N}(0, 1)$ using pinned seed.
3. Outcome & Truth Generation (FR-3, FR-4, FR-5, OD-1):
   - $a(X_i) = 1.4 + 0.3 \cdot \mathbb{I}(\text{driver\_age\_band\_ord} \ge 1) + 0.2 \cdot \mathbb{I}(\text{car\_age\_band\_ord} \le 1)$.
   - $b(X_i) = -(3.5 - 0.4 \cdot \text{driver\_age\_band\_ord} + 0.3 \cdot \text{car\_age\_band\_ord} - 0.1 \cdot \text{driver\_age\_band\_ord} \times \text{car\_age\_band\_ord})$.
   - Probability: $P_i = \sigma(a(X_i) + b(X_i) \cdot T_i)$.
   - Outcome: $Y_i \sim \text{Bernoulli}(P_i)$.
   - Truth at $T = 0$: $q_{0i} = \sigma(a(X_i))$, $\tau_i = q_{0i}(1 - q_{0i}) \cdot b(X_i)$, $\epsilon_i = (1 - q_{0i}) \cdot b(X_i)$.
4. Execute 5 Automated Self-Tests (FR-7…FR-11):
   - (1) Estimand gap $\le 0.02$ per segment against numerical average slope.
   - (2) Overlap residual variance $\ge 0.001$.
   - (3) Between-segment standard deviation of $\tau \ge 0.010$.
   - (4) Within-segment Pearson correlation $\le 0.05$ across all 35 segments.
   - (5) Bound $\max_i |\epsilon_i| \cdot 0.15 < 1.0$.
   If any self-test fails, raise `DGPVerificationError` and abort.
5. Persist Three Physical Artifacts (FR-12):
   - `elasticity_training.csv`: `PolicyID`, 25 pre-treatment feature columns, `treatment_rate_change`, `renewed`, `segment_id`.
   - `truth_sidecar.csv`: `PolicyID`, `baseline_renewal_q0`, `true_tau`, `true_epsilon`, `segment_id`.
   - `dgp_manifest.json`: seed, parameters, input SHA-256 hashes, test metrics, config fingerprint.
6. Enforce Training Contract (FR-13):
   - Validate `elasticity_training.csv` with `data_contracts/elasticity_training_suite.json`.
7. Import Boundary Enforcement (FR-14):
   - AST / inspection test asserting causal packages cannot import `truth_loader`.

---

## 3. Configuration Surface (`params.yaml`)

```yaml
tier1_ml:
  synthetic_dgp:
    random_seed: 42
    corridor_width: 0.15
    confounding_dial: 0.5
    residual_std: 0.05
    overlap_floor_var: 0.001
    estimand_gap_tolerance: 0.02
    dispersion_min_std: 0.010
    homogeneity_max_corr: 0.05
    elasticity_bound_product: 1.0
    a_coeffs:
      base: 1.4
      driver_age_ge_25: 0.3
      car_age_lt_10: 0.2
    b_coeffs:
      base: 3.5
      driver_age_ord: -0.4
      car_age_ord: 0.3
      interaction: -0.1
  dgp_paths:
    training_table: "data/processed/elasticity_training.csv"
    truth_sidecar: "data/processed/truth_sidecar.csv"
    manifest: "data/processed/dgp_manifest.json"
```

---

## 4. Great Expectations Suite Specification

File: `data_contracts/elasticity_training_suite.json`
Suite name: `elasticity_training_suite`

Expectations:
1. `expect_table_columns_to_match_set`: exact column list (29 columns: `PolicyID`, 25 pre-treatment features, `treatment_rate_change`, `renewed`, `segment_id`).
2. `expect_column_values_to_not_be_null`: for `PolicyID`, all 25 features, `treatment_rate_change`, `renewed`, `segment_id`.
3. `expect_column_values_to_be_unique`: for `PolicyID`.
4. `expect_column_values_to_be_between`: for `treatment_rate_change`, min: -0.15, max: 0.15.
5. `expect_column_values_to_be_in_set`: for `renewed`, set: `[0, 1]`.
6. `expect_no_post_treatment_leakage` (`ExpectNoPostTreatmentLeakage`): prohibited columns:
   `["tau", "epsilon", "q0", "true_effect", "tau_ground_truth", "fitted_loss_cost", "baseline_renewal_q0", "true_tau", "true_epsilon"]`.

---

## 5. Implementation Strategy and Phasing

- Task 1: Configuration surface and schema validation.
- Task 2: Typed models and exception classes.
- Task 3: DGP generator mathematics and 5 self-tests.
- Task 4: Truth loader module.
- Task 5: Great Expectations suite re-basing, fixtures, and runner.
- Task 6: CLI entrypoints and DVC stages.
- Task 7: Falsification fixtures, import-boundary test, and unit tests.
- Task 8: Full pipeline execution, gate validation, and evidence logging.
