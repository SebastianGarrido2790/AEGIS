# Specification: P2-F04 Semi-Synthetic DGP & Training Contract
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F04` (`004_semi-synthetic-dgp`)  
**Governing Decisions:** ADR-013 (benchmark design, truth isolation, DGP self-tests), ADR-015 (training table GX contract, tiered reproducibility), P2-D2 (three physically separate artifacts), P2-D2a (logistic renewal, unit, truth, ε conversion, bound), P2-D2b (observed confounding dial, overlap, heterogeneity, homogeneity), P2-D8 (training table GX suite and DAG ordering), INV-3 (data contracts), INV-4 / INV-5 (deterministic arithmetic, no LLM, no price adjustments)  
**SDD Depth:** Full (`spec.md` → approval → `plan.md` → `tasks.md`)  
**Status:** APPROVED — October 26, 2026 (OD-1…OD-4 as recommended)  
**Date:** October 26, 2026  

---

## 1. Context and Problem

The French Motor Personal Line dataset (`freMTPLfreq.csv` / `freMTPLsev.csv`) contains observational rating factors, policy exposures, and historical claim occurrences, but **does not contain policyholder premium, price changes, or renewal/lapse outcomes** (R9). To evaluate the causal machine learning estimator (`CausalForestDML`, Stage 8) against ground truth, AEGIS constructs a semi-synthetic data-generating process (DGP) where:
1. Real observational policyholder and vehicle risk covariates ($X$) are preserved.
2. Treatment (relative price change $\Delta \in [-0.15, +0.15]$) is assigned with realistic observed confounding via a risk-based pricing rule plus residual noise.
3. Renewal outcome ($Y \in \{0, 1\}$) is generated via a known logistic probability model $P(\text{renew} \mid X, \Delta) = \sigma(a(X) + b(X) \cdot \Delta)$.
4. True policyholder response metrics ($\tau_i$, $q_{0i}$, $\epsilon_i$) are mathematically defined in closed form at $\Delta = 0$.

Under **ADR-013** and **P2-D2**, the evaluation benchmark is valid if and only if the causal estimator has zero access to ground truth. Furthermore:
- Training table ingestion must be structurally guarded by a Great Expectations (GX) data contract that rejects any truth columns (ADR-015, P2-D8).
- The causal estimator codebase must be physically prevented from importing the truth sidecar loader.
- The DGP must satisfy five automated self-tests ensuring realistic overlap, informative heterogeneity, between-segment dispersion, within-segment homogeneity, and non-zero volume bounds before downstream training may commence.

---

## 2. Objective and Expected Outcome

Deliver a reproducible, deterministic semi-synthetic data generation pipeline, validation loader, self-test suite, and Great Expectations training data contract:
1. Generate synthetic treatment $\Delta_i \in [-0.15, +0.15]$ and binary renewal $Y_i \sim \text{Bernoulli}(P_i)$ for all 413,169 policies.
2. Produce three physically separate artifacts:
   - **Estimator Training Table:** `PolicyID`, pre-treatment covariates $X$ (forest view), treatment $T$, outcome $Y$, and `segment_id`. Zero truth columns.
   - **Truth Sidecar Table:** `PolicyID`, baseline renewal $q_{0i}$, treatment effect $\tau_i$, elasticity $\epsilon_i$, and `segment_id`. Persisted in an isolated path inaccessible to estimators.
   - **DGP Manifest:** JSON file capturing parameter values, random seeds, input data hash, self-test metrics, and config fingerprint.
3. Enforce five automated DGP self-tests:
   - *Estimand gap test:* per-segment divergence between closed-form truth $\tau_i$ and the estimator's average slope target over $[-0.15, 0.15]$ is within stated tolerance ($\le 0.02$).
   - *Overlap test:* residual variance $\text{Var}(T \mid X) \ge \text{floor}$ ($0.001$).
   - *Heterogeneity & dispersion test:* between-segment standard deviation of $\tau_{\text{seg}} \ge 0.01$.
   - *Homogeneity test:* within-segment Pearson correlation between $\epsilon_i$ and GLM fitted loss cost $\le 0.05$ across all 35 segments.
   - *Bound guarantee test:* $\max_i |\epsilon_i| \cdot 0.15 < 1.0$ so policy volume never floors at zero.
4. Implement and gate the training data contract (`data_contracts/elasticity_training_suite.json`) enforcing prohibited truth columns, valid corridor ranges, and binary outcome types.
5. Falsification: prove dynamically that (a) injecting a truth column (`tau_ground_truth`) fails the GX contract loudly, and (b) importing `truth_loader` from within `aegis.pipelines.causal` triggers a lint/test boundary failure.

---

## 3. Scope

### In scope
- Deterministic DGP implementation in `src/aegis/pipelines/synthetic_dgp.py` driven exclusively by `params.yaml`.
- Ground truth loader in `src/aegis/pipelines/truth_loader.py` with typed Pydantic validation.
- Unit: all 413,169 policies with equal weight; observed exposure excluded from DGP covariates (A1, R12).
- Risk-based treatment assignment with configurable observed confounding dial $\gamma \in [0, 1]$ and noise scale $\sigma_T$.
- Bounded elasticity design guaranteeing $\max_i |\epsilon_i| \cdot 0.15 < 1.0$.
- Five automated self-tests evaluated before artifacts are persisted.
- Re-basing `data_contracts/elasticity_training_suite.json` for post-DGP schema, with valid and invalid fixtures.
- Structural import-boundary test preventing causal modeling modules from importing the truth loader.
- CLI subcommand `aegis generate-dgp` and DVC stage `generate_dgp`.
- Full unit test suite and falsification fixtures.

### Out of scope
- Causal forest model training and hyperparameter tuning (Stage 8, P2-F06).
- Estimator scale and compute benchmarking (Stage 7, Qualification Q2).
- Hidden confounder sensitivity analysis (Stage 11, P2-F09 / D7).
- Downstream Phase 3 underwriting compliance checks and Phase 6 financial simulations.

---

## 4. Functional Requirements (EARS)

**FR-1 (unit and input data).** When generating semi-synthetic outcomes, the generator shall ingest all 413,169 policies from the Stage 2 forest view (`data/processed/features_forest.csv`) joined with `segment_id` from Stage 4 (`data/processed/policy_segments.csv`) and fitted loss cost from Stage 3 (`data/processed/glm_fitted_loss_cost.csv`). Each policy shall carry equal weight (weight = 1.0); observed policy exposure shall be strictly excluded from the pre-treatment covariates $X$ and DGP equations (A1, R12).

**FR-2 (risk-based treatment assignment).** The generator shall assign treatment $\Delta_i \in [-0.15, +0.15]$ using a confounding function driven by observable risk factors plus Gaussian residual noise:
$$\Delta_i = \text{clip}\left(\mu_T(X_i) + \sigma_T \cdot \eta_i, -0.15, +0.15\right)$$
where $\eta_i \sim \mathcal{N}(0, 1)$, $\sigma_T$ is the residual standard deviation, and $\mu_T(X_i) = \gamma \cdot \text{RiskSignal}(X_i)$, governed by the single confounding dial $\gamma \in [0.0, 1.0]$ configured in `params.yaml`.

**FR-3 (logistic renewal outcome).** The generator shall compute individual renewal probability using the logistic model:
$$P(\text{renew}_i \mid X_i, \Delta_i) = \sigma\left(a(X_i) + b(X_i) \cdot \Delta_i\right) = \frac{1}{1 + \exp\left(-(a(X_i) + b(X_i) \cdot \Delta_i)\right)}$$
and sample binary renewal indicator $Y_i \sim \text{Bernoulli}(P_i)$ using the pinned seed `tier1_ml.synthetic_dgp.random_seed`.

**FR-4 (ground truth definitions).** For each policy $i$, the generator shall compute the exact ground truth values at $\Delta = 0$:
- Baseline renewal probability: $q_{0i} = \sigma(a(X_i))$
- Treatment effect (local derivative): $\tau_i = \left.\frac{\partial P}{\partial \Delta}\right|_{\Delta = 0} = q_{0i}(1 - q_{0i}) \cdot b(X_i)$
- Point elasticity: $\epsilon_i = \frac{\tau_i}{q_{0i}} = (1 - q_{0i}) \cdot b(X_i)$

**FR-5 (bound guarantee).** The generator shall construct $b(X)$ such that $|b(X_i)| \le b_{\max}$ and $\max_i |\epsilon_i| \cdot 0.15 < 1.0$, guaranteeing that under the maximum price increase (+15%), projected renewal volume never floors at zero.

**FR-6 (heterogeneity design).** The generator shall formulate $a(X)$ and $b(X)$ using at most 3–4 interpretable, banded rating factors (e.g. driver age band, car age band, density band) with at least one interaction. No raw unbinned heavy-tailed variables shall enter $a(X)$ or $b(X)$ (P2-D2b).

**FR-7 (self-test: estimand gap).** The generator shall numerically integrate the average slope over the treatment corridor:
$$\bar{\tau}_i = \frac{P(\text{renew} \mid X_i, +0.15) - P(\text{renew} \mid X_i, -0.15)}{0.30}$$
and verify that across all 35 actuarial segments, the segment-aggregated estimand gap:
$$\text{Gap}_{\text{seg}} = \left| \frac{\sum_{i \in \text{seg}} \bar{\tau}_i}{\sum_{i \in \text{seg}} q_{0i}} - \frac{\sum_{i \in \text{seg}} \tau_i}{\sum_{i \in \text{seg}} q_{0i}} \right| \le \text{gap\_tolerance}$$
where `gap_tolerance` is read from `params.yaml` (default: 0.02).

**FR-8 (self-test: overlap).** The generator shall verify that the residual variance of treatment given covariates $\text{Var}(T \mid X) \ge \text{overlap\_floor\_var}$ (default: 0.001), ensuring adequate overlap across the treatment support.

**FR-9 (self-test: dispersion).** The generator shall verify that the between-segment standard deviation of ground truth $\tau_{\text{seg}} = \sum_{i \in \text{seg}} \tau_i / N_{\text{seg}}$ satisfies $\text{Std}(\tau_{\text{seg}}) \ge \text{dispersion\_min\_std}$ (default: 0.01), confirming meaningful heterogeneity across segments.

**FR-10 (self-test: within-segment homogeneity).** The generator shall compute the Pearson correlation between true elasticity $\epsilon_i$ and expected loss cost within each of the 35 segments, verifying that $\max_{\text{seg}} |\rho(\epsilon_i, \text{LossCost}_i)| \le 0.05$ (ADR-011, P2-D2b).

**FR-11 (self-test: bound verification).** The generator shall assert that $\max_i |\epsilon_i| \cdot 0.15 < 1.0$ across all 413,169 generated policies.

**FR-12 (physical artifact isolation).** The pipeline shall persist three physically distinct files:
1. `data/processed/elasticity_training.csv`: `PolicyID`, pre-treatment covariates $X$, treatment $T$, outcome $Y$, and `segment_id`.
2. `data/processed/truth_sidecar.csv`: `PolicyID`, $\tau_i$, $q_{0i}$, $\epsilon_i$, and `segment_id`.
3. `data/processed/dgp_manifest.json`: seed, parameters, input data hashes, and self-test verification metrics.

**FR-13 (training data contract enforcement).** The training table shall be validated against `data_contracts/elasticity_training_suite.json`. The contract shall enforce:
- Non-null `PolicyID`, pre-treatment features, $T$, and $Y$.
- Range check: $T \in [-0.15, +0.15]$.
- Domain check: $Y \in \{0, 1\}$.
- Prohibition of post-treatment / truth columns: `ExpectNoPostTreatmentLeakage` strictly prohibiting `tau`, `epsilon`, `q0`, `true_effect`, `tau_ground_truth`, and `fitted_loss_cost`.

**FR-14 (import boundary).** A test shall assert that no module within `src/aegis/pipelines/causal/` (or any estimator module) imports from `aegis.pipelines.truth_loader`.

---

## 5. Data Contracts

### 5.1 Input Contracts
- `data/processed/features_forest.csv`: Stage 2 feature view (413,169 rows, 27 columns).
- `data/processed/policy_segments.csv`: Stage 4 segment assignments (`PolicyID`, `segment_id`).
- `data/processed/glm_fitted_loss_cost.csv`: Stage 3 GLM fitted values (`PolicyID`, `fitted_loss_cost`).
- `params.yaml`: `tier1_ml.synthetic_dgp` parameter block.

### 5.2 Output Contracts

#### 1. Estimator Training Table (`elasticity_training.csv`)
413,169 rows, strictly pre-treatment covariates + treatment + outcome:
- `PolicyID`: `int` (unique)
- Pre-treatment features $X$ (25 columns from `features_forest.csv`, excluding `exposure_capped` and `PolicyID`):
  - `driver_age_clipped`, `car_age_clipped`, `log_density`, `driver_age_band_ord`, `car_age_band_ord`, `power_ord`
  - 7 one-hot brand indicators
  - 2 one-hot gas indicators
  - 10 one-hot region indicators
- `treatment_rate_change`: `float` in $[-0.15, +0.15]$
- `renewed`: `int` in $\{0, 1\}$
- `segment_id`: `str` (35 valid segments)

#### 2. Truth Sidecar Table (`truth_sidecar.csv`)
413,169 rows, validation-only artifact:
- `PolicyID`: `int`
- `baseline_renewal_q0`: `float` in $[0.0, 1.0]$
- `true_tau`: `float`
- `true_epsilon`: `float`
- `segment_id`: `str`

#### 3. DGP Manifest (`dgp_manifest.json` / `DGPManifest`)
Typed metadata contract:
- `random_seed`: `int`
- `sample_size`: `int` (413,169)
- `corridor_width`: `float` (0.15)
- `confounding_dial`: `float`
- `residual_std`: `float`
- `self_tests`: `dict[str, SelfTestResult]`
- `input_hashes`: `dict[str, str]` (SHA-256)
- `config_fingerprint`: `str` (SHA-256)

---

## 6. Edge Cases

| Case | Required Behavior |
|---|---|
| Treatment value clips at $[-0.15, +0.15]$ | Clipping is mathematically permitted and expected at extreme tails; contract checks ensure no value exceeds boundary. |
| Extreme covariate values | Continuous covariates are already clipped in Stage 2 (`driver_age_clipped`, `car_age_clipped`); $a(X)$ and $b(X)$ use stable banded terms. |
| Self-test failure (e.g. estimand gap > tolerance) | Generator raises `DGPVerificationError` and halts before writing training artifacts. |
| Injected truth column in training table | Great Expectations validation raises `DataContractError` and halts pipeline (Falsification). |
| Estimator attempts to import truth loader | Import-boundary test fails in CI (Falsification). |
| Repeated execution | Byte-identical CSV and JSON files under fixed seed (FR-12, ADR-015). |

---

## 7. Error Handling

Dedicated exceptions in `src/aegis/utils/exceptions.py`:
- `DGPVerificationError` (subclass of `AEGISError`): raised if any of the 5 self-tests fail.
- `DataContractError`: raised by GX runner if training contract fails.
- `TruthIsolationError`: raised if truth columns or truth loader are accessed in forbidden contexts.

Errors propagate; CLI and DVC stage exit non-zero.

---

## 8. Non-Functional Constraints

- **INV-3:** All versioned data validated against formal Great Expectations contracts.
- **INV-4 / INV-5:** Strictly deterministic execution, zero LLM calls, zero price adjustments.
- **INV-8:** All modules stay strictly below 1,000 lines.
- **Performance:** DGP generation, self-test verification, and contract evaluation over 413,169 rows execute in $< 60$ seconds.

---

## 9. Acceptance Criteria

- **AC-1:** The DGP generates 413,169 rows with treatments bounded in $[-0.15, +0.15]$ and binary renewal in $\{0, 1\}$.
- **AC-2:** All 5 automated DGP self-tests pass within configured tolerances.
- **AC-3:** Three physically separate artifacts (`elasticity_training.csv`, `truth_sidecar.csv`, `dgp_manifest.json`) are persisted deterministically.
- **AC-4:** The training table passes the Great Expectations suite `elasticity_training_suite.json`.
- **AC-5 (Falsification 1):** Planting a truth column (`tau_ground_truth`) into a test fixture causes the training suite to fail loudly with a schema violation.
- **AC-6 (Falsification 2):** An import of `truth_loader` within a causal estimator module fails the import-boundary check.
- **AC-7:** `truth_loader.py` successfully loads and validates the truth sidecar table against Pydantic schema `TruthSidecarRow`.
- **AC-8:** Re-running the generator yields identical SHA-256 hashes for all three artifacts.

---

## 10. Open Decisions for Operator Approval

### OD-1 — Ground Truth Functions $a(X)$ and $b(X)$ Parameterization (FR-3, FR-4, FR-5, FR-6)
- **Context:** Auto insurance baseline retention typically averages 80%–85% with price elasticity $\epsilon \in [-1.5, -0.4]$. In addition, FR-5 mandates $\max_i |\epsilon_i| \cdot 0.15 < 1.0$ (i.e. $|\epsilon_i| < 6.666$).
- **Recommendation:** Formulate $a(X)$ and $b(X)$ using interpretable rating factors with an interaction:
  - Baseline retention log-odds:
    $$a(X_i) = a_0 + a_1 \cdot \mathbb{I}(\text{DriverAge} \ge 25) + a_2 \cdot \mathbb{I}(\text{CarAge} < 10)$$
    With $a_0 = 1.4$, $a_1 = 0.3$, $a_2 = 0.2$. This produces $a(X_i) \in [1.4, 1.9]$, yielding baseline retention $q_{0i} = \sigma(a(X_i)) \in [0.802, 0.870]$, exactly matching industry standard renewal rates (~83% average).
  - Price sensitivity slope:
    $$b(X_i) = -(b_0 + b_1 \cdot \text{DriverAgeBandOrd} + b_2 \cdot \text{CarAgeBandOrd} + b_{12} \cdot \text{DriverAgeBandOrd} \times \text{CarAgeBandOrd})$$
    With $b_0 = 3.5$, $b_1 = -0.4$, $b_2 = 0.3$, $b_{12} = -0.1$.
    Since $\epsilon_i = (1 - q_{0i}) \cdot b(X_i)$, with $1 - q_{0i} \approx 0.13 - 0.20$ and $b(X_i) \in [-4.5, -2.0]$, the resulting elasticity $\epsilon_i$ spans $[-0.90, -0.35]$, with $\max |\epsilon_i| \approx 0.90$.
    Thus $\max |\epsilon_i| \cdot 0.15 \approx 0.135 \ll 1.0$, completely satisfying the bound guarantee with generous safety margins.
- **Alternative:** Continuous log-density entering $b(X)$. *Trade-off:* introduces intra-segment variation that slightly degrades within-segment homogeneity. Recommendation maintains clean segment-aligned structure.

### OD-2 — Confounding Assignment Rule $\mu_T(X)$ (FR-2)
- **Context:** P2-D2b mandates observed confounding via a risk-based rate-change rule plus residual noise, governed by a single confounding dial $\gamma \in [0, 1]$.
- **Recommendation:** Define normalized risk score from normalized GLM fitted loss cost $\tilde{L}_i = \frac{\text{LossCost}_i - \text{median}}{\text{IQR}}$ (or driver age / vehicle power factors):
  $$\mu_T(X_i) = \gamma \cdot 0.08 \cdot \tanh(\tilde{L}_i)$$
  $$\Delta_i = \text{clip}\left(\mu_T(X_i) + \sigma_T \cdot \eta_i, -0.15, +0.15\right)$$
  with default $\gamma = 0.5$, $\sigma_T = 0.05$. This guarantees substantial overlap across $[-0.15, +0.15]$ while imparting genuine risk-driven selection bias.

### OD-3 — Automated Self-Test Tolerances (FR-7, FR-8, FR-9, FR-10)
- **Recommendation:**
  - Estimand gap tolerance: $\le 0.02$ (max per-segment difference between local derivative $\tau$ and corridor average slope).
  - Overlap residual variance floor: $\text{Var}(T \mid X) \ge 0.001$ (residual standard deviation $\ge 0.031$).
  - Between-segment $\tau$ dispersion: $\text{Std}(\tau_{\text{seg}}) \ge 0.010$.
  - Within-segment homogeneity: $\max_{\text{seg}} |\rho(\epsilon_i, \text{LossCost}_i)| \le 0.05$.
  - Elasticity bound: $\max_i |\epsilon_i| \cdot 0.15 < 1.0$.

### OD-4 — Training Table Feature Set & Column Names (FR-12, FR-13)
- **Recommendation:**
  - Feature columns: `PolicyID`, 25 pre-treatment feature columns from `features_forest.csv` (all columns except `PolicyID` and `exposure_capped`), `treatment_rate_change`, `renewed`, and `segment_id`.
  - Column names: `treatment_rate_change` for treatment $T$, and `renewed` for binary outcome $Y$. This aligns with existing naming conventions in `data_contracts/fixtures`.

---

## 11. Definition of Done

1. This specification is reviewed and approved by the operator.
2. `plan.md` and `tasks.md` are drafted and approved.
3. `src/aegis/pipelines/synthetic_dgp.py` and `src/aegis/pipelines/truth_loader.py` implemented.
4. Five automated DGP self-tests pass and are verified via test suite.
5. `data_contracts/elasticity_training_suite.json` re-based and passing.
6. Falsification tests (truth column injection, import-boundary check) pass.
7. DVC stage `generate_dgp` reproduces cleanly.
8. `pytest`, `ruff`, `pyright`, and module-size check pass.
9. Gate 6 evidence recorded in `stage_evidence.md`.

---

## 12. Traceability

| Requirement | Source |
|---|---|
| FR-1 | P2-D2a (all 413,169 policies, equal weight, exposure excluded) |
| FR-2 | P2-D2b (observed confounding dial, risk-based rule, overlap) |
| FR-3 | P2-D2a (logistic renewal probability) |
| FR-4 | P2-D2a (closed-form truth at $\Delta = 0$) |
| FR-5 | P2-D2a (bound $\max |\epsilon| \cdot 0.15 < 1.0$) |
| FR-6 | P2-D2b (heterogeneity via 3-4 banded covariates with interaction) |
| FR-7 | P2-D2a (estimand gap self-test) |
| FR-8, FR-9, FR-10 | P2-D2b (overlap, dispersion, within-segment homogeneity self-tests) |
| FR-11 | P2-D2a (bound guarantee self-test) |
| FR-12 | P2-D2, ADR-013 (three physically separate artifacts) |
| FR-13 | P2-D8, ADR-015, INV-3 (training data contract) |
| FR-14 | ADR-013 (import-boundary enforcement) |
