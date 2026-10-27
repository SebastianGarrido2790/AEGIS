# Specification: P2-F07 Validation Harness & Q3 Threshold Calibration
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F07` (`007_validation-gate`, Stage 9)  
**Governing Decisions:** ADR-013 (validation metrics, negative/positive controls, threshold calibration, truth isolation), P2-D6 (held-out multi-seed recovery), P2-D6a (Q3 threshold calibration method and separation margin), ADR-005 (config-only tunables), ADR-015 (reproducibility), INV-4 / INV-5 (deterministic arithmetic, no LLM, no price adjustments), INV-7 (auditability), INV-8 (module ceiling).  
**SDD Depth:** Full (`spec.md` → approval → `plan.md` → `tasks.md`)  
**Status:** APPROVED — October 27, 2026 (OD-1 … OD-5 as recommended)  
**Implementation Status:** COMPLETE — Gate 9 PASSED (2026-10-27)  
**Date:** October 27, 2026  
**Amendment:** October 27, 2026 — OD-5 fixes $m_5$ at 0.90 and excludes it from empirical separation after pilot seed 42 showed $c_1=0.9429$ and $c_2=1.0000$.  

---

## 1. Context and Problem

Stage 8 (`P2-F06`) delivered the production `CausalForestDML` pipeline, producing segment-average effect and elasticity estimates with 95% confidence intervals on held-out policies. In compliance with ADR-013 and INV-4, the Stage 8 estimator was strictly isolated from the DGP truth sidecar: it produced no comparison to ground truth, and hardcoded no numeric performance thresholds.

Stage 9 implements the offline validation metric harness, executes required negative and positive controls, calibrates numeric gate thresholds on **pilot seeds** (Qualification Q3, P2-D6a), and freezes them in `params.yaml` ahead of the Stage 10 final-seed validation campaign ($K \ge 5$ disjoint seeds, Exit E1).

The validation harness is the sole authorized consumer of `src/aegis/pipelines/truth_loader.py` (ADR-013). Causal estimators and serving modules remain strictly quarantined.

---

## 2. Objective and Expected Outcome

Deliver a reproducible offline validation engine and Q3 calibration report that:

1. Ingests held-out policy predictions from an estimator run alongside the isolated truth sidecar (`truth_sidecar.csv`).
2. Calculates the six approved validation metrics ($m_1, m_{1b}, m_2, m_{3\tau}, m_{3\epsilon}, m_4, m_5$) comparing estimates to ground truth.
3. Implements and executes the three mandatory negative controls ($c_1$ unadjusted, $c_2$ shuffled, $c_3$ zero-effect DGP) and proves they fail the harness.
4. Implements and executes the positive control (randomized treatment assignment) and proves it passes cleanly.
5. Executes Qualification Q3 across a family of pilot seeds disjoint from Stage 10 final seeds.
6. Calibrates numeric pass thresholds using the approved separation-gap method (P2-D6a) with an enforced minimum margin, except for the fixed absolute $m_5$ sign gate (OD-5).
7. Publishes `reports/docs/phases/phase_2/q3_threshold_calibration_report.md` and freezes the numeric thresholds into `params.yaml` under `tier1_ml.validation_thresholds`.

---

## 3. Scope

### In scope

- `src/aegis/pipelines/validation_harness.py`: metric calculations ($m_1$–$m_5$), control runners ($c_1, c_2, c_3$, positive control), and Q3 threshold calibration.
- `src/aegis/schemas/validation.py`: typed schemas for `ValidationMetricReport`, `ControlRunResult`, `Q3CalibrationReport`, and `ValidationThresholds`.
- Configuration schema update in `src/aegis/config/schema.py` and `params.yaml` for `tier1_ml.validation_thresholds` and `tier1_ml.pilot_seeds`.
- CLI subcommand `validate-causal` in `src/aegis/pipelines/cli.py` to run validation on an estimator output against truth.
- Update structural test `tests/unit/test_truth_isolation.py` to authorize `validation_harness.py` as the sole offline consumer of `truth_loader.py`.
- Automated test suite in `tests/unit/test_validation_harness.py`.
- Q3 threshold calibration report: `reports/docs/phases/phase_2/q3_threshold_calibration_report.md`.

### Out of scope

- Final multi-seed validation campaign across $K \ge 5$ final seeds and Exit E1 verification (Stage 10).
- Confounder sensitivity sweeps (Stage 11, P2-F08).
- MLflow packaging and registry alias promotion (Stage 12, P2-F09).
- Dynamic threshold adjustment at runtime (thresholds are frozen in config).

---

## 4. Functional Requirements (EARS)

**FR-1 (truth sidecar alignment).** When evaluating a test run, the harness shall load the truth sidecar via `load_truth_sidecar()` and inner-join it with held-out policy predictions by `PolicyID`. A row count mismatch or missing policy raises `ValidationHarnessError`.

**FR-2 (validation metrics).** The harness shall compute the following metrics on held-out policies:
- **$m_1$ (segment $\tau$ rank correlation):** Spearman rank correlation $\rho(\hat{\tau}_{\text{seg}}, \tau_{\text{seg}}^*)$ across the 35 realized segments.
- **$m_{1\epsilon}$ (segment $\epsilon$ rank correlation):** Spearman rank correlation $\rho(\hat{\epsilon}_{\text{seg}}, \epsilon_{\text{seg}}^*)$ across the 35 realized segments.
- **$m_{1b}$ (policy $\tau$ rank correlation):** Spearman rank correlation $\rho(\hat{\tau}_i, \tau_i^*)$ on individual held-out test policies.
- **$m_2$ (mean absolute elasticity bias):** Mean absolute error across segments $\frac{1}{S}\sum_{s=1}^S |\hat{\epsilon}_s - \epsilon_s^*|$.
- **$m_{3\tau}$ (segment $\tau$ interval coverage):** Proportion of segments where $\hat{\tau}_{s,\text{lower}} \le \tau_s^* \le \hat{\tau}_{s,\text{upper}}$.
- **$m_{3\epsilon}$ (segment $\epsilon$ interval coverage):** Proportion of segments where $\hat{\epsilon}_{s,\text{lower}} \le \epsilon_s^* \le \hat{\epsilon}_{s,\text{upper}}$.
- **$m_4$ (interval self-consistency rate):** Proportion of segments where intervals are finite, ordered ($\text{lower} \le \text{upper}$), and contain their point estimates ($\text{lower} \le \hat{\theta} \le \text{upper}$).
- **$m_5$ (directional sign agreement):** Proportion of segments where $\text{sign}(\hat{\tau}_s) == \text{sign}(\tau_s^*)$.

**FR-3 (negative control $c_1$ — unadjusted estimator).** The harness shall implement a naive within-segment univariate OLS estimator of renewal $Y$ on rate change $T$ on confounded data ($\hat{\tau}_{\text{unadj},s} = \text{Cov}(Y,T \mid s) / \text{Var}(T \mid s)$). Because treatment is confounded by risk, $c_1$ shall fail $m_1$ and $m_2$ relative to the causal estimator.

**FR-4 (negative control $c_2$ — shuffled estimator).** The harness shall permute predicted segment effects randomly using a pinned seed. The resulting estimates shall fail rank correlation ($m_1 \approx 0$).

**FR-5 (negative control $c_3$ — zero-effect DGP).** The harness shall evaluate a DGP variant where $b(X) = 0$ (so $\tau_i = \epsilon_i = 0$ for all $i$). It shall assert that the proportion of segments falsely declared significant at $\alpha=0.05$ remains near nominal ($p \le 0.15$), and that $c_3$ fails non-zero recovery criteria.

**FR-6 (positive control — randomized treatment).** The harness shall evaluate an unconfounded DGP variant (`confounding_dial = 0.0`). The causal estimator shall pass cleanly ($m_1 \ge 0.90$).

**FR-7 (Q3 threshold calibration on pilot seeds).** The harness shall execute the estimator and negative controls across configured pilot seeds (disjoint from final evaluation seeds). For each separation-calibrated metric ($m_1, m_{1\epsilon}, m_{1b}, m_2$), it shall measure the empirical separation gap between the lower tail of the estimator distribution and the upper tail of the control distribution. If the gap satisfies the minimum margin $\Delta_{\min}$, it sets the floor and target thresholds inside the gap. If any separation-calibrated metric lacks separation, it raises `ValidationHarnessError` and halts (P2-D6a). The $m_5$ sign gate is fixed at 0.90 and excluded from separation calibration (OD-5).

**FR-8 (report generation and config freezing).** Upon successful Q3 calibration, the harness shall write `reports/docs/phases/phase_2/q3_threshold_calibration_report.md` detailing distributions, gaps, and recommended thresholds, and output the exact YAML block to freeze into `params.yaml`.

---

## 5. Open Decisions and Recommendations

### OD-1: Metric Scope for $m_1$ and $m_3$
- **Options:** (A) Report and threshold $\tau$ metrics only; (B) Report and threshold $\epsilon$ metrics only; (C) Threshold both $\tau$ and $\epsilon$ ($m_{1\tau}, m_{1\epsilon}, m_{3\tau}, m_{3\epsilon}$).
- **Recommendation:** **Option C.** In Stage 8, $\epsilon$ is derived from $\tau$ scaled by estimated $q_0$. Validating both ensures that $q_0$ estimation introduces no rank distortion or undercoverage.

### OD-2: Concrete Mathematical Formulation of Negative Control $c_1$
- **Options:** (A) Segment-level univariate OLS regression slope of $Y$ on $T$; (B) Binary difference in means $\bar{Y}_{T>0} - \bar{Y}_{T\le 0}$ normalized by treatment spread.
- **Recommendation:** **Option A.** Univariate OLS within segment is the natural econometric difference-in-means on continuous treatment and directly exhibits omitted-variable bias.

### OD-3: Pilot Seed Family vs. Final Seed Family
- **Options:** (A) 3 pilot seeds $\{42, 101, 202\}$, disjoint from 5 final seeds $\{501, 502, 503, 504, 505\}$; (B) 1 pilot seed $\{42\}$; (C) 5 pilot seeds.
- **Recommendation:** **Option A.** 3 pilot seeds provide empirical variance across seeds for threshold placement while keeping total Q3 calibration runtime around 30 minutes, cleanly separated from the Stage 10 final seeds.

### OD-4: Quantitative Threshold Placement and Minimum Margin Rule
- **Options:** (A) Midpoint rule with minimum margin: $\theta = C_{\max} + \max(\Delta_{\min}, \frac{E_{\min} - C_{\max}}{2})$, requiring $E_{\min} - C_{\max} \ge \Delta_{\min}$; (B) Fixed buffer below $E_{\min}$.
- **Recommendation:** **Option A.** Midpoint placement guarantees maximum safety buffer from both control false-positives and estimator worst-seed variations. Minimum margins: $\Delta_{\min} = 0.15$ for $m_1$/$m_{1\epsilon}$/$m_{1b}$ Spearman and $0.05$ for $m_2$ bias.

### OD-5: Handling the $m_5$ Sign Gate
- **Options:** (A) Fix $m_5$ at 0.90 and exclude it from empirical separation; (B) redesign $c_2$ as a signed permutation and redefine $m_5$; (C) halt and revisit ADR-013.
- **Recommendation:** **Option A.** Pilot seed 42 produced $m_5=1.0000$ for the estimator and $c_2$, and $0.9429$ for $c_1$. All true effects are negative, so shuffling preserves sign; the metric cannot separate empirically. The 0.90 absolute gate remains enforced, while $m_1$, $m_{1\epsilon}$, $m_{1b}$, and $m_2$ carry separation calibration.

---

## 6. Acceptance Criteria (Gate 9 Hard Close)

- **AC-1:** Validation harness correctly computes $m_1, m_{1b}, m_2, m_{3\tau}, m_{3\epsilon}, m_4, m_5$ against ground truth.
- **AC-2:** Negative control $c_1$ (unadjusted OLS) demonstrably fails the validation metrics.
- **AC-3:** Negative control $c_2$ (shuffled estimator) demonstrably fails $m_1$ ($\rho \approx 0$).
- **AC-4:** Negative control $c_3$ (zero-effect DGP) false-positive rate remains $\le 0.15$ at $\alpha=0.05$, and fails non-zero recovery.
- **AC-5:** Positive control (randomized treatment) passes all metrics cleanly.
- **AC-6:** Q3 calibration executes across the 3 pilot seeds; $m_1$, $m_{1\epsilon}$, $m_{1b}$, and $m_2$ demonstrate statistical separation exceeding $\Delta_{\min}$; $m_5$ remains fixed at 0.90 under OD-5.
- **AC-7:** `reports/docs/phases/phase_2/q3_threshold_calibration_report.md` is published; thresholds are frozen in `params.yaml`.
- **AC-8:** `tests/unit/test_truth_isolation.py` passes; `src/aegis/pipelines/validation_harness.py` is the only authorized consumer of `truth_loader.py`.

---

## 7. Traceability Matrix

| Requirement | Governing Rule / ADR | Implementation Target |
|---|---|---|
| FR-1, AC-1 | ADR-013, INV-3 | `src/aegis/pipelines/validation_harness.py` |
| FR-2, AC-1 | ADR-013, P2-D6 | `src/aegis/pipelines/validation_harness.py` |
| FR-3, AC-2 | ADR-013, P2-D6 | `src/aegis/pipelines/validation_harness.py` ($c_1$) |
| FR-4, AC-3 | ADR-013, P2-D6 | `src/aegis/pipelines/validation_harness.py` ($c_2$) |
| FR-5, AC-4 | ADR-013, P2-D6 | `src/aegis/pipelines/validation_harness.py` ($c_3$) |
| FR-6, AC-5 | ADR-013, P2-D6 | `src/aegis/pipelines/synthetic_dgp.py`, `validation_harness.py` |
| FR-7, AC-6 | ADR-013, P2-D6a | `src/aegis/pipelines/validation_harness.py` |
| FR-8, AC-7 | ADR-005, ADR-013 | `params.yaml`, `q3_threshold_calibration_report.md` |
| AC-8 | ADR-013, INV-4 | `tests/unit/test_truth_isolation.py` |
