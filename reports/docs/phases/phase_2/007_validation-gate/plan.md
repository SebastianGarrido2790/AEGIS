# Plan: P2-F07 Validation Harness & Q3 Threshold Calibration
## Phase 2 — Stage 9

**Spec:** APPROVED October 27, 2026 (OD-1 … OD-5 as recommended)  
**Status:** APPROVED — October 27, 2026  
**Implementation Status:** COMPLETE — Gate 9 PASSED (2026-10-27)
**Date:** October 27, 2026

---

## 1. Implementation Strategy

Stage 9 is structured into five pure or low-coupling units:

1. **Typed Schemas (`src/aegis/schemas/validation.py`)**:
   Pydantic contracts for metrics, per-segment validation rows, control run results, separation gaps, and the frozen thresholds block.
2. **Metric Engine (`src/aegis/pipelines/validation_harness.py`)**:
   Deterministic calculation of $m_1, m_{1\epsilon}, m_{1b}, m_2, m_{3\tau}, m_{3\epsilon}, m_4, m_5$ against ground truth.
3. **Controls Harness**:
   - $c_1$: naive unadjusted within-segment OLS regression of $Y$ on $T$.
   - $c_2$: shuffled effect predictions across segments.
   - $c_3$: zero-effect DGP ($b(X) = 0$; false-positive rate at $\alpha=0.05$ bounded).
   - Positive control: randomized treatment DGP (`confounding_dial = 0.0`) using the full frozen estimator configuration; a reduced 100k/48-tree run recovered only $m_1=0.4013$ and was rejected.
4. **Q3 Threshold Calibration**:
   Executes pilot seeds $\{42, 101, 202\}$; evaluates estimator vs. negative controls; verifies minimum separation margins $\Delta_{\min}$; generates `q3_threshold_calibration_report.md` and freezes thresholds into `params.yaml`.
5. **Config & CLI Integration**:
   Adds `ValidationThresholdsConfig` to `schema.py`; exposes CLI subcommand `calibrate-q3` / `validate-causal`; updates `test_truth_isolation.py` to authorize `validation_harness.py`.

---

## 2. Component Architecture and Interfaces

```python
# src/aegis/schemas/validation.py

class ValidationMetrics(BaseModel):
    m1_tau_spearman: float
    m1_epsilon_spearman: float
    m1b_policy_spearman: float
    m2_mean_abs_bias: float
    m3_tau_coverage: float
    m3_epsilon_coverage: float
    m4_self_consistency: float
    m5_sign_agreement: float

class ValidationThresholds(BaseModel):
    m1_tau_spearman_floor: float
    m1_tau_spearman_target: float
    m1_epsilon_spearman_floor: float
    m1_epsilon_spearman_target: float
    m1b_policy_spearman_floor: float
    m1b_policy_spearman_target: float
    m2_mean_abs_bias_ceiling: float
    m2_mean_abs_bias_target: float
    m3_tau_coverage_floor: float
    m3_epsilon_coverage_floor: float
    m4_self_consistency_floor: float
    m5_sign_agreement_floor: float

# src/aegis/pipelines/validation_harness.py

def compute_validation_metrics(
    segment_estimates: SegmentElasticityTable,
    truth_sidecar: pd.DataFrame,
    held_out_test: pd.DataFrame,
    policy_effects: np.ndarray | None = None,
) -> ValidationMetricReport:
    """Computes m1 through m5 comparing segment estimates to truth."""

def run_unadjusted_control_c1(
    test_df: pd.DataFrame,
    truth_sidecar: pd.DataFrame,
) -> ControlRunResult:
    """Evaluates within-segment naive OLS on confounded data."""

def run_shuffled_control_c2(
    segment_estimates: SegmentElasticityTable,
    truth_sidecar: pd.DataFrame,
    held_out_test: pd.DataFrame,
    random_state: int = 42,
) -> ControlRunResult:
    """Evaluates permuted segment estimates."""

def run_zero_effect_control_c3(
    config: AEGISConfig,
    random_state: int = 42,
) -> ControlRunResult:
    """Evaluates false-positive rate and recovery under b(X) = 0."""

def run_positive_control(
    config: AEGISConfig,
    random_state: int = 42,
) -> ControlRunResult:
    """Evaluates causal estimator recovery on unconfounded DGP."""

def calibrate_q3_thresholds(
    config: AEGISConfig,
    pilot_seeds: list[int] | None = None,
) -> Q3CalibrationReport:
    """Executes Q3 qualification on pilot seeds and emits recommended thresholds."""
```

---

## 3. Separation Gap and Calibration Math (P2-D6a)

For separation-calibrated metrics where higher is better ($m_1, m_{1\epsilon}, m_{1b}$):
- $E_{\min} = \min_{p \in \text{pilot}} m(p)$
- $C_{\max} = \max_{c \in \{c_1, c_2, c_3\}, p \in \text{pilot}} m(c, p)$
- Gap $\Delta = E_{\min} - C_{\max}$. Must satisfy $\Delta \ge \Delta_{\min}$.
- Floor threshold: $\theta_{\text{floor}} = C_{\max} + \max\left(\Delta_{\min}, \frac{\Delta}{2}\right)$.
- Target threshold: $\theta_{\text{target}} = \theta_{\text{floor}} + 0.5 \times (E_{\text{median}} - \theta_{\text{floor}})$.

For metrics where lower is better ($m_2$ bias):
- $E_{\max} = \max_{p \in \text{pilot}} m_2(p)$
- $C_{\min} = \min_{c \in \{c_1, c_2\}, p \in \text{pilot}} m_2(c, p)$
- Gap $\Delta = C_{\min} - E_{\max}$. Must satisfy $\Delta \ge \Delta_{\min}$.
- Ceiling threshold: $\theta_{\text{ceiling}} = C_{\min} - \max\left(\Delta_{\min}, \frac{\Delta}{2}\right)$.
- Target threshold: $\theta_{\text{target}} = \theta_{\text{ceiling}} - 0.5 \times (\theta_{\text{ceiling}} - E_{\text{median}})$.

Minimum required margins $\Delta_{\min}$:
- $\Delta_{\min}(m_1) = 0.15$
- $\Delta_{\min}(m_{1\epsilon}) = 0.15$
- $\Delta_{\min}(m_{1b}) = 0.15$
- $\Delta_{\min}(m_2) = 0.05$

$m_5$ is a fixed absolute gate at 0.90 (OD-5) and is excluded from separation calibration. `m3` and `m4` use fixed approved floors.

---

## 4. Test Strategy

1. **Unit tests (`tests/unit/test_validation_harness.py`)**:
   - Metric calculation accuracy on synthetic ground-truth and estimates.
   - $c_1$ naive OLS calculates covariance / variance properly.
   - $c_2$ shuffling breaks rank correlation.
   - $c_3$ zero-effect correctly counts significant intervals.
   - Separation gap calculation and midpoint threshold logic.
   - Schema validation and serialization round-trip.
2. **Structural and Isolation Tests**:
   - `tests/unit/test_truth_isolation.py`: `validation_harness.py` is permitted to import `truth_loader.py`; no estimator, api, or feature pipeline module may import it.
   - Module line ceiling check (INV-8): `validation_harness.py` < 1,000 lines.
3. **Q3 Qualification Execution**:
   - Runs pilot seeds $\{42, 101, 202\}$.
   - Emits `reports/docs/phases/phase_2/q3_threshold_calibration_report.md`.
   - Freezes `validation_thresholds` into `params.yaml`.
