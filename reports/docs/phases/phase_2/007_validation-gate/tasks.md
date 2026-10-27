# Tasks: P2-F07 Validation Harness & Q3 Threshold Calibration
## Phase 2 — Stage 9

**Spec:** APPROVED October 27, 2026 (OD-1 … OD-5 as recommended)  
**Plan:** APPROVED October 27, 2026  
**Status:** COMPLETE
**Date:** October 27, 2026  

Execution rule: tasks run strictly in order. Each task starts with a failing test where behavior is testable, ends with verification green on changed files, and leaves the working tree reviewable. No task starts while the previous task is failed or partial.

| Task | Scope | Status |
|---|---|---|
| T1 | Validation schemas and exception types | COMPLETE |
| T2 | Core validation metric engine ($m_1$ through $m_5$) | COMPLETE |
| T3 | Controls engine ($c_1, c_2, c_3$ and positive control) | COMPLETE |
| T4 | Separation gap calculation and Q3 calibration engine | COMPLETE |
| T5 | Config, CLI integration, and truth isolation boundary update | COMPLETE |
| T6 | Q3 pilot-seed calibration execution and threshold freeze | COMPLETE |
| T7 | Documentation, Gate 9 evidence capture, and final checks | COMPLETE |

---

## T1 — Validation schemas and exception types

**Files:** `src/aegis/schemas/validation.py`, `src/aegis/utils/exceptions.py`, `tests/unit/test_validation_harness.py`

**Actions:**
1. Add `ValidationHarnessError` to `src/aegis/utils/exceptions.py`.
2. Create `src/aegis/schemas/validation.py` defining:
   - `ValidationMetrics`: numeric fields for $m_1, m_{1\epsilon}, m_{1b}, m_2, m_{3\tau}, m_{3\epsilon}, m_4, m_5$.
   - `SegmentValidationRow`: per-segment comparison diagnostics.
   - `ValidationMetricReport`: overall run summary.
   - `ControlRunResult`: control diagnostics and failure confirmation.
   - `SeparationGap`: gap metrics ($E_{\min}, C_{\max}, \Delta, \Delta_{\min}$, separated flag).
   - `ValidationThresholds`: frozen floor/target/ceiling thresholds.
   - `Q3CalibrationReport`: full qualification summary.
3. Add unit tests for schema validation, bound invariants, and JSON serialization.

**Verification:**
```powershell
uv run pytest tests/unit/test_validation_harness.py -q
uv run ruff check src/aegis/schemas/validation.py tests/unit/test_validation_harness.py
uv run pyright src/aegis/schemas/validation.py
```

---

## T2 — Core validation metric engine ($m_1$ through $m_5$)

**Files:** `src/aegis/pipelines/validation_harness.py`, `tests/unit/test_validation_harness.py`

**Actions:**
1. Implement `compute_validation_metrics`:
   - Inner-join held-out predictions with truth sidecar by `PolicyID`.
   - Calculate segment-level true effects ($\tau_s^* = \text{mean}(\tau_i \mid s)$, $\epsilon_s^* = \sum \tau_i / \sum q_{0i}$).
   - Compute $m_1$ ($\tau$ Spearman) and $m_{1\epsilon}$ ($\epsilon$ Spearman).
   - Compute $m_{1b}$ (policy-level $\tau$ Spearman on held-out policies).
   - Compute $m_2$ (mean absolute elasticity bias across segments).
   - Compute $m_{3\tau}$ and $m_{3\epsilon}$ (empirical 95% interval coverage).
   - Compute $m_4$ (interval self-consistency rate).
   - Compute $m_5$ (directional sign agreement rate).
2. Add unit tests against synthetic fixtures with known ground truth and edge cases (empty data, mismatched IDs, non-finite values).

**Verification:**
```powershell
uv run pytest tests/unit/test_validation_harness.py -q
uv run ruff check src/aegis/pipelines/validation_harness.py
uv run pyright src/aegis/pipelines/validation_harness.py
```

---

## T3 — Controls engine ($c_1, c_2, c_3$ and positive control)

**Files:** `src/aegis/pipelines/validation_harness.py`, `tests/unit/test_validation_harness.py`

**Actions:**
1. Implement $c_1$ (`run_unadjusted_control_c1`): segment-level univariate OLS of renewal $Y$ on $T$ on confounded data.
2. Implement $c_2$ (`run_shuffled_control_c2`): randomly permute predicted segment effects with a pinned seed.
3. Implement $c_3$ (`run_zero_effect_control_c3`): evaluate zero-effect DGP ($b(X) = 0$), measuring the empirical false-positive rate at $\alpha=0.05$.
4. Implement positive control (`run_positive_control`): evaluate unconfounded DGP (`confounding_dial = 0.0`).
5. Add unit tests verifying that $c_1, c_2, c_3$ trigger control failures and positive control passes.

**Verification:**
```powershell
uv run pytest tests/unit/test_validation_harness.py -q
uv run ruff check src/aegis/pipelines/validation_harness.py
uv run pyright src/aegis/pipelines/validation_harness.py
```

---

## T4 — Separation gap calculation and Q3 calibration engine

**Files:** `src/aegis/pipelines/validation_harness.py`, `tests/unit/test_validation_harness.py`

**Actions:**
1. Implement `compute_separation_gap`:
   - Calculate $E_{\min}, E_{\text{median}}, E_{\max}$, and $C_{\max}$ across pilot seeds and negative controls.
   - Enforce minimum margins: $\Delta_{\min}(m_1) = 0.15, \Delta_{\min}(m_{1b}) = 0.15, \Delta_{\min}(m_2) = 0.05, \Delta_{\min}(m_5) = 0.10$.
   - Fail loud with `ValidationHarnessError` if any metric has no separation.
2. Implement `calibrate_thresholds`:
   - Midpoint placement for floor threshold: $\theta_{\text{floor}} = C_{\max} + \max(\Delta_{\min}, \frac{\Delta}{2})$.
   - Target threshold for median seed: $\theta_{\text{target}} = \theta_{\text{floor}} + 0.5 \times (E_{\text{median}} - \theta_{\text{floor}})$.
   - Ceiling placement for bias $m_2$.
3. Implement report generation emitting `q3_threshold_calibration_report.md`.
4. Add unit tests covering separation gaps, threshold formulas, and fail-loud branch.

**Verification:**
```powershell
uv run pytest tests/unit/test_validation_harness.py -q
uv run ruff check src/aegis/pipelines/validation_harness.py
```

---

## T5 — Config, CLI integration, and truth isolation boundary update

**Files:** `src/aegis/config/schema.py`, `params.yaml`, `src/aegis/pipelines/cli.py`, `tests/unit/test_truth_isolation.py`, `tests/unit/test_cli.py`, `tests/unit/test_config.py`

**Actions:**
1. Add `ValidationThresholdsConfig` to `src/aegis/config/schema.py` and attach to `Tier1MLConfig`.
2. Add `pilot_seeds: [42, 101, 202]` under `tier1_ml` in `params.yaml`.
3. Expose CLI commands `validate-causal` and `calibrate-q3` in `cli.py` with structured output and exit codes.
4. Update `tests/unit/test_truth_isolation.py`: add `validation_harness.py` to the permitted list of offline validation modules consuming `truth_loader.py`.
5. Add CLI and config tests.

**Verification:**
```powershell
uv run pytest tests/unit/test_truth_isolation.py tests/unit/test_config.py tests/unit/test_cli.py -q
uv run pyright
```

---

## T6 — Q3 pilot-seed calibration execution and threshold freeze

**Status:** COMPLETE. Q3 executed on pilot seeds `[42, 101, 202]`. All negative controls failed as required; positive control passed with `m1=0.9152`; separation gaps: `m1=0.1930`, `m1ε=0.1652`, `m1b=0.1568`, `m2=0.0692`; `m5` fixed at `0.90` under OD-5; thresholds frozen in `params.yaml`.

**Files:** `reports/docs/phases/phase_2/q3_threshold_calibration_report.md`, `params.yaml`

**Actions:**
1. Execute Q3 threshold calibration across the 3 pilot seeds $\{42, 101, 202\}$ and controls via the CLI / harness.
2. Verify that all negative controls fail and clear separation $\ge \Delta_{\min}$ is achieved on $m_1$, $m_{1\epsilon}$, $m_{1b}$, and $m_2$.
3. Generate and commit `reports/docs/phases/phase_2/q3_threshold_calibration_report.md`.
4. Freeze the calibrated numeric thresholds into `params.yaml` under `tier1_ml.validation_thresholds`.

**Verification:**
```powershell
uv run python -m aegis.pipelines.cli calibrate-q3
Test-Path reports/docs/phases/phase_2/q3_threshold_calibration_report.md
```

---

## T7 — Documentation, Gate 9 evidence capture, and final checks

**Files:** `reports/docs/phases/phase_2/execution_plan.md`, `reports/docs/phases/phase_2/stage_evidence.md`, `tasks.md`

**Actions:**
1. Update `execution_plan.md`: mark Stage 9 as `✅ PASSED` with Gate 9 outcome.
2. Update `stage_evidence.md`: update Stage 9 row and append detailed §Stage 9 evidence section.
3. Update `tasks.md` task table to COMPLETE.
4. Run full project checks:
   ```powershell
   uv run pytest -q
   uv run ruff check .
   uv run pyright
   uv run python scripts/check_module_size.py
   ```
5. Report Gate 9 completion to user.

---

## Execution Order

T1 → T2 → T3 → T4 → T5 → T6 → T7 (strictly serial).
