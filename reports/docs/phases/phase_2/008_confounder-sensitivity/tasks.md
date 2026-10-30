# Tasks: Stage 11 — Hidden-Confounder Sensitivity Analysis (P2-F08)

**Feature:** `P2-F08` (Light SDD: tasks.md)  
**Governing:** ADR-013, P2-D7, `system_design.md` §6  
**Status:** COMPLETED (✅ Gate 11 Passed)  
**Date:** 2026-10-30  

---

## Overview

Execute an unobserved-confounding sensitivity sweep over the known synthetic benchmark to establish the causal estimator's robustness value, generate the degradation curve, and select the representative degraded case for showcase Scenario S4.

Per P2-D7 and ADR-013:
- An unobserved confounder $U \sim \mathcal{N}(0, 1)$ is injected simultaneously into treatment assignment $T$ and renewal log-odds $Y$ with coupling strengths $(\gamma_T, \gamma_Y)$.
- A $3 \times 3$ grid plus baseline zero cell $(0.0, 0.0)$ produces 10 cells total.
- $U$ is strictly hidden from the causal estimator feature matrix $X$.
- Fail-loud execution: the harness must raise an explicit typed exception if any cell in the sensitivity grid is skipped or fails to converge.
- Metrics recorded per cell: $m_1(\tau)$ (segment rank correlation), $m_1(\epsilon)$, $m_{1b}$ (policy rank correlation), and $m_2$ (mean absolute elasticity bias).
- Robustness value calculation: minimum confounding strength $\sqrt{\gamma_T^2 + \gamma_Y^2}$ where segment rank correlation falls below the Stage 10 target threshold.
- Scenario S4 selection: identify a representative confounded cell showing clear degradation for the diagnostic showcase slice.

---

## Task Decomposition

### T1 — Schema and Configuration (`src/aegis/schemas/sensitivity.py`, `src/aegis/config/schema.py`, `params.yaml`)
- Define `SensitivityGridCell` (gamma_t, gamma_y, cell_id).
- Define `SensitivityCellResult` with metrics ($m_1, m_{1\epsilon}, m_{1b}, m_2$), fit duration, and convergence status.
- Define `SensitivitySummary` with full grid results, baseline metrics, robustness value, and Scenario S4 selection.
- Add `SensitivityConfig` and `SensitivityPathsConfig` to `src/aegis/config/schema.py` and register in `params.yaml`.

### T2 — Hidden-Confounder Injection Pipeline (`src/aegis/pipelines/sensitivity.py`)
- Implement `inject_hidden_confounder(base_features_df, glm_df, seg_df, gamma_t, gamma_y, rng)`:
  - Generate $U \sim \mathcal{N}(0, 1)$.
  - Inject $U$ into treatment assignment: $\mu_T' = \mu_T + \gamma_T \cdot U$; clip treatment to corridor.
  - Inject $U$ into renewal probability log-odds: $\text{logit}(P(\text{renew})) = a(X) + b(X) \cdot T + \gamma_Y \cdot U$.
  - Ground truth $(\tau_i, q_{0i}, \epsilon_i)$ remains the true marginal causal effect at $\Delta = 0$ (invariant to $U$ at $\Delta=0$).
  - Omit $U$ completely from the feature columns returned for estimator training.

### T3 — Sensitivity Sweep Runner with Fail-Loud Guarantees
- Implement `run_sensitivity_sweep(config, rng_seed)`:
  - Iterate strictly through all 10 cells in the configured grid.
  - Fail loud: if any cell encounters fitting failure, convergence failure, or is skipped, immediately raise `SensitivityAnalysisError`.
  - Assert exactly 10 completed cells before computing summary statistics.

### T4 — Robustness Value & Scenario S4 Selection
- Compute degradation relative to baseline $(0.0, 0.0)$.
- Identify robustness value: the smallest Euclidean confounding strength $r = \sqrt{\gamma_T^2 + \gamma_Y^2}$ where $m_1(\tau) < 0.70$ (or minimum tested if all clear).
- Select Scenario S4: choose a moderate degradation cell (e.g. $\gamma_T = 0.05, \gamma_Y = 0.5$) that clearly demonstrates estimator limits without total collapse.
- Save summary artifact to `artifacts/sensitivity_summary.json` and generate Markdown report `reports/docs/phases/phase_2/confounder_sensitivity_report.md`.

### T5 — CLI Integration (`src/aegis/pipelines/cli.py`)
- Add `run-sensitivity` subcommand to the AEGIS CLI.
- Add `--summary-path` and `--report-path` options with defaults.

### T6 — Unit Tests (`tests/unit/test_sensitivity.py`)
- Test confounder injection: verify $U$ is not present in estimator features.
- Test fail-loud enforcement: verify an exception is raised if a cell fails or is skipped.
- Test schema validation: verify `SensitivitySummary` Pydantic model serialization and deserialization.
- Test robustness value calculation logic with synthetic mock inputs.

### T7 — Execution, Verification & Evidence
- [x] Execute `run-sensitivity` on baseline seed 42.
- [x] Verify `artifacts/sensitivity_summary.json` and `confounder_sensitivity_report.md` are generated with cryptographic hashes.
- [x] Run `uv run pytest tests/unit/test_sensitivity.py -q` (7 passed).
- [x] Run `uv run ruff check .` and `uv run pyright` (clean, 0 errors).
- [x] Run `uv run python scripts/check_module_size.py` (47/47 modules within ceiling).
- [x] Update `reports/docs/phases/phase_2/execution_plan.md` and `stage_evidence.md`.

---

## Outcome (2026-10-30) — ✅ PASSED

- **10/10 Grid Cells Completed:** Baseline cell $(0.0, 0.0)$ plus 9 coupling combinations executed with zero skips verified.
- **Robustness Value:** $R_v = 0.5004$ (minimum non-zero confounding norm $\sqrt{\gamma_T^2 + \gamma_Y^2}$ breaching $m_1(\tau) < 0.60$).
- **Scenario S4 Locked Cell:** `gt_0.05_gy_0.50` ($\gamma_T = 0.05, \gamma_Y = 0.50, r = 0.502$, $m_1(\tau) = 0.6579$, bias inflation $\Delta m_2 = +0.6272$ up to $m_2 = 0.7419$).
- **Artifacts:**
  - `artifacts/sensitivity_summary.json`
  - `reports/docs/phases/phase_2/confounder_sensitivity_report.md`
- **Gate 11:** Hard close satisfied; Stage 12 (MLflow Registry & Promotion) unblocked.
