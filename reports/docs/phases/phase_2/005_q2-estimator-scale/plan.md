# Plan & Tasks: Q2 Estimator Training Scale & Budget
## Phase 2 — Stage 7

**Spec:** APPROVED October 26, 2026 (OD-1a, OD-2a)  
**Status:** COMPLETE — Gate 7 passed October 26, 2026  
**Date:** October 26, 2026  

---

## Implementation Plan

### T1 — Configuration surface for estimator block
- **Status:** ✅ Complete.
- **Objective:** Add `tier1_ml.estimator` Pydantic model and `params.yaml` block, then freeze the Q2-selected values.
- **FR:** FR-7, AC-4.
- **Files:** `src/aegis/config/schema.py`, `params.yaml`, `tests/unit/test_config.py`.
- **Done when:** Config loads; pyright and tests green.

### T2 — Q2 experiment script
- **Status:** ✅ Complete; all 8 cells ran with no skips.
- **Objective:** Implement `scripts/q2_estimator_scale.py` — loads data, runs the 4×2 grid, measures metrics, applies selection logic, writes report.
- **FR:** FR-1…FR-6, AC-1…AC-3.
- **Files:** `scripts/q2_estimator_scale.py`.
- **Done when:** Script runs to completion with all 8 cells; report written.

### T3 — Execute Q2, freeze parameters, record evidence
- **Status:** ✅ Complete; full-data GBT selected (N=413,169, 514.8 s, 2,534.3 MB RSS).
- **Objective:** Run Q2, update `params.yaml` with selected n* and nuisance family, append Stage 7 evidence.
- **FR:** FR-7, AC-4, AC-5.
- **Files:** `params.yaml`, `reports/docs/phases/phase_2/stage_evidence.md`, `q2_estimator_scale_report.md`.
- **Done when:** Gate 7 criteria satisfied.

---

## Execution Order

T1 → T2 → T3 (serial; T3 requires running T2's script).
