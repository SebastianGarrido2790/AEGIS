# Tasks: Stage 10 — Final-Seed Validation Gate (Roadmap Exit E1)

**Feature:** `P2-F07` (Full SDD approved at Stage 9; this file records the Stage 10 execution layer)
**Governs:** ADR-013 (Exit E1 verification), P2-D6, P2-D6a
**Status:** FAILED — Gate 10 not met (4/5 seeds); remediation escalated to P2-D5
**Date:** 2026-10-27

Stage 10 adds no new specification. It implements the stage actions already approved in
`execution_plan.md` (Stage 10) on top of the Stage 9 harness:

| Decision | Value | Source |
|---|---|---|
| Final seed family | `[501, 502, 503, 504, 505]` (K = 5), disjoint from pilot seeds `[42, 101, 202]` | Stage 9 OD-3 |
| Seed variation | Seed `s` sets the stratified split and estimator `random_state`; DGP noise uses `s + 10000` (`synthetic_dgp.seed_offset_noise`) | P2-D6 ("varying both DGP noise and estimator random seeds") |
| Gate statistics | Worst seed clears every frozen **floor**; every seed must pass; median meets every frozen **target** (reported) | P2-D6, execution_plan Stage 10 |
| Artifact | `artifacts/causal_validation_summary.json` + `final_seed_validation_report.md` | execution_plan Stage 10, Exit E1 checklist |
| Integrity | SHA-256 of `params.yaml` and all five fingerprinted test-data files, re-derived on verification | execution_plan Stage 10 items 4–5 |

**Out of scope:** negative/positive controls (Stage 9), confounder sensitivity (Stage 11),
MLflow packaging (Stage 12), DVC DAG assembly (Stage 14).

---

## T1 — Final-seed campaign and gate engine

**Files:** `src/aegis/pipelines/final_seed_validation.py`, `src/aegis/schemas/final_validation.py`

1. `evaluate_seed_gate` compares one seed's metrics against every frozen threshold,
   handling `m2` as a ceiling and the rest as floors.
2. `evaluate_campaign_gates` derives worst-seed and median-seed values from the per-seed
   distributions — never from a single seed — and lists failed seeds.
3. `run_final_seed_campaign` rejects seed-set violations (duplicates, pilot overlap, K < 5)
   and fails loud rather than relaxing thresholds.
4. `build_final_seed_summary` assembles the Exit E1 artifact with config and test-data
   hashes and the zero-tautology declaration.

## T2 — Persistence, verification, and tamper detection

**Files:** `src/aegis/pipelines/final_seed_validation.py`

1. `persist_summary` writes the canonical JSON artifact and the Markdown report.
2. `verify_final_validation_summary` re-hashes `params.yaml` and every fingerprinted data
   file, compares stored thresholds to the live config, and re-evaluates the gates on the
   stored metrics.
3. Verification reports an altered threshold as a failure; it does not crash on the
   resulting ordering violation.

## T3 — Configuration and CLI

**Files:** `src/aegis/config/schema.py`, `params.yaml`, `src/aegis/pipelines/cli.py`

1. `Tier1MLConfig.final_seeds` with a disjointness/unicity validator;
   `SyntheticDGPConfig.seed_offset_noise = 10000`;
   `FinalValidationPathsConfig` (artifact + report paths).
2. CLI `run-final-validation` returns 0 only when the worst seed and every seed pass.
3. CLI `verify-final-validation` returns 1 on any hash, threshold, or gate mismatch.

## T4 — Truth-isolation boundary

**Files:** `tests/unit/test_truth_isolation.py`

`final_seed_validation.py` joins `validation_harness.py` as the second authorized offline
consumer of `truth_loader.py`; all other modules stay blocked.

## T5 — Tests

**Files:** `tests/unit/test_final_seed_validation.py`, `tests/unit/test_cli.py`,
`tests/unit/test_config.py`

Gate logic, worst/median selection, seed hygiene, artifact hashing, JSON round-trip,
malformed-artifact handling, seed-overlap rejection at config load, CLI pass/fail routing,
and the threshold/data tamper falsifications.

## T6 — Campaign execution and falsification

**Commands:**

```powershell
uv run python -m aegis.pipelines.cli run-final-validation
uv run python -m aegis.pipelines.cli verify-final-validation
# Falsification: alter a frozen threshold in params.yaml, then re-verify (expect exit 1),
# then restore params.yaml and confirm verification passes again.
```

## T7 — Documentation and evidence

`execution_plan.md` Stage 10 status, `stage_evidence.md` §Stage 10, `implementation_plan.md`
§8 "Validation gate" row, and `final_seed_validation_report.md`.

---

## Outcome (2026-10-28) — FAILED

T1–T6 were executed and the artifacts produced. **The gate itself did not pass**, so Stage 10
closes as a recorded failure rather than a pass.

- **Campaign:** K = 5 (seeds 501–505), independent DGP noise and estimator random state per seed.
- **Result:** 4 of 5 seeds breach the frozen rank floors; only seed 501 clears them.
  `m2`, `m3`, `m4`, `m5` pass on every seed.
- **Root cause:** rank recovery spans 0.55–0.95 across DGP-noise draws with the estimand held
  fixed; noise draw 42, the only draw Q3 calibrated on, is the optimistic tail.
- **Falsification:** threshold-tamper detection verified — mechanism valid, verdict unaffected.
- **Thresholds:** unchanged. Recalibration cannot pass without lowering a floor to fit the
  result, which P2-D6a forbids.
- **Escalation:** estimator-family decision (P2-D5). Stages 11–15 remain blocked.

Full analysis: `reports/docs/phases/phase_2/stage10_failure_analysis.md`.

