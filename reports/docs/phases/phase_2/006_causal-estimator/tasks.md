# Tasks: P2-F06 Causal Estimator & Segment Inference
## Phase 2 — Stage 8

**Spec:** APPROVED October 26, 2026  
**Plan:** APPROVED October 26, 2026  
**Status:** ✅ COMPLETE — Gate 8 PASSED October 26, 2026  
**Date:** October 26, 2026  

Execution rule: tasks run strictly in order. Each task starts with a failing test where the behavior is testable, ends with the listed verification commands green on the changed files, and leaves the working tree reviewable. No task starts while the previous task is failed or partial.

| Task | Scope | Status |
|---|---|---|
| T1 | Configuration, exception, and dependency surface | ✅ COMPLETE |
| T2 | Causal output schemas | ✅ COMPLETE |
| T3 | Training contract, frozen split, nuisance models, and fit | ✅ COMPLETE |
| T4 | Truth-free q0, segment aggregation, and intervals | ✅ COMPLETE |
| T5 | Metadata, persistence, and end-to-end orchestration | ✅ COMPLETE |
| T6 | CLI `train-estimator` | ✅ COMPLETE |
| T7 | Structural isolation tests | ✅ COMPLETE |
| T8 | Full-data execution and Gate 8 evidence capture | ✅ COMPLETE |
| T9 | Documentation, register updates, and final validation | ✅ COMPLETE |

---

## T1 — Configuration, exception, and dependency surface

**Files:** `pyproject.toml`, `uv.lock`, `params.yaml`, `src/aegis/config/schema.py`, `src/aegis/utils/exceptions.py`, `tests/unit/test_config.py`

**Actions:**
1. Add `joblib>=1.3.0` as a direct dependency; refresh `uv.lock`.
2. Extend `params.yaml` `tier1_ml.estimator` with `n_estimators: 100`, `min_samples_leaf: 5`, `min_samples_split: 10`, `max_depth: null`, `max_samples: 0.45`, `subforest_size: 4`, `ci_alpha: 0.05`, `q0_floor: 1e-06`, `n_jobs: -1`.
3. Add `tier1_ml.estimator_paths` with the four artifact paths from `plan.md`.
4. Extend `EstimatorConfig` and add `EstimatorPathsConfig`; validate positive hyperparameters, `0 < ci_alpha < 1`, `0 < q0_floor < 1`, and `n_jobs != 0`.
5. Add `CausalEstimatorError(AEGISError)`.
6. Add config tests for defaults, real `params.yaml` load, and rejection of `n_jobs=0`, `ci_alpha=1.0`, `q0_floor=0`, and negative forest sizes.

**Verification:**
```powershell
uv lock
uv run pytest tests/unit/test_config.py -q
uv run ruff check src/aegis/config/schema.py src/aegis/utils/exceptions.py tests/unit/test_config.py
uv run pyright src/aegis/config/schema.py src/aegis/utils/exceptions.py
```

**Done when:** `params.yaml` loads through `AEGISConfig`; invalid values fail; tests and type checks pass.

---

## T2 — Causal output schemas

**Files:** `src/aegis/schemas/causal.py` (new), `tests/unit/test_causal_estimator.py`

**Actions:**
1. Write tests first: valid segment estimate round-trip; non-finite values rejected; inverted τ/ε intervals rejected; point outside interval rejected; `q0 <= 0` rejected; unknown fields rejected; duplicate segment IDs rejected; metadata rejects timings/unknown fields.
2. Implement `SegmentElasticityEstimate`, `SegmentElasticityTable`, `CausalEstimatorMetadata`, `CausalEstimatorRunManifest` with `extra="forbid"` and the validators described in `plan.md`.
3. Export no adjustment-like fields (INV-4).

**Verification:**
```powershell
uv run pytest tests/unit/test_causal_estimator.py -q -k "schema or interval or metadata"
uv run ruff check src/aegis/schemas/causal.py tests/unit/test_causal_estimator.py
uv run pyright src/aegis/schemas/causal.py
```

**Done when:** schema tests pass; valid payloads round-trip; every invalid case raises `ValidationError`.

---

## T3 — Training contract, frozen split, nuisance models, and fit

**Files:** `src/aegis/pipelines/causal_estimator.py` (new), `tests/unit/test_causal_estimator.py`

**Actions:**
1. Add tests first: 29-column/25-feature acceptance; each truth column rejected; wrong count rejected; split determinism, 80/20 sizes, stratification, disjoint IDs; oversized sample rejected; GBT nuisance types.
2. Implement constants, `validate_training_contract`, `stratified_sample_split`, `build_nuisance_models`, `fit_causal_estimator`, and `predict_policy_effects`.
3. `fit_causal_estimator` constructs `CausalForestDML` with the frozen config, `discrete_outcome=True`, `discrete_treatment=False`, `inference=True`, and `n_jobs` from config. Wrap fit exceptions in `CausalEstimatorError` with context.
4. `predict_policy_effects` reshapes `(n, 1)` output to `(n,)` and rejects non-finite values.

**Verification:**
```powershell
uv run pytest tests/unit/test_causal_estimator.py -q -k "contract or split or nuisance or effect_shape"
uv run ruff check src/aegis/pipelines/causal_estimator.py tests/unit/test_causal_estimator.py
uv run pyright src/aegis/pipelines/causal_estimator.py
```

**Done when:** contract and split tests pass; a small fit returns finite one-dimensional effects.

---

## T4 — Truth-free q0, segment aggregation, and intervals

**Files:** `src/aegis/pipelines/causal_estimator.py`, `tests/unit/test_causal_estimator.py`

**Actions:**
1. Add tests first: fold-averaged nuisance structure `(1, cv_folds)` accepted; wrong shape rejected; q0 clipped into `[q0_floor, 1]`; stub predictions produce the expected deconfounded value; interval scaling preserves ordering; hand-calculated ratio-of-sums equality; missing segment raises.
2. Implement `estimate_baseline_retention`, `scale_interval_to_elasticity`, `aggregate_segment_estimates`, and `validate_segment_coverage`.
3. Use `ate_inference(X_s, T0=0, T1=1).mean_point` as `tau_hat` and `conf_int_mean(alpha=ci_alpha)` for the τ interval. Convert with `_as_scalar`.
4. Build `SegmentElasticityEstimate` rows in sorted segment order; any schema failure propagates as `CausalEstimatorError` with the segment ID.

**Verification:**
```powershell
uv run pytest tests/unit/test_causal_estimator.py -q -k "q0 or aggregate or interval or coverage"
uv run ruff check src/aegis/pipelines/causal_estimator.py tests/unit/test_causal_estimator.py
uv run pyright src/aegis/pipelines/causal_estimator.py
```

**Done when:** all segment rows are finite and self-consistent; ratio-of-sums is verified by hand; missing-segment coverage fails loudly.

---

## T5 — Metadata, persistence, and end-to-end orchestration

**Files:** `src/aegis/pipelines/causal_estimator.py`, `src/aegis/schemas/causal.py`, `tests/unit/test_causal_estimator.py`

**Actions:**
1. Add tests first: metadata builder fields/hashes; CSV fixed column order and byte-identical repeated writes; model persistence returns a 64-hex SHA-256; fast end-to-end run on the synthetic fixture writes all four artifacts and returns validated metadata.
2. Implement `build_metadata`, `write_segment_estimates`, `write_metadata`, `persist_estimator`, `write_run_manifest`, and `run_causal_estimator_pipeline`.
3. Metadata uses canonical config fingerprint, input SHA-256 hashes, `model_version = cfdml-<nuisance>-cv<folds>-<hash12>-<hash12>`, and interval audit rates. Exclude wall time and model-binary hashes.
4. `run_causal_estimator_pipeline` loads the training table and segment definitions, fits on train, infers on test, validates coverage, writes CSV/metadata/model/run manifest, and returns metadata.
5. The run manifest holds wall seconds and the model artifact hash/size.

**Verification:**
```powershell
uv run pytest tests/unit/test_causal_estimator.py -q
uv run ruff check src/aegis/pipelines/causal_estimator.py src/aegis/schemas/causal.py tests/unit/test_causal_estimator.py
uv run pyright src/aegis/pipelines/causal_estimator.py src/aegis/schemas/causal.py
```

**Done when:** fast end-to-end test passes; repeated CSV writes are byte-identical; no timing field appears in deterministic metadata.

---

## T6 — CLI `train-estimator`

**Files:** `src/aegis/pipelines/cli.py`, `tests/unit/test_cli.py`

**Actions:**
1. Add tests first: parser accepts `train-estimator` and its optional output overrides; dispatch returns 0 on success; `CausalEstimatorError` prints `[BLOCK]` and returns 1.
2. Implement `train_estimator`, subparser, and dispatch.
3. No bare exception handler; only `CausalEstimatorError` is converted to exit code 1.

**Verification:**
```powershell
uv run pytest tests/unit/test_cli.py -q
uv run ruff check src/aegis/pipelines/cli.py tests/unit/test_cli.py
uv run pyright src/aegis/pipelines/cli.py
```

**Done when:** CLI tests pass; parser help lists the subcommand.

---

## T7 — Structural isolation tests

**Files:** `tests/unit/test_causal_estimator.py` (or `tests/unit/test_truth_isolation.py`)

**Actions:**
1. Add an AST scan over `src/aegis/api/*.py` that fails on imports of `aegis.pipelines.causal_estimator`, `econml`, `joblib`, or `pickle`.
2. Add a test that injects a forbidden import into an in-memory source string and confirms the scanner raises.
3. Confirm the existing truth-loader boundary test covers `causal_estimator.py` without modification.

**Verification:**
```powershell
uv run pytest tests/unit/test_causal_estimator.py tests/unit/test_truth_isolation.py -q
uv run ruff check tests/unit/test_causal_estimator.py
uv run pyright tests/unit/test_causal_estimator.py
```

**Done when:** serving-decoupling and truth-isolation tests pass; injected violations are detected.

---

## T8 — Full-data execution and Gate 8 evidence capture

**Files:** `data/processed/segment_elasticity_estimates.csv`, `data/processed/causal_estimator_metadata.json`, `artifacts/causal_estimator/estimator.joblib`, `artifacts/causal_estimator/run_manifest.json`, `reports/docs/phases/phase_2/stage_evidence.md`

**Actions:**
1. Run:
   ```powershell
   uv run python -m aegis.pipelines.cli train-estimator
   ```
2. Verify: 35 rows; fixed column order; all values finite; 100% interval self-consistency; held-out counts > 0; `epsilon_hat == tau_hat / baseline_retention_q0` per row within float tolerance; no adjustment fields.
3. Record SHA-256 hashes, row counts, point-estimate ranges, interval width range, model artifact size, and wall time in `stage_evidence.md`.
4. Run the pipeline a second time and compare the segment CSV SHA-256 to the first run. If the hashes differ, stop and diagnose; do not write Gate 8 evidence.
5. Confirm `artifacts/` remains gitignored and no truth column appears in any output.

**Verification:**
```powershell
uv run python -c "import pandas as pd, json, hashlib, pathlib; ..."
Get-FileHash data/processed/segment_elasticity_estimates.csv -Algorithm SHA256
```

**Done when:** all Gate 8 interval criteria are met; repeated-run segment CSV hashes match; evidence is recorded with exact commands and actual outcomes.

---

## T9 — Documentation, register updates, and final validation

**Files:** `reports/docs/phases/phase_2/execution_plan.md`, `reports/docs/phases/phase_2/stage_evidence.md`, `reports/docs/phases/phase_2/006_causal-estimator/spec.md`, `plan.md`, `tasks.md`

**Actions:**
1. Mark Stage 8 `✅ PASSED` in the execution plan with the actual gate evidence summary and date.
2. Update the feature register: `P2-F06` status to `✅ Implemented`; Stage 8 status to complete.
3. Update `stage_evidence.md` Stage 8 row and append the full evidence section.
4. Mark `tasks.md` status complete and update task statuses.
5. Run the final checks:
   ```powershell
   uv run pytest -q
   uv run ruff check <changed files>
   uv run pyright
   uv run python scripts/check_module_size.py
   ```
6. Report exact commands and outcomes, including any pre-existing failure outside the change scope (for example the known `truth_loader.py:78` E501 if still present).

**Done when:** Stage 8 is documented as passed, all checks are green on changed files, and the operator has the evidence needed to sign off Gate 8.

---

## Execution Order

T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9 (strictly serial).
