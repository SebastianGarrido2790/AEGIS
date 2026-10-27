# Plan: P2-F06 Causal Estimator & Segment Inference
## Phase 2 — Stage 8

**Spec:** APPROVED October 26, 2026 (OD-1 … OD-5 as recommended)  
**Status:** APPROVED — October 26, 2026  
**Implementation Status:** COMPLETE — Gate 8 PASSED October 26, 2026  
**Date:** October 26, 2026  

---

## 1. Implementation Strategy

Stage 8 is a deterministic pipeline with five separable responsibilities. Each is implemented as a pure or near-pure function so it can be tested without a full-data fit:

1. **Contract validation** — reject truth leakage and malformed training tables.
2. **Frozen split** — reproduce the Q2 stratified 80/20 split exactly.
3. **Fit** — construct the Q2-frozen `CausalForestDML` and fit on training rows only.
4. **Truth-free inference** — per-policy effects, deconfounded `q0_hat`, segment-average point estimates, and segment-average intervals.
5. **Persistence** — deterministic CSV/JSON plus non-promotable run artifacts.

The full-data run is executed through the CLI for gate evidence and repeated once for the AC-7 byte-identity check. Unit tests use a small synthetic table with reduced forest size so the default suite stays fast.

---

## 2. Module Design

### 2.1 `src/aegis/schemas/causal.py` (new)

Typed contracts:

- `SegmentElasticityEstimate`
  - `segment_id: str` (canonical pattern)
  - `inference_policy_count: int` (`gt=0`)
  - `baseline_retention_q0: float` (`gt=0.0, le=1.0`)
  - `tau_hat`, `tau_ci_lower`, `tau_ci_upper`: `float`
  - `epsilon_hat`, `epsilon_ci_lower`, `epsilon_ci_upper`: `float`
  - `extra="forbid"`
  - after-validator: all values finite; `lower <= point <= upper` for both τ and ε.
- `SegmentElasticityTable`
  - `segments: list[SegmentElasticityEstimate]`
  - unique `segment_id`; non-empty.
- `CausalEstimatorMetadata`
  - estimator identity/configuration, split sizes, feature/segment counts, `ci_alpha`, `q0_floor`, `q0_estimation_method`, interval audit rates, input hashes, config fingerprint, `model_version`.
  - rejects unknown fields.
- `CausalEstimatorRunManifest`
  - model SHA-256/size, fit and pipeline wall seconds, artifact path.

Schema constants (`SEGMENT_ESTIMATE_COLUMNS`) live next to the models or in the pipeline module; the CSV writer uses a single canonical order.

### 2.2 `src/aegis/pipelines/causal_estimator.py` (new)

Constants:

```python
TRUTH_COLUMNS: frozenset[str] = frozenset({
    "tau", "epsilon", "q0", "true_effect", "tau_ground_truth",
    "baseline_renewal_q0", "true_tau", "true_epsilon", "fitted_loss_cost",
})
FEATURE_METADATA_COLUMNS = ("PolicyID", "treatment_rate_change", "renewed", "segment_id")
EXPECTED_FEATURE_COUNT = 25
EXPECTED_COLUMN_COUNT = 29
SEGMENT_ESTIMATE_COLUMNS = (
    "segment_id", "inference_policy_count", "baseline_retention_q0",
    "tau_hat", "tau_ci_lower", "tau_ci_upper",
    "epsilon_hat", "epsilon_ci_lower", "epsilon_ci_upper",
)
```

Public functions (Google-style docstrings mandatory):

Final implemented signatures (the draft signatures below were refined during T3–T5):

```python
def validate_training_contract(
    data: pd.DataFrame, *, corridor_width: float,
) -> list[str]
    # Returns the 25 feature columns; raises CausalEstimatorError.

def stratified_sample_split(
    data: pd.DataFrame, sample_size: int, *,
    test_size: float, random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]

def build_nuisance_models(
    family: str, *, random_state: int, cv_folds: int,
) -> tuple[Any, Any]

def fit_causal_estimator(
    train: pd.DataFrame, feature_columns: list[str], *,
    estimator_cfg: EstimatorConfig, random_state: int,
) -> Any  # CausalForestDML; Any because EconML ships no type stubs

def predict_policy_effects(estimator: Any, X: np.ndarray) -> np.ndarray

def estimate_baseline_retention(
    estimator: Any, X: np.ndarray, *, q0_floor: float, cv_folds: int,
) -> np.ndarray

def aggregate_segment_estimates(
    estimator: Any, test: pd.DataFrame, feature_columns: list[str], *,
    ci_alpha: float, q0_floor: float, cv_folds: int,
) -> SegmentElasticityTable

def interval_quality_rates(table: SegmentElasticityTable) -> tuple[float, float]
```

`scale_interval_to_elasticity` is implemented inline in `aggregate_segment_estimates`
rather than as a separate public function; the ordering and containment behavior is
covered by schema and aggregation tests.

def validate_segment_coverage(
    estimates: list[SegmentElasticityEstimate], expected_segments: set[str],
) -> None

def build_metadata(
    *, estimator_cfg, random_state, cv_folds, train_rows, test_rows,
    feature_count, segment_count, ci_alpha, q0_floor, input_hashes,
    config_fingerprint, model_version,
) -> CausalEstimatorMetadata

def write_segment_estimates(
    estimates: list[SegmentElasticityEstimate], path: Path,
) -> None

def write_metadata(metadata: CausalEstimatorMetadata, path: Path) -> None

def persist_estimator(estimator: Any, path: Path) -> str  # SHA-256

def write_run_manifest(
    *, artifact_path: Path, artifact_sha256: str, artifact_bytes: int,
    fit_seconds: float, pipeline_seconds: float, path: Path,
) -> None

def run_causal_estimator_pipeline(
    config: AEGISConfig | None = None, *,
    training_path: Path | None = None,
    definitions_path: Path | None = None,
    segment_estimates_path: Path | None = None,
    metadata_path: Path | None = None,
    model_artifact_path: Path | None = None,
    run_manifest_path: Path | None = None,
) -> CausalEstimatorMetadata
```

Internal helpers:

- `_sha256_file(path)`, `_config_fingerprint(raw_params_path)`
- `_model_version(config_hash, training_hash)`
- `_as_scalar(value)` to normalize EconML scalar/array returns
- `_validate_nuisance_structure(estimator, cv_folds)`

### 2.3 `src/aegis/config/schema.py` (extend)

Extend `EstimatorConfig` (all defaults are the Q2-effective values):

```python
n_estimators: int = 100
min_samples_leaf: int = 5
min_samples_split: int = 10
max_depth: int | None = None
max_samples: float = 0.45
subforest_size: int = 4
ci_alpha: float = 0.05
q0_floor: float = 1e-6
n_jobs: int = -1
```

Add `EstimatorPathsConfig`:

```python
segment_estimates: str = "data/processed/segment_elasticity_estimates.csv"
metadata: str = "data/processed/causal_estimator_metadata.json"
model_artifact: str = "artifacts/causal_estimator/estimator.joblib"
run_manifest: str = "artifacts/causal_estimator/run_manifest.json"
```

Add `estimator_paths: EstimatorPathsConfig` to `Tier1MLConfig`. Validate `n_jobs != 0`, `0 < ci_alpha < 1`, `0 < q0_floor < 1`.

### 2.4 `src/aegis/utils/exceptions.py` (extend)

Add `CausalEstimatorError(AEGISError)` with a module docstring reference to P2-F06.

### 2.5 `src/aegis/pipelines/cli.py` (extend)

- Import `CausalEstimatorError` and `run_causal_estimator_pipeline`.
- Add `train_estimator(...) -> int` with `[BLOCK]` output on typed failure.
- Add `train-estimator` subparser with optional output-path overrides.
- Dispatch in `main()`.

### 2.6 `pyproject.toml` (extend)

Add direct dependency `joblib>=1.3.0` (used for the non-promotable model artifact). Run `uv lock` / `uv sync`.

---

## 3. Data Flow

```text
elasticity_training.csv
        │
        ▼
validate_training_contract ──► 25 feature columns, truth columns rejected
        │
        ▼
stratified_sample_split(sample_size=n*, test_size=0.20, seed=42)
        │
        ├── train (~330,535) ──► build_nuisance_models → CausalForestDML.fit
        │
        └── test (~82,634)
                 │
                 ├── predict_policy_effects → tau_i
                 ├── estimate_baseline_retention → q0_i
                 │       (fold-averaged E[Y|X], E[T|X]; q0 = clip(E[Y|X] - tau·E[T|X]))
                 └── per segment:
                         ate_inference(X_s).mean_point → tau_s
                         conf_int_mean(alpha) → [tau_l, tau_u]
                         q0_s = mean(q0_i)
                         epsilon_s = tau_s / q0_s
                         epsilon interval = tau interval / q0_s
        │
        ▼
SegmentElasticityEstimate × 35 ──► validate coverage / schema
        │
        ├── data/processed/segment_elasticity_estimates.csv
        ├── data/processed/causal_estimator_metadata.json
        ├── artifacts/causal_estimator/estimator.joblib (non-promotable)
        └── artifacts/causal_estimator/run_manifest.json
```

No truth sidecar read occurs anywhere in this flow.

---

## 4. Test Strategy

### 4.1 Fast unit tests (`tests/unit/test_causal_estimator.py`)

Build a synthetic 25-feature table with 2 segments and ~2,000 rows. Treatment is uniform in the corridor; outcome is a simple logistic function with a known sign. No truth sidecar is involved.

| Test | Asserts |
|---|---|
| `test_training_contract_returns_feature_columns` | 25 features returned for the valid fixture |
| `test_training_contract_rejects_truth_columns` | each truth column raises `CausalEstimatorError` |
| `test_training_contract_rejects_wrong_column_count` | 28/30 columns rejected |
| `test_split_is_deterministic_and_stratified` | repeated calls identical; sizes 80/20; both segments in both partitions; disjoint PolicyIDs |
| `test_split_rejects_oversized_sample` | `sample_size > len(data)` raises |
| `test_build_nuisance_models_gbt` | classifier/regressor types; linear family rejected in Stage 8 config? |
| `test_fast_fit_produces_finite_segment_estimates` | 2 segments; finite τ/ε; ε = τ/q0; interval containment |
| `test_fit_pins_forest_inference_to_one_thread` | fitted forest inference uses `n_jobs=1` for deterministic per-tree accumulation |
| `test_q0_is_clipped_and_positive` | stub nuisance models produce clipped q0 in `[q0_floor, 1]` |
| `test_scale_interval_is_monotone` | positive q0 preserves ordering and containment |
| `test_missing_segment_raises` | coverage validator raises with the missing ID |
| `test_csv_is_byte_identical_on_repeated_write` | same estimates → identical bytes |
| `test_interval_schema_rejects_non_finite` | `SegmentElasticityEstimate` fails |
| `test_interval_schema_rejects_inverted_or_non_containing` | both cases fail |

### 4.2 Structural tests

- Extend `tests/unit/test_truth_isolation.py` or add to `test_causal_estimator.py`: AST scan of every `src/aegis/api/*.py` rejects imports of `aegis.pipelines.causal_estimator`, `econml`, `joblib`, and `pickle` (serving decoupling, FR-11).
- Existing truth-loader AST boundary test automatically covers `causal_estimator.py`.

### 4.3 Config and CLI tests

- `tests/unit/test_config.py`: new estimator keys load; invalid `n_jobs=0`, `ci_alpha`, `q0_floor`, and negative hyperparameters are rejected.
- `tests/unit/test_cli.py`: parser exposes `train-estimator`; dispatch calls the pipeline and returns 1 on `CausalEstimatorError` (monkeypatched).

### 4.4 Full-data evidence run

- `uv run python -m aegis.pipelines.cli train-estimator`
- Verify 35 rows, 100% self-consistency, finite values, hash outputs, artifact presence.
- Run the command twice and compare segment CSV SHA-256 values for AC-7.
- Record actual wall time and memory observations in `stage_evidence.md`.

Full-data test is not part of the default pytest run; the fast fixture covers logic and the CLI run covers the gate.

---

## 5. Determinism and Reproducibility

- Fixed split and model seeds from `params.yaml`.
- Nuisance fold averaging uses a fixed fold order (`models_y[0]`, `models_t[0]`).
- Segment iteration is sorted by `segment_id`; CSV column order is fixed.
- Metadata excludes timings and binary hashes; the run manifest holds non-reproducible diagnostics.
- Model artifact bytes are not claimed deterministic; only the segment CSV and metadata are.
- EconML accumulates per-tree predictions under a lock when `n_jobs != 1`, so the fitted forest's inference is pinned to one thread after fitting; tree fitting remains parallel.

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| EconML internal nuisance arrays change shape across versions | Pin `econml 0.17.0` in the lockfile; validate `(1, cv_folds)`; raise typed error rather than guessing |
| `ate_inference` returns scalar vs. array across versions | `_as_scalar` normalization; Q2 already exercised `conf_int_mean` on 0.17.0 |
| Tiny test fixture yields degenerate intervals | Use ~2,000 rows and reduced but valid forest size; assert finiteness; increase fixture only if needed |
| q0 estimation bias | Deconfounded fold average (OD-1); Stage 9 coverage measures adequacy; do not tune on final seeds |
| Conditional ε interval undercoverage | Document limitation; Stage 9 `m3` is the explicit test; failures return to design, not threshold lowering |
| Model artifact is large | Gitignored `artifacts/`; never in serving package; hash recorded in run manifest |
| Module grows past 1,000 lines | Keep schemas separate; split inference helpers into `causal_inference.py` only if the module approaches the ceiling |

---

## 7. Execution Order

1. Config, exceptions, dependency, schema.
2. Core contract/split/fit functions with tests.
3. q0 estimation, aggregation, interval functions with tests.
4. Persistence and orchestration with tests.
5. CLI and structural tests.
6. Full-data execution and gate evidence.
7. Documentation updates and final validation.

See `tasks.md` for atomic tasks and commands.
