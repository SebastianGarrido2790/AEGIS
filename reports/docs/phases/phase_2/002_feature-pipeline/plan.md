# Technical Plan: P2-F02 Deterministic Feature Pipeline
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F02` (`002_feature-pipeline`)  
**Spec:** `reports/docs/phases/phase_2/002_feature-pipeline/spec.md` — APPROVED October 21, 2026  
**Status:** Ready for task decomposition  
**Date:** October 21, 2026

---

## 1. Architecture and Components

The feature pipeline sits between the Stage 1 raw contract and the Stage 3/6 consumers.

```text
data/versioned/raw_policy_data.csv   (Stage 1, GX-validated)
                │
                ▼
      feature_pipeline
      ├── input contract check (columns, dtypes, levels)
      ├── identity fields + exposure cap + tail clip
      ├── derive bands + log(Density)
      ├── GLM view  ────────► data/processed/features_glm.csv
      ├── forest view ──────► data/processed/features_forest.csv
      └── metadata ─────────► data/processed/feature_metadata.json
```

### Components

| Component | Responsibility | Location |
|---|---|---|
| Config schema | Typed exposure cap, age bounds, band edges, fixed level lists, output paths | `params.yaml`, `src/aegis/config/schema.py` |
| Feature errors | Typed pipeline failure carrying column, rule, count, sample keys | `src/aegis/utils/exceptions.py` |
| Metadata contracts | Pydantic models for clip log, view contract, distribution summary | `src/aegis/schemas/features.py` |
| Transform core | Deterministic derivation of both views | `src/aegis/pipelines/feature_pipeline.py` |
| CLI entrypoint | `build-features` subcommand | `src/aegis/pipelines/cli.py` |
| DVC stage | `build_features` between `version_raw` and Stage 3 | `dvc.yaml` |
| Tests | Unit/contract/determinism tests | `tests/unit/test_feature_pipeline.py` |

The package `src/aegis/schemas/` currently holds only `__init__.py`; adding one typed module is preferred over embedding Pydantic models in the pipeline module because the metadata is a persisted, versioned interface consumed by later stages.

**Covers:** FR-11, FR-13, AC-8, AC-9.

## 2. Data and Control Flow

1. Read the versioned raw CSV with `PolicyID` as a retained identifier.
2. Check the 13 expected columns, required dtypes, duplicate `PolicyID`, nulls, and category membership against fixed lists.
3. Copy identity fields (`PolicyID`, raw `DriverAge`, `CarAge`, `Density`).
4. Clip `Exposure` to 1.0 and log the count.
5. Clip `DriverAge` to `[18, 90]` and `CarAge` to `[0, 34]`; log counts by column and side.
6. Derive `log_density = ln(Density)` after checking `Density > 0`.
7. Derive DriverAge and CarAge bands from configured edges (left-closed, right-open).
8. Build the GLM view (categorical/banded) and the forest view (numeric/ordinal + one-hot).
9. Sort both views by `PolicyID`; write CSV with a fixed column order and `\n` line endings.
10. Hash the written bytes; assemble and write metadata; validate metadata against its Pydantic model before writing.

No randomness, no fitted statistics, and no timestamp enter the outputs. Metadata excludes wall-clock fields.

**Covers:** FR-1–FR-12, AC-1, AC-2, AC-6.

## 3. Interfaces and Contracts

### 3.1 Configuration additions

```yaml
tier1_ml:
  feature_pipeline:
    exposure_cap: 1.0
    driver_age_bounds: [18, 90]
    car_age_bounds: [0, 34]
    driver_age_band_edges: [18, 25, 45, 65, 100]
    driver_age_band_labels: ["18-24", "25-44", "45-64", "65+"]
    car_age_band_edges: [0, 5, 10, 101]
    car_age_band_labels: ["0-4", "5-9", "10+"]
    density_band_edges: [0, 501, 2001, 1000000000]
    density_band_labels: ["<=500", "501-2000", ">2000"]
    power_levels: ["d","e","f","g","h","i","j","k","l","m","n","o"]
    brand_levels: [...7 observed levels...]
    gas_levels: ["Regular", "Diesel"]
    region_levels: [...10 observed levels...]
  feature_paths:
    glm_view: "data/processed/features_glm.csv"
    forest_view: "data/processed/features_forest.csv"
    metadata: "data/processed/feature_metadata.json"
```

All lists and edges live in `params.yaml`. The Pydantic schema validates lengths, monotonic edges, and bound/edge consistency at load time. Edges are left-closed/right-open; final edges extend beyond observed maxima so no value is unbanded.

### 3.2 View contracts

**GLM view columns (fixed order):**
`PolicyID`, `exposure_capped`, `ClaimAmount`, `DriverAge`, `CarAge`, `Density`, `log_density`, `driver_age_band`, `car_age_band`, `density_band`, `Power`, `Brand`, `Gas`, `Region`.

`ClaimAmount` is carried only because the Stage 3 target (`capped pure premium`) needs it; it is not a covariate in this view's modeling contract.

**Forest view columns (fixed order):**
`PolicyID`, `exposure_capped`, `driver_age_clipped`, `car_age_clipped`, `log_density`, `driver_age_band_ord`, `car_age_band_ord`, `power_ord`, `brand__<slug>` ×7, `gas__<slug>` ×2, `region__<slug>` ×10.

`exposure_capped` is included as a baseline weight/offset, not as an elasticity covariate; the estimator's covariate list is the metadata `forest_covariates` field, which excludes it. Claims fields are absent by construction and by assertion.

**Metadata (JSON):**
`input_rows`, `output_rows`, `input_sha256`, `glm_view`, `forest_view`, `exposure_clip_count`, `tail_clip_counts`, `band_edges`, `level_mappings`, `one_hot_columns`, `forest_covariates`, `distribution_summary`, `config_fingerprint`.

### 3.3 Function signatures

```python
def transform_features(
    raw: pd.DataFrame,
    config: Tier1MLConfig,
) -> FeatureViews: ...

def write_feature_artifacts(
    views: FeatureViews,
    glm_path: Path,
    forest_path: Path,
    metadata_path: Path,
) -> FeatureMetadata: ...

def run_feature_pipeline(
    input_path: Path | str | None = None,
    glm_path: Path | str | None = None,
    forest_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
) -> FeatureMetadata: ...
```

`FeatureViews` is an internal frozen dataclass holding `glm`, `forest`, the fixed schema lists, the clip counts, and the config used. Public function arguments and returns are typed; no raw `dict` crosses the module boundary.

**Covers:** FR-5–FR-7, FR-11, FR-12, AC-3, AC-4, AC-8.

## 4. Algorithms and Rules

### 4.1 Encoding

| Factor | GLM view | Forest view |
|---|---|---|
| Power | original letter (categorical) | ordinal d=0 … o=11 |
| DriverAge | band label | band ordinal 0–3 |
| CarAge | band label | band ordinal 0–2 |
| Density | band label + `log_density` | `log_density` |
| Brand | original categorical | 7 one-hot columns |
| Gas | original categorical | 2 one-hot columns |
| Region | original categorical | 10 one-hot columns |

One-hot columns are generated from the **fixed configured level list**, not from observed values, so the schema cannot change between runs. `slug(value)` lowercases and replaces runs of non-alphanumerics with `_`; a startup check rejects duplicate slugs. Unknown levels fail; a configured level absent from a run yields an all-zero column, which is deterministic and schema-stable.

Discarded alternative: pandas `get_dummies` on observed categories. It silently changes the schema when a level is absent and creates an implicit unseen-level path, violating FR-13/AC-8. Rejected.

### 4.2 Clipping and logging

```text
n_exp = count(Exposure > 1.0); Exposure = min(Exposure, 1.0)
n_da_lo = count(DriverAge < 18);  DriverAge = max(DriverAge, 18)
n_da_hi = count(DriverAge > 90);  DriverAge = min(DriverAge, 90)
n_ca_lo = count(CarAge < 0);      CarAge = max(CarAge, 0)
n_ca_hi = count(CarAge > 34);     CarAge = min(CarAge, 34)
```

On the empirical input, `n_exp` must equal 421. Bounds are inclusive.

### 4.3 Bands

```text
driver_age_band = cut(clipped DriverAge, edges=[18,25,45,65,100], right=False)
car_age_band    = cut(clipped CarAge,    edges=[0,5,10,101],     right=False)
density_band    = cut(Density,           edges=[0,501,2001,1e9], right=False)
```

### 4.4 Distribution summary

For each view: row count; for numeric columns, count/min/max/mean/std/P01/P25/P50/P75/P99; for categorical columns, value counts. Rounded to six decimals for stability. The summary is descriptive only; no monitoring logic.

**Covers:** FR-1–FR-3, FR-8, FR-9, AC-2, AC-6, AC-7.

## 5. Determinism and Evidence

- Sort by `PolicyID` before writing.
- Fixed column order and fixed level order.
- `to_csv(index=False, lineterminator="\n")`.
- SHA-256 computed over written file bytes.
- Metadata has no timestamps; `config_fingerprint` is a hash of the relevant config subset.
- Gate evidence: two consecutive runs over the empirical input produce equal `glm_view.sha256` and `forest_view.sha256`; a test recomputes both hashes.

**Covers:** FR-10, FR-11, AC-1, AC-11.

## 6. Error Handling

Add `FeaturePipelineError(AEGISError)` with `details` carrying `column`, `rule`, `count`, and up to ten offending `PolicyID`s. Error conditions:

- missing/extra required columns (extra columns are rejected to prevent silent schema drift);
- duplicate `PolicyID`;
- any null in a required field;
- `Exposure <= 0`;
- `Density <= 0`;
- unknown category level;
- empty input;
- band edge/label mismatch (configuration error at load).

No `except` may swallow these into a successful return. The CLI returns exit code 1 and DVC halts.

**Covers:** FR-2, FR-13, AC-9.

## 7. Testing Strategy

| Layer | Test | Covers |
|---|---|---|
| Unit | Exposure cap: 421 empirical rows; max output 1.0 | FR-1, AC-2 |
| Unit | Invalid exposure/density/null/duplicate/unknown level raise `FeaturePipelineError` | FR-2, FR-13, AC-9 |
| Unit | `log_density == ln(Density)` | FR-3, AC-6 |
| Unit | Band boundaries at exact edges | FR-4, AC-5 |
| Unit | GLM view column contract | FR-5 |
| Unit | Forest view excludes all claims fields | FR-6, AC-3 |
| Unit | One-hot schema fixed when a level is absent | FR-7, AC-8 |
| Unit | Tail clip counts by column/side | FR-8, AC-7 |
| Unit | No composite columns; column whitelist equality | FR-9, AC-4 |
| Integration | Two runs → equal SHA-256 | FR-10, AC-1 |
| Integration | Metadata round-trips through Pydantic | FR-11, AC-8 |
| Integration | DVC `build_features` stage consumes the versioned raw file | FR-12 |
| Regression | Full empirical input transform completes and captures 421 clips | AC-2 |

Tests use small synthetic frames for rule isolation plus one empirical run for the exact counts. No test asserts a value equals itself; determinism is asserted by independent recomputation and hash comparison.

## 8. Security and Authority

- Input is untrusted data: schema, level whitelist, and numeric checks are enforced before transformation.
- No dynamic code execution, SQL, network access, or secret access.
- Output paths come from config/CLI, never from data content.
- The pipeline produces artifacts only; it does not approve, promote, or publish anything (PRODUCE ≠ APPROVE).
- No personal identifiers beyond the benchmark `PolicyID`; no external enrichment.

**Covers:** INV-3, INV-10.

## 9. Observability

- Console summary: input rows, output rows, clip counts, output paths.
- Durable evidence: metadata JSON with hashes and counts; DVC lock records artifact hashes.
- Errors include the offending column and sample keys.

## 10. Dependencies and Risks

| Risk | Mitigation |
|---|---|
| Pandas dtype inference differs across versions/OS | Pin lockfile; explicit `dtype`/conversion; tiered reproducibility contract (ADR-015) claims bit-identity only under same lockfile/OS |
| One-hot slug collision | Startup uniqueness check; fail loudly |
| Band-edge misconfiguration | Pydantic validates monotonicity and label count |
| Full-data memory | Two ~413k-row frames with <40 columns; fits available RAM |
| Schema drift from upstream | Exact column-set check; extra columns rejected |
| Future band changes | Edges are versioned config; metadata records the values used; downstream artifacts must be regenerated |

No new dependency.

## 11. Impact on Existing Systems

- `dvc.yaml`: new `build_features` stage; existing raw and regulatory stages unchanged.
- `params.yaml` / `config/schema.py`: additive configuration; existing keys unchanged.
- `cli.py`: new subcommand; existing routing unchanged.
- `pipelines/__init__.py`: optional exports; no behavior change.
- Tests: new file; existing tests unaffected.
- Stage 3/6: consume the persisted views; no feature logic duplication.

## 12. Deployment

None. Local DVC execution only. CI-smoke integration is deferred to Stage 14 by the execution plan.

## 13. Traceability

| Requirement | Component | Validation |
|---|---|---|
| FR-1, FR-2 | Transform core | T3/T4/T9 tests |
| FR-3, FR-8 | Transform core | T4/T9 tests |
| FR-4 | Config + band derivation | T1/T4 tests |
| FR-5 | GLM view builder | T5 test |
| FR-6, FR-7 | Forest view builder | T6 tests |
| FR-9 | Column whitelist | T7 test |
| FR-10, FR-11 | Writer + metadata | T7/T9 tests |
| FR-12 | CLI + DVC stage | T8 test |
| FR-13 | Input contract + typed errors | T2/T3 tests |
