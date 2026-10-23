# Technical Plan: P2-F05 Segment Definition & Tier 1 Output Contract
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F05` (`005_segments-tier1-contract`)
**Spec:** `reports/docs/phases/phase_2/005_segments-tier1-contract/spec.md` — APPROVED October 21, 2026 (OD-1…OD-5)
**Status:** Ready for task decomposition
**Date:** October 21, 2026

---

## 1. Architecture and Components

Stage 4 sits between the Stage 3 GLM fitted values and the Stage 5 cap sweep / Stage 6 DGP / Stage 8 estimator.

```text
data/processed/features_glm.csv          (Stage 2: bands, exposure)
data/processed/glm_fitted_loss_cost.csv  (Stage 3: fitted loss cost per policy)
                 │
                 ▼
        segments (assign → merge → aggregate)
        ├── base 36-cell grid from frozen band labels
        ├── OD-1 merge: 18-24 / 5-9 / >2000 → 18-24 / 5-9 / >500 (2,234 policies)
        ├── exposure-weighted loss cost = Σ(μ·w) / Σw
        ├── base premium = loss cost × (1 + 0.20)
        ├── ratio-of-sums elasticity = Στ̂ / Σq₀
        └── outputs
             ├── data/processed/segment_loss_cost.csv      (35 segments)
             ├── data/processed/segment_definitions.json   (metadata + merge log)
             └── data/processed/policy_segments.csv        (413,169 mappings)
```

### Components

| Component | Responsibility | Location |
|---|---|---|
| Config schema | `min_policy_count`, `premium_loading`, explicit `cell_merges`, output paths; cross-validated against band labels | `params.yaml`, `src/aegis/config/schema.py` |
| Segment errors | Typed assignment/aggregation failure | `src/aegis/utils/exceptions.py` |
| Contract models | `Tier1SegmentOutput`, `Tier1SegmentTable`, `SegmentDefinition`, `CellMergeRecord`, `SegmentMetadata` | `src/aegis/schemas/segments.py` |
| Assignment + merge | Band → base cell → merged segment; deterministic IDs | `src/aegis/pipelines/segments.py` |
| Aggregation | Exposure-weighted loss cost, base premium, ratio-of-sums elasticity | `src/aegis/pipelines/segments.py` |
| Orchestration | Load/validate inputs, write three artifacts | `src/aegis/pipelines/segments.py` |
| CLI entrypoint | `build-segments` subcommand | `src/aegis/pipelines/cli.py` |
| DVC stage | `build_segments` after `glm_baseline` | `dvc.yaml` |
| Tests | Contract, falsification, aggregation, determinism, empirical | `tests/unit/test_segments.py`, `tests/unit/test_config.py`, `tests/unit/test_cli.py` |

`schemas/segments.py` is separate from `pipelines/segments.py` so the published contract can be imported by Stages 5/6/8/12 without importing pipeline I/O.

**Covers:** FR-7, FR-8, FR-10, FR-11, AC-6.

## 2. Data and Control Flow

1. Read `features_glm.csv` and `glm_fitted_loss_cost.csv`; validate required columns and that `PolicyID` sets are identical and complete.
2. Build base cell keys `(driver_age_band, car_age_band, density_band)`; validate every value is a frozen config label (FR-1).
3. Apply configured `cell_merges` (OD-1): rows in the `from` and `to` density bands of the specified `(driver, car)` cohort are re-labelled to `merged_density_band`; record `is_merged` (FR-2, FR-4).
4. Count policies per realized segment; if any segment `< min_policy_count`, raise `SegmentError` (FR-3).
5. Merge the two policy tables on `PolicyID`; compute per segment: `policy_count`, `exposure_years = Σw`, `loss_cost_per_year = Σ(μ·w)/Σw` (FR-5).
6. Compute `base_premium_per_year = loss_cost_per_year × (1 + premium_loading)` (FR-6).
7. Assemble the 35-row `segment_loss_cost.csv` in canonical `segment_id` order and the merge-log `segment_definitions.json`; validate with Pydantic before writing (FR-11).
8. Write `policy_segments.csv` (`PolicyID`, `segment_id`) sorted by `PolicyID` (FR-11).
9. Expose `aggregate_ratio_of_sums(tau, q0)` implementing `Στ̂/Σq₀` with finite/length/denominator validation (FR-9); Stage 8 consumes it.

No randomness; no wall-clock fields; `lineterminator="\n"` and rounded floats for byte-identical reruns (FR-12).

## 3. Configuration Surface (`params.yaml`)

```yaml
tier1_ml:
  segments:
    min_policy_count: 1000
    premium_loading: 0.20
    cell_merges:
      - driver_age_band: "18-24"
        car_age_band: "5-9"
        from_density_band: ">2000"
        to_density_band: "501-2000"
        merged_density_band: ">500"
  segment_paths:
    segment_loss_cost: "data/processed/segment_loss_cost.csv"
    segment_definitions: "data/processed/segment_definitions.json"
    policy_segments: "data/processed/policy_segments.csv"
```

Validation (`Tier1MLConfig` validator): every merge's `driver_age_band`, `car_age_band`, `from_density_band`, and `to_density_band` must exist in the corresponding `feature_pipeline` label lists; `merged_density_band` must be unique and distinct from existing density labels; `min_policy_count >= 1`; `premium_loading >= 0`. Rejections raise `ConfigurationError` at load.

**Covers:** FR-2, FR-3, FR-6, AC-3, AC-8.

## 4. Contract Design (`schemas/segments.py`)

```python
FORBIDDEN_SUBSTRINGS = ("delta", "adjust", "proposed", "new_premium")

class Tier1SegmentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_id: str
    policy_count: int = Field(ge=1000)
    exposure_years: float = Field(gt=0.0)
    baseline_retention_q0: float = Field(ge=0.0, le=1.0)
    tau_hat: float
    epsilon_hat: float
    epsilon_ci_lower: float
    epsilon_ci_upper: float
    loss_cost_per_year: float = Field(ge=0.0)
    base_premium_per_year: float = Field(ge=0.0)
    model_version: str
    data_version: str
    semi_synthetic: bool
```

`@model_validator(mode="before")` rejects any key containing a forbidden substring (case-insensitive), so renamed adjustment fields cannot enter. `@model_validator(mode="after")` enforces `lower <= epsilon_hat <= upper`, all finite. `policy_count` enforces the ADR-014 1,000-policy floor directly on the contract (`ge=1000`); the pipeline additionally re-asserts it during aggregation with a typed `SegmentError` so a floor violation is caught before serialization.

`Tier1SegmentTable` wraps `segments: list[Tier1SegmentOutput]` and rejects duplicate `segment_id`s.

**Covers:** FR-7, FR-8, FR-10.

## 5. Aggregation Algorithms

**Loss cost (ratio of exposure-weighted sums):**
`L_s = Σ_{i∈s} μ_i w_i / Σ_{i∈s} w_i`, with `μ_i = fitted_loss_cost_per_year`, `w_i = exposure_capped`.

**Base premium:** `P_s = (1 + loading) · L_s`.

**Elasticity (ratio of sums, Stage 8 contract):**
`ε_s = Σ_{i∈s} τ̂_i / Σ_{i∈s} q0_i`. Validate: equal lengths, no non-finite values, non-empty, `Σq0 > 0`. Computed in float; no per-policy division.

**Covers:** FR-5, FR-6, FR-9, AC-4.

## 6. Error Handling

`SegmentError(AEGISError)` with `rule`, `column`, `count`, `sample_keys`. Conditions: missing/extra columns, `PolicyID` mismatch, unknown band label, realized segment below floor, empty ratio-of-sums denominator, non-finite inputs, artifact validation failure. CLI returns 1; DVC fails. No bare `except`.

**Covers:** FR-3, FR-9, AC-9.

## 7. Test Strategy

| Test | Kind | Asserts |
|---|---|---|
| `test_tier1_output_rejects_proposed_adjustment` | Falsification (required) | Payload with `proposed_adjustment=0.05` raises `ValidationError` |
| `test_tier1_output_rejects_delta_variants` | Falsification | `delta`, `proposed_premium`, `new_premium`, `price_adjustment` all rejected |
| `test_tier1_output_accepts_valid_payload` | Contract | 13-field payload validates; round-trips |
| `test_tier1_output_rejects_inverted_interval` / `_estimate_outside_interval` | Contract | FR-10 |
| `test_ratio_of_sums_matches_hand_calculation` | Unit | `Στ/Σq0` on known vectors |
| `test_ratio_of_sums_rejects_zero_denominator` / `_length_mismatch` | Unit | Typed error |
| `test_merge_rule_produces_35_segments` | Unit | Empirical cell counts; merged segment 2,234 |
| `test_all_policies_map_to_exactly_one_segment` | Unit | 413,169 rows, no nulls |
| `test_base_premium_is_loaded_loss_cost` | Unit | `P = L(1+loading)` |
| `test_floor_violation_raises` | Unit | Synthetic thin cell after merge → `SegmentError` |
| `test_artifact_determinism` | Unit | Two runs byte-identical |
| `test_empirical_segment_pipeline` | Integration (skips if absent) | 35 segments, all ≥1,000; totals reconcile |

**Covers:** AC-1…AC-9.

## 8. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Merged label `>500` not in feature labels | Merge config validated at load; `is_merged` flag and merge log in metadata |
| Silent policy loss on join | Assert `PolicyID` sets equal and output row count == 413,169 |
| Float drift breaking determinism | Fixed rounding (6 dp) before write; stable sort order |
| Contract drift with Stage 8 | `aggregate_ratio_of_sums` and interval validators unit-tested now; Stage 8 reuses them |

## 9. Traceability

| FR | Task |
|---|---|
| FR-1, FR-2, FR-3, FR-4 | T3 |
| FR-5, FR-6 | T4 |
| FR-7, FR-8, FR-10 | T2 |
| FR-9 | T5 |
| FR-11, FR-12 | T6 |
| Falsification | T7 |
| Gate 4 | T8 |
