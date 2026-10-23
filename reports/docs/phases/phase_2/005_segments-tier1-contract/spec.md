# Specification: P2-F05 Segment Definition & Tier 1 Output Contract
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F05` (`005_segments-tier1-contract`)
**Governing Decisions:** ADR-014 (segment definition, Tier 1 output contract, ratio-of-sums aggregation), P2-D3 (36-cell grid, 1,000-policy floor, merge rule, GLM loss-cost aggregation), ADR-005 (config-only tunables), INV-4 / INV-5 (no pricing adjustments in Tier 1)
**SDD Depth:** Full (`spec.md` → approval → `plan.md` → `tasks.md`)
**Status:** APPROVED — October 21, 2026 (OD-1…OD-5 as recommended)
**Date:** October 21, 2026

---

## 1. Context and Problem

Phase 3 (Underwriting Compliance Agent), Phase 6 (Financial & Revenue Impact Agent), and the Phase 2 showcase consume segment-level actuarial data rather than individual policy rows or library-bound ML estimators (ADR-014, P2-D3). To ensure cross-module auditability and eliminate circular dependencies:
1. Actuarial segments must be defined by observable pre-treatment rating factors, completely independent of the causal estimator under test.
2. The portfolio must be partitioned into segments of sufficient statistical power. Thin segments (< 1,000 policies) must be merged under a deterministic, documented actuarial rule.
3. Segment loss cost ($Losses_0$) must be aggregated as the exposure-weighted sum of GLM fitted values (capped and loaded), not raw noisy sample sums (P2-D3).
4. The output contract (`Tier1SegmentOutput`) must establish a strict structural boundary: Tier 1 models calculate baseline risk and elasticity, but **never propose price adjustments**. Any proposed adjustment field (`delta`, `proposed_premium`, `proposed_adjustment`) must be structurally rejected at schema validation time (INV-4, INV-5, ADR-014).

Stage 2 feature pipeline established the 4 DriverAge bands, 3 CarAge bands, and 3 Density bands ($4 \times 3 \times 3 = 36$ potential cells). Stage 3 fitted the Tweedie GLM and produced per-policy fitted loss costs. Stage 4 formalizes the segment mapping, resolves the cell merge rule, aggregates baseline loss cost and base premium, and locks the typed Pydantic contract.

---

## 2. Objective and Expected Outcome

Deliver a deterministic segment assignment and aggregation pipeline and typed Pydantic schema:
1. Map all 413,169 policies into the actuarial segment grid based on frozen config boundaries.
2. Execute the deterministic merge rule for any cell falling below the 1,000-policy floor, resulting in 35 valid segments with $\ge 1,000$ policies each.
3. Aggregate GLM fitted loss cost per segment via exposure-weighted means and compute technical base premium with configurable loading.
4. Implement ratio-of-sums aggregation for segment elasticity: $\hat{\epsilon}_{\text{seg}} = \sum \hat{\tau}_i / \sum q_{0i}$.
5. Enforce `Tier1SegmentOutput` schema with `extra="forbid"` and explicit rejection of proposed adjustment fields.
6. Falsification: prove dynamically that an injected `proposed_adjustment` payload fails Pydantic validation.

---

## 3. Scope

### In scope
- Deterministic 36-cell segment assignment from Stage 2 feature bands.
- Cell merge rule: merging cell `18-24 / 5-9 / >2000` (956 policies) into adjacent density band `501-2000` (combined `>500`, 2,234 policies).
- Config parameters in `params.yaml`: segment grid specifications, minimum policy floor (1,000), premium loading scalar, segment file paths.
- Exposure-weighted aggregation of GLM fitted loss cost per segment.
- Base premium computation: $\text{BasePremium} = \text{LossCost} \times (1 + \text{loading})$.
- Typed Pydantic v2 schemas: `Tier1SegmentOutput` and `Tier1SegmentTable` with `extra="forbid"`.
- Ratio-of-sums elasticity aggregation function $\hat{\epsilon}_{\text{seg}} = \sum \hat{\tau}_i / \sum q_{0i}$ and interval consistency checks.
- Artifact generation: `segment_loss_cost.csv`, `segment_definitions.json`, and `policy_segments.csv`.
- CLI subcommand `aegis build-segments` and DVC stage `build_segments`.
- Falsification test and unit/integration test suite.

### Out of scope
- Causal elasticity estimation ($\hat{\tau}$, $\hat{\epsilon}$) and confidence interval generation (Stage 8, P2-F06).
- Loss cap sweeping and final excess loading selection (Stage 5, Qualification Q1).
- Semi-synthetic DGP treatment assignment (Stage 6, P2-F04).
- Model registry and promotion to `approved` alias (Stage 12, ADR-014).
- Pricing strategy agent or bandit adjustment proposals (Tier 2 / Phase 6).

---

## 4. Functional Requirements (EARS)

**FR-1 (36-cell assignment).** When assigning policies to segments, the pipeline shall map each policy to its DriverAge band (4), CarAge band (3), and Density band (3) from the Stage 2 GLM view, defining an initial 36-cell grid ($4 \times 3 \times 3$).

**FR-2 (cell merge rule).** Where any cell in the initial 36-cell grid contains fewer policies than the configured floor (`min_policy_count: 1000`), the pipeline shall merge that cell into an adjacent band per the approved merge rule (OD-1): cell `18-24 / 5-9 / >2000` (956 policies) merges along `density_band` into adjacent `501-2000` (yielding combined density `>500` for that cohort, 2,234 policies).

**FR-3 (minimum policy floor).** Following execution of the merge rule, all realized segments shall contain $\ge 1,000$ policies. If any realized segment has $< 1,000$ policies, the pipeline shall raise a typed `SegmentError` and halt.

**FR-4 (segment identifier).** Each realized segment shall be assigned a deterministic, canonical string identifier (`segment_id`) following `D:{driver_age_band}_C:{car_age_band}_Dens:{density_band}` (e.g. `D:18-24_C:0-4_Dens:<=500`, and `D:18-24_C:5-9_Dens:>500` for the merged segment).

**FR-5 (loss-cost aggregation).** When aggregating loss cost to segment level, the pipeline shall compute the exposure-weighted mean of GLM fitted loss cost:
$$\text{LossCost}_{\text{seg}} = \frac{\sum_{i \in \text{seg}} \text{fitted\_loss\_cost}_i \times \text{exposure\_capped}_i}{\sum_{i \in \text{seg}} \text{exposure\_capped}_i}$$
using the Stage 3 GLM fitted values. Raw claim sums shall not be used (P2-D3).

**FR-6 (base premium calculation).** The pipeline shall compute segment base premium as $\text{BasePremium}_{\text{seg}} = \text{LossCost}_{\text{seg}} \times (1 + \text{loading})$, where `loading` is read from `params.yaml` (`tier1_ml.premium_loading`, default 0.20).

**FR-7 (Tier 1 Pydantic contract — fields).** The `Tier1SegmentOutput` Pydantic model shall define exactly the 13 fields mandated by ADR-014 and P2-D3:
- `segment_id`: `str`
- `policy_count`: `int` ($\ge 1000$)
- `exposure_years`: `float` ($> 0.0$)
- `baseline_retention_q0`: `float` ($0.0 \le q_0 \le 1.0$)
- `tau_hat`: `float`
- `epsilon_hat`: `float`
- `epsilon_ci_lower`: `float`
- `epsilon_ci_upper`: `float`
- `loss_cost_per_year`: `float` ($\ge 0.0$)
- `base_premium_per_year`: `float` ($\ge 0.0$)
- `model_version`: `str`
- `data_version`: `str`
- `semi_synthetic`: `bool` (strictly `True`)

**FR-8 (structural exclusion of adjustments — INV-4/INV-5).** The `Tier1SegmentOutput` model shall configure `extra = "forbid"` and shall execute a model validator rejecting any input containing keys matching `delta`, `adjust`, `proposed`, or `new_premium`. Any payload containing pricing proposals shall fail validation with a `ValidationError`.

**FR-9 (ratio-of-sums elasticity aggregation).** The pipeline shall implement the ratio-of-sums aggregation function:
$$\hat{\epsilon}_{\text{seg}} = \frac{\sum_{i \in \text{seg}} \hat{\tau}_i}{\sum_{i \in \text{seg}} q_{0i}}$$
The function shall validate that input sequences have equal non-zero lengths, all values are finite, and $\sum q_{0i} > 0$. It shall raise a typed error on non-positive denominator or mismatched lengths.

**FR-10 (interval self-consistency).** The `Tier1SegmentOutput` model shall validate that the reported 95% confidence interval is finite and encloses the point estimate: $\text{epsilon\_ci\_lower} \le \hat{\epsilon} \le \text{epsilon\_ci\_upper}$.

**FR-11 (artifacts).** The pipeline shall deterministically emit:
1. `segment_loss_cost.csv`: 35 realized segments with `segment_id`, `driver_age_band`, `car_age_band`, `density_band`, `is_merged`, `policy_count`, `exposure_years`, `loss_cost_per_year`, `base_premium_per_year`.
2. `segment_definitions.json`: metadata contract containing segment count, merge log, config fingerprint, total exposure, and total policies.
3. `policy_segments.csv`: per-policy table with `PolicyID` and `segment_id` (413,169 rows).

**FR-12 (reproducibility & determinism).** Repeated execution on the same inputs shall yield byte-identical CSV and JSON artifacts. No timestamp or nondeterministic keys shall be included in artifacts.

---

## 5. Data Contracts

### 5.1 Input Contracts
- `data/processed/features_glm.csv`: Stage 2 GLM view (413,169 rows, 14 columns including `PolicyID`, `exposure_capped`, `driver_age_band`, `car_age_band`, `density_band`).
- `data/processed/glm_fitted_loss_cost.csv`: Stage 3 GLM output (413,169 rows, including `PolicyID`, `exposure_capped`, `fitted_loss_cost`).
- `params.yaml`: `tier1_ml.segment_grid`, `tier1_ml.min_policy_count`, `tier1_ml.premium_loading`, `tier1_ml.segment_paths`.

### 5.2 Output Contracts

#### 1. `Tier1SegmentOutput` (Pydantic Model)
```python
class Tier1SegmentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(..., pattern=r"^D:[^_]+_C:[^_]+_Dens:[^_]+$")
    policy_count: int = Field(..., ge=1000)
    exposure_years: float = Field(..., gt=0.0)
    baseline_retention_q0: float = Field(..., ge=0.0, le=1.0)
    tau_hat: float
    epsilon_hat: float
    epsilon_ci_lower: float
    epsilon_ci_upper: float
    loss_cost_per_year: float = Field(..., ge=0.0)
    base_premium_per_year: float = Field(..., ge=0.0)
    model_version: str
    data_version: str
    semi_synthetic: bool = True
```

#### 2. `segment_loss_cost.csv`
Per-segment table (35 rows):
- `segment_id`: `str`
- `driver_age_band`: `str`
- `car_age_band`: `str`
- `density_band`: `str`
- `is_merged`: `bool`
- `policy_count`: `int` ($\ge 1000$)
- `exposure_years`: `float`
- `loss_cost_per_year`: `float`
- `base_premium_per_year`: `float`

#### 3. `segment_definitions.json` (`SegmentMetadata`)
Metadata JSON:
- `segment_count`: `int` (35)
- `initial_cell_count`: `int` (36)
- `min_policy_count`: `int` (1000)
- `premium_loading`: `float` (0.20)
- `merged_cells`: `list[CellMergeRecord]`
- `total_policies`: `int` (413169)
- `total_exposure_years`: `float`
- `portfolio_loss_cost_per_year`: `float`
- `portfolio_base_premium_per_year`: `float`
- `config_fingerprint`: `str` (SHA-256)

#### 4. `policy_segments.csv`
Policy-level mapping (413,169 rows):
- `PolicyID`: `int`
- `segment_id`: `str`

---

## 6. Edge Cases

| Case | Required Behavior |
|---|---|
| Cell has $< 1,000$ policies | Merged into adjacent density band per OD-1; merge logged in metadata. |
| Merged cell still has $< 1,000$ policies | Raise `SegmentError`; halt. (Not observed: French MTPL merged cell has 2,234 policies). |
| Unknown band label in policy data | Raise `SegmentError` identifying invalid row and column; halt. |
| Zero or negative exposure in segment | Raise `SegmentError` (exposure capped at 1.0, strictly positive from Stage 2). |
| Negative fitted loss cost | Impossible under Tweedie log link ($\mu = e^{\eta} > 0$); validated by schema $\ge 0.0$. |
| Elasticity denominator $\sum q_{0i} \le 0$ | Raise `SegmentAggregationError` in ratio-of-sums calculation. |
| Injected `proposed_adjustment` or `delta` | Pydantic `ValidationError` raised (Falsification control). |
| CI bounds inverted ($\text{lower} > \text{upper}$) | Pydantic `ValidationError` raised. |
| Point estimate outside CI | Pydantic `ValidationError` raised. |

---

## 7. Error Handling

A dedicated `SegmentError` (subclass of `AEGISError`) in `src/aegis/utils/exceptions.py` shall carry `rule`, `segment_id` (where applicable), `count`, and sample keys. Conditions:
- Cell count $< 1,000$ after merge.
- Missing required columns in input views.
- Policy count or PolicyID mismatch across input tables.
- Ratio-of-sums invalid inputs.
- Schema validation failure.

Errors propagate; CLI and DVC stage exit non-zero. No bare `except` may suppress errors.

---

## 8. Open Decisions

### OD-1 — Cell Merge Rule for Cells with $< 1,000$ Policies (FR-2)
- **Observation:** In the French MTPL dataset (413,169 policies), exactly ONE cell falls below the 1,000-policy floor:
  `driver_age_band == '18-24'`, `car_age_band == '5-9'`, `density_band == '>2000'` has **956** policies. All other 35 cells have $\ge 1,000$ policies (next smallest is 1,067).
- **Options:**
  - **Option A (Recommend): Merge along `density_band` into adjacent band `501-2000` (yielding combined density `>500` for that cohort).**
    - Resulting cell has $1,278 + 956 = 2,234$ policies.
    - Rationale: Driver age (18-24) is the single most critical actuarial risk differentiator; mixing young drivers into 25-44 would severely distort risk profile. Car age (5-9) is also a strong risk factor. Density is a geographic covariate where $>500$ is already urban/suburban. Merging density bands preserves age-cohort homogeneity. Total realized segments = 35.
  - **Option B: Merge along `car_age_band` into `10+` (or `0-4`).**
    - Resulting cell has $1,425 + 956 = 2,381$ policies. Mixes vehicle age cohorts.
  - **Option C: Merge along `driver_age_band` into `25-44`.**
    - Mixes young novice drivers with experienced drivers; actuarially unacceptable.
- **Recommendation:** **Option A.** Formalized as rule: merge `D:18-24_C:5-9_Dens:>2000` into `D:18-24_C:5-9_Dens:501-2000` to form `D:18-24_C:5-9_Dens:>500`.

### OD-2 — Segment Identifier Naming Format (FR-4)
- **Options:**
  - **Option A (Recommend): Structured string `D:{driver_age_band}_C:{car_age_band}_Dens:{density_band}`.**
    - E.g. `D:18-24_C:0-4_Dens:<=500`, and `D:18-24_C:5-9_Dens:>500` for the merged segment. Matches EDA notation in `implementation_plan.md` and `stage_evidence.md`. Human-readable and parseable.
  - **Option B: Hierarchical underscore string `{driver_age_band}_{car_age_band}_{density_band}`.**
    - E.g. `18-24_0-4_<=500`, `18-24_5-9_>500`.
  - **Option C: Integer code `seg_01`…`seg_35`.**
    - Opaque; requires secondary lookup.
- **Recommendation:** **Option A.**

### OD-3 — Premium Loading Default Value in `params.yaml` (FR-6)
- **Context:** Base premium is defined as $\text{BasePremium} = \text{LossCost} \times (1 + \text{loading})$. Loading covers insurer expenses, profit margin, and unmodeled tail risk (P2-D3).
- **Options:**
  - **Option A (Recommend): `tier1_ml.premium_loading: 0.20` (20%).** Standard actuarial expense+profit loading for French personal auto.
  - **Option B: `tier1_ml.premium_loading: 0.15` (15%).**
  - **Option C: `tier1_ml.premium_loading: 0.25` (25%).**
- **Recommendation:** **Option A (0.20).** Configured in `params.yaml`, never hardcoded.

### OD-4 — Scope of Stage 4 Artifacts and Pipeline Integration (FR-11)
- **Context:** `Tier1SegmentOutput` schema requires causal inference fields (`baseline_retention_q0`, `tau_hat`, `epsilon_hat`, `epsilon_ci_lower`, `epsilon_ci_upper`) that will be produced by Stage 8. What artifacts are generated in Stage 4 right now?
- **Options:**
  - **Option A (Recommend): Persist segment definitions, loss costs, and policy mappings now:**
    1. `segment_loss_cost.csv` (35 rows with `loss_cost_per_year` and `base_premium_per_year` aggregated from GLM fitted values).
    2. `segment_definitions.json` (metadata, merge log, config fingerprint).
    3. `policy_segments.csv` (413,169 rows mapping `PolicyID` to `segment_id`).
    4. Typed `Tier1SegmentOutput` Pydantic model with validation and helper methods ready for Stage 8.
    - This directly unlocks Stage 5 (loss cap sweep), which requires segment loss costs to compute Spearman rank correlation across caps!
  - **Option B: Schema only in Python; defer CSV generation to Stage 8.**
    - Blocks Stage 5 from computing segment loss-cost rank correlations.
- **Recommendation:** **Option A.**

### OD-5 — Structural Exclusion Enforcement in `Tier1SegmentOutput` (FR-8)
- **Options:**
  - **Option A (Recommend): Dual defense — Pydantic v2 `extra="forbid"` PLUS explicit field validator rejecting forbidden substrings (`delta`, `adjust`, `proposed`, `new_premium`).**
    - Guarantees neither extra fields nor maliciously renamed fields can pass.
  - **Option B: `extra="forbid"` alone.**
- **Recommendation:** **Option A.**

---

## 9. Traceability Matrix

| Requirement | Governing Artifact / ADR | Implemented In |
|---|---|---|
| FR-1 (36 cells) | ADR-014, P2-D3 | `src/aegis/pipelines/segments.py` |
| FR-2 (merge rule) | ADR-014, P2-D3 | `src/aegis/pipelines/segments.py` |
| FR-3 (policy floor) | ADR-014, P2-D3 | `src/aegis/pipelines/segments.py`, `Tier1SegmentOutput` |
| FR-4 (segment_id) | ADR-014 | `src/aegis/pipelines/segments.py` |
| FR-5 (loss-cost agg) | ADR-014, P2-D3 | `src/aegis/pipelines/segments.py` |
| FR-6 (base premium) | ADR-014, P2-D3 | `src/aegis/pipelines/segments.py` |
| FR-7 (schema fields) | ADR-014, P2-D3 | `src/aegis/schemas/segments.py` |
| FR-8 (exclusion INV-4) | INV-4, INV-5, ADR-014 | `src/aegis/schemas/segments.py`, tests |
| FR-9 (ratio of sums) | ADR-014 | `src/aegis/pipelines/segments.py` |
| FR-10 (CI check) | ADR-013, ADR-014 | `src/aegis/schemas/segments.py` |
| FR-11 (artifacts) | ADR-014, ADR-015 | `src/aegis/pipelines/cli.py`, `dvc.yaml` |
| FR-12 (determinism) | ADR-015 | tests |
| Falsification | Stage 4 execution plan | `tests/unit/test_segments.py` |
| Gate 4 | Stage 4 execution plan | Stage 4 validation gate |
