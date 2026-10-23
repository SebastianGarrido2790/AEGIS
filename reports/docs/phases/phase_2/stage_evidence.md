# Phase 2 — Stage Evidence Log
## Tier 1: Deterministic ML Baseline (EDA-First Redo)

Author: Sebastián Garrido Arévalo | Date: October 21, 2026
**Status: ACTIVE.** Evidence is appended per stage as gates are evaluated. This log is the detailed record of executed outcomes; the Execution Plan (`execution_plan.md`) holds stage goals, actions, gates and falsification requirements, and links here for the evidence.

**What this document is:** the evidence ledger for Phase 2 stages. It records live command outputs, checksums, measured values, decisions taken on evidence, and gate outcomes exactly as observed.
**How to use it:** read the relevant stage section before starting dependent work; add a new dated section per executed stage. Do not rewrite earlier entries — append corrections as new dated notes.

| Stage | Title | Gate | Date | Evidence |
|---|---|---|---|---|
| 0 | Pre-Implementation Verification & Dependency Check | ✅ PASSED | 2026-10-21 | §Stage 0 |
| 1 | Raw Intake & Raw Data Contract | ✅ PASSED | 2026-10-21 | §Stage 1 |
| 2 | Deterministic Feature Pipeline | ✅ PASSED | 2026-10-21 | §Stage 2 |
| 3 | GLM Loss-Cost Baseline & Calibration Reporting | ✅ PASSED | 2026-10-21 | §Stage 3 |
| 4 | Segment Definition & Tier 1 Output Contract | ✅ PASSED | 2026-10-21 | §Stage 4 |
| 5 | Qualification Q1: Loss Cap Sweep & Final Calibration | Pending | — | — |
| 6 | Semi-Synthetic DGP & Training Contract | Pending | — | — |
| 7 | Qualification Q2: Estimator Training Scale & Budget | Pending | — | — |
| 8 | Causal Estimator & Segment Inference | Pending | — | — |
| 9 | Validation Harness & Q3 Threshold Calibration | Pending | — | — |
| 10 | Final-Seed Validation Gate | Pending | — | — |
| 11 | Hidden-Confounder Sensitivity Analysis | Pending | — | — |
| 12 | MLflow Artifact Packaging & Human-Gated Promotion | Pending | — | — |
| 13 | Diagnostic Showcase Slice | Pending | — | — |
| 14 | DAG Assembly, Config & CI Tiers / Reproducibility | Pending | — | — |
| 15 | ADR Consolidation & Phase 2 Sign-Off | Pending | — | — |

---

## Stage 0 — Pre-Implementation Verification & Dependency Check

**Executed:** 2026-10-21 · **Gate 0:** ✅ PASSED · **Production code modified:** none

1. **Git status & starting commit:**
   - Working tree verified clean on branch `main`.
   - Starting commit hash recorded: `04903e67bcdcd60c87958430732e1d7a1f4554c9` (`04903e6 docs(phase_2): add staged execution plan`).
2. **V1 (Verification Suite Baseline):**
   - `uv run pytest -q`: **42 passed, 2 skipped** in 19.53s (skips: `test_showcase_renders_default_registered_model_scenario` and `test_showcase_renders_each_curated_preset` explicitly awaiting Phase 2 model re-training).
   - `uv run ruff check .`: **All checks passed** (0 errors).
   - `uv run pyright`: **0 errors, 0 warnings, 0 informations**.
   - `uv run python scripts/check_module_size.py`: **26/26 Python files within 1,000-line ceiling** (INV-8 held; maximum file size is `data_contracts.py` at 254 lines).
   - `uv run dvc repro`: **All 6 stages reproduced cleanly**; elasticity suite passed (12/12 expectations); regulatory corpus suite passed (9/9 expectations).
3. **V2 (Codebase Inventory):**
   - `src/aegis/pipelines`: Scaffold CLI, expectation definitions, and contract runners only (`cli.py`, `data_contracts.py`, `expectations.py`). **Zero Tier 1 modeling or training code exists on `main`.**
   - `src/aegis/tools`, `schemas`, `bandit`, `gateway`, `governance`, `agents`: Scaffold directories with empty `__init__.py` files.
   - `src/aegis/api`: Retained showcase presentation files (`app.py`, `model_service.py`, `scenarios.py`, `static/showcase.css`, `templates/showcase.html`). Verified as UI/demo harness, not Tier 1 training logic.
4. **V3 (Data Contract Schema Comparison):**
   - Compared `data_contracts/elasticity_training_suite.json` and `data_contracts/fixtures/elasticity_valid.csv` against empirical freMTPL schema.
   - Finding: Phase 1 elasticity suite was built for synthetic mock columns (`premium`, `treatment_rate_change`, `annual_mileage`) without categorical rating factors.
   - Routing: Stage 1 will implement a dedicated raw policy suite (`data_contracts/raw_policy_suite.json`) tailored to empirical freMTPL columns; Stage 6 will re-base the training suite for post-DGP covariates, treatment, and binary outcome.
5. **V4 (DVC Raw Input Sanitization):**
   - Inspected `dvc.yaml` and `data/raw/`.
   - Removed confirmed-unused placeholder files: `data/raw/elasticity_fremtpl2.csv` and `data/raw/elasticity_raw.csv`.
   - `data/raw/freMTPLfreq.csv.zip` and `data/raw/freMTPLsev.csv` are confirmed as the sole raw observational data inputs.
6. **V5 (Dependency Resolution on Python 3.12 / Windows):**
   - Verified via `uv run python -c "import econml, sklearn, statsmodels, mlflow, fastapi, jinja2"`:
     - `econml`: 0.17.0
     - `scikit-learn`: 1.9.0
     - `statsmodels`: 0.14.6
     - `mlflow`: 3.15.2
     - `fastapi`: 0.141.1
     - `jinja2`: 3.1.6
   - All core modeling, causal inference, and serving packages resolve without binary compatibility issues.
7. **V6 (EDA Regeneration & Checksum Audit):**
   - Re-ran `scripts/explore_raw_data.py --output-json artifacts/eda_summary_new.json --quiet`.
   - Verified 100% bit-identical data values against existing `artifacts/eda_summary.json` (sole difference: elapsed runtime metric `1.11` vs `1.111`).
   - Checksum of canonical `artifacts/eda_summary.json` recorded:
     `SHA-256: 7b3f9940b2f714e5bb3e87be9161569b408ebbcb025baf6e7734216566e83fb3`.
8. **V7 (Audit Branch Accessibility):**
   - Verified reference branch `archive/phase2-attempt-1` is present and accessible at commit `e5cbfb7`.
9. **V8 (Development Hardware Resource Audit):**
   - OS: Windows 11
   - CPU: 12 logical cores
   - Total physical RAM: **47.93 GB**
   - Available physical RAM: **32.19 GB**
   - 80% RAM budget ceiling for Stage 7 (Q2): **38.34 GB** (with 30-minute runtime budget).
10. **V9 (MLflow Alias Tracking):**
    - Executed live SQLite test using `MlflowClient(tracking_uri="sqlite:///...")`.
    - Successfully registered model, created versions, assigned `candidate`, and created/reassigned `approved` alias.
11. **Open EDA Gaps Closed (`implementation_plan.md` §2.4):**
    - *Raw unbinned distributions:*
      - `DriverAge`: $\min=18, \max=99, P_{01}=20.0, P_{25}=34.0, P_{50}=44.0, P_{75}=54.0, P_{99}=81.0$. (Confirms 18–24, 25–44, 45–64, 65+ grid bands have robust empirical support; no invalid driver ages below 18).
      - `CarAge`: $\min=0, \max=100, P_{01}=0.0, P_{25}=3.0, P_{50}=7.0, P_{75}=12.0, P_{99}=22.0$. (Bands 0–4, 5–9, 10+ balance quartiles cleanly).
      - `Density`: $\min=2, \max=27000, P_{01}=10.0, P_{25}=67.0, P_{50}=287.0, P_{75}=1410.0, P_{99}=27000.0$. (Confirms extreme right-tail skew; $\ln(\text{Density})$ is mandatory).
      - `Power`: Categories `d` (16.5%), `e` (18.6%), `f` (23.2%), `g` (22.1%) represent 80.4% of policies.
    - *€2.04M claim & excess loss distribution:*
      - Largest claim in dataset: `PolicyID = 101091`, `ClaimAmount = €2,036,833.00` (5.91% of total portfolio losses).
      - Policy characteristics: `DriverAge = 19`, `CarAge = 13`, `Density = 93`, `Region = Centre`, `Power = i`.
      - Mapped segment: **`D:18-24_C:10+_Dens:<=500`**.
      - Impact: This single segment represents only **1.14% of portfolio exposure**, but accounts for **66.37% of excess losses** at cap €100,000, and **36.44% of excess losses** at cap €16,109. This directly explains the distorted severity relativities observed in the raw data EDA and provides empirical grounding for Stage 5 (Q1) and showcase Scenario S3.
    - *Pairwise Spearman rank correlations:*
      - All pairwise correlations among rating factors are low ($|\rho| < 0.23$), confirming factors provide independent variation without severe collinearity:
        - `DriverAge` vs `CarAge`: $\rho = -0.0652$
        - `DriverAge` vs `Density`: $\rho = -0.0626$
        - `CarAge` vs `Density`: $\rho = -0.1738$
        - `Density` vs `Region`: $\rho = +0.2279$
        - `CarAge` vs `Exposure`: $\rho = +0.1710$
        - `Density` vs `Exposure`: $\rho = -0.1876$
    - *Exposure shortfall analysis:*
      - 29.44% of policies have full-year exposure ($\text{Exposure} == 1.0$); 31.23% are in $[0.95, 1.0]$.
      - 421 policies (0.102%) exceed 1.0 (max 1.99).
      - 16.16% of policies have $\text{Exposure} < 0.10$ (~36.5 days).
      - Policies with $\text{Exposure} < 0.10$ exhibit annualized claim frequency of $0.1837$ vs $0.0556$ for full-year policies, but raw claim probability per policy is low ($0.0106$ vs $0.0538$). This confirms that short-duration policies reflect calendar-year observation window truncation (e.g. policies initiated late in the year) rather than early customer lapse or churn.
      - **Outcome:** Assumption A1 and Decision P2-D2a are confirmed: the policy record is the uniform analysis unit; synthetic renewal is defined uniformly; observed exposure is strictly used as an offset/weight for the GLM loss-cost model and not as a causal retention factor.
12. **Q2 Sample-Size Definition (Operator Decision 2026-10-21):**
    - The operator formally resolved the Stage 0 blocking clarification: **Total slice with 80/20 split**.
    - For each Q2 evaluation level $N \in \{25\text{k}, 100\text{k}, 250\text{k}, 413\text{k}\}$, the fixed 20% held-out test split is applied uniformly, resulting in:
      - $N = 25\text{k} \implies 20,000$ training / $5,000$ held-out test
      - $N = 100\text{k} \implies 80,000$ training / $20,000$ held-out test
      - $N = 250\text{k} \implies 200,000$ training / $50,000$ held-out test
      - $N = 413\text{k} \implies 330,535$ training / $82,634$ held-out test
    - This preserves the full dataset size at the top level while strictly maintaining out-of-sample evaluation across all Q2 cells.

### Gate 0 outcome

All V1–V9 checks executed and documented; all four EDA gaps resolved with empirical metrics; Q2 sample-size definition locked with operator; baseline test/lint state clean; zero production code modified. Stage 1 unlocked at commit `04903e6`.

---

## Stage 1 — Raw Intake & Raw Data Contract

**Executed:** 2026-10-21 · **Gate 1:** ✅ PASSED · **Feature:** `P2-F01` (Light SDD)

1. **Light SDD Specification:**
   - Completed `reports/docs/phases/phase_2/001_raw-intake-contract/tasks.md` with inline specification, schemas, expectation catalog, and task verification.
2. **Cryptographic Checksum Manifest (`data_contracts/raw_intake_manifest.json`):**
   - Manifest created and verified against disk with exact hashes and file sizes:
     - `data/raw/freMTPLfreq.csv.zip`: `40ace1e7d8b862fa3e46cbe9fd4ea4f6fdae2b228e0fdf3aa0f8ddf53ada4138` (size: 4,012,714 bytes; 413,169 rows; 10 columns).
     - `data/raw/freMTPLsev.csv`: `1748e198091c61275315cc11202ddca5362f1f303b34d8b1ec29697f8497a410` (size: 199,180 bytes; 16,181 rows; 2 columns).
   - `verify_checksums()` verified live against disk; tampered file test confirmed loud failure with typed `DataContractError`.
3. **Policy-Level Joining Logic (`src/aegis/pipelines/raw_intake.py`):**
   - Merged 413,169 frequency policies and 16,181 severity records on `PolicyID`.
   - Aggregated multiple claims per policy via `sum(ClaimAmount)` across all 757 policies with 2–4 claims.
   - Non-claiming policies (397,779) initialized with `ClaimAmount = 0.0` and `sev_claim_count = 0`.
   - Referential integrity confirmed on empirical data: 0 orphan severity records (`is_orphan_severity == False` for all 413,169 rows).
4. **Great Expectations Raw Policy Contract (`data_contracts/raw_policy_suite.json`):**
   - Implemented 20 critical expectations:
     - Table columns match set (10 frequency factors + `ClaimAmount`, `sev_claim_count`, `is_orphan_severity`).
     - Unique `PolicyID` (0 duplicates).
     - `ClaimNb` between 0 and 4.
     - `Exposure` strictly in (0.0, 2.0].
     - `Exposure` <= 1.0 with `mostly: 0.998` (accommodates 0.102% truncation anomaly).
     - `ClaimAmount` >= 0.0 across table.
     - Conditional `ClaimAmount` > 0.0 (`strict_min: true`) when `ClaimNb > 0`.
     - `ClaimNb` equals `sev_claim_count` for every policy (`expect_column_pair_values_to_be_equal`).
     - `is_orphan_severity` in `[false]`.
     - Custom domain expectation: `expect_zero_orphan_severity_records` (`ExpectZeroOrphanSeverityRecords`).
     - Non-null checks across all 10 frequency columns and metadata.
   - Evaluated on full empirical data (413,169 rows): **20/20 expectations PASSED in 1.4s**.
5. **Fixtures Suite (`data_contracts/fixtures/`):**
   - `raw_policy_valid.csv`: passes 20/20 expectations cleanly.
   - 8 targeted malformed fixtures independently fail their specific expectation:
     - `raw_policy_invalid_duplicate_id.csv` -> fails `expect_column_values_to_be_unique on 'PolicyID'`
     - `raw_policy_invalid_claim_nb_range.csv` -> fails `expect_column_values_to_be_between on 'ClaimNb'`
     - `raw_policy_invalid_exposure_range.csv` -> fails `expect_column_values_to_be_between on 'Exposure'`
     - `raw_policy_invalid_exposure_tolerance.csv` -> fails `mostly: 0.998 on 'Exposure'`
     - `raw_policy_invalid_claim_amount_zero.csv` -> fails conditional `ClaimAmount > 0`
     - `raw_policy_invalid_claim_count_mismatch.csv` -> fails `expect_column_pair_values_to_be_equal`
     - `raw_policy_invalid_orphan_severity.csv` -> fails `expect_zero_orphan_severity_records` and `is_orphan_severity`
     - `raw_policy_invalid_null_frequency.csv` -> fails `expect_column_values_to_not_be_null on 'DriverAge'`
6. **Selective Falsification Executed:**
   - Injected orphan severity fixture `data_contracts/fixtures/raw_policy_invalid_orphan_severity.csv` into `validate_raw_gx`.
   - Checkpoint specifically caught referential integrity violations:
     `expect_zero_orphan_severity_records on column 'table-level'`,
     `expect_column_values_to_be_in_set on column 'is_orphan_severity'`,
     `expect_column_pair_values_to_be_equal on column 'table-level'`.
   - Pipeline halted with exit code 1; `version_raw` raised `DataContractError: Cannot version unvalidated or failed data (INV-3)`.
   - Restored valid dataset; re-validated clean pass.
7. **DVC DAG Wiring (`dvc.yaml`):**
   - Wired `ingest_raw → validate_raw_gx → version_raw`.
   - `uv run dvc repro`: All 6 stages reproduced cleanly, promoting validated `data/versioned/raw_policy_data.csv`.
8. **Verification & Standards:**
   - `uv run pytest -q`: 56 passed, 2 skipped.
   - `uv run ruff check .`: clean (0 errors).
   - `uv run pyright`: 0 errors, 0 warnings.
   - `uv run python scripts/check_module_size.py`: 27/27 modules within 1,000-line ceiling.

### Gate 1 outcome

Raw intake manifest verified cryptographically; deterministic policy joining correctly aggregates multiple claims; raw policy GX contract passes 20/20 on empirical data; all 8 malformed fixtures fail specifically; orphan severity falsification halts pipeline; DVC DAG reproduces cleanly. Stage 1 PASSED; Stage 2 is unlocked.

---

## Stage 2 — Deterministic Feature Pipeline

**Executed:** 2026-10-21 — **Gate 2:** ✅ PASSED — **Feature:** `P2-F02` (Full SDD)

1. **Specification and plan:**
   - `002_feature-pipeline/spec.md` approved 2026-10-21 with OD-1 (DriverAge `[18, 90]`, CarAge `[0, 34]`) and OD-2 (one-hot unordered factors).
   - `plan.md` and `tasks.md` written before implementation; all 10 tasks completed.
2. **Configuration:**
   - `params.yaml` and `FeaturePipelineConfig` now declare exposure cap 1.0, age bounds, three band specs, and fixed level lists (Power 12, Brand 7, Gas 2, Region 10).
   - Pydantic rejects non-monotonic edges, label/edge mismatch, duplicate levels, and invalid bounds. Tests cover each rejection.
3. **Input contract:**
   - Exact 13-column set enforced; extra columns rejected to prevent schema drift.
   - Duplicate `PolicyID`, nulls, non-positive `Exposure`/`Density`, unknown levels, non-numeric columns, and empty input raise typed `FeaturePipelineError` with rule, column, count, and sample keys.
4. **Derivations (empirical run, 413,169 rows):**
   - Exposure clipped to 1.0: **421 records**.
   - DriverAge clipped to `[18, 90]`: **0 below / 307 above**.
   - CarAge clipped to `[0, 34]`: **0 below / 410 above**.
   - `log_density = ln(Density)` verified against independent computation (max abs diff < 1e-12).
   - Bands are left-closed/right-open: DriverAge 4, CarAge 3, Density 3.
5. **Dual views:**
   - GLM view: **14 columns** (identity, capped exposure, ClaimAmount, raw ages, log density, three bands, four categorical factors).
   - Forest view: **27 columns** (identity, capped exposure, clipped ages, log density, three ordinal codes, Power ordinal, **19 fixed one-hot columns**).
   - Forest covariates exclude `PolicyID`, `exposure_capped`, and all claims fields (`ClaimNb`, `ClaimAmount`, `sev_claim_count`, `is_orphan_severity`).
   - No composite-index column exists in either view (whitelist test).
6. **Determinism and artifacts:**
   - Two independent full runs produced identical view hashes:
     - GLM: `9ffe469b83da4522e0afa64fec733eef62fa1dea121e15cef591e3de48f58b95`
     - Forest: `688dceff7515882a7a26ee0bc3b344fb8bad3108ee16741c6606b4f33379321e`
   - Input SHA-256: `09249ac77dcb7b43e6efbcaf8f93fd481d976820684cd6d2251558596bcb702d`
   - Config fingerprint: `c054757bc17f9b1574cee80b0bbef1bd82bf20854e121f7fc0640c9f34d63da3`
   - Artifacts: `features_glm.csv` (45,979,423 B), `features_forest.csv` (32,589,134 B), `feature_metadata.json` (10,204 B).
   - Metadata round-trips through `FeatureMetadata`; no timestamps.
7. **DVC:**
   - Added `build_features` after `version_raw`; it depends on the versioned raw table, config, and feature modules.
   - `uv run dvc repro`: all 7 stages clean; `build_features` executed once, raw GX gate remained upstream.
8. **Verification:**
   - `uv run pytest -q`: **92 passed, 2 skipped** (skips remain the showcase model-dependent tests). 187 non-failing GX `condition_parser` deprecation warnings.
   - `uv run ruff check .`: clean.
   - `uv run pyright`: 0 errors, 0 warnings.
   - `uv run python scripts/check_module_size.py`: **29/29 modules** within 1,000 lines (`feature_pipeline.py` 669).

### Gate 2 outcome

Both views reproduce byte-identically; all clips and exclusions are counted and recorded; no composite index exists; claims variables are provably absent from the forest covariates; transformer metadata is versioned and validated. Stage 2 PASSED; Stage 3 is unlocked.

---

## Stage 3 — GLM Loss-Cost Baseline & Calibration Reporting

**Executed:** 2026-10-21 · **Gate 3:** ✅ PASSED · **Feature:** `P2-F03` (Full SDD)

1. **Specification and plan:**
   - `003_glm-baseline-loss-treatment/spec.md` APPROVED 2026-10-21 with OD-1 (profile-deviance `p` selection with `p=1.5` fallback), OD-2 (all seven rating factors, drop-first reference coding), OD-3 (default cap €100,000), OD-4 (ratio-of-sums A/E; deciles ranked on predicted pure premium), OD-5 (JSON+CSV only, no pickle).
   - `plan.md` and `tasks.md` written before implementation; all 8 tasks completed.
2. **Fit (empirical, 413,169 rows):**
   - Target = `min(ClaimAmount, cap)/exposure_capped`, frequency weights = `exposure_capped`, Tweedie log link.
   - Seeded 80/20 split (`random_state=42`): **train 330,535 / test 82,634**.
   - Selected variance power `p = 1.84` via profile deviance (21 grid points, refinement around the minimizer); **`fallback = False`**; dispersion 1641.32; **`converged = True`**.
   - Profile minimizer: `p=1.84` (train deviance 5,523,352.58); nearest neighbors `1.85` (5,559,782), `1.80` (5,577,123).
3. **Calibration (holdout test split):**
   - Holdout mean Tweedie deviance: **16.9016**.
   - Decile A/E (ratio of sums): 1.608, 1.111, 0.815, 0.683, 0.912, 1.447, 0.893, 1.146, 1.143, 0.730.
   - One-way segment A/E: DriverAge 18-24 0.626 / 25-44 1.123 / 45-64 0.938 / 65+ 1.319; CarAge 0.719–1.042; Density ≤500 0.948 / 501-2000 1.045 / >2000 1.186 (plus the 36-cell cross).
   - Secondary diagnostics (reference only, not gates): MAE 136.59, RMSE 1141.62.
4. **Single-model invariant:** no statsmodels Poisson or Gamma family anywhere in the fit path (structural source assertion); Tweedie is the sole combined model (P2-D1).
5. **Artifacts (SHA-256):**
   - `glm_baseline_model.json` (3,942 B): `7bbe17fd926a39a7cc09f29f4949374991b2d50f6b02b359daa7d7169820a360`
   - `glm_fitted_loss_cost.csv` (13,717,080 B): `9be5875d34dc7062bad740dcf4dd2fc63aa812b5ba0d0b6eab9e42e81a7946b9`
   - `glm_calibration_report.json` (11,196 B): `4fae22711d9a1905c7dfd0b9394607c85f03d6dde8ae737ba5f4a89e5809b9a9`
   - Config fingerprint: `fd39a517612d01cbb27ec24b985f72b8a5631ab175c0637c129eb9b03a55ad88`
   - Artifacts round-trip through `GLMBaselineMetadata` / `GLMCalibrationReport`; no timestamps.
6. **DVC:** added `glm_baseline` after `build_features`; `uv run dvc repro` ran all 8 stages cleanly.
7. **Verification:**
   - `uv run pytest -q`: **116 passed, 2 skipped** (187 non-failing GX deprecation warnings).
   - `uv run ruff check .`: clean; `uv run pyright`: 0 errors, 0 warnings.
   - `uv run python scripts/check_module_size.py`: **32/32 modules** within 1,000 lines (`glm_baseline.py` 409, `glm_calibration.py` 143, `schemas/glm.py` 75).

### Gate 3 outcome

The Tweedie GLM converged without numerical errors; holdout mean deviance and decile/segment A/E ratios were generated and persisted; the variance power is documented with a recorded profile-deviance table; MAE/RMSE are present only as secondary diagnostics and are referenced by no gate. Stage 3 PASSED; Stage 4 is unlocked.

---

## Stage 4 — Segment Definition & Tier 1 Output Contract

**Executed:** 2026-10-21 · **Gate 4:** ✅ PASSED · **Feature:** `P2-F05` (Full SDD)

1. **Specification and plan:**
   - `005_segments-tier1-contract/spec.md` APPROVED 2026-10-21 with OD-1 (merge thin cell along density into `>500`), OD-2 (canonical `D:_C:_Dens:` identifiers), OD-3 (`premium_loading = 0.20`), OD-4 (persist all three tables now), OD-5 (dual defense against adjustment fields).
   - `plan.md` and `tasks.md` written before implementation; all 8 tasks completed.
2. **Cell counts (empirical, 413,169 policies):**
   - Base grid: **36 cells** (4 DriverAge × 3 CarAge × 3 Density).
   - Exactly one cell below the 1,000 floor: `18-24 / 5-9 / >2000` = **956**.
   - Approved merge pools it with adjacent `501-2000` into `D:18-24_C:5-9_Dens:>500` = **2,234** policies.
   - Realized: **35 segments**; minimum segment size **1,067**; no segment below 1,000.
3. **Economics:**
   - Exposure-weighted segment loss cost (GLM fitted, capped at €100,000): portfolio **€126.53 per policy-year**; segment range €83.18–€507.19.
   - Base premium = loss cost × 1.20: portfolio **€151.84 per policy-year**.
   - Total exposure 231,780.21 policy-years; all 413,169 policies mapped.
4. **Tier 1 contract:**
   - `Tier1SegmentOutput` carries the 13 ADR-014 fields with `extra="forbid"`.
   - Dual defense: pre-validation scan rejects keys containing `delta`, `adjust`, `proposed`, or `new_premium`; interval and finiteness validators enforce `ci_lower ≤ ε̂ ≤ ci_upper` and `semi_synthetic = true`.
   - Falsification: `proposed_adjustment: 0.05` raises `ValidationError`; `delta`, `proposed_premium`, `new_premium`, `price_adjustment`, `rate_adjustment` all rejected.
5. **Ratio-of-sums:** `ε̂_seg = Στ̂ / Σq₀` verified against hand calculation; empty, length-mismatch, non-finite, and non-positive-denominator inputs raise typed `SegmentError`.
6. **Artifacts (SHA-256):**
   - `segment_loss_cost.csv` (3,160 B): `319c371dd69d13bedbad8e1f0782153f0dde81ab5da0e6e4b18335386aa96758`
   - `segment_definitions.json` (629 B): `079407763318d30e8cf99a5568602ea6a24cf160d21b143d4d778455917edf8f`
   - `policy_segments.csv` (13,254,695 B): `396574cb5f49e651a0b2d2410156a6523bb62801a36ade240c1e0b5d9a4fd3ba`
   - Config fingerprint: `55ce36728dbdb617ec9332c1aaa32938bf4e93c70872f24a7bd370657bed4bcb`
   - Two consecutive runs produced byte-identical artifacts; no timestamps.
7. **DVC:** added `build_segments` after `glm_baseline`; `uv run dvc repro` ran all 9 stages cleanly.
8. **Verification:**
   - `uv run pytest -q`: **153 passed, 2 skipped** (187 non-failing GX deprecation warnings).
   - `uv run ruff check .`: clean; `uv run pyright`: 0 errors, 0 warnings.
   - `uv run python scripts/check_module_size.py`: **34/34 modules** within 1,000 lines (`segments.py` 479, `schemas/segments.py` 163, `schema.py` 306, `cli.py` 536).

### Gate 4 outcome

All policies map to exactly one realized segment and every segment meets the 1,000-policy floor after the approved merge; ratio-of-sums aggregation passes unit tests; Pydantic validation structurally blocks `proposed_adjustment` and all adjustment-like fields; a typed error halts any floor violation. Stage 4 PASSED; Stage 5 is unlocked.


