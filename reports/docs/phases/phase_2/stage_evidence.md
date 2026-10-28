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
| 5 | Qualification Q1: Loss Cap Sweep & Final Calibration | ✅ PASSED | 2026-10-24 | §Stage 5 |
| 6 | Semi-Synthetic DGP & Training Contract | ✅ PASSED | 2026-10-26 | §Stage 6 |
| 7 | Qualification Q2: Estimator Training Scale & Budget | ✅ PASSED | 2026-10-26 | §Stage 7 |
| 8 | Causal Estimator & Segment Inference | ✅ PASSED | 2026-10-26 | §Stage 8 |
| 9 | Validation Harness & Q3 Threshold Calibration | ✅ PASSED | 2026-10-26 | §Stage 9 |
| 10 | Final-Seed Validation Gate | ❌ FAILED | 2026-10-28 | §Stage 10 |
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

---

## Stage 5 — Qualification Q1: Loss Cap Sweep & Final Calibration

**Executed:** 2026-10-24 · **Gate 5:** ✅ PASSED · **Qualification:** Q1 (P2-D1a, ADR-012)

1. **Sweep (4 EDA candidates, GLM re-fit per cap, training-split metrics):**

   | Cap (€) | Loss removed (€) | Removed % | Claims truncated | Excess load | p̂ | Holdout deviance |
   |---|---:|---:|---:|---:|---:|---:|
   | 16,109 (P99) | 8,801,359 | 31.34% | 126 (1.03%) | 1.4565 | 1.81 | 16.3187 |
   | 35,432 (P99.5) | 7,045,693 | 25.09% | 67 (0.55%) | 1.3349 | 1.81 | 16.6300 |
   | 100,000 | 4,656,129 | 16.58% | 20 (0.16%) | 1.1988 | 1.84 | 16.9016 |
   | 131,834 (P99.9) | 4,087,724 | 14.56% | 14 (0.11%) | 1.1704 | 1.84 | 16.9702 |

2. **Selection criteria (adjacent-cap transitions):**

   | Transition | Spearman (segment loss-cost) | Holdout deviance change |
   |---|---:|---:|
   | 16,109 → 35,432 | 0.9972 | 1.908% (fail) |
   | 35,432 → 100,000 | 0.9927 | 1.633% (fail) |
   | 100,000 → 131,834 | 0.9992 | 0.406% (pass) |

   Pass criterion (Spearman ≥ 0.98 and deviance change ≤ 1%) is first satisfied at the
   €100,000 → €131,834 transition, so the **lowest truncation level selected is €100,000**.
   The Tweedie variance power is stable (p̂ = 1.81 below €100k, 1.84 above).

3. **Frozen config:** `tier1_ml.loss_cap = 100000.0`, `tier1_ml.excess_load = 1.1988`
   (uniform excess load = 1/(1−16.58%)). `glm_baseline.default_cap` already equals €100,000,
   so the final calibration report (`glm_calibration_report.json`, deviance 16.9016) is the
   Stage 3 baseline at the selected cap.

4. **Excess concentration (known limitation, operator decision 2026-10-24):** at the selected
   cap, segment `D:18-24_C:10+_Dens:<=500` (contains the €2.04M claim) holds **72.0% of excess
   loss against 1.14% of exposure (ratio 62.9×)**. The uniform load therefore understates
   `Losses₀` for that segment and overstates it elsewhere; recorded as a benchmark limitation
   carried to Phase 6 (operator chose uniform load + documented caveat).

5. **Report:** `reports/docs/phases/phase_2/q1_cap_sweep_report.md` (full per-cap table,
   transitions, selection, concentration caveat).

6. **Verification:**
   - `uv run pytest tests/unit/test_cap_sweep.py tests/unit/test_config.py tests/unit/test_cli.py -q`:
     **52 passed** (cap-sweep metrics, selection, config, CLI).
   - `uv run ruff check .`: clean; `uv run pyright`: 0 errors, 0 warnings.
   - `uv run python scripts/check_module_size.py`: all modules within 1,000 lines
     (`cap_sweep.py` 339).

### Gate 5 outcome

The sweep report is complete; the selected cap satisfies the Spearman ≥0.98 and deviance
≤1% criteria; the excess-load distribution was computed and its material concentration
recorded and escalated to the operator (uniform load + documented caveat); the cap and
excess load are frozen in `params.yaml`. Stage 5 PASSED; Stage 6 is unlocked.

---

## Stage 6 — Semi-Synthetic DGP & Training Contract

**Executed:** 2026-10-26 · **Gate 6:** ✅ PASSED · **Feature:** `P2-F04` (Full SDD)

1. **Specification and plan:**
   - `004_semi-synthetic-dgp/spec.md` APPROVED 2026-10-26 with OD-1 (bounded logistic retention $a(X)$ and price slope $b(X)$ with interaction), OD-2 (confounded pricing rule driven by standardized GLM loss cost plus noise), OD-3 (self-test thresholds: gap $\le 0.02$, overlap $\text{Var} \ge 0.001$, dispersion $\text{Std} \ge 0.010$, homogeneity $\max |\rho| \le 0.05$, bound $\max |\epsilon| \cdot 0.15 < 1.0$), and OD-4 (25 pre-treatment feature columns, `treatment_rate_change`, `renewed`, `segment_id`).
   - `plan.md` and `tasks.md` written before implementation; all 8 tasks completed.

2. **Automated DGP self-tests (empirical, 413,169 policies):**
   - **Estimand gap:** Numerical integration of average slope over $[-0.15, +0.15]$ vs closed-form $\tau$: max segment gap = **0.00544** $\le 0.02000$ (worst segment: `D:25-44_C:5-9_Dens:501-2000`). ✅ PASSED.
   - **Overlap:** Residual variance $\text{Var}(T \mid X) = \mathbf{0.00248} \ge 0.00100$ (residual std $\approx 0.0498$). ✅ PASSED.
   - **Dispersion:** Between-segment standard deviation of ground truth $\tau_{\text{seg}} = \mathbf{0.11557} \ge 0.01000$. ✅ PASSED.
   - **Within-segment homogeneity:** Max Pearson correlation between true elasticity $\epsilon_i$ and expected loss cost within segment = **0.00000** $\le 0.05000$ across all 35 segments (constant within segments by construction). ✅ PASSED.
   - **Bound guarantee:** Max $|\epsilon_i| \cdot 0.15 = \mathbf{0.12166} < 1.00000$ (max $|\epsilon_i| \approx 0.811$). Volume never floors at zero. ✅ PASSED.

3. **Data contracts and Falsification:**
   - Re-based `data_contracts/elasticity_training_suite.json` with 12 expectations enforcing exact 29 columns, unique `PolicyID`, $T \in [-0.15, +0.15]$, $Y \in \{0, 1\}$, and prohibiting all truth columns.
   - Full 413,169-row training table validated against suite: **12/12 expectations PASSED** (`data/validated/elasticity_training_validation_report.json`).
   - **Falsification 1:** Planted truth column (`tau_ground_truth`) into training fixture failed GX checkpoint loudly with `DataContractError` and identified prohibited column.
   - **Falsification 2:** AST import-boundary test verified that no module in `src/aegis` imports `truth_loader`; caught and halted injected import.

4. **Artifacts (SHA-256):**
   - `elasticity_training.csv` (45,779,288 B): `f08918631e891203f845f7107d9e7c33572d817b16f81d75b7f737424471cdb8`
   - `truth_sidecar.csv` (25,145,399 B): `21b1713ba4089c536aea6d0b47770fb8683cd10f26d5efd7d7e64a7e231cda1e`
   - `dgp_manifest.json` (1,805 B): `b735363459b337f0751ba56218e4c533205be8c92ce0afade475c3a164302121`
   - Config fingerprint: `09ff04b0df4aa448ff5b643f8d0ddab57700308c14a58d24d6d5c37991fbc887`
   - Two consecutive generation runs produced 100% bit-identical artifacts.

5. **DVC:** added `generate_dgp` and `validate_training_gx` stages after `build_segments`; `uv run dvc repro` ran all 11 stages cleanly.

6. **Verification:**
   - `uv run pytest -q`: **194 passed, 2 skipped** (showcase presets pending Stage 8/12 model registration).
   - `uv run ruff check .`: clean; `uv run pyright`: 0 errors, 0 warnings.
   - `uv run python scripts/check_module_size.py`: **38/38 modules** within 1,000 lines (`synthetic_dgp.py` 493 lines, `cli.py` 598 lines).

### Gate 6 outcome

All 5 automated DGP self-tests passed within stated tolerances; training-table Great Expectations contract passed the full 413,169-row dataset and rejected the truth-column fixture; truth loader import boundary is verified; three separate physical artifacts are produced deterministically. Stage 6 PASSED; Stage 7 (Qualification Q2: Estimator Scale & Compute Benchmark) is unlocked.

## Stage 7 — Qualification Q2: Estimator Training Scale & Budget

**Executed:** 2026-10-26 · **Gate 7:** ✅ PASSED · **Feature:** `P2-F06` (Q2 qualification)

1. **Experiment:** `uv run python scripts/q2_estimator_scale.py` completed all eight
   combinations of N ∈ {25,000, 100,000, 250,000, 413,169} and nuisance family
   (`gbt`, `linear`) with no skipped cells. Each N was stratified by `segment_id` and
   split 80/20; seed 42 was used for sampling, nuisance estimators, and CausalForestDML.
2. **Full-grid results:** 35/35 held-out segments were evaluated in each cell. Across the
   eight cells, wall times were 24.4–514.8 seconds, peak RSS was 601.3–2,792.3 MB,
   segment Spearman ranged from -0.1217 to 0.9031, and nominal 95% interval coverage was
   1.000 in every cell. The full-data GBT fit took 514.8 seconds (8.58 minutes) with
   2,534.3 MB peak RSS; the full-data linear fit took 171.4 seconds (2.86 minutes) with
   2,792.3 MB peak RSS. Both linear nuisance estimators used five-fold regularization CV;
   CausalForestDML also used five-fold cross-fitting.
3. **Budget and selection:** V8 reports 47.93 GB total RAM and a 38.34 GB (80%) ceiling;
   the run environment reported 49,077.6 MB RAM and a 39,262.1 MB ceiling. Both full-data
   fits were below the 30-minute and RAM limits. Full-data Spearman was 0.9014 (GBT) and
   0.9031 (linear); the 0.0017 gap is within the approved 0.02 tolerance, so the approved
   tie-break selects GBT at N=413,169 with cv_folds=5.
4. **Frozen parameters:** `params.yaml` now records `training_sample_size: 413169`,
   `nuisance_model: "gbt"`, `cv_folds: 5`, and `runtime_budget_minutes: 30.0`.
5. **Artifacts:** The full metrics, selection rationale, memory methodology, and SHA-256
   input hashes are in `q2_estimator_scale_report.md`.
6. **Verification:** `uv run pytest -q`: **203 passed, 2 skipped** (187 existing GX
   deprecation warnings). `uv run pyright`: **0 errors, 0 warnings**.
   `uv run python scripts/check_module_size.py`: all **38/38** source modules within the
   limit. `uv run ruff check scripts/q2_estimator_scale.py
   tests/unit/test_q2_estimator_scale.py tests/unit/test_config.py`: passed.
   Repository-wide `uv run ruff check .` reports an existing E501 in the unchanged
   `src/aegis/pipelines/truth_loader.py:78`; it is unrelated to Stage 7 and was not altered.
   The Q2 config loaded successfully; the real Stage 6 table validated as 29 columns with
   25 estimator features and no truth leakage. Regression tests cover EconML's N×1
   single-treatment effect output shape and reject truth columns at the estimator boundary.

### Gate 7 outcome

All eight cells completed; the full-data configuration met approved time and memory
budgets; GBT and the full 413,169-policy scale were selected and frozen. Stage 7 PASSED;
Stage 8 (Causal Estimator & Segment Inference) is unlocked.

---

## Stage 8 — Causal Estimator & Segment Inference

**Executed:** 2026-10-26 · **Gate 8:** ✅ PASSED · **Feature:** `P2-F06` (Full SDD)

1. **Specification and plan:**
   - `006_causal-estimator/spec.md`, `plan.md`, and `tasks.md` approved 2026-10-26 (OD-1 … OD-5 as recommended).
   - T1–T9 completed in order. No scope expansion; one reproducibility-only inference pin was added after the determinism diagnosis (plan §5). Implementation status is recorded in all three feature artifacts.
2. **Production pipeline (`src/aegis/pipelines/causal_estimator.py`, `causal_artifacts.py`, `src/aegis/schemas/causal.py`):**
   - Validates the exact 29-column Stage 6 contract, rejects all nine truth columns before fit, and enforces the 25-feature boundary.
   - Reproduces the frozen stratified 80/20 split and fits the Q2-frozen `CausalForestDML` on training rows only.
   - Estimates truth-free `q0` from cross-fitted nuisance models, aggregates held-out policies by segment, and builds segment-mean τ intervals via `ate_inference(...).conf_int_mean`; ε intervals are the τ interval scaled by the segment `q0` point estimate (OD-3).
   - Persists the segment CSV, deterministic metadata, non-promotable joblib artifact, and run-only manifest.
3. **Full-data execution:**
   - Command: `uv run python -m aegis.pipelines.cli train-estimator`
   - Model version: `cfdml-gbt-cv5-n100-leaf5-split10-depthnone-ms0.45-sub4-dataf08918631e89`
   - Rows: **330,535 training / 82,634 held-out**; segments: **35**.
   - `segment_elasticity_estimates.csv` (5,997 B): `SHA-256 48ee4db2c9557ec4778b9126a82bd633828c05731ffa818f2f3a401d02d75ad5`
   - `causal_estimator_metadata.json`: `SHA-256 60baa589aa6dbaa738217b29acc962ddcb12a1cd0653ec16e467261030de3ee4`
   - `artifacts/causal_estimator/estimator.joblib`: **290,221,911 B**; current run SHA-256 `a1585837a22df532890ab1753ca311c290d015547235593b96528d45792455b3`
   - `artifacts/causal_estimator/run_manifest.json`: `SHA-256 9e34c9aa92918d5e8a4a471bacaa1f00a086688558c54782cb3ca4f49a57e8f3`; fit **625.81 s**, pipeline **637.83 s**.
   - Peak process memory was not separately instrumented; no memory failure occurred, and the 290 MB artifact is non-promotable and gitignored.
4. **Gate 8 interval and estimate checks:**
   - 35 rows, exact nine-column contract, sorted by `segment_id`; all numeric values finite.
   - All 35 rows validate against `SegmentElasticityEstimate`; `segment_id` values are sorted and unique; metadata validates against `CausalEstimatorMetadata`.
   - `inference_policy_count` range **213–8,862**, sum **82,634**; held-out count is non-zero for every segment.
   - `baseline_retention_q0` range **0.807704–0.874939**.
   - `tau_hat` range **-0.877755…-0.064519**; `epsilon_hat` range **-1.085789…-0.074297**.
   - τ interval width range **2.540543–3.263501** (mean 2.824379); ε interval width range **2.914320–3.965046** (mean 3.318752).
   - 100% of τ and ε intervals contain their point estimate; `finite_interval_rate = 1.0` and `interval_self_consistency_rate = 1.0`.
   - Maximum per-row deviation from `epsilon_hat = tau_hat / baseline_retention_q0` is **2.22e-16**.
5. **AC-4 independent hand calculation:**
   - Segment: `D:18-24_C:0-4_Dens:501-2000`; 213 held-out policies.
   - Independent per-policy effects and fold-averaged `q0` from the persisted artifact:
     - `sum(tau_i) = -124.3663230298657`
     - `sum(q0_i) = 176.98368858901182`
     - `sum(tau_i) / sum(q0_i) = -0.7026993505523937`
   - Published CSV: `tau_hat = -0.5838794508444399`, `baseline_retention_q0 = 0.830909336098647`, `epsilon_hat = -0.7026993505523937`.
   - Absolute and relative difference: **0.0** (exact match).
6. **Determinism (AC-7):**
   - Initial full runs produced different segment CSVs (`0cb42a0347e43cb78974322d9fe2b04177c68e5ba2afddfb0a1e2fea46f3db4d` vs `97808435f09f23a128cdd540c89c591f097692e8ddc6c0eb31d4a3582245b946`).
   - Diagnosis: EconML accumulates per-tree predictions under a lock when `n_jobs != 1`; thread scheduling changed floating-point summation order by last-bit amounts. The 10k probe reproduced differing hashes with `n_jobs=-1` and identical hashes with `n_jobs=1`.
   - Fix: tree fitting remains parallel (`n_jobs=-1`); fitted-forest inference is pinned to one thread after fit (`_pin_forest_inference_to_one_thread`). The 10k probe then produced identical hashes on repeated runs (`0f2e616f763485c260b4cc08dc3caa189b961dd38cc2dd529710d3013b436cba`).
   - Final two consecutive full runs produced byte-identical segment CSVs: both `48ee4db2c9557ec4778b9126a82bd633828c05731ffa818f2f3a401d02d75ad5`.
   - Run-1 wall times were fit **497.96 s** / pipeline **509.94 s**; run-2 wall times were fit **625.81 s** / pipeline **637.83 s**. Both are within the 30-minute budget.
   - Joblib artifact hashes differ across runs (`e34ef8fb…` vs `a1585837…`); artifact bytes are run-only and are not claimed deterministic (plan §5). The CSV and metadata are the reproducibility contract.
7. **Isolation and persistence:**
   - `tests/unit/test_stage8_isolation.py`: 4 tests; API modules do not import the estimator, `econml`, `joblib`, or `pickle`; estimator modules do not import the truth loader or API; injected forbidden imports are detected.
   - Existing `tests/unit/test_truth_isolation.py` continues to scan all `src/aegis` modules and passes.
   - `artifacts/` remains gitignored (`.gitignore:43`); `data/processed/estimator.joblib` does not exist; the estimator artifact is present only under `artifacts/causal_estimator/`.
   - No truth column appears in the CSV, metadata, or manifest; metadata contains no timings or artifact hashes.
8. **Verification:**
   - `uv run pytest -q`: **287 passed, 2 skipped** in 368.34 s (187 existing GX deprecation warnings).
   - Stage 8 focused suite: **96 passed** in 16.81 s (`test_causal_estimator.py`, `test_cli.py`, `test_stage8_isolation.py`).
   - Final focused rerun including `test_truth_isolation.py`: **108 passed** in 22.75 s.
   - `uv run ruff check` on all changed Python files: **All checks passed**.
   - `uv run pyright`: **0 errors, 0 warnings, 0 informations**.
   - `uv run python scripts/check_module_size.py`: **41/41 modules** within 1,000 lines (`causal_estimator.py` 858, `causal_artifacts.py` 209, `schemas/causal.py` 155).
   - Repository-wide `uv run ruff check .` still reports the unchanged pre-existing `E501` at `src/aegis/pipelines/truth_loader.py:78`.

### Gate 8 outcome

The full-data estimator produced 35 segment-average estimates with 100% finite, ordered, self-consistent intervals; an independent hand calculation confirmed the ratio-of-sums identity exactly; two consecutive full runs produced byte-identical segment CSVs; serving and truth isolation tests pass; the fitted estimator is confined to gitignored non-promotable artifacts. Stage 8 PASSED; Stage 9 is unlocked.

---

## Stage 9 — Validation Harness & Q3 Threshold Calibration

**Executed:** 2026-10-26 · **Gate 9:** ✅ PASSED · **Feature:** `P2-F07` (Full SDD)

1. **Specification and plan:**
   - `007_validation-gate/spec.md`, `plan.md`, and `tasks.md` approved 2026-10-26 (OD-1 … OD-4).
   - OD-5 amended 2026-10-26: `m5` fixed at `0.90` and excluded from empirical separation after seed 42 showed estimator `1.0000`, `c1=0.9429`, `c2=1.0000` (gap `0.0000`).
   - T1–T7 completed in order.

2. **Implementation:**
   - Added `src/aegis/pipelines/validation_harness.py` (995 lines) and `src/aegis/schemas/validation.py` (162 lines).
   - Implements `m1`, `m1ε`, `m1b`, `m2`, `m3τ`, `m3ε`, `m4`, `m5`; controls `c1`/`c2`/`c3`; positive control; separation gaps; threshold freezing; markdown reporting.
   - Added `ValidationHarnessError`, `ValidationThresholdsConfig`, `pilot_seeds`, CLI commands `validate-causal` and `calibrate-q3`.
   - Truth isolation preserved: `validation_harness.py` is the only authorized consumer of `truth_loader.py`.

3. **Q3 execution:**
   - Command: `uv run python -m aegis.pipelines.cli calibrate-q3`
   - Pilot seeds: `[42, 101, 202]`; seed 42 reused the Stage 8 full-sample estimates, 101/202 were refit on 413,169 rows.
   - Estimator metrics:
     - Seed 42: `m1=0.8858`, `m1ε=0.8983`, `m1b=0.7853`, `m2=0.0976`, `m3/m4=1.0`, `m5=1.0`.
     - Seed 101: `m1=0.9479`, `m1ε=0.9447`, `m1b=0.8776`, `m2=0.0823`, `m3/m4=1.0`, `m5=0.971`.
     - Seed 202: `m1=0.9537`, `m1ε=0.9551`, `m1b=0.9342`, `m2=0.0909`, `m3/m4=1.0`, `m5=1.0`.
   - Negative controls all failed as required:
     - `c1`: `m1=0.6928`, `m2=0.1667`, `m5=0.9429`.
     - `c2`: `m1=-0.0402`, `m2=0.2114`, `m5=1.0000`.
     - `c3`: false-discovery rate `0.000`, `m1=-0.3727`, `m2=0.3900`, `m5=0.6857`.
   - Positive control passed: `m1=0.9152`, `m2=0.0727`, `m5=1.0000`. A reduced 100k/48-tree attempt recovered only `m1=0.4013` and was rejected; the full frozen configuration was used.
   - Separation gaps: `m1=0.1930`, `m1ε=0.1652`, `m1b=0.1568`, `m2=0.0692`; all exceed margins.

4. **Frozen thresholds (`params.yaml`):**
   - `m1_tau_spearman_floor=0.8428`, target `0.8954`
   - `m1_epsilon_spearman_floor=0.8831`, target `0.9139`
   - `m1b_policy_spearman_floor=0.7784`, target `0.8280`
   - `m2_mean_abs_bias_ceiling=0.1167`, target `0.1038`
   - `m3_tau_coverage_floor=0.90`, `m3_epsilon_coverage_floor=0.90`, `m4_self_consistency_floor=1.00`
   - `m5_sign_agreement_floor=0.90` (fixed absolute gate, OD-5)

5. **Verification:**
   - `uv run pytest -q`: **308 passed, 2 skipped** in 393.67 s (187 existing GX deprecation warnings).
   - Stage 9 focused suite: **17 passed** (`test_validation_harness.py`).
   - Config/CLI/truth boundary: **39 config + 28 CLI + 12 truth-isolation tests passed**.
   - `uv run ruff check .`: **All checks passed**.
   - `uv run pyright`: **0 errors, 0 warnings, 0 informations**.
   - `uv run python scripts/check_module_size.py`: **43/43 modules** within 1,000 lines (`validation_harness.py` 995, `config/schema.py` 476).

### Gate 9 outcome

Q3 executed across the three pilot seeds; all three negative controls failed; the positive control passed; `m1`, `m1ε`, `m1b`, and `m2` show separation above the approved margins; `m5` remains a fixed `0.90` gate under OD-5; thresholds are frozen in `params.yaml`. Stage 9 PASSED; Stage 10 final-seed validation is unlocked.

---

## Stage 10 — Final-Seed Validation Gate

**Executed:** 2026-10-27 (campaign), 2026-10-28 (verdict) · **Gate 10:** ❌ FAILED · **Exit E1:** NOT MET · **Feature:** `P2-F07`

1. **Implementation:**
   - Added `src/aegis/pipelines/final_seed_validation.py` and `src/aegis/schemas/final_validation.py`.
   - Added `run-final-validation` and `verify-final-validation` CLI commands; `final_seeds: [501, 502, 503, 504, 505]` and `synthetic_dgp.seed_offset_noise: 10000` in `params.yaml`.
   - Gate aggregation corrected before the verdict: `passes_floor` and `meets_target` are now distinct, worst-seed naming is recorded, and median targets are reported rather than gated. Under either framing the seed-level verdict is unchanged.
   - `final_seed_validation.py` is an offline validation module and is added to the truth-isolation allowlist alongside `validation_harness.py`; it never feeds truth values to an estimator, and it stays under the INV-8 ceiling by keeping DGP-table loading local rather than in `validation_harness.py` (995 lines, unchanged).

2. **Campaign:**
   - Command: `uv run python -m aegis.pipelines.cli run-final-validation`
   - K = 5 seeds `[501…505]`, disjoint from pilot seeds; each seed sets the stratified split and estimator `random_state`, with DGP noise at `seed + 10000`.
   - Artifact: `artifacts/causal_validation_summary.json`, 18,699 B, SHA-256 `a716acb72971e58959efdf0de168b0d2b18d950931ff473bd939b3181269e789`.
   - Frozen config SHA-256: `2eb6ff23488041e933d4d5cb00dbc6d481b9f4d2dccbc3e7c1ab0d1618fb5f18`.

3. **Result — 4 of 5 seeds breach the frozen floors:**

   | Seed | m1(τ) | m1(ε) | m1b | m2 | Floors |
   |---:|---:|---:|---:|---:|:---:|
   | 501 | 0.8737 | 0.8910 | 0.8854 | 0.0795 | ✅ |
   | 502 | 0.6813 | 0.7234 | 0.5775 | 0.0729 | ❌ |
   | 503 | 0.6796 | 0.7308 | 0.7621 | 0.0723 | ❌ |
   | 504 | 0.5517 | 0.6345 | 0.7308 | 0.0783 | ❌ |
   | 505 | 0.7762 | 0.7878 | 0.7017 | 0.0681 | ❌ |

   `m3`/`m4`/`m5` were `1.000` on every seed and `m2` stayed within ceiling throughout — the failure is confined to rank recovery.

4. **Root cause (measured):**
   - D1 probe refit on fixed seed-42 DGP tables with failing split/estimator seeds recovered `m1 = 0.9046 / 0.9295` → variation sits on the DGP-noise axis, not the split or estimator seed.
   - Independent noise-draw sweep (full n = 413,169, split seed 42): noise 42 `0.8858`, 10042 `0.8193`, 10101 `0.6956`, 10202 `0.8023`; on draw 10101 across split seeds `0.6956 / 0.5941 / 0.6587`.
   - `true_tau` is constant across draws (mean −0.3680, sd 0.0758, between-segment sd 0.1156), so the estimand does not move — only the realized noise does. **Noise draw 42 is the optimistic tail**, and it is the only draw Q3 calibrated on.
   - Bias (0.068–0.080) is 0.6–0.7× the between-segment spread, which is why ranking is fragile while bias, coverage and sign are stable.

5. **Recalibration rejected:**
   - Proposed amendment (3 noise draws × 3 seeds, unchanged separation gap, Δmin = 0.15) cannot pass: pooled gap is `−0.0987` (no separation); fresh-draws-only gap is `0.0317` (< margin); the per-draw paired variant separating correctly yields floor `≈0.7124`, still above four of five observed seeds.
   - Every passing variant required lowering a floor to fit the result — forbidden by P2-D6a. **Frozen thresholds were left unchanged.**

6. **Falsification:** threshold-tamper detection verified at unit level (`test_falsification_threshold_tamper_is_detected`, `test_cli.py` tamper case): edit a frozen threshold → verification exits 1; restore → exits 0. Mechanism valid; verdict unaffected.
   - **Disclosed limitation:** the persisted artifact is `artifact_version: 1.0`, written before the gate-aggregation correction, so `verify-final-validation` rejects it as `malformed_summary` (extra `median_seed_all_floors_cleared`; missing `median_seed_targets_met` and `campaign_passed`). It fails closed — this is schema drift, not a tamper result. The artifact is retained unmodified; the E1 verdict rests on its per-seed metric values (unaffected) and the unmodified frozen floors. Artifact-level tamper verification would require rerunning the campaign and would not change the verdict.

7. **Verification:**
   - `uv run pytest tests/unit/test_validation_harness.py tests/unit/test_final_seed_validation.py tests/unit/test_cli.py -q`: **63 passed**.
   - `uv run pytest` full suite: see commit record.
   - `uv run ruff check .`: **All checks passed**.
   - `uv run pyright`: **0 errors, 0 warnings, 0 informations**.

### Gate 10 outcome

❌ **FAILED.** Campaign executed exactly as specified; 4/5 seeds breach the frozen rank floors. Exit Criterion **E1 is NOT met**. The gate caught a real property — rank recovery is unstable across DGP-noise realizations — and the finding is recorded rather than tuned away. Escalated to the estimator-family decision (**P2-D5**). Stages 11–15 remain blocked; Phase 3 may not begin. Full analysis: `stage10_failure_analysis.md`.
