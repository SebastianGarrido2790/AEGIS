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
| 2 | Deterministic Feature Pipeline | Pending | — | — |
| 3 | GLM Loss-Cost Baseline & Calibration Reporting | Pending | — | — |
| 4 | Segment Definition & Tier 1 Output Contract | Pending | — | — |
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
