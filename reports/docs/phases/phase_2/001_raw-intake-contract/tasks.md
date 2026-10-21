# Tasks: P2-F01 Raw Intake & Raw Data Contract
## Phase 2 — Tier 1: Deterministic ML Baseline (Light SDD)

**Feature ID:** `P2-F01` (`001_raw-intake-contract`)  
**Parent Decisions:** ADR-015 (dataset identity, raw contract, checksum manifest), ADR-003 amendment (two elasticity-side GX suites)  
**SDD Depth:** Light (`tasks.md` with inline specification)  
**Status:** Completed (Stage 1 Passed)

---

## 1. Inline Specification

### 1.1 Objective & Boundary
Enforce INV-3 at the very first pipeline node: raw observational French MTPL data cannot be versioned or processed by downstream feature engineering without:
1. Cryptographic integrity verification against a committed SHA-256 checksum manifest.
2. Deterministic policy-level joining of frequency and severity records with sum aggregation of multiple claims per policy.
3. Passing a Great Expectations contract (`data_contracts/raw_policy_suite.json`) enforcing 20 empirical rules grounded in the Stage 0 EDA findings.

### 1.2 Raw Inputs & Checksum Manifest
- **Source Files:**
  - `data/raw/freMTPLfreq.csv.zip`: 413,169 policies, 10 frequency columns.
    - Expected SHA-256: `40ace1e7d8b862fa3e46cbe9fd4ea4f6fdae2b228e0fdf3aa0f8ddf53ada4138`
    - Expected Size: `4,012,714` bytes
  - `data/raw/freMTPLsev.csv`: 16,181 claims, 2 severity columns (`PolicyID`, `ClaimAmount`).
    - Expected SHA-256: `1748e198091c61275315cc11202ddca5362f1f303b34d8b1ec29697f8497a410`
    - Expected Size: `199,180` bytes
- **Manifest Location:** `data_contracts/raw_intake_manifest.json`
- **Failure Behavior:** Any missing file or SHA-256 checksum mismatch raises typed `DataContractError` and halts execution immediately.

### 1.3 Policy-Level Joining Logic
1. Group `sev` by `PolicyID`:
   - `ClaimAmount = sum(ClaimAmount)` (aggregates multi-claim policies, accounting for all 757 policies with 2–4 claims).
   - `sev_claim_count = count(ClaimAmount)`
2. Merge `freq` and aggregated `sev` via outer join on `PolicyID`:
   - For non-claiming policies (`PolicyID` in `freq` but not in `sev`): `ClaimAmount = 0.0`, `sev_claim_count = 0`, `is_orphan_severity = False`.
   - For claiming policies (`PolicyID` in both): `ClaimAmount = sum(ClaimAmount)`, `sev_claim_count = count`, `is_orphan_severity = False`.
   - For orphan severity policies (`PolicyID` in `sev` but not in `freq`): frequency columns are null, `is_orphan_severity = True`.
3. Output dataset schema:
   `[PolicyID, ClaimNb, Exposure, Power, CarAge, DriverAge, Brand, Gas, Region, Density, ClaimAmount, sev_claim_count, is_orphan_severity]`.

### 1.4 Raw Policy Great Expectations Suite (`raw_policy_suite.json`)
The suite defines 20 critical expectations:
1. `expect_table_columns_to_match_set`: Enforce exact column set above.
2. `expect_column_values_to_be_unique`: `PolicyID` has zero duplicates.
3. `expect_column_values_to_be_between`: `ClaimNb` in `[0, 4]`.
4. `expect_column_values_to_be_between`: `Exposure` in `(0.0, 2.0]` (`strict_min: true`).
5. `expect_column_values_to_be_between`: `Exposure` in `(0.0, 1.0]` with `mostly: 0.998` (tolerating the 0.102% truncation anomaly).
6. `expect_column_values_to_be_between`: `ClaimAmount >= 0.0` across all policies.
7. `expect_column_values_to_be_between`: `ClaimAmount > 0.0` (`strict_min: true`) conditioned on `ClaimNb > 0`.
8. `expect_column_pair_values_to_be_equal`: `ClaimNb` matches `sev_claim_count` for every policy.
9. `expect_column_values_to_be_in_set`: `is_orphan_severity` must be `[false]` for all records.
10. `expect_zero_orphan_severity_records`: Custom domain expectation verifying zero orphan severity records.
11–20. `expect_column_values_to_not_be_null`: Zero nulls across all 10 frequency columns and metadata (`PolicyID`, `ClaimNb`, `Exposure`, `Power`, `CarAge`, `DriverAge`, `Brand`, `Gas`, `Region`, `Density`).

### 1.5 Fixtures Suite (`data_contracts/fixtures/`)
- `raw_policy_valid.csv`: Clean synthetic dataset passing 20/20 expectations.
- `raw_policy_invalid_duplicate_id.csv`: Duplicate `PolicyID`.
- `raw_policy_invalid_claim_nb_range.csv`: `ClaimNb = 5`.
- `raw_policy_invalid_exposure_range.csv`: `Exposure = 0.0` or `Exposure = 2.5`.
- `raw_policy_invalid_exposure_tolerance.csv`: 20% of rows exceeding `Exposure = 1.0`.
- `raw_policy_invalid_claim_amount_zero.csv`: Claimed policy with `ClaimAmount = 0.0`.
- `raw_policy_invalid_claim_count_mismatch.csv`: `ClaimNb = 1` but `sev_claim_count = 2`.
- `raw_policy_invalid_orphan_severity.csv`: Orphan severity row (`is_orphan_severity = True`).
- `raw_policy_invalid_null_frequency.csv`: Missing frequency values (`Power = null`).

### 1.6 Falsification Protocol
Inject an orphan severity record into a raw fixture and run `validate_raw_gx`. The runner must fail specifically on referential integrity (`expect_zero_orphan_severity_records` and `is_orphan_severity in [false]`) and exit with status code 1, halting downstream DVC execution.

---

## 2. Implementation Tasks

- [x] **Task 1: Intake Checksum Manifest**
  - Create `data_contracts/raw_intake_manifest.json` recording SHA-256 and size for `freMTPLfreq.csv.zip` and `freMTPLsev.csv`.
  - Update `params.yaml` and `src/aegis/config/schema.py` to declare `raw_intake_manifest_path`, `raw_policy_suite_path`, and `raw_policy_valid_fixture`.

- [x] **Task 2: Raw Intake & Joining Module**
  - Implement `src/aegis/pipelines/raw_intake.py` with:
    - `verify_checksums()`
    - `load_and_join_raw_policy_data()`
    - `ingest_raw_data()` runner

- [x] **Task 3: Custom GX Referential Integrity Expectation**
  - Implement `ExpectZeroOrphanSeverityRecords` in `src/aegis/pipelines/expectations.py`.
  - Register in `src/aegis/pipelines/data_contracts.py`.

- [x] **Task 4: Raw Policy GX Contract Suite**
  - Define `data_contracts/raw_policy_suite.json` containing the 20 rules.
  - Implement `validate_raw_policy_data()` in `src/aegis/pipelines/data_contracts.py`.

- [x] **Task 5: Valid and Malformed Fixtures**
  - Generate `raw_policy_valid.csv` and all 8 targeted malformed fixtures in `data_contracts/fixtures/`.

- [x] **Task 6: DVC DAG Wiring & CLI Subcommands**
  - Add `ingest-raw`, `validate-raw`, `version-raw` to `src/aegis/pipelines/cli.py`.
  - Wire `ingest_raw → validate_raw_gx → version_raw` in `dvc.yaml`.

- [x] **Task 7: Comprehensive Unit Tests & Falsification Verification**
  - Add `tests/unit/test_raw_policy_data_contract.py`.
  - Verify clean pass on valid data, specific failures on all 8 malformed fixtures, checksum tamper rejection, and orphan severity pipeline blockage.

- [x] **Task 8: Verification & Gate 1 Sign-Off**
  - Run full suite: `pytest`, `ruff`, `pyright`, `check_module_size.py`, and `dvc repro`.
