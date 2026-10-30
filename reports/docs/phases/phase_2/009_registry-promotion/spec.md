# Specification: P2-F09 MLflow Artifact Packaging & Human-Gated Promotion
## Phase 2 — Tier 1: Deterministic ML Baseline

**Feature ID:** `P2-F09` (`009_registry-promotion`)  
**Governing Decisions:** ADR-014 (promotable segment-level artifact, SQLite tracking, human-gated alias promotion, plain data serving), P2-D11 (promotable unit scope; fitted estimator retained as non-promotable run artifact), P2-D11a (SQLite tracking backend, mutable `candidate`/`approved` aliases, human review verification), ADR-005 (config keys in `params.yaml`), INV-4 / INV-5 (no pricing adjustments in Tier 1), INV-7 / INV-10 (complete audit lineage, no live carrier data)  
**SDD Depth:** Full (`spec.md` → operator approval → `plan.md` → `tasks.md`), lean depth  
**Status:** APPROVED — October 30, 2026 (OD-1, OD-2, OD-3 as recommended)  
**Date:** 2026-10-30  

---

## 1. Context and Problem

Downstream consumers (Phase 3 Underwriting Compliance, Phase 6 Financial & Revenue Impact, and Phase 2 Showcase Slice) require an authoritative, auditable source for actuarial segments and baseline causal elasticity metrics (ADR-014, P2-D11).

In Phase 2 attempt 1 and standard ML setups, two anti-patterns arose:
1. **Serialized Estimator Serving Fragility:** Serving code deserialized pickled scikit-learn or EconML estimator objects, introducing fragile library-version coupling, arbitrary code execution risks (pickle), and training-serving skew for tabular lookup tasks.
2. **Ungated Promotion / Silent Drift:** Automated pipelines promoted models directly to active status without human review, calibration verification, or complete cryptographic provenance.

P2-D11 and P2-D11a establish a governed, auditable artifact lifecycle:
- The promotable unit is a **plain-data segment output package** with its complete calibration, validation, and source lineage.
- The fitted `CausalForestDML` estimator is preserved as a **non-promotable run artifact** for auditability, but is **never deserialized in serving paths**.
- Automated pipelines are restricted to assigning the `candidate` alias.
- Promotion to `approved` is strictly gated on human review: requiring a verified reviewer ID, a matching calibration report hash, and complete lineage tags.
- Rollback is supported by re-pointing the mutable `approved` alias to a previously approved version.

---

## 2. Objective and Expected Outcome

Deliver an MLflow artifact packaging pipeline and CLI promotion tool:
1. Package Tier 1 outputs into an immutable MLflow run artifact bundle conforming to the `Tier1SegmentOutput` contract (ADR-014).
2. Log complete audit lineage metadata: Git commit hash, DVC data hash, DGP manifest ID, MLflow run ID, validation-gate pass tag, and calibration report SHA-256.
3. Automatically register the model under `aegis-causal-elasticity` and assign the `candidate` alias.
4. Implement `scripts/promote_model.py` enforcing human sign-off: rejecting promotion unless provided a valid reviewer ID, a matching calibration report hash, and all lineage tags.
5. Implement rollback functionality to re-point `approved` to a prior model version.
6. Provide a verified plain-data serving loader (`load_approved_segments`) that reads strictly from the `approved` MLflow alias without importing or deserializing estimator objects.
7. Satisfy Roadmap Exit Criterion **E4** and Gate 12 hard close.

---

## 3. Scope

### In scope
- MLflow tracking configuration with SQLite backend (`sqlite:///artifacts/mlflow.db`) injected via `MLFLOW_TRACKING_URI` (ADR-014, P2-D11a).
- Packaging pipeline (`src/aegis/pipelines/registry.py`):
  - Segment output table (`tier1_segments.csv` / `Tier1SegmentTable`).
  - GLM calibration report (`glm_calibration_report.json`) and loss cap manifest.
  - Causal validation summary (`causal_validation_summary.json`).
  - Sensitivity sweep summary (`sensitivity_summary.json`).
  - Non-promotable training run artifact: serialized estimator model logged to run artifacts only.
- Registration and automated `candidate` alias assignment on pipeline execution.
- Human-gated promotion script (`scripts/promote_model.py` and `aegis promote-model` CLI):
  - Validates reviewer ID and calibration report hash match.
  - Validates required lineage tags (`git_commit`, `dvc_hash`, `dgp_manifest_id`, `gate_pass`).
  - Assigns mutable alias `approved` upon successful review.
- Rollback command: re-points `approved` alias to a specified previous version.
- Serving interface (`src/aegis/api/model_service.py`): resolves and loads plain segment data and gate metadata from the `approved` alias.
- Falsification test: automated attempt to assign `approved` without human review credentials or with mismatched calibration hash fails closed with an explicit typed error.

### Out of scope
- Live inference execution from serialized forest models (serving is plain data lookup per P2-D11).
- Rate adjustment proposals or bandit optimization (Phase 3 / Phase 6).
- Automated live deployment to external clouds (ADR-008, local SQLite first).
- Remote MLflow server orchestration (local SQLite backend governed by `params.yaml`).

---

## 4. Functional Requirements (EARS)

**FR-1 (tracking backend).** When initializing MLflow tracking, the system shall resolve the tracking URI from `MLFLOW_TRACKING_URI` environment variable, defaulting to `sqlite:///artifacts/mlflow.db` as configured in `params.yaml` (`tier1_ml.registry.tracking_uri`).

**FR-2 (artifact bundle packaging).** When packaging a Tier 1 candidate model, `aegis.pipelines.registry.package_and_register_candidate` shall bundle:
1. `tier1_segments.csv`: The 35 validated segment outputs conforming to `Tier1SegmentOutput` (P2-F05, ADR-014).
2. `glm_calibration_report.json`: GLM deviance, deciles, and loss cap manifest (€100k, load 1.1988) (Stage 5).
3. `causal_validation_summary.json`: Multi-seed validation gate summary and Option F closure record (Stage 10).
4. `sensitivity_summary.json`: 10-cell sensitivity degradation table and Scenario S4 parameters (Stage 11).
5. Lineage metadata dictionary (`lineage.json`).

**FR-3 (estimator run-artifact isolation).** When logging the run to MLflow, the pipeline shall log the fitted `CausalForestDML` object strictly as an unpromoted run artifact (e.g. `artifacts/estimator/`) for offline audit. The serving directory package shall NOT include pickle or estimator binaries (P2-D11).

**FR-4 (candidate alias assignment).** When candidate registration completes successfully, the pipeline shall register a new model version under registered model name `aegis-causal-elasticity` and assign the mutable alias `candidate`. The pipeline shall NEVER assign `approved` automatically.

**FR-5 (lineage tag enforcement).** Every registered model version shall include the following mandatory tags:
- `git_commit`: Current git HEAD commit hash (40-char SHA).
- `dvc_data_hash`: Hash of versioned input training data.
- `dgp_manifest_id`: SHA-256 fingerprint from `dgp_manifest.json`.
- `calibration_hash`: SHA-256 hash of `glm_calibration_report.json`.
- `validation_status`: `GATE_PASSED_OPTION_F`.
- `registered_by`: Pipeline executor identifier / user.

**FR-6 (human-gated promotion).** When `promote_model.py` is invoked to promote a model version to `approved`, the tool shall require:
1. Target model version number.
2. Reviewer ID (`--reviewer-id`, non-empty string).
3. Calibration report path or explicit SHA-256 hash (`--calibration-hash`).
4. Human review record note / rationale (`--review-note`).

If the reviewer ID is missing, if any required lineage tag is absent, or if the provided calibration hash does not match the model version's recorded `calibration_hash` tag, promotion shall fail closed, raise `PromotionGateError`, and exit non-zero without modifying aliases.

**FR-7 (approved alias assignment & audit log).** Upon successful human-gated validation, the tool shall assign the alias `approved` to the target model version and record promotion audit tags:
- `promoted_by`: Reviewer ID.
- `promoted_at`: UTC ISO timestamp.
- `promotion_rationale`: Review note.

**FR-8 (rollback capability).** When rollback is requested (`scripts/promote_model.py rollback --version <prior_version> --reviewer-id <id>`), the tool shall verify that `<prior_version>` exists and was previously evaluated, re-point the `approved` alias to `<prior_version>`, and log a `rollback_from` audit tag.

**FR-9 (plain data serving loader).** When `load_approved_segments(tracking_uri)` is called by serving modules (Showcase API or Phase 3), it shall:
1. Query MLflow for the model version pointed to by alias `approved`.
2. Download or read the artifact directory for that version.
3. Parse and validate `tier1_segments.csv` into a list of `Tier1SegmentOutput` instances.
4. Verify that zero estimator objects or pickle streams are loaded.
5. Return the verified `Tier1SegmentTable` alongside model version metadata.

---

## 5. Data Contracts & Schemas

### 5.1 Promotion Request & Audit Schemas (`src/aegis/schemas/registry.py`)

```python
class ModelPromotionRequest(BaseModel):
    """Input payload for human-gated model promotion."""
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="aegis-causal-elasticity")
    target_version: str = Field(description="Model version number to promote")
    reviewer_id: str = Field(min_length=3, description="Identifier of human reviewer")
    calibration_report_hash: str = Field(min_length=64, max_length=64, description="SHA-256 of reviewed calibration report")
    review_notes: str = Field(min_length=10, description="Actuarial sign-off rationale")


class ModelLineageMetadata(BaseModel):
    """Cryptographic lineage metadata attached to registered candidate."""
    model_config = ConfigDict(extra="forbid")

    model_name: str
    model_version: str
    run_id: str
    git_commit: str
    dvc_data_hash: str
    dgp_manifest_id: str
    calibration_hash: str
    validation_status: str
    created_at_utc: str
```

### 5.2 Registered Package Structure (`artifacts/models/aegis-causal-elasticity/<version>/`)

```text
├── tier1_segments.csv             # 35-cell segment table (Tier1SegmentOutput schema)
├── glm_calibration_report.json    # Holdout deviance, deciles, cap manifest
├── causal_validation_summary.json # Option F recovery metrics & seed stats
├── sensitivity_summary.json       # Degradation table, Rv=0.5004, Scenario S4
└── lineage.json                   # Machine-readable lineage metadata
```

---

## 6. Edge Cases & Error Handling

| Edge Case | Required System Behavior |
|---|---|
| Pipeline attempts to set `approved` alias automatically | Pipeline code lacks permission/method to set `approved`; only `scripts/promote_model.py` can assign it. |
| Promotion attempted without `--reviewer-id` | Aborts immediately with `PromotionGateError("Reviewer ID mandatory for approved promotion")`. Exit code 1. |
| Provided calibration hash does not match artifact | Aborts with `PromotionGateError("Calibration report hash mismatch: review invalid")`. Exit code 1. |
| Model version missing required lineage tag (e.g. untracked git dirty state) | Aborts with `PromotionGateError("Missing lineage tag: git_commit")`. Exit code 1. |
| Rollback requested to nonexistent version | Aborts with `PromotionGateError("Target rollback version does not exist")`. Exit code 1. |
| Serving code attempts to load from `approved` when alias is unset | Raises `ModelNotFoundError("No model version currently assigned to alias 'approved'")`. |
| Pickled estimator placed inside serving artifact path | `load_approved_segments` performs file scan and raises `RegistrySecurityError` if `.pkl` / `.joblib` is present in serving bundle. |

---

## 7. Open Decisions

### OD-1 — Registered Model Name
- **Options:**
  - **Option A (Recommend): `aegis-causal-elasticity`** (Matches README, Scoping doc, and PRD).
  - **Option B: `tier1-deterministic-baseline`**.
- **Recommendation:** **Option A (`aegis-causal-elasticity`).**

> **APPROVED — October 30, 2026:** Option A (`aegis-causal-elasticity`).

### OD-2 — Storage Location of SQLite Registry DB
- **Options:**
  - **Option A (Recommend): `artifacts/mlflow.db`** (Inside gitignored `artifacts/`, persisted locally; configurable via `params.yaml` and `MLFLOW_TRACKING_URI`).
  - **Option B: `data/processed/mlflow.db`**.
- **Recommendation:** **Option A.** Local-first, gitignored, auditable.

> **APPROVED — October 30, 2026:** Option A (`artifacts/mlflow.db`).

### OD-3 — CLI Interface for Promotion
- **Options:**
  - **Option A (Recommend): Dual interface:** standalone script `scripts/promote_model.py` AND CLI subcommand `uv run python -m aegis.pipelines.cli promote-model`.
  - **Option B: Standalone script only.**
- **Recommendation:** **Option A.** Maximizes accessibility in both automated scripts and terminal CLI workflows.

> **APPROVED — October 30, 2026:** Option A (dual interface).

---

## 8. Falsification Protocol & Hard Gate

### Falsification Test
1. Register a mock candidate model version with valid lineage tags and calibration report.
2. Attempt to promote this candidate to `approved` using `promote_model.py`:
   - Without a reviewer ID $\rightarrow$ must fail with `PromotionGateError`.
   - With a tampered/mismatched calibration report hash $\rightarrow$ must fail with `PromotionGateError`.
   - With a missing lineage tag (e.g. removed `git_commit`) $\rightarrow$ must fail with `PromotionGateError`.
3. Provide valid reviewer ID, matching hash, and complete lineage $\rightarrow$ promotion succeeds; alias `approved` is assigned.
4. Execute rollback to prior version $\rightarrow$ verify `approved` alias points to previous version.

### Gate 12 — Hard Close
- `candidate` alias is set automatically by packaging pipeline.
- Promotion to `approved` is blocked without valid human review record and complete lineage.
- Rollback successfully re-points `approved` alias.
- Serving loader loads plain segment data and gate metadata without deserializing estimator objects.
- Roadmap Exit Criterion **E4** is satisfied.
