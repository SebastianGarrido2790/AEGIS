# Plan: P2-F09 MLflow Artifact Packaging & Human-Gated Promotion
## Phase 2 — Stage 12

**Spec:** APPROVED October 30, 2026 (OD-1, OD-2, OD-3 as recommended)  
**Status:** APPROVED — October 30, 2026 (operator: "Approve as drafted")  
**Date:** 2026-10-30  

---

## 1. Implementation Strategy

Stage 12 splits into two modules by responsibility, one schema module, and thin entrypoints:

| Unit | Module | Responsibility |
|---|---|---|
| 1. Registry schemas | `src/aegis/schemas/registry.py` | `ModelLineageMetadata`, `ModelPromotionRequest`, `PromotionAuditRecord`, `ApprovedSegmentPackage` |
| 2. Artifact packaging (pure) | `src/aegis/pipelines/artifact_package.py` | Join Stage 4 + Stage 8 tables into a validated `Tier1SegmentTable`; resolve lineage (git, DVC, DGP, calibration hash); write the staging bundle; enforce "no estimator binary in serving bundle" |
| 3. Registry client + promotion | `src/aegis/pipelines/registry.py` | MLflow logging, registration, `candidate` alias, human-gated `approved` promotion, rollback, `load_approved_segments`, CLI subparser registration |
| 4. Config | `src/aegis/config/schema.py`, `params.yaml` | `RegistryConfig`, `RegistryPathsConfig` |
| 5. Entrypoints | `scripts/promote_model.py`, `src/aegis/pipelines/cli.py` | Operator-facing promotion/rollback commands |
| 6. Tests | `tests/unit/test_registry.py`, `tests/unit/test_artifact_package.py` | Falsification suite for the promotion gate and packaging contracts |

Rationale for splitting (2) from (3): packaging and lineage are deterministic and testable without MLflow; the registry client is stateful and needs an ephemeral SQLite backend per test. Keeping them apart preserves INV-8 headroom and makes the falsification tests precise.

---

## 2. Component Architecture and Interfaces

### 2.1 `src/aegis/schemas/registry.py`

```python
class ModelLineageMetadata(BaseModel):
    """Complete provenance attached to every registered candidate."""
    model_config = ConfigDict(extra="forbid")

    model_name: str
    model_version: str
    run_id: str
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    dvc_data_hash: str = Field(pattern=r"^[0-9a-f]{32,64}$")
    dgp_manifest_id: str = Field(pattern=SHA256_PATTERN)
    calibration_hash: str = Field(pattern=SHA256_PATTERN)
    validation_status: Literal["GATE_PASSED_OPTION_F"]
    segment_count: int = Field(ge=1)
    created_at_utc: str


class ModelPromotionRequest(BaseModel):
    """Human sign-off payload required to move `candidate` -> `approved`."""
    model_config = ConfigDict(extra="forbid")

    model_name: str = "aegis-causal-elasticity"
    target_version: str
    reviewer_id: str = Field(min_length=3)
    calibration_report_hash: str = Field(min_length=64, max_length=64)
    review_notes: str = Field(min_length=10)


class PromotionAuditRecord(BaseModel):
    """Result of a successful promotion or rollback."""
    model_config = ConfigDict(extra="forbid")

    model_name: str
    previous_approved_version: str | None
    new_approved_version: str
    reviewer_id: str
    action: Literal["promote", "rollback"]
    calibration_verified: bool
    audit_tags: dict[str, str]
    timestamp_utc: str


class ApprovedSegmentPackage(BaseModel):
    """Plain-data serving payload resolved from the `approved` alias."""
    model_config = ConfigDict(extra="forbid")

    model_name: str
    model_version: str
    run_id: str
    segments: Tier1SegmentTable
    lineage: ModelLineageMetadata
    validation_summary: dict[str, Any]
```

### 2.2 `src/aegis/pipelines/artifact_package.py`

```python
STANDARD_BUNDLE_FILES: Final[tuple[str, ...]] = (
    "tier1_segments.csv",
    "glm_calibration_report.json",
    "causal_validation_summary.json",
    "sensitivity_summary.json",
    "lineage.json",
)
FORBIDDEN_BUNDLE_SUFFIXES: Final[tuple[str, ...]] = (".pkl", ".pickle", ".joblib")

def compute_file_sha256(path: Path) -> str: ...

def load_tier1_segment_table(
    segment_loss_cost_path: Path,
    segment_estimates_path: Path,
    estimator_metadata_path: Path,
    dgp_manifest_path: Path,
) -> Tier1SegmentTable:
    """Inner-joins Stage 4 economics with Stage 8 causal estimates into 35 rows.

    Field provenance:
      policy_count, exposure_years, loss_cost_per_year, base_premium_per_year
        -> data/processed/segment_loss_cost.csv   (Stage 4, P2-F05)
      baseline_retention_q0, tau_hat, epsilon_hat, epsilon_ci_lower/upper
        -> data/processed/segment_elasticity_estimates.csv (Stage 8, P2-F06)
      model_version -> causal_estimator_metadata.json["model_version"]
      data_version  -> dgp_manifest.json["config_fingerprint"]
      semi_synthetic -> True (constant; DGP provenance)

    Raises:
        RegistryPackagingError: on row-count mismatch, unknown segment_id,
            duplicate segment_id, or a failing Tier1SegmentOutput validation.
    """

def resolve_git_commit(repo_root: Path) -> tuple[str, bool]: ...

def resolve_dvc_data_hash(
    dvc_lock_path: Path,
    training_table_path: Path,
    *, lock_dep_path: str = "data/processed/elasticity_training.csv",
) -> str:
    """Returns the DVC md5 for the training table from dvc.lock; falls back to
    SHA-256 of the file when the lock has no entry (documented fallback)."""

def resolve_lineage(
    *, model_name: str, model_version: str, run_id: str, repo_root: Path,
    calibration_report_path: Path, dgp_manifest_path: Path,
    segment_count: int,
) -> ModelLineageMetadata: ...

def build_serving_bundle(
    staging_dir: Path,
    *,
    segments: Tier1SegmentTable,
    calibration_report_path: Path,
    validation_summary_path: Path,
    sensitivity_summary_path: Path,
    lineage: ModelLineageMetadata,
) -> list[Path]:
    """Writes the five standard files; rejects estimator binaries in the bundle."""

def assert_no_estimator_artifacts(paths: Iterable[Path]) -> None:
    """Raises RegistrySecurityError if any pickled/joblib estimator is present."""
```

### 2.3 `src/aegis/pipelines/registry.py`

```python
def resolve_tracking_uri(config: AEGISConfig) -> str:
    """Env-var `MLFLOW_TRACKING_URI` wins over `tier1_ml.registry.tracking_uri`."""

def package_and_register_candidate(
    config: AEGISConfig | None = None,
) -> ModelLineageMetadata:
    """Builds the bundle, logs it to a run, registers a model version under
    `aegis-causal-elasticity`, tags it, and assigns ONLY the `candidate` alias.

    The fitted estimator (if present at estimator_paths.model_artifact) is logged
    as a run artifact path outside the serving bundle (P2-D11).
    """

def promote_to_approved(
    request: ModelPromotionRequest,
    *, config: AEGISConfig | None = None, client: MlflowClient | None = None,
) -> PromotionAuditRecord:
    """Human-gated promotion. Fail-closed validation order:
       1. reviewer_id non-empty (schema-enforced)
       2. model version exists and carries alias `candidate`
       3. every mandatory lineage tag present
       4. calibration hash matches the recorded tag AND the on-disk report
       5. only then assign `approved` + audit tags
    Raises PromotionGateError with a typed `reason` on any failure.
    """

def rollback_approved(
    *, target_version: str, reviewer_id: str, reason: str,
    config: AEGISConfig | None = None, client: MlflowClient | None = None,
) -> PromotionAuditRecord: ...

def load_approved_segments(
    config: AEGISConfig | None = None, client: MlflowClient | None = None,
) -> ApprovedSegmentPackage:
    """Serving path: resolve the `approved` alias, download the bundle, parse
    `tier1_segments.csv`, and reject any estimator binary in the artifact tree."""

def add_registry_subparsers(subparsers: Any) -> None:
    """Registers `promote-model` and `rollback-model` with set_defaults(func=...)."""
```

Promotion audit tags written on success: `promoted_by`, `promoted_at`, `promotion_rationale`, `promotion_calibration_hash`, `rollback_from` (rollback only).

### 2.4 Configuration

`params.yaml` additions under `tier1_ml`:

```yaml
  # Stage 12 (P2-F09, ADR-014, P2-D11a): local SQLite registry with aliases.
  registry:
    model_name: "aegis-causal-elasticity"
    tracking_uri: "sqlite:///artifacts/mlflow.db"
    experiment_name: "aegis-tier1-registry"
    candidate_alias: "candidate"
    approved_alias: "approved"
    required_lineage_tags:
      - "git_commit"
      - "dvc_data_hash"
      - "dgp_manifest_id"
      - "calibration_hash"
      - "validation_status"
  registry_paths:
    segment_loss_cost: "data/processed/segment_loss_cost.csv"
    segment_estimates: "data/processed/segment_elasticity_estimates.csv"
    estimator_metadata: "data/processed/causal_estimator_metadata.json"
    dgp_manifest: "data/processed/dgp_manifest.json"
    calibration_report: "data/processed/glm_calibration_report.json"
    validation_summary: "artifacts/causal_validation_summary.json"
    sensitivity_summary: "artifacts/sensitivity_summary.json"
    dvc_lock: "dvc.lock"
    staging_dir: "artifacts/registry_staging"
```

`RegistryConfig` and `RegistryPathsConfig` Pydantic models mirror this block with `extra="forbid"`; `tracking_uri` is a default only — `resolve_tracking_uri` prefers the environment variable (INV: no secrets in code/config).

### 2.5 Error hierarchy (`src/aegis/utils/exceptions.py`)

| Exception | Raised when |
|---|---|
| `RegistryPackagingError(AEGISError)` | Segment-table join, schema validation, or bundle write fails |
| `PromotionGateError(AEGISError)` | Any human-review or lineage precondition fails; carries `details["reason"]` |
| `RegistrySecurityError(AEGISError)` | An estimator binary is found in the serving bundle, or serving attempts deserialization |
| `ModelNotFoundError(AEGISError)` | Registered model, version, or required alias is absent |

---

## 3. Packaging Data Flow

```text
segment_loss_cost.csv (35 rows, Stage 4)
        +
segment_elasticity_estimates.csv (35 rows, Stage 8)
        +
causal_estimator_metadata.json (model_version)
        +
dgp_manifest.json (config_fingerprint -> data_version)
        │
        ▼
Tier1SegmentTable (35 x Tier1SegmentOutput, extra="forbid", no Δ fields)
        │
        ├── glm_calibration_report.json      (Stage 5 / P2-F03)
        ├── causal_validation_summary.json   (Stage 10 / Option F)
        ├── sensitivity_summary.json         (Stage 11 / P2-F08)
        ├── lineage.json                     (git, DVC, DGP, calibration SHA-256)
        ▼
MLflow run -> artifacts/serving_bundle/  (plain data only)
           -> artifacts/estimator/       (joblib, NON-promotable run artifact)
        ▼
Registered model version + alias `candidate`
```

`loss_cap`/`excess_load` from `params.yaml` are copied into `glm_calibration_report.json` unchanged — the calibration report is the reviewed artifact, so its bytes must not be rewritten by the registry (the recorded `calibration_hash` must match what the reviewer read).

---

## 4. Promotion Gate Logic (fail-closed)

```text
promote_to_approved(request)
  ├─ schema validation (reviewer_id >= 3 chars, notes >= 10 chars, hash len 64)
  ├─ resolve version; must exist                       else PromotionGateError("unknown_version")
  ├─ version must carry alias `candidate`              else PromotionGateError("not_a_candidate")
  ├─ every tag in required_lineage_tags present        else PromotionGateError("missing_lineage_tag:<tag>")
  ├─ request.calibration_report_hash == tag value      else PromotionGateError("calibration_hash_mismatch")
  ├─ recompute SHA-256 of on-disk calibration report
  │     == tag value                                   else PromotionGateError("calibration_report_drift")
  ├─ assign alias `approved` -> version
  └─ set tags promoted_by / promoted_at / promotion_rationale / promotion_calibration_hash
     -> PromotionAuditRecord(action="promote")
```

The `candidate` alias is neither cleared nor reassigned on promotion, so the promoted version remains identifiable as the reviewed candidate. `previous_approved_version` is recorded for rollback traceability.

Rollback reuses the same gate: reviewer ID required, target version must exist, and it must have been `approved` at least once (`promoted_by` tag present) or be the immediately preceding version. A `rollback_from` tag records the discarded version.

---

## 5. Serving Path

`load_approved_segments()` is the only registry entrypoint allowed in serving code:

1. `client.get_model_version_by_alias(model_name, "approved")` — alias absent raises `ModelNotFoundError`.
2. Download the artifact tree for `version.run_id`.
3. `assert_no_estimator_artifacts()` scans for `.pkl`/`.pickle`/`.joblib`; a hit raises `RegistrySecurityError`.
4. Parse `tier1_segments.csv` into `Tier1SegmentTable` (which already forbids Δ-like fields and enforces CI self-consistency).
5. Parse `lineage.json` into `ModelLineageMetadata`; load `causal_validation_summary.json`.
6. Return `ApprovedSegmentPackage`.

Stage 13 (P2-F11) will rewire `src/aegis/api/model_service.py` (currently a legacy payload reader with a hardcoded `sqlite:///mlflow.db` and "newest usable version" fallback) to call `load_approved_segments`. Stage 12 does **not** modify the showcase; the spec's serving requirement is satisfied by providing and testing the loader.

---

## 6. CLI and Script Wiring

`src/aegis/pipelines/cli.py` is at **959/1000 lines** (INV-8). Registering two subcommands inline would breach the ceiling. Mitigation: `registry.add_registry_subparsers(subparsers)` builds the `promote-model` / `rollback-model` parsers inside `registry.py`, and `cli.py` gains only:

```python
    from aegis.pipelines.registry import add_registry_subparsers
    add_registry_subparsers(subparsers)          # inside build_parser()
    ...
    if hasattr(args, "func"):                     # inside main()
        return args.func(args)
```

Net `cli.py` growth: ~6 lines (965/1000). If it still lands above 980, extract the existing `build-segments`/`run-q1` parser blocks into a sibling helper in a follow-up task rather than at Stage 12.

`scripts/promote_model.py` is a thin argparse wrapper (subcommands `promote` and `rollback`) delegating to `registry.promote_to_approved` / `registry.rollback_approved`, printing the audit record and exiting `1` on `PromotionGateError`. It exists so promotion can run without the pipeline CLI and is exercised by the falsification test.

---

## 7. Test Strategy

### 7.1 Unit — packaging (`tests/unit/test_artifact_package.py`)

| Test | Assertion |
|---|---|
| `test_join_produces_35_valid_segments` | 35 rows; every row validates as `Tier1SegmentOutput`; ids match Stage 4 ids |
| `test_join_rejects_missing_segment` | Dropped Stage 8 segment raises `RegistryPackagingError` |
| `test_join_rejects_duplicate_segment` | Duplicate id raises `RegistryPackagingError` |
| `test_lineage_requires_clean_commit_shape` | `git_commit` is 40 hex chars; `git_dirty` reflects `git status` (mocked) |
| `test_dvc_hash_from_lock_and_fallback` | Lock md5 returned when present; SHA-256 fallback when absent |
| `test_bundle_rejects_estimator_binary` | A staged `.joblib` triggers `RegistrySecurityError` |
| `test_bundle_contains_exactly_five_files` | Standard file set; no estimator binary |

### 7.2 Unit — registry and promotion gate (`tests/unit/test_registry.py`)

Every test uses an isolated backend: `sqlite:///{tmp_path}/mlflow.db`, an explicit `MlflowClient`, and monkeypatched config paths pointing at `tmp_path` fixtures. No test touches `artifacts/mlflow.db`.

| Test | Assertion |
|---|---|
| `test_candidate_registration_sets_only_candidate` | `candidate` resolves to v1; `approved` alias does not exist |
| `test_promotion_without_reviewer_fails` | Schema/`PromotionGateError`; alias unchanged (falsification a) |
| `test_promotion_with_mismatched_hash_fails` | `PromotionGateError(reason="calibration_hash_mismatch")` (falsification b) |
| `test_promotion_with_missing_lineage_tag_fails` | Tag removed -> `PromotionGateError(reason="missing_lineage_tag:git_commit")` (falsification c) |
| `test_promotion_success_sets_approved_and_audit_tags` | `approved` -> version; `promoted_by`/`promoted_at`/`promotion_rationale` set |
| `test_promotion_rejects_non_candidate_version` | Unregistered version -> `PromotionGateError(reason="not_a_candidate")` |
| `test_calibration_drift_detected` | Report bytes changed after tagging -> `PromotionGateError(reason="calibration_report_drift")` |
| `test_rollback_repoints_approved` | v2 -> v1; `rollback_from` recorded; `PromotionAuditRecord.action == "rollback"` |
| `test_rollback_unknown_version_fails` | `PromotionGateError`; `approved` unchanged |
| `test_approved_missing_raises_model_not_found` | `ModelNotFoundError` |
| `test_serving_loader_returns_plain_segments` | `ApprovedSegmentPackage.segments.segments` length 35; no estimator import |
| `test_serving_loader_rejects_estimator_binary` | `.pkl` in artifact tree -> `RegistrySecurityError` |
| `test_cli_parsers_registered` | `build_parser()` accepts `promote-model` and `rollback-model` |

The "serving never deserializes an estimator" property is enforced structurally: `registry.py` imports only `mlflow` metadata APIs plus pandas/pydantic, and `assert_no_estimator_artifacts` blocks estimate binaries at load time. A source-level assertion (no `joblib`/`pickle`/`econml` imports in `registry.py`) is added to the same test module.

---

## 8. INV-8 Budget

| File | Estimated lines |
|---|---|
| `src/aegis/schemas/registry.py` | ~90 |
| `src/aegis/pipelines/artifact_package.py` | ~230 |
| `src/aegis/pipelines/registry.py` | ~380 |
| `src/aegis/config/schema.py` | 549 -> ~595 |
| `src/aegis/pipelines/cli.py` | 959 -> ~965 |
| `tests/unit/test_artifact_package.py` | ~180 |
| `tests/unit/test_registry.py` | ~320 |

All under the 1,000-line ceiling; `cli.py` is the only module near it and is capped by design (§6).

---

## 9. Execution Sequence

1. **T1** Schemas + exceptions + config (`schemas/registry.py`, `config/schema.py`, `params.yaml`, `exceptions.py`).
2. **T2** Packaging module (`artifact_package.py`) with tests.
3. **T3** Registry client: registration + `candidate` alias.
4. **T4** Promotion gate + rollback.
5. **T5** Serving loader + estimator-binary rejection.
6. **T6** CLI subparsers + `scripts/promote_model.py`.
7. **T7** Full verification (`pytest`, `ruff`, `pyright`, `check_module_size`).
8. **T8** Execute candidate registration on the real artifacts (`artifacts/mlflow.db`) — automated, no human gate.
9. **T9** **Human-gated promotion:** the operator supplies reviewer ID, review note, and confirms the calibration hash. The agent executes the command but cannot self-approve; without operator input Gate 12 stays open.
10. **T10** Rollback rehearsal to a scratch version, evidence capture, and status updates in `execution_plan.md`, `stage_evidence.md`, `implementation_plan.md`.

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `cli.py` INV-8 breach | Subparser delegation into `registry.py`; hard cap at 980 lines (measured in T6) |
| Legacy `model_service.py` reads "newest version" instead of `approved` | Documented; Stage 13 rewires it to `load_approved_segments`. Stage 12 tests the loader it will consume |
| MLflow SQLite locking during concurrent tests | Every test uses its own `tmp_path` DB; no shared backend |
| dvc.lock stale relative to regenerated data | `resolve_dvc_data_hash` compares the lock md5 against the recomputed file md5 and raises `RegistryPackagingError` on mismatch |
| Accidental self-approval by an automated agent | `promote-model` requires a reviewer ID supplied out-of-band; T9 is explicitly operator-gated |
| Promotion of a version whose calibration report changed after tagging | On-disk SHA-256 recomputation (`calibration_report_drift`) |
| Estimator pickle leaking into serving | `assert_no_estimator_artifacts` on both build and load; bundle file allow-list |
