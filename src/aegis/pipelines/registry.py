"""MLflow registry client, human-gated promotion, and plain-data serving (P2-F09).

Governance contract (ADR-014, P2-D11, P2-D11a):

* Packaging registers a model version and assigns **only** the mutable ``candidate``
  alias. Nothing in the pipeline path can assign ``approved``.
* ``approved`` moves only through :func:`promote_to_approved`, which requires a human
  reviewer identity, the SHA-256 of the calibration report that reviewer read, and a
  complete lineage tag set. Every precondition fails closed with ``PromotionGateError``.
* Serving resolves the ``approved`` alias and downloads a plain-data bundle. The
  fitted estimator is a non-promotable run artifact and is never deserialized here:
  this module deliberately imports neither ``joblib`` nor ``pickle`` nor ``econml``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import ValidationError

from aegis.config.loader import find_project_root, get_config
from aegis.config.schema import AEGISConfig
from aegis.pipelines.artifact_package import (
    LINEAGE_FILE,
    SEGMENT_TABLE_FILE,
    assert_no_estimator_artifacts,
    build_serving_bundle,
    compute_file_sha256,
    load_tier1_segment_table,
    read_segment_table,
    resolve_lineage,
    write_lineage_file,
)
from aegis.schemas.registry import (
    ApprovedSegmentPackage,
    ModelLineageMetadata,
    ModelPromotionRequest,
    PromotionAuditRecord,
)
from aegis.utils.exceptions import (
    ModelNotFoundError,
    PromotionGateError,
    RegistryPackagingError,
)

SERVING_ARTIFACT_PATH = "serving_bundle"
ESTIMATOR_ARTIFACT_PATH = "estimator"
CALIBRATION_REPORT_FILE = "glm_calibration_report.json"
VALIDATION_SUMMARY_FILE = "causal_validation_summary.json"

PROMOTION_TAG_KEYS: tuple[str, ...] = (
    "promoted_by",
    "promoted_at",
    "promotion_rationale",
    "promotion_calibration_hash",
)
ROLLBACK_TAG = "rollback_from"
_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"


def _cfg(config: AEGISConfig | None) -> AEGISConfig:
    """Returns the supplied configuration or the cached project configuration."""
    return config if config is not None else get_config()


def _model_name(config: AEGISConfig) -> str:
    """Returns the registered model name from configuration."""
    return config.tier1_ml.registry.model_name


def _normalize_tracking_uri(uri: str, repo_root: Path) -> str:
    """Makes a relative SQLite URI absolute so it does not depend on the cwd."""
    prefix = "sqlite:///"
    if not uri.startswith(prefix) or uri.startswith("sqlite:////"):
        return uri
    raw = uri[len(prefix) :]
    if not raw or raw.startswith(":memory:") or Path(raw).is_absolute():
        return uri
    return prefix + (repo_root / raw).as_posix()


def resolve_tracking_uri(
    config: AEGISConfig | None = None, *, repo_root: Path | None = None
) -> str:
    """Resolves the tracking URI, letting ``MLFLOW_TRACKING_URI`` override config."""
    cfg = _cfg(config)
    root = Path(repo_root) if repo_root is not None else find_project_root()
    configured = cfg.tier1_ml.registry.tracking_uri
    override = os.environ.get(_TRACKING_URI_ENV)
    return _normalize_tracking_uri(override or configured, root)


def _client(
    config: AEGISConfig | None = None,
    client: MlflowClient | None = None,
    *,
    repo_root: Path | None = None,
) -> MlflowClient:
    """Returns the supplied MLflow client or builds one from the resolved URI."""
    if client is not None:
        return client
    return MlflowClient(tracking_uri=resolve_tracking_uri(config, repo_root=repo_root))


def _utc_now() -> str:
    """Returns the current UTC timestamp in ISO 8601 form."""
    return datetime.now(UTC).isoformat()


def _ensure_experiment(client: MlflowClient, name: str, repo_root: Path) -> str:
    """Returns the experiment id, creating the experiment with a repo-local store."""
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    artifact_root = Path(repo_root) / "artifacts" / "mlflow_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(name, artifact_location=artifact_root.as_uri())


def _ensure_registered_model(client: MlflowClient, name: str) -> None:
    """Creates the registered model when it does not exist yet."""
    try:
        client.get_registered_model(name)
    except MlflowException:
        try:
            client.create_registered_model(name)
        except MlflowException:
            # A concurrent creator won the race; the model now exists.
            client.get_registered_model(name)


def _version_tags(lineage: ModelLineageMetadata) -> dict[str, str]:
    """Flattens a lineage record into MLflow model-version tags."""
    return {
        "git_commit": lineage.git_commit,
        "git_dirty": str(lineage.git_dirty).lower(),
        "dvc_data_hash": lineage.dvc_data_hash,
        "dgp_manifest_id": lineage.dgp_manifest_id,
        "calibration_hash": lineage.calibration_hash,
        "validation_status": lineage.validation_status,
        "segment_count": str(lineage.segment_count),
        "registered_by": os.environ.get("AEGIS_REGISTERED_BY")
        or os.environ.get("USERNAME", os.environ.get("USER", "unknown")),
        "registered_at": lineage.created_at_utc,
    }


def package_and_register_candidate(
    config: AEGISConfig | None = None,
    *,
    client: MlflowClient | None = None,
    repo_root: Path | None = None,
) -> ModelLineageMetadata:
    """Packages Tier 1 artifacts, registers a model version, and assigns ``candidate``.

    The fitted estimator is logged as a non-promotable run artifact. This function
    cannot assign the ``approved`` alias under any circumstances (P2-D11a).

    Args:
        config: Optional configuration override.
        client: Optional MLflow client (tests inject an ephemeral backend).
        repo_root: Optional repository root override.

    Returns:
        The lineage record as published in the bundle, with the registry version.

    Raises:
        RegistryPackagingError: If an input artifact is missing or a bundle check fails.
    """
    cfg = _cfg(config)
    registry = cfg.tier1_ml.registry
    paths = cfg.tier1_ml.registry_paths
    root = Path(repo_root) if repo_root is not None else find_project_root()
    mlflow_client = _client(cfg, client, repo_root=root)

    table = load_tier1_segment_table(
        root / paths.segment_loss_cost,
        root / paths.segment_estimates,
        root / paths.estimator_metadata,
        root / paths.dgp_manifest,
    )

    experiment_id = _ensure_experiment(mlflow_client, registry.experiment_name, root)
    run = mlflow_client.create_run(
        experiment_id, run_name=f"candidate-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    run_id = run.info.run_id

    staging_root = root / paths.staging_dir
    staging_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = Path(tempfile.mkdtemp(prefix="bundle-", dir=staging_root))

    try:
        lineage = resolve_lineage(
            model_name=_model_name(cfg),
            model_version="pending-registration",
            run_id=run_id,
            repo_root=root,
            calibration_report_path=root / paths.calibration_report,
            dgp_manifest_path=root / paths.dgp_manifest,
            segment_count=len(table.segments),
        )
        build_serving_bundle(
            bundle_dir,
            segments=table,
            calibration_report_path=root / paths.calibration_report,
            validation_summary_path=root / paths.validation_summary,
            sensitivity_summary_path=root / paths.sensitivity_summary,
            lineage=lineage,
        )
        mlflow_client.log_artifacts(run_id, str(bundle_dir), artifact_path=SERVING_ARTIFACT_PATH)

        estimator_path = root / cfg.tier1_ml.estimator_paths.model_artifact
        if estimator_path.is_file():
            mlflow_client.log_artifact(
                run_id, str(estimator_path), artifact_path=ESTIMATOR_ARTIFACT_PATH
            )

        for key, value in _version_tags(lineage).items():
            mlflow_client.set_tag(run_id, key, value)

        _ensure_registered_model(mlflow_client, _model_name(cfg))
        model_version = mlflow_client.create_model_version(
            _model_name(cfg),
            f"runs:/{run_id}/{SERVING_ARTIFACT_PATH}",
            run_id,
            description=(
                "Tier 1 plain-data segment bundle (P2-F09). Promotion to 'approved' "
                "requires a human review record."
            ),
        )

        published_lineage = lineage.model_copy(
            update={"model_version": str(model_version.version)}
        )
        write_lineage_file(published_lineage, bundle_dir / LINEAGE_FILE)
        mlflow_client.log_artifact(
            run_id, str(bundle_dir / LINEAGE_FILE), artifact_path=SERVING_ARTIFACT_PATH
        )

        for key, value in _version_tags(published_lineage).items():
            mlflow_client.set_model_version_tag(
                _model_name(cfg), published_lineage.model_version, key, value
            )
        mlflow_client.set_model_version_tag(
            _model_name(cfg),
            published_lineage.model_version,
            "registered_by_pipeline",
            "true",
        )
        mlflow_client.set_registered_model_alias(
            _model_name(cfg), registry.candidate_alias, published_lineage.model_version
        )
        mlflow_client.set_terminated(run_id, "FINISHED")
    except Exception:
        mlflow_client.set_terminated(run_id, "FAILED")
        raise
    finally:
        shutil.rmtree(bundle_dir, ignore_errors=True)

    return published_lineage


def _get_model_version(client: MlflowClient, model_name: str, version: str) -> Any:
    """Returns a model version or raises ``ModelNotFoundError``."""
    try:
        return client.get_model_version(model_name, version)
    except MlflowException as exc:
        raise ModelNotFoundError(
            f"model version '{version}' of '{model_name}' does not exist",
            details={"model_name": model_name, "version": version},
        ) from exc


def _alias_version(client: MlflowClient, model_name: str, alias: str) -> str | None:
    """Returns the version number pointed at by an alias, or ``None`` when unset."""
    try:
        return str(client.get_model_version_by_alias(model_name, alias).version)
    except MlflowException:
        return None


def _gate(message: str, reason: str, **details: Any) -> PromotionGateError:
    """Builds a typed promotion-gate failure."""
    return PromotionGateError(message, details={"reason": reason, **details})


def _require_reviewer(reviewer_id: str) -> str:
    """Rejects an absent or placeholder reviewer identity."""
    reviewer = (reviewer_id or "").strip()
    if len(reviewer) < 3:
        raise _gate(
            "Reviewer ID mandatory for approved promotion",
            "reviewer_id_missing",
        )
    return reviewer


def promote_to_approved(
    request: ModelPromotionRequest,
    *,
    config: AEGISConfig | None = None,
    client: MlflowClient | None = None,
    repo_root: Path | None = None,
    calibration_report_path: Path | None = None,
) -> PromotionAuditRecord:
    """Promotes a reviewed candidate to the ``approved`` alias (fail-closed).

    Validation order (P2-F09 FR-6): reviewer identity, version existence, candidate
    alias membership, complete lineage tags, calibration hash match, and calibration
    report drift re-check. Aliases are untouched unless every check passes.

    Args:
        request: Human review payload.
        config: Optional configuration override.
        client: Optional MLflow client.
        repo_root: Optional repository root override.
        calibration_report_path: Optional explicit report path for the drift check.

    Returns:
        The promotion audit record.

    Raises:
        PromotionGateError: If any review or lineage precondition fails.
        ModelNotFoundError: If the target version does not exist.
    """
    cfg = _cfg(config)
    registry = cfg.tier1_ml.registry
    root = Path(repo_root) if repo_root is not None else find_project_root()
    mlflow_client = _client(cfg, client, repo_root=root)
    model_name = request.model_name or _model_name(cfg)
    reviewer = _require_reviewer(request.reviewer_id)

    version = _get_model_version(mlflow_client, model_name, request.target_version)
    version_number = str(version.version)
    tags = dict(version.tags or {})

    if _alias_version(mlflow_client, model_name, registry.candidate_alias) != version_number:
        raise _gate(
            f"version {version_number} is not the current '{registry.candidate_alias}' version",
            "not_a_candidate",
            version=version_number,
        )

    missing = [tag for tag in registry.required_lineage_tags if not tags.get(tag)]
    if missing:
        raise _gate(
            f"Missing lineage tag: {missing[0]}",
            f"missing_lineage_tag:{missing[0]}",
            missing=missing,
        )

    recorded_hash = tags["calibration_hash"].lower()
    if request.calibration_report_hash.strip().lower() != recorded_hash:
        raise _gate(
            "Calibration report hash mismatch: review invalid",
            "calibration_hash_mismatch",
            recorded=recorded_hash,
        )

    report_path = Path(
        calibration_report_path
        if calibration_report_path is not None
        else root / cfg.tier1_ml.registry_paths.calibration_report
    )
    if not report_path.is_file():
        raise _gate(
            f"Calibration report unavailable for drift check: {report_path}",
            "calibration_report_unavailable",
            path=str(report_path),
        )
    actual_hash = compute_file_sha256(report_path)
    if actual_hash != recorded_hash:
        raise _gate(
            "Calibration report changed after registration",
            "calibration_report_drift",
            recorded=recorded_hash,
            actual=actual_hash,
        )

    previous = _alias_version(mlflow_client, model_name, registry.approved_alias)
    promoted_at = _utc_now()
    audit_tags = {
        "promoted_by": reviewer,
        "promoted_at": promoted_at,
        "promotion_rationale": request.review_notes,
        "promotion_calibration_hash": recorded_hash,
    }
    mlflow_client.set_registered_model_alias(
        model_name, registry.approved_alias, version_number
    )
    for key, value in audit_tags.items():
        mlflow_client.set_model_version_tag(model_name, version_number, key, value)

    return PromotionAuditRecord(
        model_name=model_name,
        previous_approved_version=previous,
        new_approved_version=version_number,
        reviewer_id=reviewer,
        action="promote",
        calibration_verified=True,
        audit_tags=audit_tags,
        timestamp_utc=promoted_at,
    )


def rollback_approved(
    *,
    target_version: str,
    reviewer_id: str,
    reason: str,
    config: AEGISConfig | None = None,
    client: MlflowClient | None = None,
    repo_root: Path | None = None,
) -> PromotionAuditRecord:
    """Re-points the ``approved`` alias at a previously approved version.

    Raises:
        PromotionGateError: If the reviewer, reason, or target version is invalid.
        ModelNotFoundError: If the target version does not exist.
    """
    cfg = _cfg(config)
    registry = cfg.tier1_ml.registry
    root = Path(repo_root) if repo_root is not None else find_project_root()
    mlflow_client = _client(cfg, client, repo_root=root)
    model_name = _model_name(cfg)
    reviewer = _require_reviewer(reviewer_id)
    if len((reason or "").strip()) < 5:
        raise _gate("Rollback requires an auditable reason", "rollback_reason_missing")

    version = _get_model_version(mlflow_client, model_name, str(target_version))
    version_number = str(version.version)
    tags = dict(version.tags or {})
    current = _alias_version(mlflow_client, model_name, registry.approved_alias)

    if not tags.get("promoted_by"):
        immediately_previous = (
            current is not None
            and current.isdigit()
            and version_number.isdigit()
            and int(version_number) == int(current) - 1
        )
        if not immediately_previous:
            raise _gate(
                f"version {version_number} was never approved; rollback target invalid",
                "target_never_approved",
                version=version_number,
            )

    rolled_back_at = _utc_now()
    audit_tags = {
        "promoted_by": reviewer,
        "promoted_at": rolled_back_at,
        "promotion_rationale": reason,
        ROLLBACK_TAG: current or "",
    }
    mlflow_client.set_registered_model_alias(
        model_name, registry.approved_alias, version_number
    )
    for key, value in audit_tags.items():
        mlflow_client.set_model_version_tag(model_name, version_number, key, value)

    return PromotionAuditRecord(
        model_name=model_name,
        previous_approved_version=current,
        new_approved_version=version_number,
        reviewer_id=reviewer,
        action="rollback",
        calibration_verified=bool(tags.get("promotion_calibration_hash")),
        audit_tags=audit_tags,
        timestamp_utc=rolled_back_at,
    )


def load_approved_segments(
    config: AEGISConfig | None = None,
    *,
    client: MlflowClient | None = None,
    repo_root: Path | None = None,
) -> ApprovedSegmentPackage:
    """Loads the plain-data segment bundle behind the ``approved`` alias.

    Raises:
        ModelNotFoundError: If no version is assigned to the ``approved`` alias.
        RegistrySecurityError: If the artifact tree contains an estimator binary.
        RegistryPackagingError: If the packaged segment table is invalid.
    """
    cfg = _cfg(config)
    registry = cfg.tier1_ml.registry
    root = Path(repo_root) if repo_root is not None else find_project_root()
    mlflow_client = _client(cfg, client, repo_root=root)
    model_name = _model_name(cfg)

    version = _alias_version(mlflow_client, model_name, registry.approved_alias)
    if version is None:
        raise ModelNotFoundError(
            f"No model version currently assigned to alias '{registry.approved_alias}'",
            details={"model_name": model_name, "alias": registry.approved_alias},
        )

    local_dir = Path(
        mlflow_client.download_artifacts(
            _version_run_id(mlflow_client, model_name, version), SERVING_ARTIFACT_PATH
        )
    )
    assert_no_estimator_artifacts([local_dir])

    segments = read_segment_table(local_dir / SEGMENT_TABLE_FILE)
    lineage_payload = json.loads((local_dir / LINEAGE_FILE).read_text(encoding="utf-8"))
    lineage = ModelLineageMetadata.model_validate(lineage_payload)
    if lineage.model_version != version:
        raise RegistryPackagingError(
            "bundle lineage does not describe the approved model version",
            details={"alias_version": version, "lineage_version": lineage.model_version},
        )

    summary_path = local_dir / VALIDATION_SUMMARY_FILE
    validation_summary = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    )
    run_id = _version_run_id(mlflow_client, model_name, version)

    return ApprovedSegmentPackage(
        model_name=model_name,
        model_version=version,
        run_id=run_id,
        segments=segments,
        lineage=lineage,
        validation_summary=validation_summary,
    )


def _version_run_id(client: MlflowClient, model_name: str, version: str) -> str:
    """Returns the run id that produced a model version."""
    return _get_model_version(client, model_name, version).run_id


def _audit_payload(record: PromotionAuditRecord) -> str:
    """Renders an audit record as indented JSON for CLI output."""
    return json.dumps(record.model_dump(), indent=2, sort_keys=True)


def _cli_promote(args: argparse.Namespace) -> int:
    """CLI handler for `promote-model`."""
    try:
        request = ModelPromotionRequest(
            model_name=args.model_name or get_config().tier1_ml.registry.model_name,
            target_version=str(args.target_version),
            reviewer_id=args.reviewer_id,
            calibration_report_hash=args.calibration_hash,
            review_notes=args.review_note,
        )
    except ValidationError as exc:
        print(f"[BLOCK] Invalid promotion request: {exc}")
        return 1
    try:
        record = promote_to_approved(
            request,
            calibration_report_path=Path(args.calibration_report)
            if args.calibration_report
            else None,
        )
    except (PromotionGateError, ModelNotFoundError) as exc:
        print(f"[BLOCK] Promotion refused: {exc}")
        return 1
    print(_audit_payload(record))
    return 0


def _cli_rollback(args: argparse.Namespace) -> int:
    """CLI handler for `rollback-model`."""
    try:
        record = rollback_approved(
            target_version=str(args.target_version),
            reviewer_id=args.reviewer_id,
            reason=args.reason,
        )
    except (PromotionGateError, ModelNotFoundError) as exc:
        print(f"[BLOCK] Rollback refused: {exc}")
        return 1
    print(_audit_payload(record))
    return 0


def _cli_register_candidate(_args: argparse.Namespace) -> int:
    """CLI handler for `register-candidate` (never assigns `approved`)."""
    try:
        lineage = package_and_register_candidate()
    except RegistryPackagingError as exc:
        print(f"[BLOCK] Candidate registration failed: {exc}")
        return 1
    print(json.dumps(lineage.model_dump(), indent=2, sort_keys=True))
    return 0


def add_registry_subparsers(subparsers: Any) -> None:
    """Registers the registry subcommands on the pipeline CLI.

    Args:
        subparsers: The ``add_subparsers`` action from the pipeline parser.
    """
    p_register = subparsers.add_parser(
        "register-candidate",
        help="Package Tier 1 artifacts and register a candidate version (never 'approved')",
    )
    p_register.set_defaults(func=_cli_register_candidate)

    p_promote = subparsers.add_parser(
        "promote-model",
        help="Human-gated promotion of a candidate to the 'approved' alias (P2-F09)",
    )
    p_promote.add_argument("--version", dest="target_version", required=True)
    p_promote.add_argument("--reviewer-id", required=True)
    p_promote.add_argument("--calibration-hash", required=True)
    p_promote.add_argument("--review-note", required=True)
    p_promote.add_argument("--model-name", default=None)
    p_promote.add_argument("--calibration-report", default=None)
    p_promote.set_defaults(func=_cli_promote)

    p_rollback = subparsers.add_parser(
        "rollback-model",
        help="Re-point the 'approved' alias to a previously approved version (P2-F09)",
    )
    p_rollback.add_argument("--version", dest="target_version", required=True)
    p_rollback.add_argument("--reviewer-id", required=True)
    p_rollback.add_argument("--reason", required=True)
    p_rollback.set_defaults(func=_cli_rollback)
