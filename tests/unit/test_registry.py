"""Unit tests for the MLflow registry, promotion gate, and serving loader (P2-F09).

Every test runs against an isolated SQLite backend under ``tmp_path``; the real
``artifacts/mlflow.db`` is never touched. The suite encodes the Gate 12 falsification
protocol: promotion without a human review record, with a mismatched calibration hash,
or with incomplete lineage must fail closed and leave aliases unchanged (P2-D11a).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from pydantic import ValidationError

from aegis.config.loader import get_config
from aegis.config.schema import AEGISConfig
from aegis.pipelines import artifact_package
from aegis.pipelines.cli import build_parser
from aegis.pipelines.registry import (
    load_approved_segments,
    package_and_register_candidate,
    promote_to_approved,
    resolve_tracking_uri,
    rollback_approved,
)
from aegis.schemas.registry import ModelPromotionRequest
from aegis.utils.exceptions import (
    ModelNotFoundError,
    PromotionGateError,
    RegistrySecurityError,
)

MODEL_NAME = "aegis-causal-elasticity"
SEGMENT_A = "D:18-24_C:0-4_Dens:<=500"
SEGMENT_B = "D:35-49_C:5-9_Dens:501-2000"
CALIBRATION_HASH_KEYS = ("git_commit", "dvc_data_hash", "dgp_manifest_id", "calibration_hash")


@dataclass
class RegistryEnv:
    """An isolated registry backend plus the configuration that points at it."""

    config: AEGISConfig
    client: MlflowClient
    root: Path

    @property
    def calibration_report(self) -> Path:
        """Path to the calibration report the reviewer inspects."""
        return self.root / self.config.tier1_ml.registry_paths.calibration_report


def _write_repo(root: Path) -> None:
    """Materializes a minimal repository tree with all packaging inputs."""
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    artifacts = root / "artifacts"
    (artifacts / "causal_estimator").mkdir(parents=True)

    pd.DataFrame(
        {
            "segment_id": [SEGMENT_A, SEGMENT_B],
            "driver_age_band": ["18-24", "35-49"],
            "car_age_band": ["0-4", "5-9"],
            "density_band": ["<=500", "501-2000"],
            "is_merged": [False, False],
            "policy_count": [1200, 1500],
            "exposure_years": [400.5, 500.25],
            "loss_cost_per_year": [410.0, 380.5],
            "base_premium_per_year": [520.0, 470.25],
        }
    ).to_csv(processed / "segment_loss_cost.csv", index=False)
    pd.DataFrame(
        {
            "segment_id": [SEGMENT_A, SEGMENT_B],
            "inference_policy_count": [300, 350],
            "baseline_retention_q0": [0.83, 0.87],
            "tau_hat": [-0.58, -0.41],
            "tau_ci_lower": [-1.9, -1.5],
            "tau_ci_upper": [0.7, 0.6],
            "epsilon_hat": [-0.70, -0.52],
            "epsilon_ci_lower": [-2.4, -1.9],
            "epsilon_ci_upper": [1.0, 0.9],
        }
    ).to_csv(processed / "segment_elasticity_estimates.csv", index=False)
    (processed / "causal_estimator_metadata.json").write_text(
        json.dumps({"model_version": "cfdml-test-version"}), encoding="utf-8"
    )
    (processed / "dgp_manifest.json").write_text(
        json.dumps({"config_fingerprint": "a" * 64}), encoding="utf-8"
    )
    (processed / "glm_calibration_report.json").write_text(
        json.dumps({"cap": 100000.0, "excess_load": 1.1988}), encoding="utf-8"
    )
    (processed / "elasticity_training.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (artifacts / "causal_validation_summary.json").write_text(
        json.dumps({"status": "GATE_PASSED_OPTION_F"}), encoding="utf-8"
    )
    (artifacts / "sensitivity_summary.json").write_text(
        json.dumps({"robustness_value": 0.5004}), encoding="utf-8"
    )
    (artifacts / "causal_estimator" / "estimator.joblib").write_bytes(b"opaque-estimator")


def _fake_run_git(_repo_root: Path, *args: str) -> str:
    """Stands in for git so tests do not depend on the ambient worktree state."""
    if args[0] == "rev-parse":
        return "b" * 40
    if args[0] == "status":
        return ""
    return ""


def _config_for(root: Path, tracking_uri: str) -> AEGISConfig:
    """Builds a project configuration pointed at the isolated repository."""
    base = get_config()
    registry = base.tier1_ml.registry.model_copy(
        update={
            "tracking_uri": tracking_uri,
            "experiment_name": "test-registry",
            "model_name": MODEL_NAME,
        }
    )
    paths = base.tier1_ml.registry_paths.model_copy(
        update={
            "segment_loss_cost": "data/processed/segment_loss_cost.csv",
            "segment_estimates": "data/processed/segment_elasticity_estimates.csv",
            "estimator_metadata": "data/processed/causal_estimator_metadata.json",
            "dgp_manifest": "data/processed/dgp_manifest.json",
            "calibration_report": "data/processed/glm_calibration_report.json",
            "validation_summary": "artifacts/causal_validation_summary.json",
            "sensitivity_summary": "artifacts/sensitivity_summary.json",
            "dvc_lock": "dvc.lock",
            "staging_dir": "artifacts/registry_staging",
        }
    )
    tier1 = base.tier1_ml.model_copy(
        update={"registry": registry, "registry_paths": paths}
    )
    return base.model_copy(update={"tier1_ml": tier1})


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    """Provides an isolated SQLite registry backend and matching configuration."""
    root = tmp_path / "repo"
    root.mkdir()
    _write_repo(root)
    monkeypatch.setattr(artifact_package, "_run_git", _fake_run_git)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    return RegistryEnv(
        config=_config_for(root, tracking_uri),
        client=MlflowClient(tracking_uri=tracking_uri),
        root=root,
    )


def _register(env: RegistryEnv) -> str:
    """Registers a candidate and returns its model version."""
    lineage = package_and_register_candidate(
        env.config, client=env.client, repo_root=env.root
    )
    return lineage.model_version


def _promote(env: RegistryEnv, version: str, **overrides: str) -> str:
    """Promotes a version with a valid-looking human review record."""
    payload = {
        "model_name": MODEL_NAME,
        "target_version": version,
        "reviewer_id": "actuarial.reviewer",
        "review_notes": "Reviewed calibration deciles and loss-cap manifest.",
    }
    payload.update(overrides)
    if "calibration_report_hash" not in payload:
        payload["calibration_report_hash"] = _report_hash(env)
    record = promote_to_approved(
        ModelPromotionRequest(**payload),
        config=env.config,
        client=env.client,
        repo_root=env.root,
    )
    return record.new_approved_version


def _report_hash(env: RegistryEnv) -> str:
    """Returns the SHA-256 of the calibration report the reviewer inspects."""
    from aegis.pipelines.artifact_package import compute_file_sha256

    return compute_file_sha256(env.calibration_report)


def _alias(env: RegistryEnv, alias: str) -> str | None:
    """Returns the version behind an alias, or None when the alias is unset."""
    try:
        return str(env.client.get_model_version_by_alias(MODEL_NAME, alias).version)
    except Exception:
        return None


def test_candidate_registration_sets_only_candidate(env: RegistryEnv) -> None:
    """Verifies registration assigns `candidate` and never touches `approved`."""
    version = _register(env)

    assert _alias(env, "candidate") == version
    assert _alias(env, "approved") is None

    tags = env.client.get_model_version(MODEL_NAME, version).tags
    for key in CALIBRATION_HASH_KEYS + ("validation_status",):
        assert tags.get(key), f"missing lineage tag {key}"
    assert tags["validation_status"] == "GATE_PASSED_OPTION_F"
    assert tags["git_commit"] == "b" * 40


def test_estimator_logged_outside_serving_bundle(env: RegistryEnv) -> None:
    """Verifies the fitted estimator is a run artifact, not part of the bundle."""
    version = _register(env)
    run_id = str(env.client.get_model_version(MODEL_NAME, version).run_id)
    estimator_dir = Path(env.client.download_artifacts(run_id, "estimator"))
    assert [entry.name for entry in estimator_dir.iterdir()] == ["estimator.joblib"]

    bundle_dir = Path(env.client.download_artifacts(run_id, "serving_bundle"))
    assert sorted(entry.name for entry in bundle_dir.iterdir()) == [
        "causal_validation_summary.json",
        "glm_calibration_report.json",
        "lineage.json",
        "sensitivity_summary.json",
        "tier1_segments.csv",
    ]


def test_promotion_without_reviewer_fails(env: RegistryEnv) -> None:
    """Falsification (a): a missing reviewer identity is rejected before any alias change."""
    version = _register(env)

    with pytest.raises(ValidationError):
        ModelPromotionRequest(
            model_name=MODEL_NAME,
            target_version=version,
            reviewer_id="",
            calibration_report_hash=_report_hash(env),
            review_notes="Reviewed calibration deciles.",
        )

    bypassed = ModelPromotionRequest.model_construct(
        model_name=MODEL_NAME,
        target_version=version,
        reviewer_id="  ",
        calibration_report_hash=_report_hash(env),
        review_notes="Reviewed calibration deciles.",
    )
    with pytest.raises(PromotionGateError) as excinfo:
        promote_to_approved(
            bypassed, config=env.config, client=env.client, repo_root=env.root
        )
    assert excinfo.value.details["reason"] == "reviewer_id_missing"
    assert _alias(env, "approved") is None


def test_promotion_with_mismatched_hash_fails(env: RegistryEnv) -> None:
    """Falsification (b): a mismatched calibration hash blocks promotion."""
    version = _register(env)

    with pytest.raises(PromotionGateError) as excinfo:
        _promote(env, version, calibration_report_hash="0" * 64)
    assert excinfo.value.details["reason"] == "calibration_hash_mismatch"
    assert _alias(env, "approved") is None


def test_promotion_with_missing_lineage_tag_fails(env: RegistryEnv) -> None:
    """Falsification (c): incomplete lineage blocks promotion."""
    version = _register(env)
    env.client.delete_model_version_tag(MODEL_NAME, version, "git_commit")

    with pytest.raises(PromotionGateError) as excinfo:
        _promote(env, version)
    assert excinfo.value.details["reason"] == "missing_lineage_tag:git_commit"
    assert _alias(env, "approved") is None


def test_promotion_success_sets_approved_and_audit_tags(env: RegistryEnv) -> None:
    """Verifies a valid human review record promotes and records the audit tags."""
    version = _register(env)
    promoted = _promote(env, version)

    assert promoted == version
    assert _alias(env, "approved") == version
    assert _alias(env, "candidate") == version

    tags = env.client.get_model_version(MODEL_NAME, version).tags
    assert tags["promoted_by"] == "actuarial.reviewer"
    assert tags["promotion_rationale"].startswith("Reviewed calibration")
    assert tags["promotion_calibration_hash"] == _report_hash(env)
    assert tags["promoted_at"]


def test_promotion_rejects_non_candidate_version(env: RegistryEnv) -> None:
    """Verifies a version that is not the current candidate cannot be promoted."""
    first = _register(env)
    second = _register(env)
    assert second != first
    assert _alias(env, "candidate") == second

    with pytest.raises(PromotionGateError) as excinfo:
        _promote(env, first)
    assert excinfo.value.details["reason"] == "not_a_candidate"
    assert _alias(env, "approved") is None


def test_calibration_drift_detected(env: RegistryEnv) -> None:
    """Verifies a calibration report edited after registration blocks promotion."""
    version = _register(env)
    recorded_hash = _report_hash(env)
    env.calibration_report.write_text(
        json.dumps({"cap": 90000.0, "excess_load": 1.2}), encoding="utf-8"
    )

    with pytest.raises(PromotionGateError) as excinfo:
        _promote(env, version, calibration_report_hash=recorded_hash)
    assert excinfo.value.details["reason"] == "calibration_report_drift"
    assert _alias(env, "approved") is None


def test_promotion_without_calibration_report_on_disk_fails(env: RegistryEnv) -> None:
    """Verifies the drift re-check fails closed when the report cannot be read."""
    version = _register(env)
    recorded_hash = _report_hash(env)
    env.calibration_report.unlink()

    with pytest.raises(PromotionGateError) as excinfo:
        _promote(env, version, calibration_report_hash=recorded_hash)
    assert excinfo.value.details["reason"] == "calibration_report_unavailable"


def test_rollback_repoints_approved(env: RegistryEnv) -> None:
    """Verifies rollback re-points `approved` and records the discarded version."""
    first = _promote(env, _register(env))
    second = _promote(env, _register(env))
    assert _alias(env, "approved") == second

    record = rollback_approved(
        target_version=first,
        reviewer_id="actuarial.reviewer",
        reason="Calibration deciles degraded in production monitoring.",
        config=env.config,
        client=env.client,
        repo_root=env.root,
    )

    assert record.action == "rollback"
    assert record.previous_approved_version == second
    assert record.new_approved_version == first
    assert _alias(env, "approved") == first
    tags = env.client.get_model_version(MODEL_NAME, first).tags
    assert tags["rollback_from"] == second


def test_rollback_unknown_version_fails(env: RegistryEnv) -> None:
    """Verifies rollback to a nonexistent version fails and leaves the alias alone."""
    version = _promote(env, _register(env))

    with pytest.raises(ModelNotFoundError):
        rollback_approved(
            target_version="999",
            reviewer_id="actuarial.reviewer",
            reason="Attempting an invalid rollback target.",
            config=env.config,
            client=env.client,
            repo_root=env.root,
        )
    assert _alias(env, "approved") == version


def test_rollback_requires_reason_and_reviewer(env: RegistryEnv) -> None:
    """Verifies rollback demands an auditable reviewer and rationale."""
    first = _promote(env, _register(env))
    _promote(env, _register(env))

    with pytest.raises(PromotionGateError) as excinfo:
        rollback_approved(
            target_version=first,
            reviewer_id="ab",
            reason="Calibration degraded materially in monitoring.",
            config=env.config,
            client=env.client,
            repo_root=env.root,
        )
    assert excinfo.value.details["reason"] == "reviewer_id_missing"

    with pytest.raises(PromotionGateError) as excinfo:
        rollback_approved(
            target_version=first,
            reviewer_id="actuarial.reviewer",
            reason="bad",
            config=env.config,
            client=env.client,
            repo_root=env.root,
        )
    assert excinfo.value.details["reason"] == "rollback_reason_missing"


def test_approved_missing_raises_model_not_found(env: RegistryEnv) -> None:
    """Verifies the serving loader fails closed when no version is approved."""
    _register(env)

    with pytest.raises(ModelNotFoundError, match="alias 'approved'"):
        load_approved_segments(env.config, client=env.client, repo_root=env.root)


def test_serving_loader_returns_plain_segments(env: RegistryEnv) -> None:
    """Verifies the approved loader returns validated plain-data segments and lineage."""
    version = _promote(env, _register(env))

    package = load_approved_segments(env.config, client=env.client, repo_root=env.root)

    assert package.model_version == version
    assert package.model_name == MODEL_NAME
    assert len(package.segments.segments) == 2
    assert package.lineage.model_version == version
    assert package.validation_summary["status"] == "GATE_PASSED_OPTION_F"
    assert package.segments.segments[0].policy_count == 1200


def test_serving_loader_rejects_estimator_binary(env: RegistryEnv) -> None:
    """Verifies a pickle placed in the serving bundle blocks the loader."""
    version = _promote(env, _register(env))
    run_id = env.client.get_model_version(MODEL_NAME, version).run_id
    stray = env.root / "stray.pkl"
    stray.write_bytes(b"pickled-estimator")
    env.client.log_artifact(run_id, str(stray), artifact_path="serving_bundle")

    with pytest.raises(RegistrySecurityError, match="serialized estimator artifacts"):
        load_approved_segments(env.config, client=env.client, repo_root=env.root)


def test_registry_module_never_imports_deserializers() -> None:
    """Structurally enforces that the serving path cannot deserialize an estimator."""
    source = Path(artifact_package.__file__).with_name("registry.py").read_text(encoding="utf-8")
    for forbidden in ("import joblib", "import pickle", "from joblib", "import econml"):
        assert forbidden not in source


def test_resolve_tracking_uri_prefers_environment(env: RegistryEnv, monkeypatch) -> None:
    """Verifies MLFLOW_TRACKING_URI overrides the configured default."""
    from_config = resolve_tracking_uri(env.config, repo_root=env.root)
    assert from_config.endswith("mlflow.db")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///override.db")
    assert resolve_tracking_uri(env.config, repo_root=env.root).endswith("override.db")


def test_cli_parsers_registered() -> None:
    """Verifies the registry subcommands are wired into the pipeline CLI."""
    parser = build_parser()
    promote = parser.parse_args(
        [
            "promote-model",
            "--version",
            "1",
            "--reviewer-id",
            "actuarial.reviewer",
            "--calibration-hash",
            "a" * 64,
            "--review-note",
            "Reviewed calibration deciles and cap.",
        ]
    )
    assert promote.command == "promote-model"
    assert callable(promote.func)

    rollback = parser.parse_args(
        [
            "rollback-model",
            "--version",
            "1",
            "--reviewer-id",
            "actuarial.reviewer",
            "--reason",
            "Degraded monitoring.",
        ]
    )
    assert rollback.command == "rollback-model"
    assert callable(rollback.func)

    register = parser.parse_args(["register-candidate"])
    assert register.command == "register-candidate"
    assert callable(register.func)


def test_cli_promote_reports_gate_failure(env: RegistryEnv, monkeypatch, capsys) -> None:
    """Verifies the CLI handler exits non-zero and prints the gate reason."""
    from aegis.pipelines.registry import _cli_promote

    version = _register(env)
    args = build_parser().parse_args(
        [
            "promote-model",
            "--version",
            version,
            "--reviewer-id",
            "actuarial.reviewer",
            "--calibration-hash",
            "0" * 64,
            "--review-note",
            "Reviewed calibration deciles.",
        ]
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{env.root.parent / 'mlflow.db'}")
    assert _cli_promote(args) == 1
    assert "Promotion refused" in capsys.readouterr().out
