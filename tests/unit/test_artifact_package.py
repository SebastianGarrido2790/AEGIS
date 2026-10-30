"""Unit tests for Tier 1 serving-bundle packaging (P2-F09, Stage 12).

Covers the Stage 4 x Stage 8 segment join, provenance resolution, the five-file
bundle contract, and the structural rule that no serialized estimator can enter a
serving bundle (ADR-014, P2-D11, INV-7).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aegis.pipelines.artifact_package import (
    STANDARD_BUNDLE_FILES,
    assert_no_estimator_artifacts,
    build_serving_bundle,
    compute_file_md5,
    compute_file_sha256,
    load_tier1_segment_table,
    read_segment_table,
    resolve_dvc_data_hash,
    resolve_git_commit,
    resolve_lineage,
    write_segment_table,
)
from aegis.schemas.registry import ModelLineageMetadata
from aegis.utils.exceptions import RegistryPackagingError, RegistrySecurityError

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_SEGMENT_ECONOMICS = REPO_ROOT / "data/processed/segment_loss_cost.csv"
REAL_ESTIMATES = REPO_ROOT / "data/processed/segment_elasticity_estimates.csv"
REAL_ESTIMATOR_METADATA = REPO_ROOT / "data/processed/causal_estimator_metadata.json"
REAL_DGP_MANIFEST = REPO_ROOT / "data/processed/dgp_manifest.json"

SEGMENT_A = "D:18-24_C:0-4_Dens:<=500"
SEGMENT_B = "D:35-49_C:5-9_Dens:501-2000"


def _write_inputs(tmp_path: Path, *, drop_estimate: bool = False) -> tuple[Path, Path, Path, Path]:
    """Writes a minimal Stage 4/8/10 artifact set and returns its paths."""
    economics = pd.DataFrame(
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
    )
    estimates = pd.DataFrame(
        {
            "segment_id": [SEGMENT_B] if drop_estimate else [SEGMENT_A, SEGMENT_B],
            "inference_policy_count": [300, 350] if not drop_estimate else [350],
            "baseline_retention_q0": [0.83, 0.87] if not drop_estimate else [0.87],
            "tau_hat": [-0.58, -0.41] if not drop_estimate else [-0.41],
            "tau_ci_lower": [-1.9, -1.5] if not drop_estimate else [-1.5],
            "tau_ci_upper": [0.7, 0.6] if not drop_estimate else [0.6],
            "epsilon_hat": [-0.70, -0.52] if not drop_estimate else [-0.52],
            "epsilon_ci_lower": [-2.4, -1.9] if not drop_estimate else [-1.9],
            "epsilon_ci_upper": [1.0, 0.9] if not drop_estimate else [0.9],
        }
    )
    economics_path = tmp_path / "segment_loss_cost.csv"
    estimates_path = tmp_path / "segment_estimates.csv"
    economics.to_csv(economics_path, index=False)
    estimates.to_csv(estimates_path, index=False)

    metadata_path = tmp_path / "estimator_metadata.json"
    metadata_path.write_text(json.dumps({"model_version": "cfdml-test-version"}), encoding="utf-8")
    manifest_path = tmp_path / "dgp_manifest.json"
    manifest_path.write_text(json.dumps({"config_fingerprint": "a" * 64}), encoding="utf-8")
    return economics_path, estimates_path, metadata_path, manifest_path


def _lineage(tmp_path: Path, *, segment_count: int = 2) -> ModelLineageMetadata:
    """Builds a lineage record pointing at an on-disk calibration report."""
    report = tmp_path / "glm_calibration_report.json"
    report.write_text(json.dumps({"cap": 100000.0}), encoding="utf-8")
    return ModelLineageMetadata(
        model_name="aegis-causal-elasticity",
        model_version="cfdml-test-version",
        run_id="run-1",
        git_commit="b" * 40,
        git_dirty=False,
        dvc_data_hash="c" * 32,
        dgp_manifest_id="d" * 64,
        calibration_hash=compute_file_sha256(report),
        validation_status="GATE_PASSED_OPTION_F",
        segment_count=segment_count,
        created_at_utc="2026-10-30T00:00:00+00:00",
    )


def test_join_produces_valid_segments(tmp_path: Path) -> None:
    """Verifies the join yields one validated row per segment with Stage 4 policy counts."""
    economics, estimates, metadata, manifest = _write_inputs(tmp_path)
    table = load_tier1_segment_table(economics, estimates, metadata, manifest)

    assert len(table.segments) == 2
    assert [segment.segment_id for segment in table.segments] == [SEGMENT_A, SEGMENT_B]
    assert table.segments[0].policy_count == 1200
    assert table.segments[0].model_version == "cfdml-test-version"
    assert table.segments[0].data_version == "a" * 64
    assert all(segment.semi_synthetic for segment in table.segments)


@pytest.mark.skipif(
    not (REAL_SEGMENT_ECONOMICS.is_file() and REAL_ESTIMATES.is_file()),
    reason="Phase 2 processed segment artifacts are not materialized",
)
def test_join_on_real_artifacts_yields_35_segments() -> None:
    """Verifies the real Stage 4/8 artifacts join into the 35 published segments."""
    table = load_tier1_segment_table(
        REAL_SEGMENT_ECONOMICS,
        REAL_ESTIMATES,
        REAL_ESTIMATOR_METADATA,
        REAL_DGP_MANIFEST,
    )
    assert len(table.segments) == 35
    assert all(segment.policy_count >= 1000 for segment in table.segments)


def test_join_rejects_missing_segment(tmp_path: Path) -> None:
    """Verifies a segment present in economics but absent from estimates fails loud."""
    economics, estimates, metadata, manifest = _write_inputs(tmp_path, drop_estimate=True)
    with pytest.raises(RegistryPackagingError, match="different segment sets"):
        load_tier1_segment_table(economics, estimates, metadata, manifest)


def test_join_rejects_duplicate_segment(tmp_path: Path) -> None:
    """Verifies duplicate segment ids are rejected before the join."""
    economics, estimates, metadata, manifest = _write_inputs(tmp_path)
    duplicated = pd.read_csv(estimates)
    pd.concat([duplicated, duplicated.iloc[[0]]]).to_csv(estimates, index=False)

    with pytest.raises(RegistryPackagingError, match="duplicate segment_id"):
        load_tier1_segment_table(economics, estimates, metadata, manifest)


def test_join_rejects_floor_violation(tmp_path: Path) -> None:
    """Verifies a segment below the 1,000-policy floor cannot be packaged."""
    economics, estimates, metadata, manifest = _write_inputs(tmp_path)
    frame = pd.read_csv(economics)
    frame.loc[0, "policy_count"] = 999
    frame.to_csv(economics, index=False)

    with pytest.raises(RegistryPackagingError, match="violates the Tier 1 contract"):
        load_tier1_segment_table(economics, estimates, metadata, manifest)


def test_dvc_hash_from_lock_and_fallback(tmp_path: Path) -> None:
    """Verifies the lock MD5 is authoritative and the SHA-256 fallback is used when absent."""
    training = tmp_path / "elasticity_training.csv"
    training.write_text("a,b\n1,2\n", encoding="utf-8")
    recorded = compute_file_md5(training)

    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "stages:\n"
        "  build_dgp:\n"
        "    outs:\n"
        "    - path: data/processed/elasticity_training.csv\n"
        "      hash: md5\n"
        f"      md5: {recorded}\n",
        encoding="utf-8",
    )
    from_lock = resolve_dvc_data_hash(lock, training)
    assert from_lock == recorded

    lock.write_text("stages: {}\n", encoding="utf-8")
    assert resolve_dvc_data_hash(lock, training) == compute_file_sha256(training)
    assert resolve_dvc_data_hash(tmp_path / "missing.lock", training) == compute_file_sha256(
        training
    )


def test_dvc_hash_rejects_stale_lock(tmp_path: Path) -> None:
    """Verifies a lock hash that no longer matches the file fails closed."""
    training = tmp_path / "elasticity_training.csv"
    training.write_text("a,b\n1,2\n", encoding="utf-8")
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "stages:\n"
        "  build_dgp:\n"
        "    outs:\n"
        "    - path: data/processed/elasticity_training.csv\n"
        "      hash: md5\n"
        "      md5: 0123456789abcdef0123456789abcdef\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistryPackagingError, match="does not match the training table"):
        resolve_dvc_data_hash(lock, training)


def test_lineage_requires_clean_commit_shape(tmp_path: Path) -> None:
    """Verifies lineage records a 40-hex commit and reports the dirty flag."""
    training = tmp_path / "elasticity_training.csv"
    training.write_text("a,b\n1,2\n", encoding="utf-8")
    report = tmp_path / "glm_calibration_report.json"
    report.write_text(json.dumps({"cap": 100000.0}), encoding="utf-8")
    manifest = tmp_path / "dgp_manifest.json"
    manifest.write_text(json.dumps({"config_fingerprint": "e" * 64}), encoding="utf-8")

    lineage = resolve_lineage(
        model_name="aegis-causal-elasticity",
        model_version="v1",
        run_id="run-1",
        repo_root=REPO_ROOT,
        calibration_report_path=report,
        dgp_manifest_path=manifest,
        segment_count=35,
        created_at_utc="2026-10-30T00:00:00+00:00",
    )
    assert len(lineage.git_commit) == 40
    assert set(lineage.git_commit) <= set("0123456789abcdef")
    assert isinstance(lineage.git_dirty, bool)
    assert lineage.calibration_hash == compute_file_sha256(report)
    assert lineage.validation_status == "GATE_PASSED_OPTION_F"


def test_git_commit_rejects_non_repository(tmp_path: Path) -> None:
    """Verifies provenance resolution fails loud outside a git worktree."""
    with pytest.raises(RegistryPackagingError, match="git rev-parse HEAD failed"):
        resolve_git_commit(tmp_path)


def test_bundle_contains_exactly_five_files(tmp_path: Path) -> None:
    """Verifies the bundle is exactly the standard five plain-data files."""
    economics, estimates, metadata, manifest = _write_inputs(tmp_path)
    table = load_tier1_segment_table(economics, estimates, metadata, manifest)
    lineage = _lineage(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in (
        "glm_calibration_report.json",
        "causal_validation_summary.json",
        "sensitivity_summary.json",
    ):
        (inputs / name).write_text(json.dumps({"ok": True}), encoding="utf-8")

    written = build_serving_bundle(
        tmp_path / "bundle",
        segments=table,
        calibration_report_path=inputs / "glm_calibration_report.json",
        validation_summary_path=inputs / "causal_validation_summary.json",
        sensitivity_summary_path=inputs / "sensitivity_summary.json",
        lineage=lineage,
    )

    assert [path.name for path in written] == list(STANDARD_BUNDLE_FILES)
    assert sorted(entry.name for entry in (tmp_path / "bundle").iterdir()) == sorted(
        STANDARD_BUNDLE_FILES
    )
    assert read_segment_table(tmp_path / "bundle" / "tier1_segments.csv").segments == table.segments
    stored = json.loads((tmp_path / "bundle" / "lineage.json").read_text(encoding="utf-8"))
    assert stored["calibration_hash"] == lineage.calibration_hash


def test_bundle_rejects_estimator_binary(tmp_path: Path) -> None:
    """Verifies a staged estimator binary blocks bundle construction."""
    staging = tmp_path / "bundle"
    staging.mkdir()
    (staging / "estimator.joblib").write_bytes(b"not-a-real-model")

    with pytest.raises(RegistrySecurityError, match="serialized estimator artifacts"):
        build_serving_bundle(
            staging,
            segments=load_tier1_segment_table(*_write_inputs(tmp_path)),
            calibration_report_path=tmp_path / "missing.json",
            validation_summary_path=tmp_path / "missing.json",
            sensitivity_summary_path=tmp_path / "missing.json",
            lineage=_lineage(tmp_path),
        )


def test_assert_no_estimator_artifacts_scans_directories(tmp_path: Path) -> None:
    """Verifies the guard walks a directory tree and accepts plain-data bundles."""
    clean = tmp_path / "clean"
    clean.mkdir()
    write_segment_table(
        load_tier1_segment_table(*_write_inputs(tmp_path)), clean / "tier1_segments.csv"
    )
    assert_no_estimator_artifacts([clean])

    (clean / "nested").mkdir()
    (clean / "nested" / "model.pkl").write_bytes(b"pickle")
    with pytest.raises(RegistrySecurityError, match="forbidden in the serving bundle"):
        assert_no_estimator_artifacts([clean])
