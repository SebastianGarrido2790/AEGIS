"""DAG assembly, INV-3 ordering, and reproducibility contracts (P2-F10, Gate 14).

The suite reads the live ``dvc.yaml`` rather than a copy, so a stage that is renamed or
rewired without its upstream gate fails here. It also proves the version manifest is
deterministic and fails closed, and that the Tier 1 gate values load from ``params.yaml``
instead of being hardcoded in CI (ADR-015, P2-D10).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from aegis.config.loader import get_config
from aegis.pipelines.causal_artifacts import build_causal_validation_report
from aegis.pipelines.cli import build_parser
from aegis.pipelines.versioning import (
    CANONICAL_ARTIFACTS,
    build_version_manifest,
    combined_digest,
)
from aegis.schemas.validation import ValidationMetricReport, ValidationMetrics
from aegis.utils.exceptions import VersioningError

DVC_PATH = Path("dvc.yaml")
REGISTER_COMMAND = "register-candidate"


def _stages() -> dict[str, dict[str, Any]]:
    """Loads the live DVC stage definitions."""
    return yaml.safe_load(DVC_PATH.read_text(encoding="utf-8"))["stages"]


def _producers() -> dict[str, str]:
    """Maps each declared output path to the stage that produces it."""
    mapping: dict[str, str] = {}
    for name, stage in _stages().items():
        for out in stage.get("outs", []) or []:
            mapping[str(out)] = name
    return mapping


def _ancestors(stage: str) -> set[str]:
    """Returns every stage that ``stage`` transitively depends on."""
    stages = _stages()
    producers = _producers()
    seen: set[str] = set()
    pending = [stage]
    while pending:
        current = pending.pop()
        for dep in stages[current].get("deps", []) or []:
            parent = producers.get(str(dep))
            if parent is not None and parent not in seen:
                seen.add(parent)
                pending.append(parent)
    return seen


def test_new_stages_complete_the_policy_dag() -> None:
    """The estimator, recovery, and artifact-versioning nodes exist with their outputs."""
    stages = _stages()

    assert "data/processed/segment_elasticity_estimates.csv" in stages["train_estimator"]["outs"]
    assert (
        "data/validated/causal_validation_report.json" in stages["validate_recovery"]["outs"]
    )
    assert "data/versioned/version_manifest.json" in stages["version_artifacts"]["outs"]


def test_registration_is_a_governed_manual_step() -> None:
    """Registry mutation stays out of the reproducible DAG (P2-D11a, clean-git lineage)."""
    stages = _stages()

    assert "register_candidate" not in stages
    assert all(REGISTER_COMMAND not in str(stage.get("cmd", "")) for stage in stages.values())


def test_versioning_stage_is_terminal() -> None:
    """`version_artifacts` consumes products but feeds no later stage."""
    stages = _stages()
    manifest = "data/versioned/version_manifest.json"

    assert not any(manifest in (stage.get("deps") or []) for stage in stages.values())
    assert _ancestors("version_artifacts") >= {
        "train_estimator",
        "validate_recovery",
        "version_raw",
    }
    assert stages["version_artifacts"]["deps"]


def test_inv3_ordering_gates_every_versioned_product() -> None:
    """No versioned or training product is reachable without its upstream GX gate."""
    assert "validate_raw_gx" in _ancestors("version_raw")
    assert "validate_training_gx" in _ancestors("train_estimator")
    assert "validate_training_gx" in _ancestors("validate_recovery")
    assert "validate_regulatory_gx" in _ancestors("version_regulatory")
    assert {"validate_raw_gx", "validate_training_gx"} <= _ancestors("version_artifacts")


def test_version_manifest_covers_declared_stage_outputs() -> None:
    """Every canonical manifest artifact is produced by a declared stage or is a GX report."""
    produced = set(_producers())

    for artifact in CANONICAL_ARTIFACTS:
        assert artifact in produced or artifact.startswith("data/validated/"), artifact


def test_version_manifest_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two builds over identical bytes produce an identical manifest payload."""
    files = []
    for index in range(2):
        target = tmp_path / f"artifact_{index}.json"
        target.write_text(json.dumps({"index": index}), encoding="utf-8")
        files.append(str(target))

    first = build_version_manifest(files)
    second = build_version_manifest(files)

    assert first == second
    assert first.combined_sha256 == combined_digest(first.artifacts)
    assert first.artifact_count == 2


def test_version_manifest_changes_when_an_artifact_changes(tmp_path: Path) -> None:
    """A single byte change moves the combined digest, localizing drift."""
    target = tmp_path / "artifact.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    before = build_version_manifest([str(target)])

    target.write_text("a,b\n1,3\n", encoding="utf-8")
    after = build_version_manifest([str(target)])

    assert before.combined_sha256 != after.combined_sha256


def test_version_manifest_fails_closed_on_missing_artifact(tmp_path: Path) -> None:
    """A missing declared artifact blocks versioning instead of silently skipping it."""
    with pytest.raises(VersioningError):
        build_version_manifest([str(tmp_path / "absent.csv")])


def test_version_artifacts_command_is_registered() -> None:
    """The terminal DVC stage command exists on the pipeline CLI."""
    parser = build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["version-artifacts", "--help"])

    assert excinfo.value.code == 0


def test_recovery_report_is_deterministic_and_gated() -> None:
    """The recovery artifact has no timestamp and reports frozen-threshold verdicts."""
    config = get_config()
    report = ValidationMetricReport(
        seed=42,
        sample_size=1000,
        metrics=ValidationMetrics(
            m1_tau_spearman=0.90,
            m1_epsilon_spearman=0.90,
            m1b_policy_spearman=0.90,
            m2_mean_abs_bias=0.10,
            m3_tau_coverage=1.0,
            m3_epsilon_coverage=1.0,
            m4_self_consistency=1.0,
            m5_sign_agreement=1.0,
        ),
    )
    thresholds = config.tier1_ml.validation_thresholds

    first = build_causal_validation_report(report, thresholds)
    second = build_causal_validation_report(report, thresholds)

    assert first == second
    assert first["thresholds_configured"] is True
    assert first["gate_passed"] is True
    assert "timestamp" not in json.dumps(first).lower()
    assert set(first["gate_checks"]) == {
        "m1_tau_spearman",
        "m1_epsilon_spearman",
        "m1b_policy_spearman",
        "m2_mean_abs_bias",
        "m3_tau_coverage",
        "m3_epsilon_coverage",
        "m4_self_consistency",
        "m5_sign_agreement",
    }


def test_recovery_report_without_frozen_thresholds_reports_no_verdict() -> None:
    """An unfrozen threshold set yields an explicit null verdict, never a silent pass."""
    report = ValidationMetricReport(
        seed=None,
        sample_size=1000,
        metrics=ValidationMetrics(
            m1_tau_spearman=0.5,
            m1_epsilon_spearman=0.5,
            m1b_policy_spearman=0.5,
            m2_mean_abs_bias=0.5,
            m3_tau_coverage=0.5,
            m3_epsilon_coverage=0.5,
            m4_self_consistency=0.5,
            m5_sign_agreement=0.5,
        ),
    )

    payload = build_causal_validation_report(report, None)

    assert payload["thresholds_configured"] is False
    assert payload["gate_passed"] is None


def test_execution_tiers_load_from_params() -> None:
    """Tier settings come from params.yaml, not from duplicated CI constants."""
    tiers = get_config().execution_tiers

    assert tiers.tier1_ci_smoke.coverage_floor == pytest.approx(0.65)
    assert tiers.tier1_ci_smoke.require_gx_fixture_falsification is True
    assert tiers.tier2_local_repro.require_bit_identical_products is True
    assert tiers.tier2_local_repro.register_candidate_manual is True
    assert tiers.tier3_validation_campaign.on_demand is True
    assert tiers.reproducibility.metric_abs_tolerance == pytest.approx(0.001)


def test_coverage_floor_cannot_be_fitted_down() -> None:
    """The Tier 1 coverage floor rejects a value outside (0, 1]."""
    from aegis.config.schema import Tier1CISmokeConfig

    with pytest.raises(ValidationError):
        Tier1CISmokeConfig(coverage_floor=0.0)
    with pytest.raises(ValidationError):
        Tier1CISmokeConfig(coverage_floor=1.5)
