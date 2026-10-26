"""Unit tests for DGP Pydantic schemas and exceptions (P2-F04)."""

import pytest
from pydantic import ValidationError

from aegis.schemas.dgp import DGPManifest, SelfTestResult, TruthSidecarRow
from aegis.utils.exceptions import DGPVerificationError, TruthIsolationError


def test_self_test_result_valid() -> None:
    res = SelfTestResult(
        name="overlap",
        passed=True,
        observed_value=0.0025,
        threshold=0.001,
        details={"min_var": 0.0025},
    )
    assert res.passed is True
    assert res.observed_value == 0.0025


def test_dgp_manifest_valid() -> None:
    manifest = DGPManifest(
        random_seed=42,
        sample_size=413169,
        corridor_width=0.15,
        confounding_dial=0.5,
        residual_std=0.05,
        self_tests={
            "overlap": SelfTestResult(
                name="overlap",
                passed=True,
                observed_value=0.0025,
                threshold=0.001,
            )
        },
        input_hashes={"features": "a" * 64},
        config_fingerprint="b" * 64,
    )
    assert manifest.sample_size == 413169
    assert manifest.self_tests["overlap"].passed is True


def test_truth_sidecar_row_valid() -> None:
    row = TruthSidecarRow(
        PolicyID=123,
        baseline_renewal_q0=0.835,
        true_tau=-0.25,
        true_epsilon=-0.30,
        segment_id="D:25-44_C:0-4_Dens:<=500",
    )
    assert row.PolicyID == 123
    assert 0.0 <= row.baseline_renewal_q0 <= 1.0


def test_truth_sidecar_row_invalid_q0() -> None:
    with pytest.raises(ValidationError):
        TruthSidecarRow(
            PolicyID=123,
            baseline_renewal_q0=1.2,  # > 1.0 invalid
            true_tau=-0.25,
            true_epsilon=-0.30,
            segment_id="D:25-44_C:0-4_Dens:<=500",
        )


def test_dgp_exceptions() -> None:
    dgp_err = DGPVerificationError("Self test failed", {"test": "overlap"})
    assert "Self test failed" in str(dgp_err)
    assert dgp_err.details["test"] == "overlap"

    iso_err = TruthIsolationError("Truth accessed", {"caller": "causal"})
    assert "Truth accessed" in str(iso_err)
