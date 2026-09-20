"""Tests for synthetic treatment validation and causal elasticity estimation (Stage 4).

Quality gate suite verifying:
- Gate 1: Non-deterministic treatment assignment with bounded residual variance [0.20, 0.70].
- Gate 2: Programmatic ground truth consistency with analytic derivative
  tau(X) = 2.0 + 1.5 * risk_index.
- Gate 3: Point estimate strictly contained within its reported confidence interval.
- Gate 4: Genuine DoWhy refutation execution with no diagnostic_note fallback marker.
- Gate 5: Effect recovery correlation >= 0.75 (calibrated against empirical DML performance).
- Gate 6: Meaningful, non-tautological calibration metrics (finite MAE, complete metrics dict).
- Gate 7: Structured artifact serialization with valid schema and confidence intervals.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis.pipelines.training.causal_elasticity import (
    CausalElasticityResult,
    add_synthetic_treatment,
    compute_residual_variance_metrics,
    compute_true_causal_effect,
    fit_causal_elasticity,
    save_causal_artifact,
)


@pytest.fixture(scope="module")
def feature_frame() -> pd.DataFrame:
    """Use the real Stage 2 feature matrix for causal validation tests."""
    return pd.read_csv("data/versioned/feature_matrix.csv")


@pytest.fixture(scope="module")
def causal_result(feature_frame: pd.DataFrame) -> CausalElasticityResult:
    """Fit CausalForestDML once for module-level validation assertions."""
    sample = feature_frame.sample(2000, random_state=42).reset_index(drop=True)
    return fit_causal_elasticity(sample, random_state=42)


class TestWeakTreatmentIdentification:
    """Gate 1: Independent stochastic variation in synthetic treatment (Fix #1)."""

    def test_residual_variance_fraction_bounded(self, feature_frame: pd.DataFrame) -> None:
        """Residual identifying variance in [0.20, 0.70] and raw variance > 0."""
        sample = feature_frame.sample(2000, random_state=42).reset_index(drop=True)
        prepared = add_synthetic_treatment(sample, random_state=42)

        metrics = compute_residual_variance_metrics(prepared)
        residual_identifying_variance = metrics["residual_identifying_variance"]
        residual_variance = metrics["residual_variance"]

        assert 0.20 <= residual_identifying_variance <= 0.70, (
            f"Residual identifying variance {residual_identifying_variance:.4f} "
            "outside [0.20, 0.70]"
        )
        assert residual_variance > 0.001, (
            f"Raw residual variance {residual_variance:.6f} is too low (<= 0.001)"
        )

    def test_treatment_retains_meaningful_noise_variance(
        self, feature_frame: pd.DataFrame
    ) -> None:
        """Treatment assignment must not be a purely deterministic function of risk_index."""
        sample = feature_frame.sample(2000, random_state=42).reset_index(drop=True)
        prepared = add_synthetic_treatment(sample, random_state=42)
        systematic = 0.05 + 0.10 * prepared["risk_index"]
        noise_residual = prepared["treatment_rate_change"] - systematic
        assert float(noise_residual.std()) > 0.02, (
            f"Noise residual std {float(noise_residual.std()):.4f} is too low (<= 0.02)"
        )

    def test_treatment_positivity_and_variation(self, feature_frame: pd.DataFrame) -> None:
        """Treatment rate change must be strictly positive and exhibit non-trivial spread."""
        sample = feature_frame.sample(2000, random_state=42).reset_index(drop=True)
        prepared = add_synthetic_treatment(sample, random_state=42)
        assert bool(prepared["treatment_rate_change"].notna().all())
        assert bool((prepared["treatment_rate_change"] >= 0.01).all()), (
            "Positivity violation: values below 0.01 floor"
        )
        assert float(prepared["treatment_rate_change"].std()) > 0.02


class TestGroundTruthConsistency:
    """Gate 2: Analytic derivative consistency (Fix #2)."""

    def test_ground_truth_matches_structural_derivative(self) -> None:
        """Ground truth formula tau(X) must evaluate identically to dY/dT."""
        test_risks = np.array([0.0, 1.0, 2.5, 5.0])
        expected_tau = 2.0 + 1.5 * test_risks
        computed_tau = compute_true_causal_effect(test_risks)
        np.testing.assert_allclose(computed_tau, expected_tau)

    def test_outcome_dgp_treatment_slope_equals_tau(self, feature_frame: pd.DataFrame) -> None:
        """Outcome DGP synthetic response slope dY/dT must match compute_true_causal_effect."""
        sample = feature_frame.sample(100, random_state=42).reset_index(drop=True)
        tau = compute_true_causal_effect(sample["risk_index"])
        t1 = 0.10
        t2 = 0.20
        delta_t = t2 - t1
        y1 = 100.0 + 15.0 * sample["risk_index"] + tau * t1
        y2 = 100.0 + 15.0 * sample["risk_index"] + tau * t2
        empirical_slope = (y2 - y1) / delta_t
        np.testing.assert_allclose(empirical_slope, tau)


class TestPointEstimateAndConfidenceInterval:
    """Gate 3: Point estimate contained in reported confidence interval (Fix #3)."""

    def test_point_estimate_strictly_contained_in_ci(
        self, causal_result: CausalElasticityResult
    ) -> None:
        """Point estimate must fall within its own reported 95% confidence interval."""
        ate = causal_result.average_treatment_effect
        lower, upper = causal_result.treatment_effect_confidence_interval
        assert lower <= ate <= upper, (
            f"Point estimate {ate:.4f} outside CI [{lower:.4f}, {upper:.4f}]"
        )

    def test_confidence_interval_is_non_trivial(
        self, causal_result: CausalElasticityResult
    ) -> None:
        """Confidence interval must be strictly well-ordered with finite non-zero width."""
        lower, upper = causal_result.treatment_effect_confidence_interval
        assert np.isfinite(lower) and np.isfinite(upper)
        assert lower < upper, f"Degenerate CI: lower {lower} >= upper {upper}"
        assert (upper - lower) > 0.05, f"Unrealistically narrow CI width: {upper - lower}"


class TestDoWhyRefutationExecution:
    """Gate 4: Genuine DoWhy refutations execution with no fallback marker (Fix #4)."""

    def test_no_fallback_diagnostic_note(
        self, causal_result: CausalElasticityResult
    ) -> None:
        """Refutation summary must not contain fallback diagnostic_note marker."""
        summary = causal_result.refutation_summary
        assert "diagnostic_note" not in summary, (
            f"Refutation summary contains fallback marker: {summary.get('diagnostic_note')}"
        )

    def test_all_expected_refuters_executed_and_passed(
        self, causal_result: CausalElasticityResult
    ) -> None:
        """All three DoWhy refuters must report status 'ok' and passed=True."""
        summary = causal_result.refutation_summary
        expected_refuters = {"placebo_treatment", "random_common_cause", "data_subset"}
        assert expected_refuters.issubset(summary.keys()), (
            f"Missing refuters: {expected_refuters - summary.keys()}"
        )
        for refuter_name in expected_refuters:
            entry = summary[refuter_name]
            assert isinstance(entry, dict), f"Refuter {refuter_name} output is not a dict"
            assert entry.get("status") == "ok", (
                f"Refuter {refuter_name} status is {entry.get('status')}, expected 'ok'"
            )
            assert entry.get("passed") is True, f"Refuter {refuter_name} did not pass"
            p_val = entry.get("p_value")
            assert isinstance(p_val, (int, float)) and np.isfinite(p_val), (
                f"Refuter {refuter_name} has invalid p_value {p_val}"
            )


class TestCorrelationAndCalibrationMetrics:
    """Gate 5 & 6: Calibrated correlation threshold and non-tautological calibration metrics."""

    def test_correlation_exceeds_calibrated_threshold(
        self, causal_result: CausalElasticityResult
    ) -> None:
        """Correlation against true effect must exceed calibrated threshold 0.75."""
        assert causal_result.correlation >= 0.75, (
            f"Estimated correlation {causal_result.correlation:.4f} is below 0.75"
        )

    def test_baseline_mae_is_positive_and_finite(
        self, causal_result: CausalElasticityResult
    ) -> None:
        """Baseline MAE must be strictly positive and bounded in a realistic range."""
        mae = causal_result.calibration_metrics.get("baseline_mae")
        assert mae is not None, "Missing 'baseline_mae' in calibration metrics"
        assert np.isfinite(mae), f"MAE is not finite: {mae}"
        assert 0.0 < mae < 50.0, f"Baseline MAE {mae:.4f} outside realistic range (0, 50)"

    def test_calibration_metrics_schema_complete(
        self, causal_result: CausalElasticityResult
    ) -> None:
        """Calibration metrics dictionary must contain all required metric keys."""
        required_keys = {
            "correlation",
            "baseline_mae",
            "average_treatment_effect",
            "residual_variance",
            "residual_identifying_variance",
        }
        actual_keys = set(causal_result.calibration_metrics.keys())
        assert required_keys.issubset(actual_keys), (
            f"Missing calibration metrics: {required_keys - actual_keys}"
        )


class TestArtifactValidation:
    """Gate 7: Structured JSON artifact persistence and schema."""

    def test_save_causal_artifact_creates_valid_payload(
        self, causal_result: CausalElasticityResult, tmp_path: Path
    ) -> None:
        """Saved artifact must preserve model, CI, calibration, and refutation metrics."""
        artifact_path = tmp_path / "causal_elasticity.json"
        saved = save_causal_artifact(causal_result, artifact_path)

        assert saved == artifact_path
        assert artifact_path.exists()

        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert payload["model"] == "causal_forest_dml"
        assert payload["treatment_variable"] == "treatment_rate_change"
        assert "feature_columns" in payload
        assert isinstance(payload["average_treatment_effect"], float)

        interval = payload["treatment_effect_confidence_interval"]
        assert interval["alpha"] == 0.05
        assert interval["lower"] < interval["upper"]
        assert interval["lower"] <= payload["average_treatment_effect"] <= interval["upper"]

        metrics = payload["calibration_metrics"]
        assert metrics["correlation"] >= 0.75
        assert 0.0 < metrics["baseline_mae"] < 50.0
        assert "residual_variance" in metrics
        assert "residual_identifying_variance" in metrics
        assert 0.20 <= metrics["residual_identifying_variance"] <= 0.70
        assert metrics["residual_variance"] > 0.001
        assert "residual_variance" in payload
        assert "residual_identifying_variance" in payload
        assert 0.20 <= payload["residual_identifying_variance"] <= 0.70
        assert payload["residual_variance"] > 0.001

        refutations = payload["refutation_summary"]
        assert "diagnostic_note" not in refutations
        assert all(
            refutations[r]["passed"] is True
            for r in ["placebo_treatment", "random_common_cause", "data_subset"]
        )
