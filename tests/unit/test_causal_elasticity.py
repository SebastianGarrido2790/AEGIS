"""Tests for synthetic treatment validation and causal elasticity estimation (Stage 4).

Quality gate suite verifying:
- Gate 1: Non-deterministic treatment assignment with bounded residual variance [0.20, 0.70].
- Gate 2: Programmatic ground truth consistency with analytic derivative
  tau(X) = 2.0 + 1.5 * risk_index.
- Gate 3: Point estimate strictly contained within its reported confidence interval.
- Gate 4: Genuine DoWhy refutation execution with no diagnostic_note fallback marker.
- Gate 5: Effect recovery correlation >= calibrated threshold at this module's sample size.
- Gate 6: Meaningful, non-tautological calibration metrics (finite MAE, complete metrics dict).
- Gate 7: Structured artifact serialization with valid schema and confidence intervals.
- Gate 8: The actual DVC-produced, MLflow-registered artifact independently clears a
  production-scale threshold — closes the gap where CI only ever validated a small,
  freshly-refit sample and never the artifact the pipeline actually ships.

IMPORTANT — read before changing any threshold in this file:
Correlation-to-ground-truth is a sample-size-sensitive statistic for this estimator, not
a fixed property. A calibration pass (2026-09) found that fit_causal_elasticity's PREVIOUS
default (max_rows=5,000, ~1,000-row test split) produced correlations ranging from -0.65 to
+0.98 across just 15 seeds, including two seeds with strongly NEGATIVE correlation — the
model appearing to recover the opposite of the true effect, purely from an undersized test
split. The default was raised to max_rows=20,000 as a result (see causal_elasticity.py's
module docstring for the full evidence trail). A full multi-seed calibration at 20,000 rows
could not be completed in the sandbox that produced this fix (memory/time constraints); only
3 clean seeds were obtained (0.7606, 0.6415, 0.4520), all positive and much tighter than the
5,000-row distribution, but NOT a distribution you should treat as final. Run
scripts/calibrate_causal_threshold.py yourself with more seeds before trusting
_PRODUCTION_ARTIFACT_CORRELATION_FLOOR below as anything more than a conservative placeholder.
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

# Sample size used by this test module's CI-fast fixture. Matches
# fit_causal_elasticity's current max_rows default (20,000) — deliberately NOT smaller,
# since the whole reason that default was raised is that smaller samples produced unstable,
# occasionally negative correlations. If CI runtime forces this back down, the threshold
# below MUST be re-calibrated at whatever size is actually used — it is not portable
# across sample sizes.
_UNIT_TEST_SAMPLE_SIZE = 20000
_UNIT_TEST_CORRELATION_THRESHOLD = 0.40

# Production-artifact threshold. Derived from 3 real observations at max_rows=20,000
# (0.7606, 0.6415, 0.4520 — see module docstring), set well below the worst of those
# three, not from a full calibrated distribution.
#
# TODO(calibration, blocking before this is treated as final): run
# `uv run python scripts/calibrate_causal_threshold.py --seeds 15` in a real (non-sandboxed)
# environment and replace this constant with a value derived from the actual observed
# distribution at production sample size, per the script's own output guidance.
_PRODUCTION_ARTIFACT_CORRELATION_FLOOR = 0.30
_PRODUCTION_ARTIFACT_PATH = Path("data/validated/causal_elasticity.json")
_PRODUCTION_MIN_EXPECTED_SAMPLE_SIZE = 15000


@pytest.fixture(scope="module")
def feature_frame() -> pd.DataFrame:
    """Use the real Stage 2 feature matrix for causal validation tests."""
    return pd.read_csv("data/versioned/feature_matrix.csv")


@pytest.fixture(scope="module")
def causal_result(feature_frame: pd.DataFrame) -> CausalElasticityResult:
    """Fit CausalForestDML once for module-level validation assertions."""
    sample = feature_frame.sample(_UNIT_TEST_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    return fit_causal_elasticity(sample, random_state=42)


class TestWeakTreatmentIdentification:
    """Gate 1: Independent stochastic variation in synthetic treatment (Fix #1)."""

    def test_residual_variance_fraction_bounded(self, feature_frame: pd.DataFrame) -> None:
        """Residual identifying variance in [0.20, 0.70] and raw variance > 0."""
        sample = feature_frame.sample(
            _UNIT_TEST_SAMPLE_SIZE, random_state=42
        ).reset_index(drop=True)
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
        sample = feature_frame.sample(
            _UNIT_TEST_SAMPLE_SIZE, random_state=42
        ).reset_index(drop=True)
        prepared = add_synthetic_treatment(sample, random_state=42)
        systematic = 0.05 + 0.10 * prepared["risk_index"]
        noise_residual = prepared["treatment_rate_change"] - systematic
        assert float(noise_residual.std()) > 0.02, (
            f"Noise residual std {float(noise_residual.std()):.4f} is too low (<= 0.02)"
        )

    def test_treatment_positivity_and_variation(self, feature_frame: pd.DataFrame) -> None:
        """Treatment rate change must be strictly positive and exhibit non-trivial spread."""
        sample = feature_frame.sample(
            _UNIT_TEST_SAMPLE_SIZE, random_state=42
        ).reset_index(drop=True)
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
    """Gate 5 & 6: Calibrated correlation threshold and non-tautological calibration metrics.

    Threshold is specific to `_UNIT_TEST_SAMPLE_SIZE` — see TestRegisteredArtifactRegression
    for the production-scale check, and this module's docstring for calibration status.
    """

    def test_correlation_exceeds_calibrated_threshold(
        self, causal_result: CausalElasticityResult
    ) -> None:
        """Correlation against true effect must exceed the sample-size-calibrated threshold."""
        assert causal_result.correlation >= _UNIT_TEST_CORRELATION_THRESHOLD, (
            f"Estimated correlation {causal_result.correlation:.4f} is below "
            f"{_UNIT_TEST_CORRELATION_THRESHOLD} at sample size "
            f"{causal_result.test_sample_size}"
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
    """Gate 7: Structured JSON artifact persistence and schema (fresh in-test fit)."""

    def test_save_causal_artifact_creates_valid_payload(
        self, causal_result: CausalElasticityResult, tmp_path: Path
    ) -> None:
        """Saved artifact must preserve model, CI, calibration, refutation, and sample-size metadata."""
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
        assert metrics["correlation"] >= _UNIT_TEST_CORRELATION_THRESHOLD
        assert 0.0 < metrics["baseline_mae"] < 50.0
        assert "residual_variance" in metrics
        assert "residual_identifying_variance" in metrics
        assert 0.20 <= metrics["residual_identifying_variance"] <= 0.70
        assert metrics["residual_variance"] > 0.001

        assert payload["total_sample_size"] == causal_result.total_sample_size
        assert payload["test_sample_size"] == causal_result.test_sample_size
        assert payload["test_sample_size"] <= payload["total_sample_size"]

        refutations = payload["refutation_summary"]
        assert "diagnostic_note" not in refutations
        assert all(
            refutations[r]["passed"] is True
            for r in ["placebo_treatment", "random_common_cause", "data_subset"]
        )


class TestRegisteredArtifactRegression:
    """Gate 8: the actual DVC-produced, MLflow-registered artifact — not a fresh in-test fit.

    Every other test in this module fits CausalForestDML fresh inside the test run itself.
    None of them ever load or check the file that `cli.py`'s `train-causal` command actually
    produces, registers to MLflow, and that the evaluation report's numbers come from — which
    meant CI could stay green indefinitely regardless of what the real registered model's
    correlation actually was. This class closes that gap directly.

    NOTE: because max_rows changed from 5,000 to 20,000 in this same round of fixes, the
    EXISTING data/validated/causal_elasticity.json on disk (if any) was almost certainly
    produced under the OLD default. Re-run `dvc repro` (or the `train-causal` stage directly)
    under the corrected code before trusting this test's result — a stale artifact from the
    old default will not reflect the fix and should not be treated as a pass or fail of it.
    """

    @pytest.fixture(scope="class")
    def registered_artifact(self) -> dict:
        if not _PRODUCTION_ARTIFACT_PATH.is_file():
            pytest.skip(
                f"{_PRODUCTION_ARTIFACT_PATH} not present — run the `train-causal` "
                "DVC stage (or `dvc repro`) before this regression check can run."
            )
        return json.loads(_PRODUCTION_ARTIFACT_PATH.read_text(encoding="utf-8"))

    def test_artifact_has_sample_size_provenance(self, registered_artifact: dict) -> None:
        """The real artifact must record what sample size actually produced it."""
        assert "total_sample_size" in registered_artifact, (
            "Registered artifact is missing total_sample_size — cannot verify which "
            "sample size this correlation figure corresponds to."
        )
        assert "test_sample_size" in registered_artifact
        assert registered_artifact["total_sample_size"] >= _PRODUCTION_MIN_EXPECTED_SAMPLE_SIZE, (
            f"Registered artifact total_sample_size="
            f"{registered_artifact['total_sample_size']} is below the expected production "
            f"floor of {_PRODUCTION_MIN_EXPECTED_SAMPLE_SIZE} — verify train-causal was not "
            "accidentally run against a stale build or an overridden max_rows."
        )

    def test_registered_artifact_correlation_meets_production_floor(
        self, registered_artifact: dict
    ) -> None:
        """The real, registered artifact must independently clear a production-scale floor.

        See module docstring: this floor is a conservative placeholder from 3 observations,
        not a finished calibration.
        """
        correlation = registered_artifact["calibration_metrics"]["correlation"]
        sample_size = registered_artifact.get("total_sample_size", "unknown")
        assert correlation >= _PRODUCTION_ARTIFACT_CORRELATION_FLOOR, (
            f"Registered artifact correlation {correlation:.4f} "
            f"(sample_size={sample_size}) is below the production floor "
            f"{_PRODUCTION_ARTIFACT_CORRELATION_FLOOR} — this is the number the "
            "pipeline actually ships and registers to MLflow, not a unit-test sample."
        )

    def test_registered_artifact_point_estimate_contained_in_ci(
        self, registered_artifact: dict
    ) -> None:
        """The real artifact's point estimate must fall within its own reported interval."""
        ate = registered_artifact["average_treatment_effect"]
        interval = registered_artifact["treatment_effect_confidence_interval"]
        assert interval["lower"] <= ate <= interval["upper"], (
            f"Registered artifact point estimate {ate:.4f} outside its own reported "
            f"interval [{interval['lower']:.4f}, {interval['upper']:.4f}]"
        )

    def test_registered_artifact_refutations_are_live(self, registered_artifact: dict) -> None:
        """The real artifact's refutation summary must show no fallback marker."""
        refutations = registered_artifact["refutation_summary"]
        assert "diagnostic_note" not in refutations, (
            "Registered artifact's refutation_summary contains a fallback marker — "
            "DoWhy did not execute live for the production run."
        )