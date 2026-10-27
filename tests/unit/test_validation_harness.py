"""Unit tests for validation schemas, metrics, controls, and calibration (Stage 9, P2-F07)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aegis.schemas.validation import (
    SegmentValidationRow,
    ValidationMetrics,
    ValidationThresholds,
)
from aegis.utils.exceptions import ValidationHarnessError


def _valid_metrics_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "m1_tau_spearman": 0.90,
        "m1_epsilon_spearman": 0.90,
        "m1b_policy_spearman": 0.85,
        "m2_mean_abs_bias": 0.05,
        "m3_tau_coverage": 0.95,
        "m3_epsilon_coverage": 0.95,
        "m4_self_consistency": 1.0,
        "m5_sign_agreement": 1.0,
    }
    payload.update(overrides)
    return payload


def test_validation_metrics_valid() -> None:
    metrics = ValidationMetrics.model_validate(_valid_metrics_payload())
    assert metrics.m1_tau_spearman == 0.90
    assert metrics.m2_mean_abs_bias == 0.05
    assert metrics.m4_self_consistency == 1.0


def test_validation_metrics_rejects_non_finite() -> None:
    with pytest.raises(ValidationError):
        ValidationMetrics.model_validate(_valid_metrics_payload(m1_tau_spearman=float("nan")))
    with pytest.raises(ValidationError):
        ValidationMetrics.model_validate(_valid_metrics_payload(m2_mean_abs_bias=float("inf")))


def test_validation_metrics_rejects_out_of_bounds() -> None:
    with pytest.raises(ValidationError):
        ValidationMetrics.model_validate(_valid_metrics_payload(m1_tau_spearman=1.5))
    with pytest.raises(ValidationError):
        ValidationMetrics.model_validate(_valid_metrics_payload(m3_tau_coverage=1.1))
    with pytest.raises(ValidationError):
        ValidationMetrics.model_validate(_valid_metrics_payload(m2_mean_abs_bias=-0.01))


def test_segment_validation_row_valid() -> None:
    row = SegmentValidationRow(
        segment_id="D:18-24_C:0-4_Dens:501-2000",
        policy_count=213,
        tau_hat=-0.58,
        tau_true=-0.56,
        tau_ci_lower=-0.70,
        tau_ci_upper=-0.45,
        tau_covered=True,
        epsilon_hat=-0.70,
        epsilon_true=-0.68,
        epsilon_ci_lower=-0.85,
        epsilon_ci_upper=-0.55,
        epsilon_covered=True,
        baseline_q0_hat=0.83,
        baseline_q0_true=0.82,
        sign_agreed=True,
        abs_bias=0.02,
    )
    assert row.tau_covered is True
    assert row.abs_bias == 0.02


def test_validation_thresholds_ordering() -> None:
    valid_thresholds = ValidationThresholds(
        m1_tau_spearman_floor=0.70,
        m1_tau_spearman_target=0.85,
        m1_epsilon_spearman_floor=0.70,
        m1_epsilon_spearman_target=0.85,
        m1b_policy_spearman_floor=0.60,
        m1b_policy_spearman_target=0.75,
        m2_mean_abs_bias_ceiling=0.20,
        m2_mean_abs_bias_target=0.10,
        m3_tau_coverage_floor=0.90,
        m3_epsilon_coverage_floor=0.90,
        m4_self_consistency_floor=1.0,
        m5_sign_agreement_floor=0.95,
    )
    assert valid_thresholds.m1_tau_spearman_floor <= valid_thresholds.m1_tau_spearman_target

    with pytest.raises(ValidationError):
        ValidationThresholds(
            m1_tau_spearman_floor=0.90,
            m1_tau_spearman_target=0.70,  # Floor exceeds target
            m1_epsilon_spearman_floor=0.70,
            m1_epsilon_spearman_target=0.85,
            m1b_policy_spearman_floor=0.60,
            m1b_policy_spearman_target=0.75,
            m2_mean_abs_bias_ceiling=0.20,
            m2_mean_abs_bias_target=0.10,
            m3_tau_coverage_floor=0.90,
            m3_epsilon_coverage_floor=0.90,
            m4_self_consistency_floor=1.0,
            m5_sign_agreement_floor=0.95,
        )


def test_validation_harness_error() -> None:
    with pytest.raises(ValidationHarnessError) as exc_info:
        raise ValidationHarnessError("Control failed", details={"control": "c1"})
    assert "Control failed" in str(exc_info.value)
    assert exc_info.value.details["control"] == "c1"


def test_compute_spearman_corr_basic() -> None:
    from aegis.pipelines.validation_harness import compute_spearman_corr

    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert compute_spearman_corr(a, b) == pytest.approx(1.0)

    c = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert compute_spearman_corr(a, c) == pytest.approx(-1.0)

    # Constant variance returns 0.0
    const = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert compute_spearman_corr(a, const) == 0.0

    with pytest.raises(ValidationHarnessError):
        compute_spearman_corr([1.0, 2.0], [1.0])


def test_compute_validation_metrics_synthetic() -> None:
    import numpy as np
    import pandas as pd

    from aegis.pipelines.validation_harness import compute_validation_metrics

    # Create synthetic test policies and truth sidecar
    seg_ids = ["D:18-24_C:0-4_Dens:501-2000", "D:25-44_C:5-9_Dens:501-2000"]
    n_per_seg = 50
    policy_ids = [f"POL_{i:04d}" for i in range(n_per_seg * 2)]
    segments = [seg_ids[0]] * n_per_seg + [seg_ids[1]] * n_per_seg

    held_out_test = pd.DataFrame({"PolicyID": policy_ids, "segment_id": segments})

    # Truth: seg 0 has tau=-0.5, q0=0.8; seg 1 has tau=-0.2, q0=0.8
    true_tau = [-0.5] * n_per_seg + [-0.2] * n_per_seg
    true_q0 = [0.8] * (n_per_seg * 2)
    true_eps = [t / q for t, q in zip(true_tau, true_q0, strict=True)]

    truth_df = pd.DataFrame(
        {
            "PolicyID": policy_ids,
            "segment_id": segments,
            "true_tau": true_tau,
            "baseline_renewal_q0": true_q0,
            "true_epsilon": true_eps,
        }
    )

    # Segment estimates: accurate estimates with covering CIs
    est_df = pd.DataFrame(
        {
            "segment_id": seg_ids,
            "inference_policy_count": [n_per_seg, n_per_seg],
            "baseline_retention_q0": [0.80, 0.80],
            "tau_hat": [-0.51, -0.19],
            "tau_ci_lower": [-0.60, -0.30],
            "tau_ci_upper": [-0.40, -0.10],
            "epsilon_hat": [-0.51 / 0.80, -0.19 / 0.80],
            "epsilon_ci_lower": [-0.60 / 0.80, -0.30 / 0.80],
            "epsilon_ci_upper": [-0.40 / 0.80, -0.10 / 0.80],
        }
    )

    report = compute_validation_metrics(
        segment_estimates=est_df,
        truth_sidecar=truth_df,
        held_out_test=held_out_test,
        policy_effects=np.array(true_tau),
        seed=42,
    )

    assert report.seed == 42
    assert report.sample_size == 100
    assert report.metrics.m1_tau_spearman == pytest.approx(1.0)
    assert report.metrics.m1_epsilon_spearman == pytest.approx(1.0)
    assert report.metrics.m1b_policy_spearman == pytest.approx(1.0)
    assert report.metrics.m3_tau_coverage == 1.0
    assert report.metrics.m3_epsilon_coverage == 1.0
    assert report.metrics.m4_self_consistency == 1.0
    assert report.metrics.m5_sign_agreement == 1.0
    assert report.metrics.m2_mean_abs_bias < 0.02
    assert len(report.segment_details) == 2


def test_unadjusted_control_c1() -> None:
    import numpy as np
    import pandas as pd

    from aegis.pipelines.validation_harness import run_unadjusted_control_c1

    # In confounded data, higher treatment goes to lower renewal (confounded)
    seg_ids = ["D:18-24_C:0-4_Dens:501-2000", "D:25-44_C:5-9_Dens:501-2000"]
    n_per_seg = 100
    policy_ids = [f"POL_{i:04d}" for i in range(n_per_seg * 2)]
    segments = [seg_ids[0]] * n_per_seg + [seg_ids[1]] * n_per_seg

    # Confounded treatment: T in [-0.15, 0.15]
    rng = np.random.default_rng(42)
    t = rng.uniform(-0.15, 0.15, size=n_per_seg * 2)
    # y confounded with high noise
    y = (rng.uniform(0.0, 1.0, size=n_per_seg * 2) < 0.5).astype(int)

    held_out_test = pd.DataFrame(
        {
            "PolicyID": policy_ids,
            "segment_id": segments,
            "treatment_rate_change": t,
            "renewed": y,
        }
    )

    truth_df = pd.DataFrame(
        {
            "PolicyID": policy_ids,
            "segment_id": segments,
            "true_tau": [-0.5] * n_per_seg + [-0.1] * n_per_seg,
            "baseline_renewal_q0": [0.8] * (n_per_seg * 2),
            "true_epsilon": [-0.625] * n_per_seg + [-0.125] * n_per_seg,
        }
    )

    res = run_unadjusted_control_c1(held_out_test, truth_df)
    assert res.control_id == "c1_unadjusted"
    assert res.is_negative_control is True
    assert res.failed_gate_as_expected is True


def test_shuffled_control_c2() -> None:
    import pandas as pd

    from aegis.pipelines.validation_harness import run_shuffled_control_c2

    # 4 segments
    seg_ids = [f"D:{i}_C:0-4_Dens:501-2000" for i in range(4)]
    policy_ids = [f"POL_{i:04d}" for i in range(40)]
    segments = [seg_ids[i // 10] for i in range(40)]

    held_out_test = pd.DataFrame({"PolicyID": policy_ids, "segment_id": segments})
    truth_df = pd.DataFrame(
        {
            "PolicyID": policy_ids,
            "segment_id": segments,
            "true_tau": [-0.8, -0.6, -0.4, -0.2] * 10,
            "baseline_renewal_q0": [0.8] * 40,
            "true_epsilon": [-1.0, -0.75, -0.5, -0.25] * 10,
        }
    )

    est_df = pd.DataFrame(
        {
            "segment_id": seg_ids,
            "inference_policy_count": [10] * 4,
            "baseline_retention_q0": [0.8] * 4,
            "tau_hat": [-0.8, -0.6, -0.4, -0.2],
            "tau_ci_lower": [-0.9, -0.7, -0.5, -0.3],
            "tau_ci_upper": [-0.7, -0.5, -0.3, -0.1],
            "epsilon_hat": [-1.0, -0.75, -0.5, -0.25],
            "epsilon_ci_lower": [-1.125, -0.875, -0.625, -0.375],
            "epsilon_ci_upper": [-0.875, -0.625, -0.375, -0.125],
        }
    )

    res = run_shuffled_control_c2(est_df, truth_df, held_out_test, random_state=123)
    assert res.control_id == "c2_shuffled"
    assert res.is_negative_control is True
    assert res.failed_gate_as_expected is True


def test_zero_effect_control_c3() -> None:
    import pandas as pd

    from aegis.pipelines.validation_harness import run_zero_effect_control_c3

    seg_ids = ["D:18-24_C:0-4_Dens:501-2000", "D:25-44_C:5-9_Dens:501-2000"]
    policy_ids = [f"POL_{i:04d}" for i in range(20)]
    segments = [seg_ids[0]] * 10 + [seg_ids[1]] * 10

    held_out_test = pd.DataFrame({"PolicyID": policy_ids, "segment_id": segments})
    truth_df = pd.DataFrame(
        {
            "PolicyID": policy_ids,
            "segment_id": segments,
            "true_tau": [-0.5] * 10 + [-0.1] * 10,
            "baseline_renewal_q0": [0.8] * 20,
            "true_epsilon": [-0.625] * 10 + [-0.125] * 10,
        }
    )

    # Zero-effect estimates (centered near 0 with wide intervals covering 0)
    zero_df = pd.DataFrame(
        {
            "segment_id": seg_ids,
            "inference_policy_count": [10, 10],
            "baseline_retention_q0": [0.8, 0.8],
            "tau_hat": [0.005, -0.003],
            "tau_ci_lower": [-0.15, -0.15],
            "tau_ci_upper": [0.15, 0.15],
            "epsilon_hat": [0.005 / 0.8, -0.003 / 0.8],
            "epsilon_ci_lower": [-0.15 / 0.8, -0.15 / 0.8],
            "epsilon_ci_upper": [0.15 / 0.8, 0.15 / 0.8],
        }
    )

    res = run_zero_effect_control_c3(zero_df, truth_df, held_out_test)
    assert res.control_id == "c3_zero_effect"
    assert res.is_negative_control is True
    assert res.failed_gate_as_expected is True


def test_positive_control() -> None:
    import pandas as pd

    from aegis.pipelines.validation_harness import run_positive_control

    seg_ids = [f"D:{i}_C:0-4_Dens:501-2000" for i in range(4)]
    policy_ids = [f"POL_{i:04d}" for i in range(40)]
    segments = [seg_ids[i // 10] for i in range(40)]

    held_out_test = pd.DataFrame({"PolicyID": policy_ids, "segment_id": segments})
    truth_df = pd.DataFrame(
        {
            "PolicyID": policy_ids,
            "segment_id": segments,
            "true_tau": [-0.8] * 10 + [-0.6] * 10 + [-0.4] * 10 + [-0.2] * 10,
            "baseline_renewal_q0": [0.8] * 40,
            "true_epsilon": [-1.0] * 10 + [-0.75] * 10 + [-0.5] * 10 + [-0.25] * 10,
        }
    )

    # Accurate unconfounded estimates
    est_df = pd.DataFrame(
        {
            "segment_id": seg_ids,
            "inference_policy_count": [10] * 4,
            "baseline_retention_q0": [0.8] * 4,
            "tau_hat": [-0.79, -0.61, -0.39, -0.21],
            "tau_ci_lower": [-0.90, -0.72, -0.50, -0.32],
            "tau_ci_upper": [-0.68, -0.50, -0.28, -0.10],
            "epsilon_hat": [-0.79 / 0.8, -0.61 / 0.8, -0.39 / 0.8, -0.21 / 0.8],
            "epsilon_ci_lower": [-0.90 / 0.8, -0.72 / 0.8, -0.50 / 0.8, -0.32 / 0.8],
            "epsilon_ci_upper": [-0.68 / 0.8, -0.50 / 0.8, -0.28 / 0.8, -0.10 / 0.8],
        }
    )

    res = run_positive_control(est_df, truth_df, held_out_test)
    assert res.control_id == "positive_control"
    assert res.is_negative_control is False
    assert res.failed_gate_as_expected is False  # Passed cleanly!


def test_compute_separation_gap_higher_is_better() -> None:
    from aegis.pipelines.validation_harness import compute_separation_gap

    e_vals = [0.88, 0.90, 0.92]
    c_vals = [0.10, 0.20, 0.25]
    gap = compute_separation_gap("m1", "higher_is_better", e_vals, c_vals, min_margin=0.15)
    assert gap.separated is True
    assert gap.estimator_min == 0.88
    assert gap.control_max == 0.25
    assert gap.gap == pytest.approx(0.63)
    assert gap.recommended_threshold == pytest.approx(0.25 + 0.63 / 2.0, abs=1e-4)


def test_compute_separation_gap_lower_is_better() -> None:
    from aegis.pipelines.validation_harness import compute_separation_gap

    e_vals = [0.04, 0.05, 0.06]
    c_vals = [0.25, 0.30, 0.35]
    gap = compute_separation_gap("m2", "lower_is_better", e_vals, c_vals, min_margin=0.05)
    assert gap.separated is True
    assert gap.estimator_min == 0.06
    assert gap.control_max == 0.25
    assert gap.gap == pytest.approx(0.19)
    assert gap.recommended_threshold == pytest.approx(0.25 - 0.19 / 2.0, abs=1e-4)


def test_compute_separation_gap_unseparated() -> None:
    from aegis.pipelines.validation_harness import compute_separation_gap

    e_vals = [0.30, 0.35]
    c_vals = [0.25, 0.28]
    gap = compute_separation_gap("m1", "higher_is_better", e_vals, c_vals, min_margin=0.15)
    assert gap.separated is False


def test_build_calibrated_thresholds() -> None:
    from aegis.pipelines.validation_harness import build_calibrated_thresholds
    from aegis.schemas.validation import ControlRunResult, ValidationMetricReport

    est_runs = [
        ValidationMetricReport(
            seed=42,
            sample_size=1000,
            metrics=ValidationMetrics.model_validate(
                _valid_metrics_payload(m1_tau_spearman=0.90, m2_mean_abs_bias=0.05)
            ),
        ),
        ValidationMetricReport(
            seed=101,
            sample_size=1000,
            metrics=ValidationMetrics.model_validate(
                _valid_metrics_payload(m1_tau_spearman=0.88, m2_mean_abs_bias=0.06)
            ),
        ),
    ]

    neg_controls = [
        ControlRunResult(
            control_id="c1_unadjusted",
            control_name="Unadjusted OLS",
            is_negative_control=True,
            metrics=ValidationMetrics.model_validate(
                _valid_metrics_payload(
                    m1_tau_spearman=0.20,
                    m1_epsilon_spearman=0.20,
                    m1b_policy_spearman=0.15,
                    m2_mean_abs_bias=0.35,
                    m5_sign_agreement=0.50,
                )
            ),
            failed_gate_as_expected=True,
            diagnostic_message="failed",
        ),
        ControlRunResult(
            control_id="c2_shuffled",
            control_name="Shuffled",
            is_negative_control=True,
            metrics=ValidationMetrics.model_validate(
                _valid_metrics_payload(
                    m1_tau_spearman=0.05,
                    m1_epsilon_spearman=0.05,
                    m1b_policy_spearman=0.02,
                    m2_mean_abs_bias=0.40,
                    m5_sign_agreement=1.0,
                )
            ),
            failed_gate_as_expected=True,
            diagnostic_message="failed",
        ),
    ]

    gaps, thresholds = build_calibrated_thresholds(est_runs, neg_controls)
    assert all(g.separated for g in gaps.values())
    assert "m5_sign_agreement" not in gaps
    assert thresholds.m5_sign_agreement_floor == 0.90
    assert thresholds.m1_tau_spearman_floor > 0.20
    assert thresholds.m1_tau_spearman_target >= thresholds.m1_tau_spearman_floor
    assert thresholds.m2_mean_abs_bias_ceiling < 0.35


def test_format_q3_calibration_report() -> None:
    from aegis.pipelines.validation_harness import format_q3_calibration_report
    from aegis.schemas.validation import (
        ControlRunResult,
        Q3CalibrationReport,
        SeparationGap,
        ValidationMetricReport,
    )

    metrics = ValidationMetrics.model_validate(_valid_metrics_payload())
    est = ValidationMetricReport(seed=42, sample_size=1000, metrics=metrics)
    neg = ControlRunResult(
        control_id="c1_unadjusted",
        control_name="Unadjusted OLS",
        is_negative_control=True,
        metrics=ValidationMetrics.model_validate(
            _valid_metrics_payload(m1_tau_spearman=0.20, m5_sign_agreement=0.50)
        ),
        failed_gate_as_expected=True,
        diagnostic_message="failed",
    )
    pos = ControlRunResult(
        control_id="positive_control",
        control_name="Randomized",
        is_negative_control=False,
        metrics=metrics,
        failed_gate_as_expected=False,
        diagnostic_message="passed",
    )
    gap = SeparationGap(
        metric_name="m1_tau_spearman",
        direction="higher_is_better",
        estimator_min=0.90,
        control_max=0.20,
        gap=0.70,
        min_margin=0.15,
        separated=True,
        recommended_threshold=0.55,
    )
    thresholds = ValidationThresholds(
        m1_tau_spearman_floor=0.55,
        m1_tau_spearman_target=0.85,
        m1_epsilon_spearman_floor=0.55,
        m1_epsilon_spearman_target=0.85,
        m1b_policy_spearman_floor=0.55,
        m1b_policy_spearman_target=0.85,
        m2_mean_abs_bias_ceiling=0.15,
        m2_mean_abs_bias_target=0.10,
        m3_tau_coverage_floor=0.90,
        m3_epsilon_coverage_floor=0.90,
        m4_self_consistency_floor=1.0,
        m5_sign_agreement_floor=0.90,
    )
    report = Q3CalibrationReport(
        pilot_seeds=[42],
        estimator_runs=[est],
        negative_control_results=[neg],
        positive_control_result=pos,
        separation_gaps={"m1_tau_spearman": gap},
        recommended_thresholds=thresholds,
        all_negative_controls_failed=True,
        positive_control_passed=True,
        clear_separation_achieved=True,
    )

    text = format_q3_calibration_report(report, "  validation_thresholds:\n    test: 1")
    assert "# Qualification Q3" in text
    assert "PASSED" in text
    assert "m1_tau_spearman" in text
    assert "excluded" in text
    assert "```yaml" in text
    assert "test: 1" in text


