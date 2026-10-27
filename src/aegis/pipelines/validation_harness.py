"""Offline validation harness and threshold calibration engine (P2-F07, Stage 9).

Under ADR-013, this module is the authorized offline validation consumer of the truth
sidecar. It calculates validation metrics m1 through m5, executes negative and positive
controls, and calibrates numeric gate thresholds on pilot seeds without peeking at final
evaluation seeds. Truth values never enter causal estimators or production pipelines.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis.config.loader import get_config
from aegis.config.schema import AEGISConfig
from aegis.pipelines.causal_estimator import (
    aggregate_segment_estimates,
    fit_causal_estimator,
    stratified_sample_split,
    validate_training_contract,
)
from aegis.pipelines.synthetic_dgp import generate_semi_synthetic_dataset
from aegis.pipelines.truth_loader import REQUIRED_TRUTH_COLUMNS, load_truth_sidecar
from aegis.schemas.causal import SEGMENT_ESTIMATE_COLUMNS, SegmentElasticityTable
from aegis.schemas.validation import (
    ControlRunResult,
    Q3CalibrationReport,
    SegmentValidationRow,
    SeparationGap,
    ValidationMetricReport,
    ValidationMetrics,
    ValidationThresholds,
)
from aegis.utils.exceptions import ValidationHarnessError

M5_SIGN_AGREEMENT_FLOOR = 0.90


def compute_spearman_corr(
    a: Sequence[float] | np.ndarray | pd.Series,
    b: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Computes Spearman rank correlation, handling zero-variance gracefully.

    Args:
        a: First sequence of numeric values.
        b: Second sequence of numeric values.

    Returns:
        float: Spearman rank correlation in [-1.0, 1.0], or 0.0 if variance is zero.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if len(arr_a) != len(arr_b):
        raise ValidationHarnessError(
            f"Spearman correlation length mismatch: {len(arr_a)} != {len(arr_b)}"
        )
    if len(arr_a) < 2 or float(np.std(arr_a)) < 1e-12 or float(np.std(arr_b)) < 1e-12:
        return 0.0

    series_a = pd.Series(arr_a)
    series_b = pd.Series(arr_b)
    corr = float(series_a.corr(series_b, method="spearman"))
    if not math.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _extract_segment_df(
    segment_estimates: pd.DataFrame | SegmentElasticityTable,
) -> pd.DataFrame:
    """Normalizes segment estimates into a DataFrame and validates required columns."""
    if isinstance(segment_estimates, SegmentElasticityTable):
        df = pd.DataFrame([seg.model_dump() for seg in segment_estimates.segments])
    elif isinstance(segment_estimates, pd.DataFrame):
        df = segment_estimates.copy()
    else:
        raise ValidationHarnessError(
            f"Expected SegmentElasticityTable or DataFrame; got {type(segment_estimates)}"
        )

    missing = [c for c in SEGMENT_ESTIMATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationHarnessError(
            f"Segment estimates missing required columns: {missing}",
            details={"missing_columns": missing},
        )
    return df.sort_values("segment_id").reset_index(drop=True)


def compute_validation_metrics(
    segment_estimates: pd.DataFrame | SegmentElasticityTable,
    truth_sidecar: pd.DataFrame | str | None,
    held_out_test: pd.DataFrame,
    policy_effects: np.ndarray | None = None,
    seed: int | None = None,
) -> ValidationMetricReport:
    """Computes validation metrics m1 through m5 against the ground truth sidecar.

    Args:
        segment_estimates: Estimated segment table from Stage 8 inference.
        truth_sidecar: Ground truth sidecar DataFrame or path (if None, loaded from config).
        held_out_test: Held-out test policies containing PolicyID and segment_id.
        policy_effects: Optional 1D array of predicted policy effects tau_hat for held-out data.
        seed: Random seed of the evaluated run (for metadata reporting).

    Returns:
        ValidationMetricReport: Complete metric report and per-segment diagnostics.

    Raises:
        ValidationHarnessError: If inputs are mismatched, empty, or have invalid formats.
    """
    est_df = _extract_segment_df(segment_estimates)
    if est_df.empty:
        raise ValidationHarnessError("Segment estimates table is empty")

    if isinstance(truth_sidecar, (str, type(None))):
        truth_df = load_truth_sidecar(truth_sidecar)
    elif isinstance(truth_sidecar, pd.DataFrame):
        truth_df = truth_sidecar
    else:
        raise ValidationHarnessError(f"Invalid truth_sidecar type: {type(truth_sidecar)}")

    missing_truth = [c for c in REQUIRED_TRUTH_COLUMNS if c not in truth_df.columns]
    if missing_truth:
        raise ValidationHarnessError(f"Truth sidecar missing columns: {missing_truth}")

    if "PolicyID" not in held_out_test.columns or "segment_id" not in held_out_test.columns:
        raise ValidationHarnessError("held_out_test must contain 'PolicyID' and 'segment_id'")

    # Align held-out test policies with truth sidecar
    test_truth = pd.merge(
        held_out_test[["PolicyID", "segment_id"]],
        truth_df[["PolicyID", "baseline_renewal_q0", "true_tau", "true_epsilon"]],
        on="PolicyID",
        how="inner",
    )
    if len(test_truth) != len(held_out_test):
        raise ValidationHarnessError(
            f"Policy ID mismatch: {len(test_truth)} matched out of "
            f"{len(held_out_test)} test policies",
            details={"matched": len(test_truth), "expected": len(held_out_test)},
        )

    # Calculate truth at segment level
    seg_truth_rows: list[dict[str, Any]] = []
    for seg_id, group in test_truth.groupby("segment_id"):
        sum_tau = float(group["true_tau"].sum())
        sum_q0 = float(group["baseline_renewal_q0"].sum())
        mean_tau = float(group["true_tau"].mean())
        mean_q0 = float(group["baseline_renewal_q0"].mean())
        # Ratio of sums for elasticity (ADR-013, ADR-014, Stage 4/Stage 8)
        true_eps = sum_tau / sum_q0 if sum_q0 > 1e-12 else 0.0

        seg_truth_rows.append(
            {
                "segment_id": str(seg_id),
                "true_tau_seg": mean_tau,
                "true_epsilon_seg": true_eps,
                "true_q0_seg": mean_q0,
                "policy_count": len(group),
            }
        )
    truth_seg_df = pd.DataFrame(seg_truth_rows).sort_values("segment_id").reset_index(drop=True)

    # Merge estimates and truth at segment level
    merged = pd.merge(est_df, truth_seg_df, on="segment_id", how="inner")
    if len(merged) != len(est_df):
        raise ValidationHarnessError(
            f"Segment coverage mismatch between estimates ({len(est_df)}) and truth ({len(merged)})"
        )

    # Compute per-segment diagnostics
    segment_details: list[SegmentValidationRow] = []
    for _, row in merged.iterrows():
        t_hat = float(row["tau_hat"])
        t_true = float(row["true_tau_seg"])
        t_lower = float(row["tau_ci_lower"])
        t_upper = float(row["tau_ci_upper"])
        t_covered = bool(t_lower <= t_true <= t_upper)

        e_hat = float(row["epsilon_hat"])
        e_true = float(row["true_epsilon_seg"])
        e_lower = float(row["epsilon_ci_lower"])
        e_upper = float(row["epsilon_ci_upper"])
        e_covered = bool(e_lower <= e_true <= e_upper)

        q0_hat = float(row["baseline_retention_q0"])
        q0_true = float(row["true_q0_seg"])

        sign_agreed = bool(
            (t_hat >= 0 and t_true >= 0) or (t_hat < 0 and t_true < 0)
        )
        abs_bias = abs(e_hat - e_true)

        segment_details.append(
            SegmentValidationRow(
                segment_id=str(row["segment_id"]),
                policy_count=int(row["inference_policy_count"]),
                tau_hat=t_hat,
                tau_true=t_true,
                tau_ci_lower=t_lower,
                tau_ci_upper=t_upper,
                tau_covered=t_covered,
                epsilon_hat=e_hat,
                epsilon_true=e_true,
                epsilon_ci_lower=e_lower,
                epsilon_ci_upper=e_upper,
                epsilon_covered=e_covered,
                baseline_q0_hat=q0_hat,
                baseline_q0_true=q0_true,
                sign_agreed=sign_agreed,
                abs_bias=abs_bias,
            )
        )

    # Metric m1: Segment-level tau Spearman correlation
    m1_tau = compute_spearman_corr(
        merged["tau_hat"].to_numpy(dtype=float),
        merged["true_tau_seg"].to_numpy(dtype=float),
    )

    # Metric m1_eps: Segment-level epsilon Spearman correlation
    m1_eps = compute_spearman_corr(
        merged["epsilon_hat"].to_numpy(dtype=float),
        merged["true_epsilon_seg"].to_numpy(dtype=float),
    )

    # Metric m1b: Policy-level tau Spearman correlation
    if policy_effects is not None:
        effects_arr = np.asarray(policy_effects, dtype=float).reshape(-1)
        if len(effects_arr) != len(test_truth):
            raise ValidationHarnessError(
                f"policy_effects length ({len(effects_arr)}) does not match "
                f"test policies ({len(test_truth)})"
            )
        m1b = compute_spearman_corr(
            effects_arr,
            test_truth["true_tau"].to_numpy(dtype=float),
        )
    else:
        # Broadcast segment tau_hat to each policy
        seg_map = dict(
            zip(
                merged["segment_id"].astype(str),
                merged["tau_hat"].astype(float),
                strict=True,
            )
        )
        broadcast_tau = np.array(
            [seg_map[str(sid)] for sid in test_truth["segment_id"]],
            dtype=float,
        )
        m1b = compute_spearman_corr(
            broadcast_tau,
            test_truth["true_tau"].to_numpy(dtype=float),
        )

    # Metric m2: Mean absolute elasticity bias across segments
    m2_bias = float(np.mean([r.abs_bias for r in segment_details]))

    # Metric m3: 95% interval empirical coverage
    m3_tau = float(np.mean([1.0 if r.tau_covered else 0.0 for r in segment_details]))
    m3_eps = float(np.mean([1.0 if r.epsilon_covered else 0.0 for r in segment_details]))

    # Metric m4: Interval self-consistency rate
    consistent_count = sum(
        1
        for _, r in merged.iterrows()
        if (
            math.isfinite(r["tau_ci_lower"])
            and math.isfinite(r["tau_ci_upper"])
            and r["tau_ci_lower"] <= r["tau_hat"] <= r["tau_ci_upper"]
            and math.isfinite(r["epsilon_ci_lower"])
            and math.isfinite(r["epsilon_ci_upper"])
            and r["epsilon_ci_lower"] <= r["epsilon_hat"] <= r["epsilon_ci_upper"]
        )
    )
    m4_self_cons = float(consistent_count / len(merged))

    # Metric m5: Directional sign agreement rate
    m5_sign = float(np.mean([1.0 if r.sign_agreed else 0.0 for r in segment_details]))

    metrics = ValidationMetrics(
        m1_tau_spearman=m1_tau,
        m1_epsilon_spearman=m1_eps,
        m1b_policy_spearman=m1b,
        m2_mean_abs_bias=m2_bias,
        m3_tau_coverage=m3_tau,
        m3_epsilon_coverage=m3_eps,
        m4_self_consistency=m4_self_cons,
        m5_sign_agreement=m5_sign,
    )

    return ValidationMetricReport(
        seed=seed,
        sample_size=len(test_truth),
        metrics=metrics,
        segment_details=segment_details,
    )


def run_unadjusted_control_c1(
    held_out_test: pd.DataFrame,
    truth_sidecar: pd.DataFrame | str | None = None,
) -> ControlRunResult:
    """Evaluates naive unadjusted within-segment OLS on confounded data (c1).

    Under observed confounding, naive regression omits risk factors that drive
    both treatment and renewal, causing bias and rank distortion.

    Args:
        held_out_test: Held-out test policies with treatment_rate_change and renewed.
        truth_sidecar: Ground truth sidecar.

    Returns:
        ControlRunResult: Control evaluation confirming expected failure.
    """
    from aegis.schemas.causal import SegmentElasticityEstimate, SegmentElasticityTable

    required = {"segment_id", "treatment_rate_change", "renewed"}
    missing = required - set(held_out_test.columns)
    if missing:
        raise ValidationHarnessError(f"held_out_test missing required columns for c1: {missing}")

    rows: list[SegmentElasticityEstimate] = []
    for seg_id, group in held_out_test.groupby("segment_id"):
        t = group["treatment_rate_change"].to_numpy(dtype=float)
        y = group["renewed"].to_numpy(dtype=float)
        n = len(t)
        var_t = float(np.var(t))
        if var_t < 1e-12 or n < 3:
            slope = 0.0
            se = 0.1
        else:
            cov_ty = float(np.cov(t, y)[0, 1])
            slope = cov_ty / var_t
            alpha = float(np.mean(y) - slope * np.mean(t))
            residuals = y - (alpha + slope * t)
            s2 = float(np.sum(residuals**2) / max(n - 2, 1))
            se = float(math.sqrt(s2 / (n * var_t)))

        q0_hat = float(np.clip(np.mean(y), 1e-6, 1.0))
        tau_lower = slope - 1.96 * se
        tau_upper = slope + 1.96 * se
        eps_hat = slope / q0_hat
        eps_lower = tau_lower / q0_hat
        eps_upper = tau_upper / q0_hat

        rows.append(
            SegmentElasticityEstimate(
                segment_id=str(seg_id),
                inference_policy_count=n,
                baseline_retention_q0=q0_hat,
                tau_hat=slope,
                tau_ci_lower=tau_lower,
                tau_ci_upper=tau_upper,
                epsilon_hat=eps_hat,
                epsilon_ci_lower=eps_lower,
                epsilon_ci_upper=eps_upper,
            )
        )

    unadj_table = SegmentElasticityTable(segments=rows)
    report = compute_validation_metrics(unadj_table, truth_sidecar, held_out_test)

    failed_expected = (
        report.metrics.m1_tau_spearman < 0.65
        or report.metrics.m2_mean_abs_bias > 0.15
        or report.metrics.m5_sign_agreement < M5_SIGN_AGREEMENT_FLOOR
    )

    return ControlRunResult(
        control_id="c1_unadjusted",
        control_name="Unadjusted naive within-segment OLS on confounded data",
        is_negative_control=True,
        metrics=report.metrics,
        failed_gate_as_expected=failed_expected,
        diagnostic_message=(
            f"c1 m1_tau={report.metrics.m1_tau_spearman:.4f}, "
            f"m2_bias={report.metrics.m2_mean_abs_bias:.4f}, "
            f"m5_sign={report.metrics.m5_sign_agreement:.4f}"
        ),
    )


def run_shuffled_control_c2(
    segment_estimates: SegmentElasticityTable | pd.DataFrame,
    truth_sidecar: pd.DataFrame | str | None,
    held_out_test: pd.DataFrame,
    random_state: int = 42,
) -> ControlRunResult:
    """Evaluates randomly shuffled segment effect predictions (c2).

    Permuting estimates across segments destroys correlation with ground truth.

    Args:
        segment_estimates: Valid segment elasticity table.
        truth_sidecar: Ground truth sidecar.
        held_out_test: Held-out test policies.
        random_state: Seed for reproducible permutation.

    Returns:
        ControlRunResult: Control evaluation confirming expected failure.
    """
    from aegis.schemas.causal import SegmentElasticityEstimate, SegmentElasticityTable

    est_df = _extract_segment_df(segment_estimates)
    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(len(est_df))

    shuffled_rows: list[SegmentElasticityEstimate] = []
    for i, orig_idx in enumerate(shuffled_indices):
        source = est_df.iloc[orig_idx]
        target_seg_id = str(est_df.iloc[i]["segment_id"])
        target_count = int(est_df.iloc[i]["inference_policy_count"])
        q0 = float(source["baseline_retention_q0"])
        tau_hat = float(source["tau_hat"])
        tau_lower = float(source["tau_ci_lower"])
        tau_upper = float(source["tau_ci_upper"])
        eps_hat = float(source["epsilon_hat"])
        eps_lower = float(source["epsilon_ci_lower"])
        eps_upper = float(source["epsilon_ci_upper"])

        shuffled_rows.append(
            SegmentElasticityEstimate(
                segment_id=target_seg_id,
                inference_policy_count=target_count,
                baseline_retention_q0=q0,
                tau_hat=tau_hat,
                tau_ci_lower=tau_lower,
                tau_ci_upper=tau_upper,
                epsilon_hat=eps_hat,
                epsilon_ci_lower=eps_lower,
                epsilon_ci_upper=eps_upper,
            )
        )

    shuffled_table = SegmentElasticityTable(segments=shuffled_rows)
    report = compute_validation_metrics(shuffled_table, truth_sidecar, held_out_test)
    failed_expected = abs(report.metrics.m1_tau_spearman) < 0.50

    return ControlRunResult(
        control_id="c2_shuffled",
        control_name="Randomly permuted segment effect predictions",
        is_negative_control=True,
        metrics=report.metrics,
        failed_gate_as_expected=failed_expected,
        diagnostic_message=f"c2 m1_tau={report.metrics.m1_tau_spearman:.4f} (near 0 as expected)",
    )


def run_zero_effect_control_c3(
    zero_effect_estimates: SegmentElasticityTable | pd.DataFrame,
    benchmark_truth: pd.DataFrame | str | None,
    held_out_test: pd.DataFrame,
    max_false_discovery_rate: float = 0.15,
) -> ControlRunResult:
    """Evaluates zero-effect DGP predictions (c3).

    Asserts that the proportion of segments falsely flagged significant at alpha=0.05
    (i.e. where 95% CI excludes 0.0) remains near nominal (<= max_false_discovery_rate),
    and that recovery against non-zero benchmark truth fails.

    Args:
        zero_effect_estimates: Estimates from an estimator fit on b(X)=0 data.
        benchmark_truth: Benchmark non-zero truth sidecar.
        held_out_test: Held-out test policies.
        max_false_discovery_rate: Upper bound on false discovery rate.

    Returns:
        ControlRunResult: Control evaluation confirming nominal FDR and recovery failure.
    """
    est_df = _extract_segment_df(zero_effect_estimates)

    # Check false discovery rate (CI excludes zero)
    flagged_significant = sum(
        1
        for _, row in est_df.iterrows()
        if float(row["tau_ci_lower"]) > 0.0 or float(row["tau_ci_upper"]) < 0.0
    )
    fdr = float(flagged_significant / len(est_df))

    # Evaluate against benchmark truth (must fail recovery)
    report = compute_validation_metrics(est_df, benchmark_truth, held_out_test)
    failed_expected = fdr <= max_false_discovery_rate and (
        abs(report.metrics.m1_tau_spearman) < 0.50 or report.metrics.m5_sign_agreement < 0.90
    )

    return ControlRunResult(
        control_id="c3_zero_effect",
        control_name="Zero-effect DGP (b(X)=0)",
        is_negative_control=True,
        metrics=report.metrics,
        failed_gate_as_expected=failed_expected,
        diagnostic_message=(
            f"c3 false_discovery_rate={fdr:.3f} (<= {max_false_discovery_rate}), "
            f"m1_tau={report.metrics.m1_tau_spearman:.4f}"
        ),
    )


def run_positive_control(
    randomized_estimates: SegmentElasticityTable | pd.DataFrame,
    randomized_truth: pd.DataFrame | str | None,
    held_out_test: pd.DataFrame,
    min_m1: float = 0.90,
) -> ControlRunResult:
    """Evaluates estimator recovery on unconfounded randomized treatment DGP.

    Args:
        randomized_estimates: Estimates from estimator fit on confounding_dial=0.0 data.
        randomized_truth: Ground truth sidecar for the randomized DGP.
        held_out_test: Held-out test policies.
        min_m1: Minimum required Spearman correlation.

    Returns:
        ControlRunResult: Positive control outcome confirming clean recovery.
    """
    report = compute_validation_metrics(randomized_estimates, randomized_truth, held_out_test)
    passed = bool(
        report.metrics.m1_tau_spearman >= min_m1
        and report.metrics.m5_sign_agreement >= 0.90
    )

    return ControlRunResult(
        control_id="positive_control",
        control_name="Randomized treatment assignment variant (unconfounded)",
        is_negative_control=False,
        metrics=report.metrics,
        failed_gate_as_expected=not passed,
        diagnostic_message=(
            f"positive_control m1_tau={report.metrics.m1_tau_spearman:.4f} "
            f"(>= {min_m1}: {passed})"
        ),
    )


def compute_separation_gap(
    metric_name: str,
    direction: str,
    estimator_values: list[float],
    control_values: list[float],
    min_margin: float,
) -> SeparationGap:
    """Computes empirical separation between estimator and control distributions (P2-D6a).

    Args:
        metric_name: Name of metric (e.g. m1_tau_spearman).
        direction: 'higher_is_better' or 'lower_is_better'.
        estimator_values: Observed metric values across estimator pilot runs.
        control_values: Observed metric values across negative controls.
        min_margin: Minimum required separation gap Delta_min.

    Returns:
        SeparationGap: Detailed gap diagnostics and recommended threshold.
    """
    if not estimator_values or not control_values:
        raise ValidationHarnessError(
            f"Cannot compute separation gap for '{metric_name}' with empty values"
        )

    if direction == "higher_is_better":
        e_min = float(min(estimator_values))
        c_max = float(max(control_values))
        gap = e_min - c_max
        separated = bool(gap >= min_margin)
        # Midpoint placement with minimum margin safety
        rec_thresh = float(c_max + max(min_margin, gap / 2.0))
        return SeparationGap(
            metric_name=metric_name,
            direction="higher_is_better",
            estimator_min=e_min,
            control_max=c_max,
            gap=gap,
            min_margin=min_margin,
            separated=separated,
            recommended_threshold=round(rec_thresh, 4),
        )
    if direction == "lower_is_better":
        e_max = float(max(estimator_values))
        c_min = float(min(control_values))
        gap = c_min - e_max
        separated = bool(gap >= min_margin)
        rec_thresh = float(c_min - max(min_margin, gap / 2.0))
        return SeparationGap(
            metric_name=metric_name,
            direction="lower_is_better",
            estimator_min=e_max,
            control_max=c_min,
            gap=gap,
            min_margin=min_margin,
            separated=separated,
            recommended_threshold=round(rec_thresh, 4),
        )

    raise ValidationHarnessError(f"Unsupported metric direction: {direction}")


def build_calibrated_thresholds(
    estimator_runs: list[ValidationMetricReport],
    negative_controls: list[ControlRunResult],
) -> tuple[dict[str, SeparationGap], ValidationThresholds]:
    """Calculates separation gaps and builds frozen thresholds (P2-D6a).

    The directional sign-agreement gate is fixed at ``M5_SIGN_AGREEMENT_FLOOR``
    rather than separation-calibrated: with homogeneous-negative true effects,
    shuffled and unadjusted controls preserve signs and cannot separate.

    Args:
        estimator_runs: Validation reports for pilot seeds.
        negative_controls: Validation results for negative controls.

    Returns:
        tuple[dict[str, SeparationGap], ValidationThresholds]: Gaps and thresholds.

    Raises:
        ValidationHarnessError: If any separation-calibrated metric fails.
    """
    if not estimator_runs:
        raise ValidationHarnessError("No estimator pilot runs provided for calibration")
    if not negative_controls:
        raise ValidationHarnessError("No negative controls provided for calibration")

    gaps: dict[str, SeparationGap] = {}
    metrics_spec = [
        ("m1_tau_spearman", "higher_is_better", 0.15, lambda m: m.m1_tau_spearman),
        ("m1_epsilon_spearman", "higher_is_better", 0.15, lambda m: m.m1_epsilon_spearman),
        ("m1b_policy_spearman", "higher_is_better", 0.15, lambda m: m.m1b_policy_spearman),
        ("m2_mean_abs_bias", "lower_is_better", 0.05, lambda m: m.m2_mean_abs_bias),
    ]

    metric_evals: dict[str, list[float]] = {}
    for name, direction, margin, getter in metrics_spec:
        e_vals = [getter(r.metrics) for r in estimator_runs]
        c_vals = [getter(c.metrics) for c in negative_controls]
        metric_evals[name] = e_vals
        gaps[name] = compute_separation_gap(name, direction, e_vals, c_vals, margin)

    unseparated = [name for name, gap in gaps.items() if not gap.separated]
    if unseparated:
        raise ValidationHarnessError(
            f"Qualification Q3 failed: no statistical separation for {unseparated}",
            details={"unseparated_metrics": unseparated},
        )

    # Calculate median targets
    m1_target = round(
        gaps["m1_tau_spearman"].recommended_threshold
        + 0.5 * (
            float(np.median(metric_evals["m1_tau_spearman"]))
            - gaps["m1_tau_spearman"].recommended_threshold
        ),
        4,
    )
    m1e_target = round(
        gaps["m1_epsilon_spearman"].recommended_threshold
        + 0.5 * (
            float(np.median(metric_evals["m1_epsilon_spearman"]))
            - gaps["m1_epsilon_spearman"].recommended_threshold
        ),
        4,
    )
    m1b_target = round(
        gaps["m1b_policy_spearman"].recommended_threshold
        + 0.5 * (
            float(np.median(metric_evals["m1b_policy_spearman"]))
            - gaps["m1b_policy_spearman"].recommended_threshold
        ),
        4,
    )
    m2_target = round(
        gaps["m2_mean_abs_bias"].recommended_threshold
        - 0.5 * (
            gaps["m2_mean_abs_bias"].recommended_threshold
            - float(np.median(metric_evals["m2_mean_abs_bias"]))
        ),
        4,
    )

    thresholds = ValidationThresholds(
        m1_tau_spearman_floor=gaps["m1_tau_spearman"].recommended_threshold,
        m1_tau_spearman_target=m1_target,
        m1_epsilon_spearman_floor=gaps["m1_epsilon_spearman"].recommended_threshold,
        m1_epsilon_spearman_target=m1e_target,
        m1b_policy_spearman_floor=gaps["m1b_policy_spearman"].recommended_threshold,
        m1b_policy_spearman_target=m1b_target,
        m2_mean_abs_bias_ceiling=gaps["m2_mean_abs_bias"].recommended_threshold,
        m2_mean_abs_bias_target=m2_target,
        m3_tau_coverage_floor=0.90,
        m3_epsilon_coverage_floor=0.90,
        m4_self_consistency_floor=1.0,
        m5_sign_agreement_floor=M5_SIGN_AGREEMENT_FLOOR,
    )

    return gaps, thresholds


def format_q3_calibration_report(
    report: Q3CalibrationReport,
    yaml_block: str,
) -> str:
    """Formats the comprehensive Q3 threshold calibration markdown report."""
    passed = (
        report.clear_separation_achieved
        and report.all_negative_controls_failed
        and report.positive_control_passed
    )
    lines: list[str] = [
        "# Qualification Q3: Threshold Calibration Report",
        "",
        "**Executed:** October 26, 2026  ",
        f"**Pilot Seeds:** `{report.pilot_seeds}`  ",
        "**Method:** Separation gap with minimum margin (ADR-013, P2-D6a)  ",
        (
            f"**Outcome:** {'✅ PASSED' if passed else '❌ FAILED'} — "
            "Requires separation, all negative-control failures, and positive-control pass."
        ),
        "",
        "## 1. Estimator Performance on Pilot Seeds",
        "",
        "| Seed | m1(tau) | m1(eps) | m1b | m2 | m3(tau) | m3(eps) | m4 | m5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for est in report.estimator_runs:
        m = est.metrics
        lines.append(
            f"| {est.seed} | {m.m1_tau_spearman:.4f} | {m.m1_epsilon_spearman:.4f} | "
            f"{m.m1b_policy_spearman:.4f} | {m.m2_mean_abs_bias:.4f} | "
            f"{m.m3_tau_coverage:.3f} | {m.m3_epsilon_coverage:.3f} | "
            f"{m.m4_self_consistency:.3f} | {m.m5_sign_agreement:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Controls Performance",
            "",
            "| Control | Name | As Expected | m1 (tau) | m2 (bias) | m5 (sign) | Diagnostic |",
            "|---|---|:---:|---:|---:|---:|---|",
        ]
    )

    all_controls = list(report.negative_control_results) + [report.positive_control_result]
    for c in all_controls:
        as_expected = (
            c.failed_gate_as_expected if c.is_negative_control else not c.failed_gate_as_expected
        )
        status = "✅ YES" if as_expected else "❌ NO"
        lines.append(
            f"| `{c.control_id}` | {c.control_name} | {status} | "
            f"{c.metrics.m1_tau_spearman:.4f} | {c.metrics.m2_mean_abs_bias:.4f} | "
            f"{c.metrics.m5_sign_agreement:.4f} | {c.diagnostic_message} |"
        )

    lines.extend(
        [
            "",
            "## 3. Separation Gaps and Threshold Calibration",
            "",
            "| Metric | Estimator Tail | Control Tail | Gap | Margin | Separated | Threshold |",
            "|---|---:|---:|---:|---:|:---:|---:|",
        ]
    )

    for name, gap in report.separation_gaps.items():
        sep_str = "✅ YES" if gap.separated else "❌ NO"
        lines.append(
            f"| `{name}` | {gap.estimator_min:.4f} | {gap.control_max:.4f} | "
            f"{gap.gap:.4f} | {gap.min_margin:.4f} | {sep_str} | "
            f"**{gap.recommended_threshold:.4f}** |"
        )

    lines.extend(
        [
            "",
            "### Fixed absolute gate",
            "",
            f"`m5_sign_agreement` is fixed at **{M5_SIGN_AGREEMENT_FLOOR:.2f}** and excluded",
            "from empirical separation calibration. Under homogeneous-negative true effects,",
            "the shuffled and unadjusted controls preserve direction and cannot separate on sign.",
            "",
            "## 4. Frozen Thresholds Block (`params.yaml`)",
            "",
            "```yaml",
            yaml_block.strip(),
            "```",
            "",
            "## 5. Gate Conclusion",
            "",
            (
                "Qualification Q3 passed with clear statistical separation across all "
                "separation-calibrated metrics. All negative controls demonstrably failed the "
                "harness, and the positive control passed cleanly."
                if passed
                else "Qualification Q3 FAILED. Do not freeze thresholds or proceed to Gate 9."
            ),
            "",
            "Thresholds are frozen in `params.yaml` only when the outcome is PASSED, ahead of",
            "the definitive Stage 10 final-seed evaluation.",
            "",
        ]
    )

    return "\n".join(lines)


def calibrate_q3_pipeline(
    config: AEGISConfig | None = None,
    report_path: Path | str | None = None,
    pilot_sample_size: int | None = None,
) -> Q3CalibrationReport:
    """Orchestrates Qualification Q3 threshold calibration on pilot seeds (P2-D6a).

    Args:
        config: AEGIS configuration. If None, loaded via get_config().
        report_path: Destination path for the Q3 markdown report.
        pilot_sample_size: Optional sample size override for pilot runs.

    Returns:
        Q3CalibrationReport: The completed calibration report with frozen thresholds.
    """
    cfg = config or get_config()
    ml_cfg = cfg.tier1_ml
    est_cfg = ml_cfg.estimator
    dgp_paths = ml_cfg.dgp_paths
    out_path = Path(report_path or "reports/docs/phases/phase_2/q3_threshold_calibration_report.md")

    sample_size = pilot_sample_size or est_cfg.training_sample_size
    pilot_seeds = list(ml_cfg.pilot_seeds)

    training_df = pd.read_csv(dgp_paths.training_table)
    truth_df = load_truth_sidecar(dgp_paths.truth_sidecar)
    feature_cols = validate_training_contract(
        training_df, corridor_width=ml_cfg.exploration_corridor_width
    )

    estimator_runs: list[ValidationMetricReport] = []
    test_seed42: pd.DataFrame | None = None
    est_table_seed42: SegmentElasticityTable | None = None

    for seed in pilot_seeds:
        train, test = stratified_sample_split(
            training_df,
            sample_size,
            test_size=ml_cfg.test_size,
            random_state=seed,
        )
        if seed == 42:
            test_seed42 = test
            stage8_csv = Path(ml_cfg.estimator_paths.segment_estimates)
            if stage8_csv.is_file() and sample_size == est_cfg.training_sample_size:
                est_df = pd.read_csv(stage8_csv)
                est_table = SegmentElasticityTable.model_validate(
                    {"segments": est_df.to_dict("records")}
                )
                est_table_seed42 = est_table
                run_rep = compute_validation_metrics(est_table, truth_df, test, seed=seed)
                estimator_runs.append(run_rep)
                continue

        fitted = fit_causal_estimator(
            train,
            feature_cols,
            estimator_cfg=est_cfg,
            random_state=seed,
        )
        est_table = aggregate_segment_estimates(
            fitted,
            test,
            feature_cols,
            ci_alpha=est_cfg.ci_alpha,
            q0_floor=est_cfg.q0_floor,
            cv_folds=est_cfg.cv_folds,
        )
        if seed == 42:
            est_table_seed42 = est_table

        run_rep = compute_validation_metrics(est_table, truth_df, test, seed=seed)
        estimator_runs.append(run_rep)

    if test_seed42 is None or est_table_seed42 is None:
        raise ValidationHarnessError(
            "Pilot seed 42 execution failed to produce reference outputs"
        )

    # Negative controls c1 and c2
    c1 = run_unadjusted_control_c1(test_seed42, truth_df)
    c2 = run_shuffled_control_c2(est_table_seed42, truth_df, test_seed42, random_state=42)

    # Negative control c3 (zero-effect DGP with b(X)=0)
    c3_dgp_cfg = cfg.model_copy(deep=True)
    c3_dgp_cfg.tier1_ml.synthetic_dgp.b_coeffs.base = 0.0
    c3_dgp_cfg.tier1_ml.synthetic_dgp.b_coeffs.driver_age_ord = 0.0
    c3_dgp_cfg.tier1_ml.synthetic_dgp.b_coeffs.car_age_ord = 0.0
    c3_dgp_cfg.tier1_ml.synthetic_dgp.b_coeffs.interaction = 0.0

    c3_train_df, _, _ = generate_semi_synthetic_dataset(
        forest_features_path=ml_cfg.feature_paths.forest_view,
        segments_path=ml_cfg.segment_paths.policy_segments,
        glm_loss_path=ml_cfg.glm_paths.fitted_loss_cost,
        config=c3_dgp_cfg,
        run_self_tests=False,
    )
    c3_sub_size = min(sample_size, 50_000)
    c3_train, c3_test = stratified_sample_split(
        c3_train_df, c3_sub_size, test_size=ml_cfg.test_size, random_state=42
    )
    c3_est_cfg = est_cfg.model_copy(update={"n_estimators": 20})
    c3_fitted = fit_causal_estimator(
        c3_train, feature_cols, estimator_cfg=c3_est_cfg, random_state=42
    )
    c3_table = aggregate_segment_estimates(
        c3_fitted,
        c3_test,
        feature_cols,
        ci_alpha=est_cfg.ci_alpha,
        q0_floor=est_cfg.q0_floor,
        cv_folds=est_cfg.cv_folds,
    )
    c3 = run_zero_effect_control_c3(c3_table, truth_df, c3_test)

    # Positive control (randomized treatment assignment)
    pos_dgp_cfg = cfg.model_copy(deep=True)
    pos_dgp_cfg.tier1_ml.synthetic_dgp.confounding_dial = 0.0
    pos_train_df, pos_truth_df, _ = generate_semi_synthetic_dataset(
        forest_features_path=ml_cfg.feature_paths.forest_view,
        segments_path=ml_cfg.segment_paths.policy_segments,
        glm_loss_path=ml_cfg.glm_paths.fitted_loss_cost,
        config=pos_dgp_cfg,
        run_self_tests=False,
    )
    pos_train, pos_test = stratified_sample_split(
        pos_train_df, sample_size, test_size=ml_cfg.test_size, random_state=42
    )
    pos_fitted = fit_causal_estimator(
        pos_train, feature_cols, estimator_cfg=est_cfg, random_state=42
    )
    pos_table = aggregate_segment_estimates(
        pos_fitted,
        pos_test,
        feature_cols,
        ci_alpha=est_cfg.ci_alpha,
        q0_floor=est_cfg.q0_floor,
        cv_folds=est_cfg.cv_folds,
    )
    pos_control = run_positive_control(pos_table, pos_truth_df, pos_test)

    # Build calibrated thresholds
    neg_controls = [c1, c2, c3]
    gaps, thresholds = build_calibrated_thresholds(estimator_runs, neg_controls)

    threshold_lines = [
        f"    {name}: {value:.{2 if name.startswith(('m3_', 'm4_')) else 4}f}\n"
        for name, value in thresholds.model_dump().items()
    ]
    yaml_block = "tier1_ml:\n  validation_thresholds:\n" + "".join(threshold_lines)

    all_neg_passed = all(c.failed_gate_as_expected for c in neg_controls)
    pos_passed = not pos_control.failed_gate_as_expected
    clear_sep = all(g.separated for g in gaps.values())

    report = Q3CalibrationReport(
        pilot_seeds=pilot_seeds,
        estimator_runs=estimator_runs,
        negative_control_results=neg_controls,
        positive_control_result=pos_control,
        separation_gaps=gaps,
        recommended_thresholds=thresholds,
        all_negative_controls_failed=all_neg_passed,
        positive_control_passed=pos_passed,
        clear_separation_achieved=clear_sep,
    )

    report_md = format_q3_calibration_report(report, yaml_block)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")

    if not all_neg_passed or not pos_passed:
        raise ValidationHarnessError(
            "Qualification Q3 failed: control outcomes did not meet gate requirements",
            details={
                "all_negative_controls_failed": all_neg_passed,
                "positive_control_passed": pos_passed,
                "report_path": str(out_path),
            },
        )

    return report
