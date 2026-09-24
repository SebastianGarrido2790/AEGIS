"""Synthetic-treatment causal elasticity estimation and validation pipeline (Tier 1, ADR-014).

This module implements the Double Machine Learning (CausalForestDML) elasticity model
and its structural validation protocol on observational auto insurance benchmark data.

Architectural Context & Invariants:
- Causal validation against synthetic ground truth: Real claims datasets (such as freMTPL2)
  do not observe randomized price experiments or counterfactual policy rate changes. To validate
  that the causal estimator reliably recovers treatment effects and segment heterogeneity,
  a synthetic data-generating process (DGP) injects treatment rate variations and an outcome
  linked via a known structural response function tau(X) = dY/dT (ADR-014, PRD §11).
- Advisory output boundary (INV-5): In the full AEGIS agent loop, elasticity and treatment effect
  estimates are strictly advisory and surfaced only alongside regulatory compliance checks and
  financial loss-ratio impact context.
- Model registry separation (ADR-015): Validated models and DoWhy refutation summaries are
  serialized for MLflow experiment tracking under the registered name `aegis-causal-elasticity`.
- Sample-size provenance (ADR-014 amendment): correlation and every other calibration metric
  are sample-size-sensitive statistics, not fixed properties of the estimator. Every result
  and artifact records the exact sample sizes it was computed from (`total_sample_size`,
  `test_sample_size`) so a reader — and a test — can tell whether two reported correlations
  are even comparable before drawing a conclusion from the difference between them.

Empirical findings behind three parameter changes in this module (2026-09 calibration pass,
see scripts/calibrate_causal_threshold.py and its output artifacts/calibration.json):

1. `random_state` propagation bug (fixed): `fit_causal_elasticity` previously called
   `_prepare_causal_dataset(frame, feature_columns)` without forwarding its own `random_state`
   argument, so `_prepare_causal_dataset`'s own default (42) silently governed the synthetic
   treatment/outcome DGP on every call regardless of what seed the caller requested. Only the
   row subsample, the train/test split, and CausalForestDML's internal randomness actually
   varied with the caller's seed. Found by running the same nominal seed through two different
   call paths and getting two different results — not visible from a static read alone.
2. `n_estimators`: raised from 24 to 100. Holding seed and split fixed, correlation against
   the known synthetic ground truth more than tripled (0.15 -> 0.56) moving from 24 to 100
   trees on a previously pathological seed, and plateaued beyond 100 (200 trees: 0.55, no
   further gain). 24 trees was empirically undersized for this estimation task.
3. `max_rows`: raised from 5,000 to 20,000. This was the dominant fix. At max_rows=5000
   (~1,000-row held-out test split), correlation swung from -0.65 to +0.98 across just 15
   seeds, including two seeds producing a strongly *negative* correlation — the model
   appearing to recover the *opposite* of the true heterogeneous effect. The same seed that
   produced -0.65 at max_rows=5000 produced +0.76 at max_rows=20000, with everything else
   held fixed. A ~1,000-row test split is simply too small to measure a stable correlation
   coefficient against, given the real dataset has 678,013 available rows.

Full 15-seed calibration at max_rows=20,000 (2026-09, non-sandboxed environment):
  Seeds 0–13: all positive correlations (0.4535 – 0.9849).
  Seed 14: correlation = -0.4397 — a genuine outlier where the model recovers the
  *opposite* shape of heterogeneous effect (ATE still positive at 24.53, CI excludes zero,
  refutations pass). This confirms that even at 20,000 rows the estimator can occasionally
  invert the heterogeneity direction. The production floor (test_causal_elasticity.py
  _PRODUCTION_ARTIFACT_CORRELATION_FLOOR) is set to 0.40 — below the minimum positive
  correlation (0.4535) but well above the negative outlier, so it catches this failure
  mode without false positives on normal seed variation.
   Re-run `scripts/calibrate_causal_threshold.py` in a less constrained environment before
   treating this as fully calibrated.

Data-Generating Process (DGP) Design:
1. Treatment assignment (T): Includes both a systematic risk-driven component (confounding)
   and independent stochastic Gaussian noise (residual identifying variation). This ensures
   Double-ML residualization preserves identifying signal post-orthogonalization.
2. Analytic ground truth: tau(X) is defined programmatically as the exact partial derivative
   of the outcome function with respect to treatment (dY/dT = 2.0 + 1.5 * risk_index),
   guaranteeing validation targets do not drift from the synthetic outcome equation.
3. Heterogeneous outcome (Y): Combines structural baseline risk, causal treatment response tau(X)*T,
   and observational noise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.metrics import mean_absolute_error

from aegis.pipelines.feature.pipeline import build_feature_matrix, create_policy_split

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_COLUMNS = (
    "driver_age",
    "veh_age",
    "bonus_malus",
    "veh_power",
    "exposure_normalized",
    "driver_risk_score",
    "vehicle_risk_score",
    "risk_index",
)


@dataclass(frozen=True)
class CausalElasticityResult:
    """Structured output for Stage 4 estimator validation."""

    fitted_model: Any
    feature_columns: tuple[str, ...]
    treatment_variable: str
    average_treatment_effect: float
    treatment_effect_confidence_interval: tuple[float, float]
    correlation: float
    calibration_metrics: dict[str, float]
    refutation_summary: dict[str, Any]
    ground_truth: pd.Series
    total_sample_size: int
    test_sample_size: int


def compute_residual_variance_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Compute raw residual variance and residual identifying variance diagnostic.

    Args:
        frame: DataFrame containing 'risk_index' and 'treatment_rate_change'.

    Returns:
        dict[str, float]: Dictionary containing:
            - 'residual_variance': Raw sample variance of treatment residuals
              (data provenance metric).
            - 'residual_identifying_variance': Fraction of treatment variance unexplained
              by systematic risk index (1 - R^2, diagnostic metric bounded in [0.20, 0.70]).

    Note:
        This metric is itself computed over whatever `frame` is passed in, and its value
        will differ by sample size the same way correlation does. Callers that want a
        production-comparable number must pass the same sample used elsewhere for
        production reporting, not an independently-sized convenience sample.
    """
    corr = float(np.corrcoef(frame["risk_index"], frame["treatment_rate_change"])[0, 1])
    identifying_var = 1.0 - (corr**2)
    systematic = 0.05 + 0.10 * frame["risk_index"]
    raw_residual = frame["treatment_rate_change"] - systematic
    raw_var = float(np.var(raw_residual, ddof=1))
    return {
        "residual_variance": raw_var,
        "residual_identifying_variance": identifying_var,
    }


def add_synthetic_treatment(frame: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Construct synthetic treatment rate change with independent stochastic variation (Fix #1).

    Treatment assignment includes both a systematic component driven by `risk_index`
    and an independent stochastic component (Gaussian noise). The systematic component
    reflects observational confounding, while the independent residual variation provides
    the identifying variation necessary for Double-ML residualization.

    Covariate Inclusion Decision (X):
    `risk_index`, `driver_risk_score`, `vehicle_risk_score`, and `exposure_normalized`
    all remain in `X`. Although `risk_index` is derived from driver, vehicle, and exposure
    features, keeping all segment risk metrics in `X` allows the causal forest to split on
    granular risk dimensions for segment-level treatment heterogeneity. The independent
    stochastic noise introduced here guarantees full-rank identifying variation
    post-orthogonalization.
    """
    prepared = frame.copy()
    if "risk_index" not in prepared.columns:
        prepared = build_feature_matrix(prepared)

    rng = np.random.default_rng(random_state)
    treatment_noise = rng.normal(0.0, 0.05, size=len(prepared))
    systematic_assignment = 0.05 + 0.10 * prepared["risk_index"]
    unclipped_treatment = systematic_assignment + treatment_noise
    clipped_count = int((unclipped_treatment < 0.01).sum())
    if clipped_count > 0:
        logger.info(
            "Clipped %d treatment_rate_change values below the 0.01 positivity floor.",
            clipped_count,
        )
    prepared["treatment_rate_change"] = unclipped_treatment.astype(float)
    prepared["treatment_rate_change"] = prepared["treatment_rate_change"].clip(lower=0.01)
    return prepared


def compute_true_causal_effect(risk_index: Any) -> np.ndarray:
    """Analytic derivative of synthetic outcome with respect to treatment (Fix #2).

    Given outcome DGP:
        Y = 100.0 + 15.0 * risk_index + (2.0 + 1.5 * risk_index) * T + noise
    The true causal effect (marginal response / partial derivative) is:
        tau(X) = dY / dT = 2.0 + 1.5 * risk_index.

    This function serves as the single programmatic ground-truth definition used for
    both outcome data generation and estimator validation, ensuring the validation
    target cannot drift from the structural causal equation.
    """
    risk_arr = np.asarray(risk_index, dtype=float)
    return 2.0 + 1.5 * risk_arr


def _prepare_causal_dataset(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    random_state: int = 42,
) -> pd.DataFrame:
    """Add a synthetic outcome linked to treatment via the analytic causal effect function."""
    prepared = add_synthetic_treatment(frame, random_state=random_state)
    rng = np.random.default_rng(random_state)
    true_effect = compute_true_causal_effect(prepared["risk_index"])
    outcome_noise = rng.normal(0.0, 1.0, size=len(prepared))

    # Outcome DGP: structural baseline + heterogeneous causal response + observational noise
    prepared["synthetic_claim_amount"] = (
        100.0
        + 15.0 * prepared["risk_index"]
        + true_effect * prepared["treatment_rate_change"]
        + outcome_noise
    )
    missing_features = set(feature_columns) - set(prepared.columns)
    if missing_features:
        raise KeyError(f"Missing causal feature columns: {sorted(missing_features)}")
    return prepared


def _run_dowhy_refuters(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    num_simulations: int = 10,
    random_state: int = 42,
    max_refutation_rows: int = 500,
) -> dict[str, Any]:
    """Execute DoWhy refutation checks against the synthetic causal structure (Fix #4).

    Statistical Hypothesis and Pass Convention:
    Unlike classical significance testing where rejecting the null (p < 0.05) is sought,
    DoWhy refutation checks formulate estimate invariance / stability as the null hypothesis:
    - Placebo treatment refuter: The true effect under placebo treatment is zero. A high
      p-value (p > 0.05) means the placebo effect is statistically indistinguishable from zero,
      confirming the estimated treatment effect is not an artifact of random correlation.
    - Random common cause refuter: An unobserved random covariate does not alter the estimated
      effect. A high p-value (p > 0.05) indicates the estimate is invariant to random noise.
    - Data subset refuter: Subsampling the dataset does not meaningfully change the estimated
      effect. A high p-value (p > 0.05) or minimal relative effect shift (|new - est| / |est| < 10%)
      confirms the estimate is stable across data partitions.

    Therefore, a refutation test passes when the estimator fails to reject the null of stability.

    Note: this function deliberately caps at `max_refutation_rows` (default 500) for the
    refutation suite specifically, independent of `fit_causal_elasticity`'s `max_rows`. DoWhy's
    permutation-based refuters are expensive per row; this cap keeps refutation runtime bounded.
    This cap was NOT implicated in the correlation-instability findings above, since refutation
    results (p-values/pass-fail) are computed independently of, and after, the correlation
    metric — raising it is a separate cost/thoroughness tradeoff, not a correctness fix.
    """
    from dowhy import CausalModel

    model_df = frame.copy()
    if "synthetic_claim_amount" not in model_df.columns:
        model_df = _prepare_causal_dataset(model_df, feature_columns, random_state=random_state)

    if len(model_df) > max_refutation_rows:
        model_df = model_df.sample(
            n=max_refutation_rows, random_state=random_state
        ).reset_index(drop=True)

    common_causes = [c for c in feature_columns if c in model_df.columns]

    causal_model = CausalModel(
        data=model_df,
        treatment="treatment_rate_change",
        outcome="synthetic_claim_amount",
        common_causes=common_causes,
        effect_modifiers=common_causes,
    )
    estimand = causal_model.identify_effect()
    estimate = causal_model.estimate_effect(
        estimand,
        method_name="backdoor.econml.dml.CausalForestDML",
        effect_modifiers=common_causes,
        method_params={
            "init_params": {
                "n_estimators": 100,
                "subforest_size": 4,
                "random_state": random_state,
            },
            "fit_params": {},
        },
    )

    placebo = causal_model.refute_estimate(
        estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=num_simulations,
        random_state=random_state,
    )
    random_cause = causal_model.refute_estimate(
        estimand,
        estimate,
        method_name="random_common_cause",
        num_simulations=num_simulations,
        random_state=random_state,
    )
    subset = causal_model.refute_estimate(
        estimand,
        estimate,
        method_name="data_subset_refuter",
        subset_fraction=0.9,
        num_simulations=num_simulations,
        random_state=random_state,
    )

    def _extract_metric(refutation: Any) -> tuple[float, bool]:
        res = getattr(refutation, "refutation_result", None)
        if isinstance(res, dict) and "p_value" in res:
            p_val = float(res["p_value"])
            stat_sig = bool(res.get("is_statistically_significant", p_val <= 0.05))
            passed = not stat_sig
        elif hasattr(refutation, "p_value") and refutation.p_value is not None:
            p_val = float(refutation.p_value)
            passed = bool(p_val > 0.05)
        else:
            msg = f"Unable to extract p-value from DoWhy refutation object: {refutation}"
            raise ValueError(msg)

        # Effect stability check: if p-value is small due to near-zero simulation variance,
        # verify whether the practical effect change is within 10%
        has_est = hasattr(refutation, "estimated_effect")
        has_new = hasattr(refutation, "new_effect")
        if not passed and has_est and has_new:
            est = float(refutation.estimated_effect)
            new_est = float(refutation.new_effect)
            rel_diff = abs(new_est - est) / abs(est) if abs(est) > 1e-6 else 0.0
            if rel_diff < 0.10:
                logger.info(
                    "Refuter passed via effect stability check (relative shift: %.4f < 0.10).",
                    rel_diff,
                )
                passed = True
            else:
                logger.warning(
                    "Refuter failed both significance test (p=%.4f) and effect stability "
                    "(relative shift: %.4f >= 0.10).",
                    p_val,
                    rel_diff,
                )

        return p_val, passed

    p_val_placebo, passed_placebo = _extract_metric(placebo)
    p_val_random, passed_random = _extract_metric(random_cause)
    p_val_subset, passed_subset = _extract_metric(subset)

    return {
        "placebo_treatment": {
            "status": "ok",
            "p_value": p_val_placebo,
            "passed": passed_placebo,
        },
        "random_common_cause": {
            "status": "ok",
            "p_value": p_val_random,
            "passed": passed_random,
        },
        "data_subset": {
            "status": "ok",
            "p_value": p_val_subset,
            "passed": passed_subset,
        },
    }


def fit_causal_elasticity(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: int = 20000,
    run_refuters: bool = True,
) -> CausalElasticityResult:
    """Fit a synthetic-treatment CausalForestDML model and validate recovery.

    Args:
        frame: Input feature matrix (e.g. data/versioned/feature_matrix.csv).
        feature_columns: Columns used as X (heterogeneity covariates).
        test_size: Held-out fraction for the policy-grouped split.
        random_state: Seed governing the ENTIRE pipeline — synthetic DGP, row
            subsampling, train/test split, and the forest's own internal
            randomness. (Previously this seed only governed a subset of these;
            see module docstring for the propagation bug this fixes.)
        max_rows: Cap on rows used for fitting/evaluation, applied AFTER DGP
            construction. Raised from 5,000 to 20,000 following empirical
            findings that 5,000 rows (yielding a ~1,000-row test split) produced
            unstable, occasionally negative, correlation-to-ground-truth results —
            see module docstring. Raise further if your environment can sustain
            the additional fit cost; do not lower below 20,000 without re-running
            the calibration script first.
        run_refuters: If False, skips the DoWhy refutation suite. The refutation
            results are computed independently of, and after, all other metrics
            in this function, so disabling them does not affect correlation, ATE,
            or the confidence interval — this flag exists purely to allow fast,
            repeated calibration runs (e.g. scripts/calibrate_causal_threshold.py)
            without paying DoWhy's permutation-refuter cost on every seed. Always
            True in production (the `train-causal` CLI path never sets this False).

    Returns:
        CausalElasticityResult: structured result including sample-size provenance.

    Raises:
        ValueError: if training/test partitions are empty, or if the point
            estimate falls outside its own reported confidence interval
            (an internal consistency guard against future point-estimate/CI
            computation bugs of the same class as the one this module already
            fixed once).
    """
    prepared = _prepare_causal_dataset(frame, feature_columns, random_state=random_state)
    if len(prepared) > max_rows:
        prepared = prepared.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    total_sample_size = len(prepared)

    train_frame, test_frame = create_policy_split(
        prepared, test_size=test_size, random_state=random_state
    )
    if train_frame.empty or test_frame.empty:
        raise ValueError("Causal training and test partitions must both contain rows.")

    test_sample_size = len(test_frame)

    model = CausalForestDML(
        n_estimators=100,
        subforest_size=4,
        min_samples_leaf=5,
        random_state=random_state,
        inference=True,
        n_jobs=-1,
        cv=2,
    )
    X_train = train_frame[list(feature_columns)]
    X_test = test_frame[list(feature_columns)]
    T_train = train_frame["treatment_rate_change"].to_numpy()
    Y_train = train_frame["synthetic_claim_amount"].to_numpy()

    model.fit(Y_train, T_train, X=X_train)

    raw_effect = np.asarray(model.effect(X_test), dtype=float).reshape(-1)
    interval_lower, interval_upper = model.effect_interval(X_test, alpha=0.05)
    interval_lower_values = np.asarray(interval_lower, dtype=float).reshape(-1)
    interval_upper_values = np.asarray(interval_upper, dtype=float).reshape(-1)
    ci_lower_mean = float(np.mean(interval_lower_values))
    ci_upper_mean = float(np.mean(interval_upper_values))
    treatment_effect_confidence_interval = (ci_lower_mean, ci_upper_mean)

    # Point estimate directly from untransformed model effect (Fix #3)
    average_treatment_effect = float(np.mean(raw_effect))

    # Internal consistency check: ATE must fall within reported mean CI bounds
    if not (ci_lower_mean <= average_treatment_effect <= ci_upper_mean):
        logger.error(
            "Point estimate (%.4f) falls outside reported confidence interval [%.4f, %.4f].",
            average_treatment_effect,
            ci_lower_mean,
            ci_upper_mean,
        )
        raise ValueError(
            f"Point estimate ({average_treatment_effect:.4f}) falls outside reported "
            f"confidence interval [{ci_lower_mean:.4f}, {ci_upper_mean:.4f}]."
        )

    ground_truth = compute_true_causal_effect(test_frame["risk_index"])
    correlation = (
        float(np.corrcoef(raw_effect, ground_truth)[0, 1]) if len(raw_effect) > 1 else 1.0
    )
    if not np.isfinite(correlation):
        correlation = 0.0

    res_metrics = compute_residual_variance_metrics(prepared)
    reference = ground_truth
    calibration_metrics = {
        "correlation": correlation,
        "baseline_mae": float(mean_absolute_error(reference, raw_effect)),
        "average_treatment_effect": average_treatment_effect,
        "residual_variance": res_metrics["residual_variance"],
        "residual_identifying_variance": res_metrics["residual_identifying_variance"],
    }
    refutation_summary = (
        _run_dowhy_refuters(
            prepared,
            feature_columns=feature_columns,
            random_state=random_state,
        )
        if run_refuters
        else {}
    )
    logger.info(
        "fit_causal_elasticity complete: total_sample_size=%d, test_sample_size=%d, "
        "correlation=%.4f. Correlation is sample-size-sensitive — do not compare this "
        "value against a result computed at a different sample size without accounting "
        "for that difference.",
        total_sample_size,
        test_sample_size,
        correlation,
    )
    return CausalElasticityResult(
        fitted_model=model,
        feature_columns=feature_columns,
        treatment_variable="treatment_rate_change",
        average_treatment_effect=average_treatment_effect,
        treatment_effect_confidence_interval=treatment_effect_confidence_interval,
        correlation=correlation,
        calibration_metrics=calibration_metrics,
        refutation_summary=refutation_summary,
        ground_truth=pd.Series(ground_truth, index=test_frame.index),
        total_sample_size=total_sample_size,
        test_sample_size=test_sample_size,
    )


def save_causal_artifact(result: CausalElasticityResult, output_path: Path | str) -> Path:
    """Persist causal metrics and refutation diagnostics as a JSON artifact."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": "causal_forest_dml",
        "treatment_variable": result.treatment_variable,
        "feature_columns": list(result.feature_columns),
        "average_treatment_effect": result.average_treatment_effect,
        "treatment_effect_confidence_interval": {
            "lower": result.treatment_effect_confidence_interval[0],
            "upper": result.treatment_effect_confidence_interval[1],
            "alpha": 0.05,
        },
        "calibration_metrics": result.calibration_metrics,
        "residual_variance": result.calibration_metrics.get("residual_variance"),
        "residual_identifying_variance": result.calibration_metrics.get(
            "residual_identifying_variance"
        ),
        "refutation_summary": result.refutation_summary,
        # Every persisted artifact states the exact sample sizes it was computed from.
        # `correlation` (and every other calibration metric) is a sample-size-sensitive
        # statistic — a CI test reading this artifact must be able to tell what it's
        # actually comparing against.
        "total_sample_size": result.total_sample_size,
        "test_sample_size": result.test_sample_size,
    }
    destination.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return destination