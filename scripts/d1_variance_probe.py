#!/usr/bin/env python3
"""D1 variance-decomposition probe (Stage 10 remediation, operator-approved).

Fits the Q2-frozen estimator on the FIXED seed-42 DGP tables (no fresh DGP noise)
with the failing final split/estimator seeds (504, then 502). If metrics recover to
~pilot levels, the Stage 10 failure is DGP-noise variance the Q3 calibration never
sampled (Option A). If they stay degraded, the estimator itself is split-sensitive
(Option B).

Usage: uv run python scripts/d1_variance_probe.py
"""

from __future__ import annotations

import sys

import pandas as pd

from aegis.config.loader import get_config
from aegis.pipelines.causal_estimator import (
    aggregate_segment_estimates,
    fit_causal_estimator,
    stratified_sample_split,
    validate_training_contract,
)
from aegis.pipelines.truth_loader import load_truth_sidecar
from aegis.pipelines.validation_harness import compute_validation_metrics


def main() -> int:
    """Runs the fixed-DGP probe for seeds 504 and 502."""
    config = get_config()
    ml = config.tier1_ml
    training = pd.read_csv(ml.dgp_paths.training_table)
    truth = load_truth_sidecar(ml.dgp_paths.truth_sidecar)
    features = validate_training_contract(training, corridor_width=ml.exploration_corridor_width)
    print(f"fixed DGP tables: {len(training)} rows, {len(features)} features", flush=True)
    for seed in (504, 502):
        train, test = stratified_sample_split(
            training,
            ml.estimator.training_sample_size,
            test_size=ml.test_size,
            random_state=seed,
        )
        fitted = fit_causal_estimator(
            train, features, estimator_cfg=ml.estimator, random_state=seed
        )
        estimates = aggregate_segment_estimates(
            fitted,
            test,
            features,
            ci_alpha=ml.estimator.ci_alpha,
            q0_floor=ml.estimator.q0_floor,
            cv_folds=ml.estimator.cv_folds,
        )
        report = compute_validation_metrics(estimates, truth, test, seed=seed)
        m = report.metrics
        print(
            f"[D1 seed={seed}] m1={m.m1_tau_spearman:.4f} "
            f"m1e={m.m1_epsilon_spearman:.4f} m1b={m.m1b_policy_spearman:.4f} "
            f"m2={m.m2_mean_abs_bias:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
