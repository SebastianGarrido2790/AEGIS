#!/usr/bin/env python3
"""P2-D5a Phase 0 — estimator-seed variance screening (diagnostic only).

Purpose
-------
Decide which lever applies to the Gate 10 rank-recovery failure. The data draw and
the train/test split are held FIXED; only the estimator ``random_state`` varies.

If rank recovery is stable across estimator seeds, the swing seen in Stage 10 is
data-driven (irreducible given the DGP) and forest regularisation (Option A) will
not help -> go to Option C (CATE family change).
If it moves widely, forest Monte-Carlo variance dominates -> Option A is the lever.

Pre-registered interpretation rule (fixed before execution)
-----------------------------------------------------------
    spread = max(m1_tau) - min(m1_tau) across estimator seeds

    spread <  0.03   -> STABLE / data-driven  -> Option C
    spread >  0.10   -> NOISY / forest-driven -> Option A
    0.03..0.10       -> MIXED (inconclusive)  -> Option C (stronger lever), noted

Seed 504 reproduces the Stage 10 worst configuration exactly (DGP noise 10504,
split seed 504), so its m1 should match the recorded 0.5517. That is the probe's
sanity check against the gate artifact.

This script writes no config, no artifacts, and no thresholds. It is read-only
diagnostics. Truth values are used solely to score the held-out predictions.

Usage: uv run python scripts/p2_d5a_phase0_screening.py
"""

from __future__ import annotations

import time

from aegis.config.loader import get_config
from aegis.pipelines.causal_estimator import (
    aggregate_segment_estimates,
    fit_causal_estimator,
    stratified_sample_split,
)
from aegis.pipelines.final_seed_validation import _load_or_create_tables
from aegis.pipelines.validation_harness import compute_validation_metrics

# Worst observed Stage 10 draw (seed 504 -> DGP noise 10504).
NOISE_SEED = 10504
SPLIT_SEED = 504
ESTIMATOR_SEEDS = (504, 1001, 2002)

STABLE_BELOW = 0.03
NOISY_ABOVE = 0.10


def main() -> int:
    """Runs the fixed-draw, fixed-split estimator-seed sweep."""
    config = get_config()
    estimator_cfg = config.tier1_ml.estimator

    print(f"Loading DGP tables for noise_seed={NOISE_SEED} ...", flush=True)
    training, truth, features = _load_or_create_tables(config, NOISE_SEED)
    train, test = stratified_sample_split(
        training,
        estimator_cfg.training_sample_size,
        test_size=config.tier1_ml.test_size,
        random_state=SPLIT_SEED,
    )
    print(f"train={len(train)} test={len(test)} features={len(features)}", flush=True)

    results: dict[int, float] = {}
    for estimator_seed in ESTIMATOR_SEEDS:
        started = time.perf_counter()
        fitted = fit_causal_estimator(
            train, features, estimator_cfg=estimator_cfg, random_state=estimator_seed
        )
        estimates = aggregate_segment_estimates(
            fitted,
            test,
            features,
            ci_alpha=estimator_cfg.ci_alpha,
            q0_floor=estimator_cfg.q0_floor,
            cv_folds=estimator_cfg.cv_folds,
        )
        report = compute_validation_metrics(
            estimates, truth, test, seed=estimator_seed
        )
        metrics = report.metrics
        results[estimator_seed] = metrics.m1_tau_spearman
        print(
            f"estimator_seed={estimator_seed} "
            f"m1_tau={metrics.m1_tau_spearman:.4f} "
            f"m1_eps={metrics.m1_epsilon_spearman:.4f} "
            f"m1b={metrics.m1b_policy_spearman:.4f} "
            f"m2={metrics.m2_mean_abs_bias:.4f} "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )

    values = list(results.values())
    spread = max(values) - min(values)
    if spread < STABLE_BELOW:
        verdict = "STABLE / data-driven -> Option C"
    elif spread > NOISY_ABOVE:
        verdict = "NOISY / forest-driven -> Option A"
    else:
        verdict = "MIXED (inconclusive) -> Option C, noted"
    print("", flush=True)
    print(f"spread(m1_tau) = {spread:.4f}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
