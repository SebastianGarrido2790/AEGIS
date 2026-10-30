#!/usr/bin/env python3
"""P2-D5a Phase 1 — Bounded CATE Family Evaluation Sweep.

Governing: ADR-013 Amendment (October 29, 2026), P2-D5a Decision Record.
Pre-registered candidate configurations:
  1. CF_GBT_Reg: CausalForestDML with regularized tree partition (min_samples_leaf=50,
     max_samples=0.5, n_estimators=100) and GBT nuisance models (Non-oracle tree).
  2. LinearDML_Linear: LinearDML with linear CATE representation and GBT nuisance
     models (Near-oracle parametric).
  3. LinearDML_Poly: LinearDML with quadratic polynomial series featurizer and GBT
     nuisance models (Non-oracle series/flexible).

Evaluated across three representative DGP noise draws:
  - 10042 (moderate draw)
  - 10101 (hard draw)
  - 10504 (worst draw)

Evaluated metrics against frozen Q3 thresholds:
  - m1_tau_spearman >= 0.8428
  - m1_eps_spearman >= 0.8831
  - m1b_policy_spearman >= 0.7784
  - m2_mean_abs_bias <= 0.1167
  - m3_coverage >= 0.9000
  - m4_self_consistency == 1.0000
  - m5_sign_agreement >= 0.9000

Usage:
  uv run python scripts/p2_d5a_phase1_sweep.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures

from aegis.config.loader import get_config
from aegis.pipelines.causal_estimator import (
    aggregate_segment_estimates,
    stratified_sample_split,
)
from aegis.pipelines.final_seed_validation import _load_or_create_tables
from aegis.pipelines.validation_harness import compute_validation_metrics

NOISE_DRAWS = (10042, 10101, 10504)
SPLIT_SEED = 504  # Standard development split seed (501-505 retired to dev)
CV_FOLDS = 5
CI_ALPHA = 0.05
Q0_FLOOR = 0.01

FROZEN_M1_TAU_FLOOR = 0.8428
FROZEN_M1_EPS_FLOOR = 0.8831
FROZEN_M1B_FLOOR = 0.7784
FROZEN_M2_BIAS_CEILING = 0.1167


@dataclass(frozen=True)
class CandidateSweepResult:
    candidate_name: str
    noise_seed: int
    fit_duration_seconds: float
    m1_tau_spearman: float
    m1_eps_spearman: float
    m1b_policy_spearman: float
    m2_mean_abs_bias: float
    m3_tau_coverage: float
    m4_self_consistency: float
    m5_sign_agreement: float
    all_floors_cleared: bool


def _build_candidate_estimator(name: str, random_state: int) -> Any:
    """Builds a pre-registered candidate estimator instance."""
    model_y: Any = HistGradientBoostingClassifier(random_state=random_state)
    model_t: Any = HistGradientBoostingRegressor(random_state=random_state)

    if name == "CF_GBT_Reg":
        return CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_outcome=True,
            cv=CV_FOLDS,
            n_estimators=100,
            subforest_size=4,
            min_samples_leaf=50,
            max_samples=0.5,
            random_state=random_state,
        )
    if name == "LinearDML_Linear":
        return LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_outcome=True,
            cv=CV_FOLDS,
            random_state=random_state,
        )
    if name == "LinearDML_Poly":
        return LinearDML(
            model_y=model_y,
            model_t=model_t,
            featurizer=PolynomialFeatures(degree=2, include_bias=False),
            discrete_outcome=True,
            cv=CV_FOLDS,
            random_state=random_state,
        )
    raise ValueError(f"Unknown candidate name: {name}")


def run_sweep() -> list[CandidateSweepResult]:
    """Executes the pre-registered bounded sweep across candidates and draws."""
    config = get_config()
    candidates = ("LinearDML_Linear", "LinearDML_Poly", "CF_GBT_Reg")
    all_results: list[CandidateSweepResult] = []

    print("=" * 80)
    print("P2-D5a Phase 1 — Bounded CATE Family Evaluation Sweep")
    print(f"Candidates: {candidates}")
    print(f"DGP Noise Draws: {NOISE_DRAWS}")
    print(f"Frozen Floors: m1(tau)>={FROZEN_M1_TAU_FLOOR}, m1(eps)>={FROZEN_M1_EPS_FLOOR}, "
          f"m1b>={FROZEN_M1B_FLOOR}, m2<={FROZEN_M2_BIAS_CEILING}")
    print("=" * 80, flush=True)

    for noise_seed in NOISE_DRAWS:
        print(f"\n--- Loading DGP tables for noise_seed={noise_seed} ---", flush=True)
        training, truth, features = _load_or_create_tables(config, noise_seed)
        train, test = stratified_sample_split(
            training,
            config.tier1_ml.estimator.training_sample_size,
            test_size=config.tier1_ml.test_size,
            random_state=SPLIT_SEED,
        )
        Y = train["renewed"].to_numpy(dtype=float)
        T = train["treatment_rate_change"].to_numpy(dtype=float)
        X = train[features].to_numpy(dtype=float)

        for name in candidates:
            print(f"Running [{name}] on draw [{noise_seed}] ...", flush=True)
            est = _build_candidate_estimator(name, random_state=SPLIT_SEED)
            t0 = time.perf_counter()
            est.fit(Y, T, X=X)
            duration = time.perf_counter() - t0

            estimates = aggregate_segment_estimates(
                est,
                test,
                features,
                ci_alpha=CI_ALPHA,
                q0_floor=Q0_FLOOR,
                cv_folds=CV_FOLDS,
            )
            report = compute_validation_metrics(
                estimates, truth, test, seed=SPLIT_SEED
            )
            m = report.metrics

            floors_cleared = bool(
                m.m1_tau_spearman >= FROZEN_M1_TAU_FLOOR
                and m.m1_epsilon_spearman >= FROZEN_M1_EPS_FLOOR
                and m.m1b_policy_spearman >= FROZEN_M1B_FLOOR
                and m.m2_mean_abs_bias <= FROZEN_M2_BIAS_CEILING
                and m.m3_tau_coverage >= 0.90
                and m.m4_self_consistency == 1.00
                and m.m5_sign_agreement >= 0.90
            )

            result = CandidateSweepResult(
                candidate_name=name,
                noise_seed=noise_seed,
                fit_duration_seconds=round(duration, 2),
                m1_tau_spearman=round(m.m1_tau_spearman, 4),
                m1_eps_spearman=round(m.m1_epsilon_spearman, 4),
                m1b_policy_spearman=round(m.m1b_policy_spearman, 4),
                m2_mean_abs_bias=round(m.m2_mean_abs_bias, 4),
                m3_tau_coverage=round(m.m3_tau_coverage, 4),
                m4_self_consistency=round(m.m4_self_consistency, 4),
                m5_sign_agreement=round(m.m5_sign_agreement, 4),
                all_floors_cleared=floors_cleared,
            )
            all_results.append(result)
            status_symbol = "PASS" if floors_cleared else "FAIL"
            print(
                f"  -> [{name}] draw={noise_seed}: {status_symbol} "
                f"m1={result.m1_tau_spearman:.4f} m1_eps={result.m1_eps_spearman:.4f} "
                f"m1b={result.m1b_policy_spearman:.4f} m2={result.m2_mean_abs_bias:.4f} "
                f"cov={result.m3_tau_coverage:.4f} ({duration:.1f}s)",
                flush=True,
            )

    return all_results


def write_report(results: list[CandidateSweepResult], output_path: Path) -> None:
    """Writes the markdown summary report of the sweep."""
    lines = [
        "# Phase 1 Bounded CATE Family Sweep Report",
        "",
        "**Governing:** ADR-013 Amendment (October 29, 2026), P2-D5a Decision Record.",
        "**Date:** 2026-10-29",
        "",
        "## Summary Table",
        "",
        "| Candidate | Draw | Duration (s) | m1(tau) | m1(eps) | m1b | m2 bias | Cleared |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in results:
        status = "✅ PASS" if r.all_floors_cleared else "❌ FAIL"
        lines.append(
            f"| `{r.candidate_name}` | {r.noise_seed} | {r.fit_duration_seconds} | "
            f"{r.m1_tau_spearman:.4f} | {r.m1_eps_spearman:.4f} | {r.m1b_policy_spearman:.4f} | "
            f"{r.m2_mean_abs_bias:.4f} | {status} |"
        )

    lines.extend([
        "",
        "## Per-Candidate Aggregates",
        "",
        "| Candidate | Worst m1 | Mean m1 | Worst m2 | Mean Dur (s) | All Cleared |",
        "|---|---:|---:|---:|---:|:---:|",
    ])
    by_candidate: dict[str, list[CandidateSweepResult]] = {}
    for r in results:
        by_candidate.setdefault(r.candidate_name, []).append(r)

    for name, c_results in by_candidate.items():
        worst_m1 = min(r.m1_tau_spearman for r in c_results)
        mean_m1 = sum(r.m1_tau_spearman for r in c_results) / len(c_results)
        worst_m2 = max(r.m2_mean_abs_bias for r in c_results)
        mean_dur = sum(r.fit_duration_seconds for r in c_results) / len(c_results)
        all_cleared = all(r.all_floors_cleared for r in c_results)
        status = "✅ YES" if all_cleared else "❌ NO"
        lines.append(
            f"| `{name}` | {worst_m1:.4f} | {mean_m1:.4f} | {worst_m2:.4f} | "
            f"{mean_dur:.1f} | {status} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to: {output_path}")


def main() -> int:
    results = run_sweep()
    report_path = Path("reports/docs/phases/phase_2/p2_d5a_phase1_sweep_report.md")
    write_report(results, report_path)
    json_path = Path("artifacts/p2_d5a_phase1_sweep_results.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    print(f"Raw results JSON written to: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
