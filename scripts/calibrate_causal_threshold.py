"""Empirical calibration for the causal-elasticity production correlation threshold.

Standalone diagnostic tool — not part of every CI run (CausalForestDML at production
sample size is too slow, and apparently too memory-hungry across repeated in-process
fits, to run on every PR). Run manually when the production correlation floor in
`tests/unit/test_causal_elasticity.py` (`_PRODUCTION_ARTIFACT_CORRELATION_FLOOR`) needs
to move from "a documented placeholder derived from 3 observations" to "a threshold
grounded in an observed distribution."

Each seed is fit in its own subprocess, not in a shared long-lived loop. This is
deliberate: an in-process loop over many CausalForestDML fits was found to exhaust
memory and get killed partway through in one development environment, while isolated
single-seed calls completed cleanly every time. A subprocess-per-seed guarantees each
fit starts with a clean process, at the cost of Python startup overhead per seed —
worth it for a diagnostic tool that runs occasionally, not on a hot path.

Results are written incrementally (one line per completed seed) so a crash or timeout
partway through still leaves the completed seeds on disk instead of losing everything.

Usage:
    uv run python scripts/calibrate_causal_threshold.py
    uv run python scripts/calibrate_causal_threshold.py --seeds 15 --max-rows 20000
    uv run python scripts/calibrate_causal_threshold.py --output artifacts/calibration.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_FEATURE_MATRIX_PATH = Path("data/versioned/feature_matrix.csv")
DEFAULT_SEEDS = 15
DEFAULT_MAX_ROWS = 20000

_WORKER_SCRIPT = """
import json
import sys

import pandas as pd

from aegis.pipelines.training.causal_elasticity import fit_causal_elasticity

feature_matrix_path = sys.argv[1]
seed = int(sys.argv[2])
max_rows = int(sys.argv[3])

df = pd.read_csv(feature_matrix_path)
result = fit_causal_elasticity(
    df, random_state=seed, max_rows=max_rows, run_refuters=False
)
payload = {
    "seed": seed,
    "correlation": result.correlation,
    "average_treatment_effect": result.average_treatment_effect,
    "ci_lower": result.treatment_effect_confidence_interval[0],
    "ci_upper": result.treatment_effect_confidence_interval[1],
    "total_sample_size": result.total_sample_size,
    "test_sample_size": result.test_sample_size,
    "residual_identifying_variance": result.calibration_metrics["residual_identifying_variance"],
}
print(json.dumps(payload))
"""


def run_one_seed(
    feature_matrix_path: Path,
    seed: int,
    max_rows: int,
    timeout_seconds: int,
) -> dict | None:
    """Runs a single seed's fit in an isolated subprocess.

    Returns None (and prints a warning) on timeout or failure, rather than raising —
    a single bad seed should not abort the whole calibration run.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _WORKER_SCRIPT, str(feature_matrix_path), str(seed), str(max_rows)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=True,
        )
    except subprocess.TimeoutExpired:
        print(f"[CALIBRATE][WARN] seed={seed} timed out after {timeout_seconds}s — skipped.")
        return None
    except subprocess.CalledProcessError as exc:
        print(f"[CALIBRATE][WARN] seed={seed} failed: {exc.stderr[-2000:]}")
        return None

    # The worker prints exactly one JSON line; take the last non-empty line to be
    # tolerant of any incidental stdout noise from imported libraries.
    lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    if not lines:
        print(f"[CALIBRATE][WARN] seed={seed} produced no output — skipped.")
        return None
    return json.loads(lines[-1])


def summarize(results: list[dict]) -> dict[str, float]:
    """Computes summary statistics and a suggested conservative floor."""
    correlations = [r["correlation"] for r in results]
    mean = statistics.mean(correlations)
    stdev = statistics.stdev(correlations) if len(correlations) > 1 else 0.0
    minimum = min(correlations)
    maximum = max(correlations)

    # Conservative suggestion: two standard deviations below the observed mean,
    # floored so a suggestion never comes back non-positive. If the distribution is
    # heavily skewed or has extreme outliers (as the max_rows=5000 distribution did),
    # inspect min/max directly rather than trusting this formula blindly — a
    # mean-minus-2-sigma statistic assumes rough symmetry that a skewed or
    # outlier-heavy distribution won't actually have.
    suggested_floor = max(0.10, mean - 2 * stdev)

    return {
        "n_seeds": len(results),
        "mean_correlation": mean,
        "stdev_correlation": stdev,
        "min_correlation": minimum,
        "max_correlation": maximum,
        "suggested_floor_mean_minus_2sigma": suggested_floor,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Empirically calibrate the production correlation threshold for the "
            "causal elasticity artifact regression test. Runs each seed in its own "
            "subprocess for memory isolation."
        )
    )
    parser.add_argument(
        "--feature-matrix",
        type=Path,
        default=DEFAULT_FEATURE_MATRIX_PATH,
        help="Path to the Stage 2 feature matrix.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help="Number of independent random_state seeds to fit and compare.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="max_rows passed to fit_causal_elasticity — should match the production default.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Per-seed subprocess timeout. A seed that exceeds this is skipped, not fatal.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full per-seed results and summary as JSON.",
    )
    args = parser.parse_args()

    if not args.feature_matrix.is_file():
        print(f"[CALIBRATE][ERROR] Feature matrix not found: {args.feature_matrix}")
        return 1

    print(
        f"[CALIBRATE] Running {args.seeds} seeds at max_rows={args.max_rows}, "
        f"one subprocess per seed, timeout={args.timeout_seconds}s each."
    )

    results: list[dict] = []
    t_start = time.time()

    for seed in range(args.seeds):
        t0 = time.time()
        result = run_one_seed(args.feature_matrix, seed, args.max_rows, args.timeout_seconds)
        elapsed = time.time() - t0
        if result is None:
            continue
        results.append(result)
        print(
            f"[CALIBRATE] seed={seed:2d}  correlation={result['correlation']:8.4f}  "
            f"ATE={result['average_treatment_effect']:7.3f}  elapsed={elapsed:.1f}s"
        )
        # Write incrementally so a later crash/timeout doesn't lose completed seeds.
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({"per_seed_results": results, "summary": None}, indent=2),
                encoding="utf-8",
            )

    total_elapsed = time.time() - t_start
    print(f"\n[CALIBRATE] Completed {len(results)}/{args.seeds} seeds in {total_elapsed:.1f}s.")

    if not results:
        print("[CALIBRATE][ERROR] No seeds completed successfully — cannot summarize.")
        return 1

    summary = summarize(results)
    print("\n[CALIBRATE] === Summary ===")
    for key, value in summary.items():
        print(f"[CALIBRATE] {key}: {value}")

    if len(results) < args.seeds:
        print(
            f"\n[CALIBRATE][WARN] Only {len(results)}/{args.seeds} seeds completed — "
            "treat this summary as provisional, not a finished calibration. Re-run "
            "with a higher --timeout-seconds or investigate the skipped seeds directly."
        )

    print(
        "\n[CALIBRATE] Replace _PRODUCTION_ARTIFACT_CORRELATION_FLOOR in "
        "tests/unit/test_causal_elasticity.py with a value grounded in the above "
        "distribution. Inspect min/max directly before trusting "
        "suggested_floor_mean_minus_2sigma — that formula assumes rough symmetry, "
        "which a skewed or outlier-heavy distribution won't have."
    )

    if args.output is not None:
        args.output.write_text(
            json.dumps({"per_seed_results": results, "summary": summary}, indent=2),
            encoding="utf-8",
        )
        print(f"[CALIBRATE] Full results written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())