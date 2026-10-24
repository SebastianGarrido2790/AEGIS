"""Qualification Q1 loss-cap sweep (P2-D1a, ADR-012).

Runs the cap-parameterized Tweedie baseline at each EDA-grounded candidate cap, computes
the portfolio loss-removal metrics and segment-level excess concentration on the training
split, and selects the lowest cap whose transition to the next candidate satisfies the
approved Spearman >= threshold and holdout-deviance-change <= threshold criteria. Writes a
Markdown report for `reports/docs/phases/phase_2/q1_cap_sweep_report.md`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from aegis.config.loader import get_config
from aegis.pipelines.glm_baseline import run_glm_baseline
from aegis.pipelines.segments import build_policy_segments


@dataclass(frozen=True)
class CapResult:
    """One candidate cap's measured sweep metrics."""

    cap: float
    losses_removed: float
    loss_removed_pct: float
    claims_truncated: int
    claims_truncated_pct: float
    excess_load: float
    tweedie_p: float
    holdout_deviance: float
    decile_ae: list[float]
    segment_loss_cost: pd.Series
    max_excess_ratio: float
    max_excess_segment: str
    max_excess_share: float
    max_exposure_share: float


def compute_portfolio_metrics(claims: pd.Series, cap: float) -> dict[str, float]:
    """Computes loss-removal and truncation metrics for a single cap.

    Args:
        claims: Policy-level claim amounts (training split).
        cap: Loss cap level.

    Returns:
        Metrics: losses removed, percentage removed, truncated claim count and
        percentage, and the uniform excess load ``total / (total - excess)``.
    """
    total = float(claims.sum())
    excess = float(claims.clip(lower=cap).sub(cap).clip(lower=0).sum())
    removed_pct = excess / total * 100.0 if total > 0 else 0.0
    n_claims = int((claims > 0).sum())
    truncated = int((claims > cap).sum())
    truncated_pct = truncated / n_claims * 100.0 if n_claims else 0.0
    excess_load = total / (total - excess) if total > excess else float("inf")
    return {
        "losses_removed": excess,
        "loss_removed_pct": removed_pct,
        "claims_truncated": truncated,
        "claims_truncated_pct": truncated_pct,
        "excess_load": excess_load,
    }


def spearman_rank_corr(a: pd.Series, b: pd.Series) -> float:
    """Computes Spearman rank correlation between two aligned series."""
    return float(a.corr(b, method="spearman"))


@dataclass(frozen=True)
class ConcentrationResult:
    """Excess-loss concentration of one cap across segments."""

    max_excess_ratio: float
    max_excess_segment: str
    max_excess_share: float
    max_exposure_share: float


def excess_concentration(
    train_view: pd.DataFrame, policy_segments: pd.DataFrame, cap: float
) -> ConcentrationResult:
    """Measures the most concentrated segment's excess share relative to exposure.

    Args:
        train_view: Training-split view rows with `ClaimAmount` and `exposure_capped`.
        policy_segments: Policy-to-segment mapping (output of `build_policy_segments`).
        cap: Loss cap level.

    Returns:
        The maximum excess-share/exposure-share ratio and the responsible segment.
    """
    merged = train_view.merge(policy_segments[["PolicyID", "segment_id"]], on="PolicyID")
    excess = merged["ClaimAmount"].clip(lower=cap).sub(cap).clip(lower=0)
    total_excess = float(excess.sum())
    total_exposure = float(merged["exposure_capped"].sum())

    excess_sum = pd.Series(excess.groupby(merged["segment_id"]).sum())
    exposure_sum = pd.Series(merged.groupby("segment_id")["exposure_capped"].sum())
    excess_share = excess_sum / total_excess
    exposure_share = exposure_sum / total_exposure
    ratio = excess_share / exposure_share

    max_segment = str(ratio.idxmax())
    return ConcentrationResult(
        max_excess_ratio=float(ratio.max()),
        max_excess_segment=max_segment,
        max_excess_share=float(excess_share.loc[max_segment]),
        max_exposure_share=float(exposure_share.loc[max_segment]),
    )


def _segment_loss_cost(
    policy_segments: pd.DataFrame, fitted_train: pd.DataFrame
) -> pd.Series:
    """Exposure-weighted fitted loss cost per segment on the training split."""
    merged = policy_segments.merge(
        fitted_train[["PolicyID", "fitted_loss_cost_per_year"]],
        on="PolicyID",
        how="inner",
    )
    weighted = merged["fitted_loss_cost_per_year"] * merged["exposure_capped"]
    weighted_sum = pd.Series(weighted.groupby(merged["segment_id"]).sum())
    exposure_sum = pd.Series(merged.groupby("segment_id")["exposure_capped"].sum())
    return weighted_sum / exposure_sum


def select_cap(
    caps: list[float],
    spearman: list[float],
    deviance_change: list[float],
    spearman_threshold: float,
    deviance_threshold: float,
) -> tuple[float, int]:
    """Selects the lowest cap whose transition to the next satisfies both criteria.

    Args:
        caps: Candidate caps in increasing order.
        spearman: ``spearman[k]`` is the correlation of segment loss cost at caps ``k``
            and ``k+1``.
        deviance_change: ``deviance_change[k]`` is the relative holdout-deviance change
            from cap ``k`` to ``k+1``.
        spearman_threshold: Minimum required Spearman correlation.
        deviance_threshold: Maximum allowed relative deviance change.

    Returns:
        The selected cap and its index, falling back to the highest cap if no
        transition satisfies both criteria.
    """
    for index in range(len(caps) - 1):
        if (
            spearman[index] >= spearman_threshold
            and deviance_change[index] <= deviance_threshold
        ):
            return caps[index], index
    return caps[-1], len(caps) - 1


def run_cap_sweep(
    input_path: Path | None = None,
    work_dir: Path | None = None,
    report_path: Path | None = None,
) -> tuple[list[CapResult], float, float]:
    """Executes the Q1 cap sweep and writes the Markdown report.

    Args:
        input_path: Stage 2 GLM view. Defaults to params.yaml feature_paths.glm_view.
        work_dir: Temporary directory for per-cap fitted artifacts.
        report_path: Markdown report destination.

    Returns:
        A tuple of the per-cap results, the selected cap, and the selected excess load.
    """
    config = get_config()
    sweep_cfg = config.tier1_ml.cap_sweep
    segment_cfg = config.tier1_ml.segments
    feature_cfg = config.tier1_ml.feature_pipeline
    caps = sweep_cfg.cap_candidates

    view_path = Path(input_path or config.tier1_ml.feature_paths.glm_view)
    view = pd.read_csv(view_path)
    work = Path(work_dir or ".cap_sweep")
    work.mkdir(parents=True, exist_ok=True)

    first_fitted = work / "fitted_first.csv"
    run_glm_baseline(
        input_path=view_path,
        cap=caps[0],
        fitted_path=first_fitted,
        metadata_path=work / "meta_first.json",
        report_path=work / "rep_first.json",
    )
    split = pd.read_csv(first_fitted)[["PolicyID", "split"]]
    policy_segments = build_policy_segments(view, segment_cfg, feature_cfg)
    train_raw = view.merge(split, on="PolicyID", how="inner")
    train_view = cast(
        pd.DataFrame,
        train_raw[train_raw["split"] == "train"].reset_index(drop=True),
    )

    results: list[CapResult] = []
    segment_vectors: list[pd.Series] = []
    for cap in caps:
        fitted_path = work / f"fitted_{cap:g}.csv"
        report = run_glm_baseline(
            input_path=view_path,
            cap=cap,
            fitted_path=fitted_path,
            metadata_path=work / f"meta_{cap:g}.json",
            report_path=work / f"rep_{cap:g}.json",
        )
        fitted = pd.read_csv(fitted_path)
        train_fitted = cast(
            pd.DataFrame,
            fitted[fitted["split"] == "train"].reset_index(drop=True),
        )

        portfolio = compute_portfolio_metrics(
            cast(pd.Series, train_view["ClaimAmount"]), cap
        )
        concentration = excess_concentration(train_view, policy_segments, cap)
        segment_lc = _segment_loss_cost(policy_segments, train_fitted)
        segment_vectors.append(segment_lc)

        results.append(
            CapResult(
                cap=cap,
                losses_removed=float(portfolio["losses_removed"]),
                loss_removed_pct=float(portfolio["loss_removed_pct"]),
                claims_truncated=int(portfolio["claims_truncated"]),
                claims_truncated_pct=float(portfolio["claims_truncated_pct"]),
                excess_load=float(portfolio["excess_load"]),
                tweedie_p=report.tweedie_p,
                holdout_deviance=report.holdout_mean_deviance,
                decile_ae=[row.ae_ratio for row in report.decile_ae],
                segment_loss_cost=segment_lc,
                max_excess_ratio=concentration.max_excess_ratio,
                max_excess_segment=concentration.max_excess_segment,
                max_excess_share=concentration.max_excess_share,
                max_exposure_share=concentration.max_exposure_share,
            )
        )

    spearman = [
        spearman_rank_corr(segment_vectors[k], segment_vectors[k + 1])
        for k in range(len(caps) - 1)
    ]
    deviance_change = [
        abs(results[k + 1].holdout_deviance - results[k].holdout_deviance)
        / results[k].holdout_deviance
        for k in range(len(caps) - 1)
    ]

    selected_cap, _ = select_cap(
        caps,
        spearman,
        deviance_change,
        sweep_cfg.spearman_threshold,
        sweep_cfg.deviance_change_threshold,
    )
    selected = results[[result.cap for result in results].index(selected_cap)]
    _write_report(
        results,
        spearman,
        deviance_change,
        selected_cap,
        Path(report_path or "reports/docs/phases/phase_2/q1_cap_sweep_report.md"),
    )
    return results, selected_cap, selected.excess_load


def _write_report(
    results: list[CapResult],
    spearman: list[float],
    deviance_change: list[float],
    selected_cap: float,
    path: Path,
) -> None:
    """Writes the Q1 cap-sweep Markdown report."""
    caps = [result.cap for result in results]
    header = (
        "| Cap (€) | Loss removed (€) | Removed % | Claims truncated | Truncated % | "
        "Excess load | p̂ | Holdout deviance | Max excess ratio (segment) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for result in results:
        rows.append(
            f"| {result.cap:,.0f} | {result.losses_removed:,.0f} | {result.loss_removed_pct:.2f}% "
            f"| {result.claims_truncated} | {result.claims_truncated_pct:.2f}% "
            f"| {result.excess_load:.4f} | {result.tweedie_p} | {result.holdout_deviance:.4f} "
            f"| {result.max_excess_ratio:.1f}× ({result.max_excess_segment}) |"
        )
    transitions = []
    for k in range(len(caps) - 1):
        transitions.append(
            f"| {caps[k]:,.0f} → {caps[k+1]:,.0f} | {spearman[k]:.4f} | "
            f"{deviance_change[k]*100:.3f}% |"
        )
    selected = next(r for r in results if r.cap == selected_cap)
    lines = [
        "# Q1 — Loss Cap Sweep & Final Calibration",
        "",
        "Qualification test Q1 (P2-D1a, ADR-012). All metrics except holdout deviance "
        "are computed on the training split (330,535 policies / 12,249 claims).",
        "",
        "## Per-candidate metrics",
        "",
        header.rstrip(),
        *rows,
        "",
        "## Adjacent-cap transitions (selection criteria)",
        "",
        "| Transition | Spearman (segment loss-cost) | Holdout deviance change |",
        "|---|---:|---:|",
        *transitions,
        "",
        "## Selection",
        "",
        f"- Selected cap: **€{selected_cap:,.0f}** (lowest truncation level whose "
        f"transition satisfies Spearman ≥ {get_config().tier1_ml.cap_sweep.spearman_threshold} "
        f"and deviance change ≤ "
        f"{get_config().tier1_ml.cap_sweep.deviance_change_threshold*100:.0f}%).",
        f"- Frozen uniform excess load: **{selected.excess_load:.4f}**.",
        "",
        "## Excess concentration (known limitation)",
        "",
        f"The excess losses are materially concentrated: at the selected cap, segment "
        f"`{selected.max_excess_segment}` holds {selected.max_excess_share*100:.1f}% of excess "
        f"loss against {selected.max_exposure_share*100:.2f}% of exposure "
        f"(ratio {selected.max_excess_ratio:.1f}×). This single segment contains the "
        f"€2.04M claim. A uniform excess load therefore understates `Losses₀` for that "
        "segment and overstates it elsewhere; this is recorded as a known benchmark "
        "limitation carried to Phase 6 (operator decision 2026-10-24).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
