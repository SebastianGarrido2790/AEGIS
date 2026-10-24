"""Unit tests for the Q1 loss-cap sweep metrics and selection logic (P2-D1a)."""

import pandas as pd
import pytest

from aegis.pipelines.cap_sweep import (
    compute_portfolio_metrics,
    excess_concentration,
    select_cap,
    spearman_rank_corr,
)


def test_compute_portfolio_metrics_known_vector() -> None:
    """Verifies loss-removal and excess-load arithmetic on a known vector."""
    claims = pd.Series([100.0, 200.0, 1000.0, 2000.0])
    metrics = compute_portfolio_metrics(claims, cap=500.0)
    assert metrics["losses_removed"] == pytest.approx(2000.0)
    assert metrics["loss_removed_pct"] == pytest.approx(2000.0 / 3300.0 * 100.0)
    assert metrics["claims_truncated"] == 2
    assert metrics["claims_truncated_pct"] == pytest.approx(50.0)
    assert metrics["excess_load"] == pytest.approx(3300.0 / 1300.0)


def test_compute_portfolio_metrics_no_truncation() -> None:
    """Verifies a cap above all claims yields zero removal and unit load."""
    claims = pd.Series([100.0, 200.0])
    metrics = compute_portfolio_metrics(claims, cap=1000.0)
    assert metrics["losses_removed"] == pytest.approx(0.0)
    assert metrics["claims_truncated"] == 0
    assert metrics["excess_load"] == pytest.approx(1.0)


def test_spearman_rank_corr_perfect_and_imperfect() -> None:
    """Verifies Spearman correlation is 1 for monotone ranks and lower otherwise."""
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman_rank_corr(a, a * 2.0) == pytest.approx(1.0)
    b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    c = pd.Series([1.0, 2.0, 3.0, 4.0, 99.0])
    assert spearman_rank_corr(b, c) < 1.0


def _policy_segments() -> pd.DataFrame:
    """Builds a small two-segment policy mapping."""
    return pd.DataFrame(
        {
            "PolicyID": [1, 2, 3, 4],
            "segment_id": ["S1", "S1", "S2", "S2"],
        }
    )


def test_excess_concentration_identifies_dominant_segment() -> None:
    """Verifies the concentration ratio flags the segment holding the excess."""
    train = pd.DataFrame(
        {
            "PolicyID": [1, 2, 3, 4],
            "ClaimAmount": [500.0, 20000.0, 500.0, 500.0],
            "exposure_capped": [1.0, 1.0, 1.0, 1.0],
        }
    )
    concentration = excess_concentration(train, _policy_segments(), cap=1000.0)
    assert concentration.max_excess_segment == "S1"
    assert concentration.max_excess_ratio > 1.0


def test_select_cap_lowest_satisfying_cap() -> None:
    """Verifies the lowest cap whose transition passes is selected."""
    caps = [10.0, 20.0, 30.0]
    spearman = [0.99, 0.99]
    deviance_change = [0.005, 0.005]
    cap, index = select_cap(caps, spearman, deviance_change, 0.98, 0.01)
    assert cap == 10.0
    assert index == 0


def test_select_cap_skips_failing_first_transition() -> None:
    """Verifies a failing first transition advances to the next candidate."""
    caps = [10.0, 20.0, 30.0]
    spearman = [0.90, 0.99]
    deviance_change = [0.005, 0.005]
    cap, index = select_cap(caps, spearman, deviance_change, 0.98, 0.01)
    assert cap == 20.0
    assert index == 1


def test_select_cap_falls_back_to_highest() -> None:
    """Verifies fallback to the highest cap when no transition passes."""
    caps = [10.0, 20.0, 30.0]
    spearman = [0.90, 0.90]
    deviance_change = [0.05, 0.05]
    cap, index = select_cap(caps, spearman, deviance_change, 0.98, 0.01)
    assert cap == 30.0
    assert index == 2
