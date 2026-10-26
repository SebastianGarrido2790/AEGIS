"""Unit tests for semi-synthetic DGP generator and self-tests (P2-F04)."""

import numpy as np
import pandas as pd
import pytest

from aegis.pipelines.synthetic_dgp import (
    assign_confounded_treatment,
    compute_baseline_log_odds,
    compute_price_sensitivity_slope,
    run_dgp_self_tests,
    sigmoid,
)
from aegis.utils.exceptions import DGPVerificationError


def test_sigmoid() -> None:
    assert sigmoid(0.0) == 0.5
    assert np.isclose(sigmoid(500.0), 1.0)
    assert np.isclose(sigmoid(-500.0), 0.0)


def test_baseline_log_odds() -> None:
    df = pd.DataFrame(
        {
            "driver_age_band_ord": [0, 1, 2, 3],
            "car_age_band_ord": [0, 1, 2, 0],
        }
    )
    a_x = compute_baseline_log_odds(df, base=1.4, driver_age_ge_25=0.3, car_age_lt_10=0.2)
    # Row 0: age 0 (<1) -> 0, car 0 (<=1) -> 0.2 => 1.4 + 0.2 = 1.6
    assert np.isclose(a_x[0], 1.6)
    # Row 1: age 1 (>=1) -> 0.3, car 1 (<=1) -> 0.2 => 1.4 + 0.3 + 0.2 = 1.9
    assert np.isclose(a_x[1], 1.9)
    # Row 2: age 2 (>=1) -> 0.3, car 2 (>1) -> 0 => 1.4 + 0.3 = 1.7
    assert np.isclose(a_x[2], 1.7)


def test_price_sensitivity_slope() -> None:
    df = pd.DataFrame(
        {
            "driver_age_band_ord": [0, 1],
            "car_age_band_ord": [0, 1],
        }
    )
    b_x = compute_price_sensitivity_slope(
        df, base=3.5, driver_age_ord=-0.4, car_age_ord=0.3, interaction=-0.1
    )
    # All slopes should be negative
    assert (b_x < 0).all()
    # Row 0: -(3.5 + 0 + 0 + 0) = -3.5
    assert np.isclose(b_x[0], -3.5)
    # Row 1: -(3.5 - 0.4*1 + 0.3*1 - 0.1*1*1) = -(3.3) = -3.3
    assert np.isclose(b_x[1], -3.3)


def test_assign_confounded_treatment() -> None:
    rng = np.random.default_rng(42)
    loss_cost = np.array([50.0, 100.0, 150.0, 500.0, 1000.0] * 100)
    treatment, mu_t = assign_confounded_treatment(
        loss_cost=loss_cost,
        rng=rng,
        confounding_dial=0.5,
        residual_std=0.05,
        corridor_width=0.15,
    )
    assert len(treatment) == len(loss_cost)
    assert (treatment >= -0.15).all()
    assert (treatment <= 0.15).all()
    assert (mu_t >= -0.15).all()
    assert (mu_t <= 0.15).all()


def _make_mock_dgp_data(
    n: int = 1000,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "PolicyID": np.arange(1, n + 1),
            "driver_age_band_ord": rng.choice([0, 1, 2, 3], size=n),
            "car_age_band_ord": rng.choice([0, 1, 2], size=n),
            "fitted_loss_cost": rng.exponential(scale=100.0, size=n),
        }
    )
    df["segment_id"] = (
        "D:" + df["driver_age_band_ord"].astype(str) + "_C:" + df["car_age_band_ord"].astype(str)
    )

    treatment, mu_t = assign_confounded_treatment(
        loss_cost=df["fitted_loss_cost"].to_numpy(),
        rng=rng,
        confounding_dial=0.5,
        residual_std=0.05,
        corridor_width=0.15,
    )
    a_x = compute_baseline_log_odds(df)
    b_x = compute_price_sensitivity_slope(df)
    q0 = np.asarray(sigmoid(a_x), dtype=float)
    tau = q0 * (1.0 - q0) * b_x
    epsilon = (1.0 - q0) * b_x
    return df, treatment, mu_t, q0, tau, epsilon, a_x, b_x


def test_dgp_self_tests_pass_on_valid_data() -> None:
    df, treatment, mu_t, q0, tau, epsilon, a_x, b_x = _make_mock_dgp_data(n=2000)
    results = run_dgp_self_tests(
        df=df,
        treatment=treatment,
        mu_t=mu_t,
        q0=q0,
        tau=tau,
        epsilon=epsilon,
        a_x=a_x,
        b_x=b_x,
        corridor_width=0.15,
        estimand_gap_tolerance=0.05,
        overlap_floor_var=0.001,
        dispersion_min_std=0.005,
        homogeneity_max_corr=0.05,
        elasticity_bound_product=1.0,
    )
    assert len(results) == 5
    for name, res in results.items():
        assert res.passed is True, f"Self-test {name} failed"


def test_self_test_overlap_fails_on_zero_variance() -> None:
    df, treatment, mu_t, q0, tau, epsilon, a_x, b_x = _make_mock_dgp_data(n=500)
    # Forcing treatment to equal mu_t exactly (residual variance = 0)
    with pytest.raises(DGPVerificationError) as exc_info:
        run_dgp_self_tests(
            df=df,
            treatment=mu_t,  # residual variance is 0.0
            mu_t=mu_t,
            q0=q0,
            tau=tau,
            epsilon=epsilon,
            a_x=a_x,
            b_x=b_x,
            overlap_floor_var=0.001,
        )
    assert "Overlap self-test failed" in str(exc_info.value)


def test_self_test_bound_guarantee_fails_when_exceeded() -> None:
    df, treatment, mu_t, q0, tau, epsilon, a_x, b_x = _make_mock_dgp_data(n=500)
    # Artificially set an extreme elasticity
    bad_epsilon = epsilon.copy()
    bad_epsilon[0] = -10.0  # -10 * 0.15 = 1.5 >= 1.0
    with pytest.raises(DGPVerificationError) as exc_info:
        run_dgp_self_tests(
            df=df,
            treatment=treatment,
            mu_t=mu_t,
            q0=q0,
            tau=tau,
            epsilon=bad_epsilon,
            a_x=a_x,
            b_x=b_x,
            homogeneity_max_corr=1.0,  # Isolate bound check
            elasticity_bound_product=1.0,
        )
    assert "Elasticity bound self-test failed" in str(exc_info.value)

