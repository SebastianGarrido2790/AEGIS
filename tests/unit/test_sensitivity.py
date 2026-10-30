"""Unit tests for Hidden-Confounder Sensitivity Analysis (P2-F08, Stage 11).

Governs ADR-013, P2-D7, and fail-loud zero-skips assertions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis.config.loader import get_config
from aegis.pipelines.cli import build_parser
from aegis.pipelines.sensitivity import (
    build_sensitivity_grid,
    compute_robustness_value,
    inject_hidden_confounder,
    select_scenario_s4,
)
from aegis.schemas.sensitivity import (
    ScenarioS4Selection,
    SensitivityCellResult,
    SensitivitySummary,
)
from aegis.utils.exceptions import SensitivityAnalysisError


def test_build_sensitivity_grid_count_and_baseline() -> None:
    """Verifies that the sensitivity grid generates exactly 10 cells with baseline first."""
    gamma_t = [0.02, 0.05, 0.10]
    gamma_y = [0.2, 0.5, 1.0]

    grid = build_sensitivity_grid(gamma_t, gamma_y)
    assert len(grid) == 10

    # Baseline cell
    baseline = grid[0]
    assert baseline.cell_id == "gt_0.00_gy_0.00"
    assert baseline.gamma_t == 0.0
    assert baseline.gamma_y == 0.0
    assert baseline.is_baseline is True

    # 9 non-zero cells
    for cell in grid[1:]:
        assert cell.is_baseline is False
        assert cell.gamma_t in gamma_t
        assert cell.gamma_y in gamma_y


def test_build_sensitivity_grid_invalid_levels_raises() -> None:
    """Asserts that invalid grid level lengths raise SensitivityAnalysisError loudly."""
    with pytest.raises(SensitivityAnalysisError):
        build_sensitivity_grid([0.02, 0.05], [0.2, 0.5, 1.0])


def test_inject_hidden_confounder_isolation() -> None:
    """Verifies confounder U is injected but strictly excluded from feature columns."""
    config = get_config()
    dgp_cfg = config.tier1_ml.synthetic_dgp

    n = 100
    forest_df = pd.DataFrame(
        {
            "PolicyID": np.arange(1, n + 1),
            "driver_age_clipped": np.full(n, 35.0),
            "car_age_clipped": np.full(n, 5.0),
            "log_density": np.full(n, 4.0),
            "driver_age_band_ord": np.full(n, 1),
            "car_age_band_ord": np.full(n, 1),
        }
    )
    seg_df = pd.DataFrame(
        {
            "PolicyID": np.arange(1, n + 1),
            "segment_id": ["D:25-44_C:5-9_Dens:<=500"] * n,
        }
    )
    glm_df = pd.DataFrame(
        {
            "PolicyID": np.arange(1, n + 1),
            "fitted_loss_cost_per_year": np.full(n, 150.0),
        }
    )

    df_res, feature_cols = inject_hidden_confounder(
        forest_df=forest_df,
        seg_df=seg_df,
        glm_df=glm_df,
        dgp_cfg=dgp_cfg,
        gamma_t=0.05,
        gamma_y=0.5,
        rng_seed=42,
    )

    # U must not exist anywhere in the resulting columns or features
    assert "u" not in df_res.columns
    assert "U" not in df_res.columns
    assert "unobserved_confounder" not in df_res.columns
    assert "u" not in feature_cols
    assert "U" not in feature_cols

    # Essential columns must be present
    assert "treatment_rate_change" in df_res.columns
    assert "renewed" in df_res.columns
    assert "segment_id" in df_res.columns
    assert len(df_res) == n
    assert set(df_res["renewed"].unique()).issubset({0, 1})


def test_compute_robustness_value_breach() -> None:
    """Verifies that robustness value computes the minimum norm of breaching cells."""
    results = [
        SensitivityCellResult(
            cell_id="gt_0.00_gy_0.00",
            gamma_t=0.0,
            gamma_y=0.0,
            confounding_strength=0.0,
            is_baseline=True,
            fit_duration_seconds=10.0,
            m1_tau_spearman=0.90,
            m1_epsilon_spearman=0.92,
            m1b_policy_spearman=0.88,
            m2_mean_abs_bias=0.05,
            m3_tau_coverage=1.0,
            m4_self_consistency=1.0,
            m5_sign_agreement=1.0,
            degradation_m1_tau=0.0,
            degradation_m2_bias=0.0,
        ),
        SensitivityCellResult(
            cell_id="gt_0.02_gy_0.20",
            gamma_t=0.02,
            gamma_y=0.20,
            confounding_strength=round(math.sqrt(0.02**2 + 0.20**2), 4),
            is_baseline=False,
            fit_duration_seconds=10.0,
            m1_tau_spearman=0.80,
            m1_epsilon_spearman=0.82,
            m1b_policy_spearman=0.78,
            m2_mean_abs_bias=0.06,
            m3_tau_coverage=1.0,
            m4_self_consistency=1.0,
            m5_sign_agreement=1.0,
            degradation_m1_tau=-0.10,
            degradation_m2_bias=0.01,
        ),
        SensitivityCellResult(
            cell_id="gt_0.05_gy_0.50",
            gamma_t=0.05,
            gamma_y=0.50,
            confounding_strength=round(math.sqrt(0.05**2 + 0.50**2), 4),
            is_baseline=False,
            fit_duration_seconds=10.0,
            m1_tau_spearman=0.65,  # Breaches 0.70 threshold
            m1_epsilon_spearman=0.68,
            m1b_policy_spearman=0.66,
            m2_mean_abs_bias=0.09,
            m3_tau_coverage=1.0,
            m4_self_consistency=1.0,
            m5_sign_agreement=1.0,
            degradation_m1_tau=-0.25,
            degradation_m2_bias=0.04,
        ),
    ]

    r_val = compute_robustness_value(results, robustness_threshold=0.70)
    expected_norm = round(math.sqrt(0.05**2 + 0.50**2), 4)
    assert r_val == expected_norm


def test_select_scenario_s4_success_and_missing() -> None:
    """Verifies that select_scenario_s4 finds target cell or raises loudly."""
    results = [
        SensitivityCellResult(
            cell_id="gt_0.05_gy_0.50",
            gamma_t=0.05,
            gamma_y=0.50,
            confounding_strength=0.5025,
            is_baseline=False,
            fit_duration_seconds=10.0,
            m1_tau_spearman=0.65,
            m1_epsilon_spearman=0.68,
            m1b_policy_spearman=0.66,
            m2_mean_abs_bias=0.09,
            m3_tau_coverage=1.0,
            m4_self_consistency=1.0,
            m5_sign_agreement=1.0,
            degradation_m1_tau=-0.25,
            degradation_m2_bias=0.04,
        ),
    ]

    s4 = select_scenario_s4(results, "gt_0.05_gy_0.50")
    assert s4.cell_id == "gt_0.05_gy_0.50"
    assert s4.gamma_t == 0.05
    assert s4.gamma_y == 0.50
    assert s4.m1_tau_spearman == 0.65

    with pytest.raises(SensitivityAnalysisError):
        select_scenario_s4(results, "non_existent_cell")


def test_sensitivity_summary_pydantic_roundtrip() -> None:
    """Verifies full serialization and deserialization of SensitivitySummary."""
    base_res = SensitivityCellResult(
        cell_id="gt_0.00_gy_0.00",
        gamma_t=0.0,
        gamma_y=0.0,
        confounding_strength=0.0,
        is_baseline=True,
        fit_duration_seconds=15.0,
        m1_tau_spearman=0.90,
        m1_epsilon_spearman=0.92,
        m1b_policy_spearman=0.88,
        m2_mean_abs_bias=0.05,
        m3_tau_coverage=1.0,
        m4_self_consistency=1.0,
        m5_sign_agreement=1.0,
        degradation_m1_tau=0.0,
        degradation_m2_bias=0.0,
    )
    s4 = ScenarioS4Selection(
        cell_id="gt_0.05_gy_0.50",
        gamma_t=0.05,
        gamma_y=0.50,
        confounding_strength=0.5025,
        m1_tau_spearman=0.65,
        m1_epsilon_spearman=0.68,
        m2_mean_abs_bias=0.09,
        rationale="Showcase S4 test selection",
    )

    summary = SensitivitySummary(
        sample_size=100000,
        random_seed=42,
        grid_cells_total=10,
        grid_cells_completed=10,
        zero_skips_verified=True,
        baseline_metrics=base_res,
        grid_results=[base_res],
        robustness_value=0.5025,
        robustness_threshold=0.70,
        scenario_s4=s4,
        config_hash="abc123hash",
        timestamp="2026-10-30T12:00:00+00:00",
    )

    raw_json = summary.model_dump_json()
    reloaded = SensitivitySummary.model_validate_json(raw_json)
    assert reloaded.sample_size == 100000
    assert reloaded.zero_skips_verified is True
    assert reloaded.robustness_value == 0.5025


def test_cli_run_sensitivity_parser() -> None:
    """Verifies that the CLI parser accepts run-sensitivity arguments."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-sensitivity",
            "--summary-path",
            "artifacts/test_summary.json",
            "--report-path",
            "reports/test_report.md",
        ]
    )
    assert args.command == "run-sensitivity"
    assert args.summary_path == Path("artifacts/test_summary.json")
    assert args.report_path == Path("reports/test_report.md")
