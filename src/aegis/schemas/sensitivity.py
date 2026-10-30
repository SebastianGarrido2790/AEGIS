"""Pydantic schemas for Hidden-Confounder Sensitivity Analysis (P2-F08, Stage 11).

Governs ADR-013, P2-D7, and Showcase Scenario S4 parameter locking.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SensitivityGridCell(BaseModel):
    """Specification of a single hidden-confounder grid cell."""

    model_config = ConfigDict(extra="forbid")

    cell_id: str = Field(description="Unique cell identifier, e.g. 'gt_0.05_gy_0.50'")
    gamma_t: float = Field(ge=0.0, description="Coupling strength to treatment assignment T")
    gamma_y: float = Field(ge=0.0, description="Coupling strength to renewal log-odds Y")
    is_baseline: bool = Field(default=False, description="True if gamma_t == 0 and gamma_y == 0")


class SensitivityCellResult(BaseModel):
    """Evaluated recovery metrics for a single sensitivity grid cell."""

    model_config = ConfigDict(extra="forbid")

    cell_id: str
    gamma_t: float
    gamma_y: float
    confounding_strength: float = Field(
        description="Euclidean confounding norm: sqrt(gamma_t^2 + gamma_y^2)"
    )
    is_baseline: bool
    fit_duration_seconds: float = Field(ge=0.0)
    m1_tau_spearman: float = Field(ge=-1.0, le=1.0)
    m1_epsilon_spearman: float = Field(ge=-1.0, le=1.0)
    m1b_policy_spearman: float = Field(ge=-1.0, le=1.0)
    m2_mean_abs_bias: float = Field(ge=0.0)
    m3_tau_coverage: float = Field(ge=0.0, le=1.0)
    m4_self_consistency: float = Field(ge=0.0, le=1.0)
    m5_sign_agreement: float = Field(ge=0.0, le=1.0)
    degradation_m1_tau: float = Field(
        description="m1_tau relative to baseline: m1_cell - m1_baseline"
    )
    degradation_m2_bias: float = Field(
        description="m2 relative to baseline: m2_cell - m2_baseline"
    )


class ScenarioS4Selection(BaseModel):
    """Parameters and metrics for the representative degraded case feeding Showcase Scenario S4."""

    model_config = ConfigDict(extra="forbid")

    cell_id: str
    gamma_t: float
    gamma_y: float
    confounding_strength: float
    m1_tau_spearman: float
    m1_epsilon_spearman: float
    m2_mean_abs_bias: float
    rationale: str


class SensitivitySummary(BaseModel):
    """Cryptographically-hashed summary of the full 10-cell sensitivity analysis."""

    model_config = ConfigDict(extra="forbid")

    sample_size: int = Field(gt=0)
    random_seed: int
    grid_cells_total: int = Field(default=10, description="Total expected grid cells")
    grid_cells_completed: int = Field(description="Successfully executed cells")
    zero_skips_verified: bool = Field(
        description="Asserts that zero cells were skipped or silently dropped"
    )
    baseline_metrics: SensitivityCellResult
    grid_results: list[SensitivityCellResult]
    robustness_value: float = Field(
        description="Minimum confounding norm at which m1_tau breaches robustness_threshold"
    )
    robustness_threshold: float = Field(
        default=0.70,
        description="Segment rank correlation floor defining robustness"
    )
    scenario_s4: ScenarioS4Selection
    config_hash: str
    timestamp: str
