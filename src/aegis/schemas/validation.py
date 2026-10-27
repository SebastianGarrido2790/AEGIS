"""Typed schemas for offline validation metrics and Q3 threshold calibration (P2-F07).

These schemas define the contracts for evaluation metrics (m1 through m5), negative
and positive controls (c1, c2, c3), separation gaps, and frozen validation thresholds
under ADR-013 and P2-D6a.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aegis.schemas.segments import SEGMENT_ID_PATTERN


class ValidationMetrics(BaseModel):
    """Container for the six approved causal validation metrics."""

    model_config = ConfigDict(extra="forbid")

    m1_tau_spearman: float = Field(
        ge=-1.0, le=1.0, description="Segment-level tau Spearman correlation."
    )
    m1_epsilon_spearman: float = Field(
        ge=-1.0, le=1.0, description="Segment-level epsilon Spearman correlation."
    )
    m1b_policy_spearman: float = Field(
        ge=-1.0, le=1.0, description="Policy-level tau Spearman correlation on held-out test data."
    )
    m2_mean_abs_bias: float = Field(
        ge=0.0, description="Mean absolute error across segments on elasticity."
    )
    m3_tau_coverage: float = Field(
        ge=0.0, le=1.0, description="Empirical 95% CI coverage of true segment tau."
    )
    m3_epsilon_coverage: float = Field(
        ge=0.0, le=1.0, description="Empirical 95% CI coverage of true segment epsilon."
    )
    m4_self_consistency: float = Field(
        ge=0.0, le=1.0, description="Proportion of finite intervals that contain point estimates."
    )
    m5_sign_agreement: float = Field(
        ge=0.0, le=1.0, description="Proportion of segments with matching tau signs."
    )

    @model_validator(mode="after")
    def _validate_finite(self) -> ValidationMetrics:
        for field_name, value in self.__dict__.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Metric {field_name} must be finite; got {value}")
        return self


class SegmentValidationRow(BaseModel):
    """Detailed validation diagnostics for a single segment."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(pattern=SEGMENT_ID_PATTERN)
    policy_count: int = Field(gt=0)
    tau_hat: float
    tau_true: float
    tau_ci_lower: float
    tau_ci_upper: float
    tau_covered: bool
    epsilon_hat: float
    epsilon_true: float
    epsilon_ci_lower: float
    epsilon_ci_upper: float
    epsilon_covered: bool
    baseline_q0_hat: float = Field(gt=0.0, le=1.0)
    baseline_q0_true: float = Field(gt=0.0, le=1.0)
    sign_agreed: bool
    abs_bias: float = Field(ge=0.0)


class ValidationMetricReport(BaseModel):
    """Validation report for a single estimator execution."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    sample_size: int = Field(gt=0)
    metrics: ValidationMetrics
    segment_details: list[SegmentValidationRow] = Field(default_factory=list)


class ControlRunResult(BaseModel):
    """Execution outcome for a negative or positive control."""

    model_config = ConfigDict(extra="forbid")

    control_id: Literal["c1_unadjusted", "c2_shuffled", "c3_zero_effect", "positive_control"]
    control_name: str
    is_negative_control: bool
    metrics: ValidationMetrics
    failed_gate_as_expected: bool
    diagnostic_message: str


class SeparationGap(BaseModel):
    """Separation analysis between estimator lower tail and control upper tail."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    direction: Literal["higher_is_better", "lower_is_better"]
    estimator_min: float
    control_max: float
    gap: float
    min_margin: float
    separated: bool
    recommended_threshold: float


class ValidationThresholds(BaseModel):
    """Frozen numeric gate thresholds calibrated by Q3 (P2-D6a)."""

    model_config = ConfigDict(extra="forbid")

    m1_tau_spearman_floor: float = Field(ge=-1.0, le=1.0)
    m1_tau_spearman_target: float = Field(ge=-1.0, le=1.0)
    m1_epsilon_spearman_floor: float = Field(ge=-1.0, le=1.0)
    m1_epsilon_spearman_target: float = Field(ge=-1.0, le=1.0)
    m1b_policy_spearman_floor: float = Field(ge=-1.0, le=1.0)
    m1b_policy_spearman_target: float = Field(ge=-1.0, le=1.0)
    m2_mean_abs_bias_ceiling: float = Field(ge=0.0)
    m2_mean_abs_bias_target: float = Field(ge=0.0)
    m3_tau_coverage_floor: float = Field(ge=0.0, le=1.0)
    m3_epsilon_coverage_floor: float = Field(ge=0.0, le=1.0)
    m4_self_consistency_floor: float = Field(ge=0.0, le=1.0)
    m5_sign_agreement_floor: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_ordering(self) -> ValidationThresholds:
        if self.m1_tau_spearman_floor > self.m1_tau_spearman_target:
            raise ValueError("m1_tau_spearman_floor must not exceed target")
        if self.m1_epsilon_spearman_floor > self.m1_epsilon_spearman_target:
            raise ValueError("m1_epsilon_spearman_floor must not exceed target")
        if self.m1b_policy_spearman_floor > self.m1b_policy_spearman_target:
            raise ValueError("m1b_policy_spearman_floor must not exceed target")
        if self.m2_mean_abs_bias_target > self.m2_mean_abs_bias_ceiling:
            raise ValueError("m2_mean_abs_bias_target must not exceed ceiling")
        return self


class Q3CalibrationReport(BaseModel):
    """Complete Qualification Q3 threshold calibration summary artifact."""

    model_config = ConfigDict(extra="forbid")

    pilot_seeds: list[int]
    estimator_runs: list[ValidationMetricReport]
    negative_control_results: list[ControlRunResult]
    positive_control_result: ControlRunResult
    separation_gaps: dict[str, SeparationGap]
    recommended_thresholds: ValidationThresholds
    all_negative_controls_failed: bool
    positive_control_passed: bool
    clear_separation_achieved: bool
