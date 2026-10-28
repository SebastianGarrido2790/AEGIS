"""Typed schemas for the Stage 10 final-seed validation summary (Roadmap Exit E1).

The summary is the persisted evidence for ADR-013 / P2-D6: per-seed metrics on
held-out policies, the worst-seed and median-seed gate statistics against the frozen
Stage 9 thresholds, and cryptographic hashes of the frozen configuration and test data.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aegis.schemas.validation import ValidationMetrics

ARTIFACT_VERSION = "1.0"


class MetricGateEvaluation(BaseModel):
    """A single metric compared against its frozen floor for one seed or statistic."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    direction: Literal["higher_is_better", "lower_is_better"]
    value: float
    threshold: float
    margin: float = Field(description="Signed distance from the frozen floor or ceiling.")
    passes_floor: bool
    target: float | None = None
    meets_target: bool | None = None

    @model_validator(mode="after")
    def _validate_consistency(self) -> MetricGateEvaluation:
        floor_expected = (
            self.value >= self.threshold
            if self.direction == "higher_is_better"
            else self.value <= self.threshold
        )
        if floor_expected != self.passes_floor:
            raise ValueError(f"passes_floor for '{self.metric}' is inconsistent")
        if self.target is not None:
            target_expected = (
                self.value >= self.target
                if self.direction == "higher_is_better"
                else self.value <= self.target
            )
            if target_expected != self.meets_target:
                raise ValueError(f"meets_target for '{self.metric}' is inconsistent")
        return self


class FinalSeedRunResult(BaseModel):
    """One final seed's execution record: held-out metrics and gate outcome."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    dgp_noise_seed: int
    metrics: ValidationMetrics
    gates: dict[str, MetricGateEvaluation]
    held_out_policy_count: int = Field(gt=0)
    held_out_policy_ids: list[str] = Field(min_length=1)
    data_sha256: dict[str, str] = Field(default_factory=dict)
    passes_frozen_thresholds: bool
    segment_count: int = Field(gt=0)

    @field_validator("held_out_policy_ids")
    @classmethod
    def _validate_unique_ids(cls, ids: list[str]) -> list[str]:
        if len(set(ids)) != len(ids):
            raise ValueError("held_out_policy_ids must be unique")
        return ids


class FinalSeedGateSummary(BaseModel):
    """Campaign-level gate statistics: worst seed and median seed (P2-D6)."""

    model_config = ConfigDict(extra="forbid")

    k_seeds: int = Field(ge=5, description="Exit E1 requires at least five final seeds.")
    failed_seeds: list[int]
    worst_seed_name: int | None
    median_seed_name: int
    worst_seed: dict[str, MetricGateEvaluation]
    median_seed: dict[str, MetricGateEvaluation]
    worst_seed_all_floors_cleared: bool
    median_seed_targets_met: bool
    all_seeds_pass_frozen_thresholds: bool
    campaign_passed: bool
    seeds_at_or_past_target: list[int]


class FinalSeedValidationSummary(BaseModel):
    """The persisted ``artifacts/causal_validation_summary.json`` artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact_version: Literal["1.0"]
    generated_at: str
    final_seeds: list[int] = Field(min_length=5)
    pilot_seeds: list[int]
    seed_semantics: dict[str, str]
    seed_offset_noise: int = 0
    sample_size: int = Field(gt=0)
    test_size: float = Field(gt=0.0, lt=1.0)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_sha256: dict[str, str]
    held_out_ids_sha256: dict[int, str]
    thresholds: dict[str, float]
    metrics_by_seed: dict[int, dict[str, float]]
    segment_counts_by_seed: dict[int, int]
    gates_by_seed: dict[int, dict[str, dict[str, object]]]
    campaign_gates: FinalSeedGateSummary
    tautology_checks: dict[str, bool]

    @model_validator(mode="after")
    def _validate_consistency(self) -> FinalSeedValidationSummary:
        seeds = set(self.final_seeds)
        if len(seeds) != len(self.final_seeds):
            raise ValueError("final_seeds must be unique")
        pilot_overlap = seeds & set(self.pilot_seeds)
        if pilot_overlap:
            raise ValueError(f"final seeds overlap pilot seeds: {sorted(pilot_overlap)}")
        for name, mapping in (
            ("metrics_by_seed", self.metrics_by_seed),
            ("segment_counts_by_seed", self.segment_counts_by_seed),
            ("gates_by_seed", self.gates_by_seed),
            ("held_out_ids_sha256", self.held_out_ids_sha256),
        ):
            if set(mapping) != seeds:
                raise ValueError(f"{name} must cover exactly the final seeds")
        if any(self.tautology_checks.values()):
            raise ValueError("tautology_checks must all be False")
        for seed, metrics in self.metrics_by_seed.items():
            for value in metrics.values():
                if not math.isfinite(value):
                    raise ValueError(f"metric for seed {seed} is not finite")
        return self
