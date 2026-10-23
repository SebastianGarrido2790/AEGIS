"""Typed contracts for Tier 1 actuarial segments (P2-F05, ADR-014).

These models define the published segment interface consumed by Phases 3, 6, and the
showcase. `Tier1SegmentOutput` structurally forbids proposed price adjustments: Tier 1
reports baseline loss cost and elasticity only (INV-4, INV-5). Redefined adjustment fields
cannot enter because unknown keys are rejected and forbidden name fragments are screened.
"""

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aegis.schemas.features import SHA256_PATTERN

SEGMENT_ID_PATTERN = r"^D:[^_]+_C:[^_]+_Dens:\S+$"
DENSITY_LABEL_PATTERN = r"^\S+$"

FORBIDDEN_FIELD_FRAGMENTS: tuple[str, ...] = (
    "delta",
    "adjust",
    "proposed",
    "new_premium",
)


def _guard_forbidden_fields(payload: Any) -> None:
    """Raises if any incoming key names a proposed pricing adjustment.

    Args:
        payload: Raw input mapping passed to a Tier 1 contract.

    Raises:
        ValueError: If a key contains a forbidden adjustment fragment.
    """
    if not isinstance(payload, dict):
        return
    for key in payload:
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in FORBIDDEN_FIELD_FRAGMENTS):
            raise ValueError(
                f"field '{key}' is forbidden in Tier 1 outputs: "
                "no proposed adjustments may leave the deterministic baseline"
            )


class Tier1SegmentOutput(BaseModel):
    """Per-segment Tier 1 contract: baseline risk and elasticity, never a price proposal."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(pattern=SEGMENT_ID_PATTERN)
    policy_count: int = Field(ge=1000)
    exposure_years: float = Field(gt=0.0)
    baseline_retention_q0: float = Field(ge=0.0, le=1.0)
    tau_hat: float
    epsilon_hat: float
    epsilon_ci_lower: float
    epsilon_ci_upper: float
    loss_cost_per_year: float = Field(ge=0.0)
    base_premium_per_year: float = Field(ge=0.0)
    model_version: str
    data_version: str
    semi_synthetic: bool

    @model_validator(mode="before")
    @classmethod
    def _forbid_adjustment_fields(cls, data: Any) -> Any:
        """Rejects any field naming a proposed adjustment before validation."""
        _guard_forbidden_fields(data)
        return data

    @model_validator(mode="after")
    def _validate_values(self) -> "Tier1SegmentOutput":
        """Enforces finiteness, interval self-consistency, and semi-synthetic provenance."""
        values = (
            self.baseline_retention_q0,
            self.tau_hat,
            self.epsilon_hat,
            self.epsilon_ci_lower,
            self.epsilon_ci_upper,
            self.loss_cost_per_year,
            self.base_premium_per_year,
            self.exposure_years,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("segment metrics must be finite")
        if self.epsilon_ci_lower > self.epsilon_ci_upper:
            raise ValueError("epsilon_ci_lower must not exceed epsilon_ci_upper")
        if not self.epsilon_ci_lower <= self.epsilon_hat <= self.epsilon_ci_upper:
            raise ValueError("confidence interval must enclose the elasticity point estimate")
        if not self.semi_synthetic:
            raise ValueError("Tier 1 segment outputs must be explicitly semi-synthetic")
        return self


class Tier1SegmentTable(BaseModel):
    """The full set of published Tier 1 segments for one model/data version."""

    model_config = ConfigDict(extra="forbid")

    segments: list[Tier1SegmentOutput]

    @model_validator(mode="before")
    @classmethod
    def _forbid_adjustment_fields(cls, data: Any) -> Any:
        """Rejects adjustment-like top-level keys before validation."""
        _guard_forbidden_fields(data)
        return data

    @model_validator(mode="after")
    def _unique_segment_ids(self) -> "Tier1SegmentTable":
        """Rejects duplicate segment identifiers."""
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate segment_id in Tier 1 segment table")
        return self


class SegmentDefinition(BaseModel):
    """One realized segment row: bands, realized size, and baseline economics."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(pattern=SEGMENT_ID_PATTERN)
    driver_age_band: str
    car_age_band: str
    density_band: str = Field(pattern=DENSITY_LABEL_PATTERN)
    is_merged: bool
    policy_count: int = Field(ge=1000)
    exposure_years: float = Field(gt=0.0)
    loss_cost_per_year: float = Field(ge=0.0)
    base_premium_per_year: float = Field(ge=0.0)


class CellMergeRecord(BaseModel):
    """Audit record for one approved sub-floor cell merge."""

    model_config = ConfigDict(extra="forbid")

    driver_age_band: str
    car_age_band: str
    from_density_band: str
    to_density_band: str
    merged_density_band: str
    merged_policy_count: int = Field(ge=1)


class SegmentMetadata(BaseModel):
    """Versioned metadata artifact for the segment pipeline."""

    model_config = ConfigDict(extra="forbid")

    segment_count: int = Field(ge=1)
    initial_cell_count: int = Field(ge=1)
    min_policy_count: int = Field(ge=1)
    premium_loading: float = Field(ge=0.0)
    merged_cells: list[CellMergeRecord]
    total_policies: int = Field(ge=0)
    total_exposure_years: float = Field(gt=0.0)
    portfolio_loss_cost_per_year: float = Field(ge=0.0)
    portfolio_base_premium_per_year: float = Field(ge=0.0)
    config_fingerprint: str = Field(pattern=SHA256_PATTERN)
