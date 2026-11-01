"""Typed response contract for the Phase 2 diagnostic showcase (P2-F11).

The showcase is a read-only view of the ``approved`` Tier 1 artifact. Under the
ADR-009 Phase 2 exemption it may expose baseline elasticity, interval uncertainty,
and validation evidence only: no proposed price change, no rate publication (INV-4,
INV-5). ``extra="forbid"`` plus the shared forbidden-fragment guard make an
adjustment field structurally unrepresentable.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aegis.schemas.segments import Tier1SegmentOutput, guard_forbidden_fields

DISCLAIMER = (
    "Model validation demo on semi-synthetic data - not a pricing recommendation."
)


class StressCell(BaseModel):
    """One frozen Stage 11 hidden-confounder sensitivity cell (global, not a segment)."""

    model_config = ConfigDict(extra="forbid")

    cell_id: str = Field(min_length=1)
    gamma_t: float
    gamma_y: float
    confounding_strength: float
    m1_tau_spearman: float
    m1_epsilon_spearman: float
    m2_mean_abs_bias: float
    rationale: str = Field(min_length=1)


class ShowcaseView(BaseModel):
    """Everything one rendered showcase page may display."""

    model_config = ConfigDict(extra="forbid")

    scenario_key: str = Field(min_length=1)
    scenario_name: str = Field(min_length=1)
    subtitle: str = ""
    risk_profile: str = ""
    governance_state: str = ""
    governance_note: str = ""
    source_kind: Literal["segment", "stress"]
    segment: Tier1SegmentOutput | None = None
    stress: StressCell | None = None
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    validation_status: str = Field(min_length=1)
    segment_count: int = Field(ge=0)
    notice: str | None = None
    disclaimer: str = DISCLAIMER

    @model_validator(mode="before")
    @classmethod
    def _forbid_adjustment_fields(cls, data: Any) -> Any:
        """Rejects any field naming a proposed price change before validation."""
        guard_forbidden_fields(data)
        return data

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ShowcaseView:
        """Each view renders either a segment row or a stress cell, never both."""
        if (self.segment is None) == (self.stress is None):
            raise ValueError("showcase view must carry exactly one of segment or stress")
        return self
