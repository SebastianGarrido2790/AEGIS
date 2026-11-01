"""Curated preset scenarios for the Phase 2 diagnostic showcase (P2-F11, P2-D12).

Presets are fixed by decision, not selected dynamically: the operator approved four
named scenarios whose segment identifiers are pinned here. S1-S3 read a segment row
from the approved bundle; S4 reads the frozen Stage 11 hidden-confounder cell
``gt_0.05_gy_0.50``, which is a global stress result and therefore not a segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aegis.schemas.registry import ApprovedSegmentPackage
from aegis.schemas.showcase import DISCLAIMER, ShowcaseView, StressCell


@dataclass(frozen=True)
class Scenario:
    """A fixed, read-only scenario displayed by the showcase."""

    key: str
    name: str
    subtitle: str
    risk_profile: str
    governance_state: str
    governance_note: str
    source_kind: Literal["segment", "stress"]
    segment_id: str | None = None
    stress_cell: str | None = None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="large-segment",
        name="Large, well-recovered segment",
        subtitle="High-exposure cohort with a tight elasticity interval.",
        risk_profile="Drivers 45-64 / cars 0-4 years / low density",
        governance_state="Diagnostic only",
        governance_note="Largest segment in the approved bundle; interval is near median width.",
        source_kind="segment",
        segment_id="D:45-64_C:0-4_Dens:<=500",
    ),
    Scenario(
        key="small-segment",
        name="Sparse segment, wide interval",
        subtitle="Few policies, honestly wide uncertainty.",
        risk_profile="Drivers 18-24 / cars 0-4 years / high density",
        governance_state="Diagnostic only",
        governance_note="Smallest cohort shown; the widest interval in the approved bundle.",
        source_kind="segment",
        segment_id="D:18-24_C:0-4_Dens:>2000",
    ),
    Scenario(
        key="cap-sensitive",
        name="Cap-sensitive segment",
        subtitle="Most uncapped loss mass sits above the 100k cap.",
        risk_profile="Drivers 18-24 / cars 10+ years / low density",
        governance_state="Diagnostic only",
        governance_note="Three claims above the cap remove most of this segment's uncapped loss.",
        source_kind="segment",
        segment_id="D:18-24_C:10+_Dens:<=500",
    ),
    Scenario(
        key="confounded",
        name="Hidden-confounder stress cell",
        subtitle="Global estimator degradation at gamma_T = 0.05, gamma_Y = 0.50.",
        risk_profile="Sensitivity grid cell, not a policy segment",
        governance_state="Stress evidence",
        governance_note="Rank correlation and bias cover the whole estimator, not one segment.",
        source_kind="stress",
        stress_cell="gt_0.05_gy_0.50",
    ),
)


def get_scenario(key: str | None) -> Scenario:
    """Returns a selected scenario or raises a clear client-facing lookup error."""
    if not key:
        raise LookupError("Choose one of the curated preset scenarios.")
    for scenario in SCENARIOS:
        if scenario.key == key:
            return scenario
    raise LookupError(f"Unknown preset scenario: {key}")


def build_view_model(
    scenario: Scenario, package: ApprovedSegmentPackage
) -> ShowcaseView:
    """Projects one approved-bundle scenario into the showcase response contract.

    Args:
        scenario: Fixed preset selected by the operator.
        package: Plain-data bundle resolved from the ``approved`` alias.

    Raises:
        LookupError: If the bundle cannot satisfy the preset without inventing data.
    """
    lineage = package.lineage
    segments = package.segments.segments
    if not segments:
        raise LookupError("approved bundle contains no Tier 1 segments")

    segment = None
    stress = None
    notice = None
    if scenario.source_kind == "segment":
        segment = next(
            (item for item in segments if item.segment_id == scenario.segment_id), None
        )
        if segment is None:
            segment = segments[0]
            notice = (
                f"Preset segment '{scenario.segment_id}' is not present in the approved "
                f"bundle; showing '{segment.segment_id}' instead."
            )
    else:
        payload = package.sensitivity_summary.get("scenario_s4")
        if not isinstance(payload, dict):
            raise LookupError(
                "approved bundle does not carry the Stage 11 sensitivity summary"
            )
        stress = StressCell.model_validate(payload)
        if stress.cell_id != scenario.stress_cell:
            raise LookupError(
                f"approved bundle carries sensitivity cell '{stress.cell_id}', "
                f"expected '{scenario.stress_cell}'"
            )

    return ShowcaseView(
        scenario_key=scenario.key,
        scenario_name=scenario.name,
        subtitle=scenario.subtitle,
        risk_profile=scenario.risk_profile,
        governance_state=scenario.governance_state,
        governance_note=scenario.governance_note,
        source_kind=scenario.source_kind,
        segment=segment,
        stress=stress,
        model_name=package.model_name,
        model_version=package.model_version,
        data_version=(segment or segments[0]).data_version,
        validation_status=lineage.validation_status,
        segment_count=lineage.segment_count,
        notice=notice,
        disclaimer=DISCLAIMER,
    )
