"""Curated segment scenarios for the Phase 2 showcase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scenario:
    """A fixed, read-only scenario displayed by the showcase."""

    key: str
    name: str
    subtitle: str
    risk_profile: str
    model_multiplier: float
    governance_state: str
    governance_note: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        key="young-urban-commuter",
        name="Young Urban Commuter",
        subtitle="Higher-frequency urban exposure",
        risk_profile="Young driver / urban profile",
        model_multiplier=1.18,
        governance_state="Review corridor",
        governance_note="Illustrative segment signal is near the review corridor.",
    ),
    Scenario(
        key="experienced-rural-driver",
        name="Experienced Rural Driver",
        subtitle="Lower-risk rural exposure",
        risk_profile="Experienced driver / rural profile",
        model_multiplier=0.82,
        governance_state="Within demo corridor",
        governance_note="Illustrative segment signal remains inside the demo corridor.",
    ),
    Scenario(
        key="high-mileage-commercial-driver",
        name="High-Mileage Commercial Driver",
        subtitle="High exposure and commercial use",
        risk_profile="Commercial use / high mileage",
        model_multiplier=1.0,
        governance_state="Escalation candidate",
        governance_note="Illustrative segment is surfaced for human review in this demo.",
    ),
)


def get_scenario(key: str | None) -> Scenario:
    """Return a selected scenario or raise a clear client-facing lookup error."""
    if not key:
        raise LookupError("Choose one of the curated preset scenarios.")
    for scenario in SCENARIOS:
        if scenario.key == key:
            return scenario
    raise LookupError(f"Unknown preset scenario: {key}")


def build_view_model(scenario: Scenario, payload: dict[str, Any]) -> dict[str, Any]:
    """Combine registered-model metadata with a deterministic scenario multiplier."""
    base_effect = float(payload["average_treatment_effect"])
    interval = payload["treatment_effect_confidence_interval"]
    lower = float(interval["lower"])
    upper = float(interval["upper"])
    multiplier = scenario.model_multiplier
    return {
        "scenario": scenario,
        "elasticity": base_effect * multiplier,
        "confidence_interval": {
            "lower": lower * multiplier,
            "upper": upper * multiplier,
            "alpha": float(interval.get("alpha", 0.05)),
        },
        "model_name": payload.get("model", "causal_forest_dml"),
        "treatment_variable": payload.get("treatment_variable", "treatment_rate_change"),
    }
