from __future__ import annotations

from fastapi.testclient import TestClient

from aegis.api.app import app
from aegis.api.model_service import clear_model_cache, load_registered_model_payload
from aegis.api.scenarios import SCENARIOS

client = TestClient(app)


def test_showcase_renders_default_registered_model_scenario() -> None:
    """The default preset exposes registered-model metrics and demo labeling."""
    clear_model_cache()
    response = client.get("/")

    assert response.status_code == 200
    assert "DEMO - NOT FOR PRODUCTION PRICING" in response.text
    assert "Young Urban Commuter" in response.text
    assert "CausalForestDML" not in response.text
    assert "Estimated treatment effect" in response.text
    assert "confidence interval" in response.text
    assert "cdn.jsdelivr.net/npm/chart.js" in response.text


def test_showcase_renders_each_curated_preset() -> None:
    """Every approved segment preset renders a distinct scenario heading."""
    payload = load_registered_model_payload()
    base_effect = float(payload["average_treatment_effect"])
    base_interval = payload["treatment_effect_confidence_interval"]
    for preset, expected, multiplier in (
        ("young-urban-commuter", "Young Urban Commuter", 1.18),
        ("experienced-rural-driver", "Experienced Rural Driver", 0.82),
        ("high-mileage-commercial-driver", "High-Mileage Commercial Driver", 1.0),
    ):
        response = client.get("/", params={"preset": preset})
        assert response.status_code == 200
        assert expected in response.text
        assert "DEMO - NOT FOR PRODUCTION PRICING" in response.text
        assert f"{base_effect * multiplier:.2f}" in response.text
        assert f"{float(base_interval['lower']) * multiplier:.2f}" in response.text
        assert f"{float(base_interval['upper']) * multiplier:.2f}" in response.text

    rendered_effects = [base_effect * scenario.model_multiplier for scenario in SCENARIOS]
    assert len(set(rendered_effects)) == len(SCENARIOS)


def test_showcase_handles_invalid_preset_gracefully() -> None:
    """An unknown preset returns a usable error page instead of crashing."""
    response = client.get("/", params={"preset": "not-a-preset"})

    assert response.status_code == 200
    assert "Scenario unavailable" in response.text
    assert "Return to the default preset" in response.text
    assert "DEMO - NOT FOR PRODUCTION PRICING" in response.text


def test_health_route_is_labeled_demo() -> None:
    """The liveness route reports the showcase mode explicitly."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "aegis-showcase",
        "mode": "demo",
        "demo_label": "DEMO - NOT FOR PRODUCTION PRICING",
    }
