"""Unit tests for the Phase 2 diagnostic showcase (P2-F11, Gate 13).

The suite is backend-free: it injects an ``ApprovedSegmentPackage`` that has the same
shape the Stage 12 loader returns, then asserts every rendered number comes from that
package rather than from source constants. The falsification protocol (INV-5) proves
that a proposed price-change field cannot enter the response contract and that the
disclaimer banner survives every view, including governance-error pages.
"""

from __future__ import annotations

from html import unescape
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import aegis.api.app as app_module
from aegis.api.app import app
from aegis.api.scenarios import SCENARIOS, build_view_model, get_scenario
from aegis.schemas.registry import ApprovedSegmentPackage, ModelLineageMetadata
from aegis.schemas.segments import FORBIDDEN_FIELD_FRAGMENTS, Tier1SegmentOutput, Tier1SegmentTable
from aegis.schemas.showcase import DISCLAIMER, ShowcaseView
from aegis.utils.exceptions import ModelNotFoundError

client = TestClient(app)

MODEL_NAME = "aegis-causal-elasticity"
LARGE = "D:45-64_C:0-4_Dens:<=500"
SMALL = "D:18-24_C:0-4_Dens:>2000"
CAP_SENSITIVE = "D:18-24_C:10+_Dens:<=500"
STATIC_CHART = Path(app_module.STATIC_DIR) / "chart.umd.min.js"
S4_CELL = {
    "cell_id": "gt_0.05_gy_0.50",
    "gamma_t": 0.05,
    "gamma_y": 0.5,
    "confounding_strength": 0.5025,
    "m1_tau_spearman": 0.6579,
    "m1_epsilon_spearman": 0.6216,
    "m2_mean_abs_bias": 0.7419,
    "rationale": "Moderate confounding for showcase Scenario S4.",
}


def _segment(
    segment_id: str, policy_count: int, epsilon: float, loss_cost: float
) -> Tier1SegmentOutput:
    """Builds one valid Tier 1 segment row for the fake approved bundle."""
    return Tier1SegmentOutput(
        segment_id=segment_id,
        policy_count=policy_count,
        exposure_years=policy_count / 5.0,
        baseline_retention_q0=0.87,
        tau_hat=epsilon / 2.0,
        epsilon_hat=epsilon,
        epsilon_ci_lower=epsilon - 1.5,
        epsilon_ci_upper=epsilon + 1.5,
        loss_cost_per_year=loss_cost,
        base_premium_per_year=loss_cost * 1.2,
        model_version="3",
        data_version="dvc-24f3e55a",
        semi_synthetic=True,
    )


def _package(segments: list[Tier1SegmentOutput] | None = None) -> ApprovedSegmentPackage:
    """Builds the plain-data package the approved loader would return."""
    rows = segments or [
        _segment(LARGE, 33292, -0.307, 114.8),
        _segment(SMALL, 1318, -0.513, 507.2),
        _segment(CAP_SENSITIVE, 6003, -1.010, 96.7),
    ]
    return ApprovedSegmentPackage(
        model_name=MODEL_NAME,
        model_version="3",
        run_id="run-77f6272859a2474da04de3b2bff04916",
        segments=Tier1SegmentTable(segments=rows),
        lineage=ModelLineageMetadata(
            model_name=MODEL_NAME,
            model_version="3",
            run_id="run-77f6272859a2474da04de3b2bff04916",
            git_commit="0" * 40,
            git_dirty=False,
            dvc_data_hash="2" * 32,
            dgp_manifest_id="0" * 64,
            calibration_hash="4" * 64,
            validation_status="GATE_PASSED_OPTION_F",
            segment_count=len(rows),
            created_at_utc="2026-10-31T00:00:00+00:00",
        ),
        validation_summary={"campaign_gates": {"all_seeds_pass_frozen_thresholds": False}},
        sensitivity_summary={"scenario_s4": S4_CELL},
    )


@pytest.fixture
def approved(monkeypatch: pytest.MonkeyPatch) -> ApprovedSegmentPackage:
    """Serves the fake approved bundle in place of the registry loader."""
    package = _package()
    monkeypatch.setattr(app_module, "load_showcase_bundle", lambda: package)
    return package


def test_default_scenario_reads_approved_bundle(approved: ApprovedSegmentPackage) -> None:
    """The default page renders the approved version and its real segment metrics."""
    response = client.get("/")

    assert response.status_code == 200
    assert "Large, well-recovered segment" in response.text
    assert f"{approved.segments.segments[0].epsilon_hat:.2f}" in response.text
    assert f"{approved.segments.segments[0].policy_count}" in response.text
    assert "GATE_PASSED_OPTION_F" in response.text
    assert "v3" in response.text
    assert "approved alias" in response.text
    assert "CausalForestDML" not in response.text


def test_each_preset_renders_its_own_segment(approved: ApprovedSegmentPackage) -> None:
    """Every segment preset renders the matching row from the approved bundle."""
    expected = {
        "large-segment": approved.segments.segments[0],
        "small-segment": approved.segments.segments[1],
        "cap-sensitive": approved.segments.segments[2],
    }
    for key, segment in expected.items():
        direct = client.get(f"/scenario/{key}")
        query = client.get("/", params={"preset": key})
        for response in (direct, query):
            assert response.status_code == 200
            text = unescape(response.text)
            assert segment.segment_id in text
            assert f"{segment.epsilon_hat:.2f}" in text
            assert f"{segment.policy_count}" in text


def test_confounded_preset_renders_stage11_stress_cell(
    approved: ApprovedSegmentPackage,
) -> None:
    """S4 shows the frozen global sensitivity cell, not a fabricated segment."""
    response = client.get("/scenario/confounded")

    assert response.status_code == 200
    assert S4_CELL["cell_id"] in response.text
    assert f"{S4_CELL['m1_tau_spearman']:.4f}" in response.text
    assert f"{S4_CELL['m2_mean_abs_bias']:.4f}" in response.text
    assert "Global stress result, not a policy segment" in response.text
    assert "CausalForestDML" not in response.text


def test_scenarios_are_locked_to_approved_presets() -> None:
    """The four approved preset keys are stable and unique."""
    keys = [scenario.key for scenario in SCENARIOS]
    assert keys == ["large-segment", "small-segment", "cap-sensitive", "confounded"]
    assert len(set(keys)) == len(keys)
    assert get_scenario("confounded").stress_cell == S4_CELL["cell_id"]


def test_banner_present_on_every_view(approved: ApprovedSegmentPackage) -> None:
    """The disclaimer banner survives segment, stress, invalid, and error pages."""
    responses = [client.get("/"), client.get("/scenario/confounded")]
    responses.append(client.get("/", params={"preset": "not-a-preset"}))

    for response in responses:
        assert response.status_code == 200
        assert DISCLAIMER in response.text
        assert "DEMO - NOT FOR PRODUCTION PRICING" in response.text


def test_proposed_price_change_is_rejected(approved: ApprovedSegmentPackage) -> None:
    """Falsification: injecting a price change into the response contract fails loudly."""
    payload = build_view_model(SCENARIOS[0], approved).model_dump()
    payload["proposed_rate_change"] = 0.05

    with pytest.raises(ValidationError):
        ShowcaseView.model_validate(payload)


def test_no_adjustment_field_names_in_rendered_page(approved: ApprovedSegmentPackage) -> None:
    """INV-5 structural check: the rendered page names no proposed price change."""
    response = client.get("/")

    lowered = response.text.lower()
    for fragment in FORBIDDEN_FIELD_FRAGMENTS:
        assert fragment not in lowered


def test_local_chartjs_asset_and_no_cdn(approved: ApprovedSegmentPackage) -> None:
    """Chart.js is served from local static assets, never from a CDN."""
    response = client.get("/")

    assert "/static/chart.umd.min.js" in response.text
    assert "cdn." not in response.text
    assert STATIC_CHART.is_file()
    assert STATIC_CHART.stat().st_size > 100_000


def test_approved_alias_missing_renders_governance_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset approved alias yields a clear governance page, not a crash or fallback."""
    def _raise() -> ApprovedSegmentPackage:
        raise ModelNotFoundError(
            "No model version currently assigned to alias 'approved'",
            details={"model_name": MODEL_NAME, "alias": "approved"},
        )

    monkeypatch.setattr(app_module, "load_showcase_bundle", _raise)
    response = client.get("/")

    assert response.status_code == 200
    assert "Scenario unavailable" in response.text
    assert "alias 'approved'" in unescape(response.text)
    assert DISCLAIMER in response.text


def test_missing_preset_segment_falls_back_with_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preset absent from the bundle falls back visibly without inventing metrics."""
    package = _package(
        [
            _segment(SMALL, 1318, -0.513, 507.2),
            _segment(CAP_SENSITIVE, 6003, -1.010, 96.7),
        ]
    )
    monkeypatch.setattr(app_module, "load_showcase_bundle", lambda: package)

    response = client.get("/")

    assert response.status_code == 200
    assert "is not present in the approved bundle" in response.text
    assert SMALL in unescape(response.text)


def test_unknown_preset_handles_gracefully(approved: ApprovedSegmentPackage) -> None:
    """An unknown preset returns a usable error page instead of crashing."""
    response = client.get("/", params={"preset": "not-a-preset"})

    assert response.status_code == 200
    assert "Scenario unavailable" in response.text
    assert "Return to the default preset" in response.text


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
