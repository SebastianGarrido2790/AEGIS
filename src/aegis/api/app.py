"""FastAPI application for the AEGIS Phase 2 diagnostic showcase (P2-F11)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mlflow.exceptions import MlflowException

from aegis.api.model_service import load_showcase_bundle
from aegis.api.scenarios import SCENARIOS, build_view_model, get_scenario
from aegis.schemas.showcase import DISCLAIMER
from aegis.utils.exceptions import AEGISError

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
DEMO_LABEL = "DEMO - NOT FOR PRODUCTION PRICING"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app = FastAPI(title="AEGIS Showcase", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _render(request: Request, selected: str | None = None) -> Any:
    """Renders the selected preset, or a governance-state page when loading fails."""
    view_model = None
    error = None
    try:
        scenario = get_scenario(selected or SCENARIOS[0].key)
        view_model = build_view_model(scenario, load_showcase_bundle())
    except (
        LookupError,
        ValueError,
        OSError,
        AEGISError,
        MlflowException,
    ) as exc:
        error = str(exc)
    return templates.TemplateResponse(
        request=request,
        name="showcase.html",
        context={
            "scenarios": SCENARIOS,
            "view_model": view_model,
            "error": error,
            "disclaimer": DISCLAIMER,
            "demo_label": DEMO_LABEL,
        },
    )


@app.get("/", response_class=HTMLResponse)
def showcase(request: Request, preset: str | None = None) -> Any:
    """Renders the curated elasticity showcase and handles invalid presets safely."""
    return _render(request, selected=preset)


@app.get("/scenario/{key}", response_class=HTMLResponse)
def scenario_page(request: Request, key: str) -> Any:
    """Renders one curated preset by key."""
    return _render(request, selected=key)


@app.get("/health")
def health() -> dict[str, str]:
    """Returns a lightweight liveness response."""
    return {
        "status": "ok",
        "service": "aegis-showcase",
        "mode": "demo",
        "demo_label": DEMO_LABEL,
    }
