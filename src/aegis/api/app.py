"""FastAPI application for the AEGIS Phase 2 showcase."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aegis.api.model_service import load_registered_model_payload
from aegis.api.scenarios import SCENARIOS, build_view_model, get_scenario

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app = FastAPI(title="AEGIS Showcase", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


def _render(request: Request, selected: str | None = None, error: str | None = None) -> Any:
    """Render the showcase page with a selected scenario or a graceful error."""
    view_model = None
    if selected or error is None:
        try:
            scenario = get_scenario(selected or SCENARIOS[0].key)
            view_model = build_view_model(scenario, load_registered_model_payload())
        except (LookupError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            error = str(exc)
    return templates.TemplateResponse(
        request=request,
        name="showcase.html",
        context={
            "scenarios": SCENARIOS,
            "view_model": view_model,
            "error": error,
            "demo_label": "DEMO - NOT FOR PRODUCTION PRICING",
        },
    )


@app.get("/", response_class=HTMLResponse)
def showcase(request: Request, preset: str | None = None) -> Any:
    """Render the curated elasticity showcase and handle invalid presets safely."""
    return _render(request, selected=preset)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight liveness response."""
    return {
        "status": "ok",
        "service": "aegis-showcase",
        "mode": "demo",
        "demo_label": "DEMO - NOT FOR PRODUCTION PRICING",
    }
