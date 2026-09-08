"""Registered-model metadata service for the Phase 2 showcase."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from mlflow.tracking import MlflowClient

MODEL_NAME = "aegis-causal-elasticity"
TRACKING_URI = "sqlite:///mlflow.db"


def _flatten_model_payload(payload: Any) -> dict[str, Any]:
    """Flatten nested metadata dictionaries into the shape the showcase expects."""
    if not isinstance(payload, dict):
        return {}

    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            flattened[key] = value
            flattened.update(_flatten_model_payload(value))
        else:
            flattened[key] = value
    return flattened


@lru_cache(maxsize=1)
def load_registered_model_payload() -> dict[str, Any]:
    """Load the newest usable registered causal model payload from local MLflow."""
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = sorted(
        client.search_model_versions(f"name = '{MODEL_NAME}'"),
        key=lambda version: int(version.version),
        reverse=True,
    )
    if not versions:
        raise RuntimeError(f"Registered model not found: {MODEL_NAME}")

    candidates = (
        "model_package/metadata.json",
        "registered_model_metadata.json",
        "diagnostics/causal_refutation_summary.json",
        "diagnostics/glm_baseline_summary.json",
    )
    for version in versions:
        if version.run_id is None:
            continue
        for artifact_name in candidates:
            try:
                artifact_path = client.download_artifacts(version.run_id, artifact_name)
            except Exception:
                continue
            with open(artifact_path, encoding="utf-8") as artifact_file:
                payload = json.load(artifact_file)
            if not isinstance(payload, dict):
                continue
            flattened = _flatten_model_payload(payload)
            calibration_metrics = flattened.get("calibration_metrics", {})
            if (
                "average_treatment_effect" not in flattened
                and isinstance(calibration_metrics, dict)
                and "average_treatment_effect" in calibration_metrics
            ):
                flattened["average_treatment_effect"] = calibration_metrics[
                    "average_treatment_effect"
                ]
            if (
                "treatment_effect_confidence_interval" not in flattened
                and isinstance(calibration_metrics, dict)
                and "treatment_effect_confidence_interval" in calibration_metrics
            ):
                flattened["treatment_effect_confidence_interval"] = calibration_metrics[
                    "treatment_effect_confidence_interval"
                ]
            if {
                "average_treatment_effect",
                "treatment_effect_confidence_interval",
            }.issubset(flattened):
                return flattened
    raise TypeError("Registered causal model did not return a usable metadata payload")


def clear_model_cache() -> None:
    """Clear the registered-model cache for tests or a deliberate model refresh."""
    load_registered_model_payload.cache_clear()
