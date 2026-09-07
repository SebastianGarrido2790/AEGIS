"""Registered-model metadata service for the Phase 2 showcase."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from mlflow.tracking import MlflowClient

MODEL_NAME = "aegis-causal-elasticity"


@lru_cache(maxsize=1)
def load_registered_model_payload() -> dict[str, Any]:
    """Load the latest registered causal model payload from local MLflow."""
    client = MlflowClient()
    versions = list(client.search_model_versions(f"name = '{MODEL_NAME}'"))
    if not versions:
        raise RuntimeError(f"Registered model not found: {MODEL_NAME}")
    latest = max(versions, key=lambda version: int(version.version))
    if latest.run_id is None:
        raise RuntimeError(f"Registered model version has no source run: {latest.version}")
    artifact_path = client.download_artifacts(latest.run_id, "model_package/metadata.json")
    with open(artifact_path, encoding="utf-8") as artifact_file:
        payload = json.load(artifact_file)
    if not isinstance(payload, dict):
        raise TypeError("Registered causal model did not return a metadata payload")
    return payload


def clear_model_cache() -> None:
    """Clear the registered-model cache for tests or a deliberate model refresh."""
    load_registered_model_payload.cache_clear()
