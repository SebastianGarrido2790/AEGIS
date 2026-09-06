from __future__ import annotations

from pathlib import Path

import mlflow

from aegis.pipelines.training.mlflow_tracking import (
    configure_mlflow_tracking,
    track_and_register_model,
)


def test_track_and_register_model_logs_registry_version_and_artifact(tmp_path: Path) -> None:
    """Verifies that Stage 5 registers a model under the approved name and logs an artifact."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = tmp_path / "artifacts" / "mlflow"
    configure_mlflow_tracking(tracking_uri=tracking_uri, artifact_location=str(artifact_root))

    model = {"name": "demo-model", "score": 0.87}
    run = track_and_register_model(
        model_name="glm_baseline",
        registered_name="aegis-glm-baseline",
        model_object=model,
        params={"random_state": 42},
        metrics={"test_mae": 0.12},
        artifact_payload={"refutation_summary": {"placebo_treatment": {"passed": True}}},
        artifact_name="refutation_summary.json",
        tracking_uri=tracking_uri,
        artifact_location=str(artifact_root),
    )

    assert run is not None
    assert run.info.run_id
    client = mlflow.MlflowClient()
    registered = client.search_registered_models(filter_string="name = 'aegis-glm-baseline'")
    assert registered and registered[0].name == "aegis-glm-baseline"

    version = client.get_model_version(name="aegis-glm-baseline", version=1)
    assert version.name == "aegis-glm-baseline"
    artifact_paths = [
        item.path for item in client.list_artifacts(run.info.run_id, path="diagnostics")
    ]
    assert any(path.endswith("refutation_summary.json") for path in artifact_paths)
