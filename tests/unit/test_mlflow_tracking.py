from __future__ import annotations

from pathlib import Path

import mlflow

from aegis.pipelines.training.mlflow_tracking import (
    attach_report_artifact,
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

    version = client.get_model_version(name="aegis-glm-baseline", version="1")
    assert version.name == "aegis-glm-baseline"
    artifact_paths = [
        item.path for item in client.list_artifacts(run.info.run_id, path="diagnostics")
    ]
    assert any(path.endswith("refutation_summary.json") for path in artifact_paths)


def test_configure_mlflow_tracking_recovers_from_stale_windows_artifact_uri(tmp_path: Path) -> None:
    """An old Windows-form artifact path should trigger a fresh valid experiment, not a crash."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = tmp_path / "artifacts" / "mlflow"
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    stale_location = "C:\\Users\\sebas\\Desktop\\AEGIS\\artifacts\\mlflow"
    db_path = tmp_path / "mlflow.db"
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO experiments
                (name, workspace, artifact_location, lifecycle_stage,
                 creation_time, last_update_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "aegis_stale_uri_recovery",
                "default",
                stale_location,
                "active",
                1,
                1,
            ),
        )

    experiment_id = configure_mlflow_tracking(
        tracking_uri=tracking_uri,
        artifact_location=str(artifact_root),
        experiment_name="aegis_stale_uri_recovery",
    )

    experiment = client.get_experiment(experiment_id)
    assert experiment is not None
    assert experiment.artifact_location.startswith("file://")


def test_attach_report_artifact_logs_repository_report(tmp_path: Path) -> None:
    """An evaluation report should be retrievable from the target MLflow run."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = tmp_path / "artifacts" / "mlflow"
    run = track_and_register_model(
        model_name="causal_elasticity",
        registered_name="aegis-causal-elasticity",
        model_object={"model": "demo"},
        tracking_uri=tracking_uri,
        artifact_location=str(artifact_root),
    )
    report = tmp_path / "phase_2_evaluation_report.md"
    report.write_text("stage 6 report", encoding="utf-8")

    attach_report_artifact(run.info.run_id, report, tracking_uri=tracking_uri)

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    artifacts = client.list_artifacts(run.info.run_id, path="evaluation")
    assert [artifact.path for artifact in artifacts] == ["evaluation/phase_2_evaluation_report.md"]
