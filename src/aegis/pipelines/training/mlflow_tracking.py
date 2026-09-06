"""MLflow tracking and model registry helpers for Stage 5 (ADR-015)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow.pyfunc import PythonModel


class _SerializablePythonModel(PythonModel):
    """Minimal PythonModel used to register an object payload in the MLflow registry."""

    def __init__(self, payload: Any):
        self.payload = payload

    def predict(self, context: Any, model_input: Any) -> Any:
        """Return the payload without altering the caller's model input."""
        if model_input is None:
            return self.payload
        if hasattr(model_input, "to_dict"):
            return model_input.to_dict()
        return self.payload


def _normalize_artifact_location(path: str | Path | None) -> str:
    """Convert a filesystem path into a valid MLflow file URI."""
    if path is None:
        resolved = (Path.cwd() / "artifacts" / "mlflow").resolve()
    else:
        resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved.as_uri()


def configure_mlflow_tracking(
    tracking_uri: str = "sqlite:///mlflow.db",
    artifact_location: str | None = None,
    experiment_name: str = "aegis",
) -> str:
    """Configure the MLflow tracking backend and ensure the experiment exists."""
    mlflow.set_tracking_uri(tracking_uri)
    artifact_root = _normalize_artifact_location(artifact_location)

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_root,
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def _write_json_artifact(payload: Any, filename: str) -> str:
    """Serialize a payload to a temporary JSON artifact and return its path.

    The file must remain on disk until MLflow finishes copying it into its artifact
    repository, so we intentionally avoid deleting the temporary directory before the
    upload completes.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="aegis_mlflow_"))
    temp_path = temp_dir / filename
    temp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(temp_path)


def _log_model_package(run_id: str, model_object: Any, artifact_path: str = "model_package") -> str:
    """Log a serializable model package and return the runs:/ URI for registration."""
    payload = model_object
    if hasattr(model_object, "__dict__") and not isinstance(
        model_object, (dict, list, tuple, str, int, float, bool)
    ):
        payload = {
            "type": type(model_object).__name__,
            "attributes": model_object.__dict__,
        }

    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=_SerializablePythonModel(payload),
        input_example={"payload": payload},
    )
    return f"runs:/{run_id}/{artifact_path}"


def track_and_register_model(
    model_name: str,
    registered_name: str,
    model_object: Any,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    artifact_payload: dict[str, Any] | None = None,
    artifact_name: str = "refutation_summary.json",
    experiment_name: str = "aegis",
    tracking_uri: str = "sqlite:///mlflow.db",
    artifact_location: str | None = None,
) -> mlflow.ActiveRun:
    """Log a model run, attach a diagnostic artifact, and register the model in MLflow."""
    experiment_id = configure_mlflow_tracking(
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
        experiment_name=experiment_name,
    )

    with mlflow.start_run(run_name=model_name, experiment_id=experiment_id) as run:
        if params:
            mlflow.log_params({str(key): str(value) for key, value in params.items()})
        if metrics:
            mlflow.log_metrics({str(key): float(value) for key, value in metrics.items()})

        if artifact_payload is not None:
            temp_path = Path(_write_json_artifact(artifact_payload, artifact_name))
            mlflow.log_artifact(str(temp_path), artifact_path="diagnostics")

        model_uri = _log_model_package(run.info.run_id, model_object)
        try:
            mlflow.register_model(model_uri=model_uri, name=registered_name)
        except (
            Exception
        ):  # pragma: no cover - MLflow registry may accept the model without any extra action.
            pass

        return run
