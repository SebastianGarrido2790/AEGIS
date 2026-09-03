import json
from pathlib import Path

import numpy as np
import pandas as pd

from aegis.pipelines.training.glm_baseline import (
    fit_tweedie_baseline,
    save_baseline_artifact,
)


def baseline_frame() -> pd.DataFrame:
    rows = 40
    exposure = np.linspace(0.2, 1.0, rows)
    driver_age = np.linspace(22.0, 70.0, rows)
    claim_amount = exposure * (80.0 + 2.5 * driver_age + 15.0 * (np.arange(rows) % 3))
    return pd.DataFrame(
        {
            "policy_id": [f"POL-{index}" for index in range(rows)],
            "driver_age": driver_age,
            "veh_age": np.linspace(1.0, 12.0, rows),
            "bonus_malus": np.linspace(50.0, 100.0, rows),
            "veh_power": np.linspace(4.0, 9.0, rows),
            "exposure": exposure,
            "exposure_normalized": exposure,
            "driver_risk_score": np.linspace(1.0, 2.0, rows),
            "vehicle_risk_score": np.linspace(1.0, 2.0, rows),
            "risk_index": np.linspace(1.0, 3.0, rows),
            "claim_amount": claim_amount,
            "claim_count": (np.arange(rows) % 3).astype(float),
            "premium": np.full(rows, 250.0),
        }
    )


def test_tweedie_baseline_fits_with_finite_intervals() -> None:
    result = fit_tweedie_baseline(baseline_frame(), test_size=0.25, random_state=42)

    assert result.converged is True
    assert result.confidence_intervals.shape[1] == 2
    assert np.isfinite(result.confidence_intervals.to_numpy()).all()
    assert result.calibration_metrics["test_mae"] >= 0.0
    assert result.calibration_metrics["test_rmse"] >= 0.0
    assert len(result.predictions) == 10


def test_baseline_artifact_records_metric_and_intervals(tmp_path: Path) -> None:
    result = fit_tweedie_baseline(baseline_frame(), test_size=0.25, random_state=42)
    artifact_path = tmp_path / "glm_baseline.json"

    save_baseline_artifact(result, artifact_path)

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["model"] == "tweedie_glm"
    assert "test_mae" in artifact["calibration_metrics"]
    assert artifact["confidence_intervals"]
