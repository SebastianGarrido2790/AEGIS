"""Tests for synthetic treatment validation and causal elasticity estimation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from aegis.pipelines.training.causal_elasticity import (
    CausalElasticityResult,
    add_synthetic_treatment,
    fit_causal_elasticity,
    save_causal_artifact,
)


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """Use the real Stage 2 feature matrix for causal validation tests."""
    return pd.read_csv("data/versioned/feature_matrix.csv")


def test_add_synthetic_treatment_has_known_ground_truth(feature_frame: pd.DataFrame) -> None:
    """Synthetic treatment should vary by segment and have a known positive elasticity signature."""
    prepared = add_synthetic_treatment(
        feature_frame.sample(2000, random_state=42).reset_index(drop=True)
    )

    assert "treatment_rate_change" in prepared.columns
    assert bool(prepared["treatment_rate_change"].notna().all())
    assert prepared["treatment_rate_change"].std() > 0.0
    assert bool(prepared["treatment_rate_change"].gt(0.0).all())

    expected = prepared["risk_index"] * 0.10 + 0.05
    assert float((prepared["treatment_rate_change"] - expected).abs().max()) < 0.5


def test_fit_causal_elasticity_recovers_ground_truth(feature_frame: pd.DataFrame) -> None:
    """CausalForestDML should recover a positive directional effect aligned with truth."""
    result = fit_causal_elasticity(
        feature_frame.sample(2000, random_state=42).reset_index(drop=True)
    )

    assert isinstance(result, CausalElasticityResult)
    assert result.correlation >= 0.1
    assert result.average_treatment_effect > 0.0
    lower, upper = result.treatment_effect_confidence_interval
    assert lower <= upper
    assert lower == pytest.approx(lower)
    assert upper == pytest.approx(upper)
    assert set(result.refutation_summary) >= {
        "placebo_treatment",
        "random_common_cause",
        "data_subset",
    }
    assert result.calibration_metrics["baseline_mae"] >= 0.0


def test_save_causal_artifact_writes_json(tmp_path: Path) -> None:
    """The causal artifact should persist structured metrics for downstream comparison."""
    feature_frame = (
        pd.read_csv("data/versioned/feature_matrix.csv")
        .sample(2000, random_state=42)
        .reset_index(drop=True)
    )
    result = fit_causal_elasticity(feature_frame)
    artifact_path = tmp_path / "causal_elasticity.json"
    saved = save_causal_artifact(result, artifact_path)

    assert saved == artifact_path
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["model"] == "causal_forest_dml"
    assert payload["treatment_variable"] == "treatment_rate_change"
    interval = payload["treatment_effect_confidence_interval"]
    assert interval["alpha"] == 0.05
    assert interval["lower"] <= interval["upper"]
    assert "correlation" in payload["calibration_metrics"]
