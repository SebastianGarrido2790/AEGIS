"""Actuarial Tweedie GLM baseline for Stage 3 (ADR-013).

This module implements a deterministic Tweedie GLM baseline for pure premium prediction
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from aegis.pipelines.feature.pipeline import build_feature_matrix, create_policy_split

DEFAULT_FEATURE_COLUMNS = (
    "driver_age",
    "veh_age",
    "bonus_malus",
    "veh_power",
    "exposure_normalized",
    "driver_risk_score",
    "vehicle_risk_score",
    "risk_index",
)


@dataclass(frozen=True)
class GLMBaselineResult:
    """Structured Stage 3 output used by later model comparison stages."""

    fitted_model: Any
    feature_columns: tuple[str, ...]
    confidence_intervals: pd.DataFrame
    calibration_metrics: dict[str, float]
    predictions: pd.DataFrame
    converged: bool


def _prepare_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate inputs and derive pure premium without using derived premium."""
    required = {"claim_amount", "exposure", "policy_id"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing GLM target columns: {sorted(missing)}")

    output = frame.copy()
    output["exposure"] = pd.to_numeric(output["exposure"], errors="raise")
    output["claim_amount"] = pd.to_numeric(output["claim_amount"], errors="raise")
    if (output["exposure"] <= 0).any():
        raise ValueError("GLM exposure must be strictly positive.")
    if (output["claim_amount"] < 0).any():
        raise ValueError("GLM claim_amount must be non-negative.")

    output["pure_premium"] = output["claim_amount"] / output["exposure"]
    return output


def fit_tweedie_baseline(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    test_size: float = 0.2,
    random_state: int = 42,
) -> GLMBaselineResult:
    """Fit a weighted Tweedie GLM and calculate a held-out calibration baseline.

    The response is pure premium (claim amount divided by exposure), while exposure
    is used as a frequency weight. The derived tariff ``premium`` is deliberately
    excluded from predictors to prevent outcome leakage.
    """
    matrix = build_feature_matrix(frame) if "risk_index" not in frame.columns else frame.copy()
    prepared = _prepare_target(matrix)
    train_frame, test_frame = create_policy_split(
        prepared,
        test_size=test_size,
        random_state=random_state,
    )
    if train_frame.empty or test_frame.empty:
        raise ValueError("GLM training and test partitions must both contain rows.")

    missing_features = set(feature_columns) - set(train_frame.columns)
    if missing_features:
        raise KeyError(f"Missing GLM feature columns: {sorted(missing_features)}")

    train_x = sm.add_constant(train_frame[list(feature_columns)], has_constant="add")
    test_x = sm.add_constant(test_frame[list(feature_columns)], has_constant="add")
    model = sm.GLM(
        train_frame["pure_premium"],
        train_x,
        family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log()),
        freq_weights=train_frame["exposure"],
    )
    fitted = model.fit(maxiter=200)
    confidence_intervals = fitted.conf_int()
    if not np.isfinite(confidence_intervals.to_numpy()).all():
        raise ValueError("Tweedie GLM produced non-finite parameter confidence intervals.")

    predictions = np.maximum(fitted.predict(test_x), 0.0)
    actual = test_frame["pure_premium"].to_numpy()
    metrics = {
        "test_mae": float(mean_absolute_error(actual, predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
    }
    prediction_frame = pd.DataFrame(
        {
            "policy_id": test_frame["policy_id"].to_numpy(),
            "actual_pure_premium": actual,
            "predicted_pure_premium": predictions,
        }
    )

    return GLMBaselineResult(
        fitted_model=fitted,
        feature_columns=feature_columns,
        confidence_intervals=confidence_intervals,
        calibration_metrics=metrics,
        predictions=prediction_frame,
        converged=bool(getattr(fitted, "converged", True)),
    )


def save_baseline_artifact(result: GLMBaselineResult, output_path: Path | str) -> Path:
    """Persist baseline metrics, intervals, and model metadata as JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": "tweedie_glm",
        "family": "Tweedie",
        "response": "pure_premium",
        "feature_columns": list(result.feature_columns),
        "converged": result.converged,
        "calibration_metrics": result.calibration_metrics,
        "confidence_intervals": {
            str(name): {"lower": float(row.iloc[0]), "upper": float(row.iloc[1])}
            for name, row in result.confidence_intervals.iterrows()
        },
    }
    destination.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return destination
