"""Exposure normalization transformers for Stage 2 feature engineering."""

from __future__ import annotations

import pandas as pd


def normalize_exposure(frame: pd.DataFrame, exposure_col: str = "exposure") -> pd.DataFrame:
    """Normalize exposure into a bounded, deterministic feature.

    Args:
        frame: Input policy-level dataset.
        exposure_col: Column containing exposure values.

    Returns:
        A copy of the frame with an ``exposure_normalized`` column.
    """
    output = frame.copy()
    if exposure_col not in output.columns:
        raise KeyError(f"Missing required exposure column: {exposure_col}")

    exposure = output[exposure_col].astype(float).clip(lower=0.0)
    maximum = exposure.max()
    output["exposure_normalized"] = exposure / maximum if maximum > 0 else 0.0
    return output
