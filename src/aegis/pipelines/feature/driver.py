"""Driver risk factor features for Stage 2 feature engineering."""

from __future__ import annotations

import pandas as pd


def compute_driver_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic driver risk score from age and bonus malus."""
    output = frame.copy()
    required = {"driver_age", "bonus_malus"}
    missing = required - set(output.columns)
    if missing:
        raise KeyError(f"Missing required driver columns: {sorted(missing)}")

    driver_age = output["driver_age"].astype(float).fillna(output["driver_age"].median())
    bonus_malus = output["bonus_malus"].astype(float).fillna(output["bonus_malus"].median())

    adjusted_age = (driver_age - 35.0) / 25.0
    adjusted_bonus = (bonus_malus - 50.0) / 100.0
    output["driver_risk_score"] = (0.6 * adjusted_age**2 + 0.4 * adjusted_bonus**2 + 1.0).clip(
        lower=0.0
    )
    return output
