"""Vehicle risk factor features for Stage 2 feature engineering."""

from __future__ import annotations

import pandas as pd


def compute_vehicle_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic vehicle risk score from age and power."""
    output = frame.copy()
    required = {"veh_age", "veh_power", "density"}
    missing = required - set(output.columns)
    if missing:
        raise KeyError(f"Missing required vehicle columns: {sorted(missing)}")

    veh_age = output["veh_age"].astype(float).fillna(output["veh_age"].median())
    veh_power = output["veh_power"].astype(float).fillna(output["veh_power"].median())
    density = output["density"].astype(float).fillna(output["density"].median())

    age_component = (veh_age / 10.0) ** 2
    power_component = (veh_power / 10.0) ** 2
    density_component = (density / 5000.0) ** 2
    output["vehicle_risk_score"] = (
        0.5 * age_component + 0.3 * power_component + 0.2 * density_component + 1.0
    ).clip(lower=0.0)
    return output
