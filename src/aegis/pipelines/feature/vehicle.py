"""Vehicle and territorial risk factor feature engineering for Tier 1 actuarial pipelines.

This module implements deterministic vehicle and geographic density risk transformations
for Stage 2 feature engineering. In auto insurance tariff modeling (calibrated against
the freMTPL2 benchmark), vehicle characteristics and territorial density serve as primary
actuarial predictors:

Actuarial & Mathematical Formulation:
    1. Vehicle Age Component:
       Scaled relative to a typical vehicle lifespan threshold of 10 years:
           age_component = (veh_age / 10.0)^2
       Older vehicles often correlate with higher mechanical failure rates, reduced safety
       technology (e.g., lack of modern ADAS), and varied maintenance rigor.

    2. Vehicle Power Component:
       Scaled relative to average engine fiscal power (10 horsepower/CV):
           power_component = (veh_power / 10.0)^2
       Higher power vehicles correlate actuarially with aggressive acceleration profiles,
       elevated severity when collisions occur, and higher claim probabilities.

    3. Territorial Density Component:
       Scaled relative to urban threshold density (5,000 inhabitants/km^2):
           density_component = (density / 5000.0)^2
       High population density indicates dense urban traffic environments, increased traffic
       congestion, and significantly higher collision frequency relative to rural territories.

    4. Composite Vehicle Risk Score:
       Convex weighted combination with baseline floor 1.0:
           vehicle_risk_score = clip(
               0.5 * age_component + 0.3 * power_component + 0.2 * density_component + 1.0,
               lower=0.0
           )

Quality & Governance Invariants:
    - Zero Silent Imputation: Imputation counts are logged at INFO level for veh_age,
      veh_power, and density before median imputation (Phase 2 Remediation Stage 5).
    - Determinism: The transformation is functional, deterministic, and idempotent across
      training and serving paths.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_vehicle_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic vehicle and territorial risk scores.

    Extracts vehicle age, vehicle power, and geographic population density, logs any
    missing value counts, imputes missing values using column medians, and computes
    the composite `vehicle_risk_score` column.

    Args:
        frame: Input DataFrame containing vehicle and territorial variables.
            Must contain columns 'veh_age', 'veh_power', and 'density'.

    Returns:
        pd.DataFrame: A shallow copy of the input DataFrame augmented with the
            float column 'vehicle_risk_score'.

    Raises:
        KeyError: If any of 'veh_age', 'veh_power', or 'density' is missing from `frame`.

    Notes:
        - Imputation: Missing values in 'veh_age', 'veh_power', or 'density' are logged
          at INFO level ('<column>: <count> imputed') and replaced by the column median.
        - Training-Serving Parity: This transformation is applied identically during
          offline model training and online policy evaluation.
    """
    output = frame.copy()
    required = {"veh_age", "veh_power", "density"}
    missing = required - set(output.columns)
    if missing:
        raise KeyError(f"Missing required vehicle columns: {sorted(missing)}")

    veh_age_imputed = int(output["veh_age"].isna().sum())
    veh_power_imputed = int(output["veh_power"].isna().sum())
    density_imputed = int(output["density"].isna().sum())
    logger.info("veh_age: %d imputed", veh_age_imputed)
    logger.info("veh_power: %d imputed", veh_power_imputed)
    logger.info("density: %d imputed", density_imputed)

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
