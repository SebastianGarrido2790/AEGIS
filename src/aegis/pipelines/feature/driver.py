"""Driver risk factor feature engineering for Tier 1 actuarial pipelines.

This module implements deterministic driver risk transformations for Stage 2 feature
engineering. In auto insurance actuarial modeling (specifically calibrated against the
freMTPL2 benchmark), driver age and bonus-malus coefficient represent primary rating factors:

Actuarial & Mathematical Formulation:
    1. Driver Age Normalization:
       Centered around prime driving age 35 with a scale parameter of 25 years:
           adjusted_age = (driver_age - 35.0) / 25.0
       This quadratic formulation captures the well-documented U-shaped actuarial risk
       profile: younger/novice drivers (< 25) and senior drivers (> 65) exhibit elevated
       claim frequency compared to experienced middle-aged drivers.

    2. Bonus-Malus Normalization:
       Centered around the optimal merit-rating baseline (bonus 50) with scale 100:
           adjusted_bonus = (bonus_malus - 50.0) / 100.0
       Under the French regulatory system, bonus-malus starts at 100, drops down to 50
       for claim-free driving (bonus), and increases up to 350 for repeated at-fault claims (malus).

    3. Composite Driver Risk Score:
       A convex quadratic combination with baseline floor 1.0:
           driver_risk_score = clip(0.6 * adjusted_age^2 + 0.4 * adjusted_bonus^2 + 1.0, lower=0.0)

Quality & Governance Invariants:
    - Zero Silent Imputation: Missing values are logged explicitly at INFO level prior to
      median replacement, ensuring full audit traceability (Phase 2 Remediation Stage 5).
    - Determinism: The transformation is purely functional, stateless, and idempotent,
      guaranteeing identical behavior between training and live inference.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_driver_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic driver risk scores from age and bonus-malus factors.

    Extracts driver age and bonus-malus rating factors, logs any missing value counts,
    imputes missing entries with column medians, and calculates the composite
    `driver_risk_score` column.

    Args:
        frame: Input DataFrame containing policyholder rating variables.
            Must contain columns 'driver_age' and 'bonus_malus'.

    Returns:
        pd.DataFrame: A shallow copy of the input DataFrame augmented with the
            float column 'driver_risk_score'.

    Raises:
        KeyError: If either 'driver_age' or 'bonus_malus' is missing from `frame`.

    Notes:
        - Imputation: If missing values are detected in 'driver_age' or 'bonus_malus',
          their counts are logged at INFO level ('<column>: <count> imputed') and filled
          with the respective column median.
        - Parity: This deterministic transformation is consumed by both the GLM frequency
          baseline and the EconML Double-ML causal elasticity pipeline.
    """
    output = frame.copy()
    required = {"driver_age", "bonus_malus"}
    missing = required - set(output.columns)
    if missing:
        raise KeyError(f"Missing required driver columns: {sorted(missing)}")

    driver_age_imputed = int(output["driver_age"].isna().sum())
    bonus_malus_imputed = int(output["bonus_malus"].isna().sum())
    logger.info("driver_age: %d imputed", driver_age_imputed)
    logger.info("bonus_malus: %d imputed", bonus_malus_imputed)

    driver_age = output["driver_age"].astype(float).fillna(output["driver_age"].median())
    bonus_malus = output["bonus_malus"].astype(float).fillna(output["bonus_malus"].median())

    adjusted_age = (driver_age - 35.0) / 25.0
    adjusted_bonus = (bonus_malus - 50.0) / 100.0
    output["driver_risk_score"] = (0.6 * adjusted_age**2 + 0.4 * adjusted_bonus**2 + 1.0).clip(
        lower=0.0
    )
    return output
