"""Shared Stage 2 feature engineering pipeline.

This module is the single deterministic transformation used by Stage 2 and later
inference, ensuring that training and serving reuse the same logic.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from aegis.pipelines.feature.driver import compute_driver_risk
from aegis.pipelines.feature.exposure import normalize_exposure
from aegis.pipelines.feature.vehicle import compute_vehicle_risk


def build_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the shared feature matrix used across training and inference.

    The output contains the canonical policy-level features plus deterministic risk
    components required by the actuarial models in later stages.
    """
    if frame.empty:
        raise ValueError("Feature matrix construction requires a non-empty input frame.")

    output = frame.copy().sort_values("policy_id").reset_index(drop=True)
    required = {
        "policy_id",
        "driver_age",
        "veh_age",
        "bonus_malus",
        "veh_power",
        "exposure",
        "premium",
    }
    missing = required - set(output.columns)
    if missing:
        raise KeyError(f"Missing required feature columns: {sorted(missing)}")

    output = output.assign(
        policy_id=output["policy_id"].astype(str),
        driver_age=output["driver_age"].astype(float),
        veh_age=output["veh_age"].astype(float),
        bonus_malus=output["bonus_malus"].astype(float),
        veh_power=output["veh_power"].astype(float),
        exposure=output["exposure"].astype(float),
        premium=output["premium"].astype(float),
    )

    output = normalize_exposure(output)
    output = compute_driver_risk(output)
    output = compute_vehicle_risk(output)

    output["risk_index"] = (
        0.5 * output["driver_risk_score"]
        + 0.5 * output["vehicle_risk_score"]
        + output["exposure_normalized"]
    ).astype(float)

    audit_columns = [
        "policy_id",
        "driver_age",
        "veh_age",
        "bonus_malus",
        "veh_power",
        "exposure",
        "exposure_normalized",
        "driver_risk_score",
        "vehicle_risk_score",
        "risk_index",
        "premium",
    ]
    for column in ["claim_amount", "claim_count", "treatment_rate_change"]:
        if column in output.columns:
            audit_columns.append(column)
    for column in ["policy_id", "premium"]:
        if column not in output.columns:
            raise KeyError(f"Required output column missing after transformation: {column}")

    return cast(pd.DataFrame, output.loc[:, audit_columns])


def create_policy_split(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a grouped policy-level train/test split without overlap."""
    if "policy_id" not in frame.columns:
        raise KeyError("Policy split requires a policy_id column.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must fall strictly between 0 and 1.")

    policies = sorted(
        cast(list[str], frame["policy_id"].drop_duplicates().tolist()),
        key=str,
    )
    if len(policies) < 2:
        return frame.copy(), frame.iloc[0:0].copy()

    rng = pd.Series(range(len(policies)), index=policies).sample(
        frac=1.0,
        random_state=random_state,
    )
    test_count = max(1, round(len(policies) * test_size))
    selected = rng.iloc[:test_count].index.tolist()

    train_frame = frame.loc[~frame["policy_id"].isin(selected)].copy().reset_index(drop=True)
    test_frame = frame.loc[frame["policy_id"].isin(selected)].copy().reset_index(drop=True)
    return train_frame, test_frame
