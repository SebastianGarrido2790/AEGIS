"""Shared Stage 2 feature engineering and dataset partitioning pipeline.

This module serves as the single source of truth for policy feature transformations
and grouped evaluation partitioning across the AEGIS system. By providing a unified
transformation entry point, it strictly enforces Training-Serving Parity (INV-10 / ADR-019):
the exact same mathematical transformations and schema validations executed during offline
model training (GLM frequency baseline, EconML CausalForestDML elasticity estimation)
are executed during online inference and agentic decision flows.

Core Pipeline Responsibilities:
    1. Schema Validation & Type Normalization:
       Enforces presence and strict casting of required actuarial fields (`policy_id`,
       `driver_age`, `veh_age`, `bonus_malus`, `veh_power`, `exposure`, `premium`).
    2. Submodule Orchestration:
       Chains exposure normalization (`normalize_exposure`), driver risk scoring
       (`compute_driver_risk`), and vehicle risk scoring (`compute_vehicle_risk`).
    3. Composite Risk Index Synthesis:
       Derives the canonical composite actuarial risk index:
           risk_index = 0.5 * driver_risk_score + 0.5 * vehicle_risk_score + exposure_normalized
       This metric acts as a primary confounder in synthetic treatment assignment and
       heterogeneity modifier in causal elasticity estimation.
    4. Grouped Policy Partitioning (`create_policy_split`):
       Partitions datasets into train and test sets by unique policy identifier rather than
       individual observation rows, strictly preventing multi-vehicle or longitudinal
       policy data leakage across evaluation boundaries.

Architectural Context:
    - Phase 2 Feature Pipeline (ADR-019).
    - Phase 2 Remediation Stage 5 (Imputation Logging and Split-Logic Documentation).
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from aegis.pipelines.feature.driver import compute_driver_risk
from aegis.pipelines.feature.exposure import normalize_exposure
from aegis.pipelines.feature.vehicle import compute_vehicle_risk


def build_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the shared feature matrix used across training and inference.

    Validates schema requirements, casts columns to standardized types, executes
    domain transformations (exposure normalization, driver risk score, vehicle risk score),
    computes the composite actuarial `risk_index`, and filters output to canonical audit columns.

    Args:
        frame: Input DataFrame containing policy-level insurance records.
            Must contain at minimum: 'policy_id', 'driver_age', 'veh_age', 'bonus_malus',
            'veh_power', 'exposure', 'premium', and 'density' (required by vehicle risk).

    Returns:
        pd.DataFrame: Transformed feature matrix sorted by `policy_id` containing the
            canonical audit features:
            - 'policy_id' (str)
            - 'driver_age' (float)
            - 'veh_age' (float)
            - 'bonus_malus' (float)
            - 'veh_power' (float)
            - 'exposure' (float)
            - 'exposure_normalized' (float)
            - 'driver_risk_score' (float)
            - 'vehicle_risk_score' (float)
            - 'risk_index' (float)
            - 'premium' (float)
            and optionally 'claim_amount', 'claim_count', 'treatment_rate_change' if present.

    Raises:
        ValueError: If `frame` is empty.
        KeyError: If any required columns are missing from `frame` or missing after transformation.

    Notes:
        - Idempotent and deterministic: Repeated calls on identical input yield
          bitwise identical output.
        - Training-serving parity: Consumed uniformly by offline training and
          online serving pipelines.
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
    """Create a grouped policy-level train/test split without data leakage.

    Partitions policy records into mutually exclusive train and test sets based on unique
    `policy_id` values. This grouped split guarantees that policies with multiple vehicles
    or multiple exposure periods are never split across train and test partitions, preventing
    target leakage.

    Sampling Mechanism:
        1. Canonical Lexicographic Sort:
           Unique policy IDs are extracted and sorted as strings. While standard string sorting
           places "POL-10" before "POL-2", this initial ordering is deterministic across platforms.
        2. Uniform Pseudo-Random Permutation:
           The sorted policy array is permuted using
           `pd.Series.sample(frac=1.0, random_state=random_state)`.
           Because the permutation is uniform, any deterministic initial ordering is statistically
           equivalent under shuffling and has zero bearing on split validity or randomness.
        3. Partition Assignment:
           The top `round(N * test_size)` policies from the permuted sequence form the test set;
           the remaining policies form the training set.

    Args:
        frame: Input DataFrame containing a 'policy_id' column.
        test_size: Proportion of unique policies to allocate to the test partition.
            Must satisfy 0.0 < test_size < 1.0. Defaults to 0.2 (20% test split).
        random_state: Random seed for reproducible permutation. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A 2-tuple containing:
            - train_frame (pd.DataFrame): Training subset with ~ (1 - test_size) of policies.
            - test_frame (pd.DataFrame): Testing subset with ~ test_size of policies.
            Both partitions have disjoint policy sets:
            `set(train.policy_id).isdisjoint(test.policy_id)`.

    Raises:
        KeyError: If 'policy_id' is missing from `frame`.
        ValueError: If `test_size` is not strictly between 0.0 and 1.0.
    """
    if "policy_id" not in frame.columns:
        raise KeyError("Policy split requires a policy_id column.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must fall strictly between 0 and 1.")

    # Grouped policy partitioning logic:
    # 1. Sort unique policy IDs lexicographically to establish a canonical, deterministic baseline
    #    ordering across platforms and environments before applying pseudo-random permutation.
    #    Note on lexicographic ordering: standard string sorting places "POL-10" before "POL-2"
    #    (e.g., ["POL-1", "POL-10", "POL-2"]). This behavior is deliberate and completely harmless:
    #    because the entire list is subsequently uniformly permuted via pd.Series.sample(frac=1.0)
    #    with a fixed random_state seed, any deterministic initial ordering is statistically
    #    equivalent under pseudo-random permutation. It has zero bearing on split validity.
    # 2. Permuting unique policy IDs rather than individual observation rows guarantees grouped
    #    splitting: all records associated with a given policy_id are assigned strictly to either
    #    train or test without overlap, preventing policy-level data leakage across partitions.
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
