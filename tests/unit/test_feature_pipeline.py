"""Unit test suite for Stage 2 feature engineering pipeline and grouped partitioning.

This suite validates the core feature pipeline components defined in
`aegis.pipelines.feature.pipeline`, `driver.py`, and `vehicle.py`. It tests:
    1. Deterministic Execution: Ensuring bitwise identical feature output across runs.
    2. Data Leakage Prevention: Ensuring grouped policy splits partition disjoint policy ID sets.
    3. Schema & Type Integrity: Ensuring canonical output columns and numeric dtypes exist.
    4. Imputation Logging (Stage 5): Ensuring visible, exact log outputs for missing values.
    5. Clean Fixture Logging (Stage 5): Ensuring zero-imputation counts are explicitly logged.
    6. Lexicographic Stability (Stage 5): Ensuring string sorting quirks ("POL-10" < "POL-2")
       do not affect split validity, coverage, or disjointness.
"""

import logging

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from aegis.pipelines.feature.pipeline import build_feature_matrix, create_policy_split


@pytest.fixture
def policy_frame() -> pd.DataFrame:
    """Provide a canonical synthetic policy DataFrame for feature transformation testing."""
    return pd.DataFrame(
        {
            "policy_id": ["POL-1", "POL-2", "POL-3", "POL-4"],
            "driver_age": [25.0, 42.0, 59.0, 33.0],
            "veh_age": [1.0, 5.0, 9.0, 3.0],
            "bonus_malus": [50.0, 75.0, 95.0, 60.0],
            "veh_power": [4.0, 6.0, 7.0, 5.0],
            "exposure": [0.8, 1.2, 0.6, 1.0],
            "claim_count": [0.0, 1.0, 2.0, 0.0],
            "claim_amount": [0.0, 2500.0, 4600.0, 0.0],
            "area": ["A", "B", "C", "A"],
            "region": ["R82", "R24", "R11", "R82"],
            "density": [1800.0, 820.0, 4300.0, 1700.0],
            "premium": [250.0, 420.0, 610.0, 280.0],
        }
    )


def test_build_feature_matrix_is_deterministic(policy_frame: pd.DataFrame) -> None:
    """Verify that build_feature_matrix produces bitwise identical outputs across executions."""
    left = build_feature_matrix(policy_frame.copy())
    right = build_feature_matrix(policy_frame.copy())

    assert_frame_equal(left, right)
    assert {"policy_id", "driver_age", "veh_age", "exposure", "premium"}.issubset(left.columns)
    assert bool(left["exposure_normalized"].notna().all())
    assert bool(left["driver_risk_score"].notna().all())


def test_create_policy_split_has_no_overlap(policy_frame: pd.DataFrame) -> None:
    """Verify that create_policy_split partitions policies without overlap or leakage."""
    train_df, test_df = create_policy_split(policy_frame, test_size=0.5, random_state=42)

    assert set(train_df["policy_id"]).isdisjoint(set(test_df["policy_id"]))
    assert len(train_df) + len(test_df) == len(policy_frame)
    assert set(train_df["policy_id"]).union(set(test_df["policy_id"])) == set(
        policy_frame["policy_id"]
    )


def test_build_feature_matrix_schema_is_valid(policy_frame: pd.DataFrame) -> None:
    """Verify that build_feature_matrix contains all required audit columns and data types."""
    matrix = build_feature_matrix(policy_frame)

    expected = {
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
    }
    assert expected.issubset(set(matrix.columns))
    assert pd.api.types.is_numeric_dtype(matrix["exposure_normalized"])
    assert pd.api.types.is_numeric_dtype(matrix["risk_index"])


def test_build_feature_matrix_imputation_logging_with_nulls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Deliberately nulled fixture must produce visible, correct imputation counts in logs."""
    nulled_frame = pd.DataFrame(
        {
            "policy_id": ["POL-1", "POL-2", "POL-3", "POL-4", "POL-5"],
            "driver_age": [25.0, None, 59.0, None, 45.0],  # 2 nulls
            "bonus_malus": [50.0, 75.0, None, 60.0, 80.0],  # 1 null
            "veh_age": [1.0, None, 9.0, 3.0, None],  # 2 nulls
            "veh_power": [4.0, 6.0, None, 5.0, 7.0],  # 1 null
            "exposure": [0.8, 1.2, 0.6, 1.0, 0.5],
            "density": [1800.0, None, None, 1700.0, None],  # 3 nulls
            "premium": [250.0, 420.0, 610.0, 280.0, 350.0],
        }
    )

    with caplog.at_level(logging.INFO):
        matrix = build_feature_matrix(nulled_frame)

    # Verify imputation output is clean and non-null
    assert bool(matrix["driver_risk_score"].notna().all())
    assert bool(matrix["vehicle_risk_score"].notna().all())
    assert bool(matrix["risk_index"].notna().all())

    # Verify logged imputation counts are visible and exact
    log_text = caplog.text
    assert "driver_age: 2 imputed" in log_text
    assert "bonus_malus: 1 imputed" in log_text
    assert "veh_age: 2 imputed" in log_text
    assert "veh_power: 1 imputed" in log_text
    assert "density: 3 imputed" in log_text


def test_build_feature_matrix_imputation_logging_zero_when_clean(
    policy_frame: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Clean fixture must explicitly log '0 imputed' rather than remaining silent."""
    with caplog.at_level(logging.INFO):
        _ = build_feature_matrix(policy_frame)

    log_text = caplog.text
    assert "driver_age: 0 imputed" in log_text
    assert "bonus_malus: 0 imputed" in log_text
    assert "veh_age: 0 imputed" in log_text
    assert "veh_power: 0 imputed" in log_text
    assert "density: 0 imputed" in log_text


def test_create_policy_split_lexicographic_order_stability() -> None:
    """Verifies that lexicographic order ('POL-10' before 'POL-2') is stable and leak-free."""
    frame = pd.DataFrame(
        {
            "policy_id": ["POL-1", "POL-2", "POL-10", "POL-20"],
            "driver_age": [25.0, 35.0, 45.0, 55.0],
            "veh_age": [2.0, 4.0, 6.0, 8.0],
            "bonus_malus": [50.0, 60.0, 70.0, 80.0],
            "veh_power": [4.0, 5.0, 6.0, 7.0],
            "exposure": [1.0, 1.0, 1.0, 1.0],
            "density": [1000.0, 1200.0, 1400.0, 1600.0],
            "premium": [300.0, 350.0, 400.0, 450.0],
        }
    )

    # Standard string sort orders "POL-10" before "POL-2"
    unique_ids = sorted(frame["policy_id"].tolist(), key=str)
    assert unique_ids == ["POL-1", "POL-10", "POL-2", "POL-20"]

    # create_policy_split yields disjoint non-leaking partitions despite the string sorting quirk
    train_df, test_df = create_policy_split(frame, test_size=0.5, random_state=42)
    assert set(train_df["policy_id"]).isdisjoint(set(test_df["policy_id"]))
    assert len(train_df) == 2
    assert len(test_df) == 2
    assert set(train_df["policy_id"]).union(set(test_df["policy_id"])) == {
        "POL-1",
        "POL-2",
        "POL-10",
        "POL-20",
    }

