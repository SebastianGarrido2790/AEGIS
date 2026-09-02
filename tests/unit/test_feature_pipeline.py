import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from aegis.pipelines.feature.pipeline import build_feature_matrix, create_policy_split


@pytest.fixture
def policy_frame() -> pd.DataFrame:
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
    left = build_feature_matrix(policy_frame.copy())
    right = build_feature_matrix(policy_frame.copy())

    assert_frame_equal(left, right)
    assert {"policy_id", "driver_age", "veh_age", "exposure", "premium"}.issubset(left.columns)
    assert left["exposure_normalized"].notna().all()
    assert left["driver_risk_score"].notna().all()


def test_create_policy_split_has_no_overlap(policy_frame: pd.DataFrame) -> None:
    train_df, test_df = create_policy_split(policy_frame, test_size=0.5, random_state=42)

    assert set(train_df["policy_id"]).isdisjoint(set(test_df["policy_id"]))
    assert len(train_df) + len(test_df) == len(policy_frame)
    assert set(train_df["policy_id"]).union(set(test_df["policy_id"])) == set(
        policy_frame["policy_id"]
    )


def test_build_feature_matrix_schema_is_valid(policy_frame: pd.DataFrame) -> None:
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
