"""Unit tests for freMTPL2 dataset ingestion and schema mapping (ADR-011, ADR-019).

Verifies that:
1. Frequency and severity tables merge correctly with proper claim amount aggregation.
2. Canonical column renaming matches the expected contract.
3. Cleaned output DataFrame satisfies Great Expectations data contracts (INV-3).
4. Ingestion pipeline caches existing files without re-fetching.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from aegis.config.loader import get_config
from aegis.pipelines.data_contracts import validate_dataframe
from aegis.pipelines.feature.ingest import (
    clean_and_merge_fremtpl2,
    fetch_fremtpl2_data,
    ingest_fremtpl2_pipeline,
)
from aegis.utils.exceptions import DataContractError


@pytest.fixture
def mock_freq_df() -> pd.DataFrame:
    """Provides synthetic freMTPL2 frequency table matching OpenML schema."""
    return pd.DataFrame(
        {
            "IDpol": [1001.0, 1002.0, 1003.0, 1004.0],
            "ClaimNb": [0, 2, 1, 0],
            "Exposure": [1.0, 0.75, 0.50, 1.0],
            "Area": ["D", "B", "E", "C"],
            "VehPower": [5, 7, 6, 4],
            "VehAge": [3, 7, 1, 10],
            "DrivAge": [35, 52, 24, 67],
            "BonusMalus": [50, 112, 100, 60],
            "VehBrand": ["B12", "B1", "B2", "B3"],
            "VehGas": ["Regular", "Diesel", "Regular", "Regular"],
            "Density": [1200, 300, 4500, 800],
            "Region": ["R82", "R24", "R11", "R53"],
        }
    )


@pytest.fixture
def mock_sev_df() -> pd.DataFrame:
    """Provides synthetic freMTPL2 severity table matching OpenML schema."""
    return pd.DataFrame(
        {
            "IDpol": [1002.0, 1002.0, 1003.0],
            "ClaimAmount": [1200.50, 800.00, 3450.75],
        }
    )


def test_clean_and_merge_fremtpl2_structure(
    mock_freq_df: pd.DataFrame,
    mock_sev_df: pd.DataFrame,
) -> None:
    """Verifies that clean_and_merge_fremtpl2 applies canonical schema mapping and claims merge."""
    df = clean_and_merge_fremtpl2(mock_freq_df, mock_sev_df)

    # 1. Check row count preserved
    assert len(df) == 4

    # 2. Check canonical column names
    expected_cols = {
        "policy_id",
        "claim_count",
        "exposure",
        "driver_age",
        "veh_age",
        "claim_amount",
        "premium",
        "treatment_rate_change",
        "bonus_malus",
        "veh_power",
        "density",
        "area",
        "region",
        "veh_brand",
        "veh_gas",
    }
    assert expected_cols.issubset(set(df.columns))

    # 3. Check policy_id formatting
    assert df["policy_id"].iloc[0] == "POL-1001"
    assert df["policy_id"].iloc[1] == "POL-1002"

    # 4. Check severity aggregation
    # Policy 1001: 0 claims -> 0.0
    assert df.loc[df["policy_id"] == "POL-1001", "claim_amount"].iloc[0] == 0.0
    # Policy 1002: 1200.50 + 800.00 = 2000.50
    assert df.loc[df["policy_id"] == "POL-1002", "claim_amount"].iloc[0] == 2000.50
    # Policy 1003: 3450.75
    assert df.loc[df["policy_id"] == "POL-1003", "claim_amount"].iloc[0] == 3450.75

    # 5. Check derived premium and baseline treatment
    assert (df["premium"] > 0.0).all()
    assert (df["treatment_rate_change"] == 0.0).all()


def test_cleaned_fremtpl2_passes_data_contract(
    mock_freq_df: pd.DataFrame,
    mock_sev_df: pd.DataFrame,
) -> None:
    """Verifies that cleaned freMTPL2 DataFrame passes the elasticity GX suite (INV-3, ADR-019)."""
    config = get_config()
    df = clean_and_merge_fremtpl2(mock_freq_df, mock_sev_df)

    result = validate_dataframe(
        df=df,
        suite_path=config.data_contracts.elasticity_suite_path,
        raise_on_failure=False,
    )
    assert result.success is True
    assert len(result.failed_expectations) == 0


def test_ingest_pipeline_skips_when_file_exists(tmp_path: Path) -> None:
    """Verifies that ingest_fremtpl2_pipeline uses existing file without downloading (INV-10)."""
    target_file = tmp_path / "existing_dataset.csv"
    target_file.write_text("policy_id,driver_age\nPOL-1,30\n", encoding="utf-8")

    with patch("aegis.pipelines.feature.ingest.fetch_fremtpl2_data") as mock_fetch:
        result_path = ingest_fremtpl2_pipeline(output_path=target_file, force_download=False)
        assert result_path == target_file
        mock_fetch.assert_not_called()


def test_fetch_fremtpl2_error_handling() -> None:
    """Verifies that OpenML fetch failure raises typed DataContractError."""
    with (
        patch("aegis.pipelines.feature.ingest.fetch_openml", side_effect=RuntimeError("Network")),
        pytest.raises(DataContractError) as exc_info,
    ):
        fetch_fremtpl2_data()
    assert "Failed to fetch freMTPL2" in str(exc_info.value)

