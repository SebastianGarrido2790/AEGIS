"""Unit tests for the raw freMTPL exploratory data analysis script.

Verifies that:
1. Datasets load correctly from zip archives and CSVs.
2. Metadata, relational integrity, frequency, and severity statistics are accurate.
3. The fundamental actuarial pure premium identity holds.
4. CLI execution runs cleanly, generating valid JSON and Markdown outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from scripts.explore_raw_data import (
    DEFAULT_FREQ_PATH,
    DEFAULT_SEV_PATH,
    analyze_all_factors,
    analyze_frequency,
    analyze_pure_premium,
    analyze_severity,
    audit_relational_integrity,
    detect_anomalies,
    inspect_metadata,
    main,
)


@pytest.fixture
def mock_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provides a minimal in-memory actuarial dataset for fast isolated unit testing."""
    freq_data = {
        "PolicyID": [101, 102, 103, 104, 105],
        "ClaimNb": [0, 1, 2, 0, 1],
        "Exposure": [0.50, 1.00, 0.75, 0.25, 1.00],
        "Power": ["d", "e", "f", "d", "g"],
        "CarAge": [2, 0, 8, 12, 16],
        "DriverAge": [22, 35, 45, 60, 78],
        "Brand": ["Fiat", "Renault, Nissan or Citroen", "Fiat", "other", "Fiat"],
        "Gas": ["Regular", "Diesel", "Regular", "Diesel", "Regular"],
        "Region": ["Centre", "Ile-de-France", "Centre", "Bretagne", "Aquitaine"],
        "Density": [50, 5000, 200, 80, 12000],
    }
    sev_data = {
        "PolicyID": [102, 103, 103, 105],
        "ClaimAmount": [1200, 800, 2500, 500],
    }
    return pd.DataFrame(freq_data), pd.DataFrame(sev_data)


def test_inspect_metadata(tmp_path: Path, mock_datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verifies that metadata inspection accurately computes rows, columns, and memory."""
    freq_df, _ = mock_datasets
    sample_file = tmp_path / "sample_freq.csv"
    freq_df.to_csv(sample_file, index=False)

    meta = inspect_metadata(sample_file, freq_df, is_zip=False)
    assert meta.num_rows == 5
    assert meta.num_cols == 10
    assert "PolicyID" in meta.columns
    assert sum(meta.missing_values.values()) == 0


def test_audit_relational_integrity(mock_datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verifies relational integrity and discrepancy tracking."""
    freq_df, sev_df = mock_datasets
    audit = audit_relational_integrity(freq_df, sev_df)

    assert audit.freq_total_policies == 5
    assert audit.freq_unique_policies == 5
    assert audit.sev_total_claims == 4
    assert audit.sev_unique_policies == 3
    assert audit.orphan_sev_records == 0
    assert audit.count_mismatch_records == 0
    assert audit.perfect_match is True
    # Policy 103 has 2 claims, others have 1
    assert audit.claims_per_policy_distribution[1] == 2
    assert audit.claims_per_policy_distribution[2] == 1


def test_actuarial_frequency_and_severity(
    mock_datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verifies actuarial frequency, severity, and pure premium computations."""
    freq_df, sev_df = mock_datasets

    freq_res = analyze_frequency(freq_df)
    assert freq_res.total_exposure_years == 3.50
    assert freq_res.total_claim_count == 4
    assert pytest.approx(freq_res.empirical_frequency, rel=1e-3) == 4 / 3.5

    sev_res = analyze_severity(sev_df)
    assert sev_res.total_loss_amount == 5000.0
    assert sev_res.num_claims == 4
    assert sev_res.mean_severity == 1250.0
    assert sev_res.min_severity == 500.0
    assert sev_res.max_severity == 2500.0

    pure_prem = analyze_pure_premium(freq_df, sev_df)
    assert pytest.approx(pure_prem.empirical_pure_premium, rel=1e-3) == 5000.0 / 3.5
    assert pure_prem.identity_discrepancy < 1e-4


def test_factor_segmentation(mock_datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verifies that all categorical and continuous rating factors are segmented."""
    freq_df, sev_df = mock_datasets
    factors = analyze_all_factors(freq_df, sev_df)

    expected_factors = ["Gas", "Power", "Brand", "Region", "DriverAge", "CarAge", "Density"]
    for factor in expected_factors:
        assert factor in factors
        assert len(factors[factor]) > 0
        total_pols = sum(r["policies"] for r in factors[factor])
        assert total_pols == 5


def test_anomaly_detection(mock_datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verifies detection of extreme claims, out-of-bound exposures, and ages."""
    freq_df, sev_df = mock_datasets
    anomalies = detect_anomalies(freq_df, sev_df)

    assert anomalies["zero_or_negative_exposure_count"] == 0
    assert anomalies["exposure_greater_than_one_count"] == 0
    assert anomalies["claims_exceeding_100k_count"] == 0
    assert anomalies["underage_drivers_count"] == 0


def test_cli_execution_with_artifacts(tmp_path: Path) -> None:
    """Verifies end-to-end CLI execution with JSON and Markdown export."""
    if not DEFAULT_FREQ_PATH.is_file() or not DEFAULT_SEV_PATH.is_file():
        pytest.skip("Raw benchmark datasets not available for full integration test.")

    json_out = tmp_path / "eda_test.json"
    md_out = tmp_path / "eda_test.md"

    exit_code = main([
        "--sample-rows", "2000",
        "--output-json", str(json_out),
        "--output-markdown", str(md_out),
        "--quiet",
    ])

    assert exit_code == 0
    assert json_out.is_file()
    assert md_out.is_file()

    with json_out.open(encoding="utf-8") as f:
        data = json.load(f)
    assert "metadata" in data
    assert "relational_audit" in data
    assert "pure_premium" in data

    md_content = md_out.read_text(encoding="utf-8")
    assert "# freMTPL Benchmark Raw Data Exploratory Report" in md_content
