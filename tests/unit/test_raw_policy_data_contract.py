"""Unit tests for the Raw Policy Great Expectations data contract (INV-3, ADR-015, ADR-003).

Verifies that:
1. Checksum manifest verification passes on genuine freMTPL files and fails on tampered data.
2. Policy-level joining logic correctly aggregates multi-claim severity records via
   sum of ClaimAmount.
3. Valid raw policy fixture passes 100% of data contract checks (20/20 expectations).
4. Each malformed fixture fails its specific targeted expectation rule.
5. Falsification: orphan severity records specifically fail referential integrity and block DAG.
6. CLI subcommands ingest-raw, validate-raw, and version-raw execute correctly.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aegis.config.loader import get_config
from aegis.pipelines.cli import main
from aegis.pipelines.data_contracts import validate_raw_policy_data
from aegis.pipelines.raw_intake import (
    compute_sha256,
    load_and_join_raw_policy_data,
    verify_checksums,
)
from aegis.utils.exceptions import DataContractError


@pytest.fixture
def config():
    """Returns application configuration."""
    return get_config()


def test_compute_sha256(tmp_path: Path) -> None:
    """Verifies that compute_sha256 computes correct hexadecimal hashes."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("AEGIS Data Contract Test", encoding="utf-8")
    hash_val = compute_sha256(test_file)
    assert len(hash_val) == 64
    assert hash_val.islower()


def test_verify_checksums_genuine_manifest(config) -> None:
    """Verifies that actual freMTPL source files match the committed manifest."""
    records = verify_checksums(manifest_path=config.data_contracts.raw_intake_manifest_path)
    assert len(records) == 2
    for rec in records:
        assert rec["status"] == "VERIFIED"


def test_verify_checksums_tampered_file_fails(tmp_path: Path) -> None:
    """Verifies that a tampered file raises DataContractError during checksum verification."""
    dummy_file = tmp_path / "dummy.csv"
    dummy_file.write_text("PolicyID,ClaimNb\n1,0", encoding="utf-8")

    manifest = {
        "files": [
            {
                "path": str(dummy_file),
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "size_bytes": 100,
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataContractError) as exc_info:
        verify_checksums(manifest_path=manifest_path, base_dir=tmp_path)
    assert "checksum verification failed" in str(exc_info.value).lower()


def test_load_and_join_aggregates_multiple_claims(tmp_path: Path) -> None:
    """Verifies that multi-claim severity records are summed per policy."""
    freq_data = (
        "PolicyID,ClaimNb,Exposure,Power,CarAge,DriverAge,Brand,Gas,Region,Density\n"
        "101,0,0.5,d,1,30,Renault,Regular,Centre,100\n"
        "102,1,0.8,e,5,40,Peugeot,Diesel,Aquitaine,200\n"
        "103,3,1.0,f,2,25,Citroen,Regular,Ile-de-France,1500\n"
    )
    sev_data = (
        "PolicyID,ClaimAmount\n"
        "102,1200.0\n"
        "103,500.0\n"
        "103,1500.0\n"
        "103,750.0\n"
    )
    freq_file = tmp_path / "freq.csv"
    sev_file = tmp_path / "sev.csv"
    freq_file.write_text(freq_data, encoding="utf-8")
    sev_file.write_text(sev_data, encoding="utf-8")

    joined = load_and_join_raw_policy_data(freq_path=freq_file, sev_path=sev_file)

    assert len(joined) == 3
    # Policy 101: 0 claims
    row_101 = joined[joined["PolicyID"] == 101].iloc[0]
    assert row_101["ClaimNb"] == 0
    assert row_101["ClaimAmount"] == 0.0
    assert row_101["sev_claim_count"] == 0
    assert row_101["is_orphan_severity"] is False or row_101["is_orphan_severity"] == 0

    # Policy 102: 1 claim of 1200.0
    row_102 = joined[joined["PolicyID"] == 102].iloc[0]
    assert row_102["ClaimNb"] == 1
    assert row_102["ClaimAmount"] == 1200.0
    assert row_102["sev_claim_count"] == 1

    # Policy 103: 3 claims summing to 2750.0
    row_103 = joined[joined["PolicyID"] == 103].iloc[0]
    assert row_103["ClaimNb"] == 3
    assert row_103["ClaimAmount"] == 2750.0
    assert row_103["sev_claim_count"] == 3


def test_raw_policy_valid_fixture_passes_all_expectations(config) -> None:
    """Verifies that the valid raw policy fixture passes all 20 data contract checks."""
    result = validate_raw_policy_data(
        data_path=config.data_contracts.raw_policy_valid_fixture,
        suite_path=config.data_contracts.raw_policy_suite_path,
        raise_on_failure=False,
    )
    assert result.success is True
    assert result.successful_expectations == result.total_expectations
    assert result.total_expectations == 20
    assert len(result.failed_expectations) == 0
    assert len(result.failure_rules) == 0


def test_raw_policy_invalid_duplicate_id_fails_specifically(config) -> None:
    """Verifies that duplicate PolicyID triggers expect_column_values_to_be_unique."""
    bad_fixture = Path(config.data_contracts.fixtures_dir) / "raw_policy_invalid_duplicate_id.csv"
    result = validate_raw_policy_data(
        data_path=bad_fixture,
        suite_path=config.data_contracts.raw_policy_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert any(
        "expect_column_values_to_be_unique on column 'PolicyID'" in r for r in result.failure_rules
    )


def test_raw_policy_invalid_claim_nb_range_fails_specifically(config) -> None:
    """Verifies that ClaimNb > 4 triggers expect_column_values_to_be_between."""
    bad_fixture = Path(config.data_contracts.fixtures_dir) / "raw_policy_invalid_claim_nb_range.csv"
    result = validate_raw_policy_data(
        data_path=bad_fixture,
        suite_path=config.data_contracts.raw_policy_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert any(
        "expect_column_values_to_be_between on column 'ClaimNb'" in r for r in result.failure_rules
    )


def test_raw_policy_invalid_exposure_range_fails_specifically(config) -> None:
    """Verifies that Exposure > 2.0 triggers expect_column_values_to_be_between."""
    bad_fixture = Path(config.data_contracts.fixtures_dir) / "raw_policy_invalid_exposure_range.csv"
    result = validate_raw_policy_data(
        data_path=bad_fixture,
        suite_path=config.data_contracts.raw_policy_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert any(
        "expect_column_values_to_be_between on column 'Exposure'" in r
        for r in result.failure_rules
    )


def test_raw_policy_invalid_exposure_tolerance_fails_specifically(config) -> None:
    """Verifies that too many policies exceeding 1.0 triggers the tolerance expectation."""
    bad_fixture = (
        Path(config.data_contracts.fixtures_dir) / "raw_policy_invalid_exposure_tolerance.csv"
    )
    result = validate_raw_policy_data(
        data_path=bad_fixture,
        suite_path=config.data_contracts.raw_policy_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert any(
        "expect_column_values_to_be_between on column 'Exposure'" in r
        for r in result.failure_rules
    )


def test_raw_policy_invalid_claim_amount_zero_fails_specifically(config) -> None:
    """Verifies that ClaimAmount == 0 when ClaimNb > 0 triggers conditional expectation."""
    bad_fixture = (
        Path(config.data_contracts.fixtures_dir) / "raw_policy_invalid_claim_amount_zero.csv"
    )
    result = validate_raw_policy_data(
        data_path=bad_fixture,
        suite_path=config.data_contracts.raw_policy_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert any(
        "expect_column_values_to_be_between on column 'ClaimAmount'" in r
        for r in result.failure_rules
    )


def test_raw_policy_invalid_claim_count_mismatch_fails_specifically(config) -> None:
    """Verifies that ClaimNb != sev_claim_count triggers expect_column_pair_values_to_be_equal."""
    bad_fixture = (
        Path(config.data_contracts.fixtures_dir) / "raw_policy_invalid_claim_count_mismatch.csv"
    )
    result = validate_raw_policy_data(
        data_path=bad_fixture,
        suite_path=config.data_contracts.raw_policy_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert any("expect_column_pair_values_to_be_equal" in r for r in result.failure_rules)


def test_raw_policy_invalid_null_frequency_fails_specifically(config) -> None:
    """Verifies that null in frequency factor triggers expect_column_values_to_not_be_null."""
    bad_fixture = (
        Path(config.data_contracts.fixtures_dir) / "raw_policy_invalid_null_frequency.csv"
    )
    result = validate_raw_policy_data(
        data_path=bad_fixture,
        suite_path=config.data_contracts.raw_policy_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert any(
        "expect_column_values_to_not_be_null on column 'DriverAge'" in r
        for r in result.failure_rules
    )


def test_falsification_orphan_severity_fails_referential_integrity(config) -> None:
    """Selective Falsification: orphan severity records fail referential integrity (INV-3)."""
    bad_fixture = (
        Path(config.data_contracts.fixtures_dir) / "raw_policy_invalid_orphan_severity.csv"
    )
    result = validate_raw_policy_data(
        data_path=bad_fixture,
        suite_path=config.data_contracts.raw_policy_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    # Verifies both custom expectation and in-set flag trigger failure
    assert any("expect_zero_orphan_severity_records" in r for r in result.failure_rules)
    assert any("is_orphan_severity" in r for r in result.failure_rules)

    # Verifies that raise_on_failure=True raises typed DataContractError
    with pytest.raises(DataContractError) as exc_info:
        validate_raw_policy_data(
            data_path=bad_fixture,
            suite_path=config.data_contracts.raw_policy_suite_path,
            raise_on_failure=True,
        )
    assert "Data contract validation failed" in str(exc_info.value)
    assert "expect_zero_orphan_severity_records" in str(exc_info.value)


def test_cli_raw_pipeline_commands(tmp_path: Path, config) -> None:
    """Verifies end-to-end CLI execution for ingest-raw, validate-raw, and version-raw."""
    joined_csv = tmp_path / "raw_policy.csv"
    report_json = tmp_path / "report.json"
    versioned_csv = tmp_path / "versioned_policy.csv"

    # 1. Ingest raw (using valid fixture as test source)
    with patch(
        "sys.argv",
        [
            "cli.py",
            "ingest-raw",
            "--freq-path",
            "data/raw/freMTPLfreq.csv.zip",
            "--sev-path",
            "data/raw/freMTPLsev.csv",
            "--output-path",
            str(joined_csv),
        ],
    ):
        code_ingest = main()
        assert code_ingest == 0
        assert joined_csv.is_file()

    # 2. Validate raw
    with patch(
        "sys.argv",
        [
            "cli.py",
            "validate-raw",
            "--data-path",
            str(joined_csv),
            "--suite-path",
            config.data_contracts.raw_policy_suite_path,
            "--report-path",
            str(report_json),
        ],
    ):
        code_val = main()
        assert code_val == 0
        assert report_json.is_file()
        with report_json.open("r", encoding="utf-8") as f:
            report_data = json.load(f)
        assert report_data["success"] is True
        assert report_data["successful_expectations"] == 20

    # 3. Version raw
    with patch(
        "sys.argv",
        [
            "cli.py",
            "version-raw",
            "--raw-path",
            str(joined_csv),
            "--report-path",
            str(report_json),
            "--output-path",
            str(versioned_csv),
        ],
    ):
        code_ver = main()
        assert code_ver == 0
        assert versioned_csv.is_file()
