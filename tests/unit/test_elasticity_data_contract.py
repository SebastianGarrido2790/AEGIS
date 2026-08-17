"""Unit tests for the Elasticity Training Great Expectations data contract (INV-3, ADR-006).

Verifies that:
1. Valid elasticity training fixture passes 100% of data contract checks.
2. Value range violations (e.g. negative exposure) specifically fail the range expectation.
3. Post-treatment feature leakage violations specifically trip ExpectNoPostTreatmentLeakage.
4. Blocking validation mode (raise_on_failure=True) raises typed DataContractError.
"""

from pathlib import Path

import pytest

from aegis.config.loader import get_config
from aegis.pipelines.data_contracts import validate_elasticity_data
from aegis.utils.exceptions import DataContractError


@pytest.fixture
def config():
    """Returns application configuration."""
    return get_config()


def test_elasticity_valid_fixture_passes(config) -> None:
    """Verifies that the valid elasticity fixture passes 100% of data contract checks."""
    result = validate_elasticity_data(
        data_path=config.data_contracts.elasticity_valid_fixture,
        suite_path=config.data_contracts.elasticity_suite_path,
        raise_on_failure=False,
    )
    assert result.success is True
    assert result.successful_expectations == result.total_expectations
    assert len(result.failed_expectations) == 0
    assert len(result.failure_rules) == 0


def test_elasticity_range_violation_fixture_fails_specifically(config) -> None:
    """Verifies that negative exposure triggers expect_column_values_to_be_between."""
    result = validate_elasticity_data(
        data_path=config.data_contracts.elasticity_invalid_range_fixture,
        suite_path=config.data_contracts.elasticity_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert len(result.failed_expectations) > 0

    failed_columns = [f["column"] for f in result.failed_expectations]
    assert "exposure" in failed_columns
    assert any("expect_column_values_to_be_between" in rule for rule in result.failure_rules)


def test_elasticity_leakage_fixture_fails_specifically(config) -> None:
    """Verifies that post-treatment feature leakage trips ExpectNoPostTreatmentLeakage."""
    result = validate_elasticity_data(
        data_path=config.data_contracts.elasticity_invalid_leakage_fixture,
        suite_path=config.data_contracts.elasticity_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert len(result.failed_expectations) > 0

    leakage_failures = [
        f for f in result.failed_expectations
        if f["expectation_type"] == "expect_no_post_treatment_leakage"
    ]
    assert len(leakage_failures) == 1
    leaked_cols = leakage_failures[0]["result"]["details"]["leaked_columns"]
    assert "post_treatment_retention" in leaked_cols or "churn_flag" in leaked_cols


def test_elasticity_blocking_validation_raises_data_contract_error(config) -> None:
    """Verifies that raise_on_failure=True raises typed DataContractError (INV-3)."""
    with pytest.raises(DataContractError) as exc_info:
        validate_elasticity_data(
            data_path=config.data_contracts.elasticity_invalid_leakage_fixture,
            suite_path=config.data_contracts.elasticity_suite_path,
            raise_on_failure=True,
        )
    assert "Data contract validation failed" in str(exc_info.value)
    assert "expect_no_post_treatment_leakage" in str(exc_info.value)


def test_elasticity_missing_file_raises_data_contract_error(tmp_path: Path, config) -> None:
    """Verifies that attempting to validate a non-existent file raises DataContractError."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(DataContractError) as exc_info:
        validate_elasticity_data(
            data_path=non_existent,
            suite_path=config.data_contracts.elasticity_suite_path,
            raise_on_failure=False,
        )
    assert "Elasticity training data file not found" in str(exc_info.value)
