"""Unit tests for the Regulatory Corpus Great Expectations data contract (INV-3, ADR-006).

Verifies that:
1. Valid regulatory corpus fixture passes cleanly with 0 failures.
2. Each malformed fixture trips specifically and exclusively its named expectation rule.
3. Hard blocking behavior (raise_on_failure=True) raises DataContractError.
"""

from pathlib import Path

import pytest

from aegis.config.loader import get_config
from aegis.pipelines.data_contracts import validate_regulatory_corpus
from aegis.utils.exceptions import DataContractError


@pytest.fixture
def config():
    """Returns application configuration."""
    return get_config()


def test_regulatory_valid_fixture_passes(config) -> None:
    """Verifies that the valid fixture passes 100% of data contract checks."""
    result = validate_regulatory_corpus(
        data_path=config.data_contracts.regulatory_valid_fixture,
        suite_path=config.data_contracts.regulatory_corpus_suite_path,
        raise_on_failure=False,
    )
    assert result.success is True
    assert result.successful_expectations == result.total_expectations
    assert len(result.failed_expectations) == 0
    assert len(result.failure_rules) == 0


def test_regulatory_missing_metadata_fixture_fails_specifically(config) -> None:
    """Verifies that missing effective_date triggers null expectation exclusively."""
    result = validate_regulatory_corpus(
        data_path=config.data_contracts.regulatory_invalid_missing_meta_fixture,
        suite_path=config.data_contracts.regulatory_corpus_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert len(result.failed_expectations) == 1
    expected_type = "expect_column_values_to_not_be_null"
    assert result.failed_expectations[0]["expectation_type"] == expected_type
    assert result.failed_expectations[0]["column"] == "effective_date"


def test_regulatory_empty_chunk_fixture_fails_specifically(config) -> None:
    """Verifies that empty chunk_text triggers length expectation exclusively."""
    result = validate_regulatory_corpus(
        data_path=config.data_contracts.regulatory_invalid_empty_fixture,
        suite_path=config.data_contracts.regulatory_corpus_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert len(result.failed_expectations) == 1
    expected_type = "expect_column_value_lengths_to_be_between"
    assert result.failed_expectations[0]["expectation_type"] == expected_type
    assert result.failed_expectations[0]["column"] == "chunk_text"


def test_regulatory_duplicate_fixture_fails_specifically(config) -> None:
    """Verifies that duplicate chunk_id triggers unique expectation exclusively (R-1)."""
    result = validate_regulatory_corpus(
        data_path=config.data_contracts.regulatory_invalid_duplicate_fixture,
        suite_path=config.data_contracts.regulatory_corpus_suite_path,
        raise_on_failure=False,
    )
    assert result.success is False
    assert len(result.failed_expectations) == 1
    expected_type = "expect_column_values_to_be_unique"
    assert result.failed_expectations[0]["expectation_type"] == expected_type
    assert result.failed_expectations[0]["column"] == "chunk_id"


def test_blocking_validation_raises_data_contract_error(config) -> None:
    """Verifies that raise_on_failure=True raises typed DataContractError (INV-3)."""
    with pytest.raises(DataContractError) as exc_info:
        validate_regulatory_corpus(
            data_path=config.data_contracts.regulatory_invalid_missing_meta_fixture,
            suite_path=config.data_contracts.regulatory_corpus_suite_path,
            raise_on_failure=True,
        )
    assert "Data contract validation failed" in str(exc_info.value)
    assert "effective_date" in str(exc_info.value)


def test_missing_data_file_raises_data_contract_error(tmp_path: Path, config) -> None:
    """Verifies that attempting to validate a non-existent file raises DataContractError."""
    non_existent = tmp_path / "does_not_exist.json"
    with pytest.raises(DataContractError) as exc_info:
        validate_regulatory_corpus(
            data_path=non_existent,
            suite_path=config.data_contracts.regulatory_corpus_suite_path,
            raise_on_failure=False,
        )
    assert "Regulatory data file not found" in str(exc_info.value)
