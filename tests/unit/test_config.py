"""Unit tests for configuration schema, loader, and error handling."""

from pathlib import Path

import pytest
import yaml

from aegis.config.loader import load_params
from aegis.config.schema import AEGISConfig
from aegis.utils.exceptions import ConfigurationError


def test_load_valid_params(tmp_path: Path) -> None:
    """Verifies that the actual params.yaml loads cleanly and validates against the schema."""
    config = load_params()
    assert isinstance(config, AEGISConfig)
    assert config.gateway.model_name == "gpt-4o-mini"
    assert config.tier1_ml.test_size == 0.20
    assert config.tier2_agents.groundedness_threshold == 0.85
    assert config.governance.audit_storage_backend == "sqlite"
    assert config.data_contracts.fixtures_dir == "data_contracts/fixtures"
    assert config.dvc.remote_name == "local_storage"


def test_load_missing_file_raises_configuration_error(tmp_path: Path) -> None:
    """Verifies that attempting to load a non-existent configuration raises ConfigurationError."""
    non_existent = tmp_path / "non_existent_params.yaml"
    with pytest.raises(ConfigurationError) as exc_info:
        load_params(non_existent)
    assert "Configuration file not found" in str(exc_info.value)


def test_load_malformed_yaml_raises_configuration_error(tmp_path: Path) -> None:
    """Verifies that malformed YAML syntax fails loudly with ConfigurationError."""
    bad_yaml = tmp_path / "bad_syntax.yaml"
    bad_yaml.write_text("gateway: [unclosed_list", encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        load_params(bad_yaml)
    assert "Failed to parse YAML configuration" in str(exc_info.value)


def test_missing_required_key_raises_validation_error(tmp_path: Path) -> None:
    """Verifies that omitting a required top-level section triggers a typed ConfigurationError."""
    incomplete_data = {
        "gateway": {
            "model_name": "gpt-4o-mini",
            "fallback_model": "claude-3-5-sonnet",
        },
        # Missing tier1_ml, tier2_agents, governance, data_contracts, dvc
    }
    incomplete_yaml = tmp_path / "incomplete_params.yaml"
    with incomplete_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(incomplete_data, f)

    with pytest.raises(ConfigurationError) as exc_info:
        load_params(incomplete_yaml)
    assert "Configuration validation failed against schema" in str(exc_info.value)


def test_invalid_type_raises_validation_error(tmp_path: Path) -> None:
    """Verifies that an invalid data type (e.g. string for float) triggers ConfigurationError."""
    bad_type_data = {
        "gateway": {
            "model_name": "gpt-4o-mini",
            "fallback_model": "claude-3-5-sonnet",
            "temperature": "not-a-number",  # Should be float
        },
        "tier1_ml": {"test_size": 0.2},
        "tier2_agents": {"groundedness_threshold": 0.85},
        "governance": {"audit_storage_backend": "sqlite"},
        "data_contracts": {
            "elasticity_suite_path": "data_contracts/elasticity_suite.json",
            "regulatory_corpus_suite_path": "data_contracts/regulatory_suite.json",
            "elasticity_valid_fixture": "f1",
            "elasticity_invalid_leakage_fixture": "f2",
            "elasticity_invalid_range_fixture": "f3",
            "regulatory_valid_fixture": "f4",
            "regulatory_invalid_missing_meta_fixture": "f5",
            "regulatory_invalid_empty_fixture": "f6",
            "regulatory_invalid_duplicate_fixture": "f7",
        },
        "dvc": {"remote_name": "local_storage"},
    }
    bad_type_yaml = tmp_path / "bad_type_params.yaml"
    with bad_type_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(bad_type_data, f)

    with pytest.raises(ConfigurationError) as exc_info:
        load_params(bad_type_yaml)
    assert "Configuration validation failed against schema" in str(exc_info.value)


def test_extra_unknown_key_raises_validation_error(tmp_path: Path) -> None:
    """Verifies that unknown extra keys are forbidden by strict ConfigDict(extra='forbid')."""
    extra_key_data = {
        "gateway": {
            "model_name": "gpt-4o-mini",
            "fallback_model": "claude-3-5-sonnet",
            "unrecognized_field": "disallowed",
        },
        "tier1_ml": {},
        "tier2_agents": {},
        "governance": {},
        "data_contracts": {
            "elasticity_suite_path": "a",
            "regulatory_corpus_suite_path": "b",
            "elasticity_valid_fixture": "c",
            "elasticity_invalid_leakage_fixture": "d",
            "elasticity_invalid_range_fixture": "e",
            "regulatory_valid_fixture": "f",
            "regulatory_invalid_missing_meta_fixture": "g",
            "regulatory_invalid_empty_fixture": "h",
            "regulatory_invalid_duplicate_fixture": "i",
        },
        "dvc": {},
    }
    extra_key_yaml = tmp_path / "extra_key_params.yaml"
    with extra_key_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(extra_key_data, f)

    with pytest.raises(ConfigurationError) as exc_info:
        load_params(extra_key_yaml)
    assert "Configuration validation failed against schema" in str(exc_info.value)
