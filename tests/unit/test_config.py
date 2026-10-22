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


def test_feature_pipeline_config_values() -> None:
    """Verifies that approved P2-F02 bounds, bands, levels, and paths load."""
    config = load_params()
    fp = config.tier1_ml.feature_pipeline
    assert fp.exposure_cap == 1.0
    assert fp.driver_age_bounds == (18, 90)
    assert fp.car_age_bounds == (0, 34)
    assert fp.driver_age_band_edges == [18, 25, 45, 65, 100]
    assert len(fp.driver_age_band_labels) == 4
    assert len(fp.car_age_band_labels) == 3
    assert len(fp.density_band_labels) == 3
    assert len(fp.power_levels) == 12
    assert len(fp.brand_levels) == 7
    assert len(fp.gas_levels) == 2
    assert len(fp.region_levels) == 10
    assert config.tier1_ml.feature_paths.glm_view == "data/processed/features_glm.csv"


def _load_mutated_feature_config(tmp_path: Path, mutation: dict) -> ConfigurationError:
    """Writes params.yaml with a mutated feature_pipeline block and returns the load error."""
    base = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    base["tier1_ml"]["feature_pipeline"].update(mutation)
    target = tmp_path / "mutated_params.yaml"
    with target.open("w", encoding="utf-8") as f:
        yaml.dump(base, f)

    with pytest.raises(ConfigurationError) as exc_info:
        load_params(target)
    return exc_info.value


def test_feature_pipeline_band_label_mismatch_raises(tmp_path: Path) -> None:
    """Verifies that an edge/label count mismatch is rejected at config load."""
    error = _load_mutated_feature_config(
        tmp_path, {"driver_age_band_labels": ["18-24", "25-44", "45-64"]}
    )
    assert "Configuration validation failed" in str(error)


def test_feature_pipeline_non_monotonic_edges_raise(tmp_path: Path) -> None:
    """Verifies that non-increasing band edges are rejected at config load."""
    error = _load_mutated_feature_config(tmp_path, {"car_age_band_edges": [0, 10, 5, 101]})
    assert "Configuration validation failed" in str(error)


def test_feature_pipeline_duplicate_levels_raise(tmp_path: Path) -> None:
    """Verifies that duplicate factor levels are rejected at config load."""
    error = _load_mutated_feature_config(
        tmp_path, {"gas_levels": ["Regular", "Regular"]}
    )
    assert "Configuration validation failed" in str(error)


def test_glm_baseline_config_values() -> None:
    """Verifies that approved P2-F03 GLM defaults, grid, and paths load."""
    config = load_params()
    glm = config.tier1_ml.glm_baseline
    assert glm.default_cap == 100000.0
    assert glm.tweedie_p_min == 1.05
    assert glm.tweedie_p_max == 1.95
    assert glm.tweedie_p_step == 0.05
    assert glm.tweedie_p_fallback == 1.5
    assert glm.decile_count == 10
    assert glm.link == "log"
    assert config.tier1_ml.glm_paths.fitted_loss_cost == (
        "data/processed/glm_fitted_loss_cost.csv"
    )
    assert config.tier1_ml.glm_paths.model_metadata == "data/processed/glm_baseline_model.json"
    assert config.tier1_ml.glm_paths.calibration_report == (
        "data/processed/glm_calibration_report.json"
    )


def _load_mutated_glm_config(tmp_path: Path, mutation: dict) -> ConfigurationError:
    """Writes params.yaml with a mutated glm_baseline block and returns the load error."""
    base = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    base["tier1_ml"]["glm_baseline"].update(mutation)
    target = tmp_path / "mutated_glm_params.yaml"
    with target.open("w", encoding="utf-8") as f:
        yaml.dump(base, f)

    with pytest.raises(ConfigurationError) as exc_info:
        load_params(target)
    return exc_info.value


def test_glm_baseline_inverted_p_grid_raises(tmp_path: Path) -> None:
    """Verifies that p_min >= p_max is rejected at config load."""
    error = _load_mutated_glm_config(
        tmp_path, {"tweedie_p_min": 1.9, "tweedie_p_max": 1.1}
    )
    assert "Configuration validation failed" in str(error)


def test_glm_baseline_non_log_link_raises(tmp_path: Path) -> None:
    """Verifies that a non-log link is rejected at config load."""
    error = _load_mutated_glm_config(tmp_path, {"link": "identity"})
    assert "Configuration validation failed" in str(error)


def test_glm_baseline_fallback_out_of_range_raises(tmp_path: Path) -> None:
    """Verifies that a fallback power outside (1, 2) is rejected at config load."""
    error = _load_mutated_glm_config(tmp_path, {"tweedie_p_fallback": 1.0})
    assert "Configuration validation failed" in str(error)
