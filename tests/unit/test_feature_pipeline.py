"""Unit tests for the deterministic feature pipeline (P2-F02, ADR-012).

Covers typed metadata contracts, input validation, derivations, dual views,
artifact determinism, and the CLI/DVC integration.
"""

import hashlib
from math import log
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from aegis.config.loader import get_config
from aegis.pipelines.feature_pipeline import (
    GLM_COLUMNS,
    derive_features,
    run_feature_pipeline,
    transform_features,
    validate_feature_input,
    write_feature_artifacts,
)
from aegis.schemas.features import FeatureMetadata
from aegis.utils.exceptions import FeaturePipelineError

SHA = "a" * 64


@pytest.fixture
def fp_config():
    """Returns the approved feature-pipeline configuration."""
    return get_config().tier1_ml.feature_pipeline


def _valid_frame() -> pd.DataFrame:
    """Builds a minimal valid raw-policy frame."""
    return pd.DataFrame(
        {
            "PolicyID": [1, 2, 3],
            "ClaimNb": [0, 1, 0],
            "Exposure": [0.5, 0.85, 1.0],
            "Power": ["d", "e", "f"],
            "CarAge": [2, 5, 0],
            "DriverAge": [24, 32, 45],
            "Brand": [
                "Renault, Nissan or Citroen",
                "Fiat",
                "other",
            ],
            "Gas": ["Regular", "Diesel", "Regular"],
            "Region": ["Centre", "Aquitaine", "Bretagne"],
            "Density": [120, 450, 2500],
            "ClaimAmount": [0.0, 1450.0, 0.0],
            "sev_claim_count": [0, 1, 0],
            "is_orphan_severity": [False, False, False],
        }
    )


def _band(edges: list[float], labels: list[str]) -> dict[str, Any]:
    """Builds a band-spec payload."""
    return {"edges": edges, "labels": labels}


def _numeric() -> dict[str, float | int]:
    """Builds a numeric-summary payload."""
    return {
        "count": 2,
        "min": 0.0,
        "max": 1.0,
        "mean": 0.5,
        "std": 0.5,
        "p01": 0.0,
        "p25": 0.0,
        "p50": 0.5,
        "p75": 1.0,
        "p99": 1.0,
    }


def _metadata_payload() -> dict[str, Any]:
    """Builds a minimal valid FeatureMetadata payload."""
    return {
        "input_rows": 2,
        "output_rows": 2,
        "input_sha256": SHA,
        "config_fingerprint": SHA,
        "exposure_clip_count": 0,
        "tail_clip_counts": {
            "driver_age_below": 0,
            "driver_age_above": 0,
            "car_age_below": 0,
            "car_age_above": 0,
        },
        "band_edges": {
            "driver_age": _band([18, 25, 45, 65, 100], ["18-24", "25-44", "45-64", "65+"]),
            "car_age": _band([0, 5, 10, 101], ["0-4", "5-9", "10+"]),
            "density": _band([0, 501, 2001, 1000000000], ["<=500", "501-2000", ">2000"]),
        },
        "level_mappings": {"gas": ["Regular", "Diesel"]},
        "one_hot_columns": ["gas__regular", "gas__diesel"],
        "forest_covariates": ["log_density", "power_ord"],
        "glm_view": {"columns": ["PolicyID", "ClaimAmount"], "row_count": 2, "sha256": SHA},
        "forest_view": {"columns": ["PolicyID", "log_density"], "row_count": 2, "sha256": SHA},
        "distribution_summary": {
            "numeric": {"log_density": _numeric()},
            "categorical": {"gas": {"counts": {"Regular": 2}, "missing": 0}},
        },
    }


def test_metadata_round_trip() -> None:
    """Verifies that valid metadata validates and serializes without loss."""
    metadata = FeatureMetadata.model_validate(_metadata_payload())
    restored = FeatureMetadata.model_validate_json(metadata.model_dump_json())
    assert restored == metadata


def test_metadata_rejects_missing_field() -> None:
    """Verifies that metadata missing a required field is rejected."""
    payload = _metadata_payload()
    del payload["input_sha256"]
    with pytest.raises(ValidationError):
        FeatureMetadata.model_validate(payload)


def test_metadata_rejects_extra_field() -> None:
    """Verifies that metadata with an unknown field is rejected (schema stability)."""
    payload = _metadata_payload()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        FeatureMetadata.model_validate(payload)


def test_metadata_rejects_invalid_sha() -> None:
    """Verifies that non-SHA-256 hash fields are rejected."""
    payload = _metadata_payload()
    payload["input_sha256"] = "not-a-hash"
    with pytest.raises(ValidationError):
        FeatureMetadata.model_validate(payload)


def test_feature_pipeline_error_carries_details() -> None:
    """Verifies that the typed error exposes column, rule, count, and sample keys."""
    error = FeaturePipelineError(
        "bad input",
        details={"column": "Gas", "rule": "unknown_level", "count": 1, "sample_keys": [7]},
    )
    assert error.details["column"] == "Gas"
    assert error.details["rule"] == "unknown_level"
    assert error.details["count"] == 1
    assert "bad input" in str(error)


def test_validate_input_accepts_valid_frame(fp_config) -> None:
    """Verifies that the valid raw frame passes input validation."""
    validate_feature_input(_valid_frame(), fp_config)


def test_validate_input_rejects_empty_frame(fp_config) -> None:
    """Verifies that an empty input raises empty_input."""
    empty = _valid_frame().iloc[0:0]
    with pytest.raises(FeaturePipelineError) as exc_info:
        validate_feature_input(empty, fp_config)
    assert exc_info.value.details["rule"] == "empty_input"


def test_validate_input_rejects_missing_column(fp_config) -> None:
    """Verifies that a missing required column is named."""
    frame = _valid_frame().drop(columns=["Gas"])
    with pytest.raises(FeaturePipelineError) as exc_info:
        validate_feature_input(frame, fp_config)
    assert exc_info.value.details["rule"] == "missing_columns"
    assert "Gas" in exc_info.value.details["sample_keys"]


def test_validate_input_rejects_unexpected_column(fp_config) -> None:
    """Verifies that schema drift via an extra column is rejected."""
    frame = _valid_frame().assign(composite_risk=1.0)
    with pytest.raises(FeaturePipelineError) as exc_info:
        validate_feature_input(frame, fp_config)
    assert exc_info.value.details["rule"] == "unexpected_columns"
    assert exc_info.value.details["sample_keys"] == ["composite_risk"]


def test_validate_input_rejects_duplicate_policy_id(fp_config) -> None:
    """Verifies that duplicate PolicyIDs are rejected with a count."""
    frame = _valid_frame()
    frame.loc[1, "PolicyID"] = 1
    with pytest.raises(FeaturePipelineError) as exc_info:
        validate_feature_input(frame, fp_config)
    assert exc_info.value.details["rule"] == "duplicate_policy_id"
    assert exc_info.value.details["count"] == 1


def test_validate_input_rejects_null(fp_config) -> None:
    """Verifies that a null in any required column is rejected."""
    frame = _valid_frame()
    frame.loc[0, "Gas"] = None
    with pytest.raises(FeaturePipelineError) as exc_info:
        validate_feature_input(frame, fp_config)
    assert exc_info.value.details["rule"] == "null_values"
    assert exc_info.value.details["column"] == "Gas"


def test_validate_input_rejects_non_positive_exposure(fp_config) -> None:
    """Verifies that Exposure <= 0 is rejected."""
    frame = _valid_frame()
    frame.loc[0, "Exposure"] = 0.0
    with pytest.raises(FeaturePipelineError) as exc_info:
        validate_feature_input(frame, fp_config)
    assert exc_info.value.details["rule"] == "invalid_exposure"
    assert exc_info.value.details["column"] == "Exposure"


def test_validate_input_rejects_non_positive_density(fp_config) -> None:
    """Verifies that Density <= 0 is rejected before the log transform."""
    frame = _valid_frame()
    frame.loc[0, "Density"] = 0
    with pytest.raises(FeaturePipelineError) as exc_info:
        validate_feature_input(frame, fp_config)
    assert exc_info.value.details["rule"] == "invalid_density"
    assert exc_info.value.details["column"] == "Density"


def test_validate_input_rejects_unknown_level(fp_config) -> None:
    """Verifies that an unknown categorical level fails loudly."""
    frame = _valid_frame()
    frame.loc[0, "Region"] = "Atlantis"
    with pytest.raises(FeaturePipelineError) as exc_info:
        validate_feature_input(frame, fp_config)
    assert exc_info.value.details["rule"] == "unknown_level"
    assert exc_info.value.details["column"] == "Region"


def test_validate_input_rejects_non_numeric_column(fp_config) -> None:
    """Verifies that a non-numeric numeric column is rejected."""
    frame = _valid_frame()
    frame["Exposure"] = frame["Exposure"].astype(str)
    with pytest.raises(FeaturePipelineError) as exc_info:
        validate_feature_input(frame, fp_config)
    assert exc_info.value.details["rule"] == "invalid_dtype"
    assert exc_info.value.details["column"] == "Exposure"


def test_exposure_cap_counts_and_caps(fp_config) -> None:
    """Verifies exposure clipping at 1.0 and exact counts."""
    frame = _valid_frame()
    frame["Exposure"] = [0.5, 1.0, 1.99]
    derived = derive_features(frame, fp_config)
    assert derived.exposure_clip_count == 1
    assert derived.frame["exposure_capped"].max() == 1.0
    assert derived.frame.loc[1, "exposure_capped"] == 1.0
    assert derived.frame.loc[2, "exposure_capped"] == 1.0


def test_log_density_matches_math_log(fp_config) -> None:
    """Verifies that log_density equals ln(Density)."""
    derived = derive_features(_valid_frame(), fp_config)
    expected = [log(value) for value in [120, 450, 2500]]
    assert derived.frame["log_density"].tolist() == pytest.approx(expected)


def test_band_boundaries_are_left_closed(fp_config) -> None:
    """Verifies DriverAge, CarAge, and Density band boundaries."""
    frame = pd.concat([_valid_frame()] * 3, ignore_index=True).iloc[:8].reset_index(drop=True)
    frame["DriverAge"] = [18, 24, 25, 44, 45, 64, 65, 90]
    frame["CarAge"] = [0, 4, 5, 9, 10, 34, 5, 8]
    frame["Density"] = [500, 501, 2000, 2001, 2, 27000, 500, 501]
    frame["PolicyID"] = list(range(1, 9))
    frame["ClaimNb"] = [0] * 8
    frame["ClaimAmount"] = [0.0] * 8
    frame["sev_claim_count"] = [0] * 8
    derived = derive_features(frame, fp_config)
    assert derived.frame["driver_age_band"].tolist() == [
        "18-24", "18-24", "25-44", "25-44", "45-64", "45-64", "65+", "65+"
    ]
    assert derived.frame["car_age_band"].tolist() == [
        "0-4", "0-4", "5-9", "5-9", "10+", "10+", "5-9", "5-9"
    ]
    assert derived.frame["density_band"].tolist() == [
        "<=500", "501-2000", "501-2000", ">2000", "<=500", ">2000", "<=500", "501-2000"
    ]


def test_tail_clip_counts_and_values(fp_config) -> None:
    """Verifies approved tail bounds and per-side counts."""
    frame = _valid_frame()
    frame["DriverAge"] = [17, 91, 24]
    frame["CarAge"] = [-1, 40, 5]
    derived = derive_features(frame, fp_config)
    assert derived.tail_clip_counts.driver_age_below == 1
    assert derived.tail_clip_counts.driver_age_above == 1
    assert derived.tail_clip_counts.car_age_below == 1
    assert derived.tail_clip_counts.car_age_above == 1
    assert derived.frame["driver_age_clipped"].tolist() == [18, 90, 24]
    assert derived.frame["car_age_clipped"].tolist() == [0, 34, 5]


def test_power_ordinal_mapping(fp_config) -> None:
    """Verifies the approved Power ordinal mapping d=0 � o=11."""
    frame = _valid_frame()
    frame["Power"] = ["d", "f", "o"]
    derived = derive_features(frame, fp_config)
    assert derived.frame["power_ord"].tolist() == [0, 2, 11]


def test_band_derivation_has_no_missing(fp_config) -> None:
    """Verifies that every derived band is assigned."""
    derived = derive_features(_valid_frame(), fp_config)
    for column in ["driver_age_band", "car_age_band", "density_band"]:
        assert int(derived.frame[column].isna().sum()) == 0


def test_glm_view_contract(fp_config) -> None:
    """Verifies the fixed GLM view columns and row count."""
    views = transform_features(_valid_frame(), fp_config)
    assert views.glm.columns.tolist() == GLM_COLUMNS
    assert len(views.glm) == 3
    assert "ClaimNb" not in views.glm.columns
    assert "ClaimAmount" in views.glm.columns


def test_forest_view_excludes_claims_and_identifiers(fp_config) -> None:
    """Verifies claims fields are absent from the forest view and its covariates."""
    views = transform_features(_valid_frame(), fp_config)
    claims = {"ClaimNb", "ClaimAmount", "sev_claim_count", "is_orphan_severity", "PurePremium"}
    assert claims.isdisjoint(set(views.forest.columns))
    assert "PolicyID" not in views.forest_covariates
    assert "exposure_capped" not in views.forest_covariates
    assert "ClaimAmount" not in views.forest_covariates


def test_forest_one_hot_schema_is_fixed(fp_config) -> None:
    """Verifies the fixed one-hot schema and an all-zero absent level."""
    views = transform_features(_valid_frame(), fp_config)
    assert len(views.one_hot_columns) == 19
    assert len(set(views.one_hot_columns)) == 19
    absent = "brand__japanese_except_nissan_or_korean"
    assert absent in views.forest.columns
    assert int(views.forest[absent].sum()) == 0
    assert "brand__renault_nissan_or_citroen" in views.forest.columns
    assert int(views.forest["brand__renault_nissan_or_citroen"].sum()) == 1


def test_write_artifacts_and_metadata(tmp_path, fp_config) -> None:
    """Verifies artifact hashes, metadata round-trip, and summary contents."""
    views = transform_features(_valid_frame(), fp_config)
    glm_path = tmp_path / "features_glm.csv"
    forest_path = tmp_path / "features_forest.csv"
    metadata_path = tmp_path / "feature_metadata.json"

    metadata = write_feature_artifacts(
        views,
        glm_path,
        forest_path,
        metadata_path,
        input_sha256=SHA,
        input_rows=3,
    )
    assert metadata.glm_view.sha256 == hashlib.sha256(glm_path.read_bytes()).hexdigest()
    assert metadata.forest_view.sha256 == hashlib.sha256(forest_path.read_bytes()).hexdigest()
    reloaded = FeatureMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    assert reloaded == metadata
    assert reloaded.one_hot_columns == views.one_hot_columns
    assert "glm.log_density" in reloaded.distribution_summary.numeric
    assert "glm.Gas" in reloaded.distribution_summary.categorical
    assert reloaded.tail_clip_counts.driver_age_above == 0


def test_writer_is_byte_deterministic(tmp_path, fp_config) -> None:
    """Verifies that repeated writes produce byte-identical outputs."""
    views = transform_features(_valid_frame(), fp_config)
    first = tmp_path / "first"
    second = tmp_path / "second"
    for target in [first, second]:
        write_feature_artifacts(
            views,
            target / "features_glm.csv",
            target / "features_forest.csv",
            target / "feature_metadata.json",
            input_sha256=SHA,
            input_rows=3,
        )
    for name in ["features_glm.csv", "features_forest.csv", "feature_metadata.json"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_run_feature_pipeline_end_to_end(tmp_path) -> None:
    """Verifies the public runner reads a CSV, writes artifacts, and reports counts."""
    raw_path = tmp_path / "raw_policy_data.csv"
    _valid_frame().to_csv(raw_path, index=False)
    metadata = run_feature_pipeline(
        input_path=raw_path,
        glm_path=tmp_path / "glm.csv",
        forest_path=tmp_path / "forest.csv",
        metadata_path=tmp_path / "metadata.json",
    )
    assert metadata.input_rows == 3
    assert metadata.output_rows == 3
    assert metadata.exposure_clip_count == 0
    assert (tmp_path / "glm.csv").is_file()
    assert (tmp_path / "forest.csv").is_file()
    assert (tmp_path / "metadata.json").is_file()


def test_run_feature_pipeline_missing_input(tmp_path) -> None:
    """Verifies that a missing input file raises a typed error."""
    with pytest.raises(FeaturePipelineError) as exc_info:
        run_feature_pipeline(input_path=tmp_path / "missing.csv")
    assert exc_info.value.details["rule"] == "input_not_found"


def test_no_composite_columns(fp_config) -> None:
    """Verifies that neither view contains a composite-index column."""
    views = transform_features(_valid_frame(), fp_config)
    for column in list(views.glm.columns) + list(views.forest.columns):
        assert "composite" not in column.lower()
        assert "index" not in column.lower()
    assert views.glm.columns.tolist() == GLM_COLUMNS


def test_empirical_determinism_and_counts(tmp_path) -> None:
    """Verifies empirical clip counts, deterministic hashes, and log equality."""
    source = Path("data/versioned/raw_policy_data.csv")
    if not source.is_file():
        pytest.skip("Versioned raw policy data not present; run dvc repro first.")

    first = run_feature_pipeline(
        input_path=source,
        glm_path=tmp_path / "run1" / "glm.csv",
        forest_path=tmp_path / "run1" / "forest.csv",
        metadata_path=tmp_path / "run1" / "metadata.json",
    )
    second = run_feature_pipeline(
        input_path=source,
        glm_path=tmp_path / "run2" / "glm.csv",
        forest_path=tmp_path / "run2" / "forest.csv",
        metadata_path=tmp_path / "run2" / "metadata.json",
    )

    assert first.input_rows == 413_169
    assert first.output_rows == 413_169
    assert first.exposure_clip_count == 421
    assert first.tail_clip_counts.driver_age_below == 0
    assert first.tail_clip_counts.driver_age_above == 307
    assert first.tail_clip_counts.car_age_below == 0
    assert first.tail_clip_counts.car_age_above == 410
    assert first.glm_view.sha256 == second.glm_view.sha256
    assert first.forest_view.sha256 == second.forest_view.sha256
    assert first.input_sha256 == second.input_sha256

    raw = pd.read_csv(source)
    glm = pd.read_csv(tmp_path / "run1" / "glm.csv")
    merged = raw[["PolicyID", "Density"]].merge(
        glm[["PolicyID", "log_density"]], on="PolicyID", how="inner"
    )
    expected = merged["Density"].map(log)
    assert float((merged["log_density"] - expected).abs().max()) < 1e-12
