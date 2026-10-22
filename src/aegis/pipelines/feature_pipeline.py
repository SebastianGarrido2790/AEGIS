"""Deterministic dual-view feature pipeline (P2-F02, ADR-012).

Transforms the Stage 1 raw policy table into a categorical/banded GLM view and a
numeric/ordinal causal-forest view, with explicit clipping, fixed encodings, and
persisted metadata. The pipeline never fits a model or learns from the data.
"""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any, cast

import pandas as pd

from aegis.config.loader import get_config
from aegis.config.schema import FeaturePipelineConfig
from aegis.schemas.features import (
    BandSpec,
    CategoricalSummary,
    DistributionSummary,
    FeatureMetadata,
    NumericSummary,
    TailClipCounts,
    ViewContract,
)
from aegis.utils.exceptions import FeaturePipelineError

RAW_REQUIRED_COLUMNS: list[str] = [
    "PolicyID",
    "ClaimNb",
    "Exposure",
    "Power",
    "CarAge",
    "DriverAge",
    "Brand",
    "Gas",
    "Region",
    "Density",
    "ClaimAmount",
    "sev_claim_count",
    "is_orphan_severity",
]

NUMERIC_COLUMNS: list[str] = [
    "PolicyID",
    "ClaimNb",
    "Exposure",
    "CarAge",
    "DriverAge",
    "Density",
    "ClaimAmount",
    "sev_claim_count",
]

CATEGORICAL_COLUMNS: list[str] = ["Power", "Brand", "Gas", "Region"]


def _fail(
    message: str,
    rule: str,
    column: str | None = None,
    count: int = 0,
    sample_keys: list[Any] | None = None,
) -> None:
    """Raises a typed feature-pipeline error with structured details."""
    details: dict[str, Any] = {"rule": rule, "count": count}
    if column is not None:
        details["column"] = column
    if sample_keys:
        details["sample_keys"] = sample_keys[:10]
    raise FeaturePipelineError(message, details=details)


def validate_feature_input(raw: pd.DataFrame, config: FeaturePipelineConfig) -> None:
    """Validates the raw table before any transformation.

    Args:
        raw: Input policy table produced by the Stage 1 raw contract.
        config: Approved feature-pipeline configuration.

    Raises:
        FeaturePipelineError: If the input violates columns, dtypes, uniqueness,
            nullability, numeric positivity, or the fixed category levels.
    """
    if raw.empty:
        _fail("Feature input is empty; no artifacts will be produced.", "empty_input")

    actual = list(raw.columns)
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in actual]
    if missing:
        _fail(
            f"Feature input is missing required columns: {missing}",
            "missing_columns",
            count=len(missing),
            sample_keys=missing,
        )

    unexpected = [col for col in actual if col not in RAW_REQUIRED_COLUMNS]
    if unexpected:
        _fail(
            f"Feature input contains unexpected columns: {unexpected}",
            "unexpected_columns",
            count=len(unexpected),
            sample_keys=unexpected,
        )

    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(raw[column]):
            _fail(
                f"Column '{column}' must be numeric.",
                "invalid_dtype",
                column=column,
            )

    duplicate_count = int(raw["PolicyID"].duplicated().sum())
    if duplicate_count:
        duplicates = raw.loc[raw["PolicyID"].duplicated(), "PolicyID"].tolist()
        _fail(
            f"PolicyID contains {duplicate_count} duplicate value(s).",
            "duplicate_policy_id",
            column="PolicyID",
            count=duplicate_count,
            sample_keys=duplicates,
        )

    for column in RAW_REQUIRED_COLUMNS:
        null_count = int(raw[column].isna().sum())
        if null_count:
            keys = raw.loc[raw[column].isna(), "PolicyID"].tolist()
            _fail(
                f"Column '{column}' contains {null_count} null value(s).",
                "null_values",
                column=column,
                count=null_count,
                sample_keys=keys,
            )

    invalid_exposure = raw["Exposure"] <= 0
    if bool(invalid_exposure.any()):
        keys = raw.loc[invalid_exposure, "PolicyID"].tolist()
        _fail(
            "Exposure must be strictly positive.",
            "invalid_exposure",
            column="Exposure",
            count=int(invalid_exposure.sum()),
            sample_keys=keys,
        )

    invalid_density = raw["Density"] <= 0
    if bool(invalid_density.any()):
        keys = raw.loc[invalid_density, "PolicyID"].tolist()
        _fail(
            "Density must be strictly positive for the log transform.",
            "invalid_density",
            column="Density",
            count=int(invalid_density.sum()),
            sample_keys=keys,
        )

    level_specs = [
        ("Power", config.power_levels),
        ("Brand", config.brand_levels),
        ("Gas", config.gas_levels),
        ("Region", config.region_levels),
    ]
    for column, allowed in level_specs:
        unknown_mask = ~raw[column].isin(allowed)
        if bool(unknown_mask.any()):
            keys = raw.loc[unknown_mask, "PolicyID"].tolist()
            values = sorted({str(value) for value in raw.loc[unknown_mask, column]})
            _fail(
                f"Column '{column}' contains unknown level(s): {values}",
                "unknown_level",
                column=column,
                count=int(unknown_mask.sum()),
                sample_keys=keys,
            )


@dataclass(frozen=True)
class DerivedFeatures:
    """Intermediate derivations shared by the two views."""

    frame: pd.DataFrame
    exposure_clip_count: int
    tail_clip_counts: TailClipCounts


@dataclass(frozen=True)
class FeatureViews:
    """Model-ready views and their fixed schema metadata."""

    glm: pd.DataFrame
    forest: pd.DataFrame
    one_hot_columns: list[str]
    forest_covariates: list[str]
    exposure_clip_count: int
    tail_clip_counts: TailClipCounts
    config: FeaturePipelineConfig


GLM_COLUMNS: list[str] = [
    "PolicyID",
    "exposure_capped",
    "ClaimAmount",
    "DriverAge",
    "CarAge",
    "Density",
    "log_density",
    "driver_age_band",
    "car_age_band",
    "density_band",
    "Power",
    "Brand",
    "Gas",
    "Region",
]

FOREST_ID_COLUMNS: list[str] = ["PolicyID", "exposure_capped"]


def _slug(value: str) -> str:
    """Converts a factor level to a stable, lowercase column suffix."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        _fail(f"Factor level '{value}' cannot be slugified.", "invalid_level_slug")
    return slug


def _one_hot_columns(prefix: str, levels: list[str]) -> list[str]:
    """Builds deterministic one-hot column names and rejects collisions."""
    columns = [f"{prefix}__{_slug(level)}" for level in levels]
    if len(set(columns)) != len(columns):
        _fail(
            f"One-hot column slug collision for factor '{prefix}'.",
            "slug_collision",
            column=prefix,
            count=len(columns) - len(set(columns)),
        )
    return columns


def build_glm_view(derived: DerivedFeatures) -> pd.DataFrame:
    """Builds the categorical/banded GLM view with the fixed column order."""
    frame = derived.frame
    view = pd.DataFrame(
        {
            "PolicyID": frame["PolicyID"],
            "exposure_capped": frame["exposure_capped"],
            "ClaimAmount": frame["ClaimAmount"],
            "DriverAge": frame["driver_age_raw"],
            "CarAge": frame["car_age_raw"],
            "Density": frame["density_raw"],
            "log_density": frame["log_density"],
            "driver_age_band": frame["driver_age_band"],
            "car_age_band": frame["car_age_band"],
            "density_band": frame["density_band"],
            "Power": frame["Power"],
            "Brand": frame["Brand"],
            "Gas": frame["Gas"],
            "Region": frame["Region"],
        }
    )
    return cast(pd.DataFrame, view[GLM_COLUMNS])


def build_forest_view(
    derived: DerivedFeatures, config: FeaturePipelineConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Builds the numeric/ordinal causal-forest view and its one-hot columns."""
    frame = derived.frame
    view = pd.DataFrame(
        {
            "PolicyID": frame["PolicyID"],
            "exposure_capped": frame["exposure_capped"],
            "driver_age_clipped": frame["driver_age_clipped"],
            "car_age_clipped": frame["car_age_clipped"],
            "log_density": frame["log_density"],
            "driver_age_band_ord": frame["driver_age_band_ord"],
            "car_age_band_ord": frame["car_age_band_ord"],
            "power_ord": frame["power_ord"],
        }
    )

    one_hot_columns: list[str] = []
    for factor, levels in [
        ("brand", config.brand_levels),
        ("gas", config.gas_levels),
        ("region", config.region_levels),
    ]:
        columns = _one_hot_columns(factor, levels)
        values = cast(pd.Series, frame[factor.capitalize()])
        for level, column in zip(levels, columns, strict=True):
            view[column] = (values == level).astype(int)
        one_hot_columns.extend(columns)

    return view, one_hot_columns


def transform_features(
    raw: pd.DataFrame,
    config: FeaturePipelineConfig,
) -> FeatureViews:
    """Validates and transforms the raw table into both model-ready views.

    Args:
        raw: Raw policy table from the Stage 1 contract.
        config: Approved feature-pipeline configuration.

    Returns:
        FeatureViews with GLM and forest views and their schema metadata.

    Raises:
        FeaturePipelineError: If validation or derivation fails.
    """
    derived = derive_features(raw, config)
    glm = build_glm_view(derived).sort_values("PolicyID").reset_index(drop=True)
    forest, one_hot_columns = build_forest_view(derived, config)
    forest = forest.sort_values("PolicyID").reset_index(drop=True)
    forest_covariates = [col for col in forest.columns if col not in FOREST_ID_COLUMNS]
    return FeatureViews(
        glm=glm,
        forest=forest,
        one_hot_columns=one_hot_columns,
        forest_covariates=forest_covariates,
        exposure_clip_count=derived.exposure_clip_count,
        tail_clip_counts=derived.tail_clip_counts,
        config=config,
    )


def _band_series(
    values: pd.Series,
    edges: Sequence[float],
    labels: Sequence[str],
    name: str,
) -> pd.Series:
    """Assigns left-closed, right-open bands and fails on unbanded values."""
    cut = pd.cut(values, bins=list(edges), labels=list(labels), right=False)
    bands = pd.Series(cut, index=values.index)
    if bool(bands.isna().any()):
        unbanded = values.loc[bands.isna()].head(10).tolist()
        _fail(
            f"Values fall outside the configured {name} band edges.",
            "unbanded_value",
            column=name,
            count=int(bands.isna().sum()),
            sample_keys=unbanded,
        )
    return pd.Series(bands.astype(str), index=values.index)


def _ordinal(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    """Maps fixed levels to ordinal codes, failing on unknown levels."""
    return series.map(lambda value: mapping[str(value)]).astype(int)


def derive_features(raw: pd.DataFrame, config: FeaturePipelineConfig) -> DerivedFeatures:
    """Applies approved clipping, log and band derivations to the raw table.

    Args:
        raw: Validated raw policy table.
        config: Approved feature-pipeline configuration.

    Returns:
        DerivedFeatures with the intermediate frame and exact clip counts.

    Raises:
        FeaturePipelineError: If a value cannot be banded or the log is undefined.
    """
    validate_feature_input(raw, config)

    exposure_clip_count = int((raw["Exposure"] > config.exposure_cap).sum())
    driver_age_below = int((raw["DriverAge"] < config.driver_age_bounds[0]).sum())
    driver_age_above = int((raw["DriverAge"] > config.driver_age_bounds[1]).sum())
    car_age_below = int((raw["CarAge"] < config.car_age_bounds[0]).sum())
    car_age_above = int((raw["CarAge"] > config.car_age_bounds[1]).sum())

    frame = pd.DataFrame(
        {
            "PolicyID": raw["PolicyID"].astype(int),
            "ClaimAmount": raw["ClaimAmount"].astype(float),
            "exposure_capped": raw["Exposure"].clip(
                lower=None, upper=config.exposure_cap
            ).astype(float),
            "driver_age_raw": raw["DriverAge"].astype(int),
            "car_age_raw": raw["CarAge"].astype(int),
            "density_raw": raw["Density"].astype(float),
            "driver_age_clipped": raw["DriverAge"].clip(
                lower=config.driver_age_bounds[0], upper=config.driver_age_bounds[1]
            ).astype(int),
            "car_age_clipped": raw["CarAge"].clip(
                lower=config.car_age_bounds[0], upper=config.car_age_bounds[1]
            ).astype(int),
            "Power": raw["Power"].astype(str),
            "Brand": raw["Brand"].astype(str),
            "Gas": raw["Gas"].astype(str),
            "Region": raw["Region"].astype(str),
        }
    )
    frame["log_density"] = cast(pd.Series, frame["density_raw"]).map(log)

    frame["driver_age_band"] = _band_series(
        cast(pd.Series, frame["driver_age_clipped"]),
        config.driver_age_band_edges,
        config.driver_age_band_labels,
        "driver_age",
    )
    frame["car_age_band"] = _band_series(
        cast(pd.Series, frame["car_age_clipped"]),
        config.car_age_band_edges,
        config.car_age_band_labels,
        "car_age",
    )
    frame["density_band"] = _band_series(
        cast(pd.Series, frame["density_raw"]),
        config.density_band_edges,
        config.density_band_labels,
        "density",
    )

    driver_band_order = {label: index for index, label in enumerate(config.driver_age_band_labels)}
    car_band_order = {label: index for index, label in enumerate(config.car_age_band_labels)}
    power_order = {level: index for index, level in enumerate(config.power_levels)}
    frame["driver_age_band_ord"] = _ordinal(
        cast(pd.Series, frame["driver_age_band"]), driver_band_order
    )
    frame["car_age_band_ord"] = _ordinal(
        cast(pd.Series, frame["car_age_band"]), car_band_order
    )
    frame["power_ord"] = _ordinal(cast(pd.Series, frame["Power"]), power_order)

    tail_clip_counts = TailClipCounts(
        driver_age_below=driver_age_below,
        driver_age_above=driver_age_above,
        car_age_below=car_age_below,
        car_age_above=car_age_above,
    )
    return DerivedFeatures(
        frame=frame,
        exposure_clip_count=exposure_clip_count,
        tail_clip_counts=tail_clip_counts,
    )


GLM_NUMERIC_SUMMARY: list[str] = [
    "exposure_capped",
    "ClaimAmount",
    "DriverAge",
    "CarAge",
    "Density",
    "log_density",
]

GLM_CATEGORICAL_SUMMARY: list[str] = [
    "driver_age_band",
    "car_age_band",
    "density_band",
    "Power",
    "Brand",
    "Gas",
    "Region",
]

FOREST_NUMERIC_SUMMARY: list[str] = [
    "exposure_capped",
    "driver_age_clipped",
    "car_age_clipped",
    "log_density",
    "driver_age_band_ord",
    "car_age_band_ord",
    "power_ord",
]


def _sha256_bytes(data: bytes) -> str:
    """Returns the lowercase SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def _numeric_summary(series: pd.Series) -> NumericSummary:
    """Builds a rounded descriptive summary for a numeric series."""
    values = series.astype(float)
    quantiles = values.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
    std = float(values.std()) if len(values) > 1 else 0.0
    return NumericSummary(
        count=int(values.count()),
        min=round(float(values.min()), 6),
        max=round(float(values.max()), 6),
        mean=round(float(values.mean()), 6),
        std=round(std, 6),
        p01=round(float(quantiles.loc[0.01]), 6),
        p25=round(float(quantiles.loc[0.25]), 6),
        p50=round(float(quantiles.loc[0.50]), 6),
        p75=round(float(quantiles.loc[0.75]), 6),
        p99=round(float(quantiles.loc[0.99]), 6),
    )


def _categorical_summary(series: pd.Series) -> CategoricalSummary:
    """Builds a deterministic value-count summary for a categorical series."""
    counts = {str(key): int(value) for key, value in series.value_counts().items()}
    return CategoricalSummary(
        counts=dict(sorted(counts.items())),
        missing=int(series.isna().sum()),
    )


def build_distribution_summary(views: FeatureViews) -> DistributionSummary:
    """Builds the persisted baseline distribution summary for both views."""
    numeric: dict[str, NumericSummary] = {}
    categorical: dict[str, CategoricalSummary] = {}
    for column in GLM_NUMERIC_SUMMARY:
        numeric[f"glm.{column}"] = _numeric_summary(cast(pd.Series, views.glm[column]))
    for column in GLM_CATEGORICAL_SUMMARY:
        categorical[f"glm.{column}"] = _categorical_summary(
            cast(pd.Series, views.glm[column])
        )
    for column in FOREST_NUMERIC_SUMMARY:
        numeric[f"forest.{column}"] = _numeric_summary(
            cast(pd.Series, views.forest[column])
        )
    return DistributionSummary(numeric=numeric, categorical=categorical)


def _config_fingerprint(config: FeaturePipelineConfig) -> str:
    """Hashes the canonical feature-pipeline configuration."""
    payload = config.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(canonical.encode("utf-8"))


def _write_view(frame: pd.DataFrame, path: Path) -> ViewContract:
    """Writes one view deterministically and returns its contract and hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    data = path.read_bytes()
    return ViewContract(
        columns=list(frame.columns),
        row_count=int(len(frame)),
        sha256=_sha256_bytes(data),
    )


def write_feature_artifacts(
    views: FeatureViews,
    glm_path: Path | str,
    forest_path: Path | str,
    metadata_path: Path | str,
    *,
    input_sha256: str,
    input_rows: int,
) -> FeatureMetadata:
    """Writes both views and the validated metadata artifact.

    Args:
        views: Transformed GLM and forest views.
        glm_path: Destination for the GLM CSV.
        forest_path: Destination for the forest CSV.
        metadata_path: Destination for the metadata JSON.
        input_sha256: SHA-256 of the input file bytes.
        input_rows: Input row count.

    Returns:
        The validated FeatureMetadata that was written.

    Raises:
        FeaturePipelineError: If a view cannot be written or metadata is invalid.
    """
    glm_contract = _write_view(views.glm, Path(glm_path))
    forest_contract = _write_view(views.forest, Path(forest_path))
    config = views.config

    metadata = FeatureMetadata(
        input_rows=input_rows,
        output_rows=int(len(views.glm)),
        input_sha256=input_sha256,
        config_fingerprint=_config_fingerprint(config),
        exposure_clip_count=views.exposure_clip_count,
        tail_clip_counts=views.tail_clip_counts,
        band_edges={
            "driver_age": BandSpec(
                edges=[float(edge) for edge in config.driver_age_band_edges],
                labels=config.driver_age_band_labels,
            ),
            "car_age": BandSpec(
                edges=[float(edge) for edge in config.car_age_band_edges],
                labels=config.car_age_band_labels,
            ),
            "density": BandSpec(
                edges=[float(edge) for edge in config.density_band_edges],
                labels=config.density_band_labels,
            ),
        },
        level_mappings={
            "power": config.power_levels,
            "brand": config.brand_levels,
            "gas": config.gas_levels,
            "region": config.region_levels,
        },
        one_hot_columns=views.one_hot_columns,
        forest_covariates=views.forest_covariates,
        glm_view=glm_contract,
        forest_view=forest_contract,
        distribution_summary=build_distribution_summary(views),
    )

    target = Path(metadata_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return metadata


def run_feature_pipeline(
    input_path: Path | str | None = None,
    glm_path: Path | str | None = None,
    forest_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
) -> FeatureMetadata:
    """Runs the full deterministic feature pipeline.

    Args:
        input_path: Raw policy CSV. Defaults to data/versioned/raw_policy_data.csv.
        glm_path: GLM view output. Defaults to params.yaml.
        forest_path: Forest view output. Defaults to params.yaml.
        metadata_path: Metadata output. Defaults to params.yaml.

    Returns:
        The validated metadata for the written artifacts.

    Raises:
        FeaturePipelineError: If the input is missing/unreadable or transform fails.
    """
    config = get_config()
    fp_config = config.tier1_ml.feature_pipeline
    paths = config.tier1_ml.feature_paths

    source = Path(input_path or "data/versioned/raw_policy_data.csv")
    if not source.is_file():
        _fail(f"Raw policy input not found: {source}", "input_not_found", column="input")
    try:
        input_bytes = source.read_bytes()
        raw = pd.read_csv(source)
    except Exception as exc:
        raise FeaturePipelineError(
            f"Failed to read raw policy input: {exc}",
            details={"rule": "input_read_error", "column": "input", "count": 1},
        ) from exc

    views = transform_features(raw, fp_config)
    metadata = write_feature_artifacts(
        views,
        glm_path or paths.glm_view,
        forest_path or paths.forest_view,
        metadata_path or paths.metadata,
        input_sha256=_sha256_bytes(input_bytes),
        input_rows=int(len(raw)),
    )
    print(
        f"[FEATURES] {metadata.input_rows:,} rows -> "
        f"{metadata.glm_view.row_count:,} GLM / {metadata.forest_view.row_count:,} forest; "
        f"exposure clips={metadata.exposure_clip_count}; "
        f"tail clips={metadata.tail_clip_counts.model_dump()}"
    )
    return metadata
