"""Deterministic Tier 1 segment assignment and aggregation (P2-F05, ADR-014).

Maps every policy into the frozen 36-cell DriverAge x CarAge x Density grid, applies the
approved sub-floor cell merge, aggregates GLM fitted loss cost by exposure-weighted sums,
adds the configured premium loading, and writes the segment table, merge metadata, and
policy-to-segment mapping. Also provides the ratio-of-sums elasticity aggregation used by
the Stage 8 estimator. No pricing adjustment is produced or accepted (INV-4, INV-5).
"""

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from aegis.config.loader import get_config
from aegis.config.schema import FeaturePipelineConfig, SegmentsConfig
from aegis.schemas.segments import (
    CellMergeRecord,
    SegmentDefinition,
    SegmentMetadata,
)
from aegis.utils.exceptions import SegmentError

REQUIRED_FEATURE_COLUMNS: list[str] = [
    "PolicyID",
    "exposure_capped",
    "driver_age_band",
    "car_age_band",
    "density_band",
]

REQUIRED_FITTED_COLUMNS: list[str] = [
    "PolicyID",
    "exposure_capped",
    "fitted_loss_cost_per_year",
]

SEGMENT_OUTPUT_COLUMNS: list[str] = [
    "segment_id",
    "driver_age_band",
    "car_age_band",
    "density_band",
    "is_merged",
    "policy_count",
    "exposure_years",
    "loss_cost_per_year",
    "base_premium_per_year",
]


def _fail(
    message: str,
    rule: str,
    column: str | None = None,
    count: int = 0,
    sample_keys: list[Any] | None = None,
) -> None:
    """Raises a typed segment error with structured details."""
    details: dict[str, Any] = {"rule": rule, "count": count}
    if column is not None:
        details["column"] = column
    if sample_keys:
        details["sample_keys"] = sample_keys
    raise SegmentError(message, details=details)


def segment_id_for(driver_age_band: str, car_age_band: str, density_band: str) -> str:
    """Builds the canonical segment identifier for one band combination.

    Args:
        driver_age_band: Frozen DriverAge band label.
        car_age_band: Frozen CarAge band label.
        density_band: Density band label (possibly a merged label).

    Returns:
        The canonical `D:<driver>_C:<car>_Dens:<density>` identifier.
    """
    return f"D:{driver_age_band}_C:{car_age_band}_Dens:{density_band}"


def validate_segment_inputs(view: pd.DataFrame, fitted: pd.DataFrame) -> None:
    """Validates required columns, identifiers, row alignment, and positivity.

    Args:
        view: Stage 2 GLM feature view.
        fitted: Stage 3 fitted loss-cost table.

    Raises:
        SegmentError: If columns are missing, identifiers are duplicated or mismatched,
            row counts differ, or exposure/loss-cost values are invalid.
    """
    for name, frame, required in (
        ("features", view, REQUIRED_FEATURE_COLUMNS),
        ("fitted", fitted, REQUIRED_FITTED_COLUMNS),
    ):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            _fail(
                f"{name} input is missing required columns: {missing}",
                "missing_columns",
                column=",".join(missing),
                count=len(missing),
            )

    if view["PolicyID"].duplicated().any():
        _fail("features input has duplicate PolicyID values", "duplicate_policy_id", count=1)
    if fitted["PolicyID"].duplicated().any():
        _fail("fitted input has duplicate PolicyID values", "duplicate_policy_id", count=1)
    if len(view) != len(fitted):
        _fail(
            f"row count mismatch: features={len(view):,} fitted={len(fitted):,}",
            "row_count_mismatch",
            count=abs(len(view) - len(fitted)),
        )
    if set(view["PolicyID"].tolist()) != set(fitted["PolicyID"].tolist()):
        _fail(
            "PolicyID sets differ between features and fitted inputs",
            "policy_id_mismatch",
            count=1,
        )

    exposure = view["exposure_capped"].astype(float)
    if bool((exposure <= 0).any()):
        _fail(
            "exposure_capped must be strictly positive",
            "non_positive_exposure",
            column="exposure_capped",
            count=int((exposure <= 0).sum()),
        )
    fitted_lc = fitted["fitted_loss_cost_per_year"].astype(float)
    non_finite = int(
        (fitted_lc.isna() | fitted_lc.isin([float("inf"), float("-inf")])).sum()
    )
    if non_finite:
        _fail(
            "fitted_loss_cost_per_year must be finite",
            "non_finite_loss_cost",
            column="fitted_loss_cost_per_year",
            count=non_finite,
        )


def build_policy_segments(
    view: pd.DataFrame,
    segments_cfg: SegmentsConfig,
    feature_cfg: FeaturePipelineConfig,
) -> pd.DataFrame:
    """Assigns each policy to its realized segment, applying the approved merge rule.

    Args:
        view: Stage 2 GLM feature view.
        segments_cfg: Validated segment configuration (floor, loading, merges).
        feature_cfg: Frozen feature band labels used for validation.

    Returns:
        A per-policy frame with `PolicyID`, `exposure_capped`, realized bands,
        `is_merged`, and `segment_id`.

    Raises:
        SegmentError: If an unknown band label is present.
    """
    label_checks = (
        ("driver_age_band", feature_cfg.driver_age_band_labels),
        ("car_age_band", feature_cfg.car_age_band_labels),
        ("density_band", feature_cfg.density_band_labels),
    )
    for column, allowed in label_checks:
        unknown = sorted(set(view[column].astype(str)) - set(allowed))
        if unknown:
            _fail(
                f"unknown {column} labels: {unknown[:5]}",
                "unknown_band_label",
                column=column,
                count=len(unknown),
                sample_keys=unknown[:5],
            )

    driver = view["driver_age_band"].astype(str).copy()
    car = view["car_age_band"].astype(str).copy()
    density = view["density_band"].astype(str).copy()
    is_merged = pd.Series(False, index=view.index, dtype=bool)

    for merge in segments_cfg.cell_merges:
        mask = (
            (driver == merge.driver_age_band)
            & (car == merge.car_age_band)
            & density.isin([merge.from_density_band, merge.to_density_band])
        )
        if not bool(mask.any()):
            continue
        density = density.where(~mask, merge.merged_density_band)
        is_merged = is_merged | mask

    segment_id = "D:" + driver + "_C:" + car + "_Dens:" + density

    return pd.DataFrame(
        {
            "PolicyID": view["PolicyID"].to_numpy(),
            "exposure_capped": view["exposure_capped"].astype(float).to_numpy(),
            "driver_age_band": driver.to_numpy(),
            "car_age_band": car.to_numpy(),
            "density_band": density.to_numpy(),
            "is_merged": is_merged.to_numpy(),
            "segment_id": segment_id.to_numpy(),
        }
    )


def aggregate_segment_economics(
    policy_segments: pd.DataFrame,
    fitted: pd.DataFrame,
    segments_cfg: SegmentsConfig,
) -> list[SegmentDefinition]:
    """Aggregates exposure-weighted loss cost and loaded base premium per segment.

    Args:
        policy_segments: Output of `build_policy_segments`.
        fitted: Stage 3 fitted loss-cost table.
        segments_cfg: Validated segment configuration.

    Returns:
        Segment rows sorted by `segment_id`.

    Raises:
        SegmentError: If the join drops policies, exposure is non-positive, or a
            realized segment falls below the policy floor.
    """
    merged = policy_segments.merge(
        fitted[["PolicyID", "fitted_loss_cost_per_year"]],
        on="PolicyID",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(policy_segments):
        _fail(
            "policy/fitted join dropped rows",
            "join_row_mismatch",
            count=len(policy_segments) - len(merged),
        )

    merged["weighted_loss"] = (
        merged["fitted_loss_cost_per_year"].astype(float) * merged["exposure_capped"].astype(float)
    )

    definitions: list[SegmentDefinition] = []
    for segment_id, group in merged.groupby("segment_id", sort=True):
        policy_count = int(len(group))
        if policy_count < segments_cfg.min_policy_count:
            _fail(
                f"segment {segment_id} has {policy_count} policies below floor "
                f"{segments_cfg.min_policy_count}",
                "segment_below_floor",
                count=policy_count,
                sample_keys=[segment_id],
            )
        exposure = float(group["exposure_capped"].sum())
        if exposure <= 0:
            _fail(
                f"segment {segment_id} has non-positive exposure",
                "non_positive_segment_exposure",
                count=1,
                sample_keys=[segment_id],
            )
        loss_cost = float(group["weighted_loss"].sum()) / exposure
        base_premium = loss_cost * (1.0 + segments_cfg.premium_loading)
        definitions.append(
            SegmentDefinition(
                segment_id=str(segment_id),
                driver_age_band=str(group["driver_age_band"].iloc[0]),
                car_age_band=str(group["car_age_band"].iloc[0]),
                density_band=str(group["density_band"].iloc[0]),
                is_merged=bool(group["is_merged"].iloc[0]),
                policy_count=policy_count,
                exposure_years=round(exposure, 6),
                loss_cost_per_year=round(loss_cost, 6),
                base_premium_per_year=round(base_premium, 6),
            )
        )
    return definitions


def aggregate_ratio_of_sums(
    tau: Sequence[float] | pd.Series,
    q0: Sequence[float] | pd.Series,
) -> float:
    """Computes segment elasticity as the ratio of sums: sum(tau) / sum(q0).

    Aggregating the ratio of sums (rather than the mean of per-policy ratios) avoids
    division by near-zero retention probabilities and weights by baseline volume.

    Args:
        tau: Per-policy treatment effects on retention.
        q0: Per-policy baseline retention probabilities.

    Returns:
        The segment-level ratio-of-sums elasticity.

    Raises:
        SegmentError: If inputs are empty, mismatched, non-finite, or have a
            non-positive denominator.
    """
    tau_values = [float(value) for value in tau]
    q0_values = [float(value) for value in q0]
    if not tau_values or not q0_values:
        _fail("ratio-of-sums requires non-empty inputs", "empty_input")
    if len(tau_values) != len(q0_values):
        _fail(
            f"ratio-of-sums length mismatch: tau={len(tau_values)} q0={len(q0_values)}",
            "length_mismatch",
            count=abs(len(tau_values) - len(q0_values)),
        )
    if not all(math.isfinite(value) for value in tau_values) or not all(
        math.isfinite(value) for value in q0_values
    ):
        _fail("ratio-of-sums inputs must be finite", "non_finite_input")
    denominator = sum(q0_values)
    if denominator <= 0:
        _fail(
            "ratio-of-sums denominator must be positive",
            "non_positive_denominator",
        )
    return sum(tau_values) / denominator


def _config_fingerprint(segments_cfg: SegmentsConfig) -> str:
    """Hashes the canonical segment configuration."""
    payload = segments_cfg.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_metadata(
    definitions: list[SegmentDefinition],
    policy_segments: pd.DataFrame,
    merged: pd.DataFrame,
    segments_cfg: SegmentsConfig,
    feature_cfg: FeaturePipelineConfig,
) -> SegmentMetadata:
    """Assembles and validates the segment metadata artifact."""
    initial_cell_count = (
        len(feature_cfg.driver_age_band_labels)
        * len(feature_cfg.car_age_band_labels)
        * len(feature_cfg.density_band_labels)
    )
    by_id = {definition.segment_id: definition for definition in definitions}
    merged_cells: list[CellMergeRecord] = []
    for merge in segments_cfg.cell_merges:
        realized_id = segment_id_for(
            merge.driver_age_band, merge.car_age_band, merge.merged_density_band
        )
        realized = by_id.get(realized_id)
        if realized is None:
            continue
        merged_cells.append(
            CellMergeRecord(
                driver_age_band=merge.driver_age_band,
                car_age_band=merge.car_age_band,
                from_density_band=merge.from_density_band,
                to_density_band=merge.to_density_band,
                merged_density_band=merge.merged_density_band,
                merged_policy_count=realized.policy_count,
            )
        )

    total_exposure = float(merged["exposure_capped"].sum())
    portfolio_loss = float(merged["weighted_loss"].sum()) / total_exposure
    portfolio_premium = portfolio_loss * (1.0 + segments_cfg.premium_loading)

    return SegmentMetadata(
        segment_count=len(definitions),
        initial_cell_count=initial_cell_count,
        min_policy_count=segments_cfg.min_policy_count,
        premium_loading=segments_cfg.premium_loading,
        merged_cells=merged_cells,
        total_policies=int(len(policy_segments)),
        total_exposure_years=round(total_exposure, 6),
        portfolio_loss_cost_per_year=round(portfolio_loss, 6),
        portfolio_base_premium_per_year=round(portfolio_premium, 6),
        config_fingerprint=_config_fingerprint(segments_cfg),
    )


def _write_json(model: Any, path: Path) -> None:
    """Writes a validated Pydantic model as deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def run_segment_pipeline(
    features_path: Path | None = None,
    fitted_path: Path | None = None,
    loss_cost_path: Path | None = None,
    definitions_path: Path | None = None,
    policy_segments_path: Path | None = None,
) -> SegmentMetadata:
    """Assigns segments, aggregates economics, and writes the three Tier 1 artifacts.

    Args:
        features_path: Stage 2 GLM view. Defaults to params.yaml feature_paths.glm_view.
        fitted_path: Stage 3 fitted loss cost. Defaults to params.yaml.
        loss_cost_path: Segment table CSV. Defaults to params.yaml.
        definitions_path: Segment metadata JSON. Defaults to params.yaml.
        policy_segments_path: Policy-to-segment mapping CSV. Defaults to params.yaml.

    Returns:
        The validated segment metadata that was written.

    Raises:
        SegmentError: If inputs are invalid, a segment violates the floor, or writing fails.
    """
    config = get_config()
    segments_cfg = config.tier1_ml.segments
    feature_cfg = config.tier1_ml.feature_pipeline
    paths = config.tier1_ml.segment_paths

    features_source = (
        Path(features_path)
        if features_path is not None
        else Path(config.tier1_ml.feature_paths.glm_view)
    )
    fitted_source = (
        Path(fitted_path)
        if fitted_path is not None
        else Path(config.tier1_ml.glm_paths.fitted_loss_cost)
    )
    for label, source in (("features", features_source), ("fitted", fitted_source)):
        if not source.is_file():
            _fail(f"{label} input not found: {source}", "input_not_found", column=label)

    view = pd.read_csv(features_source)
    fitted = pd.read_csv(fitted_source)
    validate_segment_inputs(view, fitted)

    policy_segments = build_policy_segments(view, segments_cfg, feature_cfg)
    definitions = aggregate_segment_economics(policy_segments, fitted, segments_cfg)

    merged = policy_segments.merge(
        fitted[["PolicyID", "fitted_loss_cost_per_year"]],
        on="PolicyID",
        how="inner",
        validate="one_to_one",
    )
    merged["weighted_loss"] = (
        merged["fitted_loss_cost_per_year"].astype(float) * merged["exposure_capped"].astype(float)
    )
    metadata = _build_metadata(
        definitions, policy_segments, merged, segments_cfg, feature_cfg
    )

    table = pd.DataFrame([definition.model_dump() for definition in definitions])[
        SEGMENT_OUTPUT_COLUMNS
    ]
    loss_target = Path(loss_cost_path or paths.segment_loss_cost)
    loss_target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(loss_target, index=False, lineterminator="\n")

    _write_json(metadata, Path(definitions_path or paths.segment_definitions))

    mapping = pd.DataFrame(
        {
            "PolicyID": policy_segments["PolicyID"].to_numpy(),
            "segment_id": policy_segments["segment_id"].to_numpy(),
        }
    ).sort_values(by="PolicyID").reset_index(drop=True)
    mapping_target = Path(policy_segments_path or paths.policy_segments)
    mapping_target.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_target, index=False, lineterminator="\n")

    print(
        f"[SEGMENTS] segments={metadata.segment_count} "
        f"policies={metadata.total_policies:,} "
        f"loss_cost={metadata.portfolio_loss_cost_per_year:,.2f}/yr "
        f"base_premium={metadata.portfolio_base_premium_per_year:,.2f}/yr"
    )
    return metadata
