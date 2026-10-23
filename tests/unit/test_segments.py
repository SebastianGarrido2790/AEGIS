"""Unit and integration tests for Tier 1 segments and the contract (P2-F05)."""

import random
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from aegis.config.loader import get_config
from aegis.config.schema import Tier1MLConfig
from aegis.pipelines.segments import (
    aggregate_ratio_of_sums,
    aggregate_segment_economics,
    build_policy_segments,
    run_segment_pipeline,
    segment_id_for,
)
from aegis.schemas.segments import Tier1SegmentOutput, Tier1SegmentTable
from aegis.utils.exceptions import SegmentError


def _valid_payload(**overrides: object) -> dict[str, object]:
    """Builds a minimal valid Tier 1 segment payload."""
    payload: dict[str, object] = {
        "segment_id": "D:18-24_C:0-4_Dens:<=500",
        "policy_count": 2000,
        "exposure_years": 1500.0,
        "baseline_retention_q0": 0.80,
        "tau_hat": 0.02,
        "epsilon_hat": 0.025,
        "epsilon_ci_lower": 0.015,
        "epsilon_ci_upper": 0.035,
        "loss_cost_per_year": 1200.0,
        "base_premium_per_year": 1440.0,
        "model_version": "candidate-1",
        "data_version": "sha256:abc",
        "semi_synthetic": True,
    }
    payload.update(overrides)
    return payload


def test_tier1_output_accepts_valid_payload() -> None:
    """Verifies a complete 13-field segment payload validates and round-trips."""
    output = Tier1SegmentOutput.model_validate(_valid_payload())
    assert output.segment_id == "D:18-24_C:0-4_Dens:<=500"
    assert output.policy_count == 2000
    assert output.semi_synthetic is True
    restored = Tier1SegmentOutput.model_validate(output.model_dump())
    assert restored == output


def test_tier1_output_rejects_proposed_adjustment() -> None:
    """Falsification: a proposed adjustment payload must be rejected by the contract."""
    payload = _valid_payload()
    payload["proposed_adjustment"] = 0.05

    with pytest.raises(ValidationError) as exc_info:
        Tier1SegmentOutput.model_validate(payload)

    assert "proposed_adjustment" in str(exc_info.value)
    assert "forbidden" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "field",
    ["delta", "proposed_premium", "new_premium", "price_adjustment", "rate_adjustment"],
)
def test_tier1_output_rejects_adjustment_variants(field: str) -> None:
    """Verifies any adjustment-like field name is rejected, not just the exact example."""
    payload = _valid_payload()
    payload[field] = 0.05

    with pytest.raises(ValidationError):
        Tier1SegmentOutput.model_validate(payload)


def test_tier1_output_rejects_inverted_interval() -> None:
    """Verifies lower > upper is rejected."""
    payload = _valid_payload(epsilon_ci_lower=0.05, epsilon_ci_upper=0.01)
    with pytest.raises(ValidationError):
        Tier1SegmentOutput.model_validate(payload)


def test_tier1_output_rejects_estimate_outside_interval() -> None:
    """Verifies an elasticity point estimate outside its interval is rejected."""
    payload = _valid_payload(epsilon_hat=0.10)
    with pytest.raises(ValidationError):
        Tier1SegmentOutput.model_validate(payload)


def test_tier1_output_rejects_non_finite_value() -> None:
    """Verifies a non-finite metric is rejected."""
    payload = _valid_payload(tau_hat=float("inf"))
    with pytest.raises(ValidationError):
        Tier1SegmentOutput.model_validate(payload)


def test_tier1_output_rejects_below_floor() -> None:
    """Verifies the 1,000-policy floor is enforced on the contract."""
    payload = _valid_payload(policy_count=999)
    with pytest.raises(ValidationError):
        Tier1SegmentOutput.model_validate(payload)


def test_tier1_output_requires_semi_synthetic_true() -> None:
    """Verifies non-synthetic provenance is rejected."""
    payload = _valid_payload(semi_synthetic=False)
    with pytest.raises(ValidationError):
        Tier1SegmentOutput.model_validate(payload)


def test_tier1_output_rejects_unknown_field() -> None:
    """Verifies extra fields are forbidden."""
    payload = _valid_payload(unexpected="x")
    with pytest.raises(ValidationError):
        Tier1SegmentOutput.model_validate(payload)


def test_tier1_output_rejects_invalid_segment_id() -> None:
    """Verifies the canonical segment id pattern is enforced."""
    payload = _valid_payload(segment_id="seg-1")
    with pytest.raises(ValidationError):
        Tier1SegmentOutput.model_validate(payload)


def test_tier1_table_rejects_duplicate_segment_ids() -> None:
    """Verifies duplicate segment ids are rejected at table level."""
    segment = Tier1SegmentOutput.model_validate(_valid_payload())
    with pytest.raises(ValidationError):
        Tier1SegmentTable(segments=[segment, segment])


def test_feature_labels_available_to_segment_tests() -> None:
    """Verifies the frozen band labels used by merge tests load from config."""
    fp = get_config().tier1_ml.feature_pipeline
    assert len(fp.driver_age_band_labels) * len(fp.car_age_band_labels) * len(
        fp.density_band_labels
    ) == 36


def _build_view(rows_by_cell: dict[tuple[str, str, str], int], seed: int = 0) -> pd.DataFrame:
    """Builds a synthetic GLM view with a requested number of policies per cell."""
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    policy_id = 1
    for (driver, car, density), count in rows_by_cell.items():
        for _ in range(count):
            records.append(
                {
                    "PolicyID": policy_id,
                    "exposure_capped": round(rng.uniform(0.2, 1.0), 6),
                    "driver_age_band": driver,
                    "car_age_band": car,
                    "density_band": density,
                }
            )
            policy_id += 1
    return pd.DataFrame.from_records(records)


def _segment_config() -> Tier1MLConfig:
    """Returns the loaded tier1_ml config for segment tests."""
    return get_config().tier1_ml


def test_segment_id_for_is_canonical() -> None:
    """Verifies the canonical identifier format."""
    assert segment_id_for("18-24", "5-9", ">500") == "D:18-24_C:5-9_Dens:>500"


def test_merge_rule_pools_thin_cell_into_adjacent_density() -> None:
    """Verifies the approved OD-1 merge re-labels both density bands into one segment."""
    config = _segment_config()
    view = _build_view(
        {
            ("18-24", "5-9", ">2000"): 5,
            ("18-24", "5-9", "501-2000"): 7,
            ("18-24", "0-4", "<=500"): 3,
        }
    )
    assigned = build_policy_segments(view, config.segments, config.feature_pipeline)

    merged = assigned[assigned["is_merged"]]
    assert len(merged) == 12
    assert set(merged["segment_id"]) == {"D:18-24_C:5-9_Dens:>500"}
    assert set(merged["density_band"]) == {">500"}
    untouched = assigned[~assigned["is_merged"]]
    assert set(untouched["segment_id"]) == {"D:18-24_C:0-4_Dens:<=500"}


def test_unknown_band_label_raises() -> None:
    """Verifies an unknown band label raises a typed error."""
    config = _segment_config()
    view = _build_view({("18-24", "5-9", ">2000"): 3})
    view.loc[view.index[0], "density_band"] = "not-a-band"
    with pytest.raises(SegmentError) as exc_info:
        build_policy_segments(view, config.segments, config.feature_pipeline)
    assert "unknown_band_label" in str(exc_info.value)


def _aggregation_fixture() -> tuple[pd.DataFrame, pd.DataFrame, Tier1MLConfig]:
    """Builds a two-segment view/fitted pair with 1,050 policies each."""
    config = _segment_config()
    view = _build_view(
        {
            ("25-44", "0-4", "<=500"): 1050,
            ("45-64", "10+", ">2000"): 1050,
        }
    )
    fitted = pd.DataFrame(
        {
            "PolicyID": view["PolicyID"].to_numpy(),
            "fitted_loss_cost_per_year": [
                100.0 if density == "<=500" else 200.0
                for density in view["density_band"].tolist()
            ],
        }
    )
    return view, fitted, config


def test_loss_cost_is_exposure_weighted_mean() -> None:
    """Verifies loss cost equals the exposure-weighted fitted mean per segment."""
    view, fitted, config = _aggregation_fixture()
    assigned = build_policy_segments(view, config.segments, config.feature_pipeline)
    definitions = aggregate_segment_economics(assigned, fitted, config.segments)

    joined = assigned.merge(fitted, on="PolicyID", how="inner")
    for definition in definitions:
        group = joined[joined["segment_id"] == definition.segment_id]
        expected = float(
            (group["fitted_loss_cost_per_year"] * group["exposure_capped"]).sum()
            / group["exposure_capped"].sum()
        )
        assert definition.loss_cost_per_year == pytest.approx(round(expected, 6))
        assert definition.policy_count == 1050


def test_base_premium_is_loaded_loss_cost() -> None:
    """Verifies base premium equals loss cost times one plus loading."""
    view, fitted, config = _aggregation_fixture()
    assigned = build_policy_segments(view, config.segments, config.feature_pipeline)
    definitions = aggregate_segment_economics(assigned, fitted, config.segments)

    for definition in definitions:
        expected = definition.loss_cost_per_year * (1.0 + config.segments.premium_loading)
        assert definition.base_premium_per_year == pytest.approx(round(expected, 6))


def test_floor_violation_raises() -> None:
    """Verifies a realized segment below the configured floor raises a typed error."""
    view, fitted, config = _aggregation_fixture()
    assigned = build_policy_segments(view, config.segments, config.feature_pipeline)
    strict = config.segments.model_copy(update={"min_policy_count": 2000})

    with pytest.raises(SegmentError) as exc_info:
        aggregate_segment_economics(assigned, fitted, strict)
    assert "segment_below_floor" in str(exc_info.value)


# --- Ratio-of-sums aggregation ---


def test_ratio_of_sums_matches_hand_calculation() -> None:
    """Verifies sum(tau)/sum(q0) on a known vector."""
    assert aggregate_ratio_of_sums([1.0, 2.0, 3.0], [0.5, 0.5, 0.5]) == pytest.approx(4.0)
    assert aggregate_ratio_of_sums(pd.Series([0.1, 0.2]), pd.Series([0.8, 0.2])) == pytest.approx(
        0.3
    )


def test_ratio_of_sums_rejects_zero_denominator() -> None:
    """Verifies a non-positive denominator raises."""
    with pytest.raises(SegmentError) as exc_info:
        aggregate_ratio_of_sums([1.0, 2.0], [0.0, 0.0])
    assert "non_positive_denominator" in str(exc_info.value)


def test_ratio_of_sums_rejects_length_mismatch() -> None:
    """Verifies mismatched lengths raise."""
    with pytest.raises(SegmentError) as exc_info:
        aggregate_ratio_of_sums([1.0], [0.5, 0.5])
    assert "length_mismatch" in str(exc_info.value)


def test_ratio_of_sums_rejects_empty_input() -> None:
    """Verifies empty input raises."""
    with pytest.raises(SegmentError) as exc_info:
        aggregate_ratio_of_sums([], [])
    assert "empty_input" in str(exc_info.value)


def test_ratio_of_sums_rejects_non_finite() -> None:
    """Verifies non-finite input raises."""
    with pytest.raises(SegmentError) as exc_info:
        aggregate_ratio_of_sums([float("inf")], [0.5])
    assert "non_finite_input" in str(exc_info.value)


# --- Orchestration, determinism, and empirical integration ---


def _empirical_inputs() -> tuple[Path, Path]:
    """Returns the empirical features/fitted paths, skipping if absent."""
    features = Path("data/processed/features_glm.csv")
    fitted = Path("data/processed/glm_fitted_loss_cost.csv")
    if not features.is_file() or not fitted.is_file():
        pytest.skip("Processed feature/GLM artifacts not present; run dvc repro first.")
    return features, fitted


def test_empirical_segment_pipeline(tmp_path: Path) -> None:
    """Runs the full segment pipeline on the empirical data and checks Gate 4 facts."""
    features, fitted = _empirical_inputs()

    metadata = run_segment_pipeline(
        features_path=features,
        fitted_path=fitted,
        loss_cost_path=tmp_path / "segments.csv",
        definitions_path=tmp_path / "definitions.json",
        policy_segments_path=tmp_path / "policy_segments.csv",
    )

    assert metadata.segment_count == 35
    assert metadata.initial_cell_count == 36
    assert metadata.total_policies == 413_169
    assert metadata.min_policy_count == 1000

    table = pd.read_csv(tmp_path / "segments.csv")
    assert len(table) == 35
    assert int(table["policy_count"].min()) >= 1000
    assert int(table["policy_count"].sum()) == 413_169
    merged = table[table["is_merged"]]
    assert len(merged) == 1
    assert merged.iloc[0]["segment_id"] == "D:18-24_C:5-9_Dens:>500"
    assert int(merged.iloc[0]["policy_count"]) == 2234

    mapping = pd.read_csv(tmp_path / "policy_segments.csv")
    assert len(mapping) == 413_169
    assert mapping["segment_id"].nunique() == 35
    assert int(mapping.isna().sum().sum()) == 0


def test_artifact_determinism(tmp_path: Path) -> None:
    """Verifies two consecutive runs produce byte-identical artifacts."""
    features, fitted = _empirical_inputs()
    outputs = {
        "loss_cost_path": tmp_path / "a_segments.csv",
        "definitions_path": tmp_path / "a_definitions.json",
        "policy_segments_path": tmp_path / "a_policy_segments.csv",
    }
    run_segment_pipeline(features_path=features, fitted_path=fitted, **outputs)
    first = {name: path.read_bytes() for name, path in outputs.items()}

    run_segment_pipeline(features_path=features, fitted_path=fitted, **outputs)
    second = {name: path.read_bytes() for name, path in outputs.items()}

    assert first == second

