"""Unit tests for the Stage 7 Q2 qualification helpers."""

import numpy as np
import pandas as pd
import pytest
from scripts.q2_estimator_scale import (
    CellResult,
    aggregate_segment_effects,
    estimator_feature_columns,
    select_estimator_configuration,
    stratified_sample_split,
)


def _cell(
    sample_size: int,
    nuisance_model: str,
    spearman: float,
    runtime_seconds: float = 60.0,
    peak_rss_mb: float = 100.0,
) -> CellResult:
    return CellResult(
        sample_size=sample_size,
        nuisance_model=nuisance_model,
        wall_seconds=runtime_seconds,
        peak_rss_mb=peak_rss_mb,
        peak_traced_memory_mb=80.0,
        segment_spearman=spearman,
        ci_coverage=0.9,
        segment_count=35,
    )


def test_aggregate_segment_effects_uses_ratio_of_sums() -> None:
    effects = aggregate_segment_effects(
        predicted_tau=np.array([1.0, 3.0]),
        baseline_q0=np.array([0.9, 0.1]),
        segment_ids=np.array(["A", "A"]),
    )

    assert effects == {"A": 4.0}


def test_aggregate_segment_effects_accepts_single_treatment_column() -> None:
    effects = aggregate_segment_effects(
        predicted_tau=np.array([[1.0], [3.0]]),
        baseline_q0=np.array([0.9, 0.1]),
        segment_ids=np.array(["A", "A"]),
    )

    assert effects == {"A": 4.0}


def test_stratified_sample_split_preserves_sample_size_and_segment_coverage() -> None:
    data = pd.DataFrame(
        {
            "segment_id": ["A"] * 50 + ["B"] * 50,
            "PolicyID": range(100),
        }
    )

    train, test = stratified_sample_split(data, 50, test_size=0.2, random_state=42)

    assert len(train) == 40
    assert len(test) == 10
    assert set(train["segment_id"]) == {"A", "B"}
    assert set(test["segment_id"]) == {"A", "B"}
    assert set(train["PolicyID"]).isdisjoint(test["PolicyID"])


def test_estimator_feature_columns_excludes_training_metadata() -> None:
    feature_columns = [f"feature_{index}" for index in range(25)]
    columns = [
        "PolicyID",
        *feature_columns,
        "treatment_rate_change",
        "renewed",
        "segment_id",
    ]

    assert estimator_feature_columns(columns) == feature_columns


def test_estimator_feature_columns_rejects_truth_leakage() -> None:
    columns = [
        "PolicyID",
        *[f"feature_{index}" for index in range(24)],
        "true_tau",
        "treatment_rate_change",
        "renewed",
        "segment_id",
    ]

    with pytest.raises(ValueError, match="Truth columns cannot enter"):
        estimator_feature_columns(columns)


def test_selection_prefers_full_scale_and_gbt_when_scores_are_tied() -> None:
    results = [
        _cell(25_000, "gbt", 0.80),
        _cell(25_000, "linear", 0.81),
        _cell(100_000, "gbt", 0.86),
        _cell(100_000, "linear", 0.86),
        _cell(250_000, "gbt", 0.89),
        _cell(250_000, "linear", 0.90),
        _cell(413_169, "gbt", 0.91),
        _cell(413_169, "linear", 0.92),
    ]

    selected = select_estimator_configuration(
        results, runtime_budget_seconds=1800, ram_budget_mb=1000
    )

    assert selected.sample_size == 413_169
    assert selected.nuisance_model == "gbt"


def test_selection_uses_smallest_eligible_sample_when_full_scale_exceeds_budget() -> None:
    results = [
        _cell(25_000, "gbt", 0.80),
        _cell(25_000, "linear", 0.81),
        _cell(100_000, "gbt", 0.86),
        _cell(100_000, "linear", 0.87),
        _cell(250_000, "gbt", 0.88),
        _cell(250_000, "linear", 0.92),
        _cell(413_169, "gbt", 0.90, runtime_seconds=1900),
        _cell(413_169, "linear", 0.91, runtime_seconds=1900),
    ]

    selected = select_estimator_configuration(
        results, runtime_budget_seconds=1800, ram_budget_mb=1000
    )

    assert selected.sample_size == 250_000
    assert selected.nuisance_model == "linear"


def test_selection_fails_when_no_candidate_meets_the_budget() -> None:
    results = [
        _cell(25_000, "gbt", 0.80, runtime_seconds=1900),
        _cell(25_000, "linear", 0.81, runtime_seconds=1900),
    ]

    with pytest.raises(ValueError, match="No Q2 configuration meets"):
        select_estimator_configuration(results, runtime_budget_seconds=1800, ram_budget_mb=1000)
