"""Tests for the Stage 8 causal estimator and segment inference (P2-F06)."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from aegis.config.loader import get_config
from aegis.config.schema import AEGISConfig, EstimatorConfig, EstimatorPathsConfig
from aegis.pipelines.causal_artifacts import (
    build_model_version,
    compute_config_fingerprint,
    compute_file_sha256,
    write_estimator_artifact,
    write_segment_table,
)
from aegis.pipelines.causal_estimator import (
    TRUTH_COLUMNS,
    aggregate_segment_estimates,
    build_nuisance_models,
    estimate_baseline_retention,
    fit_causal_estimator,
    interval_quality_rates,
    predict_policy_effects,
    run_causal_estimator_pipeline,
    stratified_sample_split,
    validate_training_contract,
)
from aegis.schemas.causal import (
    CausalEstimatorMetadata,
    CausalEstimatorRunManifest,
    SegmentElasticityEstimate,
    SegmentElasticityTable,
)
from aegis.utils.exceptions import CausalEstimatorError

_SEGMENT_A = "D:25-44_C:0-4_Dens:<=500"
_SEGMENT_B = "D:45-64_C:10+_Dens:>2000"


def _synthetic_training_frame(rows: int = 1_000, seed: int = 7) -> pd.DataFrame:
    """Builds a contract-compliant synthetic table for fast estimator tests."""
    rng = np.random.default_rng(seed)
    segment_ids = [_SEGMENT_A if index % 2 == 0 else _SEGMENT_B for index in range(rows)]
    frame = pd.DataFrame(
        {
            "PolicyID": [f"P{index:06d}" for index in range(rows)],
            "treatment_rate_change": rng.uniform(-0.15, 0.15, rows),
            "segment_id": segment_ids,
        }
    )
    for index in range(25):
        frame[f"feature_{index:02d}"] = rng.normal(size=rows)
    logits = 1.0 - 4.0 * frame["treatment_rate_change"].to_numpy()
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    frame["renewed"] = rng.binomial(1, probabilities)
    return frame


def _fast_estimator_config() -> EstimatorConfig:
    """Returns a small estimator configuration for unit-speed fits."""
    return EstimatorConfig(
        n_estimators=10,
        min_samples_leaf=2,
        min_samples_split=4,
        max_samples=0.5,
        cv_folds=2,
        subforest_size=2,
        n_jobs=-1,
    )


# --- Training contract ---


def _valid_segment_payload(**overrides: object) -> dict[str, object]:
    """Builds a self-consistent segment estimate payload."""
    q0 = 0.83
    tau = -0.30
    tau_lower = -0.32
    tau_upper = -0.28
    payload: dict[str, object] = {
        "segment_id": "D:25-44_C:0-4_Dens:<=500",
        "inference_policy_count": 200,
        "baseline_retention_q0": q0,
        "tau_hat": tau,
        "tau_ci_lower": tau_lower,
        "tau_ci_upper": tau_upper,
        "epsilon_hat": tau / q0,
        "epsilon_ci_lower": tau_lower / q0,
        "epsilon_ci_upper": tau_upper / q0,
    }
    payload.update(overrides)
    return payload


def _valid_metadata_payload(**overrides: object) -> dict[str, object]:
    """Builds a valid deterministic metadata payload."""
    payload: dict[str, object] = {
        "nuisance_model": "gbt",
        "cv_folds": 5,
        "random_state": 42,
        "n_estimators": 100,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_depth": None,
        "max_samples": 0.45,
        "subforest_size": 4,
        "training_sample_size": 1000,
        "train_rows": 800,
        "test_rows": 200,
        "feature_count": 25,
        "segment_count": 35,
        "ci_alpha": 0.05,
        "q0_floor": 1e-6,
        "q0_estimation_method": "deconfounded_nuisance_fold_average",
        "interval_self_consistency_rate": 1.0,
        "finite_interval_rate": 1.0,
        "input_hashes": {"training_table": "a" * 64},
        "config_fingerprint": "b" * 64,
        "model_version": "cfdml-gbt-cv5-test",
    }
    payload.update(overrides)
    return payload


# --- SegmentElasticityEstimate ---


def test_segment_estimate_accepts_valid_payload_and_round_trips() -> None:
    estimate = SegmentElasticityEstimate.model_validate(_valid_segment_payload())
    restored = SegmentElasticityEstimate.model_validate(estimate.model_dump())
    assert restored == estimate


@pytest.mark.parametrize(
    "field",
    [
        "tau_hat",
        "tau_ci_lower",
        "tau_ci_upper",
        "epsilon_hat",
        "epsilon_ci_lower",
        "epsilon_ci_upper",
    ],
)
def test_segment_estimate_rejects_non_finite_values(field: str) -> None:
    payload = _valid_segment_payload()
    payload[field] = math.inf
    with pytest.raises(ValidationError, match="finite"):
        SegmentElasticityEstimate.model_validate(payload)


def test_segment_estimate_rejects_infinite_q0() -> None:
    payload = _valid_segment_payload(baseline_retention_q0=math.inf)
    with pytest.raises(ValidationError, match="less than or equal"):
        SegmentElasticityEstimate.model_validate(payload)


def test_segment_estimate_rejects_inverted_tau_interval() -> None:
    payload = _valid_segment_payload(tau_ci_lower=-0.20, tau_ci_upper=-0.40)
    with pytest.raises(ValidationError, match="tau_ci_lower"):
        SegmentElasticityEstimate.model_validate(payload)


def test_segment_estimate_rejects_tau_point_outside_interval() -> None:
    payload = _valid_segment_payload(tau_hat=-0.10)
    with pytest.raises(ValidationError, match="tau interval"):
        SegmentElasticityEstimate.model_validate(payload)


def test_segment_estimate_rejects_inverted_epsilon_interval() -> None:
    payload = _valid_segment_payload(epsilon_ci_lower=0.5, epsilon_ci_upper=-0.5)
    with pytest.raises(ValidationError, match="epsilon_ci_lower"):
        SegmentElasticityEstimate.model_validate(payload)


def test_segment_estimate_rejects_epsilon_point_outside_interval() -> None:
    payload = _valid_segment_payload(epsilon_hat=-0.10)
    with pytest.raises(ValidationError, match="epsilon interval"):
        SegmentElasticityEstimate.model_validate(payload)


def test_segment_estimate_rejects_non_positive_q0() -> None:
    payload = _valid_segment_payload(baseline_retention_q0=0.0)
    with pytest.raises(ValidationError):
        SegmentElasticityEstimate.model_validate(payload)


def test_segment_estimate_rejects_inconsistent_epsilon() -> None:
    payload = _valid_segment_payload(epsilon_hat=-0.36)
    with pytest.raises(ValidationError, match="epsilon_hat must equal"):
        SegmentElasticityEstimate.model_validate(payload)


def test_segment_estimate_rejects_unknown_field() -> None:
    payload = _valid_segment_payload(proposed_adjustment=0.05)
    with pytest.raises(ValidationError):
        SegmentElasticityEstimate.model_validate(payload)


def test_segment_estimate_rejects_invalid_segment_id() -> None:
    payload = _valid_segment_payload(segment_id="segment-1")
    with pytest.raises(ValidationError):
        SegmentElasticityEstimate.model_validate(payload)


# --- SegmentElasticityTable ---


def test_segment_table_rejects_duplicate_segment_ids() -> None:
    estimate = SegmentElasticityEstimate.model_validate(_valid_segment_payload())
    with pytest.raises(ValidationError, match="duplicate segment_id"):
        SegmentElasticityTable(segments=[estimate, estimate])


def test_segment_table_rejects_empty_table() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        SegmentElasticityTable(segments=[])


# --- CausalEstimatorMetadata ---


def test_metadata_accepts_valid_payload_and_round_trips() -> None:
    metadata = CausalEstimatorMetadata.model_validate(_valid_metadata_payload())
    restored = CausalEstimatorMetadata.model_validate(metadata.model_dump())
    assert restored == metadata


def test_metadata_requires_full_interval_rates() -> None:
    with pytest.raises(ValidationError, match="interval_self_consistency_rate"):
        CausalEstimatorMetadata.model_validate(
            _valid_metadata_payload(interval_self_consistency_rate=0.99)
        )
    with pytest.raises(ValidationError, match="finite_interval_rate"):
        CausalEstimatorMetadata.model_validate(
            _valid_metadata_payload(finite_interval_rate=0.99)
        )


def test_metadata_rejects_split_accounting_mismatch() -> None:
    with pytest.raises(ValidationError, match="train_rows"):
        CausalEstimatorMetadata.model_validate(
            _valid_metadata_payload(train_rows=700, test_rows=200)
        )


def test_metadata_rejects_bad_input_hash() -> None:
    with pytest.raises(ValidationError, match="lowercase SHA-256"):
        CausalEstimatorMetadata.model_validate(
            _valid_metadata_payload(input_hashes={"training_table": "not-a-hash"})
        )


def test_metadata_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        CausalEstimatorMetadata.model_validate(
            _valid_metadata_payload(fit_wall_seconds=12.3)
        )


# --- CausalEstimatorRunManifest ---


def test_run_manifest_accepts_valid_payload() -> None:
    manifest = CausalEstimatorRunManifest(
        estimator_artifact_sha256="c" * 64,
        estimator_artifact_bytes=1024,
        fit_wall_seconds=500.0,
        pipeline_wall_seconds=540.0,
        artifact_path="artifacts/causal_estimator/estimator.joblib",
    )
    assert manifest.estimator_artifact_bytes == 1024


def test_run_manifest_rejects_bad_hash_and_negative_timing() -> None:
    with pytest.raises(ValidationError):
        CausalEstimatorRunManifest(
            estimator_artifact_sha256="bad",
            estimator_artifact_bytes=1024,
            fit_wall_seconds=1.0,
            pipeline_wall_seconds=2.0,
            artifact_path="estimator.joblib",
        )
    with pytest.raises(ValidationError):
        CausalEstimatorRunManifest(
            estimator_artifact_sha256="c" * 64,
            estimator_artifact_bytes=1024,
            fit_wall_seconds=-1.0,
            pipeline_wall_seconds=2.0,
            artifact_path="estimator.joblib",
        )

# --- Training contract ---


def test_validate_training_contract_returns_ordered_feature_columns() -> None:
    frame = _synthetic_training_frame()
    features = validate_training_contract(frame, corridor_width=0.15)
    assert features == [f"feature_{index:02d}" for index in range(25)]
    assert not set(features) & set(TRUTH_COLUMNS)


@pytest.mark.parametrize("truth_column", sorted(TRUTH_COLUMNS))
def test_validate_training_contract_rejects_truth_columns(truth_column: str) -> None:
    frame = _synthetic_training_frame()
    frame[truth_column] = 0.0
    with pytest.raises(CausalEstimatorError, match="Truth columns"):
        validate_training_contract(frame, corridor_width=0.15)


def test_validate_training_contract_rejects_missing_metadata() -> None:
    frame = _synthetic_training_frame().drop(columns=["segment_id"])
    with pytest.raises(CausalEstimatorError, match="missing required columns"):
        validate_training_contract(frame, corridor_width=0.15)


def test_validate_training_contract_rejects_extra_feature() -> None:
    frame = _synthetic_training_frame()
    frame["feature_extra"] = 0.0
    with pytest.raises(CausalEstimatorError, match="exactly 29 columns"):
        validate_training_contract(frame, corridor_width=0.15)


def test_validate_training_contract_rejects_duplicate_policy_ids() -> None:
    frame = _synthetic_training_frame()
    frame.loc[1, "PolicyID"] = frame.loc[0, "PolicyID"]
    with pytest.raises(CausalEstimatorError, match="duplicate PolicyID"):
        validate_training_contract(frame, corridor_width=0.15)


def test_validate_training_contract_rejects_out_of_corridor_treatment() -> None:
    frame = _synthetic_training_frame()
    frame.loc[0, "treatment_rate_change"] = 0.20
    with pytest.raises(CausalEstimatorError, match="treatment_rate_change must lie"):
        validate_training_contract(frame, corridor_width=0.15)


def test_validate_training_contract_rejects_non_binary_outcome() -> None:
    frame = _synthetic_training_frame()
    frame.loc[0, "renewed"] = 2
    with pytest.raises(CausalEstimatorError, match="renewed must be binary"):
        validate_training_contract(frame, corridor_width=0.15)


# --- Split ---


def test_split_preserves_segments_and_is_deterministic() -> None:
    frame = _synthetic_training_frame(rows=2_000)
    first = stratified_sample_split(
        frame, 2_000, test_size=0.20, random_state=42
    )
    second = stratified_sample_split(
        frame, 2_000, test_size=0.20, random_state=42
    )
    train, test = first
    assert train.equals(second[0])
    assert test.equals(second[1])
    assert len(train) == 1_600
    assert len(test) == 400
    assert set(train["segment_id"]) == {_SEGMENT_A, _SEGMENT_B}
    assert set(test["segment_id"]) == {_SEGMENT_A, _SEGMENT_B}
    assert set(train["PolicyID"]).isdisjoint(set(test["PolicyID"]))


def test_split_honors_sample_size_before_split() -> None:
    frame = _synthetic_training_frame(rows=2_000)
    train, test = stratified_sample_split(
        frame, 1_000, test_size=0.20, random_state=42
    )
    assert len(train) == 800
    assert len(test) == 200


def test_split_rejects_oversized_sample() -> None:
    frame = _synthetic_training_frame(rows=100)
    with pytest.raises(CausalEstimatorError, match="sample_size must be"):
        stratified_sample_split(frame, 101, test_size=0.20, random_state=42)


# --- Nuisance models ---


def test_build_nuisance_models_gbt() -> None:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    model_y, model_t = build_nuisance_models("gbt", random_state=42, cv_folds=5)
    assert isinstance(model_y, GradientBoostingClassifier)
    assert isinstance(model_t, GradientBoostingRegressor)


def test_build_nuisance_models_linear() -> None:
    from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV

    model_y, model_t = build_nuisance_models("linear", random_state=42, cv_folds=3)
    assert isinstance(model_y, LogisticRegressionCV)
    assert isinstance(model_t, ElasticNetCV)


def test_build_nuisance_models_rejects_unknown_family() -> None:
    with pytest.raises(CausalEstimatorError, match="Unsupported nuisance"):
        build_nuisance_models("random_forest", random_state=42, cv_folds=5)


# --- Fit and per-policy effects ---


@pytest.fixture(scope="module")
def fitted_estimator() -> tuple[object, pd.DataFrame, list[str]]:
    """Fits one fast estimator on synthetic data for inference tests."""
    frame = _synthetic_training_frame(rows=1_200)
    features = validate_training_contract(frame, corridor_width=0.15)
    train, test = stratified_sample_split(
        frame, 1_200, test_size=0.20, random_state=42
    )
    estimator = fit_causal_estimator(
        train,
        features,
        estimator_cfg=_fast_estimator_config(),
        random_state=42,
    )
    return estimator, test, features


def test_fit_and_predict_effects_are_finite_and_negative(
    fitted_estimator: tuple[object, pd.DataFrame, list[str]],
) -> None:
    estimator, test, features = fitted_estimator
    effects = predict_policy_effects(estimator, test[features].to_numpy())
    assert effects.shape == (len(test),)
    assert np.isfinite(effects).all()
    assert effects.mean() < 0.0


def test_fit_pins_forest_inference_to_one_thread(
    fitted_estimator: tuple[object, pd.DataFrame, list[str]],
) -> None:
    estimator = cast(Any, fitted_estimator[0])
    fitted_models = estimator.model_final_.estimators_
    assert fitted_models
    assert all(model.n_jobs == 1 for model in fitted_models)


class _StubEstimator:
    """Minimal estimator stub for shape and finiteness tests."""

    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def effect(self, X: np.ndarray, T0: int = 0, T1: int = 1) -> np.ndarray:
        return self.output


def test_predict_policy_effects_flattens_single_column() -> None:
    stub = _StubEstimator(np.full((3, 1), -0.2))
    effects = predict_policy_effects(stub, np.zeros((3, 2)))
    assert effects.shape == (3,)
    assert np.allclose(effects, -0.2)


def test_predict_policy_effects_rejects_non_finite_values() -> None:
    stub = _StubEstimator(np.array([0.1, np.inf, 0.2]))
    with pytest.raises(CausalEstimatorError, match="non-finite"):
        predict_policy_effects(stub, np.zeros((3, 2)))


def test_predict_policy_effects_rejects_wrong_shape() -> None:
    stub = _StubEstimator(np.zeros((3, 2)))
    with pytest.raises(CausalEstimatorError, match="shape"):
        predict_policy_effects(stub, np.zeros((3, 2)))


# --- Truth-free baseline retention (q0) ---


class _ProbabilityModel:
    """Fixed-probability outcome nuisance stub."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probability = np.full(len(X), self.probability)
        return np.column_stack([1.0 - probability, probability])


class _TreatmentModel:
    """Fixed-value treatment nuisance stub."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value)


class _NuisanceStub:
    """Estimator stub exposing cross-fitted nuisance models and a fixed effect."""

    def __init__(
        self,
        probabilities: list[float],
        treatment_values: list[float],
        tau: float,
    ) -> None:
        self.models_y = [[_ProbabilityModel(value) for value in probabilities]]
        self.models_t = [[_TreatmentModel(value) for value in treatment_values]]
        self.tau = tau

    def effect(self, X: np.ndarray, T0: int = 0, T1: int = 1) -> np.ndarray:
        return np.full(len(X), self.tau)


def test_q0_is_fold_averaged_and_deconfounded() -> None:
    stub = _NuisanceStub(
        probabilities=[0.7, 0.9],
        treatment_values=[0.2, 0.0],
        tau=-0.5,
    )
    q0 = estimate_baseline_retention(
        stub,
        np.zeros((4, 2)),
        q0_floor=1e-6,
        cv_folds=2,
    )
    assert np.allclose(q0, (0.8 + 0.9) / 2.0)


def test_q0_is_clipped_to_the_floor() -> None:
    stub = _NuisanceStub(
        probabilities=[0.1, 0.1],
        treatment_values=[0.5, 0.5],
        tau=0.8,
    )
    q0 = estimate_baseline_retention(
        stub,
        np.zeros((3, 2)),
        q0_floor=1e-6,
        cv_folds=2,
    )
    assert np.all(q0 == 1e-6)


def test_q0_rejects_fold_count_mismatch() -> None:
    stub = _NuisanceStub(
        probabilities=[0.7, 0.9],
        treatment_values=[0.2, 0.0],
        tau=-0.5,
    )
    with pytest.raises(CausalEstimatorError, match="fold count"):
        estimate_baseline_retention(
            stub,
            np.zeros((3, 2)),
            q0_floor=1e-6,
            cv_folds=3,
        )


def test_q0_rejects_non_finite_nuisance_output() -> None:
    stub = _NuisanceStub(
        probabilities=[math.nan, 0.9],
        treatment_values=[0.2, 0.0],
        tau=-0.5,
    )
    with pytest.raises(CausalEstimatorError, match="non-finite"):
        estimate_baseline_retention(
            stub,
            np.zeros((3, 2)),
            q0_floor=1e-6,
            cv_folds=2,
        )


def test_q0_on_fitted_estimator_is_finite_and_bounded(
    fitted_estimator: tuple[object, pd.DataFrame, list[str]],
) -> None:
    estimator, test, features = fitted_estimator
    q0 = estimate_baseline_retention(
        estimator,
        test[features].to_numpy(),
        q0_floor=1e-6,
        cv_folds=2,
    )
    assert q0.shape == (len(test),)
    assert np.isfinite(q0).all()
    assert np.all(q0 >= 1e-6)
    assert np.all(q0 <= 1.0)


# --- Segment aggregation and intervals ---


def test_aggregate_segment_estimates_are_finite_and_self_consistent(
    fitted_estimator: tuple[object, pd.DataFrame, list[str]],
) -> None:
    estimator, test, features = fitted_estimator
    table = aggregate_segment_estimates(
        estimator,
        test,
        features,
        ci_alpha=0.05,
        q0_floor=1e-6,
        cv_folds=2,
    )
    assert len(table.segments) == 2
    assert [segment.segment_id for segment in table.segments] == sorted(
        test["segment_id"].unique().tolist()
    )
    assert sum(segment.inference_policy_count for segment in table.segments) == len(test)
    for segment in table.segments:
        assert math.isclose(
            segment.epsilon_hat,
            segment.tau_hat / segment.baseline_retention_q0,
            rel_tol=1e-12,
        )
        assert segment.tau_ci_lower <= segment.tau_hat <= segment.tau_ci_upper
        assert (
            segment.epsilon_ci_lower
            <= segment.epsilon_hat
            <= segment.epsilon_ci_upper
        )
    assert interval_quality_rates(table) == (1.0, 1.0)


def test_aggregate_segment_estimates_rejects_invalid_alpha(
    fitted_estimator: tuple[object, pd.DataFrame, list[str]],
) -> None:
    estimator, test, features = fitted_estimator
    with pytest.raises(CausalEstimatorError, match="ci_alpha"):
        aggregate_segment_estimates(
            estimator,
            test,
            features,
            ci_alpha=1.0,
            q0_floor=1e-6,
            cv_folds=2,
        )


# --- Metadata, persistence, and orchestration ---


def test_compute_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"aegis")
    assert compute_file_sha256(path) == hashlib.sha256(b"aegis").hexdigest()


def test_config_fingerprint_ignores_non_behavioral_settings() -> None:
    cfg = _fast_estimator_config()
    baseline = compute_config_fingerprint(
        cfg, random_state=42, test_size=0.20, corridor_width=0.15
    )
    assert baseline == compute_config_fingerprint(
        cfg.model_copy(update={"n_jobs": 1, "runtime_budget_minutes": 90.0}),
        random_state=42,
        test_size=0.20,
        corridor_width=0.15,
    )
    assert baseline != compute_config_fingerprint(
        cfg.model_copy(update={"n_estimators": 20}),
        random_state=42,
        test_size=0.20,
        corridor_width=0.15,
    )
    assert baseline != compute_config_fingerprint(
        cfg, random_state=43, test_size=0.20, corridor_width=0.15
    )


def test_build_model_version_is_deterministic() -> None:
    cfg = _fast_estimator_config()
    version = build_model_version(cfg)
    assert version == build_model_version(cfg)
    assert version.startswith("cfdml-gbt-cv2-n10")
    assert build_model_version(cfg.model_copy(update={"max_depth": 4})).endswith(
        "-depth4-ms0.5-sub2"
    )


def test_write_segment_table_round_trips(tmp_path: Path) -> None:
    estimate = SegmentElasticityEstimate.model_validate(_valid_segment_payload())
    table = SegmentElasticityTable(segments=[estimate])
    path = tmp_path / "segments.csv"
    write_segment_table(table, path)
    restored = pd.read_csv(path)
    assert restored.columns.tolist() == [
        "segment_id",
        "inference_policy_count",
        "baseline_retention_q0",
        "tau_hat",
        "tau_ci_lower",
        "tau_ci_upper",
        "epsilon_hat",
        "epsilon_ci_lower",
        "epsilon_ci_upper",
    ]
    assert restored.loc[0, "segment_id"] == estimate.segment_id
    assert restored.loc[0, "inference_policy_count"] == estimate.inference_policy_count


def test_write_estimator_artifact_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "estimator.joblib"
    payload = {"model": "stub", "n": 1}
    write_estimator_artifact(payload, path)
    assert joblib.load(path) == payload


def _pipeline_config(
    training_path: Path, definitions_path: Path, root: Path
) -> AEGISConfig:
    config = get_config().model_copy(deep=True)
    config.tier1_ml.dgp_paths.training_table = str(training_path)
    config.tier1_ml.segment_paths.segment_definitions = str(definitions_path)
    config.tier1_ml.estimator = _fast_estimator_config().model_copy(
        update={"training_sample_size": 1_200}
    )
    config.tier1_ml.estimator_paths = EstimatorPathsConfig(
        segment_estimates=str(root / "segment_elasticity_estimates.csv"),
        metadata=str(root / "causal_estimator_metadata.json"),
        model_artifact=str(root / "estimator.joblib"),
        run_manifest=str(root / "run_manifest.json"),
    )
    return config


def test_run_pipeline_writes_artifacts_and_metadata(tmp_path: Path) -> None:
    root = tmp_path
    training_path = root / "elasticity_training.csv"
    _synthetic_training_frame(rows=1_200).to_csv(training_path, index=False)
    definitions_path = root / "segment_definitions.json"
    definitions_path.write_text(json.dumps({"segment_count": 2}), encoding="utf-8")
    config = _pipeline_config(training_path, definitions_path, root)

    metadata = run_causal_estimator_pipeline(config)

    assert metadata.training_sample_size == 1_200
    assert metadata.train_rows == 960
    assert metadata.test_rows == 240
    assert metadata.segment_count == 2
    assert metadata.interval_self_consistency_rate == 1.0
    assert metadata.finite_interval_rate == 1.0
    assert metadata.input_hashes["training_table"] == compute_file_sha256(training_path)
    assert len(metadata.config_fingerprint) == 64
    assert metadata.model_version.startswith("cfdml-gbt")

    table_path = root / "segment_elasticity_estimates.csv"
    table = pd.read_csv(table_path)
    assert len(table) == 2
    assert math.isclose(
        table.loc[0, "epsilon_hat"],
        table.loc[0, "tau_hat"] / table.loc[0, "baseline_retention_q0"],
        rel_tol=1e-12,
    )

    artifact_path = root / "estimator.joblib"
    assert artifact_path.is_file()
    assert artifact_path.stat().st_size > 0
    manifest = json.loads(
        (root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["estimator_artifact_sha256"] == compute_file_sha256(artifact_path)
    assert manifest["estimator_artifact_bytes"] == artifact_path.stat().st_size
    saved_metadata = json.loads(
        (root / "causal_estimator_metadata.json").read_text(encoding="utf-8")
    )
    assert saved_metadata["segment_count"] == 2
    assert joblib.load(artifact_path) is not None


def test_run_pipeline_rejects_segment_coverage_mismatch(tmp_path: Path) -> None:
    root = tmp_path
    training_path = root / "elasticity_training.csv"
    _synthetic_training_frame(rows=1_200).to_csv(training_path, index=False)
    definitions_path = root / "segment_definitions.json"
    definitions_path.write_text(json.dumps({"segment_count": 3}), encoding="utf-8")
    config = _pipeline_config(training_path, definitions_path, root)

    with pytest.raises(CausalEstimatorError, match="segments"):
        run_causal_estimator_pipeline(config)


def test_run_pipeline_rejects_missing_training_table(tmp_path: Path) -> None:
    root = tmp_path
    definitions_path = root / "segment_definitions.json"
    definitions_path.write_text(json.dumps({"segment_count": 2}), encoding="utf-8")
    config = _pipeline_config(root / "missing.csv", definitions_path, root)

    with pytest.raises(CausalEstimatorError, match="not found"):
        run_causal_estimator_pipeline(config)
