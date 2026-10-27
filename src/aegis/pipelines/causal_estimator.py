"""CausalForestDML training and truth-free segment inference (P2-F06, Stage 8).

The pipeline validates the Stage 6 training contract, reproduces the Q2-frozen
stratified split, fits only on the training partition, estimates baseline retention
without the truth sidecar, and emits finite, self-consistent segment-average intervals.
Truth values never enter this module (ADR-013). No pricing adjustment is produced or
accepted (INV-4, INV-5).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from pydantic import ValidationError
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split

from aegis.config.schema import AEGISConfig, EstimatorConfig
from aegis.pipelines.causal_artifacts import (
    build_model_version,
    compute_config_fingerprint,
    compute_file_sha256,
    write_estimator_artifact,
    write_json_atomic,
    write_segment_table,
)
from aegis.schemas.causal import (
    CausalEstimatorMetadata,
    CausalEstimatorRunManifest,
    SegmentElasticityEstimate,
    SegmentElasticityTable,
)
from aegis.utils.exceptions import CausalEstimatorError

TRUTH_COLUMNS: frozenset[str] = frozenset(
    {
        "tau",
        "epsilon",
        "q0",
        "true_effect",
        "tau_ground_truth",
        "baseline_renewal_q0",
        "true_tau",
        "true_epsilon",
        "fitted_loss_cost",
    }
)

FEATURE_METADATA_COLUMNS: tuple[str, ...] = (
    "PolicyID",
    "treatment_rate_change",
    "renewed",
    "segment_id",
)
EXPECTED_FEATURE_COUNT = 25
EXPECTED_COLUMN_COUNT = 29


def _fail(
    message: str,
    rule: str,
    *,
    column: str | None = None,
    count: int = 0,
    sample_keys: list[str] | None = None,
) -> NoReturn:
    """Raises a typed estimator error with structured details."""
    details: dict[str, Any] = {"rule": rule, "count": count}
    if column is not None:
        details["column"] = column
    if sample_keys:
        details["sample_keys"] = sample_keys
    raise CausalEstimatorError(message, details=details)


def validate_training_contract(
    data: pd.DataFrame,
    *,
    corridor_width: float,
) -> list[str]:
    """Validates the Stage 6 training table and returns its 25 feature columns.

    Args:
        data: Candidate estimator training table.
        corridor_width: Maximum absolute treatment value from configuration.

    Returns:
        The ordered list of pre-treatment feature columns.

    Raises:
        CausalEstimatorError: If required columns are missing, truth columns are
            present, identifiers are invalid, or values fall outside the contract.
    """
    if data.empty:
        _fail("Training table is empty", "empty_training_table")

    columns = list(data.columns)
    if len(columns) != len(set(columns)):
        _fail("Training table has duplicate column names", "duplicate_columns")

    missing = [column for column in FEATURE_METADATA_COLUMNS if column not in columns]
    if missing:
        _fail(
            f"Training table is missing required columns: {missing}",
            "missing_columns",
            column=",".join(missing),
            count=len(missing),
        )

    leaked = sorted(TRUTH_COLUMNS & set(columns))
    if leaked:
        _fail(
            f"Truth columns cannot enter estimator inputs: {leaked}",
            "truth_column_leakage",
            column=",".join(leaked),
            count=len(leaked),
            sample_keys=leaked,
        )

    if len(columns) != EXPECTED_COLUMN_COUNT:
        _fail(
            f"Training table must have exactly {EXPECTED_COLUMN_COUNT} columns; "
            f"found {len(columns)}",
            "column_count_mismatch",
            count=len(columns),
        )

    feature_columns = [
        column for column in columns if column not in FEATURE_METADATA_COLUMNS
    ]
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        _fail(
            f"Expected {EXPECTED_FEATURE_COUNT} pre-treatment features; "
            f"found {len(feature_columns)}",
            "feature_count_mismatch",
            count=len(feature_columns),
        )

    policy_ids = cast(pd.Series, data["PolicyID"])
    if bool(policy_ids.isna().any()):
        _fail("PolicyID contains null values", "null_policy_id")
    if bool(policy_ids.duplicated().any()):
        _fail("Training table has duplicate PolicyID values", "duplicate_policy_id")

    segment_ids = cast(pd.Series, data["segment_id"])
    if bool(segment_ids.isna().any()) or bool((segment_ids.astype(str) == "").any()):
        _fail("segment_id contains null or empty values", "invalid_segment_id")

    treatment = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, data["treatment_rate_change"]), errors="coerce"),
    )
    if bool(treatment.isna().any()):
        _fail(
            "treatment_rate_change contains non-numeric values",
            "invalid_treatment",
            column="treatment_rate_change",
            count=int(treatment.isna().sum()),
        )
    out_of_corridor = treatment.lt(-corridor_width) | treatment.gt(corridor_width)
    if bool(out_of_corridor.any()):
        _fail(
            f"treatment_rate_change must lie in [{-corridor_width}, {corridor_width}]",
            "treatment_out_of_corridor",
            column="treatment_rate_change",
            count=int(out_of_corridor.sum()),
        )

    outcome = cast(
        pd.Series, pd.to_numeric(cast(pd.Series, data["renewed"]), errors="coerce")
    )
    if bool(outcome.isna().any()) or not bool(outcome.isin([0, 1]).all()):
        _fail(
            "renewed must be binary in {0, 1}",
            "non_binary_outcome",
            column="renewed",
            count=int(outcome.isna().sum()),
        )

    non_finite_features = int(
        np.sum(~np.isfinite(data[feature_columns].to_numpy(dtype=float)))
    )
    if non_finite_features:
        _fail(
            "Pre-treatment features contain non-finite values",
            "non_finite_features",
            count=non_finite_features,
        )

    return feature_columns


def stratified_sample_split(
    data: pd.DataFrame,
    sample_size: int,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draws a stratified total sample, then a stratified train/test split.

    Args:
        data: Validated estimator training table.
        sample_size: Total sample size before the train/test split.
        test_size: Held-out fraction in (0, 1).
        random_state: Seed for both stratified draws.

    Returns:
        The training and held-out partitions, with reset indices.

    Raises:
        CausalEstimatorError: If the requested sample size is invalid or the split
            drops a segment.
    """
    if "segment_id" not in data.columns:
        _fail(
            "Training data must include segment_id for stratification",
            "missing_segment_id",
        )
    if not 0 < test_size < 1:
        _fail(
            f"test_size must be in (0, 1); got {test_size}",
            "invalid_test_size",
        )
    if not 0 < sample_size <= len(data):
        _fail(
            f"sample_size must be in [1, {len(data)}]; got {sample_size}",
            "invalid_sample_size",
            count=sample_size,
        )

    if sample_size == len(data):
        sample = data
    else:
        sample, _ = cast(
            tuple[pd.DataFrame, pd.DataFrame],
            train_test_split(
                data,
                train_size=sample_size,
                stratify=data["segment_id"],
                random_state=random_state,
            ),
        )

    train, test = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            sample,
            test_size=test_size,
            stratify=sample["segment_id"],
            random_state=random_state,
        ),
    )
    if train.empty or test.empty:
        _fail("Stratified split produced an empty partition", "empty_split")

    missing_test_segments = sorted(
        set(sample["segment_id"].astype(str)) - set(test["segment_id"].astype(str))
    )
    if missing_test_segments:
        _fail(
            "Held-out split is missing segments",
            "missing_test_segments",
            count=len(missing_test_segments),
            sample_keys=missing_test_segments,
        )

    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_nuisance_models(
    family: str,
    *,
    random_state: int,
    cv_folds: int,
) -> tuple[Any, Any]:
    """Builds the Q2-approved outcome and treatment nuisance models.

    Args:
        family: Either `gbt` or `linear`.
        random_state: Seed shared by the nuisance estimators.
        cv_folds: Regularization cross-validation folds for the linear family.

    Returns:
        The outcome and treatment nuisance model instances.

    Raises:
        CausalEstimatorError: If the nuisance family is unsupported.
    """
    if family == "gbt":
        return (
            GradientBoostingClassifier(random_state=random_state),
            GradientBoostingRegressor(random_state=random_state),
        )
    if family == "linear":
        return (
            LogisticRegressionCV(
                cv=cv_folds, max_iter=1000, n_jobs=-1, random_state=random_state
            ),
            ElasticNetCV(
                cv=cv_folds, max_iter=10_000, n_jobs=-1, random_state=random_state
            ),
        )
    _fail(f"Unsupported nuisance model family: {family}", "unsupported_nuisance_family")


def fit_causal_estimator(
    train: pd.DataFrame,
    feature_columns: list[str],
    *,
    estimator_cfg: EstimatorConfig,
    random_state: int,
) -> Any:
    """Fits the Q2-frozen CausalForestDML on the training partition only.

    Args:
        train: Training rows. Must not contain held-out or truth data.
        feature_columns: The 25 pre-treatment feature columns.
        estimator_cfg: Frozen estimator configuration.
        random_state: Seed shared by the nuisance and forest models.

    Returns:
        The fitted CausalForestDML instance.

    Raises:
        CausalEstimatorError: If fitting fails.
    """
    model_y, model_t = build_nuisance_models(
        estimator_cfg.nuisance_model,
        random_state=random_state,
        cv_folds=estimator_cfg.cv_folds,
    )
    estimator = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_outcome=True,
        discrete_treatment=False,
        cv=estimator_cfg.cv_folds,
        n_estimators=estimator_cfg.n_estimators,
        min_samples_leaf=estimator_cfg.min_samples_leaf,
        min_samples_split=estimator_cfg.min_samples_split,
        max_depth=estimator_cfg.max_depth,
        max_samples=estimator_cfg.max_samples,
        subforest_size=estimator_cfg.subforest_size,
        inference=True,
        random_state=random_state,
        n_jobs=estimator_cfg.n_jobs,
    )

    x_train = train[feature_columns].to_numpy(dtype=float)
    treatment = train["treatment_rate_change"].to_numpy(dtype=float)
    outcome = train["renewed"].to_numpy(dtype=int)
    try:
        estimator.fit(outcome, treatment, X=x_train)
    except Exception as exc:
        raise CausalEstimatorError(
            f"CausalForestDML fit failed: {exc}",
            details={"rule": "fit_failed", "error_type": type(exc).__name__},
        ) from exc
    _pin_forest_inference_to_one_thread(estimator)
    return estimator


def _pin_forest_inference_to_one_thread(estimator: Any) -> None:
    """Serializes fitted-forest inference so per-tree accumulation is reproducible.

    EconML accumulates tree predictions under a lock when ``n_jobs != 1``, so
    floating-point summation order varies across runs. Tree fitting remains
    parallel; only prediction and inference are pinned to one thread.
    """
    for fitted_model in getattr(estimator.model_final_, "estimators_", []):
        fitted_model.n_jobs = 1


def predict_policy_effects(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Predicts one treatment effect per policy and normalizes the output shape.

    Args:
        estimator: Fitted CausalForestDML instance.
        X: Two-dimensional pre-treatment feature matrix.

    Returns:
        A one-dimensional finite array of per-policy effects.

    Raises:
        CausalEstimatorError: If prediction fails, the shape is invalid, or values
            are non-finite.
    """
    features = np.asarray(X, dtype=float)
    if features.ndim != 2:
        _fail("Effect features must be a two-dimensional matrix", "invalid_feature_shape")
    try:
        raw = estimator.effect(features, T0=0, T1=1)
    except Exception as exc:
        raise CausalEstimatorError(
            f"Causal effect prediction failed: {exc}",
            details={"rule": "effect_failed", "error_type": type(exc).__name__},
        ) from exc

    effects = np.asarray(raw, dtype=float)
    if effects.ndim == 2 and effects.shape[1] == 1:
        effects = effects[:, 0]
    if effects.ndim != 1:
        _fail(
            f"Expected one effect per policy; got shape {effects.shape}",
            "invalid_effect_shape",
        )
    if len(effects) != len(features):
        _fail(
            "Effect row count does not match feature row count",
            "effect_row_mismatch",
            count=len(effects),
        )
    non_finite = int(np.sum(~np.isfinite(effects)))
    if non_finite:
        _fail(
            "Per-policy effects contain non-finite values",
            "non_finite_effects",
            count=non_finite,
        )
    return effects


def estimate_baseline_retention(
    estimator: Any,
    X: np.ndarray,
    *,
    q0_floor: float,
    cv_folds: int,
) -> np.ndarray:
    """Estimates baseline retention without the truth sidecar.

    For every cross-fitting fold, the deconfounded nuisance estimate is
    `E[Y|X] - tau(X) * E[T|X]`, clipped to `[q0_floor, 1]`; the per-policy q0 is the
    fold average, clipped again. Truth values are never consulted (ADR-013, OD-1).

    Args:
        estimator: Fitted CausalForestDML instance.
        X: Two-dimensional pre-treatment feature matrix.
        q0_floor: Lower clip bound for baseline retention.
        cv_folds: Expected number of cross-fitting folds.

    Returns:
        A one-dimensional finite array of per-policy baseline retention estimates.

    Raises:
        CausalEstimatorError: If nuisance models are missing, shapes are invalid,
            values are non-finite, or the fold count differs from configuration.
    """
    if not 0.0 < q0_floor < 1.0:
        _fail(
            f"q0_floor must be in (0, 1); got {q0_floor}",
            "invalid_q0_floor",
        )
    features = np.asarray(X, dtype=float)
    if features.ndim != 2:
        _fail("q0 features must be a two-dimensional matrix", "invalid_feature_shape")

    try:
        outcome_models = estimator.models_y
        treatment_models = estimator.models_t
    except AttributeError as exc:
        raise CausalEstimatorError(
            "Fitted estimator does not expose cross-fitted nuisance models",
            details={"rule": "missing_nuisance_models"},
        ) from exc

    try:
        fold_outcome_models = outcome_models[0]
        fold_treatment_models = treatment_models[0]
    except (IndexError, TypeError) as exc:
        raise CausalEstimatorError(
            "Fitted estimator nuisance models are empty",
            details={"rule": "empty_nuisance_models"},
        ) from exc
    if len(fold_outcome_models) != cv_folds or len(fold_treatment_models) != cv_folds:
        _fail(
            "Cross-fitting fold count does not match configuration",
            "nuisance_fold_mismatch",
            count=len(fold_outcome_models),
        )

    effects = predict_policy_effects(estimator, features)
    fold_q0: list[np.ndarray] = []
    for fold_index, (model_y, model_t) in enumerate(
        zip(fold_outcome_models, fold_treatment_models, strict=True)
    ):
        try:
            probabilities = np.asarray(model_y.predict_proba(features), dtype=float)
            treatment_prediction = np.asarray(
                model_t.predict(features), dtype=float
            ).reshape(-1)
        except Exception as exc:
            raise CausalEstimatorError(
                f"q0 nuisance prediction failed for fold {fold_index}: {exc}",
                details={
                    "rule": "q0_nuisance_failed",
                    "fold": fold_index,
                    "error_type": type(exc).__name__,
                },
            ) from exc

        if (
            probabilities.ndim != 2
            or probabilities.shape[1] < 2
            or len(probabilities) != len(features)
        ):
            _fail(
                f"Outcome nuisance probabilities have invalid shape for fold {fold_index}",
                "invalid_q0_probability_shape",
                count=fold_index,
            )
        if len(treatment_prediction) != len(features):
            _fail(
                f"Treatment nuisance predictions have invalid shape for fold {fold_index}",
                "invalid_q0_treatment_shape",
                count=fold_index,
            )
        if not np.isfinite(probabilities).all() or not np.isfinite(
            treatment_prediction
        ).all():
            _fail(
                f"q0 nuisance outputs contain non-finite values in fold {fold_index}",
                "non_finite_q0_nuisance",
                count=fold_index,
            )

        outcome_probability = probabilities[:, 1]
        deconfounded = outcome_probability - effects * treatment_prediction
        fold_q0.append(np.clip(deconfounded, q0_floor, 1.0))

    averaged = np.clip(np.mean(np.vstack(fold_q0), axis=0), q0_floor, 1.0)
    non_finite = int(np.sum(~np.isfinite(averaged)))
    if non_finite:
        _fail(
            "Baseline retention estimates contain non-finite values",
            "non_finite_q0",
            count=non_finite,
        )
    return averaged


def aggregate_segment_estimates(
    estimator: Any,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    ci_alpha: float,
    q0_floor: float,
    cv_folds: int,
) -> SegmentElasticityTable:
    """Produces segment-average elasticity estimates and confidence intervals.

    Only held-out test policies are aggregated (OD-2). `tau` intervals come from
    `ate_inference`; `epsilon` intervals reuse the `tau` interval scaled by the
    segment-average q0 point estimate (OD-3).

    Args:
        estimator: Fitted CausalForestDML instance.
        test: Held-out policies; must include `segment_id` and the feature columns.
        feature_columns: The 25 pre-treatment feature columns.
        ci_alpha: Two-sided confidence-interval alpha.
        q0_floor: Lower clip bound for baseline retention.
        cv_folds: Expected number of cross-fitting folds.

    Returns:
        A validated, segment-sorted elasticity table.

    Raises:
        CausalEstimatorError: If inputs are invalid, inference fails, or a segment
            estimate cannot satisfy the output contract.
    """
    if not 0.0 < ci_alpha < 1.0:
        _fail(
            f"ci_alpha must be in (0, 1); got {ci_alpha}",
            "invalid_ci_alpha",
        )
    if test.empty:
        _fail("Held-out test table is empty", "empty_test_table")
    if "segment_id" not in test.columns:
        _fail("Held-out table must include segment_id", "missing_segment_id")

    features = test[feature_columns].to_numpy(dtype=float)
    q0 = estimate_baseline_retention(
        estimator,
        features,
        q0_floor=q0_floor,
        cv_folds=cv_folds,
    )
    segment_ids = cast(pd.Series, test["segment_id"]).astype(str).to_numpy()
    if np.any(segment_ids == ""):
        _fail("segment_id contains empty values", "invalid_segment_id")

    estimates: list[SegmentElasticityEstimate] = []
    for segment_id in sorted(set(segment_ids.tolist())):
        mask = segment_ids == segment_id
        policy_count = int(mask.sum())
        if policy_count == 0:
            continue
        segment_features = features[mask]
        try:
            inference = estimator.ate_inference(segment_features, T0=0, T1=1)
            interval = np.asarray(
                inference.conf_int_mean(alpha=ci_alpha), dtype=float
            ).reshape(-1)
            tau_hat = float(inference.mean_point)
        except Exception as exc:
            raise CausalEstimatorError(
                f"Segment inference failed for '{segment_id}': {exc}",
                details={
                    "rule": "segment_inference_failed",
                    "segment_id": segment_id,
                    "error_type": type(exc).__name__,
                },
            ) from exc

        if interval.size != 2:
            _fail(
                f"Expected a two-sided interval for '{segment_id}'",
                "invalid_interval_shape",
                count=int(interval.size),
            )
        tau_lower, tau_upper = float(interval[0]), float(interval[1])
        if not all(
            np.isfinite(value) for value in (tau_hat, tau_lower, tau_upper)
        ):
            _fail(
                f"Segment tau values are non-finite for '{segment_id}'",
                "non_finite_tau",
            )
        if tau_lower > tau_upper:
            _fail(
                f"Segment tau interval is inverted for '{segment_id}'",
                "inverted_tau_interval",
            )
        baseline_q0 = float(np.mean(q0[mask]))
        try:
            estimates.append(
                SegmentElasticityEstimate(
                    segment_id=segment_id,
                    inference_policy_count=policy_count,
                    baseline_retention_q0=baseline_q0,
                    tau_hat=tau_hat,
                    tau_ci_lower=tau_lower,
                    tau_ci_upper=tau_upper,
                    epsilon_hat=tau_hat / baseline_q0,
                    epsilon_ci_lower=tau_lower / baseline_q0,
                    epsilon_ci_upper=tau_upper / baseline_q0,
                )
            )
        except ValidationError as exc:
            raise CausalEstimatorError(
                f"Segment estimate failed output validation for '{segment_id}': {exc}",
                details={
                    "rule": "segment_output_invalid",
                    "segment_id": segment_id,
                },
            ) from exc

    if not estimates:
        _fail("No segment estimates were produced", "empty_segment_table")
    return SegmentElasticityTable(segments=estimates)


def interval_quality_rates(table: SegmentElasticityTable) -> tuple[float, float]:
    """Computes finite-interval and self-consistency rates for gate evidence.

    Args:
        table: Validated segment elasticity table.

    Returns:
        A pair `(finite_rate, self_consistency_rate)` in [0, 1]. The self-consistency
        rate counts intervals with ordered bounds that enclose their point estimate.
    """
    total = len(table.segments)
    finite_count = 0
    consistent_count = 0
    for segment in table.segments:
        values = (
            segment.tau_hat,
            segment.tau_ci_lower,
            segment.tau_ci_upper,
            segment.epsilon_hat,
            segment.epsilon_ci_lower,
            segment.epsilon_ci_upper,
        )
        if not all(math.isfinite(value) for value in values):
            continue
        finite_count += 1
        tau_consistent = (
            segment.tau_ci_lower <= segment.tau_hat <= segment.tau_ci_upper
        )
        epsilon_consistent = (
            segment.epsilon_ci_lower
            <= segment.epsilon_hat
            <= segment.epsilon_ci_upper
        )
        if tau_consistent and epsilon_consistent:
            consistent_count += 1
    return finite_count / total, consistent_count / total


def run_causal_estimator_pipeline(config: AEGISConfig) -> CausalEstimatorMetadata:
    """Runs the complete Stage 8 estimator pipeline and writes its artifacts.

    Args:
        config: Validated AEGIS configuration.

    Returns:
        Metadata describing the completed estimator run.

    Raises:
        CausalEstimatorError: If inputs are missing, the contract is violated, the
            fit fails, segment coverage is incomplete, or an artifact cannot be
            written.
    """
    started = time.perf_counter()
    ml_config = config.tier1_ml
    estimator_cfg = ml_config.estimator
    paths = ml_config.estimator_paths
    training_path = Path(ml_config.dgp_paths.training_table)
    definitions_path = Path(ml_config.segment_paths.segment_definitions)

    training_hash = compute_file_sha256(training_path)
    definitions_hash = compute_file_sha256(definitions_path)

    try:
        data = pd.read_csv(training_path)
    except Exception as exc:
        raise CausalEstimatorError(
            f"Failed to read training table: {exc}",
            details={"rule": "training_read_failed", "path": str(training_path)},
        ) from exc

    feature_columns = validate_training_contract(
        data, corridor_width=ml_config.exploration_corridor_width
    )

    try:
        definitions: Any = json.loads(definitions_path.read_text(encoding="utf-8"))
        expected_segment_count = int(definitions["segment_count"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CausalEstimatorError(
            f"Failed to read segment definitions: {exc}",
            details={
                "rule": "segment_definitions_invalid",
                "path": str(definitions_path),
            },
        ) from exc

    training_segments = set(
        cast(pd.Series, data["segment_id"]).astype(str).tolist()
    )
    if len(training_segments) != expected_segment_count:
        _fail(
            f"Training table covers {len(training_segments)} segments; "
            f"expected {expected_segment_count}",
            "segment_coverage_mismatch",
            count=len(training_segments),
        )

    train, test = stratified_sample_split(
        data,
        estimator_cfg.training_sample_size,
        test_size=ml_config.test_size,
        random_state=ml_config.random_state,
    )

    fit_started = time.perf_counter()
    estimator = fit_causal_estimator(
        train,
        feature_columns,
        estimator_cfg=estimator_cfg,
        random_state=ml_config.random_state,
    )
    fit_wall_seconds = time.perf_counter() - fit_started

    table = aggregate_segment_estimates(
        estimator,
        test,
        feature_columns,
        ci_alpha=estimator_cfg.ci_alpha,
        q0_floor=estimator_cfg.q0_floor,
        cv_folds=estimator_cfg.cv_folds,
    )
    table_segments = {segment.segment_id for segment in table.segments}
    if table_segments != training_segments:
        _fail(
            "Segment estimate coverage does not match the training table",
            "segment_table_coverage_mismatch",
            count=len(table_segments),
            sample_keys=sorted(training_segments.symmetric_difference(table_segments)),
        )
    finite_rate, consistency_rate = interval_quality_rates(table)

    metadata = CausalEstimatorMetadata(
        nuisance_model=estimator_cfg.nuisance_model,
        cv_folds=estimator_cfg.cv_folds,
        random_state=ml_config.random_state,
        n_estimators=estimator_cfg.n_estimators,
        min_samples_leaf=estimator_cfg.min_samples_leaf,
        min_samples_split=estimator_cfg.min_samples_split,
        max_depth=estimator_cfg.max_depth,
        max_samples=estimator_cfg.max_samples,
        subforest_size=estimator_cfg.subforest_size,
        training_sample_size=estimator_cfg.training_sample_size,
        train_rows=len(train),
        test_rows=len(test),
        feature_count=len(feature_columns),
        segment_count=len(table.segments),
        ci_alpha=estimator_cfg.ci_alpha,
        q0_floor=estimator_cfg.q0_floor,
        q0_estimation_method="deconfounded_nuisance_fold_average",
        interval_self_consistency_rate=consistency_rate,
        finite_interval_rate=finite_rate,
        input_hashes={
            "training_table": training_hash,
            "segment_definitions": definitions_hash,
        },
        config_fingerprint=compute_config_fingerprint(
            estimator_cfg,
            random_state=ml_config.random_state,
            test_size=ml_config.test_size,
            corridor_width=ml_config.exploration_corridor_width,
        ),
        model_version=f"{build_model_version(estimator_cfg)}-data{training_hash[:12]}",
    )

    table_path = Path(paths.segment_estimates)
    artifact_path = Path(paths.model_artifact)
    write_segment_table(table, table_path)
    write_estimator_artifact(estimator, artifact_path)
    write_json_atomic(Path(paths.metadata), metadata.model_dump(mode="json"))

    try:
        artifact_bytes = artifact_path.stat().st_size
    except OSError as exc:
        raise CausalEstimatorError(
            f"Failed to read estimator artifact size: {exc}",
            details={"rule": "artifact_stat_failed", "path": str(artifact_path)},
        ) from exc
    manifest = CausalEstimatorRunManifest(
        estimator_artifact_sha256=compute_file_sha256(artifact_path),
        estimator_artifact_bytes=artifact_bytes,
        fit_wall_seconds=fit_wall_seconds,
        pipeline_wall_seconds=time.perf_counter() - started,
        artifact_path=str(artifact_path),
    )
    write_json_atomic(Path(paths.run_manifest), manifest.model_dump(mode="json"))
    return metadata
