"""Atomic persistence and lineage helpers for Stage 8 causal estimator artifacts.

This module owns filesystem effects: hashing inputs, building deterministic version and
configuration labels, and writing the segment table, joblib model, metadata, and run
manifest. It cannot fit models or estimate effects.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from aegis.config.schema import EstimatorConfig, ValidationThresholdsConfig
from aegis.schemas.causal import SEGMENT_ESTIMATE_COLUMNS, SegmentElasticityTable
from aegis.schemas.validation import ValidationMetricReport
from aegis.utils.exceptions import CausalEstimatorError

CAUSAL_VALIDATION_REPORT_VERSION = "1.0"


def compute_file_sha256(path: Path) -> str:
    """Computes a streaming SHA-256 digest for a file.

    Args:
        path: File to hash.

    Returns:
        A lowercase hexadecimal SHA-256 digest.

    Raises:
        CausalEstimatorError: If the file is missing or cannot be read.
    """
    if not path.is_file():
        raise CausalEstimatorError(
            f"Required input file not found: {path}",
            details={"rule": "missing_input_file", "path": str(path)},
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CausalEstimatorError(
            f"Failed to hash file {path}: {exc}",
            details={"rule": "hash_failed", "path": str(path)},
        ) from exc
    return digest.hexdigest()


def build_model_version(estimator_cfg: EstimatorConfig) -> str:
    """Builds a deterministic human-readable model version label.

    Args:
        estimator_cfg: Frozen estimator configuration.

    Returns:
        A version label derived only from behavioral estimator settings.
    """
    depth = "none" if estimator_cfg.max_depth is None else str(estimator_cfg.max_depth)
    return (
        f"cfdml-{estimator_cfg.nuisance_model}"
        f"-cv{estimator_cfg.cv_folds}"
        f"-n{estimator_cfg.n_estimators}"
        f"-leaf{estimator_cfg.min_samples_leaf}"
        f"-split{estimator_cfg.min_samples_split}"
        f"-depth{depth}"
        f"-ms{estimator_cfg.max_samples}"
        f"-sub{estimator_cfg.subforest_size}"
    )


def compute_config_fingerprint(
    estimator_cfg: EstimatorConfig,
    *,
    random_state: int,
    test_size: float,
    corridor_width: float,
) -> str:
    """Computes a canonical fingerprint of behavioral training configuration.

    Non-behavioral settings such as worker count and runtime budget are excluded so
    the fingerprint changes only when model behavior or data handling changes.

    Args:
        estimator_cfg: Frozen estimator configuration.
        random_state: Seed shared by the nuisance and forest models.
        test_size: Held-out fraction.
        corridor_width: Maximum absolute treatment value.

    Returns:
        A lowercase hexadecimal SHA-256 digest.
    """
    payload = {
        "nuisance_model": estimator_cfg.nuisance_model,
        "cv_folds": estimator_cfg.cv_folds,
        "n_estimators": estimator_cfg.n_estimators,
        "min_samples_leaf": estimator_cfg.min_samples_leaf,
        "min_samples_split": estimator_cfg.min_samples_split,
        "max_depth": estimator_cfg.max_depth,
        "max_samples": estimator_cfg.max_samples,
        "subforest_size": estimator_cfg.subforest_size,
        "ci_alpha": estimator_cfg.ci_alpha,
        "q0_floor": estimator_cfg.q0_floor,
        "training_sample_size": estimator_cfg.training_sample_size,
        "random_state": random_state,
        "test_size": test_size,
        "corridor_width": corridor_width,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _temporary_path(target: Path) -> Path:
    """Creates a unique temporary sibling path for atomic replacement."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    return Path(name)


def _replace_atomically(temporary: Path, target: Path, *, rule: str) -> None:
    """Replaces a target with a completed temporary file, cleaning up on failure."""
    try:
        temporary.replace(target)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise CausalEstimatorError(
            f"Failed to write {target}: {exc}",
            details={"rule": rule, "path": str(target)},
        ) from exc


def write_segment_table(table: SegmentElasticityTable, path: Path) -> None:
    """Atomically writes the segment elasticity table as CSV.

    Args:
        table: Validated segment elasticity table.
        path: Destination CSV path.

    Raises:
        CausalEstimatorError: If serialization or file replacement fails.
    """
    frame = pd.DataFrame(
        [segment.model_dump() for segment in table.segments],
        columns=pd.Index(SEGMENT_ESTIMATE_COLUMNS),
    )
    temporary = _temporary_path(path)
    try:
        frame.to_csv(temporary, index=False)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise CausalEstimatorError(
            f"Failed to serialize segment table: {exc}",
            details={"rule": "segment_table_write_failed", "path": str(path)},
        ) from exc
    _replace_atomically(temporary, path, rule="segment_table_replace_failed")


def write_estimator_artifact(estimator: Any, path: Path) -> None:
    """Atomically persists the fitted estimator as a non-promotable joblib artifact.

    Args:
        estimator: Fitted CausalForestDML instance.
        path: Destination joblib path.

    Raises:
        CausalEstimatorError: If serialization or file replacement fails.
    """
    temporary = _temporary_path(path)
    try:
        joblib.dump(estimator, temporary)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise CausalEstimatorError(
            f"Failed to persist estimator artifact: {exc}",
            details={"rule": "estimator_artifact_write_failed", "path": str(path)},
        ) from exc
    _replace_atomically(temporary, path, rule="estimator_artifact_replace_failed")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically writes a JSON document with stable formatting.

    Args:
        path: Destination JSON path.
        payload: JSON-serializable dictionary.

    Raises:
        CausalEstimatorError: If serialization or file replacement fails.
    """
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise CausalEstimatorError(
            f"Failed to serialize JSON artifact: {exc}",
            details={"rule": "json_write_failed", "path": str(path)},
        ) from exc
    _replace_atomically(temporary, path, rule="json_replace_failed")


def _threshold_checks(
    metrics: Any, thresholds: ValidationThresholdsConfig
) -> dict[str, dict[str, Any]]:
    """Compares each frozen Stage 9 metric against its floor or ceiling.

    Args:
        metrics: Validated Stage 9 metric record.
        thresholds: Frozen validation thresholds from configuration.

    Returns:
        A mapping of metric name to value, threshold, direction, and pass verdict.
    """
    floors = {
        "m1_tau_spearman": (metrics.m1_tau_spearman, thresholds.m1_tau_spearman_floor),
        "m1_epsilon_spearman": (
            metrics.m1_epsilon_spearman,
            thresholds.m1_epsilon_spearman_floor,
        ),
        "m1b_policy_spearman": (
            metrics.m1b_policy_spearman,
            thresholds.m1b_policy_spearman_floor,
        ),
        "m3_tau_coverage": (metrics.m3_tau_coverage, thresholds.m3_tau_coverage_floor),
        "m3_epsilon_coverage": (
            metrics.m3_epsilon_coverage,
            thresholds.m3_epsilon_coverage_floor,
        ),
        "m4_self_consistency": (
            metrics.m4_self_consistency,
            thresholds.m4_self_consistency_floor,
        ),
        "m5_sign_agreement": (
            metrics.m5_sign_agreement,
            thresholds.m5_sign_agreement_floor,
        ),
    }
    checks: dict[str, dict[str, Any]] = {
        name: {
            "value": value,
            "threshold": threshold,
            "direction": "floor",
            "passes": bool(value >= threshold),
        }
        for name, (value, threshold) in floors.items()
    }
    checks["m2_mean_abs_bias"] = {
        "value": metrics.m2_mean_abs_bias,
        "threshold": thresholds.m2_mean_abs_bias_ceiling,
        "direction": "ceiling",
        "passes": bool(metrics.m2_mean_abs_bias <= thresholds.m2_mean_abs_bias_ceiling),
    }
    return dict(sorted(checks.items()))


def build_causal_validation_report(
    report: ValidationMetricReport,
    thresholds: ValidationThresholdsConfig | None,
) -> dict[str, Any]:
    """Builds the deterministic Stage 9 recovery-verification artifact payload.

    The payload carries no timestamp or wall-clock measurement so that two runs over
    identical inputs produce byte-identical JSON. Whether the estimator itself is
    bit-reproducible is reported by comparing consecutive runs, not hidden here.

    Args:
        report: Single-run validation report from the Stage 9 harness.
        thresholds: Frozen thresholds, or ``None`` when the campaign has not frozen them.

    Returns:
        A JSON-serializable dictionary with metrics and per-metric gate verdicts.
    """
    checks = None if thresholds is None else _threshold_checks(report.metrics, thresholds)
    return {
        "report_version": CAUSAL_VALIDATION_REPORT_VERSION,
        "sample_size": report.sample_size,
        "seed": report.seed,
        "segment_count": len(report.segment_details),
        "thresholds_configured": thresholds is not None,
        "metrics": report.metrics.model_dump(mode="json"),
        "gate_checks": checks,
        "gate_passed": None if checks is None else all(c["passes"] for c in checks.values()),
    }


def write_causal_validation_report(path: Path, payload: dict[str, Any]) -> None:
    """Writes the Stage 9 recovery-verification artifact with stable formatting.

    Args:
        path: Destination JSON path (``data/validated/causal_validation_report.json``).
        payload: Output of :func:`build_causal_validation_report`.
    """
    write_json_atomic(path, payload)
