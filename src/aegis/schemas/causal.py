"""Typed contracts for causal estimator segment inference (P2-F06, ADR-013/ADR-014).

These models define the Stage 8 intermediate segment table and its lineage metadata.
They contain elasticity estimates and confidence intervals only; no proposed pricing
adjustment or rate delta can appear (INV-4, INV-5). The truth sidecar is not represented
here because Stage 8 never reads it.
"""

import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aegis.schemas.features import SHA256_PATTERN
from aegis.schemas.segments import SEGMENT_ID_PATTERN

SEGMENT_ESTIMATE_COLUMNS: tuple[str, ...] = (
    "segment_id",
    "inference_policy_count",
    "baseline_retention_q0",
    "tau_hat",
    "tau_ci_lower",
    "tau_ci_upper",
    "epsilon_hat",
    "epsilon_ci_lower",
    "epsilon_ci_upper",
)

_SHA256_RE = re.compile(SHA256_PATTERN)


class SegmentElasticityEstimate(BaseModel):
    """Segment-average treatment effect and elasticity with 95% confidence intervals."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(pattern=SEGMENT_ID_PATTERN)
    inference_policy_count: int = Field(gt=0)
    baseline_retention_q0: float = Field(gt=0.0, le=1.0)
    tau_hat: float
    tau_ci_lower: float
    tau_ci_upper: float
    epsilon_hat: float
    epsilon_ci_lower: float
    epsilon_ci_upper: float

    @model_validator(mode="after")
    def _validate_values(self) -> "SegmentElasticityEstimate":
        """Enforces finiteness, interval ordering, containment, and epsilon consistency."""
        values = (
            self.baseline_retention_q0,
            self.tau_hat,
            self.tau_ci_lower,
            self.tau_ci_upper,
            self.epsilon_hat,
            self.epsilon_ci_lower,
            self.epsilon_ci_upper,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("segment estimates and intervals must be finite")
        if self.tau_ci_lower > self.tau_ci_upper:
            raise ValueError("tau_ci_lower must not exceed tau_ci_upper")
        if not self.tau_ci_lower <= self.tau_hat <= self.tau_ci_upper:
            raise ValueError("tau interval must enclose the tau point estimate")
        if self.epsilon_ci_lower > self.epsilon_ci_upper:
            raise ValueError("epsilon_ci_lower must not exceed epsilon_ci_upper")
        if not self.epsilon_ci_lower <= self.epsilon_hat <= self.epsilon_ci_upper:
            raise ValueError("epsilon interval must enclose the elasticity point estimate")
        expected_epsilon = self.tau_hat / self.baseline_retention_q0
        if not math.isclose(
            self.epsilon_hat, expected_epsilon, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("epsilon_hat must equal tau_hat / baseline_retention_q0")
        return self


class SegmentElasticityTable(BaseModel):
    """The complete set of Stage 8 segment estimates for one model/data version."""

    model_config = ConfigDict(extra="forbid")

    segments: list[SegmentElasticityEstimate]

    @model_validator(mode="after")
    def _validate_table(self) -> "SegmentElasticityTable":
        """Rejects empty tables and duplicate segment identifiers."""
        if not self.segments:
            raise ValueError("segment elasticity table must not be empty")
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate segment_id in segment elasticity table")
        return self


class CausalEstimatorMetadata(BaseModel):
    """Deterministic lineage and configuration metadata for the Stage 8 estimator run."""

    model_config = ConfigDict(extra="forbid")

    estimator_name: Literal["CausalForestDML"] = "CausalForestDML"
    nuisance_model: Literal["gbt", "linear"]
    cv_folds: int = Field(gt=1)
    random_state: int
    n_estimators: int = Field(gt=0)
    min_samples_leaf: int = Field(gt=0)
    min_samples_split: int = Field(gt=1)
    max_depth: int | None = Field(default=None, gt=0)
    max_samples: float = Field(gt=0.0, le=1.0)
    subforest_size: int = Field(gt=0)
    training_sample_size: int = Field(gt=0)
    train_rows: int = Field(gt=0)
    test_rows: int = Field(gt=0)
    feature_count: int = Field(gt=0)
    segment_count: int = Field(gt=0)
    ci_alpha: float = Field(gt=0.0, lt=1.0)
    q0_floor: float = Field(gt=0.0, lt=1.0)
    q0_estimation_method: Literal["deconfounded_nuisance_fold_average"]
    interval_self_consistency_rate: float = Field(ge=0.0, le=1.0)
    finite_interval_rate: float = Field(ge=0.0, le=1.0)
    input_hashes: dict[str, str]
    config_fingerprint: str = Field(pattern=SHA256_PATTERN)
    model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_metadata(self) -> "CausalEstimatorMetadata":
        """Enforces split accounting, full interval validity, and hash formats."""
        if self.train_rows + self.test_rows != self.training_sample_size:
            raise ValueError(
                "train_rows + test_rows must equal training_sample_size"
            )
        if self.interval_self_consistency_rate != 1.0:
            raise ValueError("interval_self_consistency_rate must equal 1.0 at gate time")
        if self.finite_interval_rate != 1.0:
            raise ValueError("finite_interval_rate must equal 1.0 at gate time")
        if not self.input_hashes:
            raise ValueError("input_hashes must not be empty")
        for name, digest in self.input_hashes.items():
            if not name or not _SHA256_RE.fullmatch(digest):
                raise ValueError(
                    f"input hash for '{name}' must be a lowercase SHA-256 digest"
                )
        return self


class CausalEstimatorRunManifest(BaseModel):
    """Non-deterministic run-only diagnostics for the persisted estimator artifact."""

    model_config = ConfigDict(extra="forbid")

    estimator_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    estimator_artifact_bytes: int = Field(gt=0)
    fit_wall_seconds: float = Field(ge=0.0)
    pipeline_wall_seconds: float = Field(ge=0.0)
    artifact_path: str = Field(min_length=1)
