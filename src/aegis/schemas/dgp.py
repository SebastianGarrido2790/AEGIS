"""Typed contracts for the Semi-Synthetic DGP and Truth Sidecar (P2-F04, ADR-013).

These models define the persisted DGP manifest and the isolated truth sidecar
interface consumed only by validation harnesses (P2-D2).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aegis.schemas.features import SHA256_PATTERN


class SelfTestResult(BaseModel):
    """Result of an automated DGP self-test (ADR-013, P2-D2a/b)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    observed_value: float
    threshold: float
    details: dict[str, Any] = Field(default_factory=dict)


class DGPManifest(BaseModel):
    """Versioned metadata artifact emitted by the semi-synthetic DGP generator."""

    model_config = ConfigDict(extra="forbid")

    random_seed: int
    sample_size: int = Field(ge=1)
    corridor_width: float = Field(gt=0.0, lt=1.0)
    confounding_dial: float = Field(ge=0.0, le=1.0)
    residual_std: float = Field(gt=0.0)
    self_tests: dict[str, SelfTestResult]
    input_hashes: dict[str, str]
    config_fingerprint: str = Field(pattern=SHA256_PATTERN)


class TruthSidecarRow(BaseModel):
    """One policy record from the isolated truth sidecar table (P2-D2)."""

    model_config = ConfigDict(extra="forbid")

    PolicyID: int
    baseline_renewal_q0: float = Field(ge=0.0, le=1.0)
    true_tau: float
    true_epsilon: float
    segment_id: str
