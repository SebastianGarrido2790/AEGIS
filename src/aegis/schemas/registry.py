"""Typed contracts for the Tier 1 MLflow registry and human-gated promotion (P2-F09).

The registry publishes a plain-data segment bundle only (ADR-014, P2-D11): no
serialized estimator ever enters the serving path. Promotion to the `approved` alias
is gated on an explicit human review record carrying a reviewer identity and the
SHA-256 of the calibration report that reviewer inspected (P2-D11a, INV-7).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aegis.schemas.features import SHA256_PATTERN
from aegis.schemas.segments import Tier1SegmentTable

GIT_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
DVC_HASH_PATTERN = r"^[0-9a-f]{32,64}$"
VALIDATION_STATUS = "GATE_PASSED_OPTION_F"


class ModelLineageMetadata(BaseModel):
    """Complete cryptographic provenance attached to every registered candidate."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    git_commit: str = Field(pattern=GIT_COMMIT_PATTERN)
    git_dirty: bool
    dvc_data_hash: str = Field(pattern=DVC_HASH_PATTERN)
    dgp_manifest_id: str = Field(pattern=SHA256_PATTERN)
    calibration_hash: str = Field(pattern=SHA256_PATTERN)
    validation_status: Literal["GATE_PASSED_OPTION_F"]
    segment_count: int = Field(ge=1)
    created_at_utc: str = Field(min_length=1)


class ModelPromotionRequest(BaseModel):
    """Human sign-off payload required to move `candidate` -> `approved`."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="aegis-causal-elasticity", min_length=1)
    target_version: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=3)
    calibration_report_hash: str = Field(min_length=64, max_length=64)
    review_notes: str = Field(min_length=10)


class PromotionAuditRecord(BaseModel):
    """Auditable outcome of a successful promotion or rollback."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    previous_approved_version: str | None = None
    new_approved_version: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=3)
    action: Literal["promote", "rollback"]
    calibration_verified: bool
    audit_tags: dict[str, str] = Field(default_factory=dict)
    timestamp_utc: str = Field(min_length=1)


class ApprovedSegmentPackage(BaseModel):
    """Plain-data serving payload resolved from the `approved` alias."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    segments: Tier1SegmentTable
    lineage: ModelLineageMetadata
    validation_summary: dict[str, Any] = Field(default_factory=dict)
