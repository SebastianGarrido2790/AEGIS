"""Typed contract for the Phase 2 artifact version manifest (P2-F10, ADR-015).

The manifest is the reproducibility evidence artifact: one SHA-256 per declared data
product plus a combined digest. Two runs over identical inputs must produce an
identical manifest, so the payload deliberately carries no timestamp, git revision, or
wall-clock measurement.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from aegis.schemas.features import SHA256_PATTERN

MANIFEST_VERSION = "1.0"


class ArtifactDigest(BaseModel):
    """Content digest for one declared pipeline product."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="Repository-relative artifact path")
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0)


class VersionManifest(BaseModel):
    """Ordered digest set for every canonical Phase 2 data product."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: str = Field(default=MANIFEST_VERSION, min_length=1)
    combined_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact_count: int = Field(gt=0)
    artifacts: list[ArtifactDigest] = Field(min_length=1)
