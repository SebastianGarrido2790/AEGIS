"""Approved-artifact service for the Phase 2 diagnostic showcase (P2-F11).

The showcase reads only through :func:`aegis.pipelines.registry.load_approved_segments`,
the Stage 12 serving entry point. That path resolves the ``approved`` alias, downloads
the plain-data bundle, and refuses estimator binaries (ADR-014, INV-4).
"""

from __future__ import annotations

from functools import lru_cache

from aegis.pipelines.registry import load_approved_segments
from aegis.schemas.registry import ApprovedSegmentPackage


@lru_cache(maxsize=1)
def load_showcase_bundle() -> ApprovedSegmentPackage:
    """Loads the plain-data bundle behind the ``approved`` alias.

    Raises:
        ModelNotFoundError: If no version is assigned to the ``approved`` alias.
        RegistrySecurityError: If the artifact tree contains an estimator binary.
        RegistryPackagingError: If the packaged segment table is invalid.
    """
    return load_approved_segments()


def clear_model_cache() -> None:
    """Clears the approved-bundle cache for tests or a deliberate model refresh."""
    load_showcase_bundle.cache_clear()
