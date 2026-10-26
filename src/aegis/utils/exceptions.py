"""AEGIS custom exception hierarchy.

All domain, configuration, pipeline, and governance exceptions inherit from
AEGISError to ensure consistent error handling, typed failure modes, and
unambiguous stack traces across the platform.
"""

from typing import Any


class AEGISError(Exception):
    """Base exception for all AEGIS errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


class ConfigurationError(AEGISError):
    """Raised when application configuration is missing, malformed, or invalid."""


class DataContractError(AEGISError):
    """Raised when data contracts or expectation suites fail validation (INV-3)."""


class FeaturePipelineError(AEGISError):
    """Raised when the deterministic feature pipeline cannot produce valid outputs (P2-F02)."""


class GLMBaselineError(AEGISError):
    """Raised when the Tweedie baseline cannot fit or calibrate valid outputs (P2-F03)."""


class SegmentError(AEGISError):
    """Raised when segment assignment or Tier 1 aggregation cannot produce valid outputs."""


class GroundingThresholdError(AEGISError):
    """Raised when compliance retrieval grounding falls below threshold (INV-6)."""


class FallbackTriggeredError(AEGISError):
    """Raised when governance fallback to last approved rate table occurs."""


class ModuleSizeExceededError(AEGISError):
    """Raised when a module exceeds the 1,000 line ceiling (INV-8)."""


class DGPVerificationError(AEGISError):
    """Raised when semi-synthetic DGP self-tests or generation constraints fail (P2-F04)."""


class TruthIsolationError(AEGISError):
    """Raised when truth sidecar access or truth column isolation boundaries are violated."""


