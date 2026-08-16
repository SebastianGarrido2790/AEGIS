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


class GroundingThresholdError(AEGISError):
    """Raised when compliance retrieval grounding falls below threshold (INV-6)."""


class FallbackTriggeredError(AEGISError):
    """Raised when governance fallback to last approved rate table occurs."""


class ModuleSizeExceededError(AEGISError):
    """Raised when a module exceeds the 1,000 line ceiling (INV-8)."""
