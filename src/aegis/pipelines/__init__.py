"""AEGIS pipelines package."""

from aegis.pipelines.data_contracts import (
    ContractValidationResult,
    load_suite_from_json,
    validate_dataframe,
    validate_regulatory_corpus,
)

__all__ = [
    "ContractValidationResult",
    "load_suite_from_json",
    "validate_dataframe",
    "validate_regulatory_corpus",
]
