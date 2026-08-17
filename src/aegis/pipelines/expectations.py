"""Custom Great Expectations classes for AEGIS (ADR-006).

Implements domain-specific actuarial data contract checks, including post-treatment
feature leakage prevention for causal elasticity modeling.
"""

from typing import Any, ClassVar

from great_expectations.expectations.expectation import BatchExpectation


class ExpectNoPostTreatmentLeakage(BatchExpectation):
    """Custom expectation ensuring dataset contains no post-treatment leakage features.

    Causal pricing elasticity estimation requires strict pre-treatment features.
    Any feature observable only after treatment/pricing assignment (e.g. policyholder
    retention, cancellation post-renewal, post-loss settlements) introduces causal
    leakage and invalidates treatment effect identification.
    """

    metric_dependencies: ClassVar[tuple[str, ...]] = ("table.columns",)
    prohibited_columns: list[str]

    def _validate(
        self,
        metrics: dict[str, Any],
        runtime_configuration: dict[str, Any] | None = None,
        execution_engine: Any = None,
    ) -> dict[str, Any]:
        """Validates that no prohibited post-treatment columns exist in the table."""
        actual_columns: set[str] = set(metrics.get("table.columns", []))
        prohibited_set: set[str] = set(self.prohibited_columns)
        leaked_columns: set[str] = actual_columns.intersection(prohibited_set)

        success = len(leaked_columns) == 0
        return {
            "success": success,
            "result": {
                "observed_value": sorted(list(actual_columns)),
                "details": {
                    "prohibited_columns": sorted(list(prohibited_set)),
                    "leaked_columns": sorted(list(leaked_columns)),
                },
            },
        }
