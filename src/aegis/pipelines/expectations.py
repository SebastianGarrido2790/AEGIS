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


class ExpectZeroOrphanSeverityRecords(BatchExpectation):
    """Custom expectation enforcing referential integrity between severity and frequency data.

    Validates that every claim severity record maps to an existing observational
    frequency policy record. Orphan claims lacking policyholder and vehicle risk
    attributes introduce severe bias into pure premium estimation.
    """

    metric_dependencies: ClassVar[tuple[str, ...]] = ("table.columns",)
    orphan_column: str = "is_orphan_severity"

    def _validate(
        self,
        metrics: dict[str, Any],
        runtime_configuration: dict[str, Any] | None = None,
        execution_engine: Any = None,
    ) -> dict[str, Any]:
        """Validates that the batch contains zero orphan severity records."""
        df = getattr(execution_engine, "dataframe", None)
        if df is None:
            return {"success": True, "result": {"details": "No in-memory dataframe available"}}

        if self.orphan_column not in df.columns:
            return {
                "success": True,
                "result": {"details": f"Orphan flag '{self.orphan_column}' not present"},
            }

        orphans = df[df[self.orphan_column] == True]  # noqa: E712
        orphan_count = int(len(orphans))
        success = orphan_count == 0

        orphan_policy_ids = (
            orphans["PolicyID"].head(10).tolist() if "PolicyID" in df.columns else []
        )

        return {
            "success": success,
            "result": {
                "orphan_count": orphan_count,
                "orphan_policy_ids": orphan_policy_ids,
                "details": {
                    "violation": (
                        "Orphan severity records detected without matching frequency record"
                    ),
                    "orphan_count": orphan_count,
                },
            },
        }
