"""Data contract validation runners using Great Expectations Core (INV-3, ADR-003, ADR-006).

Provides functions to validate datasets against versioned JSON expectation suites.
Fails loudly and raises DataContractError when contract assertions are violated.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd

from aegis.config.loader import get_config
from aegis.utils.exceptions import DataContractError


@dataclass(frozen=True)
class ContractValidationResult:
    """Structured result of a Great Expectations data contract validation."""

    success: bool
    suite_name: str
    total_expectations: int
    successful_expectations: int
    failed_expectations: list[dict[str, Any]] = field(default_factory=list)
    failure_rules: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Returns a human-readable summary of the validation outcome."""
        status = "PASSED" if self.success else "FAILED"
        lines = [
            f"Data Contract [{self.suite_name}]: {status}",
            f"Passed: {self.successful_expectations}/{self.total_expectations}",
        ]
        if self.failure_rules:
            lines.append("Violated Rules:")
            for rule in self.failure_rules:
                lines.append(f"  - {rule}")
        return "\n".join(lines)


def load_suite_from_json(
    suite_path: Path | str,
    context: Any,
) -> gx.ExpectationSuite:
    """Loads an ExpectationSuite from a native GX JSON file into the context.

    Args:
        suite_path: Path to the JSON suite file.
        context: GX DataContext.

    Returns:
        gx.ExpectationSuite: Loaded and registered expectation suite.

    Raises:
        DataContractError: If the suite file cannot be found or parsed.
    """
    path = Path(suite_path)
    if not path.is_file():
        raise DataContractError(
            f"Expectation suite file not found: {path}",
            details={"suite_path": str(path)},
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            suite_dict = json.load(f)
    except Exception as exc:
        raise DataContractError(
            f"Failed to read expectation suite JSON: {exc}",
            details={"suite_path": str(path), "error": str(exc)},
        ) from exc

    suite_name = suite_dict.get("name", path.stem)
    try:
        # Check if already present in context
        try:
            return context.suites.get(suite_name)
        except Exception:
            return context.suites.add(gx.ExpectationSuite(**suite_dict))
    except Exception as exc:
        raise DataContractError(
            f"Failed to load suite into Great Expectations context: {exc}",
            details={"suite_path": str(path), "error": str(exc)},
        ) from exc


def validate_dataframe(
    df: pd.DataFrame,
    suite_path: Path | str,
    raise_on_failure: bool = False,
) -> ContractValidationResult:
    """Validates an in-memory DataFrame against a JSON expectation suite.

    Args:
        df: Pandas DataFrame containing dataset to validate.
        suite_path: Path to the GX JSON expectation suite.
        raise_on_failure: If True, raises DataContractError on failure (INV-3).

    Returns:
        ContractValidationResult: Structured validation outcome.

    Raises:
        DataContractError: If raise_on_failure is True and any expectation fails.
    """
    context = gx.get_context(mode="ephemeral")
    suite = load_suite_from_json(suite_path, context)

    # Register in-memory dataframe batch
    ds_name = f"ds_{suite.name}"
    try:
        data_source = context.data_sources.get(ds_name)
    except Exception:
        data_source = context.data_sources.add_pandas(ds_name)

    asset_name = f"asset_{suite.name}"
    try:
        asset = data_source.get_asset(asset_name)
    except Exception:
        asset = data_source.add_dataframe_asset(asset_name)

    batch_def_name = f"batch_{suite.name}"
    try:
        batch_def = asset.get_batch_definition(batch_def_name)
    except Exception:
        batch_def = asset.add_batch_definition_whole_dataframe(batch_def_name)

    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    gx_results = batch.validate(suite)

    failed_expectations: list[dict[str, Any]] = []
    failure_rules: list[str] = []
    successful_count = 0

    for res in gx_results.results:
        exp_type = getattr(res.expectation_config, "type", str(res.expectation_config))
        kwargs = getattr(res.expectation_config, "kwargs", {})
        column = kwargs.get("column", "table-level")

        if res.success:
            successful_count += 1
        else:
            rule_desc = f"{exp_type} on column '{column}'"
            failure_rules.append(rule_desc)
            failed_expectations.append(
                {
                    "expectation_type": exp_type,
                    "column": column,
                    "kwargs": kwargs,
                    "result": res.result,
                }
            )

    result = ContractValidationResult(
        success=bool(gx_results.success),
        suite_name=suite.name,
        total_expectations=len(gx_results.results),
        successful_expectations=successful_count,
        failed_expectations=failed_expectations,
        failure_rules=failure_rules,
    )

    if not result.success and raise_on_failure:
        raise DataContractError(
            f"Data contract validation failed for suite '{suite.name}'. "
            f"Violated {len(failure_rules)} rule(s): {', '.join(failure_rules)}",
            details={
                "suite_name": suite.name,
                "suite_path": str(suite_path),
                "failure_rules": failure_rules,
                "failed_expectations": failed_expectations,
            },
        )

    return result


def validate_regulatory_corpus(
    data_path: Path | str | None = None,
    suite_path: Path | str | None = None,
    raise_on_failure: bool = False,
) -> ContractValidationResult:
    """Validates regulatory corpus JSON data against the regulatory expectation suite.

    Args:
        data_path: Path to regulatory JSON data file. Defaults to params.yaml valid fixture.
        suite_path: Path to regulatory JSON suite. Defaults to params.yaml suite path.
        raise_on_failure: If True, raises DataContractError if contract is violated.

    Returns:
        ContractValidationResult: Structured validation outcome.
    """
    config = get_config()
    target_data_path = Path(data_path or config.data_contracts.regulatory_valid_fixture)
    target_suite_path = Path(suite_path or config.data_contracts.regulatory_corpus_suite_path)

    if not target_data_path.is_file():
        raise DataContractError(
            f"Regulatory data file not found: {target_data_path}",
            details={"data_path": str(target_data_path)},
        )

    try:
        with target_data_path.open("r", encoding="utf-8") as f:
            raw_json = json.load(f)
        df = pd.DataFrame(raw_json)
    except Exception as exc:
        raise DataContractError(
            f"Failed to load regulatory data into DataFrame: {exc}",
            details={"data_path": str(target_data_path), "error": str(exc)},
        ) from exc

    return validate_dataframe(df, target_suite_path, raise_on_failure=raise_on_failure)
