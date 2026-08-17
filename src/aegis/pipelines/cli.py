"""CLI entrypoints for DVC pipeline execution (INV-3, ADR-007).

Implements the fine-grained `ingest -> validate_gx -> version` pipeline stages for
both elasticity training data and the regulatory corpus.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from aegis.pipelines.data_contracts import (
    validate_elasticity_data,
    validate_regulatory_corpus,
)
from aegis.utils.exceptions import DataContractError


def ingest_file(input_path: Path, output_path: Path) -> int:
    """Ingests a source data file into raw storage."""
    if not input_path.is_file():
        print(f"[ERROR] Source file not found: {input_path}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    print(f"[INGEST] Ingested {input_path} -> {output_path}")
    return 0


def validate_file(
    data_type: str,
    data_path: Path,
    suite_path: Path,
    report_path: Path,
) -> int:
    """Executes Great Expectations validation and writes structured report."""
    print(f"[VALIDATE] Validating {data_type} data at {data_path} against {suite_path}")

    if data_type == "elasticity":
        result = validate_elasticity_data(
            data_path=data_path,
            suite_path=suite_path,
            raise_on_failure=False,
        )
    elif data_type == "regulatory":
        result = validate_regulatory_corpus(
            data_path=data_path,
            suite_path=suite_path,
            raise_on_failure=False,
        )
    else:
        print(f"[ERROR] Unknown data type: {data_type}")
        return 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "success": result.success,
        "suite_name": result.suite_name,
        "total_expectations": result.total_expectations,
        "successful_expectations": result.successful_expectations,
        "failure_rules": result.failure_rules,
        "failed_expectations": result.failed_expectations,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(result.summary())

    if not result.success:
        print("[BLOCK] Data contract validation failed. Halting pipeline (INV-3).")
        return 1

    print(f"[SUCCESS] Validation passed. Report written to {report_path}")
    return 0


def version_file(
    raw_path: Path,
    report_path: Path,
    output_path: Path,
) -> int:
    """Promotes validated data to versioned storage if validation passed."""
    if not report_path.is_file():
        print(f"[ERROR] Validation report not found at {report_path}. Stage blocked.")
        return 1

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    if not report.get("success", False):
        raise DataContractError(
            "Cannot version unvalidated or failed data (INV-3).",
            details={"report": report},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, output_path)
    print(f"[VERSION] Promoted validated dataset: {raw_path} -> {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI parser for DVC pipeline stages."""
    parser = argparse.ArgumentParser(description="AEGIS DVC Pipeline Stages")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest commands
    for stage, default_in, default_out in [
        (
            "ingest-elasticity",
            "data_contracts/fixtures/elasticity_valid.csv",
            "data/raw/elasticity_raw.csv",
        ),
        (
            "ingest-regulatory",
            "data_contracts/fixtures/regulatory_valid.json",
            "data/raw/regulatory_raw.json",
        ),
    ]:
        p = subparsers.add_parser(stage)
        p.add_argument("--input-path", type=Path, default=Path(default_in))
        p.add_argument("--output-path", type=Path, default=Path(default_out))

    # Validate commands
    for stage, _dtype, default_data, default_suite, default_rep in [
        (
            "validate-elasticity",
            "elasticity",
            "data/raw/elasticity_raw.csv",
            "data_contracts/elasticity_training_suite.json",
            "data/validated/elasticity_validation_report.json",
        ),
        (
            "validate-regulatory",
            "regulatory",
            "data/raw/regulatory_raw.json",
            "data_contracts/regulatory_corpus_suite.json",
            "data/validated/regulatory_validation_report.json",
        ),
    ]:
        p = subparsers.add_parser(stage)
        p.add_argument("--data-path", type=Path, default=Path(default_data))
        p.add_argument("--suite-path", type=Path, default=Path(default_suite))
        p.add_argument("--report-path", type=Path, default=Path(default_rep))

    # Version commands
    for stage, default_raw, default_rep, default_out in [
        (
            "version-elasticity",
            "data/raw/elasticity_raw.csv",
            "data/validated/elasticity_validation_report.json",
            "data/versioned/elasticity_training_data.csv",
        ),
        (
            "version-regulatory",
            "data/raw/regulatory_raw.json",
            "data/validated/regulatory_validation_report.json",
            "data/versioned/regulatory_corpus.json",
        ),
    ]:
        p = subparsers.add_parser(stage)
        p.add_argument("--raw-path", type=Path, default=Path(default_raw))
        p.add_argument("--report-path", type=Path, default=Path(default_rep))
        p.add_argument("--output-path", type=Path, default=Path(default_out))

    return parser


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest-elasticity":
        return ingest_file(args.input_path, args.output_path)
    if args.command == "ingest-regulatory":
        return ingest_file(args.input_path, args.output_path)
    if args.command == "validate-elasticity":
        return validate_file("elasticity", args.data_path, args.suite_path, args.report_path)
    if args.command == "validate-regulatory":
        return validate_file("regulatory", args.data_path, args.suite_path, args.report_path)
    if args.command == "version-elasticity":
        return version_file(args.raw_path, args.report_path, args.output_path)
    if args.command == "version-regulatory":
        return version_file(args.raw_path, args.report_path, args.output_path)

    return 1


if __name__ == "__main__":
    sys.exit(main())
