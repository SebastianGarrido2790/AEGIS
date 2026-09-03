"""CLI entrypoints for DVC pipeline execution (INV-3, ADR-007, R-4).

Implements the fine-grained `ingest -> validate_gx -> version` pipeline stages for
both elasticity training data and the regulatory corpus.
Defaults are sourced dynamically from params.yaml via get_config() (R-4).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from aegis.config.loader import get_config
from aegis.pipelines.data_contracts import (
    validate_elasticity_data,
    validate_regulatory_corpus,
)
from aegis.pipelines.feature.pipeline import build_feature_matrix
from aegis.pipelines.training.glm_baseline import (
    fit_tweedie_baseline,
    save_baseline_artifact,
)
from aegis.utils.exceptions import DataContractError


def ingest_file(input_path: Path | None, output_path: Path, data_type: str) -> int:
    """Ingests a source data file into raw storage.

    Defaults input_path to the valid fixture configured in params.yaml if omitted (R-4).
    """
    config = get_config()
    if input_path is None:
        if data_type == "elasticity":
            input_path = Path(config.data_contracts.elasticity_valid_fixture)
        elif data_type == "regulatory":
            input_path = Path(config.data_contracts.regulatory_valid_fixture)
        else:
            print(f"[ERROR] Unknown data type: {data_type}")
            return 1

    if not input_path.is_file():
        print(f"[ERROR] Source file not found: {input_path}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    print(f"[INGEST] Ingested {input_path} -> {output_path}")
    return 0


def validate_file(
    data_type: str,
    data_path: Path | None,
    suite_path: Path | None,
    report_path: Path,
) -> int:
    """Executes Great Expectations validation and writes structured report."""
    print(f"[VALIDATE] Validating {data_type} data...")

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
    """Constructs the CLI parser for DVC pipeline stages with params.yaml backing (R-4)."""
    parser = argparse.ArgumentParser(description="AEGIS DVC Pipeline Stages")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest commands
    p_ing_f = subparsers.add_parser("ingest-fremtpl2")
    p_ing_f.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/raw/elasticity_fremtpl2.csv"),
        help="Target CSV path for ingested freMTPL2 dataset",
    )
    p_ing_f.add_argument(
        "--force-download",
        action="store_true",
        default=False,
        help="Force download from OpenML even if output file exists",
    )

    p_ing_e = subparsers.add_parser("ingest-elasticity")
    p_ing_e.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Source path (defaults to elasticity_valid_fixture in params.yaml)",
    )
    p_ing_e.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/raw/elasticity_raw.csv"),
    )

    p_ing_r = subparsers.add_parser("ingest-regulatory")
    p_ing_r.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Source path (defaults to regulatory_valid_fixture in params.yaml)",
    )
    p_ing_r.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/raw/regulatory_raw.json"),
    )

    # Validate commands
    p_val_e = subparsers.add_parser("validate-elasticity")
    p_val_e.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/elasticity_raw.csv"),
    )
    p_val_e.add_argument(
        "--suite-path",
        type=Path,
        default=None,
        help="Suite path (defaults to elasticity_suite_path in params.yaml)",
    )
    p_val_e.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/validated/elasticity_validation_report.json"),
    )

    p_val_r = subparsers.add_parser("validate-regulatory")
    p_val_r.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/regulatory_raw.json"),
    )
    p_val_r.add_argument(
        "--suite-path",
        type=Path,
        default=None,
        help="Suite path (defaults to regulatory_corpus_suite_path in params.yaml)",
    )
    p_val_r.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/validated/regulatory_validation_report.json"),
    )

    # Version commands
    p_ver_e = subparsers.add_parser("version-elasticity")
    p_ver_e.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/elasticity_raw.csv"),
    )
    p_ver_e.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/validated/elasticity_validation_report.json"),
    )
    p_ver_e.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/versioned/elasticity_training_data.csv"),
    )

    p_ver_r = subparsers.add_parser("version-regulatory")
    p_ver_r.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/regulatory_raw.json"),
    )
    p_ver_r.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/validated/regulatory_validation_report.json"),
    )
    p_ver_r.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/versioned/regulatory_corpus.json"),
    )

    p_feature = subparsers.add_parser("build-features")
    p_feature.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/versioned/elasticity_fremtpl2.csv"),
        help="Validated source dataset for feature construction",
    )
    p_feature.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/versioned/feature_matrix.csv"),
        help="Target path for the engineered feature matrix",
    )

    p_glm = subparsers.add_parser("train-glm")
    p_glm.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/versioned/feature_matrix.csv"),
        help="Stage 2 feature matrix used for GLM training",
    )
    p_glm.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/validated/glm_baseline.json"),
        help="Target JSON path for baseline metrics and intervals",
    )

    return parser


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest-fremtpl2":
        from aegis.pipelines.feature.ingest import ingest_fremtpl2_pipeline

        ingest_fremtpl2_pipeline(
            output_path=args.output_path,
            force_download=args.force_download,
        )
        return 0
    if args.command == "ingest-elasticity":
        return ingest_file(args.input_path, args.output_path, "elasticity")
    if args.command == "ingest-regulatory":
        return ingest_file(args.input_path, args.output_path, "regulatory")
    if args.command == "validate-elasticity":
        return validate_file("elasticity", args.data_path, args.suite_path, args.report_path)
    if args.command == "validate-regulatory":
        return validate_file("regulatory", args.data_path, args.suite_path, args.report_path)
    if args.command == "version-elasticity":
        return version_file(args.raw_path, args.report_path, args.output_path)
    if args.command == "version-regulatory":
        return version_file(args.raw_path, args.report_path, args.output_path)
    if args.command == "build-features":
        dataset = pd.read_csv(args.input_path)
        feature_matrix = build_feature_matrix(dataset)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_matrix.to_csv(args.output_path, index=False)
        print(f"[FEATURE] Wrote feature matrix to {args.output_path}")
        return 0
    if args.command == "train-glm":
        dataset = pd.read_csv(args.input_path)
        result = fit_tweedie_baseline(dataset)
        save_baseline_artifact(result, args.output_path)
        print(f"[GLM] Wrote baseline artifact to {args.output_path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
