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

from aegis.config.loader import get_config
from aegis.pipelines.data_contracts import (
    validate_elasticity_data,
    validate_raw_policy_data,
    validate_regulatory_corpus,
)
from aegis.pipelines.feature_pipeline import run_feature_pipeline
from aegis.pipelines.glm_baseline import run_glm_baseline
from aegis.pipelines.raw_intake import ingest_raw_data
from aegis.pipelines.segments import run_segment_pipeline
from aegis.utils.exceptions import (
    DataContractError,
    FeaturePipelineError,
    GLMBaselineError,
    SegmentError,
)


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


def ingest_raw_policy(
    freq_path: Path | None,
    sev_path: Path | None,
    manifest_path: Path | None,
    output_path: Path,
) -> int:
    """Ingests and validates raw observational data files into joined policy table."""
    try:
        ingest_raw_data(
            freq_path=freq_path,
            sev_path=sev_path,
            manifest_path=manifest_path,
            output_path=output_path,
        )
        return 0
    except DataContractError as exc:
        print(f"[BLOCK] Raw intake integrity check failed: {exc}")
        return 1
    except Exception as exc:
        print(f"[ERROR] Failed during raw ingestion: {exc}")
        return 1


def build_feature_artifacts(
    input_path: Path | None,
    glm_output: Path | None,
    forest_output: Path | None,
    metadata_output: Path | None,
) -> int:
    """Builds the deterministic GLM and forest feature views."""
    try:
        run_feature_pipeline(
            input_path=input_path,
            glm_path=glm_output,
            forest_path=forest_output,
            metadata_path=metadata_output,
        )
        return 0
    except FeaturePipelineError as exc:
        print(f"[BLOCK] Feature pipeline failed: {exc}")
        return 1


def build_glm_baseline(
    input_path: Path | None,
    cap: float | None,
    fitted_output: Path | None,
    metadata_output: Path | None,
    report_output: Path | None,
) -> int:
    """Fits the cap-parameterized Tweedie baseline and writes calibration artifacts."""
    try:
        run_glm_baseline(
            input_path=input_path,
            cap=cap,
            fitted_path=fitted_output,
            metadata_path=metadata_output,
            report_path=report_output,
        )
        return 0
    except GLMBaselineError as exc:
        print(f"[BLOCK] GLM baseline failed: {exc}")
        return 1


def build_segments(
    features_path: Path | None,
    fitted_path: Path | None,
    loss_cost_output: Path | None,
    definitions_output: Path | None,
    policy_segments_output: Path | None,
) -> int:
    """Builds the Tier 1 segment table, metadata, and policy mapping."""
    try:
        run_segment_pipeline(
            features_path=features_path,
            fitted_path=fitted_path,
            loss_cost_path=loss_cost_output,
            definitions_path=definitions_output,
            policy_segments_path=policy_segments_output,
        )
        return 0
    except SegmentError as exc:
        print(f"[BLOCK] Segment pipeline failed: {exc}")
        return 1


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
    elif data_type in ("raw", "raw_policy"):
        result = validate_raw_policy_data(
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
        json.dump(report_data, f, indent=2, default=str)

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

    if not raw_path.is_file():
        print(f"[ERROR] Raw data file not found at {raw_path}. Stage blocked.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, output_path)
    print(f"[VERSION] Promoted validated dataset: {raw_path} -> {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI parser for DVC pipeline stages with params.yaml backing (R-4)."""
    parser = argparse.ArgumentParser(description="AEGIS DVC Pipeline Stages")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest commands
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

    p_ing_raw = subparsers.add_parser("ingest-raw")
    p_ing_raw.add_argument(
        "--freq-path",
        type=Path,
        default=Path("data/raw/freMTPLfreq.csv.zip"),
        help="Path to frequency dataset (default: data/raw/freMTPLfreq.csv.zip)",
    )
    p_ing_raw.add_argument(
        "--sev-path",
        type=Path,
        default=Path("data/raw/freMTPLsev.csv"),
        help="Path to severity dataset (default: data/raw/freMTPLsev.csv)",
    )
    p_ing_raw.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Path to checksum manifest (defaults to params.yaml)",
    )
    p_ing_raw.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/raw/raw_policy_data.csv"),
        help="Destination path for joined raw policy table",
    )

    # Validate commands
    p_val_e = subparsers.add_parser("validate-elasticity")
    p_val_e.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Data path (defaults to elasticity_valid_fixture in params.yaml)",
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

    p_val_raw = subparsers.add_parser("validate-raw")
    p_val_raw.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/raw_policy_data.csv"),
        help="Path to joined raw policy data",
    )
    p_val_raw.add_argument(
        "--suite-path",
        type=Path,
        default=None,
        help="Suite path (defaults to raw_policy_suite_path in params.yaml)",
    )
    p_val_raw.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/validated/raw_policy_validation_report.json"),
        help="Path to write validation report",
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

    p_ver_raw = subparsers.add_parser("version-raw")
    p_ver_raw.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/raw_policy_data.csv"),
        help="Path to raw policy data to version",
    )
    p_ver_raw.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/validated/raw_policy_validation_report.json"),
        help="Path to validation report to verify",
    )
    p_ver_raw.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/versioned/raw_policy_data.csv"),
        help="Destination path in versioned storage",
    )

    p_feat = subparsers.add_parser("build-features")
    p_feat.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/versioned/raw_policy_data.csv"),
        help="Validated raw policy table",
    )
    p_feat.add_argument(
        "--glm-output",
        type=Path,
        default=None,
        help="GLM view output (defaults to params.yaml)",
    )
    p_feat.add_argument(
        "--forest-output",
        type=Path,
        default=None,
        help="Forest view output (defaults to params.yaml)",
    )
    p_feat.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Metadata output (defaults to params.yaml)",
    )

    p_glm = subparsers.add_parser("build-glm")
    p_glm.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/features_glm.csv"),
        help="Stage 2 GLM view",
    )
    p_glm.add_argument(
        "--cap",
        type=float,
        default=None,
        help="Loss cap level (defaults to params.yaml glm_baseline.default_cap)",
    )
    p_glm.add_argument(
        "--fitted-output",
        type=Path,
        default=None,
        help="Fitted loss-cost CSV (defaults to params.yaml)",
    )
    p_glm.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Model metadata JSON (defaults to params.yaml)",
    )
    p_glm.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Calibration report JSON (defaults to params.yaml)",
    )

    p_seg = subparsers.add_parser("build-segments")
    p_seg.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/processed/features_glm.csv"),
        help="Stage 2 GLM view",
    )
    p_seg.add_argument(
        "--fitted-path",
        type=Path,
        default=Path("data/processed/glm_fitted_loss_cost.csv"),
        help="Stage 3 fitted loss cost",
    )
    p_seg.add_argument(
        "--loss-cost-output",
        type=Path,
        default=None,
        help="Segment table CSV (defaults to params.yaml)",
    )
    p_seg.add_argument(
        "--definitions-output",
        type=Path,
        default=None,
        help="Segment metadata JSON (defaults to params.yaml)",
    )
    p_seg.add_argument(
        "--policy-segments-output",
        type=Path,
        default=None,
        help="Policy-to-segment CSV (defaults to params.yaml)",
    )

    return parser


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest-elasticity":
        return ingest_file(args.input_path, args.output_path, "elasticity")
    if args.command == "ingest-regulatory":
        return ingest_file(args.input_path, args.output_path, "regulatory")
    if args.command == "ingest-raw":
        return ingest_raw_policy(
            args.freq_path, args.sev_path, args.manifest_path, args.output_path
        )
    if args.command == "validate-elasticity":
        return validate_file("elasticity", args.data_path, args.suite_path, args.report_path)
    if args.command == "validate-regulatory":
        return validate_file("regulatory", args.data_path, args.suite_path, args.report_path)
    if args.command == "validate-raw":
        return validate_file("raw_policy", args.data_path, args.suite_path, args.report_path)
    if args.command == "version-elasticity":
        return version_file(args.raw_path, args.report_path, args.output_path)
    if args.command == "version-regulatory":
        return version_file(args.raw_path, args.report_path, args.output_path)
    if args.command == "version-raw":
        return version_file(args.raw_path, args.report_path, args.output_path)
    if args.command == "build-features":
        return build_feature_artifacts(
            args.input_path, args.glm_output, args.forest_output, args.metadata_output
        )
    if args.command == "build-glm":
        return build_glm_baseline(
            args.input_path, args.cap, args.fitted_output, args.metadata_output, args.report_output
        )
    if args.command == "build-segments":
        return build_segments(
            args.features_path,
            args.fitted_path,
            args.loss_cost_output,
            args.definitions_output,
            args.policy_segments_output,
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
