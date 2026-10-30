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
from aegis.pipelines.cap_sweep import run_cap_sweep
from aegis.pipelines.causal_estimator import run_causal_estimator_pipeline
from aegis.pipelines.data_contracts import (
    validate_elasticity_data,
    validate_raw_policy_data,
    validate_regulatory_corpus,
)
from aegis.pipelines.feature_pipeline import run_feature_pipeline
from aegis.pipelines.glm_baseline import run_glm_baseline
from aegis.pipelines.raw_intake import ingest_raw_data
from aegis.pipelines.registry import add_registry_subparsers
from aegis.pipelines.segments import run_segment_pipeline
from aegis.pipelines.synthetic_dgp import run_synthetic_dgp_pipeline
from aegis.utils.exceptions import (
    CausalEstimatorError,
    DataContractError,
    DGPVerificationError,
    FeaturePipelineError,
    GLMBaselineError,
    SegmentError,
    ValidationHarnessError,
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


def run_q1_cap_sweep(
    input_path: Path | None,
    work_dir: Path | None,
    report_path: Path | None,
) -> int:
    """Executes the Q1 loss-cap sweep and writes the Markdown report."""
    try:
        _, selected_cap, selected_load = run_cap_sweep(
            input_path=input_path,
            work_dir=work_dir,
            report_path=report_path,
        )
        print(f"[Q1] selected_cap={selected_cap:,.0f} excess_load={selected_load:.4f}")
        return 0
    except (GLMBaselineError, SegmentError) as exc:
        print(f"[BLOCK] Cap sweep failed: {exc}")
        return 1


def generate_dgp() -> int:
    """Generates semi-synthetic training and truth datasets with 5 self-tests (P2-F04)."""
    try:
        manifest = run_synthetic_dgp_pipeline()
        print(f"[SUCCESS] Semi-synthetic DGP generated: {manifest.sample_size} policies.")
        return 0
    except DGPVerificationError as exc:
        print(f"[BLOCK] DGP verification self-test failed: {exc}")
        return 1


def train_estimator(
    segment_estimates: Path | None = None,
    metadata: Path | None = None,
    model_artifact: Path | None = None,
    run_manifest: Path | None = None,
) -> int:
    """Trains the Stage 8 causal estimator and writes segment inference artifacts."""
    config = get_config()
    if any(
        path is not None
        for path in (segment_estimates, metadata, model_artifact, run_manifest)
    ):
        config = config.model_copy(deep=True)
        paths = config.tier1_ml.estimator_paths
        if segment_estimates is not None:
            paths.segment_estimates = str(segment_estimates)
        if metadata is not None:
            paths.metadata = str(metadata)
        if model_artifact is not None:
            paths.model_artifact = str(model_artifact)
        if run_manifest is not None:
            paths.run_manifest = str(run_manifest)
    try:
        result = run_causal_estimator_pipeline(config)
    except CausalEstimatorError as exc:
        print(f"[BLOCK] Causal estimator failed: {exc}")
        return 1
    print(f"[SUCCESS] Causal estimator trained: {result.model_version}")
    print(
        f"[ARTIFACT] segments={result.segment_count} "
        f"test_rows={result.test_rows} "
        f"finite_rate={result.finite_interval_rate:.4f}"
    )
    return 0


def validate_causal(
    segment_estimates: Path | None = None,
    truth_sidecar: Path | None = None,
    training_table: Path | None = None,
) -> int:
    """Validates segment elasticity estimates against the ground truth sidecar (Stage 9)."""
    import pandas as pd

    from aegis.pipelines.causal_estimator import stratified_sample_split
    from aegis.pipelines.validation_harness import compute_validation_metrics

    config = get_config()
    seg_path = segment_estimates or Path(config.tier1_ml.estimator_paths.segment_estimates)
    truth_path = truth_sidecar or Path(config.tier1_ml.dgp_paths.truth_sidecar)
    train_path = training_table or Path(config.tier1_ml.dgp_paths.training_table)

    try:
        seg_df = pd.read_csv(seg_path)
        train_df = pd.read_csv(train_path)
        _, held_out_test = stratified_sample_split(
            train_df,
            config.tier1_ml.estimator.training_sample_size,
            test_size=config.tier1_ml.test_size,
            random_state=config.tier1_ml.random_state,
        )
        report = compute_validation_metrics(
            segment_estimates=seg_df,
            truth_sidecar=str(truth_path),
            held_out_test=held_out_test,
        )
    except ValidationHarnessError as exc:
        print(f"[BLOCK] Validation harness failed: {exc}")
        return 1
    except Exception as exc:
        print(f"[ERROR] Unexpected validation error: {exc}")
        return 1

    m = report.metrics
    print(f"[SUCCESS] Causal validation complete for {len(report.segment_details)} segments")
    print(
        f"[METRICS] m1_tau={m.m1_tau_spearman:.4f} m1_eps={m.m1_epsilon_spearman:.4f} "
        f"m1b={m.m1b_policy_spearman:.4f} m2_bias={m.m2_mean_abs_bias:.4f} "
        f"m3_tau_cov={m.m3_tau_coverage:.3f} m3_eps_cov={m.m3_epsilon_coverage:.3f} "
        f"m4_cons={m.m4_self_consistency:.3f} m5_sign={m.m5_sign_agreement:.3f}"
    )
    return 0


def calibrate_q3(
    report_path: Path | None = None,
    pilot_sample_size: int | None = None,
) -> int:
    """Executes Qualification Q3 threshold calibration on pilot seeds (P2-F07)."""
    from aegis.pipelines.validation_harness import calibrate_q3_pipeline

    dest = report_path or Path("reports/docs/phases/phase_2/q3_threshold_calibration_report.md")
    try:
        report = calibrate_q3_pipeline(
            report_path=dest,
            pilot_sample_size=pilot_sample_size,
        )
    except ValidationHarnessError as exc:
        print(f"[BLOCK] Q3 threshold calibration failed: {exc}")
        return 1
    except Exception as exc:
        print(f"[ERROR] Unexpected Q3 calibration error: {exc}")
        return 1

    print("[SUCCESS] Q3 threshold calibration complete")
    print(f"[REPORT] Report written to: {dest}")
    print(
        f"[THRESHOLDS] m1_floor={report.recommended_thresholds.m1_tau_spearman_floor:.4f} "
        f"m1_target={report.recommended_thresholds.m1_tau_spearman_target:.4f}"
    )
    return 0


def run_final_validation(sample_size: int | None = None) -> int:
    """Executes the Stage 10 final-seed campaign and persists the Exit E1 artifact."""
    from aegis.pipelines.final_seed_validation import (
        persist_summary,
        run_final_seed_campaign,
    )

    config = get_config()
    try:
        summary = run_final_seed_campaign(config, sample_size=sample_size, progress=print)
        artifact, report = persist_summary(summary, config)
    except ValidationHarnessError as exc:
        print(f"[BLOCK] Final-seed validation failed: {exc}")
        return 1

    gates = summary.campaign_gates
    outcome = (
        "PASSED"
        if gates.worst_seed_all_floors_cleared and gates.all_seeds_pass_frozen_thresholds
        else "FAILED"
    )
    print(f"[{outcome}] Final-seed campaign complete: K={gates.k_seeds}")
    print(
        f"[GATES] worst_seed={gates.worst_seed_name} median_seed={gates.median_seed_name} "
        f"failed_seeds={gates.failed_seeds} "
        f"worst_floors_cleared={gates.worst_seed_all_floors_cleared}"
    )
    for name in (
        "m1_tau_spearman",
        "m1_epsilon_spearman",
        "m1b_policy_spearman",
        "m2_mean_abs_bias",
    ):
        worst = gates.worst_seed[name]
        print(
            f"[GATE {name}] worst={worst.value:.4f} threshold={worst.threshold:.4f} "
            f"margin={worst.margin:+.4f}"
        )
    print(f"[ARTIFACT] {artifact}")
    print(f"[REPORT] {report}")
    return 0 if outcome == "PASSED" else 1


def verify_final_validation(
    summary_path: Path | None = None,
    config_path: Path | None = None,
) -> int:
    """Verifies the persisted Exit E1 artifact against the live frozen configuration."""
    from aegis.config.loader import find_project_root, load_params
    from aegis.pipelines.final_seed_validation import (
        compute_file_sha256,
        load_summary_from_payload,
        verify_final_validation_summary,
    )

    defaults = get_config()
    artifact = Path(summary_path or defaults.tier1_ml.final_validation_paths.summary)
    resolved_config = Path(config_path) if config_path else find_project_root() / "params.yaml"
    try:
        if not artifact.is_file():
            raise ValidationHarnessError(f"Validation artifact not found: {artifact}")
        summary = load_summary_from_payload(artifact.read_bytes())
        live_config = load_params(resolved_config)
        problems = verify_final_validation_summary(summary, live_config, resolved_config)
    except ValidationHarnessError as exc:
        print(f"[BLOCK] Final-seed verification failed: {exc}")
        return 1

    if problems:
        print("[BLOCK] Final-seed validation artifact is INVALID:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("[SUCCESS] Final-seed validation artifact verified")
    print(f"[CONFIG] params.yaml sha256={compute_file_sha256(resolved_config)}")
    print(
        f"[GATES] K={summary.campaign_gates.k_seeds} "
        f"failed_seeds={summary.campaign_gates.failed_seeds}"
    )
    return 0


def run_sensitivity(
    summary_path: Path | None = None,
    report_path: Path | None = None,
) -> int:
    """Executes hidden-confounder sensitivity sweep and writes summary (Stage 11, P2-F08)."""
    from aegis.pipelines.sensitivity import run_sensitivity_sweep
    from aegis.utils.exceptions import SensitivityAnalysisError

    config = get_config()
    print("[SENSITIVITY] Executing 10-cell hidden-confounder sensitivity sweep...")
    try:
        summary = run_sensitivity_sweep(
            config=config,
            output_summary_path=summary_path,
            output_report_path=report_path,
        )
    except SensitivityAnalysisError as exc:
        print(f"[BLOCK] Sensitivity analysis failed: {exc}")
        return 1

    print(f"[SUCCESS] Sensitivity sweep complete: {summary.grid_cells_completed}/10 cells.")
    print(f"[ROBUSTNESS] Robustness value R_v = {summary.robustness_value:.4f}")
    print(f"[SCENARIO S4] Locked cell: {summary.scenario_s4.cell_id}")
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

    p_q1 = subparsers.add_parser("run-q1")
    p_q1.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/features_glm.csv"),
        help="Stage 2 GLM view",
    )
    p_q1.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Temporary directory for per-cap fitted artifacts",
    )
    p_q1.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Markdown report destination (defaults to the Q1 report path)",
    )

    subparsers.add_parser(
        "generate-dgp",
        help="Generate semi-synthetic DGP training data and truth sidecar (P2-F04)",
    )
    p_est = subparsers.add_parser(
        "train-estimator",
        help="Train the Stage 8 CausalForestDML and estimate segment elasticities",
    )
    p_est.add_argument(
        "--segment-estimates",
        type=Path,
        default=None,
        help="Segment table CSV (defaults to params.yaml)",
    )
    p_est.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Deterministic metadata JSON (defaults to params.yaml)",
    )
    p_est.add_argument(
        "--model-artifact",
        type=Path,
        default=None,
        help="Non-promotable joblib artifact (defaults to params.yaml)",
    )
    p_est.add_argument(
        "--run-manifest",
        type=Path,
        default=None,
        help="Run-only manifest JSON (defaults to params.yaml)",
    )

    p_val = subparsers.add_parser(
        "validate-causal",
        help="Validate causal elasticity estimates against ground truth sidecar (Stage 9)",
    )
    p_val.add_argument(
        "--segment-estimates",
        type=Path,
        default=None,
        help="Segment table CSV (defaults to params.yaml)",
    )
    p_val.add_argument(
        "--truth-sidecar",
        type=Path,
        default=None,
        help="Ground truth sidecar CSV (defaults to params.yaml)",
    )
    p_val.add_argument(
        "--training-table",
        type=Path,
        default=None,
        help="Training table CSV (defaults to params.yaml)",
    )

    p_q3 = subparsers.add_parser(
        "calibrate-q3",
        help="Execute Qualification Q3 threshold calibration on pilot seeds (P2-F07)",
    )
    p_q3.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Destination markdown report path",
    )
    p_q3.add_argument(
        "--pilot-sample-size",
        type=int,
        default=None,
        help="Optional pilot sample size override",
    )

    p_final = subparsers.add_parser(
        "run-final-validation",
        help="Run the Stage 10 final-seed campaign and persist the Exit E1 artifact",
    )
    p_final.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size override (defaults to the Q2-frozen scale)",
    )

    p_verify = subparsers.add_parser(
        "verify-final-validation",
        help="Verify the Exit E1 artifact against the live frozen config and test data",
    )
    p_verify.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Summary artifact JSON (defaults to params.yaml)",
    )
    p_verify.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Configuration file to verify against (defaults to project params.yaml)",
    )

    p_sens = subparsers.add_parser(
        "run-sensitivity",
        help="Execute 10-cell hidden-confounder sensitivity sweep and lock Scenario S4 (P2-F08)",
    )
    p_sens.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Path for JSON sensitivity summary",
    )
    p_sens.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Path for Markdown sensitivity report",
    )

    add_registry_subparsers(subparsers)

    return parser


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "func"):
        return int(args.func(args))

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
    if args.command == "run-q1":
        return run_q1_cap_sweep(args.input_path, args.work_dir, args.report_path)
    if args.command == "generate-dgp":
        return generate_dgp()
    if args.command == "train-estimator":
        return train_estimator(
            args.segment_estimates,
            args.metadata,
            args.model_artifact,
            args.run_manifest,
        )
    if args.command == "validate-causal":
        return validate_causal(
            args.segment_estimates,
            args.truth_sidecar,
            args.training_table,
        )
    if args.command == "calibrate-q3":
        return calibrate_q3(
            args.report_path,
            args.pilot_sample_size,
        )
    if args.command == "run-final-validation":
        return run_final_validation(args.sample_size)
    if args.command == "verify-final-validation":
        return verify_final_validation(args.summary_path, args.config_path)
    if args.command == "run-sensitivity":
        return run_sensitivity(args.summary_path, args.report_path)

    return 1


if __name__ == "__main__":
    sys.exit(main())
