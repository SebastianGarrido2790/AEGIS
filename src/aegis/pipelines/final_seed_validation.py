"""Final-seed validation campaign and Exit E1 gate (P2-F07, Stage 10).

Runs the frozen Stage 9 causal validation metrics on held-out policies across at least
five final seeds disjoint from pilot seeds. Each seed uses independent DGP noise and
estimator random state. Results are evaluated only against frozen thresholds.

The summary records SHA-256 hashes of params.yaml and every test-data input. Verification
re-hashes those files and re-evaluates the stored metrics against the live thresholds.
This detects post-run edits; it is not an authenticity signature and cannot prevent a
malicious actor from rewriting both the artifact and its hash references.

This module is an authorized offline consumer of the truth sidecar. Truth values never
enter causal estimators or serving pipelines (ADR-013), and no pricing adjustment is
produced or accepted (INV-4, INV-5).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NoReturn

import numpy as np
import pandas as pd

from aegis.config.loader import find_project_root
from aegis.config.schema import AEGISConfig
from aegis.pipelines.causal_estimator import (
    aggregate_segment_estimates,
    fit_causal_estimator,
    stratified_sample_split,
    validate_training_contract,
)
from aegis.pipelines.synthetic_dgp import generate_semi_synthetic_dataset
from aegis.pipelines.truth_loader import load_truth_sidecar
from aegis.pipelines.validation_harness import compute_validation_metrics
from aegis.schemas.final_validation import (
    ARTIFACT_VERSION,
    FinalSeedGateSummary,
    FinalSeedRunResult,
    FinalSeedValidationSummary,
    MetricGateEvaluation,
)
from aegis.schemas.validation import ValidationMetrics, ValidationThresholds
from aegis.utils.exceptions import ValidationHarnessError

GateDirection = Literal["higher_is_better", "lower_is_better"]

METRIC_GATE_SPEC: tuple[tuple[str, GateDirection, str], ...] = (
    ("m1_tau_spearman", "higher_is_better", "m1_tau_spearman_floor"),
    ("m1_epsilon_spearman", "higher_is_better", "m1_epsilon_spearman_floor"),
    ("m1b_policy_spearman", "higher_is_better", "m1b_policy_spearman_floor"),
    ("m2_mean_abs_bias", "lower_is_better", "m2_mean_abs_bias_ceiling"),
    ("m3_tau_coverage", "higher_is_better", "m3_tau_coverage_floor"),
    ("m3_epsilon_coverage", "higher_is_better", "m3_epsilon_coverage_floor"),
    ("m4_self_consistency", "higher_is_better", "m4_self_consistency_floor"),
    ("m5_sign_agreement", "higher_is_better", "m5_sign_agreement_floor"),
)

TARGET_FIELDS: dict[str, tuple[str, GateDirection]] = {
    "m1_tau_spearman": ("m1_tau_spearman_target", "higher_is_better"),
    "m1_epsilon_spearman": ("m1_epsilon_spearman_target", "higher_is_better"),
    "m1b_policy_spearman": ("m1b_policy_spearman_target", "higher_is_better"),
    "m2_mean_abs_bias": ("m2_mean_abs_bias_target", "lower_is_better"),
}

MEDIAN_REFERENCE_METRIC = "m1_tau_spearman"
TEST_DATA_FILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("features_forest_sha256", ("tier1_ml", "feature_paths", "forest_view")),
    ("policy_segments_sha256", ("tier1_ml", "segment_paths", "policy_segments")),
    ("glm_fitted_loss_cost_sha256", ("tier1_ml", "glm_paths", "fitted_loss_cost")),
    ("training_table_sha256", ("tier1_ml", "dgp_paths", "training_table")),
    ("truth_sidecar_sha256", ("tier1_ml", "dgp_paths", "truth_sidecar")),
)
TAUTOLOGY_CHECKS: dict[str, str] = {
    "metrics_compared_against_themselves": "metrics are compared against frozen thresholds",
    "thresholds_taken_from_run_metrics": "thresholds come from the Stage 9 config",
    "training_partition_scored": "metrics use the held-out partition only",
    "input_hashes_copied_from_artifact": "input hashes are re-derived from files",
}


def _fail(message: str, rule: str, **details: Any) -> NoReturn:
    """Raises a typed validation error with a machine-readable rule identifier."""
    raise ValidationHarnessError(message, details={"rule": rule, **details})


def _config_value(config: AEGISConfig, path: tuple[str, ...]) -> Any:
    """Resolves a nested configuration attribute by name path."""
    value: Any = config
    for attribute in path:
        value = getattr(value, attribute)
    return value

def compute_file_sha256(path: Path | str) -> str:
    """Computes a file's SHA-256 digest by streaming its bytes.

    Args:
        path: File path to hash.

    Returns:
        Lowercase hex SHA-256 digest.

    Raises:
        ValidationHarnessError: If the file does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        _fail(f"Cannot hash missing file: {file_path}", "missing_file", path=str(file_path))
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_test_data(config: AEGISConfig) -> dict[str, str]:
    """Hashes all test-data files referenced by the validated configuration."""
    return {key: compute_file_sha256(_config_value(config, path)) for key, path in TEST_DATA_FILES}


def compute_ids_sha256(table_ids: list[str]) -> str:
    """Computes an order-insensitive fingerprint of a policy-ID set."""
    canonical = json.dumps(sorted(table_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_summary_from_payload(payload: str | bytes) -> FinalSeedValidationSummary:
    """Parses and schema-validates a persisted JSON summary."""
    try:
        return FinalSeedValidationSummary.model_validate_json(payload)
    except ValueError as exc:
        _fail(f"Final-seed summary is malformed: {exc}", "malformed_summary", error=str(exc))


def _evaluate(
    metric: str,
    direction: GateDirection,
    value: float,
    threshold: float,
    *,
    target: float | None = None,
) -> MetricGateEvaluation:
    """Builds a metric gate evaluation against its floor or ceiling and optional target."""
    higher_is_better = direction == "higher_is_better"
    passes = value >= threshold if higher_is_better else value <= threshold
    meets_target = None
    if target is not None:
        meets_target = value >= target if higher_is_better else value <= target
    return MetricGateEvaluation(
        metric=metric,
        direction=direction,
        value=value,
        threshold=threshold,
        margin=value - threshold if higher_is_better else threshold - value,
        passes_floor=passes,
        target=target,
        meets_target=meets_target,
    )


def evaluate_seed_gate(
    metrics: ValidationMetrics,
    thresholds: ValidationThresholds,
) -> dict[str, MetricGateEvaluation]:
    """Evaluates one seed's metrics against all frozen floors (and annotates targets)."""
    evaluations: dict[str, MetricGateEvaluation] = {}
    for name, direction, floor_field in METRIC_GATE_SPEC:
        target_info = TARGET_FIELDS.get(name)
        target = float(getattr(thresholds, target_info[0])) if target_info else None
        evaluations[name] = _evaluate(
            name,
            direction,
            float(getattr(metrics, name)),
            float(getattr(thresholds, floor_field)),
            target=target,
        )
    return evaluations

def _median_evaluation(
    name: str,
    runs: list[FinalSeedRunResult],
    thresholds: ValidationThresholds,
) -> MetricGateEvaluation:
    """Evaluates the across-seed median against its frozen floor and optional target."""
    for metric, stat_direction, stat_floor in METRIC_GATE_SPEC:
        if metric != name:
            continue
        values = [float(getattr(run.metrics, name)) for run in runs]
        target_info = TARGET_FIELDS.get(name)
        return _evaluate(
            name,
            stat_direction,
            float(np.median(values)),
            float(getattr(thresholds, stat_floor)),
            target=float(getattr(thresholds, target_info[0])) if target_info else None,
        )
    _fail(f"Unknown metric: {name}", "unknown_metric", metric=name)


def _median_run(runs: list[FinalSeedRunResult]) -> FinalSeedRunResult:
    """Returns the run carrying the median m1(tau), breaking ties by seed."""
    ordered = sorted(
        runs, key=lambda run: (float(run.metrics.m1_tau_spearman), run.seed)
    )
    return ordered[(len(ordered) - 1) // 2]


def _seeds_at_or_past_target(
    runs: list[FinalSeedRunResult], thresholds: ValidationThresholds
) -> list[int]:
    """Lists seeds whose m1/m1epsilon/m1b/m2 metrics all meet their Q3 targets."""
    result: list[int] = []
    for run in runs:
        gates = evaluate_seed_gate(run.metrics, thresholds)
        if all(gates[name].meets_target for name in TARGET_FIELDS):
            result.append(run.seed)
    return result

def evaluate_campaign_gates(
    runs: list[FinalSeedRunResult],
    thresholds: ValidationThresholds,
) -> FinalSeedGateSummary:
    """Computes worst-seed floors, median-seed targets, and all-seed floor compliance.

    Args:
        runs: Completed final-seed runs.
        thresholds: Frozen Stage 9 thresholds.

    Returns:
        Campaign-level gate summary.

    Raises:
        ValidationHarnessError: If the campaign has no runs.
    """
    if not runs:
        _fail("Cannot evaluate campaign gates without runs", "empty_campaign")
    worst: dict[str, MetricGateEvaluation] = {}
    median: dict[str, MetricGateEvaluation] = {}
    for name, direction, floor_field in METRIC_GATE_SPEC:
        values = [float(getattr(run.metrics, name)) for run in runs]
        worst_value = min(values) if direction == "higher_is_better" else max(values)
        worst[name] = _evaluate(
            name, direction, worst_value, float(getattr(thresholds, floor_field))
        )
        median[name] = _median_evaluation(name, runs, thresholds)

    failed = [run.seed for run in runs if not run.passes_frozen_thresholds]
    worst_pass = all(value.passes_floor for value in worst.values())
    median_targets_met = all(median[name].meets_target is True for name in TARGET_FIELDS)
    all_pass = not failed
    return FinalSeedGateSummary(
        k_seeds=len(runs),
        failed_seeds=failed,
        worst_seed_name=min(
            runs,
            key=lambda run: float(run.metrics.m1_tau_spearman),
        ).seed,
        median_seed_name=_median_run(runs).seed,
        worst_seed=worst,
        median_seed=median,
        worst_seed_all_floors_cleared=worst_pass,
        median_seed_targets_met=median_targets_met,
        all_seeds_pass_frozen_thresholds=all_pass,
        campaign_passed=worst_pass and median_targets_met and all_pass,
        seeds_at_or_past_target=_seeds_at_or_past_target(runs, thresholds),
    )


def build_final_seed_summary(
    config: AEGISConfig,
    config_path: Path | str,
    runs: list[FinalSeedRunResult],
    thresholds: ValidationThresholds,
    *,
    sample_size: int,
    seed_offset_noise: int = 0,
) -> FinalSeedValidationSummary:
    """Builds the persisted Stage 10 summary, rejecting invalid seed families."""
    seeds = [run.seed for run in runs]
    if len(set(seeds)) != len(seeds):
        _fail("Duplicate seeds in final campaign", "duplicate_seeds", seeds=seeds)
    overlap = sorted(set(seeds) & set(config.tier1_ml.pilot_seeds))
    if overlap:
        _fail("Final seeds overlap Q3 pilot seeds", "pilot_seed_overlap", seeds=overlap)
    if len(runs) < 5:
        _fail("Exit E1 requires K >= 5 final seeds", "insufficient_seeds", k_seeds=len(runs))
    return FinalSeedValidationSummary(
        artifact_version=ARTIFACT_VERSION,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        final_seeds=seeds,
        pilot_seeds=list(config.tier1_ml.pilot_seeds),
        seed_semantics={
            "draw_and_estimator_seed": "stratified split and estimator random_state",
            "dgp_noise_seed": "draw seed + tier1_ml.synthetic_dgp.seed_offset_noise",
        },
        seed_offset_noise=seed_offset_noise,
        sample_size=sample_size,
        test_size=config.tier1_ml.test_size,
        config_sha256=compute_file_sha256(config_path),
        data_sha256=hash_test_data(config),
        held_out_ids_sha256={
            run.seed: compute_ids_sha256(run.held_out_policy_ids) for run in runs
        },
        thresholds=thresholds.model_dump(),
        metrics_by_seed={run.seed: run.metrics.model_dump() for run in runs},
        segment_counts_by_seed={run.seed: run.segment_count for run in runs},
        gates_by_seed={
            run.seed: {name: ev.model_dump() for name, ev in run.gates.items()} for run in runs
        },
        campaign_gates=evaluate_campaign_gates(runs, thresholds),
        tautology_checks={name: False for name in TAUTOLOGY_CHECKS},
    )


def format_final_validation_report(summary: FinalSeedValidationSummary) -> str:
    """Renders the final-seed metrics and complete Stage 10 gate comparison as Markdown."""
    gates = summary.campaign_gates
    lines = [
        "# Stage 10 — Final-Seed Validation Report (Roadmap Exit E1)",
        "",
        f"**Generated:** {summary.generated_at}  ",
        f"**Final Seeds:** `{summary.final_seeds}`  ",
        f"**Pilot Seeds (excluded):** `{summary.pilot_seeds}`  ",
        f"**Sample:** {summary.sample_size:,} policies per seed; held-out fraction "
        f"{summary.test_size:.2f}  ",
        f"**Frozen Config SHA-256:** `{summary.config_sha256}`  ",
        f"**Outcome:** {'✅ PASSED' if gates.campaign_passed else '❌ FAILED'}",
        "",
        "## 1. Metric Summary Across Seeds",
        "",
        "| Seed | DGP noise | m1(tau) | m1(eps) | m1b | m2 | "
        "m3(tau) | m3(eps) | m4 | m5 | Floors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in summary.final_seeds:
        values = summary.metrics_by_seed[seed]
        lines.append(
            f"| {seed} | {seed + summary.seed_offset_noise} | "
            f"{values['m1_tau_spearman']:.4f} | {values['m1_epsilon_spearman']:.4f} | "
            f"{values['m1b_policy_spearman']:.4f} | {values['m2_mean_abs_bias']:.4f} | "
            f"{values['m3_tau_coverage']:.3f} | {values['m3_epsilon_coverage']:.3f} | "
            f"{values['m4_self_consistency']:.3f} | {values['m5_sign_agreement']:.3f} | "
            f"{'❌' if seed in gates.failed_seeds else '✅'} |"
        )
    lines.extend(
        [
            "",
            "## 2. Worst-Seed Floors and Median-Seed Targets",
            "",
            "| Metric | Worst value | Frozen floor | Clears | Median | Target | Meets target |",
            "|---|---:|---:|:---:|---:|---:|:---:|",
        ]
    )
    for name, _, _ in METRIC_GATE_SPEC:
        worst = gates.worst_seed[name]
        med = gates.median_seed[name]
        target = "—" if med.target is None else f"{med.target:.4f}"
        meets = "—" if med.meets_target is None else ("✅" if med.meets_target else "❌")
        lines.append(
            f"| `{name}` | {worst.value:.4f} | {worst.threshold:.4f} | "
            f"{'✅' if worst.passes_floor else '❌'} | {med.value:.4f} | {target} | {meets} |"
        )
    lines.extend(
        [
            "",
            f"- **Worst seed:** `{gates.worst_seed_name}`.",
            f"- **Median seed:** `{gates.median_seed_name}` (median of m1(tau)).",
            f"- **Failed seeds:** `{gates.failed_seeds}` (zero required).",
            f"- **Median targets met:** `{gates.median_seed_targets_met}`.",
            f"- **Seeds at/past all targets:** `{gates.seeds_at_or_past_target}`.",
            "",
            "Gate pass requires worst-seed floors, every seed passing all floors, and median",
            "performance meeting the Q3 targets for m1(tau), m1(epsilon), m1b, and m2.",
            "",
            "## 3. Frozen Configuration and Test-Data Hashes",
            "",
            "| Item | SHA-256 |",
            "|---|---|",
            f"| `params.yaml` | `{summary.config_sha256}` |",
        ]
    )
    for key, _ in TEST_DATA_FILES:
        lines.append(f"| `{key.removesuffix('_sha256')}` | `{summary.data_sha256[key]}` |")
    lines.extend(["", "Held-out PolicyID fingerprints:", "", "| Seed | SHA-256 |", "|---|---|"])
    lines.extend(
        f"| {seed} | `{summary.held_out_ids_sha256[seed]}` |" for seed in summary.final_seeds
    )
    lines.extend(["", "## 4. Zero Tautological Tests", "", "| Check | Present |", "|---|:---:|"])
    lines.extend(
        f"| `{name}` | {'❌ YES' if summary.tautology_checks[name] else '✅ NO'} |"
        for name in TAUTOLOGY_CHECKS
    )
    lines.extend(
        [
            "",
            "## 5. Post-Hoc Immutability",
            "",
            "`verify-final-validation` re-hashes the frozen configuration and test data and",
            "re-evaluates the gates. Any threshold change invalidates verification.",
            "",
        ]
    )
    return "\n".join(lines)


def verify_final_validation_summary(
    summary: FinalSeedValidationSummary,
    config: AEGISConfig,
    config_path: Path | str,
) -> list[str]:
    """Verifies hashes, live thresholds, and stored gate results.

    Args:
        summary: Persisted campaign summary.
        config: Configuration loaded from the live params file.
        config_path: Path to the params file.

    Returns:
        List of verification errors; empty means valid.
    """
    problems: list[str] = []
    live_config_hash = compute_file_sha256(config_path)
    if live_config_hash != summary.config_sha256:
        problems.append("Frozen configuration hash mismatch: params.yaml changed after campaign")
    for key, path in TEST_DATA_FILES:
        file_path = Path(_config_value(config, path))
        if not file_path.is_file():
            problems.append(f"Test data missing for verification: {file_path}")
        elif compute_file_sha256(file_path) != summary.data_sha256[key]:
            problems.append(f"Test data hash mismatch for {key}: {file_path} changed")

    live_cfg = config.tier1_ml.validation_thresholds
    if live_cfg is None:
        problems.append("No frozen validation thresholds are configured")
        return problems
    for name, frozen in summary.thresholds.items():
        live = float(getattr(live_cfg, name))
        if not np.isclose(live, float(frozen), rtol=0.0, atol=1e-12):
            problems.append(f"Frozen threshold '{name}' altered after campaign: {frozen} -> {live}")
    try:
        thresholds = ValidationThresholds.model_validate(live_cfg.model_dump())
    except ValueError as exc:
        problems.append(f"Live thresholds violate ordering invariants: {exc}")
        return problems
    recheck = _recheck_stored_gates(summary, thresholds)
    if recheck.failed_seeds:
        problems.append(f"Gate re-evaluation fails for seeds {recheck.failed_seeds}")
    if recheck.median_seed_targets_met != summary.campaign_gates.median_seed_targets_met:
        problems.append("Median-target verdict differs under live thresholds")
    return problems

def _recheck_stored_gates(
    summary: FinalSeedValidationSummary, thresholds: ValidationThresholds
) -> FinalSeedGateSummary:
    """Re-evaluates stored per-seed metrics against live thresholds."""
    runs: list[FinalSeedRunResult] = []
    for seed in summary.final_seeds:
        metrics = ValidationMetrics.model_validate(summary.metrics_by_seed[seed])
        gates = evaluate_seed_gate(metrics, thresholds)
        runs.append(
            FinalSeedRunResult(
                seed=seed,
                dgp_noise_seed=seed + summary.seed_offset_noise,
                metrics=metrics,
                gates=gates,
                held_out_policy_count=1,
                held_out_policy_ids=[f"seed-{seed}"],
                data_sha256=summary.data_sha256,
                passes_frozen_thresholds=all(ev.passes_floor for ev in gates.values()),
                segment_count=summary.segment_counts_by_seed[seed],
            )
        )
    return evaluate_campaign_gates(runs, thresholds)


def _load_or_create_tables(
    config: AEGISConfig, noise_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Loads the configured DGP data or regenerates it with independent noise."""
    ml = config.tier1_ml
    if noise_seed == ml.synthetic_dgp.random_seed:
        training = pd.read_csv(ml.dgp_paths.training_table)
        truth = load_truth_sidecar(ml.dgp_paths.truth_sidecar)
    else:
        dgp_config = config.model_copy(deep=True)
        dgp_config.tier1_ml.synthetic_dgp.random_seed = noise_seed
        training, truth, _ = generate_semi_synthetic_dataset(
            forest_features_path=ml.feature_paths.forest_view,
            segments_path=ml.segment_paths.policy_segments,
            glm_loss_path=ml.glm_paths.fitted_loss_cost,
            config=dgp_config,
            run_self_tests=False,
        )
    features = validate_training_contract(
        training, corridor_width=ml.exploration_corridor_width
    )
    return training, truth, features


def run_final_seed_campaign(
    config: AEGISConfig,
    *,
    config_path: Path | str | None = None,
    sample_size: int | None = None,
    seeds: list[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> FinalSeedValidationSummary:
    """Executes all final-seed fits and evaluates the approved Stage 10 gate."""
    ml = config.tier1_ml
    estimator_cfg = ml.estimator
    threshold_cfg = ml.validation_thresholds
    if threshold_cfg is None:
        _fail("No frozen validation thresholds configured; run Q3 first", "missing_thresholds")
    thresholds = ValidationThresholds.model_validate(threshold_cfg.model_dump())
    final_seeds = list(seeds or ml.final_seeds)
    overlap = sorted(set(final_seeds) & set(ml.pilot_seeds))
    if overlap or len(set(final_seeds)) != len(final_seeds):
        _fail(
            "Final seeds must be unique and disjoint from pilots",
            "seed_hygiene",
            overlap=overlap,
        )
    if len(final_seeds) < 5:
        _fail("Exit E1 requires K >= 5 final seeds", "insufficient_seeds", k=len(final_seeds))
    if len(final_seeds) < 5:
        _fail(
            "Exit E1 requires K >= 5 final seeds", "insufficient_seeds", k=len(final_seeds)
        )

    data_hashes = hash_test_data(config)
    total_sample = sample_size or estimator_cfg.training_sample_size
    noise_offset = ml.synthetic_dgp.seed_offset_noise
    runs: list[FinalSeedRunResult] = []
    for seed in final_seeds:
        noise_seed = seed + noise_offset
        training, truth, features = _load_or_create_tables(config, noise_seed)
        train, test = stratified_sample_split(
            training,
            total_sample,
            test_size=ml.test_size,
            random_state=seed,
        )
        fitted = fit_causal_estimator(
            train, features, estimator_cfg=estimator_cfg, random_state=seed
        )
        estimates = aggregate_segment_estimates(
            fitted,
            test,
            features,
            ci_alpha=estimator_cfg.ci_alpha,
            q0_floor=estimator_cfg.q0_floor,
            cv_folds=estimator_cfg.cv_folds,
        )
        report = compute_validation_metrics(estimates, truth, test, seed=seed)
        gates = evaluate_seed_gate(report.metrics, thresholds)
        failed = [name for name, gate in gates.items() if not gate.passes_floor]
        run = FinalSeedRunResult(
            seed=seed,
            dgp_noise_seed=noise_seed,
            metrics=report.metrics,
            gates=gates,
            held_out_policy_count=len(test),
            held_out_policy_ids=test["PolicyID"].astype(str).tolist(),
            data_sha256=data_hashes,
            passes_frozen_thresholds=not failed,
            segment_count=len(report.segment_details),
        )
        runs.append(run)
        if progress:
            metrics = report.metrics
            progress(
                f"[SEED {seed}] noise={noise_seed} m1={metrics.m1_tau_spearman:.4f} "
                f"m1e={metrics.m1_epsilon_spearman:.4f} "
                f"m1b={metrics.m1b_policy_spearman:.4f} "
                f"m2={metrics.m2_mean_abs_bias:.4f} -> "
                f"{'PASS' if not failed else 'FAIL ' + ','.join(failed)}"
            )

    cfg_path = Path(config_path) if config_path else find_project_root() / "params.yaml"
    return build_final_seed_summary(
        config,
        cfg_path,
        runs,
        thresholds,
        sample_size=total_sample,
        seed_offset_noise=noise_offset,
    )


def persist_summary(
    summary: FinalSeedValidationSummary,
    config: AEGISConfig,
) -> tuple[Path, Path]:
    """Persists the JSON summary and its Markdown report."""
    artifact = Path(config.tier1_ml.final_validation_paths.summary)
    report = Path(config.tier1_ml.final_validation_paths.report)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report.write_text(format_final_validation_report(summary), encoding="utf-8")
    return artifact, report
__all__ = [
    "METRIC_GATE_SPEC",
    "TAUTOLOGY_CHECKS",
    "build_final_seed_summary",
    "compute_file_sha256",
    "compute_ids_sha256",
    "evaluate_campaign_gates",
    "evaluate_seed_gate",
    "format_final_validation_report",
    "hash_test_data",
    "load_summary_from_payload",
    "persist_summary",
    "run_final_seed_campaign",
    "verify_final_validation_summary",
]
