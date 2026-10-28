"""Unit tests for the Stage 10 final-seed gate, artifact hashing, and tamper detection.

Every assertion compares independently derived quantities: metrics against frozen
thresholds, and files re-hashed on read against hashes stored in the artifact. No test
asserts a value against itself (Gate 10 zero-tautology requirement).
"""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from aegis.config.loader import load_params
from aegis.pipelines.final_seed_validation import (
    build_final_seed_summary,
    compute_file_sha256,
    compute_ids_sha256,
    evaluate_campaign_gates,
    evaluate_seed_gate,
    format_final_validation_report,
    load_summary_from_payload,
    verify_final_validation_summary,
)
from aegis.schemas.final_validation import FinalSeedRunResult
from aegis.schemas.validation import ValidationMetrics, ValidationThresholds
from aegis.utils.exceptions import ValidationHarnessError

CONFIG_PATH = Path("params.yaml")


def _metrics(**overrides: float) -> ValidationMetrics:
    """Builds a metrics record that clears every frozen floor unless overridden."""
    payload = {
        "m1_tau_spearman": 0.90,
        "m1_epsilon_spearman": 0.92,
        "m1b_policy_spearman": 0.84,
        "m2_mean_abs_bias": 0.09,
        "m3_tau_coverage": 1.0,
        "m3_epsilon_coverage": 1.0,
        "m4_self_consistency": 1.0,
        "m5_sign_agreement": 1.0,
    }
    payload.update(overrides)
    return ValidationMetrics.model_validate(payload)


def _thresholds() -> ValidationThresholds:
    """Returns the live frozen Stage 9 thresholds."""
    config = load_params(CONFIG_PATH)
    thresholds_cfg = config.tier1_ml.validation_thresholds
    assert thresholds_cfg is not None
    return ValidationThresholds.model_validate(thresholds_cfg.model_dump())


def _run(
    seed: int, metrics: ValidationMetrics, thresholds: ValidationThresholds
) -> FinalSeedRunResult:
    """Builds a run result whose gate verdict is derived from its metrics."""
    gates = evaluate_seed_gate(metrics, thresholds)
    return FinalSeedRunResult(
        seed=seed,
        dgp_noise_seed=seed + 10000,
        metrics=metrics,
        gates=gates,
        held_out_policy_count=3,
        held_out_policy_ids=["1", "2", "3"],
        data_sha256={"training_table_sha256": "a" * 64},
        passes_frozen_thresholds=all(ev.passes_floor for ev in gates.values()),
        segment_count=35,
    )


def _campaign(overrides: dict[int, dict[str, float]] | None = None) -> list[FinalSeedRunResult]:
    """Builds a five-seed campaign, optionally overriding metrics per seed."""
    thresholds = _thresholds()
    overrides = overrides or {}
    return [_run(seed, _metrics(**overrides.get(seed, {})), thresholds) for seed in range(501, 506)]


def test_evaluate_seed_gate_clears_frozen_floors() -> None:
    gates = evaluate_seed_gate(_metrics(), _thresholds())
    assert set(gates) == {
        "m1_tau_spearman",
        "m1_epsilon_spearman",
        "m1b_policy_spearman",
        "m2_mean_abs_bias",
        "m3_tau_coverage",
        "m3_epsilon_coverage",
        "m4_self_consistency",
        "m5_sign_agreement",
    }
    assert all(ev.passes_floor for ev in gates.values())
    assert gates["m1_tau_spearman"].margin == pytest.approx(
        _metrics().m1_tau_spearman - _thresholds().m1_tau_spearman_floor
    )


def test_evaluate_seed_gate_detects_bias_ceiling_breach() -> None:
    """m2 is a ceiling: exceeding it must fail even though other metrics are strong."""
    gates = evaluate_seed_gate(_metrics(m2_mean_abs_bias=0.20), _thresholds())
    assert not gates["m2_mean_abs_bias"].passes_floor
    assert gates["m2_mean_abs_bias"].margin < 0
    assert gates["m1_tau_spearman"].passes_floor


def test_campaign_gates_use_worst_and_median_not_single_seed() -> None:
    runs = _campaign({503: {"m1_tau_spearman": 0.80}})
    gates = evaluate_campaign_gates(runs, _thresholds())
    assert gates.worst_seed["m1_tau_spearman"].value == pytest.approx(0.80)
    assert not gates.worst_seed_all_floors_cleared
    assert gates.failed_seeds == [503]
    assert not gates.all_seeds_pass_frozen_thresholds
    # Median of [0.80, 0.90, 0.90, 0.90, 0.90] is 0.90, so the median statistic still passes.
    assert gates.median_seed["m1_tau_spearman"].passes_floor
    assert gates.worst_seed_name != gates.median_seed_name


def test_campaign_gates_clean_campaign_passes() -> None:
    """A campaign whose seeds all clear the floors passes every campaign statistic."""
    gates = evaluate_campaign_gates(_campaign(), _thresholds())
    assert gates.k_seeds == 5
    assert gates.failed_seeds == []
    assert gates.worst_seed_all_floors_cleared
    assert gates.all_seeds_pass_frozen_thresholds
    assert len(gates.seeds_at_or_past_target) == 5


def test_campaign_gates_reject_empty_campaign() -> None:
    with pytest.raises(ValidationHarnessError):
        evaluate_campaign_gates([], _thresholds())


def test_build_summary_rejects_pilot_seed_overlap() -> None:
    config = load_params(CONFIG_PATH)
    runs = _campaign()
    runs[0] = _run(42, _metrics(), _thresholds())
    with pytest.raises(ValidationHarnessError) as exc_info:
        build_final_seed_summary(config, CONFIG_PATH, runs, _thresholds(), sample_size=1000)
    assert "pilot" in str(exc_info.value)


def test_build_summary_rejects_fewer_than_five_seeds() -> None:
    config = load_params(CONFIG_PATH)
    with pytest.raises(ValidationHarnessError) as exc_info:
        build_final_seed_summary(
            config, CONFIG_PATH, _campaign()[:4], _thresholds(), sample_size=1000
        )
    assert "K >= 5" in str(exc_info.value)


def test_build_summary_records_hashes_and_zero_tautologies() -> None:
    config = load_params(CONFIG_PATH)
    summary = build_final_seed_summary(
        config, CONFIG_PATH, _campaign(), _thresholds(), sample_size=1000
    )
    assert summary.config_sha256 == compute_file_sha256(CONFIG_PATH)
    assert not any(summary.tautology_checks.values())
    assert summary.held_out_ids_sha256[501] == compute_ids_sha256(["1", "2", "3"])
    assert summary.thresholds["m1_tau_spearman_floor"] == (_thresholds().m1_tau_spearman_floor)


def test_report_renders_worst_median_and_tamper_guidance() -> None:
    config = load_params(CONFIG_PATH)
    summary = build_final_seed_summary(
        config, CONFIG_PATH, _campaign(), _thresholds(), sample_size=1000
    )
    report = format_final_validation_report(summary)
    assert "## 2. Worst-Seed Floors and Median-Seed Targets" in report
    assert "Zero Tautological Tests" in report
    assert "verify-final-validation" in report
    assert "- **Failed seeds:** `[]`" in report


def test_summary_round_trips_through_json() -> None:
    config = load_params(CONFIG_PATH)
    summary = build_final_seed_summary(
        config, CONFIG_PATH, _campaign(), _thresholds(), sample_size=1000
    )
    payload = json.dumps(summary.model_dump(mode="json"))
    assert load_summary_from_payload(payload) == summary


def test_malformed_summary_payload_fails_closed() -> None:
    with pytest.raises(ValidationHarnessError):
        load_summary_from_payload("{not json")


# ---------------------------------------------------------------------------
# Falsification: post-hoc threshold tampering invalidates the artifact
# ---------------------------------------------------------------------------


@pytest.fixture
def tamper_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Copies params.yaml and the fingerprinted test data into an isolated workspace.

    Returns:
        (config_copy, summary_path, data_dir)
    """
    config_copy = tmp_path / "params.yaml"
    shutil.copy(CONFIG_PATH, config_copy)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    raw = yaml.safe_load(config_copy.read_text(encoding="utf-8"))
    ml = raw["tier1_ml"]
    rel_paths = [
        ml["feature_paths"]["forest_view"],
        ml["segment_paths"]["policy_segments"],
        ml["glm_paths"]["fitted_loss_cost"],
        ml["dgp_paths"]["training_table"],
        ml["dgp_paths"]["truth_sidecar"],
    ]
    for relative in rel_paths:
        target = data_dir / Path(relative).name
        shutil.copy(relative, target)
    return config_copy, tmp_path / "summary.json", data_dir


def _rewrite_config_paths(config_copy: Path, data_dir: Path) -> None:
    """Repoints the workspace config at the isolated data copies."""
    raw = yaml.safe_load(config_copy.read_text(encoding="utf-8"))
    ml = raw["tier1_ml"]
    for section, key in [
        ("feature_paths", "forest_view"),
        ("segment_paths", "policy_segments"),
        ("glm_paths", "fitted_loss_cost"),
        ("dgp_paths", "training_table"),
        ("dgp_paths", "truth_sidecar"),
    ]:
        ml[section][key] = (data_dir / Path(ml[section][key]).name).as_posix()
    config_copy.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def test_falsification_threshold_tamper_is_detected(tamper_workspace) -> None:
    """Gate 10 falsification: editing a frozen threshold after the run invalidates it."""
    config_copy, _, data_dir = tamper_workspace
    _rewrite_config_paths(config_copy, data_dir)
    config = load_params(config_copy)
    summary = build_final_seed_summary(
        config, config_copy, _campaign(), _thresholds(), sample_size=1000
    )
    assert verify_final_validation_summary(summary, config, config_copy) == []

    raw = yaml.safe_load(config_copy.read_text(encoding="utf-8"))
    raw["tier1_ml"]["validation_thresholds"]["m1_tau_spearman_floor"] = 0.50
    config_copy.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    tampered = load_params(config_copy)

    problems = verify_final_validation_summary(summary, tampered, config_copy)
    joined = "\n".join(problems)
    assert "Frozen configuration hash mismatch" in joined
    assert "m1_tau_spearman_floor" in joined


def test_falsification_test_data_tamper_is_detected(tamper_workspace) -> None:
    """Altering a fingerprinted test-data file after the run invalidates the artifact."""
    config_copy, _, data_dir = tamper_workspace
    _rewrite_config_paths(config_copy, data_dir)
    config = load_params(config_copy)
    summary = build_final_seed_summary(
        config, config_copy, _campaign(), _thresholds(), sample_size=1000
    )
    truth = data_dir / Path(config.tier1_ml.dgp_paths.truth_sidecar).name
    truth.write_bytes(truth.read_bytes() + b"\n")

    problems = verify_final_validation_summary(summary, config, config_copy)
    assert any("truth_sidecar_sha256" in problem for problem in problems)


def test_verification_reports_gate_failure_under_relaxed_thresholds(tamper_workspace) -> None:
    """A hardened floor must surface as a gate re-evaluation failure, not a hash-only note."""
    config_copy, _, data_dir = tamper_workspace
    _rewrite_config_paths(config_copy, data_dir)
    config = load_params(config_copy)
    summary = build_final_seed_summary(
        config, config_copy, _campaign(), _thresholds(), sample_size=1000
    )

    raw = yaml.safe_load(config_copy.read_text(encoding="utf-8"))
    thresholds = raw["tier1_ml"]["validation_thresholds"]
    thresholds["m1_tau_spearman_floor"] = 0.95
    thresholds["m1_tau_spearman_target"] = 0.96
    config_copy.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    hardened = load_params(config_copy)

    problems = verify_final_validation_summary(summary, hardened, config_copy)
    assert any("Gate re-evaluation" in problem for problem in problems)


def test_frozen_threshold_ordering_rejects_tampered_config() -> None:
    """A floor raised above its target is rejected by the threshold schema itself."""
    tampered = _thresholds().model_dump()
    tampered["m1_tau_spearman_floor"] = 0.99
    with pytest.raises(ValueError):
        ValidationThresholds.model_validate(tampered)
