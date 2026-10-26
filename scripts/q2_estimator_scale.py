#!/usr/bin/env python3
"""Qualification Q2: select CausalForestDML training scale and nuisance family.

The script evaluates the approved 4 x 2 experiment using a single configured seed.
Ground-truth columns are joined only after fitting, for held-out evaluation.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import threading
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split

from aegis.config.loader import get_config
from aegis.pipelines.segments import aggregate_ratio_of_sums

SAMPLE_SIZES = (25_000, 100_000, 250_000, 413_169)
SPEARMAN_TOLERANCE = 0.02
REPORT_PATH = Path("reports/docs/phases/phase_2/q2_estimator_scale_report.md")
TRUTH_COLUMNS = {
    "tau",
    "epsilon",
    "q0",
    "true_effect",
    "tau_ground_truth",
    "baseline_renewal_q0",
    "true_tau",
    "true_epsilon",
    "fitted_loss_cost",
}


@dataclass(frozen=True)
class CellResult:
    """Measured outcomes for one sample-size and nuisance-family configuration."""

    sample_size: int
    nuisance_model: str
    wall_seconds: float
    peak_rss_mb: float
    peak_traced_memory_mb: float
    segment_spearman: float
    ci_coverage: float
    segment_count: int
    train_size: int = 0
    test_size: int = 0


def aggregate_segment_effects(
    predicted_tau: np.ndarray,
    baseline_q0: np.ndarray,
    segment_ids: np.ndarray,
) -> dict[str, float]:
    """Aggregates policy effects by segment with the production ratio-of-sums method."""
    effects = np.asarray(predicted_tau, dtype=float)
    if effects.ndim == 2 and effects.shape[1] == 1:
        effects = effects[:, 0]
    if effects.ndim != 1:
        raise ValueError("Expected one treatment-effect estimate per policy.")
    baseline = np.asarray(baseline_q0, dtype=float)
    segments = np.asarray(segment_ids)
    if baseline.ndim != 1 or segments.ndim != 1:
        raise ValueError("Baseline probabilities and segment IDs must be one-dimensional.")
    if not (len(effects) == len(baseline) == len(segments)):
        raise ValueError("Effect, baseline probability, and segment arrays must align.")
    if len(segments) == 0:
        raise ValueError("Cannot aggregate an empty held-out sample.")

    result: dict[str, float] = {}
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        result[str(segment_id)] = aggregate_ratio_of_sums(
            effects[mask].tolist(), baseline[mask].tolist()
        )
    return result


def estimator_feature_columns(columns: Sequence[str]) -> list[str]:
    """Validates the Stage 6 training contract and returns leakage-safe X columns."""
    column_set = set(columns)
    required = {"PolicyID", "treatment_rate_change", "renewed", "segment_id"}
    missing = required - column_set
    if missing:
        raise ValueError(f"Training data is missing required columns: {sorted(missing)}.")
    leaked = TRUTH_COLUMNS & column_set
    if leaked:
        raise ValueError(
            f"Truth columns cannot enter estimator inputs: {sorted(leaked)}."
        )
    if len(columns) != 29 or len(column_set) != len(columns):
        raise ValueError("Training data must match the unique 29-column Stage 6 contract.")
    features = [column for column in columns if column not in required]
    if len(features) != 25:
        raise ValueError(f"Expected 25 pre-treatment features; found {len(features)}.")
    return features


def stratified_sample_split(
    data: pd.DataFrame,
    sample_size: int,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draws a segment-stratified total sample, then creates its 80/20 split."""
    if "segment_id" not in data:
        raise ValueError("Training data must include segment_id for stratification.")
    if not 0 < sample_size <= len(data):
        raise ValueError(f"sample_size must be in [1, {len(data)}], got {sample_size}.")

    if sample_size == len(data):
        sample = data
    else:
        sample, _ = cast(
            tuple[pd.DataFrame, pd.DataFrame],
            train_test_split(
                data,
                train_size=sample_size,
                stratify=data["segment_id"],
                random_state=random_state,
            ),
        )
    train, test = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(
            sample,
            test_size=test_size,
            stratify=sample["segment_id"],
            random_state=random_state,
        ),
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def select_estimator_configuration(
    results: list[CellResult],
    *,
    runtime_budget_seconds: float,
    ram_budget_mb: float,
) -> CellResult:
    """Selects a budget-feasible Q2 result under the approved scale and tie-break rules."""
    if not results:
        raise ValueError("No Q2 results were provided.")
    if any(result.nuisance_model not in {"gbt", "linear"} for result in results):
        raise ValueError("Q2 results contain an unsupported nuisance family.")

    max_sample_size = max(result.sample_size for result in results)
    full_scale = [
        result
        for result in results
        if result.sample_size == max_sample_size and _within_budget(
            result, runtime_budget_seconds, ram_budget_mb
        )
    ]
    if full_scale:
        return _prefer_nuisance_family(full_scale)

    largest_scale_score = max(
        result.segment_spearman
        for result in results
        if result.sample_size == max_sample_size
    )
    qualified_results = [
        result
        for result in results
        if result.segment_spearman >= largest_scale_score - SPEARMAN_TOLERANCE
        and _within_budget(result, runtime_budget_seconds, ram_budget_mb)
    ]
    if not qualified_results:
        raise ValueError(
            "No Q2 configuration meets the runtime, RAM, and recovery criteria."
        )

    smallest_qualified_size = min(result.sample_size for result in qualified_results)
    smallest_scale_results = [
        result
        for result in qualified_results
        if result.sample_size == smallest_qualified_size
    ]
    return _prefer_nuisance_family(smallest_scale_results)


def _within_budget(
    result: CellResult, runtime_budget_seconds: float, ram_budget_mb: float
) -> bool:
    return (
        result.wall_seconds <= runtime_budget_seconds
        and result.peak_rss_mb <= ram_budget_mb
    )


def _prefer_nuisance_family(results: list[CellResult]) -> CellResult:
    """Chooses the best score, preferring GBT when scores are within the tolerance."""
    best_score = max(result.segment_spearman for result in results)
    tied_results = [
        result
        for result in results
        if best_score - result.segment_spearman <= SPEARMAN_TOLERANCE
    ]
    gbt_results = [result for result in tied_results if result.nuisance_model == "gbt"]
    return max(gbt_results or tied_results, key=lambda result: result.segment_spearman)


class _PeakRSSMonitor:
    """Samples current process RSS while one Q2 model is fitting."""

    def __init__(self) -> None:
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> _PeakRSSMonitor:
        self._sample_once()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._sample_once()
        self._stop.set()
        self._thread.join()

    def _sample(self) -> None:
        while not self._stop.wait(0.1):
            self._sample_once()

    def _sample_once(self) -> None:
        rss_bytes = _current_rss_bytes()
        if rss_bytes is not None:
            self.peak_bytes = max(self.peak_bytes, rss_bytes)


def _current_rss_bytes() -> int | None:
    """Returns the current process resident set size on Windows and Linux."""
    if sys.platform == "win32":
        return _windows_current_rss_bytes()
    if sys.platform.startswith("linux"):
        statm = Path("/proc/self/statm").read_text(encoding="ascii").split()
        return int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
    if os.name == "posix":
        import resource

        unit_bytes = 1 if sys.platform == "darwin" else 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit_bytes)
    return None


def _windows_current_rss_bytes() -> int:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_int
    success = get_process_memory_info(
        get_current_process(), ctypes.byref(counters), counters.cb
    )
    if not success:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.WorkingSetSize)


def _total_ram_bytes() -> int:
    """Returns installed physical memory on Windows and Unix-like platforms."""
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        global_memory_status = kernel32.GlobalMemoryStatusEx
        global_memory_status.argtypes = [ctypes.POINTER(MemoryStatus)]
        global_memory_status.restype = ctypes.c_int
        if not global_memory_status(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(status.ullTotalPhys)

    return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def _make_nuisance_models(
    family: str, random_state: int, cv_folds: int
) -> tuple[Any, Any]:
    """Builds the approved outcome and treatment nuisance models."""
    if family == "gbt":
        return (
            GradientBoostingClassifier(random_state=random_state),
            GradientBoostingRegressor(random_state=random_state),
        )
    if family == "linear":
        return (
            LogisticRegressionCV(
                cv=cv_folds, max_iter=1000, n_jobs=-1, random_state=random_state
            ),
            ElasticNetCV(
                cv=cv_folds, max_iter=10_000, n_jobs=-1, random_state=random_state
            ),
        )
    raise ValueError(f"Unsupported nuisance model family: {family}")


def _evaluate_cell(
    train: pd.DataFrame,
    test: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    sample_size: int,
    nuisance_model: str,
    random_state: int,
    cv_folds: int,
    expected_segment_count: int,
) -> CellResult:
    feature_columns = estimator_feature_columns(train.columns.tolist())
    model_y, model_t = _make_nuisance_models(nuisance_model, random_state, cv_folds)
    estimator = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_outcome=True,
        discrete_treatment=False,
        cv=cv_folds,
        random_state=random_state,
        n_jobs=-1,
    )

    x_train = train[feature_columns]
    treatment_train = train["treatment_rate_change"]
    outcome_train = train["renewed"]
    x_test = test[feature_columns]
    truth_test = truth.set_index("PolicyID").reindex(test["PolicyID"])
    if truth_test[["baseline_renewal_q0", "true_tau"]].isna().to_numpy().any():
        raise ValueError("Truth sidecar is missing held-out policy records.")

    tracemalloc.start()
    tracemalloc.reset_peak()
    start = time.perf_counter()
    with _PeakRSSMonitor() as rss_monitor:
        estimator.fit(outcome_train, treatment_train, X=x_train)
        predicted_tau = estimator.effect(x_test, T0=0, T1=1)
        segment_ids = test["segment_id"].to_numpy()
        q0 = truth_test["baseline_renewal_q0"].to_numpy(dtype=float)
        predicted_segments = aggregate_segment_effects(predicted_tau, q0, segment_ids)
        true_segments = aggregate_segment_effects(
            truth_test["true_tau"].to_numpy(dtype=float), q0, segment_ids
        )

        segment_intervals: dict[str, tuple[float, float]] = {}
        for segment_id in sorted(set(segment_ids)):
            mask = segment_ids == segment_id
            inference = estimator.ate_inference(x_test.loc[mask], T0=0, T1=1)
            lower, upper = inference.conf_int_mean(alpha=0.05)
            segment_intervals[str(segment_id)] = (float(lower), float(upper))

    wall_seconds = time.perf_counter() - start
    _, peak_traced_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if rss_monitor.peak_bytes == 0:
        raise RuntimeError(f"Process RSS measurement is unsupported on {sys.platform}.")

    predicted_series = pd.Series(predicted_segments, dtype=float).sort_index()
    true_series = pd.Series(true_segments, dtype=float).sort_index()
    segment_spearman = float(predicted_series.corr(true_series, method="spearman"))
    covered = [
        lower <= true_segments[segment_id] <= upper
        for segment_id, (lower, upper) in segment_intervals.items()
    ]
    if len(predicted_segments) != expected_segment_count or len(covered) != expected_segment_count:
        raise ValueError(
            "Held-out evaluation did not cover all configured segments: "
            f"expected {expected_segment_count}, predicted {len(predicted_segments)}, "
            f"intervals {len(covered)}."
        )

    return CellResult(
        sample_size=sample_size,
        nuisance_model=nuisance_model,
        wall_seconds=wall_seconds,
        peak_rss_mb=rss_monitor.peak_bytes / (1024**2),
        peak_traced_memory_mb=peak_traced_bytes / (1024**2),
        segment_spearman=segment_spearman,
        ci_coverage=sum(covered) / len(covered),
        segment_count=len(covered),
        train_size=len(train),
        test_size=len(test),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(
    results: list[CellResult],
    selected: CellResult,
    *,
    random_state: int,
    cv_folds: int,
    runtime_budget_seconds: float,
    ram_budget_mb: float,
    total_ram_mb: float,
    training_path: Path,
    truth_path: Path,
    definitions_path: Path,
) -> None:
    table = [
        "| N | Nuisance | Train/test | Wall time (s) | Peak RSS (MB) | "
        "Peak traced memory (MB) | Segment Spearman | 95% CI coverage | Segments |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: (item.sample_size, item.nuisance_model)):
        table.append(
            f"| {result.sample_size:,} | {result.nuisance_model} | "
            f"{result.train_size:,}/{result.test_size:,} | {result.wall_seconds:.1f} | "
            f"{result.peak_rss_mb:.1f} | {result.peak_traced_memory_mb:.1f} | "
            f"{result.segment_spearman:.4f} | {result.ci_coverage:.3f} | "
            f"{result.segment_count} |"
        )
    selected_ram_fraction = selected.peak_rss_mb / total_ram_mb
    full_data_size = max(result.sample_size for result in results)
    full_data_results = [
        result for result in results if result.sample_size == full_data_size
    ]
    full_data_passes = [
        result
        for result in full_data_results
        if _within_budget(result, runtime_budget_seconds, ram_budget_mb)
    ]
    if full_data_passes:
        scale_rationale = (
            f"{len(full_data_passes)} full-data fit(s) met both limits, so the full "
            "413,169-policy scale was adopted."
        )
    else:
        best_full_scale_spearman = max(
            result.segment_spearman for result in full_data_results
        )
        scale_rationale = (
            f"No full-data fit met both limits. The best full-scale Spearman was "
            f"{best_full_scale_spearman:.4f}; the selected scale is the smallest "
            f"budget-feasible sample within {SPEARMAN_TOLERANCE:.2f} of that value."
        )

    selected_scale_results = [
        result
        for result in results
        if result.sample_size == selected.sample_size
        and _within_budget(result, runtime_budget_seconds, ram_budget_mb)
    ]
    other_family = next(
        (
            result
            for result in selected_scale_results
            if result.nuisance_model != selected.nuisance_model
        ),
        None,
    )
    if other_family is not None:
        score_gap = abs(selected.segment_spearman - other_family.segment_spearman)
        if score_gap <= SPEARMAN_TOLERANCE and selected.nuisance_model == "gbt":
            family_rationale = (
                f"At N={selected.sample_size:,}, linear scored "
                f"{other_family.segment_spearman:.4f} and GBT scored "
                f"{selected.segment_spearman:.4f} (gap {score_gap:.4f}); because the gap "
                f"is within {SPEARMAN_TOLERANCE:.2f}, the approved tie-break selects GBT."
            )
        else:
            family_rationale = (
                f"At N={selected.sample_size:,}, `{selected.nuisance_model}` had the "
                f"higher budget-feasible Spearman score."
            )
    else:
        family_rationale = (
            f"`{selected.nuisance_model}` was the only budget-feasible nuisance family "
            f"at N={selected.sample_size:,}."
        )
    report = f"""# Q2 Estimator Scale Qualification Report

**Executed:** {date.today().isoformat()}  
**Seed:** {random_state} (single pilot seed; multi-seed evaluation is Stage 9/10 scope)  
**Decision:** n* = {selected.sample_size:,}, nuisance = `{selected.nuisance_model}`,
cv = {cv_folds}

## Method

Each requested N is a stratified total sample followed by an 80/20 stratified train/test
split. CausalForestDML was fit only on the training partition with {cv_folds}-fold cross-fitting.
It used 100 trees (library default) and all available CPUs. GBT nuisance models used
scikit-learn defaults; linear nuisance models used LogisticRegressionCV (max_iter=1,000)
and ElasticNetCV (max_iter=10,000), each with {cv_folds}-fold regularization CV and default
regularization grids. The run used {os.cpu_count() or "unknown"} logical CPUs.
The test partition was scored against the isolated truth sidecar; truth values were not
passed to estimator fitting. Predicted and true segment effects use the production
ratio-of-sums aggregation (`sum(tau) / sum(q0)`). Baseline renewal probability was constant
within each of the 35 configured segments in this truth sidecar, so the ratio-of-sums segment
mean and the `ate_inference` segment-average estimand coincide. Segment confidence intervals
are obtained from `ate_inference` on each held-out segment. No grid cells were skipped.

Peak process RSS was sampled from the operating system while each cell ran. Peak
`tracemalloc` memory is also recorded; unlike RSS, it does not include all native-library
allocations and is not used for the RAM budget decision.

## Results

{chr(10).join(table)}

## Selection

The full-data runtime budget is {runtime_budget_seconds / 60:.1f} minutes and the RSS
ceiling is {ram_budget_mb:.1f} MB ({ram_budget_mb / total_ram_mb:.1%} of {total_ram_mb:.1f} MB
installed RAM). {scale_rationale} {family_rationale} The selected fit ran in
{selected.wall_seconds / 60:.2f} minutes and peaked at {selected.peak_rss_mb:.1f} MB
({selected_ram_fraction:.1%} of installed RAM).

## Frozen configuration

```yaml
tier1_ml:
  estimator:
    training_sample_size: {selected.sample_size}
    nuisance_model: "{selected.nuisance_model}"
    cv_folds: {cv_folds}
    runtime_budget_minutes: {runtime_budget_seconds / 60:.1f}
```

## Reproducibility

- Training table SHA-256: `{_sha256(training_path)}`
- Truth sidecar SHA-256: `{_sha256(truth_path)}`
- Segment definitions SHA-256: `{_sha256(definitions_path)}`
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> int:
    """Runs the full Q2 grid and writes the qualification report."""
    config = get_config()
    tier1 = config.tier1_ml
    training_path = Path(tier1.dgp_paths.training_table)
    truth_path = Path(tier1.dgp_paths.truth_sidecar)
    definitions_path = Path(tier1.segment_paths.segment_definitions)
    for path in (training_path, truth_path, definitions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Q2 input is missing: {path}")

    training = pd.read_csv(training_path)
    truth = pd.read_csv(truth_path)
    definitions = json.loads(definitions_path.read_text(encoding="utf-8"))
    expected_segment_count = int(definitions["segment_count"])
    observed_segment_count = training["segment_id"].nunique()
    if observed_segment_count != expected_segment_count:
        raise ValueError(
            f"Training segment count {observed_segment_count} does not match definitions "
            f"({expected_segment_count})."
        )
    if training["PolicyID"].duplicated().any() or truth["PolicyID"].duplicated().any():
        raise ValueError("Q2 inputs contain duplicate PolicyID values.")
    if set(training["PolicyID"]) != set(truth["PolicyID"]):
        raise ValueError("Training table and truth sidecar PolicyID sets differ.")
    estimator_feature_columns(training.columns.tolist())

    random_state = tier1.random_state
    cv_folds = tier1.estimator.cv_folds
    runtime_budget_seconds = tier1.estimator.runtime_budget_minutes * 60.0
    total_ram_mb = _total_ram_bytes() / (1024**2)
    ram_budget_mb = total_ram_mb * 0.80
    results: list[CellResult] = []

    for sample_size in SAMPLE_SIZES:
        train, test = stratified_sample_split(
            training,
            sample_size,
            test_size=tier1.test_size,
            random_state=random_state,
        )
        print(
            f"Q2 N={sample_size:,}: train={len(train):,}, test={len(test):,}",
            flush=True,
        )
        for family in ("gbt", "linear"):
            print(f"  fitting nuisance family={family}", flush=True)
            result = _evaluate_cell(
                train,
                test,
                truth,
                sample_size=sample_size,
                nuisance_model=family,
                random_state=random_state,
                cv_folds=cv_folds,
                expected_segment_count=expected_segment_count,
            )
            results.append(result)
            print(
                f"  complete: {result.wall_seconds:.1f}s, "
                f"RSS={result.peak_rss_mb:.1f}MB, "
                f"Spearman={result.segment_spearman:.4f}, "
                f"coverage={result.ci_coverage:.3f}",
                flush=True,
            )

    expected_cells = len(SAMPLE_SIZES) * 2
    if len(results) != expected_cells:
        raise RuntimeError(f"Q2 completed {len(results)} cells; expected {expected_cells}.")
    selected = select_estimator_configuration(
        results,
        runtime_budget_seconds=runtime_budget_seconds,
        ram_budget_mb=ram_budget_mb,
    )
    _write_report(
        results,
        selected,
        random_state=random_state,
        cv_folds=cv_folds,
        runtime_budget_seconds=runtime_budget_seconds,
        ram_budget_mb=ram_budget_mb,
        total_ram_mb=total_ram_mb,
        training_path=training_path,
        truth_path=truth_path,
        definitions_path=definitions_path,
    )
    print(f"Q2 report written: {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
