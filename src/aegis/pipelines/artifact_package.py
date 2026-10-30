"""Deterministic packaging of Tier 1 artifacts into a governed serving bundle (P2-F09).

This module is pure: it reads validated Phase 2 artifacts, resolves provenance, and
writes a plain-data bundle. It never talks to MLflow and never touches a serialized
estimator, so packaging and lineage can be verified without a registry backend
(ADR-014, P2-D11, INV-7).

Field provenance of the published segment table:

* ``policy_count``, ``exposure_years``, ``loss_cost_per_year``,
  ``base_premium_per_year`` come from the Stage 4 segment economics table
  (P2-F05). ``policy_count`` is the full-segment policy count that satisfies the
  1,000-policy floor; the Stage 8 inference count is intentionally not used.
* ``baseline_retention_q0``, ``tau_hat``, ``epsilon_hat``, and the epsilon interval
  come from the Stage 8 causal estimate table (P2-F06).
* ``model_version`` and ``data_version`` identify the estimator configuration and
  the DGP configuration fingerprint that produced those estimates.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
import yaml
from pydantic import ValidationError

from aegis.schemas.registry import ModelLineageMetadata
from aegis.schemas.segments import Tier1SegmentOutput, Tier1SegmentTable
from aegis.utils.exceptions import RegistryPackagingError, RegistrySecurityError

SEGMENT_TABLE_FILE: Final[str] = "tier1_segments.csv"
LINEAGE_FILE: Final[str] = "lineage.json"
STANDARD_BUNDLE_FILES: Final[tuple[str, ...]] = (
    SEGMENT_TABLE_FILE,
    "glm_calibration_report.json",
    "causal_validation_summary.json",
    "sensitivity_summary.json",
    LINEAGE_FILE,
)
FORBIDDEN_BUNDLE_SUFFIXES: Final[tuple[str, ...]] = (".pkl", ".pickle", ".joblib")
DEFAULT_LOCK_DEP_PATH: Final[str] = "data/processed/elasticity_training.csv"

_ECONOMICS_COLUMNS: Final[tuple[str, ...]] = (
    "segment_id",
    "policy_count",
    "exposure_years",
    "loss_cost_per_year",
    "base_premium_per_year",
)
_ESTIMATE_COLUMNS: Final[tuple[str, ...]] = (
    "segment_id",
    "baseline_retention_q0",
    "tau_hat",
    "epsilon_hat",
    "epsilon_ci_lower",
    "epsilon_ci_upper",
)


def compute_file_sha256(path: Path) -> str:
    """Returns the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_file_md5(path: Path) -> str:
    """Returns the MD5 hex digest of a file's bytes (DVC's content hash)."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Loads a JSON object, raising a packaging error when unavailable or malformed."""
    if not Path(path).is_file():
        raise RegistryPackagingError(
            f"{label} not found: {path}", details={"path": str(path), "label": label}
        )
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RegistryPackagingError(
            f"{label} is not valid JSON: {exc}",
            details={"path": str(path), "label": label},
        ) from exc
    if not isinstance(payload, dict):
        raise RegistryPackagingError(
            f"{label} must be a JSON object",
            details={"path": str(path), "label": label},
        )
    return payload


def _require_columns(frame: pd.DataFrame, expected: Sequence[str], *, label: str) -> None:
    """Raises if a table is missing any required column."""
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise RegistryPackagingError(
            f"{label} is missing required columns: {missing}",
            details={"label": label, "missing": missing},
        )


def _require_unique_ids(frame: pd.DataFrame, *, label: str) -> None:
    """Raises if a table repeats a segment identifier."""
    duplicates = sorted(
        frame.loc[frame["segment_id"].duplicated(), "segment_id"].astype(str).unique()
    )
    if duplicates:
        raise RegistryPackagingError(
            f"{label} has duplicate segment_id values: {duplicates}",
            details={"label": label, "duplicates": duplicates},
        )


def build_segment_rows(
    economics: pd.DataFrame,
    estimates: pd.DataFrame,
    *,
    model_version: str,
    data_version: str,
) -> list[Tier1SegmentOutput]:
    """Joins Stage 4 economics with Stage 8 causal estimates into validated rows.

    Args:
        economics: Segment economics table (P2-F05).
        estimates: Segment causal estimate table (P2-F06).
        model_version: Estimator version identifier from the Stage 8 metadata.
        data_version: DGP configuration fingerprint for the training data.

    Returns:
        One validated :class:`Tier1SegmentOutput` per segment, ordered by segment id.

    Raises:
        RegistryPackagingError: On missing columns, duplicate/mismatched segment ids,
            an empty result, or a row that violates the Tier 1 contract.
    """
    _require_columns(economics, _ECONOMICS_COLUMNS, label="segment economics")
    _require_columns(estimates, _ESTIMATE_COLUMNS, label="segment estimates")
    _require_unique_ids(economics, label="segment economics")
    _require_unique_ids(estimates, label="segment estimates")

    economics_ids = set(economics["segment_id"].astype(str))
    estimate_ids = set(estimates["segment_id"].astype(str))
    if economics_ids != estimate_ids:
        raise RegistryPackagingError(
            "segment economics and causal estimates cover different segment sets",
            details={
                "missing_estimates": sorted(economics_ids - estimate_ids),
                "missing_economics": sorted(estimate_ids - economics_ids),
            },
        )

    merged = economics.merge(
        estimates[
            [
                "segment_id",
                "baseline_retention_q0",
                "tau_hat",
                "epsilon_hat",
                "epsilon_ci_lower",
                "epsilon_ci_upper",
            ]
        ],
        on="segment_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("segment_id", kind="stable")

    rows: list[Tier1SegmentOutput] = []
    for record in merged.to_dict(orient="records"):
        payload = {
            "segment_id": str(record["segment_id"]),
            "policy_count": int(record["policy_count"]),
            "exposure_years": float(record["exposure_years"]),
            "baseline_retention_q0": float(record["baseline_retention_q0"]),
            "tau_hat": float(record["tau_hat"]),
            "epsilon_hat": float(record["epsilon_hat"]),
            "epsilon_ci_lower": float(record["epsilon_ci_lower"]),
            "epsilon_ci_upper": float(record["epsilon_ci_upper"]),
            "loss_cost_per_year": float(record["loss_cost_per_year"]),
            "base_premium_per_year": float(record["base_premium_per_year"]),
            "model_version": model_version,
            "data_version": data_version,
            "semi_synthetic": True,
        }
        try:
            rows.append(Tier1SegmentOutput.model_validate(payload))
        except ValidationError as exc:
            raise RegistryPackagingError(
                f"segment '{payload['segment_id']}' violates the Tier 1 contract: {exc}",
                details={"segment_id": payload["segment_id"]},
            ) from exc

    if not rows:
        raise RegistryPackagingError("no segments were packaged into the serving bundle")
    return rows


def load_tier1_segment_table(
    segment_loss_cost_path: Path,
    segment_estimates_path: Path,
    estimator_metadata_path: Path,
    dgp_manifest_path: Path,
) -> Tier1SegmentTable:
    """Builds the served :class:`Tier1SegmentTable` from the Stage 4/8/10 artifacts.

    Args:
        segment_loss_cost_path: Stage 4 segment economics CSV.
        segment_estimates_path: Stage 8 segment causal estimate CSV.
        estimator_metadata_path: Stage 8 estimator metadata JSON.
        dgp_manifest_path: Stage 4 DGP manifest JSON.

    Returns:
        The validated segment table, one row per published segment.

    Raises:
        RegistryPackagingError: If an input is missing/unreadable or any join or
            contract check fails.
    """
    for path in (segment_loss_cost_path, segment_estimates_path):
        if not Path(path).is_file():
            raise RegistryPackagingError(
                f"segment table not found: {path}", details={"path": str(path)}
            )

    metadata = _load_json_object(estimator_metadata_path, label="estimator metadata")
    manifest = _load_json_object(dgp_manifest_path, label="DGP manifest")

    model_version = str(metadata.get("model_version") or "")
    data_version = str(manifest.get("config_fingerprint") or "")
    if not model_version:
        raise RegistryPackagingError(
            "estimator metadata is missing 'model_version'",
            details={"path": str(estimator_metadata_path)},
        )
    if not data_version:
        raise RegistryPackagingError(
            "DGP manifest is missing 'config_fingerprint'",
            details={"path": str(dgp_manifest_path)},
        )

    economics = pd.read_csv(segment_loss_cost_path)
    estimates = pd.read_csv(segment_estimates_path)
    rows = build_segment_rows(
        economics, estimates, model_version=model_version, data_version=data_version
    )
    return Tier1SegmentTable(segments=rows)


def write_segment_table(table: Tier1SegmentTable, path: Path) -> None:
    """Writes the segment table as plain CSV (no estimator, no pickles)."""
    frame = pd.DataFrame([segment.model_dump() for segment in table.segments])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_lineage_file(lineage: ModelLineageMetadata, path: Path) -> Path:
    """Writes the machine-readable lineage record as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(lineage.model_dump(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def read_segment_table(path: Path) -> Tier1SegmentTable:
    """Reads and validates a packaged segment CSV.

    Raises:
        RegistryPackagingError: If the file is absent or a row violates the contract.
    """
    if not Path(path).is_file():
        raise RegistryPackagingError(
            f"segment table not found: {path}", details={"path": str(path)}
        )
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        _ECONOMICS_COLUMNS
        + (
            "baseline_retention_q0",
            "tau_hat",
            "epsilon_hat",
            "epsilon_ci_lower",
            "epsilon_ci_upper",
            "model_version",
            "data_version",
            "semi_synthetic",
        ),
        label="packaged segment table",
    )
    try:
        return Tier1SegmentTable.model_validate({"segments": frame.to_dict(orient="records")})
    except ValidationError as exc:
        raise RegistryPackagingError(
            f"packaged segment table violates the Tier 1 contract: {exc}",
            details={"path": str(path)},
        ) from exc


def _run_git(repo_root: Path, *args: str) -> str:
    """Runs a git command in the repository and returns stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RegistryPackagingError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}",
            details={"command": "git " + " ".join(args)},
        )
    return result.stdout.strip()


def resolve_git_commit(repo_root: Path) -> tuple[str, bool]:
    """Returns the current HEAD commit hash and whether the worktree is dirty.

    Raises:
        RegistryPackagingError: If git is unavailable or the repository is invalid.
    """
    commit = _run_git(repo_root, "rev-parse", "HEAD")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RegistryPackagingError(
            f"HEAD is not a 40-character git SHA: {commit!r}",
            details={"commit": commit},
        )
    dirty = bool(_run_git(repo_root, "status", "--porcelain"))
    return commit, dirty


def resolve_dvc_data_hash(
    dvc_lock_path: Path,
    training_table_path: Path,
    *,
    lock_dep_path: str = DEFAULT_LOCK_DEP_PATH,
) -> str:
    """Returns the content hash of the versioned training table (INV-3, ADR-015).

    The DVC lock hash is authoritative when present. Because the lock can go stale
    relative to regenerated data, the recorded MD5 is compared against the file's
    recomputed MD5 and a mismatch fails loud. When no lock entry exists the SHA-256
    of the file is used as a documented fallback.

    Args:
        dvc_lock_path: Path to ``dvc.lock``.
        training_table_path: Path to the training table the hash describes.
        lock_dep_path: Repository-relative lock path of the training table.

    Returns:
        The MD5 recorded in the lock (verified) or the SHA-256 fallback.

    Raises:
        RegistryPackagingError: If the training table is missing or the lock hash
            no longer matches the file on disk.
    """
    training_table = Path(training_table_path)
    if not training_table.is_file():
        raise RegistryPackagingError(
            f"training table not found: {training_table}",
            details={"path": str(training_table)},
        )

    lock_path = Path(dvc_lock_path)
    if lock_path.is_file():
        with lock_path.open("r", encoding="utf-8") as handle:
            lock_payload = yaml.safe_load(handle) or {}
        recorded = _find_lock_md5(lock_payload, lock_dep_path)
        if recorded is not None:
            actual = compute_file_md5(training_table)
            if actual != recorded:
                raise RegistryPackagingError(
                    "dvc.lock hash does not match the training table on disk",
                    details={
                        "lock_path": lock_dep_path,
                        "lock_md5": recorded,
                        "file_md5": actual,
                    },
                )
            return recorded

    return compute_file_sha256(training_table)


def _find_lock_md5(lock_payload: Any, lock_dep_path: str) -> str | None:
    """Returns the recorded MD5 for a path anywhere in the DVC lock, if present."""
    if not isinstance(lock_payload, dict):
        return None
    for stage in lock_payload.get("stages", {}).values():
        if not isinstance(stage, dict):
            continue
        for section in ("outs", "deps"):
            for entry in stage.get(section) or []:
                if isinstance(entry, dict) and entry.get("path") == lock_dep_path:
                    recorded = entry.get("md5")
                    if isinstance(recorded, str) and recorded:
                        return recorded
    return None


def resolve_lineage(
    *,
    model_name: str,
    model_version: str,
    run_id: str,
    repo_root: Path,
    calibration_report_path: Path,
    dgp_manifest_path: Path,
    segment_count: int,
    validation_status: Literal["GATE_PASSED_OPTION_F"] = "GATE_PASSED_OPTION_F",
    created_at_utc: str | None = None,
) -> ModelLineageMetadata:
    """Assembles the complete provenance record for a candidate registration.

    Args:
        model_name: Registered model name the version belongs to.
        model_version: MLflow model version identifier.
        run_id: MLflow run that produced the version.
        repo_root: Repository root used for git and DVC provenance.
        calibration_report_path: The calibration report a reviewer will inspect.
        dgp_manifest_path: Stage 4 DGP manifest.
        segment_count: Number of segments packaged in the bundle.
        validation_status: Stage 10 gate status recorded on the version.
        created_at_utc: Override for the creation timestamp (tests).

    Returns:
        The validated lineage record.

    Raises:
        RegistryPackagingError: If git provenance, the DVC hash, the DGP manifest,
            or the calibration report cannot be resolved.
    """
    root = Path(repo_root)
    commit, dirty = resolve_git_commit(root)
    manifest = _load_json_object(dgp_manifest_path, label="DGP manifest")
    dgp_manifest_id = str(manifest.get("config_fingerprint") or "")
    if not dgp_manifest_id:
        raise RegistryPackagingError(
            "DGP manifest is missing 'config_fingerprint'",
            details={"path": str(dgp_manifest_path)},
        )
    if not Path(calibration_report_path).is_file():
        raise RegistryPackagingError(
            f"calibration report not found: {calibration_report_path}",
            details={"path": str(calibration_report_path)},
        )

    return ModelLineageMetadata(
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        git_commit=commit,
        git_dirty=dirty,
        dvc_data_hash=resolve_dvc_data_hash(
            root / "dvc.lock",
            root / DEFAULT_LOCK_DEP_PATH,
        ),
        dgp_manifest_id=dgp_manifest_id,
        calibration_hash=compute_file_sha256(Path(calibration_report_path)),
        validation_status=validation_status,
        segment_count=segment_count,
        created_at_utc=created_at_utc or datetime.now(UTC).isoformat(),
    )


def assert_no_estimator_artifacts(paths: Iterable[Path]) -> None:
    """Raises if any path is (or contains) a serialized estimator binary.

    Args:
        paths: Files or directories that will be (or were) published to serving.

    Raises:
        RegistrySecurityError: If a ``.pkl`` / ``.pickle`` / ``.joblib`` file is found.
    """
    offenders: list[str] = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_dir():
            candidates = [entry for entry in candidate.rglob("*") if entry.is_file()]
        elif candidate.is_file():
            candidates = [candidate]
        else:
            candidates = []
        offenders.extend(
            str(entry) for entry in candidates if entry.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES
        )
    if offenders:
        raise RegistrySecurityError(
            "serialized estimator artifacts are forbidden in the serving bundle",
            details={"offenders": sorted(offenders)},
        )


def build_serving_bundle(
    staging_dir: Path,
    *,
    segments: Tier1SegmentTable,
    calibration_report_path: Path,
    validation_summary_path: Path,
    sensitivity_summary_path: Path,
    lineage: ModelLineageMetadata,
) -> list[Path]:
    """Writes the five standard bundle files into a staging directory.

    The calibration report and both summary artifacts are copied byte-for-byte so
    that the recorded hashes describe exactly what a reviewer inspected.

    Args:
        staging_dir: Directory that receives the bundle.
        segments: Validated segment table to publish.
        calibration_report_path: Stage 5 GLM calibration report.
        validation_summary_path: Stage 10 final-seed validation summary.
        sensitivity_summary_path: Stage 11 sensitivity summary.
        lineage: Provenance record serialized as ``lineage.json``.

    Returns:
        The written bundle files, in ``STANDARD_BUNDLE_FILES`` order.

    Raises:
        RegistryPackagingError: If an input is missing or the bundle fails its own
            read-back validation.
        RegistrySecurityError: If an estimator binary exists in the staging directory.
    """
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    assert_no_estimator_artifacts([staging])

    copies = {
        "glm_calibration_report.json": Path(calibration_report_path),
        "causal_validation_summary.json": Path(validation_summary_path),
        "sensitivity_summary.json": Path(sensitivity_summary_path),
    }
    for name, source in copies.items():
        if not source.is_file():
            raise RegistryPackagingError(
                f"bundle input not found: {source}",
                details={"file": name, "path": str(source)},
            )

    write_segment_table(segments, staging / SEGMENT_TABLE_FILE)
    for name, source in copies.items():
        shutil.copyfile(source, staging / name)
    write_lineage_file(lineage, staging / LINEAGE_FILE)

    assert_no_estimator_artifacts([staging])
    expected = {staging / name for name in STANDARD_BUNDLE_FILES}
    present = {entry for entry in staging.iterdir() if entry.is_file()}
    unexpected = sorted(str(entry) for entry in present - expected)
    if unexpected:
        raise RegistrySecurityError(
            "unexpected files present in the serving bundle",
            details={"unexpected": unexpected},
        )

    read_segment_table(staging / SEGMENT_TABLE_FILE)
    return [staging / name for name in STANDARD_BUNDLE_FILES]
