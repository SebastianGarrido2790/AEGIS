"""Isolated truth sidecar table loader (ADR-013, P2-D2).

This module provides the dedicated, validated loader for the ground truth sidecar table.
Under ADR-013, this module is strictly quarantined from causal estimators and training
pipelines, and may only be consumed by offline validation harnesses and benchmarks.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from aegis.config.loader import get_config
from aegis.schemas.dgp import TruthSidecarRow
from aegis.utils.exceptions import TruthIsolationError

REQUIRED_TRUTH_COLUMNS = (
    "PolicyID",
    "baseline_renewal_q0",
    "true_tau",
    "true_epsilon",
    "segment_id",
)


def load_truth_sidecar(
    path: Path | str | None = None,
    config: Any | None = None,
    validate_rows: bool = False,
) -> pd.DataFrame:
    """Loads and validates the ground truth sidecar table.

    Args:
        path: Path to truth_sidecar.csv. If None, resolved from params.yaml.
        config: AEGISConfig instance. If None, loaded via get_config().
        validate_rows: If True, validates every row against TruthSidecarRow Pydantic schema.

    Returns:
        pd.DataFrame: Validated ground truth table.

    Raises:
        TruthIsolationError: If the file does not exist, has missing columns, or invalid values.
    """
    if path is None:
        if config is None:
            config = get_config()
        resolved_path = config.tier1_ml.dgp_paths.truth_sidecar
    else:
        resolved_path = path

    truth_file = Path(resolved_path)
    if not truth_file.is_file():
        raise TruthIsolationError(
            f"Ground truth sidecar file not found: {truth_file}",
            details={"path": str(truth_file)},
        )

    try:
        df = pd.read_csv(truth_file)
    except Exception as exc:
        raise TruthIsolationError(
            f"Failed to read truth sidecar file: {exc}",
            details={"path": str(truth_file), "error": str(exc)},
        ) from exc

    # Column validation
    missing_cols = [c for c in REQUIRED_TRUTH_COLUMNS if c not in df.columns]
    if missing_cols:
        raise TruthIsolationError(
            f"Truth sidecar is missing required columns: {missing_cols}",
            details={"missing_columns": missing_cols, "present_columns": list(df.columns)},
        )

    # Range and sanity validation
    if len(df) == 0:
        raise TruthIsolationError("Truth sidecar table is empty")

    q0_values = df["baseline_renewal_q0"]
    if bool((q0_values < 0.0).any()) or bool((q0_values > 1.0).any()):
        raise TruthIsolationError("baseline_renewal_q0 values must be in [0.0, 1.0]")

    if not bool(df["true_tau"].notnull().all()):
        raise TruthIsolationError("true_tau contains null or NaN values")

    if not bool(df["true_epsilon"].notnull().all()):
        raise TruthIsolationError("true_epsilon contains null or NaN values")

    if validate_rows:
        for _idx, row in df.head(100).iterrows():
            TruthSidecarRow.model_validate(row.to_dict())

    return df
