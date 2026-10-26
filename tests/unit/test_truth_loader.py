"""Unit tests for the isolated truth loader module (P2-F04, ADR-013)."""

from pathlib import Path

import pandas as pd
import pytest

from aegis.pipelines.truth_loader import load_truth_sidecar
from aegis.utils.exceptions import TruthIsolationError


def test_load_truth_sidecar_valid(tmp_path: Path) -> None:
    csv_file = tmp_path / "truth_sidecar.csv"
    df_valid = pd.DataFrame(
        {
            "PolicyID": [1, 2, 3],
            "baseline_renewal_q0": [0.80, 0.82, 0.85],
            "true_tau": [-0.20, -0.25, -0.22],
            "true_epsilon": [-0.25, -0.30, -0.26],
            "segment_id": ["S1", "S2", "S3"],
        }
    )
    df_valid.to_csv(csv_file, index=False)

    loaded = load_truth_sidecar(path=csv_file, validate_rows=True)
    assert len(loaded) == 3
    assert list(loaded.columns) == [
        "PolicyID",
        "baseline_renewal_q0",
        "true_tau",
        "true_epsilon",
        "segment_id",
    ]


def test_load_truth_sidecar_missing_file() -> None:
    with pytest.raises(TruthIsolationError) as exc_info:
        load_truth_sidecar(path="non_existent_file.csv")
    assert "Ground truth sidecar file not found" in str(exc_info.value)


def test_load_truth_sidecar_missing_columns(tmp_path: Path) -> None:
    csv_file = tmp_path / "bad_truth.csv"
    df_bad = pd.DataFrame({"PolicyID": [1], "true_tau": [-0.2]})
    df_bad.to_csv(csv_file, index=False)

    with pytest.raises(TruthIsolationError) as exc_info:
        load_truth_sidecar(path=csv_file)
    assert "Truth sidecar is missing required columns" in str(exc_info.value)


def test_load_truth_sidecar_invalid_q0_range(tmp_path: Path) -> None:
    csv_file = tmp_path / "invalid_q0.csv"
    df_bad = pd.DataFrame(
        {
            "PolicyID": [1],
            "baseline_renewal_q0": [1.5],  # Invalid > 1.0
            "true_tau": [-0.2],
            "true_epsilon": [-0.25],
            "segment_id": ["S1"],
        }
    )
    df_bad.to_csv(csv_file, index=False)

    with pytest.raises(TruthIsolationError) as exc_info:
        load_truth_sidecar(path=csv_file)
    assert "baseline_renewal_q0 values must be in [0.0, 1.0]" in str(exc_info.value)
