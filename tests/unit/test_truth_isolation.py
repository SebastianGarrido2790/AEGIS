"""Falsification and import-boundary enforcement tests for truth isolation (ADR-013, P2-D2).

Falsification 1: Proves dynamically that planting truth columns (tau_ground_truth, tau,
epsilon, etc.) into the training table causes the Great Expectations contract to fail
loudly and halt the pipeline.

Falsification 2: Proves via structural AST scanning that modules in causal estimator
packages cannot import from aegis.pipelines.truth_loader.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

from aegis.config.loader import get_config
from aegis.pipelines.data_contracts import validate_dataframe, validate_elasticity_data
from aegis.utils.exceptions import DataContractError, TruthIsolationError


def check_module_import_boundary(source_code: str, module_path: str = "<memory>") -> None:
    """Scans python source code using AST to verify no forbidden imports of truth_loader exist.

    Args:
        source_code: Python source code string.
        module_path: File path or identifier for diagnostic error reporting.

    Raises:
        TruthIsolationError: If an import of aegis.pipelines.truth_loader is detected.
    """
    tree = ast.parse(source_code, filename=module_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "truth_loader" in alias.name:
                    raise TruthIsolationError(
                        f"Forbidden import of truth_loader detected in {module_path}: "
                        f"import {alias.name}",
                        details={"file": module_path, "import": alias.name},
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if "truth_loader" in mod:
                raise TruthIsolationError(
                    f"Forbidden import from truth_loader detected in {module_path}: "
                    f"from {mod} import ...",
                    details={"file": module_path, "module": mod},
                )
            for alias in node.names:
                if "truth_loader" in alias.name:
                    raise TruthIsolationError(
                        f"Forbidden import of truth_loader detected in {module_path}: "
                        f"from {mod} import {alias.name}",
                        details={"file": module_path, "import": alias.name},
                    )


# ==========================================
# Falsification 1: Training Data Contract Leakage Defense
# ==========================================


def test_falsification_planted_truth_column_fails_contract() -> None:
    """Falsification test 1: Planted tau_ground_truth fails GX suite loudly (Gate 6)."""
    config = get_config()
    with pytest.raises(DataContractError) as exc_info:
        validate_elasticity_data(
            data_path=config.data_contracts.elasticity_invalid_leakage_fixture,
            suite_path=config.data_contracts.elasticity_suite_path,
            raise_on_failure=True,
        )
    error_msg = str(exc_info.value)
    assert "Data contract validation failed" in error_msg
    assert "expect_no_post_treatment_leakage" in error_msg
    details = exc_info.value.details
    assert "failed_expectations" in details
    leakage = [
        f for f in details["failed_expectations"]
        if f["expectation_type"] == "expect_no_post_treatment_leakage"
    ]
    assert len(leakage) == 1
    assert "tau_ground_truth" in leakage[0]["result"]["details"]["leaked_columns"]


@pytest.mark.parametrize(
    "forbidden_col",
    [
        "tau",
        "epsilon",
        "q0",
        "true_effect",
        "tau_ground_truth",
        "fitted_loss_cost",
        "baseline_renewal_q0",
        "true_tau",
        "true_epsilon",
    ],
)
def test_falsification_dynamic_truth_columns_rejected(forbidden_col: str) -> None:
    """Verifies that each individual truth column variant is dynamically rejected."""
    config = get_config()
    df_valid = pd.read_csv(config.data_contracts.elasticity_valid_fixture)
    df_leakage = df_valid.copy()
    df_leakage[forbidden_col] = -0.25

    with pytest.raises(DataContractError) as exc_info:
        validate_dataframe(
            df=df_leakage,
            suite_path=config.data_contracts.elasticity_suite_path,
            raise_on_failure=True,
        )
    assert "expect_no_post_treatment_leakage" in str(exc_info.value)


# ==========================================
# Falsification 2: Import-Boundary Enforcement
# ==========================================


OFFLINE_VALIDATION_MODULES = frozenset({"validation_harness.py", "final_seed_validation.py"})


def test_import_boundary_existing_codebase() -> None:
    """Asserts that only the offline validation modules import truth_loader."""
    src_dir = Path("src/aegis")
    for py_file in src_dir.rglob("*.py"):
        if py_file.name in ("truth_loader.py", "__init__.py", *OFFLINE_VALIDATION_MODULES):
            continue
        source = py_file.read_text(encoding="utf-8")
        # Should not raise TruthIsolationError
        check_module_import_boundary(source, str(py_file))


def test_falsification_import_boundary_catches_injected_import() -> None:
    """Falsification test 2: Injected import of truth_loader fails boundary check."""
    bad_code_1 = "import aegis.pipelines.truth_loader\n\ndef fit(): pass"
    with pytest.raises(TruthIsolationError) as exc_info1:
        check_module_import_boundary(bad_code_1, "src/aegis/pipelines/causal/estimator.py")
    assert "Forbidden import of truth_loader detected" in str(exc_info1.value)

    bad_code_2 = "from aegis.pipelines.truth_loader import load_truth_sidecar\n\ndef fit(): pass"
    with pytest.raises(TruthIsolationError) as exc_info2:
        check_module_import_boundary(bad_code_2, "src/aegis/pipelines/causal/estimator.py")
    assert "Forbidden import from truth_loader detected" in str(exc_info2.value)
