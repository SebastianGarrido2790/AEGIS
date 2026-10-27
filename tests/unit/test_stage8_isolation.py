"""Structural isolation checks for Stage 8 serving and truth boundaries (INV-4, ADR-013)."""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src" / "aegis"
API_ROOT = SRC_ROOT / "api"
ESTIMATOR_MODULES = (
    SRC_ROOT / "pipelines" / "causal_estimator.py",
    SRC_ROOT / "pipelines" / "causal_artifacts.py",
)

FORBIDDEN_SERVING_ROOTS = ("econml", "joblib", "pickle")
FORBIDDEN_SERVING_MODULES = (
    "aegis.pipelines.causal_estimator",
    "aegis.pipelines.causal_artifacts",
)
FORBIDDEN_ESTIMATOR_MODULES = (
    "aegis.api",
    "aegis.pipelines.truth_loader",
)


def _imported_modules(source: str, label: str = "<memory>") -> set[str]:
    """Collects absolute module names from import statements in Python source."""
    tree = ast.parse(source, filename=label)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _matches(module: str, forbidden: tuple[str, ...]) -> bool:
    """Returns whether a module equals or is nested under a forbidden name."""
    return any(module == item or module.startswith(f"{item}.") for item in forbidden)


def _forbidden_imports(
    source: str, label: str, forbidden: tuple[str, ...]
) -> list[str]:
    """Lists imports in source that violate a forbidden module set."""
    return [
        module
        for module in sorted(_imported_modules(source, label))
        if _matches(module, forbidden)
    ]


def test_serving_modules_do_not_import_estimator_stack() -> None:
    api_files = sorted(API_ROOT.rglob("*.py"))
    assert api_files, "expected API modules to scan"

    violations: list[str] = []
    for path in api_files:
        source = path.read_text(encoding="utf-8")
        violations.extend(
            f"{path.relative_to(PROJECT_ROOT)} imports {module}"
            for module in _forbidden_imports(
                source,
                str(path),
                FORBIDDEN_SERVING_ROOTS + FORBIDDEN_SERVING_MODULES,
            )
        )
    assert not violations, "\n".join(violations)


def test_serving_scanner_rejects_injected_estimator_import() -> None:
    source = "from econml.dml import CausalForestDML\nimport joblib\n"
    assert _forbidden_imports(
        source, "api/injected.py", FORBIDDEN_SERVING_ROOTS + FORBIDDEN_SERVING_MODULES
    ) == ["econml.dml", "joblib"]


def test_estimator_modules_do_not_import_truth_or_serving_layers() -> None:
    violations: list[str] = []
    for path in ESTIMATOR_MODULES:
        assert path.is_file()
        source = path.read_text(encoding="utf-8")
        violations.extend(
            f"{path.name} imports {module}"
            for module in _forbidden_imports(
                source, str(path), FORBIDDEN_ESTIMATOR_MODULES
            )
        )
    assert not violations, "\n".join(violations)


def test_estimator_sources_have_no_truth_sidecar_access() -> None:
    for path in ESTIMATOR_MODULES:
        source = path.read_text(encoding="utf-8")
        assert "truth_sidecar" not in source
        assert "truth_loader" not in source
