#!/usr/bin/env python3
"""Module size checker for AEGIS (INV-8 enforcement).

Verifies that no Python module under `src/` exceeds the 1,000-line ceiling.
This is a design-quality constraint that keeps units of logic testable in
isolation and prevents monster modules.
"""

import argparse
import sys
from pathlib import Path


def count_file_lines(file_path: Path) -> int:
    """Counts the total physical lines in a file.

    Args:
        file_path: Path to the target file.

    Returns:
        int: Total number of lines.
    """
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def check_module_sizes(target_dir: Path, max_lines: int = 1000) -> int:
    """Walks the directory and validates file line counts.

    Args:
        target_dir: Base directory to scan.
        max_lines: Maximum allowable lines per file (default: 1,000).

    Returns:
        int: 0 if all files comply with max_lines ceiling, 1 if any file violates it.
    """
    if not target_dir.exists():
        print(f"[ERROR] Target directory not found: {target_dir}")
        return 1

    py_files = sorted(target_dir.rglob("*.py"))
    if not py_files:
        print(f"[INFO] No Python files found in {target_dir}")
        return 0

    violations: list[tuple[Path, int]] = []
    print(f"--- AEGIS Module Size Check (INV-8: Max {max_lines} lines) ---")
    print(f"Scanning directory: {target_dir.resolve()} ({len(py_files)} python files)")

    for py_file in py_files:
        line_count = count_file_lines(py_file)
        base_dir = target_dir.parent if target_dir.parent.exists() else target_dir
        rel_path = py_file.relative_to(base_dir)

        if line_count >= max_lines:
            violations.append((rel_path, line_count))
            print(f"  [FAIL] {rel_path}: {line_count:,} lines (exceeds {max_lines} limit)")
        else:
            print(f"  [PASS] {rel_path}: {line_count:,} lines")

    print("-" * 60)
    if violations:
        print(
            f"[ERROR] INV-8 VIOLATION: {len(violations)} file(s) "
            f"exceeded {max_lines}-line ceiling."
        )
        for path, count in violations:
            print(f"  - {path}: {count} lines")
        return 1

    print(f"[SUCCESS] All {len(py_files)} Python file(s) are within the {max_lines}-line ceiling.")
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Enforces INV-8: No module under src/ exceeds 1,000 lines."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("src"),
        help="Directory to scan (default: src)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=1000,
        help="Maximum lines allowed per module (default: 1000)",
    )
    args = parser.parse_args()

    return check_module_sizes(target_dir=args.target_dir, max_lines=args.max_lines)


if __name__ == "__main__":
    sys.exit(main())
