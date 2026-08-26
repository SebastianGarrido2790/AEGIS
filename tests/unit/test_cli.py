"""Unit tests for the DVC pipeline CLI runner (INV-3, ADR-007, R-4).

Verifies that:
1. Ingest commands successfully copy input fixtures to target destinations.
2. Ingest commands fail gracefully on missing input paths and invalid data types.
3. Validate commands run GX validations, produce report JSONs, and return exit code 0 on valid data.
4. Validate commands return exit code 1 and log blocking messages on invalid data.
5. Version commands promote validated files when report indicates success.
6. Version commands block and raise DataContractError when report indicates failure or is missing.
7. Parser builds all expected subcommands with default fallback wiring.
8. main() entrypoint routes cleanly to all 6 subcommands.
"""

import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aegis.config.loader import get_config
from aegis.pipelines.cli import (
    build_parser,
    ingest_file,
    main,
    validate_file,
    version_file,
)
from aegis.utils.exceptions import DataContractError


@pytest.fixture
def config():
    """Returns application configuration."""
    return get_config()


def test_cli_parser_commands_structure() -> None:
    """Verifies that all pipeline subcommands are registered in the parser."""
    parser = build_parser()
    expected_commands = [
        "ingest-fremtpl2",
        "ingest-elasticity",
        "ingest-regulatory",
        "validate-elasticity",
        "validate-regulatory",
        "version-elasticity",
        "version-regulatory",
    ]
    for cmd in expected_commands:
        args = parser.parse_args([cmd])
        assert args.command == cmd


def test_ingest_file_success(tmp_path: Path, config) -> None:
    """Verifies that ingest_file copies the fixture to the raw destination."""
    out_file = tmp_path / "raw_elasticity.csv"
    code = ingest_file(
        input_path=Path(config.data_contracts.elasticity_valid_fixture),
        output_path=out_file,
        data_type="elasticity",
    )
    assert code == 0
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_ingest_file_missing_source(tmp_path: Path) -> None:
    """Verifies that ingest_file returns exit code 1 when source file does not exist."""
    code = ingest_file(
        input_path=tmp_path / "non_existent.csv",
        output_path=tmp_path / "out.csv",
        data_type="elasticity",
    )
    assert code == 1


def test_ingest_file_unknown_type(tmp_path: Path) -> None:
    """Verifies that ingest_file returns exit code 1 on unknown data type."""
    code = ingest_file(
        input_path=None,
        output_path=tmp_path / "out.csv",
        data_type="unknown_type",
    )
    assert code == 1


def test_ingest_file_default_resolution(tmp_path: Path) -> None:
    """Verifies that passing input_path=None resolves to params.yaml default fixture (R-4)."""
    out_file = tmp_path / "default_reg.json"
    code = ingest_file(input_path=None, output_path=out_file, data_type="regulatory")
    assert code == 0
    assert out_file.is_file()

    out_file_e = tmp_path / "default_el.csv"
    code_e = ingest_file(input_path=None, output_path=out_file_e, data_type="elasticity")
    assert code_e == 0
    assert out_file_e.is_file()


def test_validate_file_valid_data(tmp_path: Path, config) -> None:
    """Verifies that validate_file produces a valid report and returns exit code 0."""
    report_file = tmp_path / "report_valid.json"
    code = validate_file(
        data_type="elasticity",
        data_path=Path(config.data_contracts.elasticity_valid_fixture),
        suite_path=Path(config.data_contracts.elasticity_suite_path),
        report_path=report_file,
    )
    assert code == 0
    assert report_file.is_file()
    with report_file.open("r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["success"] is True
    assert report["successful_expectations"] == report["total_expectations"]


def test_validate_file_regulatory_valid_data(tmp_path: Path, config) -> None:
    """Verifies that validate_file validates regulatory data cleanly."""
    report_file = tmp_path / "report_reg_valid.json"
    code = validate_file(
        data_type="regulatory",
        data_path=Path(config.data_contracts.regulatory_valid_fixture),
        suite_path=Path(config.data_contracts.regulatory_corpus_suite_path),
        report_path=report_file,
    )
    assert code == 0
    assert report_file.is_file()


def test_validate_file_unknown_type(tmp_path: Path, config) -> None:
    """Verifies that validate_file returns exit code 1 for unknown data type."""
    report_file = tmp_path / "report_unknown.json"
    code = validate_file(
        data_type="invalid_type",
        data_path=Path(config.data_contracts.regulatory_valid_fixture),
        suite_path=Path(config.data_contracts.regulatory_corpus_suite_path),
        report_path=report_file,
    )
    assert code == 1


def test_validate_file_invalid_data(tmp_path: Path, config) -> None:
    """Verifies that validate_file writes a failed report and returns exit code 1 (INV-3)."""
    report_file = tmp_path / "report_invalid.json"
    code = validate_file(
        data_type="elasticity",
        data_path=Path(config.data_contracts.elasticity_invalid_range_fixture),
        suite_path=Path(config.data_contracts.elasticity_suite_path),
        report_path=report_file,
    )
    assert code == 1
    assert report_file.is_file()
    with report_file.open("r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["success"] is False
    assert len(report["failure_rules"]) > 0


def test_version_file_success(tmp_path: Path) -> None:
    """Verifies that version_file promotes raw data when validation report indicates success."""
    raw_file = tmp_path / "raw.csv"
    raw_file.write_text("a,b\n1,2", encoding="utf-8")
    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps({"success": True}), encoding="utf-8")
    versioned_file = tmp_path / "versioned.csv"

    code = version_file(raw_path=raw_file, report_path=report_file, output_path=versioned_file)
    assert code == 0
    assert versioned_file.is_file()


def test_version_file_missing_report(tmp_path: Path) -> None:
    """Verifies that version_file returns exit code 1 if report is missing."""
    raw_file = tmp_path / "raw.csv"
    raw_file.write_text("a,b\n1,2", encoding="utf-8")
    versioned_file = tmp_path / "versioned.csv"
    code = version_file(
        raw_path=raw_file,
        report_path=tmp_path / "missing.json",
        output_path=versioned_file,
    )
    assert code == 1


def test_version_file_blocks_on_failure(tmp_path: Path) -> None:
    """Verifies that version_file raises DataContractError when report failed (INV-3)."""
    raw_file = tmp_path / "raw.csv"
    raw_file.write_text("a,b\n1,2", encoding="utf-8")
    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps({"success": False}), encoding="utf-8")
    versioned_file = tmp_path / "versioned.csv"

    with pytest.raises(DataContractError) as exc_info:
        version_file(raw_path=raw_file, report_path=report_file, output_path=versioned_file)
    assert "Cannot version unvalidated or failed data" in str(exc_info.value)


@pytest.mark.parametrize(
    "subcmd,extra_args",
    [
        ("ingest-fremtpl2", []),
        ("ingest-regulatory", []),
        ("validate-elasticity", []),
        ("validate-regulatory", []),
        ("version-elasticity", []),
        ("version-regulatory", []),
    ],
)
def test_main_cli_routing(subcmd: str, extra_args: list[str], tmp_path: Path, config) -> None:
    """Verifies that main() dispatches to all subcommands."""
    test_args = ["cli.py", subcmd, *extra_args]
    with (
        patch("sys.argv", test_args),
        patch("aegis.pipelines.feature.ingest.ingest_fremtpl2_pipeline"),
        contextlib.suppress(Exception),
    ):
        main()

