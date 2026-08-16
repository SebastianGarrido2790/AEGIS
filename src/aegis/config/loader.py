"""AEGIS configuration loader.

Loads and validates params.yaml into strongly typed Pydantic models.
Raises ConfigurationError if the file is missing, invalid YAML, or fails schema validation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from aegis.config.schema import AEGISConfig
from aegis.utils.exceptions import ConfigurationError


def find_project_root() -> Path:
    """Locates the project root directory containing params.yaml."""
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "params.yaml").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_params(config_path: Path | str | None = None) -> AEGISConfig:
    """Loads and validates configuration from a YAML file.

    Args:
        config_path: Optional explicit path to configuration YAML file.
            Defaults to params.yaml in the project root.

    Returns:
        AEGISConfig: Validated strongly typed configuration instance.

    Raises:
        ConfigurationError: If the file is missing, unparseable, or fails schema validation.
    """
    if config_path is None:
        resolved_path = find_project_root() / "params.yaml"
    else:
        resolved_path = Path(config_path)

    if not resolved_path.is_file():
        raise ConfigurationError(
            f"Configuration file not found: {resolved_path}",
            details={"config_path": str(resolved_path)},
        )

    try:
        with resolved_path.open("r", encoding="utf-8") as f:
            raw_data: Any = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse YAML configuration: {exc}",
            details={"config_path": str(resolved_path), "error": str(exc)},
        ) from exc
    except Exception as exc:
        raise ConfigurationError(
            f"Failed to read configuration file: {exc}",
            details={"config_path": str(resolved_path), "error": str(exc)},
        ) from exc

    if not isinstance(raw_data, dict):
        raise ConfigurationError(
            "Configuration file must contain a top-level mapping/dictionary.",
            details={"config_path": str(resolved_path), "parsed_type": type(raw_data).__name__},
        )

    try:
        return AEGISConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Configuration validation failed against schema: {exc}",
            details={"config_path": str(resolved_path), "errors": exc.errors()},
        ) from exc


@lru_cache(maxsize=1)
def get_config() -> AEGISConfig:
    """Returns a cached singleton instance of the validated AEGIS configuration."""
    return load_params()


def clear_config_cache() -> None:
    """Clears the cached configuration singleton (useful for testing)."""
    get_config.cache_clear()
