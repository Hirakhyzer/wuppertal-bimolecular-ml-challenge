"""Configuration loading and dataset-schema validation.

The CSV target and pair-identifier columns are deliberately unresolved by default.
They must be filled only after `scripts/inspect_data.py` has examined the real data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SCHEMA_KEYS = ("target_column", "pair_a_column", "pair_b_column")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML configuration file and return a plain dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration {path} must contain a YAML mapping.")
    return payload


def require_confirmed_schema(config: dict[str, Any]) -> dict[str, str]:
    """Return confirmed CSV mappings or raise before any model run.

    This guard prevents scripts from inventing a target column or geometry-pair
    identifier mapping based on file names or column order.
    """
    schema = config.get("schema", {})
    unresolved = [key for key in REQUIRED_SCHEMA_KEYS if not schema.get(key)]
    if unresolved:
        keys = ", ".join(unresolved)
        raise ValueError(
            "Dataset schema is unresolved. Run `python scripts/inspect_data.py`, "
            "confirm the real CSV columns, then fill configs/*.yaml. Missing: " + keys
        )
    return {key: str(schema[key]) for key in REQUIRED_SCHEMA_KEYS}


def project_root() -> Path:
    """Return the repository root based on this source file location."""
    return Path(__file__).resolve().parents[1]
