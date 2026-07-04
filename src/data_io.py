"""Raw-file discovery and dataset inspection utilities.

These functions inspect actual local files and avoid assigning semantic meaning to
CSV columns until the user supplies confirmed names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_REQUIRED_FILES = ("Coord_A.xyz", "Coord_B.xyz", "CouplingEnergies.csv")
OPTIONAL_FILES = ("Coord_supermol.xyz",)


def expected_paths(raw_dir: str | Path) -> dict[str, Path]:
    """Return canonical paths for expected local raw files."""
    directory = Path(raw_dir)
    return {name: directory / name for name in (*EXPECTED_REQUIRED_FILES, *OPTIONAL_FILES)}


def validate_raw_inputs(raw_dir: str | Path) -> dict[str, Path]:
    """Ensure required challenge files exist, leaving the optional file optional."""
    paths = expected_paths(raw_dir)
    missing = [name for name in EXPECTED_REQUIRED_FILES if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required local data files: " + ", ".join(missing) + ". "
            "Place them in data/raw/ as documented in data/README.md."
        )
    return paths


def inspect_csv(path: str | Path, target_column: str | None = None) -> dict[str, Any]:
    """Read CSV structure and non-semantic quality summaries.

    If `target_column` is confirmed by the user, include a target distribution.
    Otherwise, summarize all numeric candidates without selecting one.
    """
    table = pd.read_csv(path)
    numeric_columns = table.select_dtypes(include=np.number).columns.tolist()
    numeric_summary = {
        column: {
            "count": int(table[column].count()),
            "mean": _finite_or_none(table[column].mean()),
            "std": _finite_or_none(table[column].std()),
            "min": _finite_or_none(table[column].min()),
            "max": _finite_or_none(table[column].max()),
        }
        for column in numeric_columns
    }
    summary: dict[str, Any] = {
        "rows": int(len(table)),
        "columns": table.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in table.dtypes.items()},
        "missing_values": {column: int(value) for column, value in table.isna().sum().items()},
        "duplicate_rows": int(table.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "numeric_column_summary": numeric_summary,
        "target_distribution": None,
    }
    if target_column is not None:
        if target_column not in table.columns:
            raise KeyError(f"Confirmed target column is not present: {target_column}")
        values = pd.to_numeric(table[target_column], errors="raise")
        summary["target_distribution"] = {
            "column": target_column,
            "count": int(values.count()),
            "missing": int(values.isna().sum()),
            "mean": _finite_or_none(values.mean()),
            "std": _finite_or_none(values.std()),
            "min": _finite_or_none(values.min()),
            "max": _finite_or_none(values.max()),
            "quantiles": {str(q): _finite_or_none(values.quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
        }
    return summary


def inspect_pair_combinations(
    table: pd.DataFrame,
    pair_a_column: str,
    pair_b_column: str,
) -> dict[str, Any]:
    """Describe observed pair identifiers without assuming completeness.

    The result reports observed distinct pairs, duplicated pair rows, and the
    Cartesian-product size implied by observed A and B identifiers. It does not
    claim that a full Cartesian product is scientifically expected.
    """
    for column in (pair_a_column, pair_b_column):
        if column not in table.columns:
            raise KeyError(f"Pair identifier column is not present: {column}")
    pair_table = table[[pair_a_column, pair_b_column]]
    a_count = int(pair_table[pair_a_column].nunique(dropna=True))
    b_count = int(pair_table[pair_b_column].nunique(dropna=True))
    observed = int(len(pair_table.drop_duplicates()))
    possible = a_count * b_count
    return {
        "pair_a_column": pair_a_column,
        "pair_b_column": pair_b_column,
        "unique_a_identifiers": a_count,
        "unique_b_identifiers": b_count,
        "observed_unique_pairs": observed,
        "duplicate_pair_rows": int(pair_table.duplicated().sum()),
        "cartesian_product_size_from_observed_ids": possible,
        "observed_pair_fraction_of_cartesian_product": (observed / possible) if possible else None,
    }


def write_json_summary(summary: dict[str, Any], destination: str | Path) -> None:
    """Write inspection output as deterministic human-readable JSON."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _finite_or_none(value: Any) -> float | None:
    """Convert finite numeric values to floats; use None for NaN or infinity."""
    if value is None or not np.isfinite(value):
        return None
    return float(value)
