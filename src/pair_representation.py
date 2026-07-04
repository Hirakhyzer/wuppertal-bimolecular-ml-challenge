"""Directed pair-feature construction and reverse-pair symmetry diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


PAIR_FEATURE_OPTIONS = ("directed_concatenation", "interaction_features")


def build_pair_features(representation_a: np.ndarray, representation_b: np.ndarray, mode: str) -> np.ndarray:
    """Build directed bi-molecular features from equally sized molecular vectors.

    No feature mode here assumes that switching A and B preserves the target.
    """
    a = np.asarray(representation_a, dtype=float)
    b = np.asarray(representation_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("A and B representations must have the same shape.")
    if a.ndim == 1:
        a, b = a[None, :], b[None, :]
        squeeze = True
    elif a.ndim == 2:
        squeeze = False
    else:
        raise ValueError("Representations must be one- or two-dimensional.")
    if mode == "directed_concatenation":
        output = np.concatenate([a, b], axis=1)
    elif mode == "interaction_features":
        output = np.concatenate([a, b, np.abs(a - b), a * b], axis=1)
    else:
        raise ValueError(f"Unknown pair feature mode {mode!r}; choose from {PAIR_FEATURE_OPTIONS}.")
    return output[0] if squeeze else output


def reverse_pair_symmetry_report(
    table: pd.DataFrame,
    pair_a_column: str,
    pair_b_column: str,
    target_column: str,
) -> dict[str, float | int | None]:
    """Compare targets for observed reverse pairs without assuming symmetry.

    The report is diagnostic only. It quantifies rows for which both `(A,B)` and
    `(B,A)` exist and summarizes their target differences. Users must interpret
    any result with the actual target units and experimental context.
    """
    required = {pair_a_column, pair_b_column, target_column}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Missing columns for reverse-pair diagnostic: {sorted(missing)}")
    working = table[[pair_a_column, pair_b_column, target_column]].copy()
    working[target_column] = pd.to_numeric(working[target_column], errors="raise")
    reverse = working.rename(columns={pair_a_column: pair_b_column, pair_b_column: pair_a_column, target_column: "reverse_target"})
    joined = working.merge(reverse, on=[pair_a_column, pair_b_column], how="inner")
    differences = np.abs(joined[target_column] - joined["reverse_target"])
    if joined.empty:
        return {"matched_reverse_rows": 0, "mean_absolute_target_difference": None, "max_absolute_target_difference": None}
    return {
        "matched_reverse_rows": int(len(joined)),
        "mean_absolute_target_difference": float(differences.mean()),
        "max_absolute_target_difference": float(differences.max()),
    }
