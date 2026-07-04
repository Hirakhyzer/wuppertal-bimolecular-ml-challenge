"""Evaluation metrics and structured result-record helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error for regression targets and predictions."""
    return float(mean_absolute_error(np.asarray(y_true), np.asarray(y_pred)))


def save_result_record(record: dict[str, Any], path: str | Path) -> None:
    """Write one structured experiment record as JSON.

    Callers should include seed, split strategy, selected hyperparameters, feature
    mode, training count, validation MAE, and test MAE only when actually run.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    """Serialize NumPy scalar types used in result metadata."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")
