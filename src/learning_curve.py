"""Repeated validation learning-curve utilities for computationally bounded KRR."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


def feasible_log_sizes(
    requested_sizes: Iterable[int],
    available_train_samples: int,
    max_krr_train_samples: int,
) -> list[int]:
    """Return sorted unique feasible training sizes without exceeding KRR limits."""
    ceiling = min(int(available_train_samples), int(max_krr_train_samples))
    sizes = sorted({int(size) for size in requested_sizes if 1 <= int(size) <= ceiling})
    if not sizes:
        raise ValueError("No requested learning-curve sizes fit available/KRR-limited training data.")
    return sizes


def repeated_learning_curve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    training_sizes: Iterable[int],
    seeds: Iterable[int],
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """Run reproducible repeated subsampling and return validation-MAE summary.

    `fit_predict` must fit exclusively on its supplied training subset and return
    predictions for the supplied validation features. Hyperparameters should be
    preselected or selected inside the training subset only.
    """
    rows: list[dict[str, float | int]] = []
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    for size in training_sizes:
        if size > len(x_train):
            raise ValueError(f"Training size {size} exceeds available samples {len(x_train)}.")
        for seed in seeds:
            generator = np.random.default_rng(int(seed))
            subset = generator.choice(len(x_train), size=int(size), replace=False)
            prediction = fit_predict(x_train[subset], y_train[subset], x_validation)
            mae = float(np.mean(np.abs(np.asarray(y_validation) - np.asarray(prediction))))
            rows.append({"training_size": int(size), "seed": int(seed), "validation_mae": mae})
    raw = pd.DataFrame(rows)
    summary = raw.groupby("training_size", as_index=False).agg(
        validation_mae_mean=("validation_mae", "mean"),
        validation_mae_std=("validation_mae", "std"),
        repetitions=("seed", "count"),
    )
    return summary
