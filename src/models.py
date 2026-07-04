"""Kernel Ridge Regression model construction and validation-only selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class KRRSelection:
    """Selected KRR hyperparameters and validation MAE."""

    alpha: float
    gamma: float
    validation_mae: float


def build_krr_pipeline(alpha: float, gamma: float) -> Pipeline:
    """Create the required StandardScaler plus RBF Kernel Ridge Regression pipeline."""
    if alpha <= 0 or gamma <= 0:
        raise ValueError("KRR alpha and gamma must both be positive.")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf", alpha=float(alpha), gamma=float(gamma))),
    ])


def select_krr_hyperparameters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    alphas: Iterable[float],
    gammas: Iterable[float],
) -> KRRSelection:
    """Select KRR settings only by validation MAE, never held-out test MAE."""
    best: KRRSelection | None = None
    for alpha in alphas:
        for gamma in gammas:
            model = build_krr_pipeline(float(alpha), float(gamma))
            model.fit(x_train, y_train)
            mae = float(mean_absolute_error(y_validation, model.predict(x_validation)))
            candidate = KRRSelection(float(alpha), float(gamma), mae)
            if best is None or candidate.validation_mae < best.validation_mae:
                best = candidate
    if best is None:
        raise ValueError("At least one alpha and gamma candidate are required.")
    return best


def fit_selected_krr(x_train: np.ndarray, y_train: np.ndarray, selection: KRRSelection) -> Pipeline:
    """Fit one final KRR pipeline using already selected hyperparameters."""
    model = build_krr_pipeline(selection.alpha, selection.gamma)
    return model.fit(x_train, y_train)


def build_optional_gpr_comparison(*args, **kwargs):
    """Reserved extension point for a future Gaussian Process comparison.

    GPR is intentionally not enabled in the first scaffold because its kernel
    scaling and uncertainty protocol should be designed after the real dataset
    dimensions and target units have been inspected.
    """
    raise NotImplementedError("Gaussian Process comparison is a documented future extension.")
