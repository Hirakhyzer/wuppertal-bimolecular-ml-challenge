"""Leakage-aware random-pair and geometry-aware data split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    """Integer indices assigned to training, validation, test, and dropped rows."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    dropped: np.ndarray


def random_pair_split(
    n_samples: int,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 2026,
) -> SplitIndices:
    """Create a standard reproducible random split over pair rows."""
    _validate_fractions(validation_fraction, test_fraction)
    if n_samples < 3:
        raise ValueError("At least three pair rows are required for train/validation/test splitting.")
    generator = np.random.default_rng(seed)
    order = generator.permutation(n_samples)
    n_test = max(1, int(round(n_samples * test_fraction)))
    n_validation = max(1, int(round(n_samples * validation_fraction)))
    if n_test + n_validation >= n_samples:
        raise ValueError("Split fractions leave no training rows.")
    return SplitIndices(
        train=np.sort(order[n_test + n_validation:]),
        validation=np.sort(order[n_test:n_test + n_validation]),
        test=np.sort(order[:n_test]),
        dropped=np.asarray([], dtype=int),
    )


def geometry_aware_split(
    geometry_a_ids: np.ndarray,
    geometry_b_ids: np.ndarray,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 2026,
) -> SplitIndices:
    """Create a strict held-out-geometry split for pair data.

    Unique geometry identifiers are assigned to train, validation, or test groups.
    A pair is retained only when both member identifiers belong to the same group;
    cross-group pairs are placed in `dropped`. This makes test geometry exclusion
    explicit at the cost of reduced sample count.
    """
    _validate_fractions(validation_fraction, test_fraction)
    a = np.asarray(geometry_a_ids)
    b = np.asarray(geometry_b_ids)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Geometry ID arrays must be one-dimensional and equally sized.")
    unique_ids = np.unique(np.concatenate([a, b]))
    if len(unique_ids) < 3:
        raise ValueError("At least three unique geometry identifiers are required.")
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(unique_ids)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_validation = max(1, int(round(len(shuffled) * validation_fraction)))
    if n_test + n_validation >= len(shuffled):
        raise ValueError("Geometry split fractions leave no training geometry identifiers.")
    test_ids = set(shuffled[:n_test])
    validation_ids = set(shuffled[n_test:n_test + n_validation])
    train_ids = set(shuffled[n_test + n_validation:])
    train, validation, test, dropped = [], [], [], []
    for index, (left, right) in enumerate(zip(a, b)):
        if left in train_ids and right in train_ids:
            train.append(index)
        elif left in validation_ids and right in validation_ids:
            validation.append(index)
        elif left in test_ids and right in test_ids:
            test.append(index)
        else:
            dropped.append(index)
    split = SplitIndices(
        train=np.asarray(train, dtype=int),
        validation=np.asarray(validation, dtype=int),
        test=np.asarray(test, dtype=int),
        dropped=np.asarray(dropped, dtype=int),
    )
    if min(len(split.train), len(split.validation), len(split.test)) == 0:
        raise ValueError(
            "Geometry-aware split produced an empty retained partition. Review pair connectivity "
            "or choose a different seed/fraction after inspecting the actual dataset."
        )
    validate_geometry_exclusion(a, b, split)
    return split


def validate_geometry_exclusion(geometry_a_ids: np.ndarray, geometry_b_ids: np.ndarray, split: SplitIndices) -> None:
    """Raise if any geometry identifier appears in multiple retained partitions."""
    a = np.asarray(geometry_a_ids)
    b = np.asarray(geometry_b_ids)
    groups = {
        "train": set(np.concatenate([a[split.train], b[split.train]])),
        "validation": set(np.concatenate([a[split.validation], b[split.validation]])),
        "test": set(np.concatenate([a[split.test], b[split.test]])),
    }
    labels = list(groups)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1:]:
            overlap = groups[left].intersection(groups[right])
            if overlap:
                raise AssertionError(f"Geometry leakage between {left} and {right}: {sorted(overlap)[:5]}")


def _validate_fractions(validation_fraction: float, test_fraction: float) -> None:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must be between zero and one.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must leave a training fraction.")
