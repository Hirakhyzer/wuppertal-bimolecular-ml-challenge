"""Coulomb-matrix construction, canonical ordering, padding, and vectorization."""

from __future__ import annotations

import numpy as np

from .xyz_parser import MolecularGeometry


def build_coulomb_matrix(geometry: MolecularGeometry) -> np.ndarray:
    """Construct the validated Coulomb matrix for one molecular geometry.

    Diagonal terms use 0.5 * Z**2.4. Off-diagonal terms use Zi*Zj/distance.
    """
    atomic_numbers = geometry.atomic_numbers
    coordinates = np.asarray(geometry.coordinates, dtype=float)
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    matrix = np.zeros((geometry.atom_count, geometry.atom_count), dtype=float)
    diagonal = 0.5 * atomic_numbers ** 2.4
    np.fill_diagonal(matrix, diagonal)
    outer_z = np.outer(atomic_numbers, atomic_numbers)
    non_diagonal = ~np.eye(geometry.atom_count, dtype=bool)
    if np.any(distances[non_diagonal] <= 0.0):
        raise ValueError("Distinct atoms share identical coordinates; Coulomb term is undefined.")
    matrix[non_diagonal] = outer_z[non_diagonal] / distances[non_diagonal]
    return matrix


def canonicalize_coulomb_matrix(matrix: np.ndarray) -> np.ndarray:
    """Reorder rows and columns with permutation-invariant row signatures.

    The primary key is descending row L2 norm. A descending sorted-row signature
    breaks non-degenerate ties without referencing input atom order. This is a
    practical canonicalization strategy, not a complete uniqueness guarantee.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Coulomb matrix must be square.")
    row_norms = np.linalg.norm(matrix, axis=1)
    signatures = np.sort(matrix, axis=1)[:, ::-1]
    keys = [tuple([-row_norms[index], *(-signatures[index])]) for index in range(len(matrix))]
    order = np.asarray(sorted(range(len(matrix)), key=lambda index: keys[index]), dtype=int)
    return matrix[np.ix_(order, order)]


def pad_matrix(matrix: np.ndarray, max_atoms: int) -> np.ndarray:
    """Pad a square matrix with zeros to a fixed atom capacity."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    if max_atoms < matrix.shape[0]:
        raise ValueError("max_atoms cannot be smaller than the input matrix.")
    padded = np.zeros((max_atoms, max_atoms), dtype=float)
    padded[: matrix.shape[0], : matrix.shape[1]] = matrix
    return padded


def upper_triangular_vector(matrix: np.ndarray) -> np.ndarray:
    """Flatten upper-triangular entries, including the diagonal, of a square matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    indices = np.triu_indices(matrix.shape[0])
    return matrix[indices]


class CoulombMatrixFeaturizer:
    """Fit fixed-size Coulomb features using a training-derived atom capacity.

    Call `fit` only on training geometries to make the representation capacity an
    explicit part of the train/validation/test protocol.
    """

    def __init__(self, max_atoms: int | None = None) -> None:
        self.max_atoms = max_atoms
        self.fitted_max_atoms_: int | None = None

    def fit(self, geometries: list[MolecularGeometry]) -> "CoulombMatrixFeaturizer":
        """Determine a fixed padding capacity from supplied training geometries."""
        if not geometries:
            raise ValueError("Cannot fit a featurizer on an empty geometry collection.")
        observed = max(geometry.atom_count for geometry in geometries)
        if self.max_atoms is not None and self.max_atoms < observed:
            raise ValueError("Configured max_atoms is smaller than observed training geometry.")
        self.fitted_max_atoms_ = self.max_atoms or observed
        return self

    def transform(self, geometries: list[MolecularGeometry]) -> np.ndarray:
        """Create canonicalized, padded, upper-triangular Coulomb features."""
        if self.fitted_max_atoms_ is None:
            raise RuntimeError("Call fit before transform.")
        features = []
        for geometry in geometries:
            if geometry.atom_count > self.fitted_max_atoms_:
                raise ValueError("Encountered geometry larger than fitted padding capacity.")
            matrix = canonicalize_coulomb_matrix(build_coulomb_matrix(geometry))
            features.append(upper_triangular_vector(pad_matrix(matrix, self.fitted_max_atoms_)))
        return np.asarray(features, dtype=float)

    def fit_transform(self, geometries: list[MolecularGeometry]) -> np.ndarray:
        """Fit capacity and transform the same training geometries."""
        return self.fit(geometries).transform(geometries)
