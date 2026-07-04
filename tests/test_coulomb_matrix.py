import numpy as np

from src.coulomb_matrix import CoulombMatrixFeaturizer, build_coulomb_matrix
from src.xyz_parser import MolecularGeometry


def make_geometry(symbols, coordinates):
    return MolecularGeometry(tuple(symbols), np.asarray(coordinates, dtype=float), "toy")


def test_coulomb_diagonal_matches_definition():
    geometry = make_geometry(["H", "He"], [[0, 0, 0], [0, 0, 1]])
    matrix = build_coulomb_matrix(geometry)
    assert np.isclose(matrix[0, 0], 0.5 * 1**2.4)
    assert np.isclose(matrix[1, 1], 0.5 * 2**2.4)
    assert np.isclose(matrix[0, 1], 2.0)
    assert np.isclose(matrix[0, 1], matrix[1, 0])


def test_feature_is_invariant_to_non_degenerate_atom_ordering():
    symbols = ["H", "C", "O"]
    coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1], [0.3, 1.4, 0.2]]
    first = make_geometry(symbols, coordinates)
    order = [2, 0, 1]
    second = make_geometry([symbols[i] for i in order], [coordinates[i] for i in order])
    featurizer = CoulombMatrixFeaturizer().fit([first, second])
    features = featurizer.transform([first, second])
    assert np.allclose(features[0], features[1])
