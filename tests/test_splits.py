import numpy as np

from src.splits import geometry_aware_split, random_pair_split, validate_geometry_exclusion


def test_random_pair_split_is_disjoint_and_complete():
    split = random_pair_split(20, validation_fraction=0.2, test_fraction=0.2, seed=9)
    merged = np.concatenate([split.train, split.validation, split.test])
    assert len(np.unique(merged)) == 20
    assert len(split.dropped) == 0


def test_geometry_aware_split_excludes_held_out_geometries():
    # Within-group duplicate pairs ensure every partition can retain samples.
    a = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    b = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    split = geometry_aware_split(a, b, validation_fraction=0.2, test_fraction=0.2, seed=3)
    validate_geometry_exclusion(a, b, split)
    retained = np.concatenate([split.train, split.validation, split.test])
    assert len(np.unique(retained)) == len(retained)
