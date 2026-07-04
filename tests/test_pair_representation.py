import numpy as np

from src.pair_representation import build_pair_features


def test_directed_concatenation_preserves_order():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 5.0])
    forward = build_pair_features(a, b, "directed_concatenation")
    reverse = build_pair_features(b, a, "directed_concatenation")
    assert np.array_equal(forward, np.array([1.0, 2.0, 3.0, 5.0]))
    assert not np.array_equal(forward, reverse)


def test_interaction_features_have_expected_blocks():
    a = np.array([2.0, 4.0])
    b = np.array([1.0, 8.0])
    feature = build_pair_features(a, b, "interaction_features")
    assert np.array_equal(feature, np.array([2.0, 4.0, 1.0, 8.0, 1.0, 4.0, 2.0, 32.0]))
