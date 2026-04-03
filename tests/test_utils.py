import numpy as np

from utils import get_length_histogram_bins


def test_get_length_histogram_bins_returns_default_bins_for_empty_input():
    bins = get_length_histogram_bins([])

    assert np.array_equal(bins, np.array([0.0, 5.0, 15.0, 30.0, 45.0]))


def test_get_length_histogram_bins_extends_upper_bound_for_long_samples():
    bins = get_length_histogram_bins([1.0, 12.0, 40.0])

    assert np.array_equal(bins[:-1], np.array([0.0, 5.0, 15.0, 30.0]))
    assert bins[-1] == 42.0
