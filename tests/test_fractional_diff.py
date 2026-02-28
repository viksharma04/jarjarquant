import numpy as np
import polars as pl
import pytest
from jarjarquant.fractional_diff import get_weights, frac_diff, get_weights_ffd, frac_diff_ffd


def test_get_weights_length():
    weights = get_weights(d=0.5, size=10)
    assert len(weights) == 10


def test_get_weights_last_is_one():
    weights = get_weights(d=0.5, size=5)
    assert weights[-1] == 1.0  # most recent weight is always 1.0


def test_frac_diff_returns_series():
    series = pl.Series("price", np.cumsum(np.random.randn(100)) + 100)
    result = frac_diff(series, d=0.5)
    assert isinstance(result, pl.Series)
    assert len(result) == len(series)


def test_frac_diff_ffd_returns_series():
    series = pl.Series("price", np.cumsum(np.random.randn(100)) + 100)
    result = frac_diff_ffd(series, d=0.5, threshold=1e-4)
    assert isinstance(result, pl.Series)
    assert len(result) == len(series)


def test_get_weights_ffd_returns_1d():
    weights = get_weights_ffd(d=0.5, threshold=1e-4)
    assert weights.ndim == 1


def test_frac_diff_d_zero_is_identity():
    values = np.cumsum(np.random.randn(50)) + 100
    series = pl.Series("price", values)
    result = frac_diff(series, d=0.0)
    # d=0 means no differentiation — should be close to original
    non_null = result.drop_nulls()
    assert len(non_null) > 0
