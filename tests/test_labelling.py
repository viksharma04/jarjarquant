"""Tests for the labelling module."""

import numpy as np
import polars as pl
import pytest
from jarjarquant.labelling import (
    inverse_cumsum_filter,
    event_sampling,
    triple_barrier_labels,
    one_period_with_sl,
    n_period_with_sl,
    get_sample_weights,
    get_vertical_barrier,
)


def _make_prices(n: int = 500) -> np.ndarray:
    return np.cumsum(np.random.randn(n)) + 100


def test_inverse_cumsum_filter():
    close = _make_prices(500)
    result = inverse_cumsum_filter(close, threshold=0.02)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool or result.dtype == np.int64  # indices or mask


def test_triple_barrier_labels_returns_polars():
    n = 200
    close = _make_prices(n)
    dates = np.array(
        pl.date_range(
            pl.date(2020, 1, 1),
            pl.date(2020, 1, 1) + pl.duration(days=n - 1),
            eager=True,
        ).cast(pl.Datetime)
    )
    result = triple_barrier_labels(
        dates=dates,
        close=close,
        span=10,
        pt_sl=2.0,
        n_days=5,
    )
    assert isinstance(result, pl.DataFrame)
    assert "label" in result.columns


def test_one_period_with_sl():
    n = 100
    close = _make_prices(n)
    low = close - np.abs(np.random.randn(n)) * 0.5
    dates = np.array(
        pl.date_range(
            pl.date(2020, 1, 1),
            pl.date(2020, 1, 1) + pl.duration(days=n - 1),
            eager=True,
        ).cast(pl.Datetime)
    )
    result = one_period_with_sl(dates=dates, close=close, low=low)
    assert isinstance(result, pl.DataFrame)
    assert "label" in result.columns
    assert len(result) == n


def test_n_period_with_sl():
    n = 100
    close = _make_prices(n)
    low = close - np.abs(np.random.randn(n)) * 0.5
    dates = np.array(
        pl.date_range(
            pl.date(2020, 1, 1),
            pl.date(2020, 1, 1) + pl.duration(days=n - 1),
            eager=True,
        ).cast(pl.Datetime)
    )
    result = n_period_with_sl(dates=dates, close=close, low=low, n_periods=3)
    assert isinstance(result, pl.DataFrame)
    assert "label" in result.columns
    assert len(result) == n


def test_get_sample_weights():
    n = 50
    close = _make_prices(n)
    dates = np.arange(
        np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + n, dtype="datetime64[D]"
    )
    # Create labels with triple barrier
    labels_df = triple_barrier_labels(
        dates=dates.astype("datetime64[ns]"),
        close=close,
        span=10,
        pt_sl=1.0,
        n_days=3,
    )
    if len(labels_df) > 0:
        result = get_sample_weights(labels_df)
        assert isinstance(result, np.ndarray)
