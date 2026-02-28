"""Pure functions for fractional differentiation of time series."""

import numpy as np
import polars as pl


def get_weights(d: float, size: int) -> np.ndarray:
    """Calculate weights for fractional differentiation.

    Args:
        d: Order of differentiation.
        size: Number of weights to compute.

    Returns:
        1D array of weights (oldest to newest).
    """
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1])
    return w


def frac_diff(series: pl.Series, d: float = 0.7, threshold: float = 0.01) -> pl.Series:
    """Apply fractional differentiation to a time series.

    Args:
        series: Input price series.
        d: Order of fractional differentiation.
        threshold: Cumulative weight threshold for skipping small weights.

    Returns:
        Fractionally differentiated series (same length, leading values are null).
    """
    values = series.to_numpy().astype(np.float64)
    n = len(values)
    w = get_weights(d, n).reshape(-1, 1)

    # Determine how many leading weights to skip
    w_cum = np.cumsum(np.abs(w))
    w_cum /= w_cum[-1]
    skip = int(np.sum(w_cum > threshold))

    result = np.full(n, np.nan)
    for iloc in range(skip, n):
        if not np.isfinite(values[iloc]):
            continue
        result[iloc] = np.dot(w[-(iloc + 1):].T, values[:iloc + 1].reshape(-1, 1))[0, 0]

    return pl.Series(series.name, result)


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Compute weights for fixed-width window fractional differencing.

    Args:
        d: Fractional differencing parameter.
        threshold: Minimum absolute weight to include.

    Returns:
        1D array of weights (oldest to newest).
    """
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < threshold:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff_ffd(
    series: pl.Series, d: float = 0.7, threshold: float = 1e-5
) -> pl.Series:
    """Apply fractional differencing with fixed-width window.

    Args:
        series: Input price series.
        d: Fractional differencing parameter.
        threshold: Minimum weight threshold.

    Returns:
        Fractionally differentiated series (same length, leading values are null).
    """
    values = series.to_numpy().astype(np.float64)
    n = len(values)
    w = get_weights_ffd(d, threshold)
    width = len(w) - 1

    result = np.full(n, np.nan)
    for iloc in range(width, n):
        if not np.isfinite(values[iloc]):
            continue
        window = values[iloc - width: iloc + 1].reshape(-1, 1)
        result[iloc] = np.dot(w.T, window)[0, 0]

    return pl.Series(series.name, result)
