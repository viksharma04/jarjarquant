"""Internal math utilities used by indicator implementations."""

from typing import Optional

import numpy as np
from numba import jit
from scipy.special import legendre

from jarjarquant.volatility import atr_volatility


def ewm(data: np.ndarray, span: int, adjust: bool = True) -> np.ndarray:
    """Exponential weighted moving average (replicates pd.Series.ewm().mean()).

    Args:
        data: Input array.
        span: EWM span (alpha = 2 / (span + 1)).
        adjust: If True, use bias-correcting denominator (matches pandas default).

    Returns:
        EWM array same length as input.
    """
    alpha = 2.0 / (span + 1)
    n = len(data)
    result = np.empty(n)

    if adjust:
        # Weighted average with decaying weights — matches pandas adjust=True
        numerator = 0.0
        denominator = 0.0
        for i in range(n):
            numerator = data[i] + (1 - alpha) * numerator
            denominator = 1.0 + (1 - alpha) * denominator
            result[i] = numerator / denominator
    else:
        result[0] = data[0]
        for i in range(1, n):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result


def rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with NaN for incomplete windows (replicates pd.Series.rolling().mean()).

    Args:
        data: Input array.
        window: Window size.

    Returns:
        Array same length as input, NaN for first (window-1) elements.
    """
    n = len(data)
    result = np.full(n, np.nan)
    cumsum = np.cumsum(data)
    result[window - 1] = cumsum[window - 1] / window
    for i in range(window, n):
        result[i] = (cumsum[i] - cumsum[i - window]) / window
    return result


def rolling_max(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling max with NaN for incomplete windows.

    Args:
        data: Input array.
        window: Window size.

    Returns:
        Array same length as input.
    """
    n = len(data)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.max(data[i - window + 1: i + 1])
    return result


def rolling_min(data: np.ndarray, window: int) -> np.ndarray:
    """Rolling min with NaN for incomplete windows.

    Args:
        data: Input array.
        window: Window size.

    Returns:
        Array same length as input.
    """
    n = len(data)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.min(data[i - window + 1: i + 1])
    return result


def shift(data: np.ndarray, periods: int) -> np.ndarray:
    """Shift array by N periods, filling with NaN.

    Args:
        data: Input array.
        periods: Number of positions to shift (positive = forward).

    Returns:
        Shifted array.
    """
    result = np.full_like(data, np.nan)
    if periods > 0:
        result[periods:] = data[:-periods]
    elif periods < 0:
        result[:periods] = data[-periods:]
    else:
        result[:] = data
    return result


def compute_legendre_coefficients(lookback: int, degree: int) -> np.ndarray:
    """Compute Legendre polynomial coefficients over a lookback window.

    Args:
        lookback: Number of points in the window.
        degree: Degree of the Legendre polynomial (1, 2, or 3).

    Returns:
        Coefficients of the Legendre polynomial for the given degree.
    """
    if degree not in [1, 2, 3]:
        raise ValueError("Only degrees 1, 2, or 3 are supported.")
    x = np.linspace(-1, 1, lookback)
    legendre_poly = legendre(degree)
    return legendre_poly(x)


def compute_normalized_legendre_coefficients(n: int, degree: int) -> np.ndarray:
    """Compute normalized orthogonal Legendre coefficient arrays.

    Args:
        n: Number of elements.
        degree: Degree (1, 2, or 3).

    Returns:
        Normalized coefficient array for the given degree.
    """
    c1 = np.linspace(-1.0, 1.0, n)
    c1 /= np.linalg.norm(c1)

    if degree == 1:
        return c1

    c2 = c1**2
    c2 -= np.mean(c2)
    c2 /= np.linalg.norm(c2)

    if degree == 2:
        return c2

    c3 = c1**3
    c3 -= np.mean(c3)
    c3 /= np.linalg.norm(c3)
    proj = np.dot(c1, c3)
    c3 = c3 - proj * c1
    c3 /= np.linalg.norm(c3)

    return c3


def calculate_regression_coefficient(
    prices: np.ndarray, legendre_coeffs: np.ndarray
) -> float:
    """Calculate linear regression coefficient via dot product.

    Args:
        prices: Price series (e.g., log prices).
        legendre_coeffs: Precomputed Legendre coefficients.

    Returns:
        Slope of the least squares line.
    """
    return np.dot(legendre_coeffs, prices)


def _determine_initial_direction(
    series: np.ndarray, start_idx: int, thresholds: np.ndarray
) -> int:
    """Look ahead to determine initial direction based on first significant move."""
    n = len(series)
    if start_idx >= n - 1:
        return 1

    initial_price = series[start_idx]
    if abs(initial_price) < 1e-10:
        return 1

    for i in range(start_idx + 1, n):
        value = series[i]
        if value == 0.0 or np.isnan(value):
            continue
        threshold = thresholds[i]
        change = (value - initial_price) / abs(initial_price)
        if abs(change) >= threshold:
            return 1 if change > 0 else -1

    return 1


@jit(nopython=True)
def _directional_change_core(series: np.ndarray, thresholds: np.ndarray) -> tuple:
    """Numba-optimized core directional change detection."""
    n = len(series)
    if n == 0:
        empty_array = np.empty(0, dtype=np.int64)
        return (empty_array, empty_array)

    start_idx = 0
    for i in range(n):
        if series[i] != 0.0 and not np.isnan(series[i]):
            start_idx = i
            break

    if start_idx == n - 1:
        result = np.empty(1, dtype=np.int64)
        result[0] = start_idx
        return (result, result)

    # Determine initial direction inline (can't call non-jit from jit)
    initial_price = series[start_idx]
    up = True
    if abs(initial_price) > 1e-10:
        for i in range(start_idx + 1, n):
            value = series[i]
            if value == 0.0 or np.isnan(value):
                continue
            threshold = thresholds[i]
            change = (value - initial_price) / abs(initial_price)
            if abs(change) >= threshold:
                up = change > 0
                break

    max_pivots = n // 2 + 1
    recognition_indices_temp = np.empty(max_pivots, dtype=np.int64)
    extreme_indices_temp = np.empty(max_pivots, dtype=np.int64)

    pivot_count = 0
    current_pivot = series[start_idx]
    current_pivot_idx = start_idx

    recognition_indices_temp[pivot_count] = start_idx
    extreme_indices_temp[pivot_count] = start_idx
    pivot_count += 1

    for i in range(start_idx + 1, n):
        value = series[i]
        if value == 0.0 or np.isnan(value):
            continue

        threshold = thresholds[i]
        if abs(current_pivot) < 1e-10:
            current_pivot = value
            current_pivot_idx = i
            continue

        if up:
            if value > current_pivot:
                current_pivot = value
                current_pivot_idx = i
            else:
                change = (value - current_pivot) / abs(current_pivot)
                if change < -threshold:
                    up = False
                    recognition_indices_temp[pivot_count] = i
                    extreme_indices_temp[pivot_count] = current_pivot_idx
                    pivot_count += 1
                    current_pivot = value
                    current_pivot_idx = i
        else:
            if value < current_pivot:
                current_pivot = value
                current_pivot_idx = i
            else:
                change = (value - current_pivot) / abs(current_pivot)
                if change > threshold:
                    up = True
                    recognition_indices_temp[pivot_count] = i
                    extreme_indices_temp[pivot_count] = current_pivot_idx
                    pivot_count += 1
                    current_pivot = value
                    current_pivot_idx = i

    return (recognition_indices_temp[:pivot_count], extreme_indices_temp[:pivot_count])


def directional_change_pivots(
    series: np.ndarray,
    threshold_type: str = "static",
    threshold_value: Optional[float] = 0.01,
    atr_window: int = 14,
    high_series: Optional[np.ndarray] = None,
    low_series: Optional[np.ndarray] = None,
    close_series: Optional[np.ndarray] = None,
    return_extremes: bool = False,
) -> tuple:
    """Detect directional change pivots using static or volatility-based thresholds.

    Args:
        series: Price series (typically close prices).
        threshold_type: "static" or "volatility".
        threshold_value: Static threshold (percentage as decimal).
        atr_window: Window for ATR calculation (volatility mode).
        high_series: High prices (required for volatility).
        low_series: Low prices (required for volatility).
        close_series: Close prices (required for volatility).
        return_extremes: If True, return (recognition_indices, extreme_indices).

    Returns:
        Recognition indices, or tuple of (recognition, extreme) indices.
    """
    if len(series) == 0:
        empty_array = np.array([], dtype=np.int64)
        return (empty_array, empty_array) if return_extremes else empty_array

    series = np.asarray(series, dtype=np.float64)

    if threshold_type == "static":
        if threshold_value is None:
            raise ValueError("threshold_value must be provided for static threshold type")
        thresholds = np.full(len(series), abs(threshold_value), dtype=np.float64)

    elif threshold_type == "volatility":
        if any(x is None for x in [high_series, low_series]):
            raise ValueError("high_series and low_series must be provided for volatility threshold")

        high_series = np.asarray(high_series, dtype=np.float64)
        low_series = np.asarray(low_series, dtype=np.float64)
        close_series = np.asarray(
            close_series if close_series is not None else series, dtype=np.float64
        )

        atr_values = atr_volatility(high_series, low_series, close_series, atr_window)

        thresholds = np.where(
            np.abs(series) > 1e-10,
            atr_values / np.abs(series),
            np.full(len(series), 0.01),
        )
        thresholds = np.where(np.isnan(thresholds), 0.01, thresholds)
    else:
        raise ValueError(f"Invalid threshold_type: {threshold_type}. Must be 'static' or 'volatility'")

    recognition_indices, extreme_indices = _directional_change_core(series, thresholds)

    if return_extremes:
        return (recognition_indices, extreme_indices)
    return recognition_indices
