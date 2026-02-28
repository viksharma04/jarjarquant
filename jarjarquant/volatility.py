import numpy as np
from numba import jit

from jarjarquant.schemas import VolatilityMeasure


def calculate_volatility(
    open: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    measure: VolatilityMeasure,
    n: int = 0,
    **kwargs,
) -> np.ndarray:
    match measure:
        case VolatilityMeasure.MAX_RANGE:
            return max_range_volatility(high, low, n, **kwargs)
        case VolatilityMeasure.ATR:
            return atr_volatility(high, low, close, n, **kwargs)
        case VolatilityMeasure.ANNUALIZED_STD_DEV:
            return annualized_std_dev_volatility(close, n, **kwargs)
        case VolatilityMeasure.HIGH_LOW:
            return high_low_volatility(high, low, n, **kwargs)
        case VolatilityMeasure.HIGH_LOW_CLOSE:
            return high_low_close_volatility(high, low, close, open, n, **kwargs)
        case VolatilityMeasure.YANG_ZHANG:
            return yang_zhang_volatility(open, high, low, close, n, **kwargs)
        case VolatilityMeasure.AVG_LOG_RETURNS:
            return avg_log_returns_volatility(close, n, **kwargs)
        case VolatilityMeasure.EWM_STD:
            return ewm_std_volatility(close, n, **kwargs)
        case _:
            raise ValueError(f"Unsupported volatility measure: {measure}")


@jit(nopython=True, cache=True, nogil=True)
def max_range_volatility(high: np.ndarray, low: np.ndarray, n: int) -> np.ndarray:
    n = n if n > 0 else len(high)
    length = len(high)
    vol = np.full(length, np.nan)

    # Pre-compute all ranges once instead of computing in each window
    ranges = np.abs(high - low)

    for i in range(n - 1, length):
        max_range = ranges[i]
        # Find max in window without creating a slice
        for j in range(i - n + 1, i):
            if ranges[j] > max_range:
                max_range = ranges[j]
        vol[i] = max_range

    return vol


@jit(nopython=True, cache=True, nogil=True)
def annualized_std_dev_volatility(close: np.ndarray, n: int) -> np.ndarray:
    n = n if n > 0 else len(close)
    length = len(close)
    vol = np.full(length, np.nan)

    # Pre-compute all log prices once
    log_close = np.log(close)

    # Pre-compute all log returns once (instead of np.diff in each window)
    log_returns = np.empty(length)
    log_returns[0] = np.nan
    for i in range(1, length):
        log_returns[i] = log_close[i] - log_close[i - 1]

    # Use cumulative sum for efficient rolling mean calculation
    cumsum = np.empty(length)
    cumsum[0] = 0.0
    for i in range(1, length):
        cumsum[i] = cumsum[i - 1] + log_returns[i]

    sqrt_252 = np.sqrt(252.0)
    inv_n_minus_1 = 1.0 / (n - 1)

    for i in range(n - 1, length):
        # Rolling mean using cumsum: O(1) instead of O(n)
        window_sum = cumsum[i] - cumsum[i - n + 1]
        mean_return = window_sum * inv_n_minus_1

        # Compute variance in single pass
        variance_sum = 0.0
        for j in range(i - n + 2, i + 1):
            diff = log_returns[j] - mean_return
            variance_sum += diff * diff

        std_dev = np.sqrt(variance_sum * inv_n_minus_1)
        vol[i] = std_dev * sqrt_252

    return vol


@jit(nopython=True, cache=True, nogil=True)
def high_low_volatility(high: np.ndarray, low: np.ndarray, n: int) -> np.ndarray:
    n = n if n > 0 else len(high)
    length = len(high)
    vol = np.full(length, np.nan)

    # Pre-compute log ranges once for entire array
    log_ranges_sq = np.empty(length)
    inv_4_log2 = 1.0 / (4.0 * np.log(2.0))
    for i in range(length):
        lr = np.log(high[i] / low[i])
        log_ranges_sq[i] = lr * lr * inv_4_log2

    # Use cumulative sum for efficient rolling mean
    cumsum = np.empty(length + 1)
    cumsum[0] = 0.0
    for i in range(length):
        cumsum[i + 1] = cumsum[i] + log_ranges_sq[i]

    inv_n = 1.0 / n
    for i in range(n - 1, length):
        window_sum = cumsum[i + 1] - cumsum[i - n + 1]
        vol[i] = np.sqrt(window_sum * inv_n)

    return vol


@jit(nopython=True, cache=True, nogil=True)
def high_low_close_volatility(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, open: np.ndarray, n: int
):
    n = n if n > 0 else len(high)
    length = len(high)
    vol = np.full(length, np.nan)

    # Pre-compute all log values once
    log_high_low = np.empty(length)
    log_close_open = np.empty(length)
    for i in range(length):
        log_high_low[i] = np.log(high[i] / low[i])
        log_close_open[i] = np.log(close[i] / open[i])

    # Pre-compute a - b values for each bar
    factor = 2.0 * np.log(2.0) - 1.0
    a_minus_b = np.empty(length)
    for i in range(length):
        a = 0.5 * log_high_low[i] * log_high_low[i]
        b = factor * log_close_open[i] * log_close_open[i]
        a_minus_b[i] = a - b

    # Use cumulative sum for efficient rolling mean
    cumsum = np.empty(length + 1)
    cumsum[0] = 0.0
    for i in range(length):
        cumsum[i + 1] = cumsum[i] + a_minus_b[i]

    inv_n = 1.0 / n
    for i in range(n - 1, length):
        window_sum = cumsum[i + 1] - cumsum[i - n + 1]
        mean_val = window_sum * inv_n
        # Handle potential negative values due to floating point
        vol[i] = np.sqrt(max(mean_val, 0.0))

    return vol


@jit(nopython=True, cache=True, nogil=True)
def yang_zhang_volatility(
    open: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int
):
    n = n if n > 0 else len(close)
    length = len(close)
    vol = np.full(length, np.nan)

    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    # Pre-compute all log values once
    log_open = np.log(open)
    log_high = np.log(high)
    log_low = np.log(low)
    log_close = np.log(close)

    # Pre-compute overnight returns squared (open/prev_close)
    log_open_close_sq = np.empty(length)
    log_open_close_sq[0] = 0.0
    for i in range(1, length):
        val = log_open[i] - log_close[i - 1]
        log_open_close_sq[i] = val * val

    # Pre-compute intraday returns squared (close/open)
    log_close_open_sq = np.empty(length)
    for i in range(length):
        val = log_close[i] - log_open[i]
        log_close_open_sq[i] = val * val

    # Pre-compute Rogers-Satchell variance component
    rs_var_component = np.empty(length)
    for i in range(length):
        h_c = log_high[i] - log_close[i]
        h_o = log_high[i] - log_open[i]
        l_c = log_low[i] - log_close[i]
        l_o = log_low[i] - log_open[i]
        rs_var_component[i] = h_c * h_o + l_c * l_o

    # Cumulative sums for rolling means
    cumsum_oc = np.empty(length + 1)
    cumsum_co = np.empty(length + 1)
    cumsum_rs = np.empty(length + 1)
    cumsum_oc[0] = 0.0
    cumsum_co[0] = 0.0
    cumsum_rs[0] = 0.0

    for i in range(length):
        cumsum_oc[i + 1] = cumsum_oc[i] + log_open_close_sq[i]
        cumsum_co[i + 1] = cumsum_co[i] + log_close_open_sq[i]
        cumsum_rs[i + 1] = cumsum_rs[i] + rs_var_component[i]

    inv_n = 1.0 / n
    inv_n_minus_1 = 1.0 / (n - 1)
    one_minus_k = 1.0 - k

    for i in range(n - 1, length):
        # Overnight variance: uses n-1 observations (from index 1 to n-1 in window)
        o_c_var = (cumsum_oc[i + 1] - cumsum_oc[i - n + 2]) * inv_n_minus_1
        # Intraday variance
        c_o_var = (cumsum_co[i + 1] - cumsum_co[i - n + 1]) * inv_n
        # Rogers-Satchell variance
        rs_var = (cumsum_rs[i + 1] - cumsum_rs[i - n + 1]) * inv_n

        vol[i] = np.sqrt(o_c_var + k * c_o_var + one_minus_k * rs_var)

    return vol


@jit(nopython=True, cache=True, nogil=True)
def _compute_true_ranges(
    high: np.ndarray, low: np.ndarray, prev_close: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized computation of true ranges.

    Args:
        high: Array of high prices
        low: Array of low prices
        prev_close: Array of previous close prices

    Returns:
        np.ndarray: True range values
    """
    n = len(high)
    true_ranges = np.empty(n)

    for i in range(n):
        if np.isnan(prev_close[i]):
            # If previous close is NaN, use high-low range
            true_ranges[i] = abs(high[i] - low[i])
        else:
            # True range = max(high-low, |high-prev_close|, |low-prev_close|)
            hl = abs(high[i] - low[i])
            hc = abs(high[i] - prev_close[i])
            lc = abs(low[i] - prev_close[i])
            true_ranges[i] = max(hl, hc, lc)

    return true_ranges


@jit(nopython=True, cache=True, nogil=True)
def _ema(data: np.ndarray, n: int) -> np.ndarray:
    """Exponential moving average with span n (alpha = 2 / (n + 1))."""
    alpha = 2.0 / (n + 1)
    result = np.empty_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


@jit(nopython=True, cache=True, nogil=True)
def atr_volatility(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int, use_ema: bool = False
) -> np.ndarray:
    """Compute ATR (Average True Range) volatility.

    Args:
        high (np.ndarray)
        low (np.ndarray)
        close (np.ndarray)
        n (int): period for ATR calculation. If n <= 0, returns one value at last index calculated over entire array.
        use_ema (bool, optional): Whether to use exponential moving average. Defaults to False.

    Returns:
        np.ndarray: ATR volatility values
    """
    n = n if n > 0 else len(close)
    length = len(close)
    vol = np.full(length, np.nan)

    # Compute previous close prices
    prev_close = np.empty(length)
    prev_close[0] = np.nan
    for i in range(1, length):
        prev_close[i] = close[i - 1]

    # Compute true ranges
    true_ranges = _compute_true_ranges(high, low, prev_close)

    if use_ema:
        # Compute ATR using exponential moving average
        ema_values = _ema(true_ranges, n)
        # Set first n-1 values to NaN to match SMA behavior
        for i in range(n - 1):
            vol[i] = np.nan
        for i in range(n - 1, length):
            vol[i] = ema_values[i]
    else:
        # Use cumulative sum for efficient rolling mean (SMA)
        cumsum = np.empty(length + 1)
        cumsum[0] = 0.0
        for i in range(length):
            cumsum[i + 1] = cumsum[i] + true_ranges[i]

        inv_n = 1.0 / n
        for i in range(n - 1, length):
            vol[i] = (cumsum[i + 1] - cumsum[i - n + 1]) * inv_n

    return vol


@jit(nopython=True, cache=True, nogil=True)
def ewm_std_volatility(close: np.ndarray, n: int) -> np.ndarray:
    """Compute exponentially weighted moving standard deviation of returns.

    Uses span-based EWM (alpha = 2 / (n + 1)) to incrementally compute
    the EWM variance of simple returns, then returns sqrt(variance).

    Args:
        close: Array of close prices.
        n: Span for the EWM calculation.

    Returns:
        np.ndarray: EWM standard deviation of returns at each step.
    """
    length = len(close)
    vol = np.full(length, np.nan)
    alpha = 2.0 / (n + 1)

    # First return is undefined
    if length < 2:
        return vol

    ret = close[1] / close[0] - 1.0
    ewm_mean = ret
    ewm_var = 0.0
    vol[1] = 0.0

    for i in range(2, length):
        ret = close[i] / close[i - 1] - 1.0
        delta = ret - ewm_mean
        ewm_mean = ewm_mean + alpha * delta
        ewm_var = (1.0 - alpha) * (ewm_var + alpha * delta * delta)
        vol[i] = np.sqrt(ewm_var)

    return vol


@jit(nopython=True, cache=True, nogil=True)
def avg_log_returns_volatility(close: np.ndarray, n: int) -> np.ndarray:
    n = n if n > 0 else len(close)
    length = len(close)
    vol = np.full(length, np.nan)

    # Pre-compute all log prices once
    log_close = np.log(close)

    # Pre-compute all log returns once
    log_returns = np.empty(length)
    log_returns[0] = 0.0
    for i in range(1, length):
        log_returns[i] = abs(log_close[i] - log_close[i - 1])

    # Use cumulative sum for efficient rolling mean
    cumsum = np.empty(length + 1)
    cumsum[0] = 0.0
    for i in range(length):
        cumsum[i + 1] = cumsum[i] + log_returns[i]

    # n-1 log returns in a window of n prices
    inv_n_minus_1 = 1.0 / (n - 1)
    for i in range(n - 1, length):
        # Sum of log returns from i-n+2 to i (inclusive)
        window_sum = cumsum[i + 1] - cumsum[i - n + 2]
        vol[i] = window_sum * inv_n_minus_1

    return vol
