"""Pure functions for labelling price data for ML.

Extracted from the Labeller class. All functions are stateless and operate
on numpy arrays / Polars DataFrames.
"""

from typing import Optional

import numpy as np
import polars as pl
from numba import jit

from jarjarquant.volatility import ewm_std_volatility


# ---------------------------------------------------------------------------
# Numba-accelerated helpers
# ---------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def _find_barrier_exits(
    close: np.ndarray,
    event_indices: np.ndarray,
    vb_indices: np.ndarray,
    pt_thresholds: np.ndarray,
    sl_thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find barrier exit indices, labels, and returns for each event.

    Args:
        close: Full array of close prices.
        event_indices: Indices into close where events occur.
        vb_indices: Indices into close for vertical barrier of each event.
        pt_thresholds: Profit-taking threshold (positive) per event.
        sl_thresholds: Stop-loss threshold (negative) per event.

    Returns:
        Tuple of (exit_indices, labels, returns) arrays.
        Labels: 1 = profit-taking, -1 = stop-loss, 0 = vertical barrier.
    """
    n_events = len(event_indices)
    exit_indices = np.empty(n_events, dtype=np.int64)
    labels = np.empty(n_events, dtype=np.int64)
    returns = np.empty(n_events, dtype=np.float64)

    for i in range(n_events):
        ev_idx = event_indices[i]
        vb_idx = vb_indices[i]
        entry_price = close[ev_idx]
        pt = pt_thresholds[i]
        sl = sl_thresholds[i]

        exit_idx = vb_idx
        label = 0
        first_pt = -1
        first_sl = -1

        for j in range(ev_idx + 1, vb_idx + 1):
            ret = close[j] / entry_price - 1.0
            if ret >= pt and first_pt == -1:
                first_pt = j
            if ret <= sl and first_sl == -1:
                first_sl = j

        if first_pt != -1 and first_sl != -1:
            if first_pt <= first_sl:
                exit_idx = first_pt
                label = 1
            else:
                exit_idx = first_sl
                label = -1
        elif first_pt != -1:
            exit_idx = first_pt
            label = 1
        elif first_sl != -1:
            exit_idx = first_sl
            label = -1
        else:
            exit_idx = vb_idx
            label = 0

        exit_indices[i] = exit_idx
        labels[i] = label
        returns[i] = close[exit_idx] / entry_price - 1.0

    return exit_indices, labels, returns


@jit(nopython=True, cache=True)
def _one_period_sl_labels(
    close: np.ndarray,
    low: np.ndarray,
    event_indices: np.ndarray,
    stop_loss_ret: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one-period-ahead labels with intraday stop-loss check."""
    n_events = len(event_indices)
    exit_indices = np.empty(n_events, dtype=np.int64)
    labels = np.empty(n_events, dtype=np.int64)
    returns = np.empty(n_events, dtype=np.float64)

    for i in range(n_events):
        idx = event_indices[i]
        next_idx = idx + 1
        intraday_ret = low[next_idx] / close[next_idx] - 1.0

        if intraday_ret < stop_loss_ret:
            exit_indices[i] = next_idx
            labels[i] = -1
            returns[i] = stop_loss_ret
        else:
            ret = close[next_idx + 1] / close[next_idx] - 1.0
            exit_indices[i] = next_idx + 1
            returns[i] = ret
            if ret > 0.0:
                labels[i] = 1
            elif ret < 0.0:
                labels[i] = -1
            else:
                labels[i] = 0

    return exit_indices, labels, returns


@jit(nopython=True, cache=True)
def _n_period_sl_labels(
    close: np.ndarray,
    low: np.ndarray,
    event_indices: np.ndarray,
    stop_loss_ret: float,
    n_periods: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute n-period-ahead labels with daily stop-loss check."""
    n_events = len(event_indices)
    exit_indices = np.empty(n_events, dtype=np.int64)
    labels = np.empty(n_events, dtype=np.int64)
    returns = np.empty(n_events, dtype=np.float64)

    for i in range(n_events):
        idx = event_indices[i]
        sl_entry = close[idx]
        stopped = False

        for j in range(idx + 1, idx + n_periods + 1):
            if low[j] / sl_entry - 1.0 < stop_loss_ret:
                exit_indices[i] = j
                labels[i] = -1
                returns[i] = stop_loss_ret
                stopped = True
                break

        if not stopped:
            exit_idx = idx + 1 + n_periods
            ret = close[exit_idx] / close[idx + 1] - 1.0
            exit_indices[i] = exit_idx
            returns[i] = ret
            if ret >= 0.0:
                labels[i] = 1
            elif ret < 0.0:
                labels[i] = -1

    return exit_indices, labels, returns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inverse_cumsum_filter(
    close: np.ndarray,
    threshold: float = 0.01,
    n: int = 2,
) -> np.ndarray:
    """Apply a cumulative sum filter to a price series.

    Args:
        close: Array of close prices.
        threshold: Threshold value for filtering.
        n: Lookback period for the rolling window.

    Returns:
        Boolean array where True indicates dates flagged by the filter
        (rolling cumulative return magnitude below threshold).
    """
    returns = np.empty(len(close), dtype=np.float64)
    returns[0] = np.nan
    returns[1:] = close[1:] / close[:-1]  # 1 + pct_change

    # Rolling product over window n
    flagged = np.full(len(close), False, dtype=bool)
    for i in range(n, len(close)):
        product = 1.0
        for j in range(i - n + 1, i + 1):
            if np.isnan(returns[j]):
                product = np.nan
                break
            product *= returns[j]
        if not np.isnan(product):
            cum_ret = product - 1.0
            flagged[i] = abs(cum_ret) < threshold

    return flagged


def event_sampling(
    close: np.ndarray,
    method: str = "vol_contraction",
    long_lookback: int = 100,
    short_lookback: int = 10,
    threshold: float = 2.0,
) -> np.ndarray:
    """Sample events based on volatility contraction or expansion.

    Args:
        close: Array of close prices.
        method: 'vol_contraction' or 'vol_expansion'.
        long_lookback: Lookback for long-term volatility.
        short_lookback: Lookback for short-term volatility.
        threshold: Ratio threshold for flagging events.

    Returns:
        Boolean array where True indicates flagged event dates.
    """
    long_vol = ewm_std_volatility(close, long_lookback)
    short_vol = ewm_std_volatility(close, short_lookback)

    if method == "vol_contraction":
        ratio = long_vol / short_vol
    elif method == "vol_expansion":
        ratio = short_vol / long_vol
    else:
        raise ValueError(
            "Method must be 'vol_contraction' or 'vol_expansion'"
        )

    flagged = ratio > threshold
    # Handle NaN values
    flagged = np.where(np.isnan(ratio), False, flagged)
    return flagged.astype(bool)


def get_vertical_barrier(
    dates: np.ndarray,
    event_indices: np.ndarray,
    n_days: int = 1,
) -> np.ndarray:
    """Get vertical barrier indices for each event.

    Args:
        dates: Array of datetime64 values (sorted ascending).
        event_indices: Indices into dates where events occur.
        n_days: Number of days for the vertical barrier.

    Returns:
        Array of indices into dates for the vertical barrier of each event.
    """
    event_dates = dates[event_indices]
    vb_dates = event_dates + np.timedelta64(n_days, "D")
    vb_indices = np.searchsorted(dates, vb_dates, side="left")
    n = len(dates)
    vb_indices = np.clip(vb_indices, 0, n - 1).astype(np.int64)
    return vb_indices


def triple_barrier_labels(
    dates: np.ndarray,
    close: np.ndarray,
    event_mask: np.ndarray | None = None,
    scale_pt_sl: bool = True,
    span: int = 10,
    pt_sl: float = 1.0,
    n_days: int = 2,
) -> pl.DataFrame:
    """Apply the triple barrier method using numpy arrays and return a Polars DataFrame.

    Args:
        dates: Array of datetime64 values (sorted ascending).
        close: Array of close prices, same length as dates.
        event_mask: Boolean mask selecting which dates are events. If None,
            all dates are used.
        scale_pt_sl: If True, scale pt_sl by EWM std volatility.
        span: Span for EWM std volatility calculation.
        pt_sl: Profit-taking / stop-loss multiplier (symmetric).
        n_days: Number of days for the vertical barrier.

    Returns:
        pl.DataFrame with columns: date, exit_date, label, returns.
    """
    n = len(dates)

    if event_mask is None:
        event_indices = np.arange(n, dtype=np.int64)
    else:
        event_indices = np.where(event_mask)[0].astype(np.int64)

    if len(event_indices) == 0:
        return pl.DataFrame(
            schema={
                "date": pl.Datetime,
                "exit_date": pl.Datetime,
                "label": pl.Int64,
                "returns": pl.Float64,
            }
        )

    # Compute volatility
    if scale_pt_sl:
        vol = ewm_std_volatility(close, span)
    else:
        vol = np.full(n, 0.01)

    # Compute vertical barrier indices
    event_dates = dates[event_indices]
    vb_dates = event_dates + np.timedelta64(n_days, "D")
    vb_indices = np.searchsorted(dates, vb_dates, side="left")
    vb_indices = np.clip(vb_indices, 0, n - 1).astype(np.int64)

    # Compute thresholds per event
    event_vol = vol[event_indices]
    nan_mask = np.isnan(event_vol)
    if np.any(nan_mask):
        event_vol = event_vol.copy()
        event_vol[nan_mask] = 0.01
    pt_thresholds = pt_sl * event_vol
    sl_thresholds = -pt_sl * event_vol

    # Run numba barrier search
    exit_indices, labels, returns = _find_barrier_exits(
        close, event_indices, vb_indices, pt_thresholds, sl_thresholds
    )

    return pl.DataFrame(
        {
            "date": dates[event_indices],
            "exit_date": dates[exit_indices],
            "label": labels,
            "returns": returns,
        }
    )


def one_period_with_sl(
    dates: np.ndarray,
    close: np.ndarray,
    low: np.ndarray,
    stop_loss_ret: float = -0.02,
    event_mask: np.ndarray | None = None,
) -> pl.DataFrame:
    """Label each bar based on next-period return with an intraday stop-loss check.

    Args:
        dates: Array of datetime64 values (sorted ascending).
        close: Array of close prices, same length as dates.
        low: Array of low prices, same length as dates.
        stop_loss_ret: Negative return threshold for intraday stop-loss.
        event_mask: Boolean mask selecting which dates are events.

    Returns:
        pl.DataFrame with columns: date, exit_date, label, returns.
    """
    n = len(dates)

    if event_mask is None:
        all_event_indices = np.arange(n, dtype=np.int64)
    else:
        all_event_indices = np.where(event_mask)[0].astype(np.int64)

    if len(all_event_indices) == 0:
        return pl.DataFrame(
            schema={
                "date": pl.Datetime,
                "exit_date": pl.Datetime,
                "label": pl.Int64,
                "returns": pl.Float64,
            }
        )

    valid_mask = all_event_indices + 2 < n
    valid_indices = all_event_indices[valid_mask]

    n_all = len(all_event_indices)
    out_exit_dates = np.empty(n_all, dtype=dates.dtype)
    out_labels = np.empty(n_all, dtype=np.int64)
    out_returns = np.empty(n_all, dtype=np.float64)

    out_exit_dates[:] = np.datetime64("NaT")
    out_labels[:] = 0
    out_returns[:] = np.nan

    if len(valid_indices) > 0:
        exit_indices, labels, returns = _one_period_sl_labels(
            close, low, valid_indices, stop_loss_ret, n
        )
        out_exit_dates[valid_mask] = dates[exit_indices]
        out_labels[valid_mask] = labels
        out_returns[valid_mask] = returns

    df = pl.DataFrame(
        {
            "date": dates[all_event_indices],
            "exit_date": out_exit_dates,
            "label": out_labels,
            "returns": out_returns,
        }
    )

    trailing_count = int((~valid_mask).sum())
    if trailing_count > 0:
        df = df.with_columns(
            [
                pl.when(pl.col("exit_date").is_null())
                .then(None)
                .otherwise(pl.col("exit_date"))
                .alias("exit_date"),
                pl.when(pl.col("returns").is_nan())
                .then(None)
                .otherwise(pl.col("label"))
                .alias("label"),
                pl.when(pl.col("returns").is_nan())
                .then(None)
                .otherwise(pl.col("returns"))
                .alias("returns"),
            ]
        )

    return df


def n_period_with_sl(
    dates: np.ndarray,
    close: np.ndarray,
    low: np.ndarray,
    n_periods: int = 1,
    stop_loss_ret: float = -0.02,
    event_mask: np.ndarray | None = None,
) -> pl.DataFrame:
    """Label each bar based on n-period return with daily stop-loss check.

    Args:
        dates: Array of datetime64 values (sorted ascending).
        close: Array of close prices, same length as dates.
        low: Array of low prices, same length as dates.
        n_periods: Number of holding periods.
        stop_loss_ret: Negative return threshold for stop-loss.
        event_mask: Boolean mask selecting which dates are events.

    Returns:
        pl.DataFrame with columns: date, exit_date, label, returns.
    """
    n = len(dates)

    if event_mask is None:
        all_event_indices = np.arange(n, dtype=np.int64)
    else:
        all_event_indices = np.where(event_mask)[0].astype(np.int64)

    if len(all_event_indices) == 0:
        return pl.DataFrame(
            schema={
                "date": pl.Datetime,
                "exit_date": pl.Datetime,
                "label": pl.Int64,
                "returns": pl.Float64,
            }
        )

    valid_mask = all_event_indices + n_periods + 1 < n
    valid_indices = all_event_indices[valid_mask]

    n_all = len(all_event_indices)
    out_exit_dates = np.empty(n_all, dtype=dates.dtype)
    out_labels = np.empty(n_all, dtype=np.int64)
    out_returns = np.empty(n_all, dtype=np.float64)

    out_exit_dates[:] = np.datetime64("NaT")
    out_labels[:] = 0
    out_returns[:] = np.nan

    if len(valid_indices) > 0:
        exit_indices, labels, returns = _n_period_sl_labels(
            close, low, valid_indices, stop_loss_ret, n_periods, n
        )
        out_exit_dates[valid_mask] = dates[exit_indices]
        out_labels[valid_mask] = labels
        out_returns[valid_mask] = returns

    df = pl.DataFrame(
        {
            "date": dates[all_event_indices],
            "exit_date": out_exit_dates,
            "label": out_labels,
            "returns": out_returns,
        }
    )

    trailing_count = int((~valid_mask).sum())
    if trailing_count > 0:
        df = df.with_columns(
            [
                pl.when(pl.col("exit_date").is_null())
                .then(None)
                .otherwise(pl.col("exit_date"))
                .alias("exit_date"),
                pl.when(pl.col("returns").is_nan())
                .then(None)
                .otherwise(pl.col("label"))
                .alias("label"),
                pl.when(pl.col("returns").is_nan())
                .then(None)
                .otherwise(pl.col("returns"))
                .alias("returns"),
            ]
        )

    return df


def num_co_events(
    dates: np.ndarray,
    event_dates: np.ndarray,
    exit_dates: np.ndarray,
) -> np.ndarray:
    """Compute the number of concurrent events per bar.

    Args:
        dates: Full array of dates covering the range.
        event_dates: Start dates for each event.
        exit_dates: End dates for each event.

    Returns:
        Array of co-event counts, same length as dates.
    """
    n = len(dates)
    count = np.zeros(n, dtype=np.int64)

    for i in range(len(event_dates)):
        start_idx = np.searchsorted(dates, event_dates[i], side="left")
        end_idx = np.searchsorted(dates, exit_dates[i], side="right") - 1
        end_idx = min(end_idx, n - 1)
        for j in range(start_idx, end_idx + 1):
            count[j] += 1

    return count


def average_uniqueness(
    dates: np.ndarray,
    event_dates: np.ndarray,
    exit_dates: np.ndarray,
    co_events: np.ndarray,
) -> np.ndarray:
    """Compute average uniqueness for each event.

    Args:
        dates: Full array of dates.
        event_dates: Start dates for each event.
        exit_dates: End dates for each event.
        co_events: Array of co-event counts (from num_co_events).

    Returns:
        Array of average uniqueness values, one per event.
    """
    n_events = len(event_dates)
    weights = np.empty(n_events, dtype=np.float64)

    for i in range(n_events):
        start_idx = np.searchsorted(dates, event_dates[i], side="left")
        end_idx = np.searchsorted(dates, exit_dates[i], side="right") - 1
        end_idx = min(end_idx, len(dates) - 1)

        if start_idx > end_idx:
            weights[i] = 1.0
            continue

        total = 0.0
        count = 0
        for j in range(start_idx, end_idx + 1):
            if co_events[j] > 0:
                total += 1.0 / co_events[j]
                count += 1

        weights[i] = total / count if count > 0 else 1.0

    return weights


def get_sample_weights(
    labels_df: pl.DataFrame,
    close: np.ndarray | None = None,
) -> np.ndarray:
    """Compute sample weights based on return attribution and event concurrency.

    Args:
        labels_df: DataFrame with columns: date, exit_date, label, returns.
        close: Optional array of close prices. If None, weights are based on
            average uniqueness only.

    Returns:
        Array of sample weights, one per event in labels_df.
    """
    event_dates = labels_df["date"].to_numpy()
    exit_dates = labels_df["exit_date"].to_numpy()

    # Build a date range covering all events
    all_dates = np.sort(
        np.unique(np.concatenate([event_dates, exit_dates]))
    )
    # Remove NaT values
    all_dates = all_dates[~np.isnat(all_dates)]

    co_events = num_co_events(all_dates, event_dates, exit_dates)
    weights = average_uniqueness(all_dates, event_dates, exit_dates, co_events)

    return weights
