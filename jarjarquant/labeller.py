"""The labeller specializes in transforming raw price data into labels for ML using various methods"""

# Imports
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from numba import jit

from .core.volatility_calculations import ewm_std_volatility
from .data_analyst import get_daily_vol


@jit(nopython=True, cache=True)
def _find_barrier_exits(
    close: np.ndarray,
    event_indices: np.ndarray,
    vb_indices: np.ndarray,
    pt_thresholds: np.ndarray,
    sl_thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find barrier exit indices, labels, and returns for each event.

    For each event, walks forward from event index to vertical barrier index,
    checking if the return crosses the profit-taking or stop-loss threshold.

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

        # Determine which barrier was hit first
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
    """Compute one-period-ahead labels with intraday stop-loss check.

    For each event at index i:
    1. Check if low[i+1]/close[i+1] - 1 < stop_loss_ret (intraday stop-loss).
       If so: label = -1, ret = stop_loss_ret, exit = i+1.
    2. Otherwise: ret = close[i+2]/close[i+1] - 1, label = sign(ret), exit = i+2.

    Args:
        close: Full array of close prices.
        low: Full array of low prices.
        event_indices: Indices into close/low where events occur (i+2 < n guaranteed).
        stop_loss_ret: Negative return threshold for intraday stop-loss.
        n: Total length of the price arrays.

    Returns:
        Tuple of (exit_indices, labels, returns) arrays.
    """
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
    """Compute n-period-ahead labels with daily stop-loss check.

    For each event at index i:
    1. Entry price for SL check = close[i], entry price for return = close[i+1].
    2. Walk forward from j = i+1 to i+n_periods (inclusive). If
       low[j] / close[i] - 1 < stop_loss_ret, exit at j with label = -1
       and return = stop_loss_ret.
    3. If no stop-loss hit: ret = close[i+1+n_periods] / close[i+1] - 1,
       label = sign(ret), exit = i+1+n_periods.

    Args:
        close: Full array of close prices.
        low: Full array of low prices.
        event_indices: Indices where events occur (i+1+n_periods < n guaranteed).
        stop_loss_ret: Negative return threshold for stop-loss.
        n_periods: Number of holding periods.
        n: Total length of the price arrays.

    Returns:
        Tuple of (exit_indices, labels, returns) arrays.
    """
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


class Labeller:
    "Class to label data - implements commonn methods used during labelling for financial ml"

    def __init__(self, ohlcv_df: pd.DataFrame):
        """Initialize Labelling

        Args:
            ohlcv_df (pd.DataFrame): DataFrame containing OHLCV data with a datetime index.
        """
        self._df = ohlcv_df

    def inverse_cumsum_filter(
        self, series: pd.Series = None, h: float = 0.01, n: int = 2
    ) -> pd.Series:
        """
        Apply a cumulative sum filter to a time series based on a rolling period.

        Parameters:
        - series: pd.Series, time series of prices with time stamp index
        - h: float, threshold value for filtering
        - n: int, lookback period for the rolling window

        Returns:
        - pd.Series, boolean series where True indicates dates flagged by the filter
        """
        if series is None:
            series = self._df["Close"]

        returns = series.pct_change()
        # Ensure the series is sorted by index (time)
        returns = returns.add(1)

        # Calculate the rolling cumulative sum over the lookback period n
        rolling_cumsum = returns.rolling(window=n).apply(np.prod) - 1

        # Flag dates where the cumulative return is less than the absolute value of h
        flagged = rolling_cumsum.abs() < h

        return flagged

    def event_sampling(
        self,
        method: str = "vol_contraction",
        long_lookback: int = 100,
        short_lookback: int = 10,
        threshold: float = 2,
        price: Optional[str] = "Close",
    ):
        """
        method can be 'vol_contraction', 'vol_expansion'
        contraction and expansion are defined based on three parameters: long lookback, short lookback, and threshold
        """
        price_series = self._df[price]
        long_vol = get_daily_vol(price_series, long_lookback)
        short_vol = get_daily_vol(price_series, short_lookback)

        if method == "vol_contraction":
            ratio = long_vol / short_vol
        elif method == "vol_expansion":
            ratio = short_vol / long_vol
        else:
            raise ValueError(
                "Method must be 'volatility_contraction' or 'volatility_expansion'"
            )

        # Flag dates where the ratio is greater than the threshold
        flagged = ratio > threshold

        # Add flagged column as 'event_flag' column to self._df
        self._df["event_flag"] = flagged

    @staticmethod
    def plot_with_flags(series: pd.Series, flagged: pd.Series):
        """
        Plots a time series and highlights flagged dates as red dots.

        Parameters:
        - series: pd.Series, the original time series of returns with timestamp index
        - flagged: pd.Series, boolean series indicating flagged dates
        """
        # Ensure the series is sorted by time index
        series = series.sort_index()

        # Plot the time series
        plt.figure(figsize=(10, 6))
        plt.plot(series.index, series.values, label="Time Series", color="blue")

        # Highlight flagged dates as red dots
        plt.scatter(
            series.index[flagged],
            series.values[flagged],
            color="red",
            label="Flagged Dates",
        )

        # Add labels and legend
        plt.title(
            f"Time Series with Flagged Dates; Percent labels = {
                np.average(flagged) * 100
            }%"
        )
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.show()

    # Getting dates for the vertical barrier
    @staticmethod
    def get_vertical_barrier(t_events, Close, num_days=1):
        """Get a datetime index of dates for the vertical barrier

        Args:
            tEvents (datetime index): dates when the algorithm should look for trades
            Close (pd.Series): series of prices
            numDays (int, optional): vertical barrier limit. Defaults to 1.

        Returns:
            pd.Series: series of datetime values
        """
        t1 = Close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
        t1 = t1[t1 < Close.shape[0]]
        t1 = pd.Series(Close.index[t1], index=t_events[: t1.shape[0]])
        t1.index = t1.index.tz_localize(None)
        return t1

    @staticmethod
    def find_min_column(row):
        if pd.isnull(row["pt"]) & pd.isnull(row["sl"]):
            min_value = row[["pt", "sl"]].min()
        else:
            min_value = pd.Timestamp(0)
        return (
            row[["pt", "sl"]].idxmin()
            if min_value <= row["vb"]
            else row[["pt", "sl", "vb"]].idxmin()
        )

    @staticmethod
    def triple_barrier_method(
        Close: pd.Series,
        t_events: pd.DatetimeIndex,
        scale_pt_sl: bool = True,
        span: int = 10,
        pt_sl: float = 1,
        n_days: int = 2,
    ):
        Close.index = pd.DatetimeIndex(Close.index)
        Close.index = Close.index.tz_localize(None)

        # If scale pt_sl is True pt_sl is multiplied by the average period volatility over the scale_lookback period
        if scale_pt_sl:
            vol = get_daily_vol(close=Close, span=span)
            # returns = Close.pct_change()
            # vol = returns.rolling(
            #     window=scale_lookback, min_periods=1).std()*np.sqrt(n_days)
            Close = Close.iloc[n_days:]
            trgt = vol[vol.index.isin(Close.index)]

        # If scale pt_sl is False pt_sl is used as absolute return i.e 1 = 1% return
        else:
            trgt = pd.Series(0.01, index=t_events)

        if t_events is None:
            t_events = Close.index
        else:
            Close = Close.loc[Close.index.intersection(t_events)]

        t_events = t_events[t_events.isin(Close.index)]

        v_bars = Labeller.get_vertical_barrier(t_events, Close, n_days)
        pt_sl = [pt_sl, -pt_sl]

        events = pd.concat({"vb": v_bars, "trgt": trgt}, axis=1).dropna(subset=["trgt"])

        exits = events[["vb"]].copy(deep=True)
        exits["sl"] = pd.NaT
        exits["pt"] = pd.NaT

        pt = pt_sl[0] * events["trgt"]
        sl = pt_sl[1] * events["trgt"]

        for event, vb in events["vb"].fillna(Close.index[-1]).items():
            price_path = Close[event:vb]
            return_path = price_path / Close[event] - 1
            exits.loc[event, "sl"] = return_path[
                return_path
                <
                # earliest stop loss
                sl[event]
            ].index.min()
            exits.loc[event, "pt"] = return_path[
                return_path
                >
                # earliest profit taking
                pt[event]
            ].index.min()

        exits["vb"] = exits["vb"].fillna(Close.index[-1])
        exits["barrier_hit"] = exits.apply(Labeller.find_min_column, axis=1)
        exits["hit_date"] = exits[["vb", "sl", "pt"]].min(axis=1)
        exits["returns"] = Close[exits["hit_date"]].values / Close[t_events].values - 1
        exits["bin"] = np.sign(exits["returns"])
        exits.loc[exits["barrier_hit"] == "vb", "bin"] = 0

        return exits[["hit_date", "bin", "returns"]]

    @staticmethod
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

        This is a faster alternative to `triple_barrier_method` that uses numba
        for the barrier search and Polars for the output DataFrame.

        Args:
            dates: Array of datetime64 values (must be sorted ascending).
            close: Array of close prices, same length as dates.
            event_mask: Boolean mask selecting which dates are events. If None,
                all dates are used.
            scale_pt_sl: If True, scale pt_sl by EWM std volatility. If False,
                use a fixed 0.01 threshold.
            span: Span for EWM std volatility calculation (used when scale_pt_sl=True).
            pt_sl: Profit-taking / stop-loss multiplier (symmetric).
            n_days: Number of days for the vertical barrier.

        Returns:
            pl.DataFrame with columns: date, exit_date, label, returns.
        """
        n = len(dates)

        # Determine event indices
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

        # Compute vertical barrier indices (first date >= event + n_days)
        event_dates = dates[event_indices]
        vb_dates = event_dates + np.timedelta64(n_days, "D")
        vb_indices = np.searchsorted(dates, vb_dates, side="left")
        # Clamp to valid range
        vb_indices = np.clip(vb_indices, 0, n - 1).astype(np.int64)

        # Compute thresholds per event
        event_vol = vol[event_indices]
        # Replace NaN vol with 0.01 fallback
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

        # Build Polars DataFrame
        return pl.DataFrame(
            {
                "date": dates[event_indices],
                "exit_date": dates[exit_indices],
                "label": labels,
                "returns": returns,
            }
        )

    @staticmethod
    def one_period_with_sl(
        dates: np.ndarray,
        close: np.ndarray,
        low: np.ndarray,
        stop_loss_ret: float = -0.02,
        event_mask: np.ndarray | None = None,
    ) -> pl.DataFrame:
        """Label each bar based on next-period return with an intraday stop-loss check.

        For each event at index i:
        1. If low[i+1]/close[i+1] - 1 < stop_loss_ret, label = -1 and
           return = stop_loss_ret (stop-loss hit intraday).
        2. Otherwise, return = close[i+2]/close[i+1] - 1 and label = sign(return).

        Events too close to the end of the array (where i+2 >= n) receive null
        exit_date, label, and returns.

        Args:
            dates: Array of datetime64 values (sorted ascending).
            close: Array of close prices, same length as dates.
            low: Array of low prices, same length as dates.
            stop_loss_ret: Negative return threshold for intraday stop-loss.
            event_mask: Boolean mask selecting which dates are events. If None,
                all dates are used.

        Returns:
            pl.DataFrame with columns: date, exit_date, label, returns.
            Same number of rows as input dates (or number of True values in event_mask).
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

        # Split into computable (i+2 < n) and trailing events
        valid_mask = all_event_indices + 2 < n
        valid_indices = all_event_indices[valid_mask]

        # Prepare full-length output arrays with nulls for trailing events
        n_all = len(all_event_indices)
        out_exit_dates = np.empty(n_all, dtype=dates.dtype)
        out_labels = np.empty(n_all, dtype=np.int64)
        out_returns = np.empty(n_all, dtype=np.float64)

        # Fill trailing events with null sentinels
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

        # Build DataFrame, converting sentinel values to proper nulls
        df = pl.DataFrame(
            {
                "date": dates[all_event_indices],
                "exit_date": out_exit_dates,
                "label": out_labels,
                "returns": out_returns,
            }
        )

        # Replace NaT / NaN with proper Polars nulls for trailing rows
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

    @staticmethod
    def n_period_with_sl(
        dates: np.ndarray,
        close: np.ndarray,
        low: np.ndarray,
        n_periods: int = 1,
        stop_loss_ret: float = -0.02,
        event_mask: np.ndarray | None = None,
    ) -> pl.DataFrame:
        """Label each bar based on n-period return with daily stop-loss check.

        For each event at index i:
        1. Check stop-loss: for each bar j from i+1 to i+n_periods, if
           low[j] / close[i] - 1 < stop_loss_ret, stop out at bar j with
           label = -1 and return = stop_loss_ret.
        2. If no stop-loss hit: return = close[i+1+n_periods] / close[i+1] - 1
           and label = sign(return).

        Events too close to the end of the array (where i+1+n_periods >= n)
        receive null exit_date, label, and returns.

        Args:
            dates: Array of datetime64 values (sorted ascending).
            close: Array of close prices, same length as dates.
            low: Array of low prices, same length as dates.
            n_periods: Number of holding periods.
            stop_loss_ret: Negative return threshold for stop-loss.
            event_mask: Boolean mask selecting which dates are events. If None,
                all dates are used.

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

        # Need access to index i+1+n_periods
        valid_mask = all_event_indices + n_periods + 1 < n
        valid_indices = all_event_indices[valid_mask]

        # Prepare full-length output arrays with null sentinels
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

        # Build DataFrame, converting sentinel values to proper nulls
        df = pl.DataFrame(
            {
                "date": dates[all_event_indices],
                "exit_date": out_exit_dates,
                "label": out_labels,
                "returns": out_returns,
            }
        )

        # Replace NaT / NaN with proper Polars nulls for trailing rows
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

    def add_labels(
        self,
        method: str = "triple_barrier",
        price: str = "Close",
        **kwargs,
    ):
        labels = None

        # If hit_date, bin, and returns column are present in the dataframe, remove them
        if "hit_date" in self._df.columns:
            self._df.drop(columns=["hit_date", "bin", "returns"], inplace=True)

        if method == "triple_barrier":
            if "event_flag" in self._df.columns:
                t_events = self._df["event_flag"].dropna().index
                labels = self.triple_barrier_method(
                    self._df[price], t_events=t_events, **kwargs
                )
            else:
                labels = self.triple_barrier_method(
                    self._df[price], t_events=self._df.index, **kwargs
                )

        self._df = self._df.join(labels, how="left")

    @staticmethod
    def num_co_events(close_idx, t_exits):
        """
        Compute the number of concurrent events per bar across the entire `closeIdx` range.

        Any event that starts before the maximum of `t1` impacts the count.
        """
        # 1) Handle unclosed events (events with NaN end date)
        t_exits.fillna(close_idx[-1])
        # unclosed events affect the count

        # 2) Find the relevant range of events
        # events that end after the first closeIdx time
        t_exits = t_exits[t_exits >= close_idx[0]]
        # events that start at or before the latest event in t1
        t_exits = t_exits.loc[: t_exits.max()]

        # 3) Initialize a count series covering the entire closeIdx range
        iloc = close_idx.searchsorted(np.array([t_exits.index[0], t_exits.max()]))
        count = pd.Series(0, index=close_idx[iloc[0] : iloc[1] + 1])

        # 4) Count events that span each bar in closeIdx
        for t_in, t_out in t_exits.items():
            count.loc[t_in:t_out] += 1

        return count

    @staticmethod
    def average_uniqueness(t_exits, co_events):
        wght = pd.Series(index=t_exits.index)
        for t_in, t_out in t_exits.items():
            wght.loc[t_in] = (1.0 / co_events.loc[t_in:t_out]).mean()

        return wght

    @staticmethod
    def get_sample_weights(Close, t_exits: pd.Series):
        """_summary_

        Args:
            Close (pd.Series): Price series
            t_exits (pd.Series): Datetime index of entry dates and values of exit dates

        Returns:
            _type_: _description_
        """

        co_events = Labeller.num_co_events(Close.index, t_exits)
        # Derive sample weight by return attribution
        ret = np.log(Close).diff()  # log-returns, so that they are additive
        wght = pd.Series(index=t_exits.index)
        for t_in, t_out in t_exits.loc[wght.index].items():
            wght.loc[t_in] = (ret.loc[t_in:t_out] / co_events.loc[t_in:t_out]).sum()
        return wght.abs()

    def add_sample_weights(self, price: Optional[str] = "Close"):
        sw = None

        if "hit_date" not in self._df.columns:
            raise ValueError(
                "hit_date column not found in the dataframe. Please add labels first."
            )

        # If 'sample_weight' column is present in the dataframe, remove it
        if "sample_weight" in self._df.columns:
            self._df.drop(columns=["sample_weight"], inplace=True)

        sw = self.get_sample_weights(
            Close=self._df[price], t_exits=self._df["hit_date"]
        )

        self._df["sample_weight"] = sw
