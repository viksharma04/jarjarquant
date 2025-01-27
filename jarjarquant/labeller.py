"""The labeller specializes in transforming raw price data into labels for ML using various methods"""
# Imports
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data_analyst import DataAnalyst


class Labeller():
    "Class to label data - implements commonn methods used during labelling for financial ml"

    def __init__(self, ohlcv_df: pd.DataFrame):
        """Initialize Labelling

        Args:
            ohlcv_df (pd.DataFrame): DataFrame containing OHLCV data with a datetime index.
        """
        self._df = ohlcv_df

    def inverse_cumsum_filter(self, series: pd.Series = None, h: float = 0.01, n: int = 2) -> pd.Series:
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
            series = self._df['Close']

        returns = series.pct_change()
        # Ensure the series is sorted by index (time)
        returns = returns.add(1)

        # Calculate the rolling cumulative sum over the lookback period n
        rolling_cumsum = returns.rolling(window=n).apply(np.prod) - 1

        # Flag dates where the cumulative return is less than the absolute value of h
        flagged = (rolling_cumsum.abs() < h)

        return flagged

    def event_sampling(self, method: str = 'vol_contraction', long_lookback: int = 100, short_lookback: int = 10, threshold: float = 2, price: Optional[str] = 'Close'):
        """
        method can be 'vol_contraction', 'vol_expansion'
        contraction and expansion are defined based on three parameters: long lookback, short lookback, and threshold
        """
        price_series = self._df[price]
        data_analyst = DataAnalyst()

        long_vol = data_analyst.get_daily_vol(price_series, long_lookback)
        short_vol = data_analyst.get_daily_vol(price_series, short_lookback)

        if method == 'vol_contraction':
            ratio = long_vol/short_vol
        elif method == 'vol_expansion':
            ratio = short_vol/long_vol
        else:
            raise ValueError(
                "Method must be 'volatility_contraction' or 'volatility_expansion'")

        # Flag dates where the ratio is greater than the threshold
        flagged = (ratio > threshold)

        # Add flagged column as 'event_flag' column to self._df
        self._df['event_flag'] = flagged

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
        plt.plot(series.index, series.values,
                 label='Time Series', color='blue')

        # Highlight flagged dates as red dots
        plt.scatter(series.index[flagged], series.values[flagged],
                    color='red', label='Flagged Dates')

        # Add labels and legend
        plt.title(f"Time Series with Flagged Dates; Percent labels = {
                  np.average(flagged)*100}%")
        plt.xlabel('Date')
        plt.ylabel('Return')
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
        t1 = Close.index.searchsorted(t_events+pd.Timedelta(days=num_days))
        t1 = t1[t1 < Close.shape[0]]
        t1 = (pd.Series(Close.index[t1], index=t_events[:t1.shape[0]]))
        t1.index = t1.index.tz_localize(None)
        return t1

    @staticmethod
    def find_min_column(row):

        if pd.isnull(row['pt']) & pd.isnull(row['sl']):
            min_value = row[['pt', 'sl']].min()
        else:
            min_value = pd.Timestamp(0)
        return row[['pt', 'sl']].idxmin() if min_value <= row['vb'] else row[['pt', 'sl', 'vb']].idxmin()

    @staticmethod
    def triple_barrier_method(Close: pd.Series, t_events: pd.DatetimeIndex = None, scale_pt_sl: bool = True, span: int = 100, pt_sl: int = 1.5, n_days: int = 10):

        Close.index = Close.index.tz_localize(None)

        # If scale pt_sl is True pt_sl is multiplied by the average period volatility over the scale_lookback period
        if scale_pt_sl:
            data_analyst = DataAnalyst()
            vol = data_analyst.get_daily_vol(close=Close, span=span)
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
            Close = Close.loc[t_events]

        t_events = t_events[t_events.isin(Close.index)]

        v_bars = Labeller.get_vertical_barrier(t_events, Close, n_days)
        pt_sl = [pt_sl, -pt_sl]

        events = pd.concat({'vb': v_bars, 'trgt': trgt},
                           axis=1).dropna(subset=['trgt'])

        exits = events[['vb']].copy(deep=True)
        exits['sl'] = pd.NaT
        exits['pt'] = pd.NaT

        pt = pt_sl[0]*events['trgt']
        sl = pt_sl[1]*events['trgt']

        for event, vb in events['vb'].fillna(Close.index[-1]).items():

            price_path = Close[event:vb]
            return_path = (price_path/Close[event]-1)
            exits.loc[event, 'sl'] = return_path[return_path <
                                                 # earliest stop loss
                                                 sl[event]].index.min()
            exits.loc[event, 'pt'] = return_path[return_path >
                                                 # earliest profit taking
                                                 pt[event]].index.min()

        exits['vb'] = exits['vb'].fillna(Close.index[-1])
        exits['barrier_hit'] = exits.apply(
            Labeller.find_min_column, axis=1)
        exits['hit_date'] = exits[['vb', 'sl', 'pt']].min(axis=1)
        exits['returns'] = Close[exits['hit_date']
                                 ].values/Close[t_events].values - 1
        exits['bin'] = np.sign(exits['returns'])
        exits.loc[exits['barrier_hit'] == 'vb', 'bin'] = 0

        return exits[['hit_date', 'bin', 'returns']]

    def add_labels(self, method: Optional[str] = 'triple_barrier', price: Optional[str] = 'Close', **kwargs):

        labels = None

        # If hit_date, bin, and returns column are present in the dataframe, remove them
        if 'hit_date' in self._df.columns:
            self._df.drop(columns=['hit_date', 'bin', 'returns'], inplace=True)

        if method == 'triple_barrier':
            if 'event_flag' in self._df.columns:
                t_events = self._df['event_flag'].dropna().index
                labels = self.triple_barrier_method(
                    self._df[price], t_events=t_events, **kwargs)
            else:
                labels = self.triple_barrier_method(
                    self._df[price], **kwargs)

        self._df = self._df.join(labels, how='left')

    @staticmethod
    def num_co_events(close_idx, t_exits):
        '''
        Compute the number of concurrent events per bar across the entire `closeIdx` range.

        Any event that starts before the maximum of `t1` impacts the count.
        '''
        # 1) Handle unclosed events (events with NaN end date)
        t_exits.fillna(close_idx[-1])
        # unclosed events affect the count

        # 2) Find the relevant range of events
        # events that end after the first closeIdx time
        t_exits = t_exits[t_exits >= close_idx[0]]
        # events that start at or before the latest event in t1
        t_exits = t_exits.loc[:t_exits.max()]

        # 3) Initialize a count series covering the entire closeIdx range
        iloc = close_idx.searchsorted(
            np.array([t_exits.index[0], t_exits.max()]))
        count = pd.Series(0, index=close_idx[iloc[0]:iloc[1] + 1])

        # 4) Count events that span each bar in closeIdx
        for t_in, t_out in t_exits.items():
            count.loc[t_in:t_out] += 1

        return count

    @staticmethod
    def average_uniqueness(t_exits, co_events):
        wght = pd.Series(index=t_exits.index)
        for t_in, t_out in t_exits.items():
            wght.loc[t_in] = (1./co_events.loc[t_in:t_out]).mean()

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
            wght.loc[t_in] = (ret.loc[t_in:t_out] /
                              co_events.loc[t_in:t_out]).sum()
        return wght.abs()

    def add_sample_weights(self, price: Optional[str] = 'Close'):

        sw = None

        if 'hit_date' not in self._df.columns:
            raise ValueError(
                "hit_date column not found in the dataframe. Please add labels first.")

        # If 'sample_weight' column is present in the dataframe, remove it
        if 'sample_weight' in self._df.columns:
            self._df.drop(columns=['sample_weight'], inplace=True)

        sw = self.get_sample_weights(
            Close=self._df[price], t_exits=self._df['hit_date'])

        self._df['sample_weight'] = sw
