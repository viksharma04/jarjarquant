# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Labeller:
    "Class to label data - implements commonn methods used during labelling for financial ml"

    def __init__(self, timeseries: pd.Series):
        """Initialize Labelling

        Args:
            timeseries (Optional[pd.Series], optional): any timeseires with a datetime index. Defaults to None.
        """
        self.series = timeseries

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
            series = self.series

        returns = series.pct_change()
        # Ensure the series is sorted by index (time)
        returns = returns.add(1)

        # Calculate the rolling cumulative sum over the lookback period n
        rolling_cumsum = returns.rolling(window=n).apply(np.prod) - 1

        # Flag dates where the cumulative return is less than the absolute value of h
        flagged = (rolling_cumsum.abs() < h)

        return flagged

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
    def get_vertical_barrier(t_events, close, num_days=1):
        """Get a datetime index of dates for the vertical barrier

        Args:
            tEvents (datetime index): dates when the algorithm should look for trades
            close (pd.Series): series of prices
            numDays (int, optional): vertical barrier limit. Defaults to 1.

        Returns:
            pd.Series: series of datetime values
        """
        t1 = close.index.searchsorted(t_events+pd.Timedelta(days=num_days))
        t1 = t1[t1 < close.shape[0]]
        t1 = (pd.Series(close.index[t1], index=t_events[:t1.shape[0]]))
        return t1

    @staticmethod
    def find_min_column(row):

        if pd.isnull(row['pt']) & pd.isnull(row['sl']):
            min_value = row[['pt', 'sl']].min()
        else:
            min_value = pd.Timestamp(0)
        return row[['pt', 'sl']].idxmin() if min_value <= row['vb'] else row[['pt', 'sl', 'vb']].idxmin()

    def triple_barrier_method(self, close: pd.Series = None, t_events: pd.DatetimeIndex = None, scale_pt_sl: bool = False, pt_sl: int = 1, scale_lookback: int = 1, n_days: int = 1, meta_labelling: bool = True):

        if close is None:
            close = self.series

        if t_events is None:
            t_events = close.index

        # If scale pt_sl is True pt_sl is multiplied by the average period volatility over the scale_lookback period
        if scale_pt_sl:
            returns = close.pct_change()
            vol = returns.rolling(
                window=scale_lookback).std()*np.sqrt(scale_lookback)
            trgt = vol[vol.index.isin(t_events)]
            close = close.iloc[scale_lookback:]
        # If scale pt_sl is False pt_sl is used as absolute return i.e 1 = 1% return
        else:
            trgt = pd.Series(0.01, index=t_events)

        t_events = t_events[t_events.isin(close.index)]

        v_bars = self.get_vertical_barrier(t_events, close, n_days)
        pt_sl = [pt_sl, -pt_sl]

        events = pd.concat({'vb': v_bars, 'trgt': trgt},
                           axis=1).dropna(subset=['trgt'])

        exits = events[['vb']].copy(deep=True)

        pt = pt_sl[0]*events['trgt']
        sl = pt_sl[1]*events['trgt']

        for event, vb in events['vb'].fillna(close.index[-1]).items():

            price_path = close[event:vb]
            return_path = (price_path/close[event]-1)
            exits.loc[event, 'sl'] = return_path[return_path <
                                                 # earliest stop loss
                                                 sl[event]].index.min()
            exits.loc[event, 'pt'] = return_path[return_path >
                                                 # earliest profit taking
                                                 pt[event]].index.min()

        exits['vb'] = exits['vb'].fillna(close.index[-1])
        exits['barrier_hit'] = exits.apply(self.find_min_column, axis=1)
        exits['hit_date'] = exits[['vb', 'sl', 'pt']].min(axis=1)
        exits['returns'] = close[exits['hit_date']
                                 ].values/close[t_events].values - 1
        exits['bin'] = np.sign(exits['returns'])
        exits.loc[exits['barrier_hit'] == 'vb', 'bin'] = 0

        if meta_labelling:
            # Create the 'bin_1' column based on the sign of the 'return' column
            exits['bin_1'] = exits['returns'].apply(
                lambda x: 1 if x >= 0 else -1)

            # Create the 'bin_2' column based on the values in the original 'bin' column
            exits['bin_2'] = exits['bin'].apply(lambda x: 0 if x == 0 else 1)

            # Return the transformed dataframe with only the specified columns
            return exits[['bin_1', 'bin_2', 'returns', 'hit_date']]
        else:
            return exits[['bin', 'returns', 'hit_date']]

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

    def get_sample_weights(self, close, t_exits):

        if close is None:
            close = self.series

        co_events = self.num_co_events(close.index, t_exits)
        # Derive sample weight by return attribution
        ret = np.log(close).diff()  # log-returns, so that they are additive
        wght = pd.Series(index=t_exits.index)
        for t_in, t_out in t_exits.loc[wght.index].items():
            wght.loc[t_in] = (ret.loc[t_in:t_out] /
                              co_events.loc[t_in:t_out]).sum()
        return wght.abs()
