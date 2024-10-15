# Imports
import numpy as np
import pandas as pd


class Labeller:
    "Class to label data - implements commonn methods used during labelling for financial ml"

    def __init__(self, timeseries: pd.Series):
        """Initialize Labelling

        Args:
            timeseries (Optional[pd.Series], optional): any timeseires with a datetime index. Defaults to None.
        """
        self.series = timeseries

    @staticmethod
    def inverse_cumsum_filter(series: pd.Series, h: float, n: int) -> pd.Series:
        """
        Apply a cumulative sum filter to a time series based on a rolling period.

        Parameters:
        - series: pd.Series, time series of prices with time stamp index
        - h: float, threshold value for filtering
        - n: int, lookback period for the rolling window

        Returns:
        - pd.Series, boolean series where True indicates dates flagged by the filter
        """
        returns = series.pct_change()
        # Ensure the series is sorted by index (time)
        returns = returns.add(1)

        # Calculate the rolling cumulative sum over the lookback period n
        rolling_cumsum = returns.rolling(window=n).apply(np.prod) - 1

        # Flag dates where the cumulative return is less than the absolute value of h
        flagged = (rolling_cumsum.abs() < h)

        return flagged
