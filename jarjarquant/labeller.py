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
