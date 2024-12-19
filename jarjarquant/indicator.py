"""Indicator class - each indicator in jarjarquant is an instance of the Indicator class.
Each indicator is a specific calculation on market data with an underlying market hypothesis"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from .data_analyst import DataAnalyst


class Indicator:
    """Base class to implement indicators"""

    def __init__(self, ohlcv_df: pd.DataFrame):
        if ohlcv_df is None or ohlcv_df.empty:
            raise ValueError("Please provide a valid OHLCV DataFrame!")

        self.df = ohlcv_df.copy()
        self.indicator_type = None
        self.data_analyst = DataAnalyst()

    def calculate(self):
        """Placeholder - to be implemented in derived classes

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "Derived classes must implement the calculate method.")

    def indicator_evaluation_report(self, n_bins_to_discretize: int = 10):
        """Runs a set of statistical tests to examine various properties of the 
        indicator series, such as, stationarity, normality, entropy, mutual
        information, etc.

        Args:
            n_bins_to_discretize (int, optional): number of bins to use if indicator
            is continous. Used for mutual information calculation. Defaults to 10.
        """
        values = self.calculate()
        if self.indicator_type == 'c':
            d_values = self.data_analyst.discretize_array(
                values, n_bins_to_discretize)
        else:
            d_values = values

        self.data_analyst.visual_stationary_test(values)
        print("/n")
        self.data_analyst.adf_test(values)
        print("/n")
        self.data_analyst.jb_normality_test(values, plot_dist=True)
        print("/n")
        print(f"Relative entropy = {
              self.data_analyst.relative_entropy(values, True)}")
        print(f"Range IQR Ratio = {
              self.data_analyst.range_iqr_ratio(values, True)}")
        print(f"Mutual information at lag 1 = {
              self.data_analyst.mutual_information(d_values, 1)}")


class RSI(Indicator):
    """Class to calculate the Relative Strength Index (RSI)"""

    def __init__(self, ohlcv_df: pd.DataFrame, period: int = 14):
        super().__init__(ohlcv_df)
        self.period = period
        self.indicator_type = 'c'  # 'c' for continuous, 'd' for discrete

    def calculate(self) -> np.ndarray:
        close = self.df['Close'].values
        n = len(close)
        front_bad = self.period
        output = np.full(n, 50.0)  # Default RSI of 50.0 for undefined values

        # Calculate initial sums for up and down movements
        deltas = np.diff(close)
        ups = np.where(deltas > 0, deltas, 0)
        downs = np.where(deltas < 0, -deltas, 0)

        # Initialize the up and down sums
        upsum = np.sum(ups[:self.period - 1]) / \
            (self.period - 1) + np.finfo(float).eps
        dnsum = np.sum(downs[:self.period - 1]) / \
            (self.period - 1) + np.finfo(float).eps

        # Compute RSI values after initial self.period period

        for i in range(front_bad, n):
            diff = deltas[i - 1]
            if diff > 0:
                upsum = ((self.period - 1) * upsum + diff) / self.period
                dnsum *= (self.period - 1) / self.period
            else:
                dnsum = ((self.period - 1) * dnsum - diff) / self.period
                upsum *= (self.period - 1) / self.period

            # RSI calculation
            output[i] = 100.0 * upsum / (upsum + dnsum)

        return output


class CMMA(Indicator):

    def __init__(self, ohlcv_df: pd.DataFrame, lookback: int = 21, atr_length: int = 21):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.atr_length = atr_length
        self.indicator_type = 'c'  # 'c' for continuous, 'd' for discrete

    def calculate(self) -> np.ndarray:
        """
        Calculate the Cumulative Moving Mean Average (CMMA) indicator.

        Parameters:
        lookback: The number of periods for calculating the rolling mean.
        atr_length: The length of the rolling window for the ATR calculation.
        df: A pandas DataFrame containing 'Close', 'Open', 'Low', and 'High' columns.

        Returns:
        A pandas Series with the CMMA values.
        """

        # Extract the relevant columns from the DataFrame
        Close = self.df['Close']
        Open = self.df['Open']
        Low = self.df['Low']
        High = self.df['High']

        # Compute the natural logarithm of the close prices
        log_close = np.log(Close)

        # Calculate the rolling mean of the log of close prices over the lookback period
        rolling_mean = log_close.shift(1).rolling(
            window=self.lookback).ewm.mean()

        # Calculate the denominator using the ATR function and adjust by the sqrt of (lookback + 1)
        denom = self.data_analyst.atr(self.atr_length, Open, High, Low, Close) * \
            np.sqrt(self.lookback + 1)

        # Normalize the output by dividing the difference between log_close and rolling_mean by denom
        # If denom is 0 or negative, set the normalized output to 0
        normalized_output = np.where(
            denom > 0,
            (log_close - rolling_mean) / denom,
            0
        )

        # Transform the normalized output using the cumulative distribution function (CDF) of the normal distribution
        output = 100 * norm.cdf(normalized_output) - 50

        # Return the final CMMA values as a pandas Series with the same index as the input DataFrame
        return output
