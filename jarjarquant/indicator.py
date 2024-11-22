"""Indicator class - each indicator in jarjarquant is an instance of the Indicator class.
Indicator evaluator accepts an object of class Indicator"""

import numpy as np
import pandas as pd
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
        self.data_analyst.adf_test(values)
        self.data_analyst.jb_normality_test(values, plot_dist=True)
        print(f"Relative entropy = {
              self.data_analyst.relative_entropy(values)}")
        print(f"Range IQR Ratio = {self.data_analyst.range_iqr_ratio(values)}")
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
