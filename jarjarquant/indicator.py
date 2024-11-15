"""Indicator class - each indicator in jarjarquant is an instance of the Indicator class. Indicator evaluator accepts an object of class Indicator"""
from typing import Optional
import numpy as np
import pandas as pd


class Indicator:
    """Base class to implement indicators"""

    def __init__(self, ohlcv_df: pd.DataFrame):
        if ohlcv_df is None or ohlcv_df.empty:
            raise ValueError("Please provide a valid OHLCV dataframe!")

        self.df = ohlcv_df

    def calculate(self):
        raise NotImplementedError(
            "Derived classes must implement the calculate method.")

    def visual_stationary_test():
        """Plots a line graph of self.value for any class that extends indicator. Draws a top and bottom horizontal threshold above/below which 15% of observations lie

        Returns:
            _type_: _description_
        """


class RSI(Indicator):
    """Class to calculate the Relative Strength Index (RSI)"""

    def __init__(self, ohlcv_df: pd.DataFrame, period: int = 14):
        super().__init__(ohlcv_df)
        self.period = period
        self.value = self.calculate()

    def calculate(self):
        close = self.df['Close']
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
