import numpy as np
import pandas as pd

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.RSI)
class RSI(Indicator):
    """Class to calculate the Relative Strength Index (RSI)"""

    def __init__(self, ohlcv_df: pd.DataFrame, period: int = 14, transform=None):
        super().__init__(ohlcv_df)
        self.period = period
        self.indicator_type = "continuous"  # continuous or discrete
        self.transform = transform

    def calculate(self) -> np.ndarray:
        close = np.asarray(self.df["Close"].values)
        n = len(close)
        front_bad = self.period
        output = np.full(n, 50.0)  # Default RSI of 50.0 for undefined values

        # Calculate initial sums for up and down movements
        deltas = np.diff(close)
        ups = np.where(deltas > 0, deltas, 0)
        downs = np.where(deltas < 0, -deltas, 0)

        # Initialize the up and down sums
        upsum = np.sum(ups[: self.period - 1]) / (self.period - 1) + np.finfo(float).eps
        dnsum = (
            np.sum(downs[: self.period - 1]) / (self.period - 1) + np.finfo(float).eps
        )

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
            if upsum + dnsum == 0:
                output[i] = 50.0  # Default RSI value when both sums are zero
            else:
                output[i] = 100.0 * upsum / (upsum + dnsum)

        output = (output - 50) / 10

        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
