import numpy as np
import pandas as pd

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType

from .rsi import RSI


@register_indicator(IndicatorType.DETRENDED_RSI)
class DetrendedRSI(Indicator):
    """Class to calculate the Detrended RSI"""

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        short_period: int = 2,
        long_period: int = 21,
        regression_length: int = 120,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.short_period = short_period
        self.long_period = long_period
        self.regression_length = regression_length
        self.indicator_type = "continuous"
        self.transform = transform

    def calculate(self) -> np.ndarray:
        close = self.df["Close"].values
        n = len(close)
        output = np.full(n, 0.0)

        short_rsi = RSI(self.df, self.short_period).calculate()
        # Apply inverse logistic transformation to the short RSI values
        # Look at pg 103 of Statistically Sound Indicators
        short_rsi = -10 * np.log((2 / (1 + 0.00999 * (2 * short_rsi - 100))) - 1)

        # Long RSI
        long_rsi = RSI(self.df, self.long_period).calculate()

        for i in range(self.regression_length + self.long_period - 1, n):
            x = long_rsi[i - self.regression_length : i]
            y = short_rsi[i - self.regression_length : i]

            x_mean = np.mean(x)
            y_mean = np.mean(y)

            x_diff = x - x_mean
            y_diff = y - y_mean

            coef = np.dot(x_diff, y_diff) / (np.dot(x_diff, x_diff) + 1e-10)

            output[i] = (y[-1] - y_mean) - coef * (x[-1] - x_mean)

        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
