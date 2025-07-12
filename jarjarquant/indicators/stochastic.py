import numpy as np
import pandas as pd

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.STOCHASTIC)
class Stochastic(Indicator):
    """Class to calculate the stochastic oscillator"""

    def __init__(self, ohlcv_df, lookback: int = 14, n_smooth: int = 2, transform=None):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.n_smooth = n_smooth
        self.indicator_type = "continuous"
        self.transform = transform

    def calculate(self) -> np.ndarray:
        close = self.df["Close"].values
        n = len(close)
        output = np.full(n, 50.0)

        # Calculate rolling max and min for Close values
        high_max = pd.Series(close).rolling(window=self.lookback).max().values
        low_min = pd.Series(close).rolling(window=self.lookback).min().values

        for i in range(self.lookback, n):
            if high_max[i] == low_min[i]:
                output[i] = 50.0
            else:
                sto_0 = 100 * (close[i] - low_min[i]) / (high_max[i] - low_min[i])
                if self.n_smooth == 0:
                    output[i] = sto_0
                elif self.n_smooth == 1:
                    if i == self.lookback:
                        output[i] = sto_0
                    else:
                        output[i] = 0.33333 * sto_0 + 0.66667 * output[i - 1]
                else:
                    if i < self.lookback + 1:
                        output[i] = sto_0
                    elif i == self.lookback + 1:
                        output[i] = 0.33333 * sto_0 + 0.66667 * output[i - 1]
                    else:
                        sto_1 = 0.33333 * sto_0 + 0.66667 * output[i - 1]
                        output[i] = 0.33333 * sto_1 + 0.66667 * output[i - 2]

        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
