import numpy as np
import pandas as pd

from jarjarquant.indicators.base import Indicator


class Aroon(Indicator):
    """Class to calculate the Aroon indicator"""

    def __init__(self, ohlcv_df: pd.DataFrame, lookback: int = 25, transform=None):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.indicator_type = "continuous"
        self.transform = transform

    def calculate(self) -> np.ndarray:
        high = self.df["High"].values
        low = self.df["Low"].values
        n = len(high)
        output = np.full(n, 0.0)

        for i in range(self.lookback, n):
            high_max = np.argmax(np.asarray(high[i - self.lookback : i]))
            low_min = np.argmin(np.asarray(low[i - self.lookback : i]))

            aroon_up = 100 * (self.lookback - (i - high_max)) / self.lookback
            aroon_down = 100 * (self.lookback - (i - low_min)) / self.lookback

            output[i] = aroon_up - aroon_down

        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
