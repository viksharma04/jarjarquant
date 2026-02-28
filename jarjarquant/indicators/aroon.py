import numpy as np
import polars as pl

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.AROON)
class Aroon(Indicator):
    """Class to calculate the Aroon indicator"""

    def __init__(self, ohlcv_df: pl.DataFrame, lookback: int = 25, transform=None):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self._transform = transform

    def _compute(self) -> np.ndarray:
        high = self._df["High"].to_numpy()
        low = self._df["Low"].to_numpy()
        n = len(high)
        output = np.full(n, 0.0)

        for i in range(self.lookback, n):
            high_max = np.argmax(np.asarray(high[i - self.lookback : i]))
            low_min = np.argmin(np.asarray(low[i - self.lookback : i]))

            aroon_up = 100 * (self.lookback - (i - high_max)) / self.lookback
            aroon_down = 100 * (self.lookback - (i - low_min)) / self.lookback

            output[i] = aroon_up - aroon_down

        return output
