import numpy as np
import polars as pl

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import IndicatorType, register_indicator


@register_indicator(IndicatorType.RSI)
class RSI(Indicator):
    """Class to calculate the Relative Strength Index (RSI)"""

    def __init__(self, ohlcv_df: pl.DataFrame, period: int = 14, transform=None):
        super().__init__(ohlcv_df)
        self.period = period
        self._transform = transform

    def _compute(self) -> np.ndarray:
        close = self._df["Close"].to_numpy()
        n = len(close)
        front_bad = self.period
        output = np.full(n, 50.0)

        deltas = np.diff(close)
        ups = np.where(deltas > 0, deltas, 0)
        downs = np.where(deltas < 0, -deltas, 0)

        upsum = np.sum(ups[: self.period - 1]) / (self.period - 1) + np.finfo(float).eps
        dnsum = (
            np.sum(downs[: self.period - 1]) / (self.period - 1) + np.finfo(float).eps
        )

        for i in range(front_bad, n):
            diff = deltas[i - 1]
            if diff > 0:
                upsum = ((self.period - 1) * upsum + diff) / self.period
                dnsum *= (self.period - 1) / self.period
            else:
                dnsum = ((self.period - 1) * dnsum - diff) / self.period
                upsum *= (self.period - 1) / self.period

            if upsum + dnsum == 0:
                output[i] = 50.0
            else:
                output[i] = 100.0 * upsum / (upsum + dnsum)

        output = (output - 50) / 10

        return output
